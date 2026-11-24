"""Scan all visible holdings at the preparation anchor and reconcile bookkeeping."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from maa.define import MaaAdbInputMethodEnum
from maa.tasker import Tasker

from ..prep_holdings import AREAS
from ..track import HAND, TEMP, Track
from .board_read import read_board_tiles
from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .deploy import read_board
from .equipment_read import read_equipment_slots
from .hand import read_hand
from .read import CATALOG, ROOT, Reader, read_json
from .shop import set_shop_open


def scan_holdings(tasker: Tasker, reader: Reader, board: dict, *,
                  areas: set[str] | None = None, rack_slots: list[int] | None = None,
                  known_layout: dict[tuple[int, int], tuple[str, str | None]] | None = None) -> dict:
    """Read only the requested areas and restore the starting shop view."""
    areas = set(AREAS) if areas is None else areas
    if not areas or not areas <= set(AREAS):
        raise ValueError("Select at least one valid holdings area")
    start = perf_counter()
    hand = field = temp = None
    phases = []
    view = set_shop_open(tasker, False) if areas & {"hand", "board"} else None
    try:
        if "hand" in areas:
            hand = read_hand(tasker, reader, skip_bare=True)
            phases.append(hand)
        if "board" in areas:
            field = read_board_tiles(tasker, reader, board, known_layout=known_layout)
            phases.append(field)
    finally:
        if view is not None:
            phases.extend((view["before"], set_shop_open(tasker, view["before"]["expanded"])["after"]))
    if "temp" in areas:
        temp = read_equipment_slots(tasker, reader, rack_slots)
        phases.append(temp)
    if len({(part["round"], part["funds"]) for part in phases}) != 1:
        raise RuntimeError("Round or funds changed during the holdings scan")
    if hand and (hand["stageId"] != board["stageId"] or field and hand["remaining"] != field["remaining"]):
        raise RuntimeError("The hand and board were read from different preparation states")
    return {"stageId": board["stageId"], "round": phases[0]["round"], "funds": phases[0]["funds"],
            "remaining": field["remaining"] if field else None,
            "hand": hand["slots"] if hand else None, "temp": temp["slots"] if temp else None,
            "rackSlots": rack_slots,
            "board": field["pieces"] if field else None, "facingSource": field["facingSource"] if field else None,
            "unreadTiles": field["unreadTiles"] if field else [], "boardSource": "screen" if field else "track",
            "totalMs": round((perf_counter() - start) * 1000)}


def reconcile_holdings(track: Track, scan: dict, *, deploy_cap: int,
                       confirmed_ids: dict[str, str] | None = None) -> dict:
    """Replace only the observed areas after their identities reconcile."""
    if scan["board"] is not None and (type(deploy_cap) is not int or deploy_cap < 0):
        raise ValueError("A verified deployment capacity is required")
    confirmed_ids = confirmed_ids or {}
    old = {**{f"hand:{i}": piece for i, piece in enumerate(track.hand) if piece is not None},
           **{f"temp:{i}": piece for i, piece in enumerate(track.temp[:TEMP]) if piece is not None},
           **{f"board:{key}": piece for key, piece in track.board.items()}}
    issues = []
    rows = {}
    for area, capacity in (("hand", HAND), ("temp", TEMP)):
        slots = scan[area]
        if slots is None:
            continue
        indices = (scan.get("rackSlots") or list(range(capacity))) if area == "temp" else list(range(capacity))
        if sorted(slot["screenSlot"] for slot in slots) != sorted(indices):
            issues.append(f"{area}: incomplete slot scan")
        for slot in slots:
            key = f"{area}:{slot['screenSlot']}"
            if slot["status"] == "occupied":
                rows[key] = ("item" if area == "temp" else slot.get("kind", "chess"), slot)
            elif slot["status"] != "empty":
                issues.append(f"{key}: occupancy could not be verified")

    if scan["unreadTiles"]:
        issues.append("board: some deployment tiles are uncalibrated")
    if scan["board"] is not None and scan["remaining"] != deploy_cap - len(scan["board"]):
        issues.append("board: observed occupants disagree with the deployment count")
    for piece in scan["board"] or []:
        key = f"board:{piece['row']},{piece['col']}"
        unknown = piece["facing"] is None and scan.get("facingSource") == "track"
        if key in rows or not unknown and piece["facing"] not in {"UP", "RIGHT", "DOWN", "LEFT"}:
            issues.append(f"{key}: position or facing could not be verified")
        rows[key] = ("chess", piece)

    for key in confirmed_ids.keys() - rows.keys():
        issues.append(f"{key}: supplied ID has no observed occupant")
    identities = {key: (piece.kind, piece.id, None) for key, piece in old.items()
                  if scan[key.split(":", 1)[0]] is None}
    if scan["temp"] is not None and scan.get("rackSlots") is not None:
        identities.update({f"temp:{i}": (piece.kind, piece.id, None)
                           for i, piece in enumerate(track.temp[:TEMP]) if piece and i not in scan["rackSlots"]})
    inferred = []
    for key, (kind, row) in rows.items():
        candidates = set(row["candidateIds"])
        exact = row["id"]
        if key in confirmed_ids:
            supplied = confirmed_ids[key]
            if supplied != exact and (exact is not None or supplied not in candidates):
                issues.append(f"{key}: supplied ID disagrees with the observed identity")
                continue
            exact = supplied
        if exact is None and key in old and old[key].kind == kind and old[key].id in candidates:
            exact = old[key].id
            inferred.append(key)
        lookup = {"chess": track.gd.chess, "item": track.gd.item, "token": track.gd.token}[kind]
        if exact is None or not lookup(exact):
            issues.append(f"{key}: exact identity unresolved")
            continue
        identities[key] = (kind, exact, row.get("facing"))
    if issues:
        return {"applied": False, "issues": issues, "reusedIdentityAt": inferred}

    assigned = {key: old[key] for key, (kind, exact, _) in identities.items()
                if key in old and (old[key].kind, old[key].id) == (kind, exact)}
    unmatched = [piece for key, piece in old.items() if key not in assigned]
    for key, (kind, exact, _) in identities.items():
        if key in assigned:
            continue
        matching = [piece for piece in unmatched if (piece.kind, piece.id) == (kind, exact)]
        destinations = [other for other, identity in identities.items()
                        if other not in assigned and identity[:2] == (kind, exact)]
        if len(matching) == len(destinations) == 1:
            assigned[key] = matching[0]
            unmatched.remove(matching[0])
    token_owners = {}
    for key, (kind, exact, _) in identities.items():
        if kind != "token" or key in assigned:
            continue
        owner_ids = (track.gd.token(exact) or {}).get("owners") or []
        owners = [owner for owner, (other_kind, other_id, _) in identities.items()
                  if other_kind == "chess" and other_id in owner_ids]
        if len(owners) != 1:
            return {"applied": False, "issues": [f"{key}: summoned unit owner unresolved"],
                    "reusedIdentityAt": inferred}
        token_owners[key] = owners[0]
    created = []
    for key, (kind, exact, _) in identities.items():
        if key not in assigned and kind != "token":
            assigned[key] = track._new(kind, exact)
            created.append(key)
    for key, owner in token_owners.items():
        assigned[key] = track._new("token", identities[key][1], owner=assigned[owner].uid)
        created.append(key)
    for key, (_, _, facing) in identities.items():
        if facing is not None or scan.get("facingSource") == "track" and key.startswith("board:") and key in rows:
            assigned[key].dir = facing
    track.hand = [assigned.get(f"hand:{i}") for i in range(HAND)]
    track.temp = [assigned.get(f"temp:{i}") for i in range(TEMP)] + track.temp[TEMP:]
    if scan["board"] is not None:
        track.board = {key.split(":", 1)[1]: assigned[key] for key in identities if key.startswith("board:")}
    return {"applied": True, "issues": [], "reusedIdentityAt": inferred, "newPiecesAt": created}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    args = parser.parse_args()
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        reader = Reader(args.catalog, read_json(ROOT / "calib/slots.json"))
        controller = connect(args.adb, args.address, input_methods=MaaAdbInputMethodEnum.Maatouch)
        result = scan_holdings(load_ocr(controller, args.ocr_resource), reader, read_board(args.board))
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
