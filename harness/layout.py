"""Build the layout model input and plan executable moves from its output."""

from copy import copy, deepcopy
from dataclasses import dataclass

from .track import Track


DIRECTIONS = {"UP", "RIGHT", "DOWN", "LEFT"}


def tile_key(tile: tuple[int, int]) -> str:
    return f"{tile[0]},{tile[1]}"


def layout_tiles(stage: dict, view: str) -> dict[str, str]:
    tiles = stage["deployTiles"][view]
    classes = (("melee", "melee"), ("rangedOnly", "ranged"))
    if view == "normal":
        return {tile_key(tuple(tile)): kind for source, kind in classes for tile in tiles[source]}
    return {tile_key((row + 7, 20 - col if view == "bossRight" else col)): kind
            for source, kind in classes for row, col in tiles[source] if 2 <= row <= 5}


def piece_features(record: dict) -> dict:
    position = (record.get("position") or "").upper()
    if not position and record.get("chessType") == "DIY":
        raise ValueError("A self-selected operator needs its resolved catalog record")
    cls = {"MELEE": "melee", "RANGED": "ranged"}.get(position, "all")
    attack = record.get("attackKind")
    heal = record.get("dmgType") == "heal" or attack == "heal"
    block = (record.get("stats") or {}).get("blockCnt")
    block = 1 if block is None else max(0, block)
    return {"cls": cls, "block": block if cls == "melee" and attack != "none" else 0,
            "heal": heal, "air": bool(record.get("canHitFly")) and not heal and attack != "none",
            "ground": not heal}


def build_layout_state(track: Track, screen: dict, selected_uids: list[int], *,
                       boss_template: str | None = None, side: str | None = None,
                       resolved_records: dict[int, dict] | None = None,
                       tracked_bonds: list[dict] | None = None, require_bonds: bool = True) -> dict:
    """Use the caller's chosen lineup and the current match/seat facts."""
    match, own = screen["match"], screen["self"]
    boss_template = boss_template or match.get("bossTemplate")
    stage_id, current_round = match.get("stageId"), match.get("round")
    stage = track.gd.stage(stage_id)
    lp, cap = own.get("lp"), own.get("deployCap")
    if not stage or type(current_round) is not int or type(lp) is not int or type(cap) is not int:
        raise ValueError("Layout needs a known stage, round, life and deployment capacity")
    if (len(selected_uids) != len(set(selected_uids)) or len(selected_uids) > cap
            or any(type(uid) is not int for uid in selected_uids)):
        raise ValueError("Selected UIDs must be unique integers within deployment capacity")
    bonds = None
    if require_bonds:
        bond_view = tracked_bonds if tracked_bonds is not None else own.get("bonds")
        if bond_view is None:
            raise ValueError("Layout needs the current bond view from screen and bookkeeping")
        bonds = [[bond.get(key) for key in ("bondId", "count", "tier", "layers")] for bond in bond_view]
        if any(not bond_id or any(type(value) is not int for value in values) for bond_id, *values in bonds):
            raise ValueError("Layout needs complete bond IDs, counts, tiers and layers")

    config = screen.get("config") or {}
    leader_rounds = (config.get("bossRound") or track.gd.boss_round,
                     config.get("hiddenRound") or track.gd.hidden_round)
    view = "normal"
    if current_round in leader_rounds:
        side = side or "L"
        if side not in {"L", "R"} or not boss_template:
            raise ValueError("Leader layout needs its template and a valid field side")
        view = "bossLeft" if side == "L" else "bossRight"

    records = resolved_records or {}
    pieces = []
    for uid in selected_uids:
        loc = track._loc_by_uid(uid)
        if loc is None or loc["area"] not in {"hand", "board"} or loc["piece"].kind != "chess":
            raise ValueError(f"UID {uid} is not an operator available in hand or on the board")
        piece = loc["piece"]
        record = records.get(uid) or track.gd.chess(piece.id)
        if record is None:
            raise ValueError(f"No operator record for UID {uid}")
        pieces.append({"uid": uid, "id": piece.id, "golden": track.gd.is_golden(piece.id),
                       "items": [item.id for item in piece.items], **piece_features(record),
                       "key": loc.get("key"), "dir": piece.dir if loc["area"] == "board" else None})
    return {"pieces": pieces, "map": layout_tiles(stage, view), "bonds": bonds,
            "bossTemplate": boss_template, "lp": lp, "deployCap": cap, "side": side}


@dataclass(frozen=True)
class LayoutStep:
    uid: int
    source: tuple[int, int] | None
    target: tuple[int, int] | None
    hand_slot: int | None
    direction: str | None


def plan_layout(track: Track, state: dict, result: dict, calibrated: set[tuple[int, int]]) -> list[LayoutStep]:
    """Preflight the entire board before any MFW movement, including temporary moves."""
    profiles = {piece["uid"]: piece for piece in state["pieces"]}
    targets = {row["uid"]: (tuple(row["to"]), row["dir"]) for row in result["board"]}
    if len(targets) != len(result["board"]) or targets.keys() != profiles.keys():
        raise ValueError("The chosen layout must place each selected UID exactly once")
    target_tiles = {tile for tile, _ in targets.values()}
    if len(target_tiles) != len(targets):
        raise ValueError("Two operators have the same layout destination")
    tiles = state["map"]
    for uid, (tile, direction) in targets.items():
        if (len(tile) != 2 or any(type(value) is not int for value in tile) or tile not in calibrated
                or not permits(tiles, profiles[uid], tile) or direction not in DIRECTIONS):
            raise ValueError(f"Invalid or uncalibrated layout destination for UID {uid}")
    for key, piece in track.board.items():
        if piece.kind not in {"chess", "token"} or as_tile(key) not in calibrated:
            raise ValueError("Layout requires a calibrated board of trackable pieces")
    for uid, profile in profiles.items():
        loc = track._loc_by_uid(uid)
        if (loc is None or loc["area"] not in {"board", "hand"} or loc["piece"].id != profile["id"]
                or loc.get("key") != profile["key"]
                or loc["area"] == "board" and loc["piece"].dir != profile["dir"]):
            raise ValueError(f"Layout input for UID {uid} differs from current bookkeeping")
    if "ops" in result:
        validate_ops(result["ops"], profiles, targets)

    shadow = copy(track)
    shadow.hand, shadow.temp, shadow.board = deepcopy((track.hand, track.temp, track.board))
    steps = []
    staged = set()
    profile_order = list(targets)
    while True:
        excess = [piece.uid for piece in sorted(shadow.board.values(), key=lambda p: p.kind != "token")
                  if piece.uid not in targets]
        pending = [uid for uid in profile_order if not at_destination(shadow._loc_by_uid(uid), targets[uid])]
        if not pending and not excess:
            return steps
        withdrawable = next((uid for uid in excess if shadow.withdraw_slot(uid) >= 0), None)
        if withdrawable is not None:
            simulate(shadow, steps, withdrawable, None, None)
            continue
        for uid in pending:
            target, direction = targets[uid]
            loc = shadow._loc_by_uid(uid)
            if tile_key(target) not in shadow.board and (loc["area"] == "board"
                    or shadow.deployed_chess_count() < state["deployCap"]):
                simulate(shadow, steps, uid, target, direction)
                break
        else:
            for uid in pending:
                loc = shadow._loc_by_uid(uid)
                if loc["area"] != "board" or uid in staged:
                    continue
                free_hand = shadow.withdraw_slot(uid) >= 0
                if free_hand and loc["key"] == tile_key(targets[uid][0]):
                    simulate(shadow, steps, uid, None, None)
                    staged.add(uid)
                    break
                free = calibrated - target_tiles - {as_tile(key) for key in shadow.board}
                compatible = [tile for tile in free if permits(tiles, profiles[uid], tile)]
                origin = as_tile(loc["key"])
                cols = [tile[1] for tile in calibrated]
                center = (min(cols) + max(cols)) / 2
                spare = min(compatible,
                            key=lambda tile: (abs(tile[0] - origin[0]), abs(tile[1] - center), tile),
                            default=None)
                if spare is not None:
                    simulate(shadow, steps, uid, spare, targets[uid][1])
                    staged.add(uid)
                    break
                if free_hand:
                    simulate(shadow, steps, uid, None, None)
                    staged.add(uid)
                    break
            else:
                raise ValueError("The layout needs a compatible empty board tile or a free hand slot")


def as_tile(key: str) -> tuple[int, int]:
    row, col = key.split(",")
    return int(row), int(col)


def permits(tiles: dict, profile: dict, tile: tuple[int, int]) -> bool:
    kind = tiles.get(tile_key(tile))
    return kind == "melee" or (kind == "ranged" and profile["cls"] != "melee")


def at_destination(loc: dict, target: tuple) -> bool:
    tile, direction = target
    return loc.get("key") == tile_key(tile) and loc["piece"].dir == direction


def validate_ops(ops: list, profiles: dict, targets: dict) -> None:
    seen = set()
    for op in ops:
        uid = op.get("uid")
        if op.get("op") != "move" or uid not in targets or uid in seen:
            raise ValueError("Layout operations must be unique moves of selected UIDs")
        key = profiles[uid]["key"]
        source = None if op["from"] is None else tuple(op["from"])
        if source != (as_tile(key) if key else None) or (tuple(op["to"]), op["dir"]) != targets[uid]:
            raise ValueError(f"Move for UID {uid} disagrees with the snapshot or final layout")
        seen.add(uid)
    for uid, (tile, direction) in targets.items():
        profile = profiles[uid]
        if (profile["key"], profile["dir"]) != (tile_key(tile), direction) and uid not in seen:
            raise ValueError(f"Layout operations omit a changed position or facing for UID {uid}")


def simulate(shadow: Track, steps: list[LayoutStep], uid: int, target: tuple | None, direction: str | None) -> None:
    loc = shadow._loc_by_uid(uid)
    source = as_tile(loc["key"]) if loc["area"] == "board" else None
    hand_slot = loc.get("idx") if source is None else (shadow.withdraw_slot(uid) if target is None else None)
    if target is None:
        shadow.bench(uid)
    else:
        shadow.move(uid, target, direction)
    steps.append(LayoutStep(uid, source, target, hand_slot, direction))
