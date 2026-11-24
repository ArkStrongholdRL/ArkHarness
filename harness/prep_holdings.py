"""Choose which holdings need a screen read at the start of regulation prep."""

from collections import Counter

from .track import TEMP, Track, rightmost_free


AREAS = ("hand", "temp", "board")
CHESS_GRANTS = {"SERVER_POOL_CHAR", "SERVER_MOST_BOND"}
ITEM_GRANTS = {
    "SERVER_GAIN_EQUIP", "SERVER_GAIN_RANDOM_EQUIP", "SERVER_GAIN_RANDOM_EQUIP_CHESS_IN_POOL",
    "SERVER_POOL_EQUIP", "SERVER_UPGRADE_SELF_EQUIP",
}
TRIGGER_GAIN = {"SERVER_TRIGGER_ANOTHER", "SERVER_TRIGGER_FRONT_COUNT"}
FRONT = {"UP": (1, 0), "RIGHT": (0, 1), "DOWN": (-1, 0), "LEFT": (0, -1)}
FIXED_GRANT = "preparation_start_gain_chess_from_round"
PERIODIC_GRANT = "preparation_start_gain_chess_every_n_round"


def scheduled_rack_slot(track: Track, band_id: str, current_round: int) -> tuple[int, str] | None:
    """Locate a single known, nonmerging grant in the rightmost free rack slot."""
    due = [buff for buff in (track.gd.band(band_id) or {}).get("buffs", [])
           if buff.get("key") == FIXED_GRANT
           and (buff.get("bb") or {}).get("round") == current_round
           and (buff.get("bbStr") or {}).get("type") == "equip"]
    if len(due) != 1 or (due[0]["bb"].get("count") or 1) != 1:
        return None
    item_id = due[0]["bbStr"].get("chess")
    item = track.gd.item(item_id)
    if not item or item.get("mergeable") or item.get("canFuse"):
        return None
    slot = rightmost_free(track.temp[:TEMP])
    return (slot, item_id) if slot >= 0 else None


def prep_read_plan(track: Track, band_id: str | None, current_round: int, tracked_round: int | None,
                   deploy_count: int | None, *, force: bool = False) -> dict[str, str]:
    """Plan reads from the previous known roster and deterministic round schedule."""
    if force or tracked_round not in (current_round - 1, current_round) or not track.gd.band(band_id):
        reason = "requested or uncertain holdings" if force else "initial calibration"
        return dict.fromkeys(AREAS, reason)
    board = {tuple(map(int, key.split(","))): piece for key, piece in track.board.items()
             if piece.kind == "chess"}
    reads = {}
    if deploy_count != len(board):
        reads["board"] = "deployment count changed"
    if tracked_round == current_round:
        return reads
    gd = track.gd
    if current_round in (gd.boss_round, gd.hidden_round) or tracked_round in (gd.boss_round, gd.hidden_round):
        reads.update(hand="leader field can return deployed operators", board="leader field transition")

    grants = 0
    for buff in gd.band(band_id).get("buffs") or []:
        key, params = buff.get("key"), buff.get("bb") or {}
        round_start = params.get("round")
        if key == "gain_bond_char_per_round" and isinstance(round_start, int):
            interval = max(1, params.get("preround") or 1)
            due = current_round >= round_start and (current_round - round_start) % interval == 0
        elif key == PERIODIC_GRANT and isinstance(round_start, int):
            due = round_start > 0 and current_round % round_start == 0
        else:
            due = round_start == current_round
        if not due:
            continue
        if key == "prep_start_random_equip_in_round":
            reads["temp"] = "scheduled random squad equipment"
        if key in {FIXED_GRANT, PERIODIC_GRANT, "prep_start_gain_chess_from_pool_in_round", "gain_bond_char_per_round"}:
            if (buff.get("bbStr") or {}).get("type") == "equip":
                reason = "scheduled fixed equipment" if key == FIXED_GRANT else "scheduled squad equipment"
                reads["temp"] = "multiple squad equipment grants" if "temp" in reads else reason
            else:
                reads["hand"] = "scheduled squad operator"
                grants += params.get("count") or 1

    def effects(piece, event):
        record = gd.chess(piece.id) or {}
        return [g for gid in record.get("garrisonIds") or [] if (g := gd.garrison(gid))
                and g.get("eventType") == event]

    row_counts = Counter(row for row, _ in board)
    for (row, col), piece in board.items():
        effects_to_read = []
        for effect in effects(piece, "SERVER_PREP_START"):
            key = effect.get("effectKey") or effect.get("effectType")
            params, cond = effect.get("bb") or {}, effect.get("bbStr") or {}
            if cond.get("conditionkey") == "character_same_row" and row_counts[row] < params.get("check_count", 0):
                continue
            if key in TRIGGER_GAIN and cond.get("event") == "SERVER_GAIN":
                directions = (FRONT[piece.dir],) if piece.dir in FRONT else FRONT.values()
                for dr, dc in directions:
                    for step in range(1, (params.get("count") or 1) + 1):
                        if target := board.get((row + dr * step, col + dc * step)):
                            effects_to_read.extend(effects(target, "SERVER_GAIN"))
            else:
                effects_to_read.append(effect)
        for effect in effects_to_read:
            key = effect.get("effectKey") or effect.get("effectType")
            if key in CHESS_GRANTS:
                reads["hand"] = "operator grant at prep start"
                grants += (effect.get("bb") or {}).get("count") or 1
            if key == "SERVER_UPGRADE_SELF_EQUIP":
                equipped = [(gd.item(item.id) or {}) for item in piece.items]
                if not any(not item.get("isGolden") and gd.item(item.get("upgradeChessId") or item.get("goldenId"))
                           for item in equipped):
                    continue
            if key in ITEM_GRANTS:
                reads["temp"] = "equipment effect at prep start"

    # A settlement transformation can replace a fielded operator and return its gear.
    all_chess = list(board.values()) + [p for p in track.hand + track.temp if p and p.kind == "chess"]
    if any(b.get("key") == "char_chess_transformation_equip"
           for piece in all_chess for item in piece.items for b in (gd.item(item.id) or {}).get("buffs", [])):
        return dict.fromkeys(AREAS, "post-battle transformation")

    copies = Counter(gd.base_id_of(p.id) for p in all_chess if not gd.is_golden(p.id))
    if grants:
        merge_bases = {base for base, count in copies.items() if count + grants >= gd.merge_count(base) > 0}
        if any(not gd.is_golden(p.id) and gd.base_id_of(p.id) in merge_bases for p in board.values()):
            reads["board"] = "fielded operator could merge"
        if any(p.items and not gd.is_golden(p.id) and gd.base_id_of(p.id) in merge_bases for p in all_chess):
            reads["temp"] = "a merge could return equipped items"

    victoria = "victoriaShip"
    members = {gd.base_id_of(p.id) for p in board.values() if victoria in (gd.chess(p.id) or {}).get("bonds", [])}
    thresholds = (gd.bond(victoria) or {}).get("thresholds") or []
    if thresholds and len(members) >= thresholds[0]:
        reads["temp"] = "active Victoria milestone can grant equipment"
    overflow = [piece for piece in track.temp[TEMP:] if piece]
    if any(piece.kind != "item" for piece in overflow):
        reads.setdefault("hand", "tracked overflow can refill the bench")
    if any(piece.kind == "item" for piece in overflow):
        reads["temp"] = "tracked overflow can refill the rack"
    return reads
