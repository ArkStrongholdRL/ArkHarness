"""Reconcile the two possible outcomes of one interrupted layout operation."""

from copy import copy, deepcopy

from .checkpoint import restore_track
from .layout import as_tile
from .screen.board_read import inspect_board_positions
from .screen.equipment_read import read_equipment_slots
from .screen.hand import read_hand
from .screen.shop import set_shop_open
from .track import TEMP


def layout_outcomes(track, step):
    loc = track._loc_by_uid(step["uid"])
    source, target = step["source"], step["target"]
    if not loc or loc["piece"].id != step["id"]:
        raise ValueError("Interrupted layout step no longer matches the saved operator")
    if source is None:
        if loc["area"] != "hand" or loc["idx"] != step["handSlot"]:
            raise ValueError("Interrupted deployment no longer matches its hand slot")
    elif loc.get("key") != f"{source[0]},{source[1]}":
        raise ValueError("Interrupted layout step no longer matches its source tile")
    changed = copy(track)
    changed.hand, changed.temp, changed.board = deepcopy((track.hand, track.temp, track.board))
    if target is None:
        if track.withdraw_slot(step["uid"]) != step["handSlot"] or not changed.bench(step["uid"]):
            raise ValueError("Interrupted withdrawal no longer matches its destination slot")
    else:
        key = f"{target[0]},{target[1]}"
        if key in track.board or not changed.move(step["uid"], target, step.get("verifiedFacing")):
            raise ValueError("Interrupted placement no longer matches its destination tile")
    return changed


def matches_piece(observed, expected):
    if observed is None:
        return expected is None
    return observed.get("status", "occupied") == "occupied" and expected is not None and (observed.get("id") == expected.id
                                     or observed.get("id") is None and expected.id in observed.get("candidateIds", ()))


def recover_layout_step(tasker, reader, board, track, step, screen):
    changed = layout_outcomes(track, step)
    piece = track._loc_by_uid(step["uid"])["piece"]
    summon_slot = step["handSlot"] if piece.kind == "token" else None
    return reconcile_layout(tasker, reader, board, track, changed, screen, step,
                            summon_slot=summon_slot)


def recover_layout_batch(tasker, reader, board, track, baseline, screen, pending=None):
    before = copy(track)
    before._bounty_picks = {}
    restore_track(before, {"track": baseline["holdings"]})
    count = screen["self"]["deployCount"]
    if before.deployed_chess_count() != count:
        raise RuntimeError(f"Neither the saved board nor the board before layout matches the live count ({count})")
    possible = layout_outcomes(track, pending) if pending else None
    return reconcile_layout(tasker, reader, board, track, before, screen, baseline,
                            possible=possible, require_count=True)


def reconcile_layout(tasker, reader, board, track, changed, screen, scope, *,
                     summon_slot=None, possible=None, require_count=False):
    rnd, own = screen["match"]["round"], screen["self"]
    if scope["round"] != rnd or scope["stageId"] != board["stageId"]:
        raise RuntimeError("The interrupted layout step is from a different preparation round or stage")
    funds, cap, count = (own.get(key) for key in ("funds", "deployCap", "deployCount"))
    if any(type(value) is not int for value in (funds, cap, count)):
        raise RuntimeError("Preparation HUD is incomplete for layout recovery")
    def signature(piece):
        return (piece.kind, piece.uid, piece.id, piece.count) if piece else None

    alternatives = [changed] if possible is None else [changed, possible]
    slots = sorted({i for candidate in alternatives for i, (before, after) in enumerate(zip(track.hand, candidate.hand))
                    if signature(before) != signature(after)} | ({summon_slot} if summon_slot is not None else set()))
    rack_slots = sorted({i for candidate in alternatives
                         for i, (before, after) in enumerate(zip(track.temp[:TEMP], candidate.temp[:TEMP]))
                         if signature(before) != signature(after)})
    keys = sorted({k for candidate in alternatives for k in track.board.keys() | candidate.board.keys()
                   if signature(track.board.get(k)) != signature(candidate.board.get(k))})
    tiles = [as_tile(key) for key in keys]
    view = set_shop_open(tasker, False)
    try:
        if (view["before"]["round"], view["before"]["funds"]) != (rnd, funds):
            raise RuntimeError("Preparation changed before layout recovery")
        hand = read_hand(tasker, reader, slots) if slots else None
        if hand and (hand["stageId"], hand["round"], hand["funds"], hand["remaining"]) != (
                board["stageId"], rnd, funds, cap - count):
            raise RuntimeError("The affected hand slots were not read in the original preparation state")
        observed_hand = {row["screenSlot"]: None if row["status"] == "empty" else row
                         for row in hand["slots"]} if hand else {}
        observed_board = inspect_board_positions(tasker, reader, board, tiles, (rnd, funds, cap - count))
    finally:
        restored = set_shop_open(tasker, view["before"]["expanded"])["after"]
    if (restored["round"], restored["funds"]) != (rnd, funds):
        raise RuntimeError("Preparation changed during layout recovery")
    rack = read_equipment_slots(tasker, reader, rack_slots) if rack_slots else None
    if rack and (rack["round"], rack["funds"]) != (rnd, funds):
        raise RuntimeError("The affected rack slots were not read in the original preparation state")
    observed_rack = {row["screenSlot"]: None if row["status"] == "empty" else row
                     for row in rack["slots"]} if rack else {}
    if observed_hand.keys() != set(slots) or observed_rack.keys() != set(rack_slots):
        raise RuntimeError("The affected holding slots could not all be read")

    def matches(candidate):
        return ((not require_count or candidate.deployed_chess_count() == count)
                and all(matches_piece(observed_hand[i], candidate.hand[i]) for i in slots)
                and all(matches_piece(observed_rack[i], candidate.temp[i]) for i in rack_slots)
                and all(matches_piece(observed_board[tile], candidate.board.get(key)) for key, tile in zip(keys, tiles)))

    outcomes = [(False, track), (True, changed)]
    verified = [(applied, candidate) for applied, candidate in outcomes if matches(candidate)]
    if len(verified) != 1:
        raise RuntimeError(f"Could not determine the interrupted layout outcome for hand slots {slots} and tiles {tiles}")
    applied, candidate = verified[0]
    if applied:
        track.hand, track.temp, track.board, track._uid = candidate.hand, candidate.temp, candidate.board, candidate._uid
    return {"applied": applied, "handSlots": slots, "rackSlots": rack_slots, "tiles": tiles}
