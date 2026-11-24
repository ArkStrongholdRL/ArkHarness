"""Execute a chosen layout on the calibrated board using MFW action receipts."""

from collections.abc import Callable
from time import perf_counter

from maa.tasker import Tasker

from ..layout import LayoutStep, as_tile, plan_layout
from ..track import Track
from .deploy import deploy, move_board
from .read import Reader
from .withdraw import withdraw_board


class LayoutUnverified(RuntimeError):
    def __init__(self, message: str, receipts: list[dict], areas: tuple[str, ...]):
        super().__init__(message)
        self.receipts = tuple(receipts)
        self.areas = areas


def execute_layout(tasker: Tasker, reader: Reader, calibration: dict, track: Track,
                   screen: dict, state: dict, result: dict, *,
                   operator_names: dict[int, str] | None = None,
                   before_step: Callable[[LayoutStep], None] | None = None,
                   on_step: Callable[[], None] | None = None) -> dict:
    """Apply only checked moves to Track. Uncertain outcomes require a new readback."""
    start = perf_counter()
    match = screen["match"]
    current_round = match.get("round")
    if (match.get("stageId") != calibration["stageId"] or type(current_round) is not int
            or state["deployCap"] != screen["self"].get("deployCap")):
        raise ValueError("Layout and calibration must match the current stage, round and capacity")
    if screen["self"].get("deployCount") != len(track.board):
        raise ValueError("The deployment count differs from the saved board")
    leader = current_round in {round_number for round_number in (
        (screen.get("config") or {}).get("bossRound") or track.gd.boss_round,
        (screen.get("config") or {}).get("hiddenRound") or track.gd.hidden_round,
    ) if round_number is not None}
    if leader and calibration.get("deployField") != "bossLeft":
        raise ValueError("The solo leader round requires its own left-field calibration")
    if not leader and calibration.get("deployField", "normal") != "normal":
        raise ValueError("The normal round requires normal field calibration")
    if not all(calibration["views"].get(view) for view in ("normal", "drag")):
        raise ValueError("Layout requires both preparation and drag calibration")
    calibrated = set.intersection(*({tuple(cell[:2]) for cell in calibration["views"][view]["groundCenters"]}
                                    for view in ("normal", "drag")))
    steps = plan_layout(track, state, result, calibrated)
    names = {}
    for step in steps:
        piece = track._loc_by_uid(step.uid)["piece"]
        name = (operator_names or {}).get(step.uid) or (track.gd.chess(piece.id) or {}).get("name")
        if not name:
            raise ValueError(f"No displayed operator name for UID {step.uid}")
        names[step.uid] = name

    receipts = []
    try:
        for step in steps:
            loc = track._loc_by_uid(step.uid)
            live_source = as_tile(loc["key"]) if loc and loc["area"] == "board" else None
            if loc is None or live_source != step.source:
                raise RuntimeError(f"Tracked source changed for UID {step.uid}")
            if step.target is None and track.withdraw_slot(step.uid) != step.hand_slot:
                raise RuntimeError("The expected free hand slot changed before withdrawal")
            if step.source is None and (loc["area"] != "hand" or loc["idx"] != step.hand_slot):
                raise RuntimeError(f"UID {step.uid} is no longer in the planned hand slot")
            action_start = perf_counter()
            name = names[step.uid]
            if before_step:
                before_step(step)
            if step.target is None:
                receipt = withdraw_board(tasker, reader.slots, calibration, name, *step.source, step.hand_slot,
                                         expected_round=current_round, stage_confirmed=True)
                track.bench(step.uid)
                operation = "bench"
            elif step.source is None:
                receipt = deploy(tasker, reader.slots, calibration, reader.stages, step.hand_slot,
                                 name, *step.target, step.direction, expected_round=current_round,
                                 stage_confirmed=True)
                track.move(step.uid, step.target, step.direction)
                operation = "deploy"
            else:
                receipt = move_board(tasker, calibration, reader.stages, name, *step.source,
                                     *step.target, step.direction, expected_round=current_round,
                                     stage_confirmed=True)
                track.move(step.uid, step.target, step.direction)
                operation = "move"
            receipts.append({"op": operation, "uid": step.uid, "ms": round((perf_counter() - action_start) * 1000),
                             "receipt": receipt})
            if on_step:
                on_step()
    except RuntimeError as error:
        areas = ("board", "hand") if step.source is None or step.target is None else ("board",)
        raise LayoutUnverified(f"Layout stopped after {len(receipts)} verified action(s), UID {step.uid}, "
                               f"source {step.source}, target {step.target}, hand slot {step.hand_slot}: {error}",
                               receipts, areas) from error
    return {"stageId": calibration["stageId"], "round": current_round, "board": result["board"],
            "actions": receipts, "totalMs": round((perf_counter() - start) * 1000)}
