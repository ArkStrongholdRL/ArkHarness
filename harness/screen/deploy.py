"""Place a named operator from the hand or another calibrated board tile."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.define import MaaAdbInputMethodEnum
from maa.tasker import Tasker

from .capture import add_session_args, session
from .native import Frame, completed_data
from .read import CATALOG, ROOT, clean, integer, read_json, screen_integer
from .session import run as run_task, task_recognition


FACING_OFFSETS = {"UP": (0, -182), "RIGHT": (182, 0), "DOWN": (0, 182), "LEFT": (-182, 0)}
FACING_MARKERS = {"UP": (0, -19), "RIGHT": (34, 4), "DOWN": (0, 25), "LEFT": (-34, 4)}
PLACEMENT_TERMINALS = {"Board.PlacementVerified", "Board.PlacementWithoutSwipe"}


def read_board(path: Path) -> dict:
    return _read_board(path.resolve(), frozenset())


def _read_board(path: Path, parents: frozenset[Path]) -> dict:
    if path in parents:
        raise ValueError(f"Board calibration has a circular geometry reference at {path}")
    board = read_json(path)
    if "groundGeometry" not in board:
        return board

    geometry = _read_board((path.parent / board["groundGeometry"]).resolve(), parents | {path})
    views = {}
    for name, terrain in board["views"].items():
        if terrain is None:
            views[name] = None
            continue
        if "groundCenters" in terrain:
            views[name] = terrain
            continue
        ground = geometry["views"].get(name)
        if ground is None:
            raise ValueError(f"No shared ground geometry for view {name!r}")
        views[name] = {
            "heightCorrection": ground.get("heightCorrection", []),
            **terrain,
            "groundCenters": ground["groundCenters"],
        }
    return {**board, "views": views}


def tile_point(board: dict, row: int, col: int, *, view_name: str = "drag") -> tuple[int, int]:
    view = board["views"].get(view_name)
    if not view:
        raise ValueError(f"No {view_name} calibration for {board['stageId']} {board.get('deployField', 'normal')}")
    point = next(((x, y) for r, c, x, y in view["groundCenters"] if (r, c) == (row, col)), None)
    if point is None:
        raise ValueError(f"The requested tile is outside the calibrated {view_name} view")
    correction = next((item for item in view["heightCorrection"]
                       if (item["row"], item["col"]) == (row, col)), {})
    return round(point[0] + correction.get("dx", 0)), round(point[1] + correction.get("dy", 0))


def deployment_tiles(stage: dict, field: str, *, include_changed: bool = True) -> set[tuple[int, int]]:
    groups = stage["deployTiles"][field]
    tiles = {tuple(tile) for kind, group in groups.items() for tile in group
             if include_changed or kind in {"melee", "rangedOnly"}}
    if field == "normal":
        return tiles
    # Track keeps the engine's own-board coordinates on both leader fields.
    return {(row + 7, 20 - col if field == "bossRight" else col) for row, col in tiles if 2 <= row <= 5}


class DeployRecognition(CustomRecognition):
    def __init__(self, stage: str, source: tuple[int, int], name: str, target: tuple[int, int],
                 normal_target: tuple[int, int], facing: str, *, from_board: bool = False,
                 expected_round: int | None = None, stage_confirmed: bool = False, summon: bool = False):
        super().__init__()
        self.stage = stage
        self.source = source
        self.name = name
        self.target = target
        self.normal_target = normal_target
        self.facing = facing
        self.from_board = from_board
        self.expected_round = expected_round
        self.stage_confirmed = stage_confirmed
        self.summon = summon
        self.reason = "The calibrated preparation view was not recognized"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        if kind == "carried":
            return self.drag_target(frame)
        if kind == "facing":
            x, y = self.target
            if self.facing_picker_open(frame) and any(
                clean(row.text) == self.name for row in frame.rows("Board.CarriedName")
            ):
                return x, y, 1, 1
            self.reason = "The direction picker for the requested tile and operator was not recognized"
            return None
        before = completed_data(argv.task_detail.nodes, "Board.DeployGuard") if kind == "verify" else None
        return self.preparation(frame, kind, before)

    def facing_picker_open(self, frame: Frame) -> bool:
        x, y = self.target
        roi = [max(0, x - 240), max(0, y - 168), 200, 96]
        return frame.hit("Board.FacingCancelIcon", roi=roi) or frame.hit("Board.FacingCancelText", roi=roi)

    def drag_target(self, frame: Frame) -> tuple | None:
        if (not self.summon and not frame.hit("Board.SellText")) or not any(
            clean(row.text) == self.name for row in frame.rows("Board.CarriedName")
        ):
            self.reason = "The dragged operator was not verified by name"
            return None
        x, y = self.target
        return x, y, 1, 1

    def preparation(self, frame: Frame, kind: str,
                    before: dict | None) -> CustomRecognition.AnalyzeResult | tuple | None:
        if kind == "verify" and self.facing_picker_open(frame):
            self.reason = "The direction picker is still open"
            return None
        if frame.scene() != "preparation" or not frame.hit("ArkRestText"):
            self.reason = "The calibrated preparation view was not recognized"
            return None
        # Moving a unit can cover a stage landmark. The map was verified before the drag.
        if kind != "verify" and not self.stage_confirmed and frame.stage_id() != self.stage:
            self.reason = "The calibration did not match the current preparation view"
            return None
        expanded = frame.hit("Shop.Expanded")
        if kind == "close":
            labels = frame.rows("Shop.CollapseText") if expanded else []
            return tuple(labels[0].box) if len(labels) == 1 else None
        if expanded:
            self.reason = "The normal board view was not recovered after closing the shop"
            return None
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        remaining = screen_integer(frame, "Screen.DeployCollapsed")
        if current_round is None or remaining is None:
            self.reason = "The round or remaining deployment capacity could not be read"
            return None
        if self.expected_round is not None and current_round != self.expected_round:
            self.reason = f"Expected preparation round {self.expected_round}, read {current_round}"
            return None
        if kind == "verify":
            delta = 0 if self.from_board or self.summon else 1
            if before is None or current_round != before["round"] or before["remaining"] - remaining != delta:
                self.reason = f"After release, round={current_round}, remaining capacity={remaining}"
                return None
            if not self.stage_confirmed and not self.summon and not matches_facing(frame, self.normal_target, self.facing):
                self.reason = "The direction was not set in the requested tile's picker"
                return None
            return self.AnalyzeResult((0, 0, 1, 1), {"round": current_round, "remainingBefore": before["remaining"],
                                                    "remainingAfter": remaining, "facing": self.facing})
        if kind != "pick":
            return None
        if not self.from_board and not self.summon and remaining <= 0:
            self.reason = "No deployment position remains"
            return None
        if not self.stage_confirmed and has_facing_marker(frame, self.normal_target):
            self.reason = "The destination tile is already occupied"
            return None
        x, y = self.source
        return self.AnalyzeResult((x, y, 1, 1), {"round": current_round, "remaining": remaining})


def facing_override(target: tuple[int, int], facing: str) -> dict:
    if facing not in FACING_OFFSETS:
        raise ValueError("A valid facing direction is required")
    x, y = target
    dx, dy = FACING_OFFSETS[facing]
    return {"begin": [x, y], "end": [max(0, min(x + dx, 1279)), max(0, min(y + dy, 719))]}


def facing_marker_roi(target: tuple[int, int], facing: str) -> list[int]:
    x, y = target
    dx, dy = FACING_MARKERS[facing]
    return [x + dx - 9, y + dy - 11, 18, 22]


def matches_facing(frame: Frame, target: tuple[int, int], facing: str) -> bool:
    if frame.hit("Board.FacingMarker", roi=facing_marker_roi(target, facing)):
        return True
    if facing != "UP":
        return False
    # A front sprite can hide the middle of the arrow on the far side of the tile.
    x, y = target
    return (frame.hit("Board.FacingMarker", roi=[x - 18, y - 21, 10, 10], count=3)
            and frame.hit("Board.FacingMarker", roi=[x + 12, y - 22, 14, 12], count=4))


def completed_facing_swipes(nodes) -> int:
    return sum(node.name == "Board.ChooseFacing" and node.action is not None and node.action.success for node in nodes)


def place_operator(tasker: Tasker, board: dict, stages: dict, source: tuple[int, int],
                   name: str, row: int, col: int, facing: str, *, from_board: bool = False,
                   expected_round: int | None = None, stage_confirmed: bool = False, summon: bool = False) -> dict:
    name = clean(name)
    if not name:
        raise ValueError("An exact displayed operator name is required")
    stage = board["stageId"]
    field = board.get("deployField", "normal")
    if field != "normal" and (not stage_confirmed or expected_round is None):
        raise ValueError("A leader move requires a verified stage and round")
    if (row, col) not in deployment_tiles(stages[stage], field, include_changed=False):
        raise ValueError(f"Tile ({row}, {col}) is outside the {field} deployable area of {stage}")
    x, y = tile_point(board, row, col)
    direction = facing_override((x, y), facing)
    normal_target = tile_point(board, row, col, view_name="normal")
    source_x, source_y = source
    recognition = DeployRecognition(stage, source, name, (x, y), normal_target, facing,
                                    from_board=from_board, expected_round=expected_round,
                                    stage_confirmed=stage_confirmed, summon=summon)
    with task_recognition(tasker, "ArkDeploy", recognition):
        job = run_task(tasker, "Board.Deploy", {
            "Board.PickHand": {"target": [source_x, source_y]},
            "Board.LiftHand": {"target": [source_x, source_y - 80]},
            "Board.CancelMove": {"target": [35, 675] if from_board else [source_x, source_y]},
            "Board.MoveToTile": {"target": [x, y]},
            "Board.ChooseFacing": direction,
            "Board.PlacementWithoutSwipe": {"enabled": summon or not stage_confirmed},
        })
    names = job.names
    swipes = completed_facing_swipes(job.nodes)
    result = completed_data(job.nodes, *PLACEMENT_TERMINALS)
    if (job.finished(*PLACEMENT_TERMINALS)
            and result is not None and (swipes or summon or not stage_confirmed)):
        receipt = {**result, "facingSwipes": swipes}
        if summon and not swipes:
            receipt["facing"] = None
        return {"name": name, "stageId": stage, "row": row, "col": col, **receipt}
    if "Board.ReleaseTarget" in names:
        raise RuntimeError(f"A release was attempted; placement is unverified after {swipes} direction swipe(s): {recognition.reason}")
    if "Board.PickHand" in names:
        raise RuntimeError(f"No tile release was sent; drag recovery requested: {recognition.reason}")
    raise RuntimeError(f"No deployment gesture was sent: {recognition.reason}")


def deploy(tasker: Tasker, slots: dict, board: dict, stages: dict, hand_slot: int,
           name: str, row: int, col: int, facing: str, *, expected_round: int | None = None,
           stage_confirmed: bool = False, summon: bool = False) -> dict:
    if type(hand_slot) is not int or not 0 <= hand_slot < slots["hand"]["capacity"]:
        raise ValueError("A valid zero-based screen hand slot is required")
    source = tuple(slots["hand"]["centersByView"]["normal"][hand_slot])
    return {"handSlot": hand_slot, **place_operator(
        tasker, board, stages, source, name, row, col, facing, expected_round=expected_round,
        stage_confirmed=stage_confirmed, summon=summon)}


def move_board(tasker: Tasker, board: dict, stages: dict, name: str,
               from_row: int, from_col: int, row: int, col: int, facing: str, *,
               expected_round: int | None = None, stage_confirmed: bool = False) -> dict:
    if (from_row, from_col) == (row, col):
        raise ValueError("The source and destination board tiles must differ")
    source = tile_point(board, from_row, from_col, view_name="normal")
    result = place_operator(tasker, board, stages, source, name, row, col, facing,
                            from_board=True, expected_round=expected_round, stage_confirmed=stage_confirmed)
    return {"fromRow": from_row, "fromCol": from_col, **result}


def has_facing_marker(frame: Frame, point: tuple[int, int]) -> bool:
    return any(matches_facing(frame, point, direction) for direction in FACING_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hand-slot", type=int, help="Zero-based screen slot from left to right")
    source.add_argument("--from-row", type=int, help="Engine row of an existing board operator")
    parser.add_argument("--from-col", type=int, help="Engine column of an existing board operator")
    parser.add_argument("--name", required=True, help="Exact displayed operator name")
    parser.add_argument("--row", required=True, type=int, help="Engine tile row")
    parser.add_argument("--col", required=True, type=int, help="Engine tile column")
    parser.add_argument("--facing", required=True, choices=FACING_OFFSETS)
    add_session_args(parser, catalog=CATALOG)
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json",
                        help="Calibration file for the current map")
    args = parser.parse_args()
    if (args.from_row is None) != (args.from_col is None):
        parser.error("Use --from-row and --from-col together")
    try:
        tasker = session(args.adb, args.address, args.ocr_resource,
                                 input_methods=MaaAdbInputMethodEnum.Maatouch).tasker
        board = read_board(args.board)
        stages = read_json(args.catalog / "stages.json")
        if args.hand_slot is not None:
            result = deploy(tasker, read_json(ROOT / "calib/slots.json"), board, stages, args.hand_slot,
                            args.name, args.row, args.col, args.facing)
        else:
            result = move_board(tasker, board, stages, args.name, args.from_row, args.from_col,
                                args.row, args.col, args.facing)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
