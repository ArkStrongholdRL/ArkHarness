"""Return a named board operator to an expected hand slot through MaaFramework."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import add_session_args, session
from .deploy import has_facing_marker, read_board, tile_point
from .native import Frame, completed_data
from .session import run as run_task, task_recognition
from .read import ROOT, clean, integer, read_json, screen_integer


class WithdrawNotSent(RuntimeError):
    """The withdrawal click never ran."""


class WithdrawRecognition(CustomRecognition):
    def __init__(self, stage: str, source: tuple[int, int], hand: tuple[int, int], name: str,
                 expected_round: int | None, stage_confirmed: bool, *, summon: bool = False):
        super().__init__()
        self.stage = stage
        self.source = source
        self.hand = hand
        self.name = name
        self.expected_round = expected_round
        self.stage_confirmed = stage_confirmed
        self.summon = summon
        self.reason = "The preparation board was not verified"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if frame.scene() != "preparation" or not frame.hit("ArkRestText") or frame.hit("Shop.Expanded"):
            return None
        kind = json.loads(argv.custom_recognition_param)
        nodes = argv.task_detail.nodes if kind not in {"reset", "pick"} else ()
        pick = completed_data(nodes, "Withdraw.Pick")
        before = pick["before"] if pick is not None else None
        if kind == "hand":
            if (before is not None and (self.summon or frame.hit("HandSale.Button") or frame.hit("Screen.OperatorDetails"))
                    and not frame.hit("Withdraw.Icon") and clean(frame.text("Board.CarriedName")) == self.name):
                return (0, 0, 1, 1)
            self.reason = "The operator name was not verified in the expected hand slot"
            return None
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        remaining = None if self.summon else screen_integer(frame, "Screen.DeployCollapsed")
        if current_round is None or funds is None:
            self.reason = "The round or balance could not be read"
            return None
        if self.expected_round is not None and current_round != self.expected_round:
            self.reason = f"Expected preparation round {self.expected_round}, read {current_round}"
            return None
        state = {"round": current_round, "funds": funds, "remaining": remaining}
        if kind == "reset":
            details_open = frame.hit("Screen.OperatorDetails") or frame.hit("Withdraw.Icon")
            return (0, 0, 1, 1) if details_open else None
        if kind == "pick":
            if frame.hit("Screen.OperatorDetails") or (not self.stage_confirmed and frame.stage_id() != self.stage):
                self.reason = "The requested source tile is not verified on this map"
                return None
            marked = not self.stage_confirmed and has_facing_marker(frame, self.source)
            return self.AnalyzeResult((*self.source, 1, 1), {"before": state, "sourceMarked": marked})
        if before is None or (current_round, funds) != (before["round"], before["funds"]):
            self.reason = "Withdraw did not preserve the preparation round and balance"
            return None
        if kind == "withdraw":
            return self.button(frame)
        hand_confirmed = any(node.name == "Withdraw.CloseHandDetails" and node.completed for node in nodes)
        if frame.hit("Screen.OperatorDetails") or frame.hit("HandSale.Button") or frame.hit("Withdraw.Icon"):
            self.reason = ("The confirmed hand details did not close" if hand_confirmed
                           else "The board details were still open before selecting the hand slot")
            return None
        remaining_increased = before["remaining"] is not None and remaining == before["remaining"] + 1
        clicked = any(node.name == "Withdraw.Click" and node.completed for node in nodes)
        if self.summon and not clicked:
            self.reason = "The summon withdrawal click was not verified"
            return None
        if (not self.summon and not self.stage_confirmed and not remaining_increased
                and (not pick["sourceMarked"] or has_facing_marker(frame, self.source))):
            self.reason = "Neither the deployment count nor the source marker confirmed the withdrawal"
            return None
        if kind == "cleared":
            return (*self.hand, 1, 1)
        if kind == "verified":
            counted = not self.summon and self.stage_confirmed and remaining_increased and clicked
            if not (counted or hand_confirmed):
                self.reason = "The operator was not confirmed in the expected hand slot"
                return None
            return self.AnalyzeResult((0, 0, 1, 1), {"before": before, "after": state, "sourceCleared": True})
        return None

    def button(self, frame: Frame) -> tuple | None:
        observed = clean(frame.text("Board.CarriedName"))
        if observed != self.name:
            self.reason = f"Expected board operator {self.name}, read {observed or 'no name'}"
            return None
        result = frame.node("Withdraw.Icon")
        if result is None or len(result.filtered_results) != 1:
            self.reason = "The selected operator's withdraw icon was not uniquely verified"
            return None
        return tuple(result.filtered_results[0].box)


def withdraw_board(tasker: Tasker, slots: dict, board: dict, name: str, row: int, col: int,
                   hand_slot: int, *, expected_round: int | None = None, stage_confirmed: bool = False,
                   summon: bool = False) -> dict:
    name = clean(name)
    hand_centers = slots["hand"]["centersByView"]["normal"]
    if not name or type(hand_slot) is not int or not 0 <= hand_slot < len(hand_centers):
        raise ValueError("An exact displayed name and a valid expected hand slot are required")
    stage = board["stageId"]
    source = tile_point(board, row, col, view_name="normal")
    recognition = WithdrawRecognition(stage, source, tuple(hand_centers[hand_slot]), name, expected_round,
                                      stage_confirmed, summon=summon)
    with task_recognition(tasker, "ArkWithdraw", recognition):
        job = run_task(tasker, "Withdraw.Board")
    result = completed_data(job.nodes, "Withdraw.Verified")
    if not job.finished("Withdraw.Verified") or result is None:
        if not any(node.name == "Withdraw.Click" and node.action is not None for node in job.nodes):
            raise WithdrawNotSent(f"Withdraw was not sent: {recognition.reason}")
        raise RuntimeError(f"Withdraw outcome unverified: {recognition.reason}")
    return {"stageId": stage, "name": name, "row": row, "col": col, "handSlot": hand_slot,
            **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--row", required=True, type=int)
    parser.add_argument("--col", required=True, type=int)
    parser.add_argument("--hand-slot", required=True, type=int, help="Expected zero-based hand slot after withdrawing")
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json")
    add_session_args(parser)
    args = parser.parse_args()
    try:
        tasker = session(args.adb, args.address, args.ocr_resource).tasker
        result = withdraw_board(tasker, read_json(ROOT / "calib/slots.json"), read_board(args.board),
                                args.name, args.row, args.col, args.hand_slot)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
