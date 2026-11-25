"""Sell a named operator from a calibrated board tile through MaaFramework."""

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
from .read import ROOT, clean, integer, read_json


class SaleNotSent(RuntimeError):
    """Board sale preflight failed before the sell button was clicked."""


class SellRecognition(CustomRecognition):
    def __init__(self, stage: str, source: tuple[int, int], layout: dict, name: str,
                 expected_price: int | None = None, *, expected_round: int | None = None,
                 stage_confirmed: bool = False, by_balance: bool = False):
        super().__init__()
        self.stage = stage
        self.source = source
        self.layout = layout
        self.name = name
        self.expected_price = expected_price
        self.expected_round = expected_round
        self.stage_confirmed = stage_confirmed
        self.by_balance = by_balance
        self.reason = "The original preparation board was not recognized"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        return self.preparation(frame, kind, argv.task_detail.nodes)

    def preparation(self, frame: Frame, kind: str, nodes) -> CustomRecognition.AnalyzeResult | tuple | None:
        if not frame.hit("ArkRestText"):
            return None
        if kind != "verify":
            if frame.scene() != "preparation":
                return None
            needs_stage = not self.stage_confirmed and kind != "reset" and completed_data(
                nodes, "Board.OpenSellDetails") is None
            if needs_stage and frame.stage_id() != self.stage:
                self.reason = f"Could not verify the terrain for {self.stage}"
                return None
            expanded = frame.hit("Shop.Expanded")
            if kind == "close":
                labels = frame.rows("Shop.CollapseText") if expanded else []
                return tuple(labels[0].box) if len(labels) == 1 else None
            if expanded:
                return None
            if kind == "reset":
                return (0, 0, 1, 1) if frame.hit("Withdraw.Icon") or frame.hit("Screen.OperatorDetails") else None
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        if current_round is None or funds is None:
            self.reason = "The round or balance was unreadable"
            return None
        if self.expected_round is not None and current_round != self.expected_round:
            self.reason = f"Expected sale in round {self.expected_round}, read {current_round}"
            return None
        state = {"round": current_round, "funds": funds}
        if kind == "quote":
            picked = completed_data(nodes, "Board.OpenSellDetails")
            return self.quote(frame, state, picked["before"] if picked is not None else None)
        if kind == "pick":
            marked = ((not self.stage_confirmed or not self.by_balance and self.expected_price in (None, 0))
                      and has_facing_marker(frame, self.source))
            if not self.stage_confirmed and not marked:
                self.reason = "The source tile did not have a verified operator marker"
                return None
            return self.AnalyzeResult((*self.source, 1, 1), {"before": state, "sourceMarked": marked})
        quoted = completed_data(nodes, "Board.ClickSell")
        if kind != "verify" or quoted is None:
            return None
        before, price = quoted["before"], quoted["price"]
        if self.by_balance:
            price = funds - before["funds"]
            if price <= 0:
                self.reason = f"The balance did not increase after selling: {state}"
                return None
        if (current_round, funds) != (before["round"], before["funds"] + price):
            self.reason = f"The expected sale proceeds were not verified: {state}"
            return None
        if price == 0:
            picked = completed_data(nodes, "Board.OpenSellDetails")
            if picked is None or not picked["sourceMarked"] or has_facing_marker(frame, self.source):
                self.reason = "The source tile did not verify clear after the zero-price sale"
                return None
        return self.AnalyzeResult((0, 0, 1, 1), {"before": before, "after": state, "price": price})

    def quote(self, frame: Frame, state: dict, before: dict | None) -> CustomRecognition.AnalyzeResult | None:
        if not frame.hit("Withdraw.Icon"):
            self.reason = "Board operator details were not recognized"
            return None
        names = [clean(row.text) for row in frame.rows("Board.CarriedName")]
        if self.name not in names:
            self.reason = f"Expected {self.name} in board details, read {names}"
            return None
        buttons = frame.rows("Board.SellButtonText", roi=self.layout["boardDetailsButtonROI"])
        if len(buttons) != 1:
            self.reason = "The sale button in board details was not uniquely verified"
            return None
        button = buttons[0].box
        if before is not None and state != before:
            self.reason = "The preparation round or balance changed while opening operator details"
            return None
        if self.by_balance:
            return self.AnalyzeResult(tuple(button), {"before": state, "price": None})
        dx, dy, width, height = self.layout["priceROIFromButton"]
        price = integer(frame.text("Board.SellButtonPrice", roi=[button[0] + dx, button[1] + dy, width, height]),
                        r"[+＋]([0-9]+)")
        if price is None:
            self.reason = "The sale price next to the button was unreadable"
            return None
        if self.expected_price is not None and price != self.expected_price:
            self.reason = f"The displayed sale price was {price}, expected {self.expected_price}"
            return None
        return self.AnalyzeResult(tuple(button), {"before": state, "price": price})


def sell_board(tasker: Tasker, slots: dict, board: dict, name: str, row: int, col: int, *,
               inspect: bool = False, expected_price: int | None = None, expected_round: int | None = None,
               stage_confirmed: bool = False, by_balance: bool = False) -> dict:
    name = clean(name)
    if not name:
        raise ValueError("An exact displayed operator name is required")
    stage = board["stageId"]
    source = tile_point(board, row, col, view_name="normal")
    recognition = SellRecognition(stage, source, slots["sale"], name, expected_price,
                                  expected_round=expected_round, stage_confirmed=stage_confirmed,
                                  by_balance=by_balance and not inspect)
    with task_recognition(tasker, "ArkSell", recognition):
        entry = "Board.InspectSale" if inspect else "Board.Sell"
        override = {"Board.OpenSellDetails": {"next": ["Board.ReadSaleQuote"]}} if inspect else {}
        job = run_task(tasker, entry, override)
    target = {"stageId": stage, "name": name, "row": row, "col": col}
    terminal = "Board.ReadSaleQuote" if inspect else "Board.SellVerified"
    result = completed_data(job.nodes, terminal)
    if job.finished(terminal) and result is not None:
        return {**target, **result}
    if "Board.ClickSell" in job.names:
        raise RuntimeError(f"Sale was attempted once and is unverified: {recognition.reason}")
    raise SaleNotSent(f"No sale click was sent at ({row}, {col}) for {name}: {recognition.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json")
    parser.add_argument("--name", required=True, help="Exact displayed operator name")
    parser.add_argument("--row", required=True, type=int, help="Engine tile row")
    parser.add_argument("--col", required=True, type=int, help="Engine tile column")
    parser.add_argument("--inspect", action="store_true", help="Open operator details if needed and read the quote without selling")
    parser.add_argument("--expected-price", type=int, help="Require this amount in the displayed quote before selling")
    add_session_args(parser)
    args = parser.parse_args()
    if args.expected_price is not None and args.expected_price < 0:
        parser.error("--expected-price must be nonnegative")
    try:
        tasker = session(args.adb, args.address, args.ocr_resource).tasker
        result = sell_board(tasker, read_json(ROOT / "calib/slots.json"), read_board(args.board),
                            args.name, args.row, args.col, inspect=args.inspect, expected_price=args.expected_price)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
