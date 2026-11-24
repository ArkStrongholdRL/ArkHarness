"""Sell a named operator from a calibrated board tile through MaaFramework."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .deploy import has_facing_marker, read_board, tile_point
from .native import Frame
from .session import task_recognition
from .read import ROOT, clean, integer, read_json


class SaleNotSent(RuntimeError):
    """Board sale preflight failed before the sell button was clicked."""


class SellRecognition(CustomRecognition):
    def __init__(self, stage: str, source: tuple[int, int], layout: dict, name: str,
                 expected_price: int | None = None, *, expected_round: int | None = None,
                 stage_confirmed: bool = False):
        super().__init__()
        self.stage = stage
        self.source = source
        self.layout = layout
        self.name = name
        self.expected_price = expected_price
        self.expected_round = expected_round
        self.stage_confirmed = stage_confirmed
        self.before = None
        self.price = None
        self.source_marked = False
        self.result = None
        self.reason = "The original preparation board was not recognized"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        return self.preparation(frame, kind)

    def preparation(self, frame: Frame, kind: str) -> tuple | None:
        if not frame.hit("ArkRestText"):
            return None
        if kind != "verify":
            if frame.scene() != "preparation":
                return None
            needs_stage = not self.stage_confirmed and kind != "reset" and self.before is None
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
            return self.quote(frame, state)
        if kind == "pick":
            if not self.stage_confirmed or self.expected_price in (None, 0):
                self.source_marked = has_facing_marker(frame, self.source)
            if not self.stage_confirmed and not self.source_marked:
                self.reason = "The source tile did not have a verified operator marker"
                return None
            self.before = state
            return (*self.source, 1, 1)
        if self.before is None or self.price is None:
            return None
        before = self.before
        if (current_round, funds) != (before["round"], before["funds"] + self.price):
            self.reason = f"The expected sale proceeds were not verified: {state}"
            return None
        if self.price == 0 and (not self.source_marked or has_facing_marker(frame, self.source)):
            self.reason = "The source tile did not verify clear after the zero-price sale"
            return None
        self.result = {"before": before, "after": state, "price": self.price}
        return (0, 0, 1, 1)

    def quote(self, frame: Frame, state: dict) -> tuple | None:
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
        dx, dy, width, height = self.layout["priceROIFromButton"]
        price = integer(frame.text("Board.SellButtonPrice", roi=[button[0] + dx, button[1] + dy, width, height]),
                        r"[+＋]([0-9]+)")
        if price is None:
            self.reason = "The sale price next to the button was unreadable"
            return None
        if self.expected_price is not None and price != self.expected_price:
            self.reason = f"The displayed sale price was {price}, expected {self.expected_price}"
            return None
        if self.before is not None and (state["round"], state["funds"]) != (
            self.before["round"], self.before["funds"]
        ):
            self.reason = "The preparation round or balance changed while opening operator details"
            return None
        self.before = state
        self.price = price
        return tuple(button)


def sell_board(tasker: Tasker, slots: dict, board: dict, name: str, row: int, col: int, *,
               inspect: bool = False, expected_price: int | None = None, expected_round: int | None = None,
               stage_confirmed: bool = False) -> dict:
    name = clean(name)
    if not name:
        raise ValueError("An exact displayed operator name is required")
    stage = board["stageId"]
    source = tile_point(board, row, col, view_name="normal")
    recognition = SellRecognition(stage, source, slots["sale"], name, expected_price,
                                  expected_round=expected_round, stage_confirmed=stage_confirmed)
    with task_recognition(tasker, "ArkSell", recognition):
        entry = "Board.InspectSale" if inspect else "Board.Sell"
        override = {"Board.OpenSellDetails": {"next": ["Board.ReadSaleQuote"]}} if inspect else {}
        job = tasker.post_task(entry, override).wait()
        detail = job.get()
        nodes = [node.name for node in detail.nodes] if detail else []
    target = {"stageId": stage, "name": name, "row": row, "col": col}
    if inspect and job.succeeded and nodes and nodes[-1] == "Board.ReadSaleQuote" and recognition.price is not None:
        return {**target, "price": recognition.price, "before": recognition.before}
    if job.succeeded and nodes and nodes[-1] == "Board.SellVerified" and recognition.result:
        return {**target, **recognition.result}
    if "Board.ClickSell" in nodes:
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
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    args = parser.parse_args()
    if args.expected_price is not None and args.expected_price < 0:
        parser.error("--expected-price must be nonnegative")
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        controller = connect(args.adb, args.address)
        tasker = load_ocr(controller, args.ocr_resource)
        result = sell_board(tasker, read_json(ROOT / "calib/slots.json"), read_board(args.board),
                            args.name, args.row, args.col, inspect=args.inspect, expected_price=args.expected_price)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
