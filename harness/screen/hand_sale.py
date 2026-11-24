"""Read or execute a named hand-slot sale through MaaFramework."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .native import Frame, completed_data
from .session import task_recognition
from .read import ROOT, clean, integer, read_json


class HandSaleRecognition(CustomRecognition):
    def __init__(self, source: tuple[int, int], name: str, expected_price: int | None):
        super().__init__()
        self.source = (*source, 1, 1)
        self.name = name
        self.expected_price = expected_price

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        if not frame.hit("ArkRestText"):
            return None
        nodes = argv.task_detail.nodes
        before = completed_data(nodes, "HandSale.Open") if kind not in {"open", "reset"} else None
        sale = completed_data(nodes, "HandSale.Click") if kind in {"check", "verified"} else None
        if kind in {"check", "verified"} and (sale is None or before is None):
            return None
        if kind == "check" and sale["price"] != 0:
            return None
        if kind == "verified" and sale["price"]:
            state = self.balance(frame)
            if state == {**before, "funds": before["funds"] + sale["price"]}:
                return self.result(before, state, sale["price"])
            return None
        if frame.scene() != "preparation" or frame.hit("Shop.Expanded"):
            return None
        buttons = frame.rows("HandSale.Button")
        details = bool(buttons) or frame.hit("Screen.OperatorDetails")
        if kind == "reset":
            return (0, 0, 1, 1) if details or frame.hit("Withdraw.Icon") else None
        state = self.balance(frame)
        if state is None:
            return None
        if kind == "open":
            return None if details else self.AnalyzeResult(self.source, state)
        if before is None:
            return None
        if kind in {"sell", "dismiss"}:
            if state != before:
                return None
            quote = self.quote(frame, buttons)
            if kind == "sell":
                if quote is None or quote[1] != self.expected_price:
                    return None
                return self.AnalyzeResult(quote[0], {"price": quote[1]})
            return self.AnalyzeResult((0, 0, 1, 1), {"price": quote[1] if quote else None}) if details else None
        if details or state != before:
            return None
        if kind == "check":
            return self.source
        if kind == "verified" and any(node.name == "HandSale.CheckSlot" and node.completed for node in nodes):
            return self.result(before, state, sale["price"])
        if kind == "unchanged":
            quoted = completed_data(nodes, "HandSale.Dismiss")
            if quoted is not None:
                return self.result(before, state, quoted["price"])
        return None

    @staticmethod
    def balance(frame: Frame) -> dict | None:
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        return None if current_round is None or funds is None else {"round": current_round, "funds": funds}

    def result(self, before: dict, after: dict, price: int | None) -> CustomRecognition.AnalyzeResult:
        return self.AnalyzeResult((0, 0, 1, 1), {"price": price, "before": before, "after": after})

    def quote(self, frame: Frame, buttons: list) -> tuple | None:
        if len(buttons) != 1 or not any(clean(row.text) == self.name for row in frame.rows("Board.CarriedName")):
            return None
        button = buttons[0].box
        x, y = button[:2]
        price = integer(frame.text("Board.SellButtonPrice", roi=[x + 55, y - 58, 58, 34]), r"[+＋]([0-9]+)")
        return (tuple(button), price) if price is not None else None


def sell_hand(tasker: Tasker, slots: dict, hand_slot: int, name: str, *,
              inspect: bool = False, expected_price: int | None = None) -> dict:
    """Quote an operator or sell it at an explicit expected price."""
    centers = slots["hand"]["centersByView"]["normal"]
    name = clean(name)
    if not name or type(hand_slot) is not int or not 0 <= hand_slot < len(centers):
        raise ValueError("An exact displayed name and valid zero-based hand slot are required")
    if (not inspect or expected_price is not None) and (type(expected_price) is not int or expected_price < 0):
        raise ValueError("A sale requires a nonnegative expected price")
    recognition = HandSaleRecognition(tuple(centers[hand_slot]), name, expected_price)
    with task_recognition(tasker, "ArkHandSale", recognition):
        if inspect:
            override = {"HandSale.Open": {"next": ["HandSale.Dismiss"]}}
        else:
            terminal = "HandSale.Verified" if expected_price else "HandSale.CheckSlot"
            override = {"HandSale.Click": {"next": [terminal]}}
        job = tasker.post_task("HandSale.Run", override).wait()
    detail = job.get()
    nodes = detail.nodes if detail else []
    result = completed_data(nodes, "HandSale.Unchanged" if inspect else "HandSale.Verified")
    if (job.succeeded and result is not None and result["price"] is not None
            and expected_price in (None, result["price"])):
        return {"handSlot": hand_slot, "name": name, **result}
    if any(node.name == "HandSale.Click" and node.action is not None for node in nodes):
        raise RuntimeError("Hand sale was attempted once and is unverified: The expected proceeds or empty slot did not verify")
    raise RuntimeError("Hand sale was not sent: The expected hand slot, operator or price did not verify")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-slot", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--expected-price", type=int)
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    args = parser.parse_args()
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        tasker = load_ocr(connect(args.adb, args.address), args.ocr_resource)
        result = sell_hand(tasker, read_json(ROOT / "calib/slots.json"), args.hand_slot, args.name,
                           inspect=args.inspect, expected_price=args.expected_price)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
