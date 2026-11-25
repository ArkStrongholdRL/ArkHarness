"""Read or execute a named hand-slot sale through MaaFramework."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import add_session_args, session
from .native import Frame, completed_data
from .session import run as run_task, task_recognition
from .read import ROOT, clean, integer, read_json


class HandSaleRecognition(CustomRecognition):
    def __init__(self, source: tuple[int, int], name: str, expected_price: int | None, *, by_balance: bool = False):
        super().__init__()
        self.source = (*source, 1, 1)
        self.name = name
        self.expected_price = expected_price
        self.by_balance = by_balance

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
        if kind == "verified" and (self.by_balance or sale["price"]):
            state = self.balance(frame)
            if self.by_balance:
                if state is not None and state["round"] == before["round"] and state["funds"] > before["funds"]:
                    return self.result(before, state, state["funds"] - before["funds"])
                return None
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
            button = self.sell_button(frame, buttons)
            if kind == "sell" and self.by_balance:
                return self.AnalyzeResult(button, {"price": None}) if button is not None else None
            quote = self.quote(frame, button) if button is not None else None
            if kind == "sell":
                if quote is None or self.expected_price not in (None, quote):
                    return None
                return self.AnalyzeResult(button, {"price": quote})
            return self.AnalyzeResult((0, 0, 1, 1), {"price": quote}) if details else None
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

    def sell_button(self, frame: Frame, buttons: list) -> tuple | None:
        if len(buttons) != 1 or not any(clean(row.text) == self.name for row in frame.rows("Board.CarriedName")):
            return None
        return tuple(buttons[0].box)

    @staticmethod
    def quote(frame: Frame, button: tuple) -> int | None:
        x, y = button[:2]
        return integer(frame.text("Board.SellButtonPrice", roi=[x + 55, y - 58, 58, 34]), r"[+＋]([0-9]+)")


def sell_hand(tasker: Tasker, slots: dict, hand_slot: int, name: str, *,
              inspect: bool = False, expected_price: int | None = None,
              by_balance: bool = False) -> dict:
    """Sell a verified operator and check the transaction in the same round."""
    centers = slots["hand"]["centersByView"]["normal"]
    name = clean(name)
    if not name or type(hand_slot) is not int or not 0 <= hand_slot < len(centers):
        raise ValueError("An exact displayed name and valid zero-based hand slot are required")
    if (expected_price is not None or not (inspect or by_balance)) and (
            type(expected_price) is not int or expected_price < 0):
        raise ValueError("A sale requires a nonnegative expected price")
    required_price = None if by_balance else expected_price
    recognition = HandSaleRecognition(tuple(centers[hand_slot]), name, required_price, by_balance=by_balance and not inspect)
    with task_recognition(tasker, "ArkHandSale", recognition):
        override = {"HandSale.Open": {"next": ["HandSale.Dismiss"]}} if inspect else None
        job = run_task(tasker, "HandSale.Run", override)
    nodes = job.nodes
    result = completed_data(nodes, "HandSale.Unchanged" if inspect else "HandSale.Verified")
    if (job.succeeded and result is not None and result["price"] is not None
            and required_price in (None, result["price"])):
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
    add_session_args(parser)
    args = parser.parse_args()
    try:
        tasker = session(args.adb, args.address, args.ocr_resource).tasker
        result = sell_hand(tasker, read_json(ROOT / "calib/slots.json"), args.hand_slot, args.name,
                           inspect=args.inspect, expected_price=args.expected_price)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
