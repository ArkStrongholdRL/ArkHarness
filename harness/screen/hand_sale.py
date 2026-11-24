"""Read or execute a named hand-slot sale through MaaFramework."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .native import Frame
from .read import ROOT, clean, integer, read_json


class HandSaleRecognition(CustomRecognition):
    def __init__(self, source: tuple[int, int], name: str, expected_price: int | None):
        super().__init__()
        self.source = (*source, 1, 1)
        self.name = name
        self.expected_price = expected_price
        self.before = None
        self.after = None
        self.price = None
        self.checked_slot = False
        self.reason = "The normal preparation hand view was not verified"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        if kind == "check" and self.price != 0:
            return None
        if not frame.hit("ArkRestText"):
            return None
        if kind == "verified" and self.price:
            current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
            funds = integer(frame.text("Screen.Funds"))
            if self.before is not None and (current_round, funds) == (
                    self.before["round"], self.before["funds"] + self.price):
                self.after = {"round": current_round, "funds": funds}
                return (0, 0, 1, 1)
            self.reason = "The expected hand sale proceeds were not verified"
            return None
        if frame.scene() != "preparation" or frame.hit("Shop.Expanded"):
            return None
        buttons = frame.rows("HandSale.Button")
        details = bool(buttons) or frame.hit("Screen.OperatorDetails")
        if kind == "reset":
            return (0, 0, 1, 1) if details or frame.hit("Withdraw.Icon") else None
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        if current_round is None or funds is None:
            self.reason = "The round or balance was unreadable"
            return None
        state = {"round": current_round, "funds": funds}
        if kind == "open":
            if details:
                return None
            self.before = state
            return self.source
        if self.before is None:
            return None
        if kind == "verified" and not self.checked_slot:
            return None
        if kind in {"sell", "dismiss"}:
            if state != self.before:
                self.reason = "The preparation state changed before sale"
                return None
            button = self.quote(frame, buttons)
            if kind == "sell":
                return button if button and self.price == self.expected_price else None
            return (0, 0, 1, 1) if details else None
        gain = 0 if kind == "unchanged" else self.price
        if details or gain is None or state != {**self.before, "funds": self.before["funds"] + gain}:
            if kind != "unchanged":
                self.reason = "The expected preparation state and empty hand slot were not verified"
            return None
        if kind == "check":
            self.checked_slot = True
            return self.source
        self.after = state
        return (0, 0, 1, 1)

    def quote(self, frame: Frame, buttons: list) -> tuple | None:
        if len(buttons) != 1 or not any(clean(row.text) == self.name for row in frame.rows("Board.CarriedName")):
            self.reason = "The requested hand operator and its sell button were not verified"
            return None
        button = buttons[0].box
        x, y = button[:2]
        self.price = integer(frame.text("Board.SellButtonPrice", roi=[x + 55, y - 58, 58, 34]),
                             r"[+＋]([0-9]+)")
        if self.price is None:
            self.reason = "The hand sale price was unreadable"
            return None
        if self.expected_price is not None and self.price != self.expected_price:
            self.reason = f"The hand sale price was {self.price}, expected {self.expected_price}"
        return tuple(button)


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
    resource = tasker.resource
    if not resource.register_custom_recognition("ArkHandSale", recognition):
        raise RuntimeError("Could not register the hand sale recognizer")
    try:
        if inspect:
            override = {"HandSale.Open": {"next": ["HandSale.Dismiss"]}}
        else:
            terminal = "HandSale.Verified" if expected_price else "HandSale.CheckSlot"
            override = {"HandSale.Click": {"next": [terminal]}}
        job = tasker.post_task("HandSale.Run", override).wait()
        detail = job.get()
        nodes = [node.name for node in detail.nodes] if detail else []
    finally:
        resource.unregister_custom_recognition("ArkHandSale")
    target = {"handSlot": hand_slot, "name": name, "price": recognition.price,
              "before": recognition.before, "after": recognition.after}
    if job.succeeded and nodes and recognition.after is not None and recognition.price is not None:
        if inspect and nodes[-1] == "HandSale.Unchanged" and expected_price in (None, recognition.price):
            return target
        if not inspect and nodes[-1] == "HandSale.Verified":
            return target
    if "HandSale.Click" in nodes:
        raise RuntimeError(f"Hand sale was attempted once and is unverified: {recognition.reason}")
    raise RuntimeError(f"Hand sale was not sent: {recognition.reason}")


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
