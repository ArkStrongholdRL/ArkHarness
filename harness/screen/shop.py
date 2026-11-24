"""Operate the expanded shop through guarded MaaFramework pipelines."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .native import Frame
from .session import task_recognition
from .read import (
    CATALOG, ROOT, Reader, clean, expanded_shop_level, integer, read_json, shop_frozen,
    shop_confirm_box, shop_reward_choices, shop_upgrade_center, shop_upgrade_price,
)


CONTROLS = {"refresh": "Shop.RefreshText", "level": "Shop.LevelText", "freeze": "Shop.FreezeText"}


def shop_status(frame: Frame) -> tuple[int, int, int] | None:
    if not frame.hit("ArkRestText") or not frame.hit("Shop.Expanded"):
        return None
    current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
    level = expanded_shop_level(frame)
    funds = integer(frame.text("Screen.Funds"))
    if current_round is None or level is None or funds is None:
        return None
    return current_round, level, funds


class ShopViewRecognition(CustomRecognition):
    def __init__(self, expanded: bool):
        super().__init__()
        self.expanded = expanded
        self.before = None
        self.result = None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if frame.scene() != "preparation" or not frame.hit("ArkRestText"):
            return None
        kind = json.loads(argv.custom_recognition_param)
        details = frame.hit("Screen.OperatorDetails") or frame.hit("Withdraw.Icon") or frame.hit("HandSale.Button")
        if kind == "dismissDetails":
            return (0, 0, 1, 1) if details or frame.hit("Shop.ConfirmLevelText") else None
        if details or frame.hit("Shop.ConfirmLevelText"):
            return None
        expanded = frame.hit("Shop.Expanded")
        label = "Shop.CollapseText" if expanded else "Shop.ExpandText"
        buttons = frame.rows(label)
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        if current_round is None or funds is None or len(buttons) != 1:
            return None
        state = {"round": current_round, "funds": funds, "expanded": expanded}
        if kind == "switch":
            if expanded == self.expanded:
                return None
            self.before = state
            return tuple(buttons[0].box)
        if expanded != self.expanded or (self.before is not None and
                (current_round, funds) != (self.before["round"], self.before["funds"])):
            return None
        self.result = {"before": self.before or state, "after": state, "clicked": self.before is not None}
        return (0, 0, 1, 1)


def set_shop_open(tasker: Tasker, expanded: bool) -> dict:
    """Set the shop view while preserving round and funds."""
    if type(expanded) is not bool:
        raise ValueError("The desired expanded state must be a boolean")
    recognition = ShopViewRecognition(expanded)
    with task_recognition(tasker, "ArkShopView", recognition):
        job = tasker.post_task("Shop.SetView").wait()
    if not job.succeeded or recognition.result is None:
        raise RuntimeError("The requested shop view could not be verified")
    return recognition.result


class BuyRecognition(CustomRecognition):
    reward_kind = None
    def __init__(self, reader: Reader, slot: int, expected_id: str | None = None,
                 expected_price: int | None = None, expected_round: int | None = None, merge: bool = False):
        super().__init__()
        self.merge = merge  # the purchase completes a three-copy merge: the card slot may not show the sold marker
        self.reader = reader
        self.slot = slot
        self.expected_id = expected_id
        self.expected_price = expected_price
        self.expected_round = expected_round
        self.purchase = None
        self.position = None
        self.result = None
        self.reason = "Preparation and expanded shop were not recognized"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        status = shop_status(frame)
        if status is None:
            return None
        if frame.hit("Shop.ConfirmLevelText"):
            self.reason = "A shop upgrade is awaiting confirmation"
            return None
        kind = json.loads(argv.custom_recognition_param)
        if self.expected_round is not None and status[0] != self.expected_round:
            self.reason = f"Expected preparation round {self.expected_round}, read {status[0]}"
            return None
        if kind in {"target", "quote"}:
            return self.target(frame, *status, quote=kind == "quote")
        if kind == "confirm":
            return self.confirm(frame, *status)
        return self.verify(frame, *status) if kind == "verify" else None

    def target(self, frame: Frame, current_round: int, level: int, funds: int, *,
               pending: bool = False, quote: bool = False) -> tuple | None:
        mode = self.reader.read_mode(frame)
        positions = self.reader.shop_positions(mode, level)
        position = next((row for row in positions if row["engineIndex"] == self.slot), None)
        if position is None:
            self.reason = f"Engine slot {self.slot} has no calibrated position for this shop"
            return None
        if pending:
            box = shop_confirm_box(frame, position)
            if box is None:
                return None
        else:
            if shop_confirm_box(frame, position) is not None:
                self.reason = f"Engine slot {self.slot} already has an unrelated purchase confirmation"
                return None
            x, y = position["center"]
            box = (x, y, 1, 1)
        card = self.reader.read_shop_slot(frame, position, reward_kind=self.reward_kind)
        if pending and card is None:
            price = integer(frame.text("Screen.ShopCardPrice", roi=position["priceROI"]))
            card = {"kind": position["kind"], "id": None, "price": price, "sold": False}
        if not card or card["sold"] or card["price"] is None:
            self.reason = f"An available card and price were not verified in engine slot {self.slot}"
            return None
        if self.expected_id is not None and card["id"] != self.expected_id:
            self.reason = f"Engine slot {self.slot} has {card['id']}, expected {self.expected_id}"
            return None
        if self.expected_price is not None and card["price"] != self.expected_price:
            self.reason = f"Engine slot {self.slot} costs {card['price']}, expected {self.expected_price}"
            return None
        if not quote and funds < card["price"]:
            self.reason = f"Engine slot {self.slot} costs {card['price']}; only {funds} funds were read"
            return None
        self.position = position
        self.purchase = {
            "slot": self.slot, "screenSlot": position["screenIndex"], "round": current_round,
            "shopLevel": level, "kind": card["kind"], "id": card["id"],
            "price": card["price"], "fundsBefore": funds,
        }
        if quote:
            self.result = {**self.purchase, "affordable": funds >= card["price"]}
        return box

    def confirm(self, frame: Frame, current_round: int, level: int, funds: int) -> tuple | None:
        if self.purchase is None:
            if any(value is not None for value in (self.expected_id, self.expected_price, self.expected_round)):
                return None
            return self.target(frame, current_round, level, funds, pending=True)
        purchase = self.purchase
        if (current_round, level, funds) != (purchase["round"], purchase["shopLevel"], purchase["fundsBefore"]):
            self.reason = (f"Confirmation read round={current_round}, shop level={level}, funds={funds}; "
                           f"expected {purchase['round']}, {purchase['shopLevel']}, {purchase['fundsBefore']}")
            return None
        confirmation = shop_confirm_box(frame, self.position)
        if confirmation is not None and self.expected_price is not None:
            price = integer(frame.text("Screen.ShopCardPrice", roi=self.position["priceROI"]))
            if price != self.expected_price:
                self.reason = f"The confirmation price was {price}, expected {self.expected_price}"
                return None
        return confirmation

    def verify(self, frame: Frame, current_round: int, level: int, funds: int) -> tuple | None:
        purchase = self.purchase
        if purchase is None or current_round != purchase["round"] or level != purchase["shopLevel"]:
            self.reason = "The original preparation round and shop level were not recovered"
            return None
        sold = frame.hit("Screen.ShopSold", roi=self.position["soldROI"])
        if funds != purchase["fundsBefore"] - purchase["price"]:
            self.reason = f"Post-click check read funds={funds}, sold={sold}"
            return None
        if self.merge or not sold:
            positions = self.reader.shop_positions(self.reader.read_mode(frame), level)
            if shop_reward_choices(self.reader.read_shop_slots(frame, positions)):
                self.result = {**purchase, "fundsAfter": funds, "sold": sold, "rewardOpened": True}
                return (0, 0, 1, 1)
        if not sold:
            if purchase["price"] > 0:
                # The funds dropped by exactly the price in the same round and shop level: the click bought it, whether or
                # not the slot shows the sold marker (merge animation, overlays). A missing marker alone is not a failure.
                self.result = {**purchase, "fundsAfter": funds, "sold": False, "soldMarkerMissing": True, "merge": self.merge}
                return (0, 0, 1, 1)
            replacement = self.reader.read_shop_slot(frame, self.position)
            if (purchase["id"] and replacement and replacement["id"] and replacement["id"] != purchase["id"]
                    and replacement["price"] is not None and replacement["price"] > 0):
                self.result = {**purchase, "fundsAfter": funds, "sold": False, "rewardClaimed": True}
                return (0, 0, 1, 1)
            self.reason = f"Post-click check read funds={funds}, sold=False and no verified reward offer"
            return None
        self.result = {**purchase, "fundsAfter": funds, "sold": True}
        return (0, 0, 1, 1)


class RewardRecognition(BuyRecognition):
    def target(self, frame: Frame, current_round: int, level: int, funds: int, *,
               pending: bool = False, quote: bool = False) -> tuple | None:
        positions = self.reader.shop_positions(self.reader.read_mode(frame), level)
        slots = self.reader.read_shop_slots(frame, positions, reward_kind=self.reward_kind)
        choices = shop_reward_choices(slots, min_cards=1, kind=self.reward_kind or "chess")
        self.offer = {index: card["id"] for index, card in choices}
        if self.offer.get(self.slot) != self.expected_id:
            self.reason = "The requested free reward was not visible"
            return None
        return super().target(frame, current_round, level, funds, pending=pending, quote=quote)

    def confirm(self, frame: Frame, current_round: int, level: int, funds: int) -> tuple | None:
        if self.purchase is None:
            return self.target(frame, current_round, level, funds, pending=True)
        return super().confirm(frame, current_round, level, funds)

    def verify(self, frame: Frame, current_round: int, level: int, funds: int) -> tuple | None:
        purchase = self.purchase
        if purchase is None or (current_round, level, funds) != (
            purchase["round"], purchase["shopLevel"], purchase["fundsBefore"],
        ):
            return None
        if shop_confirm_box(frame, self.position):
            self.reason = "The free reward still awaits confirmation"
            return None
        positions = self.reader.shop_positions(self.reader.read_mode(frame), level)
        slots = self.reader.read_shop_slots(frame, positions, reward_kind=self.reward_kind)
        paid_shop = sum(bool(card and card["id"] and (card["price"] or 0) > 0) for card in slots) >= 2
        choices = dict(shop_reward_choices(slots, min_cards=1, kind=self.reward_kind or "chess"))
        replaced = sum(index in self.offer and card["id"] != self.offer[index] for index, card in choices.items())
        if not paid_shop and replaced < 2:
            self.reason = "The shop did not return to paid offers or show a new set of free rewards"
            return None
        self.result = {**purchase, "fundsAfter": funds, "rewardClaimed": True}
        return (0, 0, 1, 1)


def pick_reward(tasker: Tasker, reader: Reader, slot: int, *, expected_id: str, expected_round: int,
                kind: str = "chess") -> dict:
    return run_purchase(tasker, reader, slot, entry="Shop.Buy", expected_id=expected_id, expected_price=0,
                        expected_round=expected_round, reward=True, reward_kind=kind)


def quote_slot(tasker: Tasker, reader: Reader, slot: int) -> dict:
    """Read the current offer without clicking the card."""
    return run_purchase(tasker, reader, slot, entry="Shop.Quote")


def buy_slot(tasker: Tasker, reader: Reader, slot: int, *, expected_id: str | None = None,
             expected_price: int | None = None, expected_round: int | None = None, merge: bool = False) -> dict:
    if expected_id is not None and (not isinstance(expected_id, str) or not expected_id.strip()):
        raise ValueError("An expected card ID must be a nonempty string")
    if expected_price is not None and (type(expected_price) is not int or expected_price < 0):
        raise ValueError("An expected price must be a nonnegative integer")
    if expected_round is not None and (type(expected_round) is not int or expected_round < 1):
        raise ValueError("An expected round must be a positive integer")
    return run_purchase(tasker, reader, slot, entry="Shop.Buy", expected_id=expected_id,
                        expected_price=expected_price, expected_round=expected_round, merge=merge)


def run_purchase(tasker: Tasker, reader: Reader, slot: int, *, entry: str,
                 expected_id: str | None = None, expected_price: int | None = None,
                 expected_round: int | None = None, reward: bool = False, merge: bool = False,
                 reward_kind: str | None = None) -> dict:
    if type(slot) is not int or not 0 <= slot < reader.slots["shop"]["capacity"]:
        raise ValueError("Slot must be a zero-based engine shop index")
    recognizer = RewardRecognition if reward else BuyRecognition
    recognition = recognizer(reader, slot, expected_id, expected_price, expected_round, **({"merge": True} if merge and not reward else {}))
    recognition.reward_kind = reward_kind
    with task_recognition(tasker, "ArkShopBuy", recognition):
        job = tasker.post_task(entry).wait()
        detail = job.get()
        nodes = [node.name for node in detail.nodes] if detail else []
    terminal = "Shop.Quoted" if entry == "Shop.Quote" else "Shop.BuyVerified"
    if job.succeeded and nodes and nodes[-1] == terminal and recognition.result is not None:
        return recognition.result
    if entry == "Shop.Quote":
        raise RuntimeError(f"No shop quote was verified: {recognition.reason}")
    if "Shop.BuySlot" in nodes or "Shop.ConfirmBuy" in nodes:
        raise RuntimeError(f"Purchase attempted once; verification failed: {recognition.reason}")
    raise RuntimeError(f"No purchase was sent: {recognition.reason}")


class ShopControlRecognition(CustomRecognition):
    def __init__(self, reader: Reader, action: str, frozen: bool | None):
        super().__init__()
        self.reader = reader
        self.action = action
        self.frozen = frozen
        self.before = None
        self.price = 0
        self.result = None
        self.reason = "Preparation, round, balance and expanded shop were not verified"

    def offers(self, frame: Frame) -> list[str]:
        return sorted({name for row in frame.rows("Shop.OfferLabels") if row.score >= 0.88
                       and (name := clean(row.text)) in self.reader.shop_names})

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        kind = json.loads(argv.custom_recognition_param)
        if ((kind == "already" and self.action != "freeze")
                or (kind == "confirm_level" and self.action != "level")):
            return None
        frame = Frame(context, argv.image)
        status = shop_status(frame)
        if status is None:
            return None
        if self.action != "level" and frame.hit("Shop.ConfirmLevelText"):
            self.reason = "A shop upgrade is awaiting confirmation"
            return None
        current_round, level, funds = status
        state = {"round": current_round, "level": level, "funds": funds}
        if self.action == "freeze":
            state["frozen"] = shop_frozen(frame)
        if kind == "already":
            if state["frozen"] == self.frozen:
                self.result = {"action": self.action, "before": state, "after": state, "clicked": False}
                return (0, 0, 1, 1)
            return None
        if kind == "verify":
            return self.verify(frame, state)
        if kind == "confirm_level":
            return self.confirm_level(frame, state)
        return self.target(frame, state)

    def target(self, frame: Frame, state: dict) -> tuple | None:
        labels = frame.rows(CONTROLS[self.action])
        if len(labels) != 1:
            self.reason = f"The {self.action} button was not verified"
            return None
        if self.action == "freeze":
            if state["frozen"] is None or state["frozen"] == self.frozen:
                return None
        else:
            price = shop_upgrade_price(frame) if self.action == "level" else integer(frame.text("Screen.RefreshPrice"))
            if price is None or price > state["funds"]:
                self.reason = f"The {self.action} price was unreadable or exceeded the available funds"
                return None
            self.price = price
            if self.action == "refresh" and price == 0:
                state["offers"] = self.offers(frame)
        self.before = state
        return tuple(labels[0].box)

    def confirm_level(self, frame: Frame, state: dict) -> tuple | None:
        labels = frame.rows("Shop.ConfirmLevelText")
        if len(labels) != 1:
            return None
        price = shop_upgrade_price(frame)
        if price is None or price > state["funds"]:
            self.reason = "The upgrade preview level, price or balance was not verified"
            return None
        if self.before is None:
            center = shop_upgrade_center(frame)
            displayed = state["level"]
            if center is None or displayed < 2 and center < 340:
                self.reason = "The previous shop level could not be recovered from the upgrade preview"
                return None
            # Level I still occupies the rightmost upgrade position until confirmation.
            self.before, self.price = {**state, "level": 1 if center >= 340 else displayed - 1}, price
        before = self.before
        if ((state["round"], state["funds"], price) != (before["round"], before["funds"], self.price)
                or state["level"] not in (before["level"], before["level"] + 1)):
            self.reason = "The upgrade preview did not match the original level and price"
            return None
        return tuple(labels[0].box)

    def verify(self, frame: Frame, state: dict) -> tuple | None:
        before = self.before
        if before is None:
            return None
        if self.action == "level" and frame.hit("Shop.ConfirmLevelText"):
            self.reason = "The shop is still showing an upgrade preview"
            return None
        expected_level = before["level"] + (self.action == "level")
        if (state["round"], state["level"], state["funds"]) != (
            before["round"], expected_level, before["funds"] - self.price
        ):
            self.reason = f"The expected round, level and balance were not verified: {state}"
            return None
        if self.action == "freeze" and state["frozen"] != self.frozen:
            self.reason = "The freeze button did not show the requested state"
            return None
        if self.action == "refresh" and self.price == 0:
            state["offers"] = self.offers(frame)
            price_after = integer(frame.text("Screen.RefreshPrice"))
            offer_changed = bool(before["offers"] and set(state["offers"]).difference(before["offers"]))
            if not price_after and not offer_changed:
                self.reason = "A free refresh did not show a price or offer change"
                return None
        self.result = {"action": self.action, "before": before, "after": state, "clicked": True}
        return (0, 0, 1, 1)


def change_shop(tasker: Tasker, reader: Reader, action: str, *, frozen: bool | None = None) -> dict:
    if action not in CONTROLS or (action == "freeze" and type(frozen) is not bool):
        raise ValueError("Specify a shop action and an explicit desired freeze state for freeze")
    recognition = ShopControlRecognition(reader, action, frozen)
    with task_recognition(tasker, "ArkShopControl", recognition):
        job = tasker.post_task("Shop.Change").wait()
    if job.succeeded and recognition.result is not None:
        return recognition.result
    if recognition.before is not None:
        raise RuntimeError(f"Shop {action} outcome is unverified: {recognition.reason}")
    raise RuntimeError(f"No {action} click was sent: {recognition.reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("open", "close", "quote", "buy", *CONTROLS))
    parser.add_argument("--slot", type=int, help="Zero-based engine shop index for quote or buy")
    parser.add_argument("--expected-id")
    parser.add_argument("--expected-price", type=int)
    parser.add_argument("--expected-round", type=int)
    parser.add_argument("--state", choices=("on", "off"), help="Desired state for freeze")
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()
    if (args.action in {"buy", "quote"}) != (args.slot is not None):
        parser.error("--slot is required for buy or quote")
    if args.action != "buy" and any(value is not None for value in
                                     (args.expected_id, args.expected_price, args.expected_round)):
        parser.error("Expected ID, price and round are only available with buy")
    if (args.action == "freeze") != (args.state is not None):
        parser.error("--state on|off is required for freeze")
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        slots = read_json(ROOT / "calib/slots.json")
        reader = Reader(args.catalog, slots)
        tasker = load_ocr(connect(args.adb, args.address), args.ocr_resource)
        if args.action == "buy":
            result = buy_slot(tasker, reader, args.slot, expected_id=args.expected_id,
                              expected_price=args.expected_price, expected_round=args.expected_round)
        elif args.action == "quote":
            result = quote_slot(tasker, reader, args.slot)
        elif args.action in {"open", "close"}:
            result = set_shop_open(tasker, args.action == "open")
        else:
            frozen = None if args.state is None else args.state == "on"
            result = change_shop(tasker, reader, args.action, frozen=frozen)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
