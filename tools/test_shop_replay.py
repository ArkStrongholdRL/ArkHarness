"""Exercise the real MFW shop pipeline against offline frames with scripted OCR."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from maa.tasker import Tasker

from harness.screen.capture import OCR_RESOURCE, load_ocr
from harness.screen.native import OfflineController
from harness.screen.shop import buy_slot, change_shop, pick_reward, quote_slot, set_shop_open


POSITION = {"engineIndex": 0, "screenIndex": 1, "kind": "chess", "center": [300, 600],
            "nameROI": [250, 600, 100, 40], "priceROI": [300, 650, 20, 30],
            "soldROI": [250, 600, 100, 40]}
CARD = {"kind": "chess", "id": "operator", "price": 2, "sold": False}
FREE_CARD = {**CARD, "price": 0}
IDLE = {"level": 1, "funds": 4, "card": CARD, "price": 1, "label": "old"}


class ReplayController(OfflineController):
    def __init__(self):
        super().__init__(np.zeros((720, 1280, 3), dtype=np.uint8))
        self.clicks = []

    def touch_down(self, contact, x, y, pressure):
        self.clicks.append((x, y))
        return True

    def touch_up(self, contact):
        self.image[0, 0, 0] = len(self.clicks)
        return True


class ReplayFrame:
    def __init__(self, image, states):
        self.state = states[min(int(image[0, 0, 0]), len(states) - 1)]

    def text(self, name, **_):
        if name == "ArkRoundNumber":
            return "回合9"
        if name == "Screen.Funds":
            return str(self.state["funds"])
        if name == "Screen.ShopCardPrice":
            return str(self.state["card"]["price"])
        assert name == "Screen.RefreshPrice"
        return str(self.state["price"])

    def hit(self, name, **_):
        return (name == "ArkRestText" or (name == "Shop.Expanded" and self.state.get("expanded", False))
                or (name == "Shop.ConfirmLevelText" and self.state.get("confirmation") == "level")
                or (name == "Screen.ShopSold" and self.state.get("sold", False)))

    def scene(self):
        return "preparation"

    def rows(self, name, **_):
        if name == "Shop.OfferLabels":
            return [SimpleNamespace(score=0.99, text=self.state["label"])]
        if name == "Shop.ConfirmBuyText":
            if self.state.get("confirmation") != "buy":
                return []
        elif name == "Shop.ConfirmLevelText":
            if self.state.get("confirmation") != "level":
                return []
        return [SimpleNamespace(box=(499, 649, 2, 2))]


def status(frame):
    return 9, frame.state["level"], frame.state["funds"]


def read_slots(frame, *_args, **_kwargs):
    return frame.state.get("cards", [frame.state["card"]])


def replay(tasker, controller, states, action, *, error=None, clicks):
    controller.clicks.clear()
    controller.image[0, 0, 0] = 0
    with (patch("harness.screen.shop.Frame", side_effect=lambda _, image: ReplayFrame(image, states)),
          patch("harness.screen.shop.shop_status", side_effect=status),
          patch("harness.screen.shop.shop_upgrade_price", side_effect=lambda f: f.state["price"]),
          patch("harness.screen.shop.shop_upgrade_center", return_value=350)):
        try:
            result = action()
        except RuntimeError as caught:
            assert error and error in str(caught), caught
            result = None
        else:
            assert error is None, result
    assert controller.clicks == clicks, controller.clicks
    return result


def main():
    Tasker.set_log_dir(Path("debug/mfw-shop-replay").resolve())
    Tasker.set_save_draw(False)
    Tasker.set_save_on_error(False)
    controller = ReplayController()
    assert controller.set_screenshot_use_raw_size(True)
    assert controller.post_connection().wait().succeeded
    tasker = load_ocr(controller, OCR_RESOURCE)
    for expanded in (True, False):
        before = {**IDLE, "expanded": not expanded}
        after = {**IDLE, "expanded": expanded}
        expected = {"round": 9, "funds": 4, "expanded": expanded}
        switched = replay(tasker, controller, [before, after], lambda: set_shop_open(tasker, expanded),
                          clicks=[(500, 650)])
        assert switched == {"before": {**expected, "expanded": not expanded}, "after": expected, "clicked": True}
        already_set = replay(tasker, controller, [after], lambda: set_shop_open(tasker, expanded), clicks=[])
        assert already_set == {"before": expected, "after": expected, "clicked": False}
    replay(tasker, controller, [{**IDLE, "expanded": True}, {**IDLE, "funds": 3}],
           lambda: set_shop_open(tasker, False),
           clicks=[(500, 650)], error="The requested shop view could not be verified")

    reader = SimpleNamespace(
        slots={"shop": {"capacity": 1}}, shop_names={"old", "new"},
        read_mode=lambda _: None, shop_positions=lambda *_: [POSITION],
        read_shop_slot=lambda frame, *_args, **_kwargs: frame.state["card"],
        read_shop_slots=read_slots,
    )
    click = (500, 650)
    card_click = (300, 600)
    paid = {**IDLE, "funds": 2, "sold": True}
    purchase = lambda: buy_slot(tasker, reader, 0, expected_id="operator", expected_price=2, expected_round=9)
    quote = replay(tasker, controller, [IDLE], lambda: quote_slot(tasker, reader, 0), clicks=[])
    assert quote["id"] == "operator" and quote["affordable"]
    for states, clicks in (([IDLE, paid], [card_click]),
                           ([IDLE, {**IDLE, "confirmation": "buy"}, paid], [card_click, click])):
        result = replay(tasker, controller, states, purchase, clicks=clicks)
        assert result["fundsAfter"] == 2 and result["sold"]
    result = replay(tasker, controller, [IDLE, {**paid, "sold": False}], purchase, clicks=[card_click])
    assert result["soldMarkerMissing"]
    replay(tasker, controller, [IDLE, IDLE], purchase, clicks=[card_click], error="Purchase attempted once")
    replay(tasker, controller, [{**IDLE, "confirmation": "buy"}, paid], purchase,
           clicks=[], error="No purchase was sent")
    result = replay(tasker, controller, [{**IDLE, "confirmation": "buy"}, paid],
                    lambda: buy_slot(tasker, reader, 0), clicks=[click])
    assert result["fundsAfter"] == 2
    reward = {**IDLE, "card": FREE_CARD}
    paid_shop = {**IDLE, "cards": [CARD, CARD]}
    result = replay(tasker, controller, [reward, paid_shop],
                    lambda: pick_reward(tasker, reader, 0, expected_id="operator", expected_round=9),
                    clicks=[card_click])
    assert result["rewardClaimed"] and result["fundsBefore"] == result["fundsAfter"]
    refresh = lambda: change_shop(tasker, reader, "refresh")
    result = replay(tasker, controller, [IDLE, {**IDLE, "funds": 3}], refresh, clicks=[click])
    assert result["before"]["funds"] == 4 and result["after"]["funds"] == 3
    free_refresh = {**IDLE, "price": 0}
    result = replay(tasker, controller, [free_refresh, {**free_refresh, "label": "new"}], refresh, clicks=[click])
    assert result["before"]["offers"] == ["old"] and result["after"]["offers"] == ["new"]
    replay(tasker, controller, [free_refresh], refresh, clicks=[click], error="outcome is unverified")
    preview = {**IDLE, "confirmation": "level", "level": 2}
    upgraded = {**IDLE, "level": 2, "funds": 3}
    for states, clicks in (([IDLE, preview, upgraded], [click, click]), ([preview, upgraded], [click])):
        result = replay(tasker, controller, states, lambda: change_shop(tasker, reader, "level"), clicks=clicks)
        assert result["before"] == {"round": 9, "level": 1, "funds": 4}
        assert result["after"] == {"round": 9, "level": 2, "funds": 3}
    print("MFW shop replay passed")


if __name__ == "__main__":
    main()
