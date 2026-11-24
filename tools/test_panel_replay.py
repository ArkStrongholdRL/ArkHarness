"""Offline MFW panel transitions with scripted recognition frames."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maa.tasker import Tasker

from harness.screen.capture import OCR_RESOURCE, load_ocr
from harness.screen.equip_choice import choose_equipment_dialog
from tools.test_shop_replay import ReplayController


PICK = (399, 199, 2, 2)
CONFIRM = (399, 329, 2, 2)


def reset(controller):
    controller.clicks.clear()
    controller.image[0, 0, 0] = 0


def equipment_choice(tasker, controller):
    for closes_after in (1, 2):
        reset(controller)

        def frame(_, image):
            phase = int(image[0, 0, 0])

            def rows(node, **params):
                assert node == "Bounty.NameLeft"
                if phase >= closes_after:
                    return []
                if "expected" in params:
                    return [SimpleNamespace(text="装备替换", box=PICK)]
                text, box = ("精打细算玩偶", PICK) if phase == 0 else ("确认替换", CONFIRM)
                return [SimpleNamespace(text=text, box=box, score=0.99)]

            return SimpleNamespace(rows=rows, text=lambda node: {"ArkRoundNumber": "回合9", "Screen.Funds": "4"}[node],
                                   hit=lambda node: node == "ArkRestText" and phase >= closes_after)

        with patch("harness.screen.equip_choice.Frame", side_effect=frame):
            result = choose_equipment_dialog(tasker, ("装备替换",), {"精打细算玩偶": "item"}, 9)
        assert result == {"id": "item", "round": 9, "funds": 4}
        expected_clicks = [(400, 200), (400, 330)]
        assert controller.clicks == expected_clicks[:closes_after]


def main():
    Tasker.set_log_dir(Path("debug/mfw-panel-replay").resolve())
    Tasker.set_save_draw(False)
    Tasker.set_save_on_error(False)
    controller = ReplayController()
    assert controller.set_screenshot_use_raw_size(True)
    assert controller.post_connection().wait().succeeded
    tasker = load_ocr(controller, OCR_RESOURCE)
    equipment_choice(tasker, controller)
    print("MFW panel replay passed")


if __name__ == "__main__":
    main()
