"""Offline MFW panel transitions with scripted recognition frames."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from harness.screen.briefing import choose_bond_group, inspect_briefing
from harness.screen.capture import OCR_RESOURCE, load_ocr
from harness.screen.command import inspect_command
from harness.screen.equip_choice import choose_equipment_dialog
from harness.screen.equipment_read import read_equipment_slots
from harness.screen.hand import read_hand
from harness.screen.read import CATALOG
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


def command_details(tasker, controller):
    for phases in (["preparation", "commandDetails", "preparation"], ["commandDetails", "preparation"]):
        reset(controller)

        def frame(_, image):
            scene = phases[int(image[0, 0, 0])]
            return SimpleNamespace(
                scene=lambda: scene, hit=lambda node: node == "ArkRestText",
                rows=lambda node: [SimpleNamespace(box=PICK)] if node == "Command.View" else [],
                text=lambda node: {"ArkRoundNumber": "回合9", "Screen.Funds": "4"}.get(node, ""),
            )

        with patch("harness.screen.command.Frame", side_effect=frame):
            result = inspect_command(tasker, CATALOG)
        assert result["round"] == 9 and result["funds"] == 4
        assert controller.clicks == [(400, 200)] * (len(phases) - 1)


class StrategyMarker(CustomRecognition):
    def analyze(self, context, argv):
        return PICK if int(argv.image[0, 0, 0]) >= self.confirmed_phase else None


def briefing_groups(tasker, controller):
    groups = [[f"left{i}" for i in range(5)], [f"right{i}" for i in range(5)]]
    marker = StrategyMarker()
    assert tasker.resource.register_custom_recognition("ReplayStrategy", marker)
    post_task = tasker.post_task
    override = {
        "Briefing.Confirmed": {"all_of": ["Replay.Strategy"]},
        "Replay.Strategy": {"recognition": "Custom", "custom_recognition": "ReplayStrategy"},
    }
    for initial, changed_groups in ((0, False), (1, False), (0, True)):
        reset(controller)
        marker.confirmed_phase = 2 - initial

        def frame(_, image):
            phase = int(image[0, 0, 0])
            return SimpleNamespace(
                hit=lambda node: node == "Lifecycle.Briefing", phase=phase,
                rows=lambda node, **opts: [SimpleNamespace(box=PICK if node == "Briefing.GroupButton" else CONFIRM)],
            )

        def read(frame):
            selected = min(initial + frame.phase, 1)
            displayed = groups[::-1] if changed_groups and frame.phase else groups
            return {"match": {"bondChoice": {"groups": displayed}, "disabledBonds": displayed[1 - selected],
                              "bossId": "boss"}, "config": {"modeId": "mode"}}

        reader = SimpleNamespace(read=read)
        with (patch("harness.screen.briefing.Frame", side_effect=frame),
              patch.object(tasker, "post_task", side_effect=lambda entry: post_task(entry, override))):
            preview = inspect_briefing(tasker, reader)
            assert preview["previewSelected"] == initial and not controller.clicks
            if changed_groups:
                try:
                    choose_bond_group(tasker, reader, 1)
                except RuntimeError as error:
                    assert "briefing was not confirmed" in str(error)
                else:
                    raise AssertionError("Changed briefing groups were confirmed")
                assert controller.clicks == [(400, 200)]
            else:
                result = choose_bond_group(tasker, reader, 1, expected_groups=groups)
                assert result["bondChoice"] == {"groups": groups, "picked": 1}
                assert controller.clicks == ([(400, 330)] if initial else [(400, 200), (400, 330)])
    tasker.resource.unregister_custom_recognition("ReplayStrategy")


class RackReplay(ReplayController):
    def __init__(self):
        super().__init__()
        self.moves = []

    def touch_move(self, contact, x, y, pressure):
        self.moves.append((x, y))
        return True


def equipment_rack():
    controller = RackReplay()
    assert controller.set_screenshot_use_raw_size(True)
    assert controller.post_connection().wait().succeeded
    tasker = load_ocr(controller, OCR_RESOURCE)
    post_task = tasker.post_task
    overrides = {
        "EquipmentRead.Pick": {"all_of": ["EquipmentRead.Before"]},
        "EquipmentRead.Held": {"all_of": ["EquipmentRead.ReadName"]},
    }
    reader = SimpleNamespace(slots={"equipment": {"centersByView": {"shopExpanded": [[400, 400], [500, 400]]},
                                                  "liftOffset": [0, -100]}})
    view = {"after": {"round": 9, "funds": 4}, "clicked": False}
    identity = {"name": "精打细算玩偶", "id": "item", "baseId": "item", "candidateIds": ["item"]}
    with (patch.object(tasker, "post_task", side_effect=lambda entry, params: post_task(
              entry, {**params, **{name: {**params.get(name, {}), **patch}
                                  for name, patch in overrides.items()}})),
          patch("harness.screen.equipment_read.Frame", return_value=SimpleNamespace(hit=lambda _: False)),
          patch("harness.screen.equipment_read.hud_status", return_value=(9, 4)),
          patch("harness.screen.equipment_read.set_shop_open", return_value=view),
          patch("harness.screen.equipment_read.catalog_identity", return_value=identity)):
        result = read_equipment_slots(tasker, reader, [1])
    assert result == {"round": 9, "funds": 4, "slots": [{"screenSlot": 1, "status": "occupied", **identity}]}
    assert controller.clicks == [(500, 400)] and controller.moves == [(500, 300), (35, 690)]


def hand_slots(tasker, controller):
    reset(controller)
    stage = "act3autochess_m02"
    reader = SimpleNamespace(stage_hint=stage, slots={"hand": {"centersByView": {"normal": [[300, 600], [400, 600]]}}})
    identity = {"id": "chess", "name": "干员", "kind": "chess", "candidateIds": ["chess"]}

    def frame(_, image):
        phase = int(image[0, 0, 0])
        return SimpleNamespace(
            scene=lambda: "preparation",
            hit=lambda node: node == "ArkRestText" or (node == "HandSale.Button" and phase == 1),
            text=lambda node: {"ArkRoundNumber": "回合9", "Screen.Funds": "4",
                               "Screen.DeployCollapsed": "剩余可放置角色:2"}[node],
        )

    with (patch("harness.screen.hand.Frame", side_effect=frame),
          patch("harness.screen.hand.chess_identity", return_value=identity)):
        result = read_hand(tasker, reader, [0, 1])
    assert (result["stageId"], result["round"], result["funds"], result["remaining"]) == (stage, 9, 4, 2)
    assert result["slots"][0] == {"screenSlot": 0, "status": "occupied", **identity}
    assert result["slots"][1]["screenSlot"] == 1 and result["slots"][1]["status"] == "empty"
    assert controller.clicks == [(300, 600), (640, 495), (400, 600)]


def main():
    Tasker.set_log_dir(Path("debug/mfw-panel-replay").resolve())
    Tasker.set_save_draw(False)
    Tasker.set_save_on_error(False)
    controller = ReplayController()
    assert controller.set_screenshot_use_raw_size(True)
    assert controller.post_connection().wait().succeeded
    tasker = load_ocr(controller, OCR_RESOURCE)
    equipment_choice(tasker, controller)
    command_details(tasker, controller)
    briefing_groups(tasker, controller)
    equipment_rack()
    hand_slots(tasker, controller)
    print("MFW panel replay passed")


if __name__ == "__main__":
    main()
