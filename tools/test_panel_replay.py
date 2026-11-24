"""Offline MFW panel transitions with scripted recognition frames."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker
from PIL import Image

from harness.screen.briefing import choose_bond_group, inspect_briefing
from harness.screen.capture import OCR_RESOURCE, load_ocr
from harness.screen.command import inspect_command
from harness.screen.commander import CommanderRecognition, run as run_commander, run_pipeline, save_portrait
from harness.screen.equip_choice import choose_equipment_dialog
from harness.screen.equipment_read import read_equipment_slots
from harness.screen.hand import read_hand
from harness.screen.hand_sale import sell_hand
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


def commander_preview(tasker, controller):
    for starting_name in ("", "领袖"):
        reset(controller)
        marker = StrategyMarker()
        marker.confirmed_phase = 0
        assert tasker.resource.register_custom_recognition("ReplayCommander", marker)
        post_task = tasker.post_task
        overrides = {
            "Replay.Strategy": {"recognition": "Custom", "custom_recognition": "ReplayCommander"},
            "Commander.ReadPage": {"all_of": ["Replay.Strategy", "Commander.Snapshot"]},
            "Commander.ClickSlot": {"all_of": ["Replay.Strategy", "Commander.Slot"]},
            "Commander.PreviewOpened": {
                "all_of": ["Replay.Strategy", "Commander.SelectionChanged", "Commander.Snapshot"],
            },
        }
        layouts = {"gridPlain": {"rows": [300], "columns": [400]},
                   "gridWithDetail": {"rows": [300], "columns": [400]}}

        def frame(_, image):
            phase = int(image[0, 0, 0])
            name = "领袖" if phase else starting_name
            hits = {
                "Commander.ConfirmText": bool(name), "Commander.GridWithDetail": bool(name),
                "Commander.GridPlain": not name, "Commander.SelectedBadge": phase > 0,
            }
            return SimpleNamespace(text=lambda _: name, hit=lambda node, **_: hits[node])

        with (patch.object(tasker, "post_task", side_effect=lambda entry, params: post_task(
                   entry, {**params, **overrides})),
              patch("harness.screen.commander.Frame", side_effect=frame)):
            observed = run_commander(tasker, "observe", layouts)
            assert observed["commander"] == (starting_name or None) and not controller.clicks
            result = run_pipeline(tasker, "preview", CommanderRecognition(layouts, (1, 1), capture_portrait=True))
        assert result["snapshot"] == {"screen": "strategySelection", "commander": "领袖", "canConfirm": True}
        assert controller.clicks == [(400, 300)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "portrait.png"
            assert save_portrait(path, result["portrait"])
            with Image.open(path) as portrait:
                assert portrait.size == (80, 80)
        tasker.resource.unregister_custom_recognition("ReplayCommander")


def hand_sale(tasker, controller):
    slots = {"hand": {"centersByView": {"normal": [[300, 600]]}}}
    for inspect, price in ((True, 2), (False, 2), (False, 0)):
        reset(controller)

        def frame(_, image):
            phase = int(image[0, 0, 0])
            funds = 4 if inspect or phase < 2 else 4 + price
            texts = {"ArkRoundNumber": "回合9", "Screen.Funds": str(funds), "Board.SellButtonPrice": f"+{price}"}
            rows = {"HandSale.Button": [SimpleNamespace(box=PICK)], "Board.CarriedName": [SimpleNamespace(text="干员")]}
            return SimpleNamespace(
                scene=lambda: "overlay" if not inspect and price and phase >= 2 else "preparation",
                hit=lambda node: node == "ArkRestText",
                text=lambda node, **_: texts[node],
                rows=lambda node: rows[node] if phase == 1 else [],
            )

        with patch("harness.screen.hand_sale.Frame", side_effect=frame):
            result = sell_hand(tasker, slots, 0, "干员", inspect=inspect, expected_price=price)
        assert result["price"] == price and result["before"] == {"round": 9, "funds": 4}
        assert result["after"] == {"round": 9, "funds": 4 + (0 if inspect else price)}
        clicks = [(300, 600), (1030, 70) if inspect else (400, 200)]
        assert controller.clicks == (clicks if inspect or price else [*clicks, (300, 600)])


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
    commander_preview(tasker, controller)
    hand_sale(tasker, controller)
    print("MFW panel replay passed")


if __name__ == "__main__":
    main()
