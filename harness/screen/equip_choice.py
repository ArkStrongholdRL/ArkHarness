"""Resolve an equipment dialog by its exact heading and a named option."""

import json
import re

from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .native import Frame
from .read import clean, integer


class EquipmentChoice(CustomRecognition):
    def __init__(self, headings: tuple[str, ...], options: dict[str, str], expected_round: int):
        super().__init__()
        self.heading_pattern = "^(?:" + "|".join(map(re.escape, headings)) + ")$"
        self.options = options
        self.round = expected_round
        self.selection = None
        self.result = None

    def analyze(self, context, argv):
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        heading = frame.rows("Bounty.NameLeft", roi=[100, 18, 1080, 220], expected=self.heading_pattern)
        rnd = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        if kind == "closed":
            funds = integer(frame.text("Screen.Funds"))
            if heading or not self.selection or rnd != self.round or funds is None or not frame.hit("ArkRestText"):
                return None
            self.result = {"id": self.selection[0], "round": rnd, "funds": funds}
            return (0, 0, 1, 1)
        if len(heading) != 1 or rnd not in (None, self.round):
            return None
        body = frame.rows("Bounty.NameLeft", roi=[104, 170, 1072, 405])
        if kind == "pick":
            for title, item_id in self.options.items():
                matches = [row for row in body if clean(row.text).strip("【】[]") == title and row.score >= 0.9]
                if len(matches) == 1:
                    box = tuple(matches[0].box)
                    self.selection = (item_id, box)
                    return box
            return None
        if kind == "confirm" and self.selection:
            x, _, w, _ = self.selection[1]
            labels = [row for row in body if clean(row.text) in {"确认替换", "确认更换", "确认选择"}
                      or clean(row.text) == "确认" and abs(row.box[0] + row.box[2] / 2 - (x + w / 2)) < 125]
            if len(labels) == 1:
                return tuple(labels[0].box)
        return None


def choose_equipment_dialog(tasker: Tasker, headings: tuple[str, ...], options: dict[str, str],
                            expected_round: int) -> dict:
    recognition = EquipmentChoice(headings, options, expected_round)
    entry = "EquipChoice.Start"
    overrides = {entry: {"next": ["EquipChoice.Pick"], "timeout": 4500}}
    for node, kind, following in (("Pick", "pick", ["Closed", "Confirm"]),
                                  ("Confirm", "confirm", ["Closed"]), ("Closed", "closed", [])):
        overrides[f"EquipChoice.{node}"] = {
            "recognition": "Custom", "custom_recognition": "ArkEquipChoice", "custom_recognition_param": kind,
            "action": "DoNothing" if kind == "closed" else "Click", "max_hit": 1, "timeout": 4500,
            "next": [f"EquipChoice.{name}" for name in following],
        }
    if not tasker.resource.register_custom_recognition("ArkEquipChoice", recognition):
        raise RuntimeError("Could not register equipment choice recognition")
    try:
        job = tasker.post_task(entry, overrides).wait()
        if not job.succeeded or not recognition.result:
            raise RuntimeError(f"Equipment dialog was not verified for {' / '.join(headings)}")
        return recognition.result
    finally:
        tasker.resource.unregister_custom_recognition("ArkEquipChoice")
