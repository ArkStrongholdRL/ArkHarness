"""Resolve an equipment dialog by its exact heading and a named option."""

import json
import re

from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .native import Frame, completed_data
from .read import clean, integer
from .session import run as run_task, task_recognition


class EquipmentChoice(CustomRecognition):
    def __init__(self, headings: tuple[str, ...], options: dict[str, str], expected_round: int):
        super().__init__()
        self.heading_pattern = "^(?:" + "|".join(map(re.escape, headings)) + ")$"
        self.options = options
        self.round = expected_round

    def analyze(self, context, argv):
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        heading = frame.rows("Bounty.NameLeft", roi=[100, 18, 1080, 220], expected=self.heading_pattern)
        rnd = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        selection = completed_data(argv.task_detail.nodes, "EquipChoice.Pick") if kind != "pick" else None
        if kind == "closed":
            funds = integer(frame.text("Screen.Funds"))
            if heading or selection is None or rnd != self.round or funds is None or not frame.hit("ArkRestText"):
                return None
            return self.AnalyzeResult((0, 0, 1, 1), {"id": selection["id"], "round": rnd, "funds": funds})
        if len(heading) != 1 or rnd not in (None, self.round):
            return None
        body = frame.rows("Bounty.NameLeft", roi=[104, 170, 1072, 405])
        if kind == "pick":
            for title, item_id in self.options.items():
                matches = [row for row in body if clean(row.text).strip("【】[]") == title and row.score >= 0.9]
                if len(matches) == 1:
                    box = tuple(matches[0].box)
                    return self.AnalyzeResult(box, {"id": item_id, "box": box})
            return None
        if kind == "confirm" and selection is not None:
            x, _, w, _ = selection["box"]
            labels = [row for row in body if clean(row.text) in {"确认替换", "确认更换", "确认选择"}
                      or clean(row.text) == "确认" and abs(row.box[0] + row.box[2] / 2 - (x + w / 2)) < 125]
            if len(labels) == 1:
                return tuple(labels[0].box)
        return None


def choose_equipment_dialog(tasker: Tasker, headings: tuple[str, ...], options: dict[str, str],
                            expected_round: int) -> dict:
    recognition = EquipmentChoice(headings, options, expected_round)
    with task_recognition(tasker, "ArkEquipChoice", recognition):
        job = run_task(tasker, "EquipChoice.Start")
    result = completed_data(job.nodes, "EquipChoice.Closed")
    if not job.succeeded or result is None:
        raise RuntimeError(f"Equipment dialog was not verified for {' / '.join(headings)}")
    return result
