"""Resolve an equipment dialog by its exact heading and a named option."""

import json
import re

from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .native import Frame, completed_data
from .read import clean, integer, screen_integer
from .session import run as run_task, task_recognition

REPLACEMENT_COLUMNS = (
    ([824, 301, 142, 42], [748, 575, 218, 41]),
    ([1100, 301, 139, 42], [1015, 575, 218, 41]),
)


class EquipmentChoice(CustomRecognition):
    def __init__(self, headings: tuple[str, ...], options: dict[str, str], expected_round: int,
                 *, replace_on: str | None = None):
        super().__init__()
        self.heading_pattern = "^(?:" + "|".join(map(re.escape, headings)) + ")$"
        self.options = options
        self.round = expected_round
        self.replace_on = replace_on

    def analyze(self, context, argv):
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        rnd = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        if self.replace_on is not None or kind in {"cancel", "cleared"}:
            return self.replacement(frame, kind, rnd, argv)
        heading = frame.rows("Bounty.NameLeft", roi=[100, 18, 1080, 220], expected=self.heading_pattern)
        selection = completed_data(argv.task_detail.nodes, "EquipChoice.Pick") if kind != "pick" else None
        if kind == "closed":
            funds = screen_integer(frame, "Screen.Funds")
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

    def replacement(self, frame, kind, rnd, argv):
        if not frame.hit("ArkRestText"):
            return None
        opened = frame.hit("EquipChoice.ReplaceTitle")
        if kind in {"closed", "cleared"}:
            if opened or any(frame.hit("EquipChoice.ReplaceButton", roi=button)
                             for _, button in REPLACEMENT_COLUMNS):
                return None
            if kind == "cleared":
                return (0, 0, 1, 1)
            selection = completed_data(argv.task_detail.nodes, "EquipChoice.Pick", "EquipChoice.RetryPick")
            funds = screen_integer(frame, "Screen.Funds")
            if selection is not None and rnd == self.round and funds is not None:
                return self.AnalyzeResult((0, 0, 1, 1), {"id": selection["id"], "round": rnd, "funds": funds})
            return None
        if not opened or rnd != self.round:
            return None
        if kind == "cancel":
            return (0, 0, 1, 1) if any(frame.hit(node, roi=button)
                                       for _, button in REPLACEMENT_COLUMNS
                                       for node in ("EquipChoice.ReplaceButton", "EquipChoice.ReplaceConfirm")) else None
        if kind == "confirm":
            selection = completed_data(argv.task_detail.nodes, "EquipChoice.Pick", "EquipChoice.RetryPick")
            if selection is not None:
                buttons = frame.rows("EquipChoice.ReplaceConfirm", roi=selection["buttonROI"])
                if len(buttons) == 1:
                    x, _, _, _ = selection["buttonROI"]
                    return (x + 132, 596, 1, 1)
            return None
        if kind != "pick" or clean(frame.text("EquipChoice.ReplaceCarrier", min_score=0.70)) != self.replace_on:
            return None
        matches = []
        for name_roi, button_roi in REPLACEMENT_COLUMNS:
            name = clean(frame.text("EquipChoice.ReplaceItem", roi=name_roi, min_score=0.90))
            if name in self.options:
                matches.append((self.options[name], button_roi))
        if len(matches) != 1:
            return None
        item_id, button_roi = matches[0]
        buttons = frame.rows("EquipChoice.ReplaceButton", roi=button_roi)
        if len(buttons) == 1:
            return self.AnalyzeResult((button_roi[0] + 132, 596, 1, 1), {"id": item_id, "buttonROI": button_roi})
        return None


def choose_equipment_dialog(tasker: Tasker, headings: tuple[str, ...], options: dict[str, str],
                            expected_round: int, *, replace_on: str | None = None) -> dict:
    recognition = EquipmentChoice(headings, options, expected_round, replace_on=replace_on)
    override = {"EquipChoice.Pick": {"pre_delay": 300, "post_delay": 500},
                "EquipChoice.RetryPick": {"enabled": True}} if replace_on is not None else None
    with task_recognition(tasker, "ArkEquipChoice", recognition):
        job = run_task(tasker, "EquipChoice.Start", override)
    result = completed_data(job.nodes, "EquipChoice.Closed")
    if not job.succeeded or result is None:
        if replace_on is not None:
            raise RuntimeError(f"Equipment replacement was not verified for {replace_on}: {', '.join(options)}")
        raise RuntimeError(f"Equipment dialog was not verified for {' / '.join(headings)}")
    return result


def cancel_equipment_replacement(tasker: Tasker, expected_round: int) -> bool:
    recognition = EquipmentChoice((), {}, expected_round)
    with task_recognition(tasker, "ArkEquipChoice", recognition):
        job = run_task(tasker, "EquipChoice.Recover")
    if not job.finished("EquipChoice.ReplacementCleared"):
        raise RuntimeError("The interrupted equipment replacement did not return to preparation")
    return any(node.name == "EquipChoice.CancelReplacement" and node.completed for node in job.nodes)
