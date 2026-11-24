"""Inspect and confirm one of the two briefing bond groups through MaaFramework."""

import json

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .native import Frame
from .session import task_recognition
from .read import Reader


GROUP_REGIONS = ([22, 130, 423, 447], [445, 130, 444, 447])


class BriefingRecognition(CustomRecognition):
    def __init__(self, reader: Reader, group: int | None, expected_groups: list[list[str]] | None):
        super().__init__()
        self.reader = reader
        self.group = group
        self.expected_groups = expected_groups
        self.result = None
        self.reason = "Both core-bond groups were not verified on the briefing page"

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if not frame.hit("Lifecycle.Briefing"):
            return None
        observed = self.reader.read(frame)
        state = observed["match"]
        choice = state["bondChoice"]
        if choice is None:
            return None
        groups = choice["groups"]
        disabled = set(state["disabledBonds"] or [])
        selected = next((idx for idx in range(2) if set(groups[1 - idx]) == disabled), None)
        if self.expected_groups is not None and any(set(read) != set(expected)
                for read, expected in zip(groups, self.expected_groups, strict=True)):
            self.reason = "The displayed bond groups differ from the expected groups"
            return None
        kind = json.loads(argv.custom_recognition_param)
        if kind == "switch":
            if selected is None or selected == self.group:
                return None
            candidates = frame.rows("Briefing.GroupButton", roi=GROUP_REGIONS[self.group])
            if len(candidates) != 1:
                self.reason = "No unique selection button was verified for the requested group"
                return None
            self.expected_groups = groups
            return tuple(candidates[0].box)
        if kind == "confirm":
            if selected != self.group:
                self.reason = "The requested bond group was not highlighted"
                return None
            buttons = frame.rows("Entry.ReadyButton")
            if len(buttons) != 1:
                self.reason = "No unique briefing confirmation button was verified"
                return None
            hit = tuple(buttons[0].box)
        else:
            hit = (0, 0, 1, 1)
        self.result = {"groups": groups, "previewSelected": selected, "disabledBonds": state["disabledBonds"],
                       "bossId": state["bossId"], "modeId": observed["config"]["modeId"]}
        return hit


def inspect_briefing(tasker: Tasker, reader: Reader) -> dict:
    return run_briefing(tasker, reader, group=None, expected_groups=None)


def choose_bond_group(tasker: Tasker, reader: Reader, group: int, *,
                      expected_groups: list[list[str]] | None = None) -> dict:
    """Select a group by screen index and verify the transition to commander selection."""
    if type(group) is not int or group not in (0, 1):
        raise ValueError("Bond group must be 0 for left or 1 for right")
    if expected_groups is not None and (len(expected_groups) != 2 or any(
        len(row) != 5 or any(not isinstance(key, str) or not key for key in row) or len(set(row)) != 5
        for row in expected_groups
    )):
        raise ValueError("Expected groups must each contain five distinct bond IDs")
    return run_briefing(tasker, reader, group=group, expected_groups=expected_groups)


def run_briefing(tasker: Tasker, reader: Reader, *, group: int | None,
                 expected_groups: list[list[str]] | None) -> dict:
    recognition = BriefingRecognition(reader, group, expected_groups)
    entry, terminal = ("Briefing.Inspect", "Briefing.Read") if group is None else (
        "Briefing.Select", "Briefing.Confirmed")
    with task_recognition(tasker, "ArkBriefing", recognition):
        job = tasker.post_task(entry).wait()
        detail = job.get()
        steps = [node.name for node in detail.nodes] if detail else []
    if job.succeeded and steps and steps[-1] == terminal and recognition.result is not None:
        result = recognition.result
        if group is None:
            return result
        return {**result, "bondChoice": {"groups": result["groups"], "picked": group},
                "after": "strategySelection", "steps": steps}
    if "Briefing.Confirm" in steps:
        raise RuntimeError("Briefing confirmation was attempted; the next screen could not be verified")
    if "Briefing.Switch" in steps:
        raise RuntimeError(f"Group preview was changed; briefing was not confirmed: {recognition.reason}")
    raise RuntimeError(f"No briefing input was sent: {recognition.reason}")
