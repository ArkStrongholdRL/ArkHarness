"""Inspect and confirm one of the two briefing bond groups through MaaFramework."""

import json
import logging

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .names import match_name
from .native import MIN_NAME_SCORE, Frame, completed_data
from .session import run as run_task, task_recognition
from .read import Reader, clean


GROUP_REGIONS = ([22, 130, 423, 447], [445, 130, 444, 447])
# The briefing layout is fixed: page title, enemy leader name, core bond names. Reading just these beats a full-screen OCR.
BRIEFING_TEXT_ROIS = ([400, 10, 320, 50], [915, 295, 345, 140], [15, 240, 875, 235])
LOG = logging.getLogger(__name__)


def enabled_addon_bonds(frame: Frame, reader: Reader) -> list[str]:
    """Add-on bond ids listed in the briefing's add-on row. A bond missing from the row is disabled for the match."""
    found = set()
    for row in frame.rows("Briefing.AddonName"):
        if row.score < MIN_NAME_SCORE:
            continue
        candidates = [key for key, data in reader.bonds.get(match_name(clean(row.text), reader.bonds), [])
                      if not data.get("isCore")]
        if len(candidates) == 1:
            found.add(candidates[0])
    return sorted(found)


class BriefingRecognition(CustomRecognition):
    def __init__(self, reader: Reader, group: int | None, expected_groups: list[list[str]] | None):
        super().__init__()
        self.reader = reader
        self.group = group
        self.expected_groups = expected_groups

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if not frame.hit("Lifecycle.Briefing"):
            return None
        frame.text_rois = BRIEFING_TEXT_ROIS
        observed = self.reader.read(frame)
        state = observed["match"]
        choice = state["bondChoice"]
        if choice is None:
            return None
        groups = choice["groups"]
        disabled = set(state["disabledBonds"] or [])
        selected = next((idx for idx in range(2) if set(groups[1 - idx]) == disabled), None)
        kind = json.loads(argv.custom_recognition_param)
        switched = completed_data(argv.task_detail.nodes, "Briefing.Switch") if kind == "confirm" else None
        expected_groups = switched["groups"] if switched is not None else self.expected_groups
        if expected_groups is not None and any(set(read) != set(expected)
                for read, expected in zip(groups, expected_groups, strict=True)):
            LOG.debug("The displayed bond groups differ from the expected groups")
            return None
        if kind == "switch":
            if selected is None or selected == self.group:
                return None
            # Verified on device: the unselected group has no text button; tapping inside its panel selects it.
            x, y, width, height = GROUP_REGIONS[self.group]
            return self.AnalyzeResult((x + width // 2 - 20, y + height // 2 - 20, 40, 40), {"groups": groups})
        if kind == "confirm":
            if selected != self.group:
                LOG.debug("The requested bond group was not highlighted")
                return None
            buttons = frame.rows("Entry.ReadyButton")
            if len(buttons) != 1:
                LOG.debug("No unique briefing confirmation button was verified")
                return None
            hit = tuple(buttons[0].box)
        elif kind == "read":
            hit = (0, 0, 1, 1)
        else:
            return None
        detail = {"groups": groups, "previewSelected": selected,
                  "disabledBonds": state["disabledBonds"], "bossId": state["bossId"],
                  "modeId": observed["config"]["modeId"]}
        if kind == "read":  # the add-on row costs an extra OCR pass; only the inspection result needs it
            detail["enabledAddonBonds"] = enabled_addon_bonds(frame, self.reader)
        return self.AnalyzeResult(hit, detail)


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
        job = run_task(tasker, entry)
    nodes = job.nodes
    steps = [node.name for node in nodes]
    result = completed_data(nodes, "Briefing.Read" if group is None else "Briefing.Confirm")
    if job.finished(terminal) and result is not None:
        if group is None:
            return result
        return {**result, "bondChoice": {"groups": result["groups"], "picked": group},
                "after": "strategySelection", "steps": steps}
    if any(node.name == "Briefing.Confirm" and node.action is not None for node in nodes):
        raise RuntimeError("Briefing confirmation was attempted; the next screen could not be verified")
    if any(node.name == "Briefing.Switch" and node.action is not None for node in nodes):
        raise RuntimeError("Group preview switch was attempted; briefing was not confirmed")
    raise RuntimeError("No briefing input was sent: The bond groups or briefing controls could not be verified")
