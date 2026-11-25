"""Read the displayed commander skill through its MaaFramework details pipeline."""

import argparse
import json
import re
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import add_session_args, session
from .native import Frame, completed_data
from .session import run as run_task, task_recognition
from .read import CATALOG, clean, index_names, integer, read_json


def command_choice_names(catalog: Path) -> dict[str, str]:
    effects = read_json(catalog / "effects.json")
    bands = read_json(catalog / "bands.json")
    candidates = {}
    for band in bands.values():
        for skill in band.get("skills") or []:
            effect_id = skill["effectId"]
            for raw in (effects.get(effect_id, {}).get("name"), (skill.get("desc") or "").partition("\n")[0]):
                label = clean(raw or "").strip("【】[]")
                if label:
                    candidates.setdefault(label, {})[effect_id] = skill.get("type")
    ordered = sorted(((label, *next(iter(ids.items()))) for label, ids in candidates.items() if len(ids) == 1),
                     key=lambda row: (row[2] != "PASSIVE", row[0]))
    return {label: effect_id for label, effect_id, _ in ordered}


class CommandRecognition(CustomRecognition):
    def __init__(self, catalog: Path):
        super().__init__()
        self.bands = index_names(read_json(catalog / "bands.json"))
        self.effects = read_json(catalog / "effects.json")
        self.commands = command_choice_names(catalog)

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        scene = frame.scene()
        if scene not in {"preparation", "commandDetails"} or not frame.hit("ArkRestText"):
            return None
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        if current_round is None or funds is None:
            return None
        state = {"round": current_round, "funds": funds}
        kind = json.loads(argv.custom_recognition_param)
        if kind == "closed":
            read = completed_data(argv.task_detail.nodes, "Command.ReadClose")
            if scene != "preparation" or read is None or read["state"] != state:
                return None
            return self.AnalyzeResult((0, 0, 1, 1), read["command"])
        if (kind, scene) not in {("read", "commandDetails"), ("open", "preparation")}:
            return None
        labels = frame.rows("Command.View")
        if len(labels) != 1 or frame.hit("Screen.OperatorDetails"):
            return None
        if kind == "read":
            opened = completed_data(argv.task_detail.nodes, "Command.Open")
            if opened is not None and opened["state"] != state:
                return None
            return self.AnalyzeResult(tuple(labels[0].box),
                                      {"state": state, "command": {**state, **self.details(frame)}})
        return self.AnalyzeResult(tuple(labels[0].box), {"state": state})

    def details(self, frame: Frame) -> dict:
        name = clean(frame.text("Command.Name"))
        title = clean(frame.text("Command.Title")).strip("【】[]")
        lines = frame.rows("Command.Body")
        top = min((row.box[1] for row in lines), default=0)
        lines.sort(key=lambda row: ((row.box[1] - top + 5) // 10, row.box[0]))
        description = "\n".join(row.text.strip() for row in lines)
        bands = self.bands.get(name, [])
        band_id, band = bands[0] if len(bands) == 1 else (None, {})
        skills = [skill for skill in band.get("skills", []) if title in {
            clean(self.effects.get(skill["effectId"], {}).get("name") or ""),
            clean((skill.get("desc") or "").split("\n", 1)[0]).strip("【】[]"),
        }] if title else []
        skill = skills[0] if len(skills) == 1 else {}
        display_counter = re.fullmatch(r"([0-9]+)/([0-9]+)", clean(frame.text("Command.Counter")))
        progress = re.search(r"进度[:(]?([0-9]+)/([0-9]+)", clean(description))
        return {
            "bandId": band_id, "bandName": name or None,
            "effectId": skill.get("effectId") or self.commands.get(title), "effectName": title or None,
            "type": skill.get("type"), "targetType": skill.get("targetType"),
            "usable": False if skill.get("type") in {"PASSIVE", "NONE"} else None,
            "displayCounter": list(map(int, display_counter.groups())) if display_counter else None,
            "progress": list(map(int, progress.groups())) if progress else None,
            "description": description or None,
        }


def inspect_command(tasker: Tasker, catalog: Path) -> dict:
    """Read the held command and verify return to the same preparation round."""
    recognition = CommandRecognition(catalog)
    with task_recognition(tasker, "ArkCommand", recognition):
        job = run_task(tasker, "Command.Inspect")
    result = completed_data(job.nodes, "Command.Closed")
    if not job.succeeded or result is None:
        raise RuntimeError("The command details or return to preparation could not be verified")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_session_args(parser, catalog=CATALOG)
    args = parser.parse_args()
    try:
        tasker = session(args.adb, args.address, args.ocr_resource).tasker
        print(json.dumps(inspect_command(tasker, args.catalog), ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
