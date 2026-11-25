"""Inspect or operate the opening commander page through MaaFramework."""

import argparse
import json
import re
from base64 import b64decode, b64encode
from collections.abc import Collection
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker
from PIL import Image

from .capture import OCR_RESOURCE, add_session_args, session
from .native import MIN_NAME_SCORE, Frame, completed_data
from .names import match_name
from .session import run as run_task, task_recognition
from .read import CATALOG, ROOT, clean, index_names, read_json


ENTRIES = {
    "observe": ("Commander.Observe", "Commander.ReadPage"),
    "preview": ("Commander.Preview", "Commander.PreviewOpened"),
    "confirm": ("Commander.Confirm", "Lifecycle.Squad"),
}
PORTRAIT_RADIUS = 40


class CommanderRecognition(CustomRecognition):
    def __init__(self, layouts: dict, slot: tuple[int, int] | None = None, *,
                 templates: dict[str, str] | None = None, capture_portrait: bool = False, names: Collection[str] = ()):
        super().__init__()
        self.layouts = layouts
        self.slot = slot
        self.templates = templates or {}
        self.capture_portrait = capture_portrait
        self.names = names

    def grid(self, frame: Frame, has_confirm: bool) -> dict | None:
        detail = frame.hit("Commander.GridWithDetail")
        plain = frame.hit("Commander.GridPlain")
        if plain != detail and has_confirm == detail:
            return self.layouts["gridWithDetail" if detail else "gridPlain"]
        return None

    def match_portraits(self, frame: Frame, grid: dict) -> dict:
        columns, rows = grid["columns"], grid["rows"]
        tolerance = 20
        padding = PORTRAIT_RADIUS + tolerance
        roi = [columns[0] - padding, rows[0] - padding,
               columns[-1] - columns[0] + 2 * padding, rows[-1] - rows[0] + 2 * padding]
        roi[3] = min(roi[3], frame.image.shape[0] - roi[1])
        matched = {}
        for name, template in self.templates.items():
            result = frame.node("Commander.Portrait", template=template, roi=roi)
            slots = set()
            for match in result.filtered_results if result is not None else ():
                x, y, width, height = match.box
                cx, cy = x + width / 2, y + height / 2
                column = next((i for i, center in enumerate(columns, 1) if abs(cx - center) <= tolerance), None)
                row = next((i for i, center in enumerate(rows, 1) if abs(cy - center) <= tolerance), None)
                if row is not None and column is not None:
                    slots.add((row, column))
            if len(slots) == 1:
                matched[name] = slots.pop()
        return matched

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        has_confirm = frame.hit("Commander.ConfirmText")
        observed = clean(frame.text("Commander.SelectedName", min_score=MIN_NAME_SCORE)) if has_confirm else ""
        name = match_name(observed, self.names) if self.names else observed
        if kind == "snapshot":
            result = {"snapshot": {"screen": "strategySelection", "commander": name or None,
                                   "canConfirm": bool(has_confirm and name)}}
            if name and name == observed:
                result["exactName"] = name
            if self.templates and (grid := self.grid(frame, has_confirm)):
                result["matchedSlots"] = self.match_portraits(frame, grid)
            return self.AnalyzeResult((0, 0, 1, 1), result)
        if self.slot is None:
            return None
        row, col = self.slot
        if kind == "slot":
            grid = self.grid(frame, has_confirm)
            if grid is None:
                return None
            x, y = grid["columns"][col - 1], grid["rows"][row - 1]
            picked = {"previousName": name}
            if self.capture_portrait:
                radius = PORTRAIT_RADIUS
                portrait = argv.image[y - radius:y + radius, x - radius:x + radius, ::-1]
                with BytesIO() as png:
                    Image.fromarray(portrait).save(png, format="PNG")
                    picked["portrait"] = b64encode(png.getvalue()).decode("ascii")
            return self.AnalyzeResult((x, y, 1, 1), picked)
        if kind == "selection" and name:
            picked = completed_data(argv.task_detail.nodes, "Commander.ClickSlot")
            if picked is None:
                return None
            if name != picked["previousName"]:
                return (0, 0, 1, 1)
            layout = self.layouts["gridWithDetail"]
            x, y = layout["columns"][col - 1], layout["rows"][row - 1]
            if frame.hit("Commander.SelectedBadge", roi=[x - 45, y - 81, 90, 25]):
                return (0, 0, 1, 1)
        return None


def run(tasker: Tasker, action: str, layouts: dict, slot: tuple[int, int] | None = None,
        expected_name: str | None = None, *, names: Collection[str] = ()) -> dict:
    if action == "confirm" and not expected_name:
        raise ValueError("confirm requires the exact displayed commander name")
    if action == "preview" and slot is None:
        raise ValueError("preview requires a visible row and column")
    result = run_pipeline(tasker, action, CommanderRecognition(layouts, slot, names=names), expected_name)
    return result if action == "confirm" else result["snapshot"]


def run_pipeline(tasker: Tasker, action: str, recognition: CommanderRecognition,
                 expected_name: str | None = None) -> dict:
    entry, terminal = ENTRIES[action]
    override = {}
    if expected_name:
        override = {"Commander.ExpectedName": {"expected": "^" + re.escape(expected_name) + "$"}}
    with task_recognition(tasker, "ArkCommander", recognition):
        job = run_task(tasker, entry, override)
    nodes = job.nodes

    if not job.finished(terminal):
        if any(node.name == "Commander.ConfirmExpected" and node.action is not None for node in nodes):
            raise RuntimeError("Confirm was attempted once; the squad page was not verified")
        if any(node.name == "Commander.ClickSlot" and node.action is not None for node in nodes):
            raise RuntimeError("Slot was clicked once; the requested preview was not verified")
        raise RuntimeError("Commander page, slot, or expected name did not match; no input was sent")
    if action == "confirm":
        return {"commander": expected_name, "after": "squadSelection"}
    result = completed_data(nodes, terminal)
    if result is None:
        raise RuntimeError("Commander page was not read")
    if action == "preview":
        picked = completed_data(nodes, "Commander.ClickSlot")
        if picked is not None and "portrait" in picked:
            result["portrait"] = picked["portrait"]
    return result


def save_portrait(path: Path, portrait: str | None) -> bool:
    if portrait is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as directory:
            temp_path = Path(directory) / path.name
            temp_path.write_bytes(b64decode(portrait))
            temp_path.replace(path)
    except OSError:
        return False
    return True


def portrait_template(name: str) -> str:
    return f"commander/{sha256(name.encode()).hexdigest()}.png"


def select(tasker: Tasker, layouts: dict, name: str, *,
           resource_path: Path = OCR_RESOURCE, fallback_names: tuple[str, ...] = (), catalog: Path = CATALOG) -> dict:
    if not name:
        raise ValueError("select requires the exact displayed commander name")

    image_dir = resource_path / "image"
    ordered = tuple(dict.fromkeys((name, *fallback_names)))
    templates = {choice: portrait_template(choice) for choice in ordered}
    cached = {choice: path for choice, path in templates.items() if (image_dir / path).is_file()}
    names = index_names(read_json(catalog / "bands.json"))
    probe = CommanderRecognition(layouts, templates=cached, names=names)
    observed = run_pipeline(tasker, "observe", probe)
    current = observed["snapshot"]
    scanned = []
    if current["commander"] == name:
        return {**run(tasker, "confirm", layouts, expected_name=name), "scanned": scanned,
                "matchedBy": "currentPreview"}

    matched = observed.get("matchedSlots", {})
    for choice in ordered:
        if choice not in matched:
            continue
        candidate = tuple(matched[choice])
        row, column = candidate
        try:
            preview = run_pipeline(tasker, "preview", CommanderRecognition(layouts, candidate, names=names))
        except RuntimeError:
            continue
        scanned.append({"row": row, "column": column, "name": preview["snapshot"]["commander"]})
        if preview.get("exactName") == choice:
            return {**run(tasker, "confirm", layouts, expected_name=choice), "scanned": scanned,
                    "matchedBy": "template", "requested": name}

    visited = {(entry["row"], entry["column"]) for entry in scanned}
    captured = []
    grid = layouts["gridWithDetail"]
    for row, _ in enumerate(grid["rows"], 1):
        for column, _ in enumerate(grid["columns"], 1):
            if (row, column) in visited:
                continue
            recognition = CommanderRecognition(layouts, (row, column), capture_portrait=True, names=names)
            try:
                observed = run_pipeline(tasker, "preview", recognition)
            except RuntimeError as error:
                if fallback_names:
                    continue
                raise RuntimeError(f"Commander preview at row {row}, column {column}: {error}") from error
            preview = observed["snapshot"]
            actual = preview["commander"]
            scanned.append({"row": row, "column": column, "name": actual})
            if exact := observed.get("exactName"):
                path = image_dir / portrait_template(exact)
                if not path.is_file() and save_portrait(path, observed.get("portrait")):
                    captured.append(exact)
            if preview["commander"] == name:
                result = run(tasker, "confirm", layouts, expected_name=name)
                return {**result, "scanned": scanned, "matchedBy": "grid",
                        "templatesCached": captured}
    for alternative in fallback_names:
        slot = next(((entry["row"], entry["column"]) for entry in scanned if entry["name"] == alternative), None)
        if slot is not None and run(tasker, "preview", layouts, slot, names=names)["commander"] == alternative:
            result = run(tasker, "confirm", layouts, expected_name=alternative)
            return {**result, "scanned": scanned, "matchedBy": "availableCommander", "requested": name,
                    "templatesCached": captured}
    raise RuntimeError(f"Commander {name!r} was not found in the visible strategy grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(*ENTRIES, "select"))
    parser.add_argument("--slot", nargs=2, type=int, metavar=("ROW", "COLUMN"))
    parser.add_argument("--name", help="Exact displayed commander name for select or confirm")
    add_session_args(parser)
    args = parser.parse_args()
    slot = tuple(args.slot) if args.slot else None
    if slot and any(value not in range(1, 5) for value in slot):
        parser.error("--slot uses visible rows and columns 1 through 4")
    if args.action == "preview" and slot is None:
        parser.error("preview requires --slot ROW COLUMN")
    name = clean(args.name) if args.name else None
    if args.action in {"select", "confirm"} and not name:
        parser.error(f"{args.action} requires --name")
    try:
        slots = read_json(ROOT / "calib/slots.json")
        tasker = session(args.adb, args.address, args.ocr_resource).tasker
        layouts = slots["commander"]
        if args.action == "select":
            result = select(tasker, layouts, name, resource_path=args.ocr_resource)
        else:
            result = run(tasker, args.action, layouts, slot, name)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
