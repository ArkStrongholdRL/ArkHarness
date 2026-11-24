"""Inspect or operate the opening commander page through MaaFramework."""

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker
from PIL import Image

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .native import Frame
from .read import ROOT, clean, read_json


ENTRIES = {
    "observe": ("Commander.Observe", "Commander.ReadPage"),
    "preview": ("Commander.Preview", "Commander.PreviewOpened"),
    "confirm": ("Commander.Confirm", "Lifecycle.Squad"),
}
PORTRAIT_RADIUS = 40


class CommanderRecognition(CustomRecognition):
    def __init__(self, layouts: dict, slot: tuple[int, int] | None = None, *,
                 template: str | None = None, capture_portrait: bool = False):
        super().__init__()
        self.layouts = layouts
        self.slot = slot
        self.template = template
        self.capture_portrait = capture_portrait
        self.previous_name = ""
        self.snapshot = None
        self.portrait = None
        self.matched_slot = None

    def grid(self, frame: Frame, has_confirm: bool) -> dict | None:
        detail = frame.hit("Commander.GridWithDetail")
        plain = frame.hit("Commander.GridPlain")
        if plain != detail and has_confirm == detail:
            return self.layouts["gridWithDetail" if detail else "gridPlain"]
        return None

    def match_portrait(self, frame: Frame, grid: dict) -> tuple[int, int] | None:
        columns, rows = grid["columns"], grid["rows"]
        tolerance = 20
        padding = PORTRAIT_RADIUS + tolerance
        roi = [columns[0] - padding, rows[0] - padding,
               columns[-1] - columns[0] + 2 * padding, rows[-1] - rows[0] + 2 * padding]
        roi[3] = min(roi[3], frame.image.shape[0] - roi[1])
        result = frame.node("Commander.Portrait", template=self.template, roi=roi)
        if result is None:
            return None
        slots = set()
        for match in result.filtered_results:
            x, y, width, height = match.box
            cx, cy = x + width / 2, y + height / 2
            column = next((i for i, center in enumerate(columns, 1) if abs(cx - center) <= tolerance), None)
            row = next((i for i, center in enumerate(rows, 1) if abs(cy - center) <= tolerance), None)
            if row is not None and column is not None:
                slots.add((row, column))
        return slots.pop() if len(slots) == 1 else None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        has_confirm = frame.hit("Commander.ConfirmText")
        name = clean(frame.text("Commander.SelectedName")) if has_confirm else ""
        if kind == "snapshot":
            self.snapshot = {"screen": "strategySelection", "commander": name or None,
                             "canConfirm": bool(has_confirm and name)}
            if self.template and (grid := self.grid(frame, has_confirm)):
                self.matched_slot = self.match_portrait(frame, grid)
            return (0, 0, 1, 1)
        if self.slot is None:
            return None
        row, col = self.slot
        if kind == "slot":
            grid = self.grid(frame, has_confirm)
            if grid is None:
                return None
            self.previous_name = name
            x, y = grid["columns"][col - 1], grid["rows"][row - 1]
            if self.capture_portrait:
                radius = PORTRAIT_RADIUS
                self.portrait = argv.image[y - radius:y + radius, x - radius:x + radius, ::-1].copy()
            return x, y, 1, 1
        if kind == "selection" and name:
            if name != self.previous_name:
                return (0, 0, 1, 1)
            layout = self.layouts["gridWithDetail"]
            x, y = layout["columns"][col - 1], layout["rows"][row - 1]
            if frame.hit("Commander.SelectedBadge", roi=[x - 45, y - 81, 90, 25]):
                return (0, 0, 1, 1)
        return None


def run(tasker: Tasker, action: str, layouts: dict, slot: tuple[int, int] | None = None,
        expected_name: str | None = None) -> dict:
    if action == "confirm" and not expected_name:
        raise ValueError("confirm requires the exact displayed commander name")
    if action == "preview" and slot is None:
        raise ValueError("preview requires a visible row and column")
    return run_pipeline(tasker, action, CommanderRecognition(layouts, slot), expected_name)


def run_pipeline(tasker: Tasker, action: str, recognition: CommanderRecognition,
                 expected_name: str | None = None) -> dict:
    resource = tasker.resource
    if not resource.register_custom_recognition("ArkCommander", recognition):
        raise RuntimeError("Could not register the commander recognizer")
    entry, terminal = ENTRIES[action]
    override = {}
    if expected_name:
        override = {"Commander.ExpectedName": {"expected": "^" + re.escape(expected_name) + "$"}}
    try:
        job = tasker.post_task(entry, override).wait()
        detail = job.get()
        nodes = [node.name for node in detail.nodes] if detail else []
    finally:
        resource.unregister_custom_recognition("ArkCommander")

    if not job.succeeded or not nodes or nodes[-1] != terminal:
        if "Commander.ConfirmExpected" in nodes:
            raise RuntimeError("Confirm was attempted once; the squad page was not verified")
        if "Commander.ClickSlot" in nodes:
            raise RuntimeError("Slot was clicked once; the requested preview was not verified")
        raise RuntimeError("Commander page, slot, or expected name did not match; no input was sent")
    if action == "confirm":
        return {"commander": expected_name, "after": "squadSelection"}
    if recognition.snapshot is None:
        raise RuntimeError("Commander page was not read")
    return recognition.snapshot


def save_portrait(path: Path, recognition: CommanderRecognition) -> bool:
    if recognition.portrait is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent) as directory:
            temp_path = Path(directory) / path.name
            Image.fromarray(recognition.portrait).save(temp_path)
            temp_path.replace(path)
    except OSError:
        return False
    return True


def select(tasker: Tasker, layouts: dict, name: str, *,
           resource_path: Path = OCR_RESOURCE, fallback_names: tuple[str, ...] = ()) -> dict:
    if not name:
        raise ValueError("select requires the exact displayed commander name")

    template = f"commander/{sha256(name.encode()).hexdigest()}.png"
    cache_path = resource_path / "image" / template
    probe = CommanderRecognition(layouts, template=template if cache_path.is_file() else None)
    current = run_pipeline(tasker, "observe", probe)
    scanned = []
    if current["commander"] == name:
        return {**run(tasker, "confirm", layouts, expected_name=name), "scanned": scanned,
                "matchedBy": "currentPreview"}

    candidate = probe.matched_slot
    if candidate is not None:
        row, column = candidate
        try:
            preview = run(tasker, "preview", layouts, candidate)
        except RuntimeError:
            candidate = None
        else:
            scanned.append({"row": row, "column": column, "name": preview["commander"]})
            if preview["commander"] == name:
                return {**run(tasker, "confirm", layouts, expected_name=name), "scanned": scanned,
                        "matchedBy": "template"}

    grid = layouts["gridWithDetail"]
    for row, _ in enumerate(grid["rows"], 1):
        for column, _ in enumerate(grid["columns"], 1):
            if (row, column) == candidate:
                continue
            recognition = CommanderRecognition(layouts, (row, column), capture_portrait=True)
            try:
                preview = run_pipeline(tasker, "preview", recognition)
            except RuntimeError as error:
                if fallback_names:
                    continue
                raise RuntimeError(f"Commander preview at row {row}, column {column}: {error}") from error
            scanned.append({"row": row, "column": column, "name": preview["commander"]})
            if preview["commander"] == name:
                result = run(tasker, "confirm", layouts, expected_name=name)
                return {**result, "scanned": scanned, "matchedBy": "grid",
                        "templateCached": save_portrait(cache_path, recognition)}
    for alternative in fallback_names:
        slot = next(((entry["row"], entry["column"]) for entry in scanned if entry["name"] == alternative), None)
        if slot is not None and run(tasker, "preview", layouts, slot)["commander"] == alternative:
            result = run(tasker, "confirm", layouts, expected_name=alternative)
            return {**result, "scanned": scanned, "matchedBy": "availableCommander", "requested": name}
    raise RuntimeError(f"Commander {name!r} was not found in the visible strategy grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(*ENTRIES, "select"))
    parser.add_argument("--slot", nargs=2, type=int, metavar=("ROW", "COLUMN"))
    parser.add_argument("--name", help="Exact displayed commander name for select or confirm")
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
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
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        slots = read_json(ROOT / "calib/slots.json")
        tasker = load_ocr(connect(args.adb, args.address), args.ocr_resource)
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
