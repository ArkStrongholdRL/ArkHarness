"""Verify hand-slot occupants through the MaaFramework preparation pipeline."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .native import Frame
from .read import CATALOG, ROOT, Reader, chess_identity, integer, read_json


# An operator standing on the bench carries a dark tier badge above its slot (collapsed-shop view). Measured on device:
# occupied slots 0.51–0.52 dark pixels in the badge ROI, empty bench 0.00–0.04, one slot beside dark terrain 0.18.
# Anything at or above the threshold is still clicked, so only slots that are clearly bare skip the click.
BADGE_OFFSET = (-29, -86)
BADGE_HALF = (18, 14)
BADGE_DARK = 90
BADGE_MAYBE = 0.12


def bench_candidates(image, reader: Reader) -> list[int]:
    """Screen slots that may hold an operator, from one frame of the collapsed-shop view (no input)."""
    centers = reader.slots["hand"]["centersByView"]["normal"]
    found = []
    for slot, (x, y) in enumerate(centers):
        cx, cy = x + BADGE_OFFSET[0], y + BADGE_OFFSET[1]
        roi = image[cy - BADGE_HALF[1]:cy + BADGE_HALF[1], cx - BADGE_HALF[0]:cx + BADGE_HALF[0]]
        if roi.size and float((roi.max(axis=2) < BADGE_DARK).mean()) >= BADGE_MAYBE:
            found.append(slot)
    return found


class HandRecognition(CustomRecognition):
    def __init__(self, reader: Reader):
        super().__init__()
        self.reader = reader
        self.stage = None
        self.hud = None
        self.slot = None
        self.center = None
        self.result = None

    def select_slot(self, slot: int) -> None:
        self.slot = slot
        self.center = self.reader.slots["hand"]["centersByView"]["normal"][slot]
        self.result = None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if frame.scene() != "preparation" or not frame.hit("ArkRestText") or frame.hit("Shop.Expanded"):
            return None
        kind = json.loads(argv.custom_recognition_param)
        board_details = frame.hit("Withdraw.Icon")
        hand_details = frame.hit("HandSale.Button") or frame.hit("Screen.OperatorDetails")
        details = board_details or hand_details
        hit = (0, 0, 1, 1)
        if kind == "reset":
            return hit if details else None
        if kind == "read":
            if not hand_details or board_details:
                return None
            self.result = {"screenSlot": self.slot, "status": "occupied", **chess_identity(self.reader, frame)}
            return hit

        hud = (
            integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)"),
            integer(frame.text("Screen.Funds")),
            integer(frame.text("Screen.DeployCollapsed"), r"剩余可放置角色:([0-9]+)"),
        )
        if None in hud or (self.hud is not None and self.hud != hud):
            return None
        if details:
            return None
        stage = self.reader.stage_hint or frame.stage_id()
        if stage is None or (self.stage is not None and stage != self.stage):
            return None
        if kind == "pick":
            self.stage, self.hud = stage, hud
            return (*self.center, 1, 1)
        if kind == "empty":
            self.result = {"screenSlot": self.slot, "status": "empty", "name": None,
                           "id": None, "baseId": None, "candidateIds": []}
        return hit


def read_hand(tasker: Tasker, reader: Reader, indices: list[int] | None = None, *, skip_bare: bool = False) -> dict:
    """Return hand slots from one preparation round and restore its anchor.

    skip_bare: slots whose tier badge is clearly absent are reported empty without a click (collapsed-shop view only).
    At least one slot is always clicked, which also reads the HUD (round, funds, remaining placements, stage).
    """
    capacity = len(reader.slots["hand"]["centersByView"]["normal"])
    selected = list(range(capacity)) if indices is None else indices
    bare = []
    if skip_bare and indices is None:
        job = tasker.controller.post_screencap().wait()
        if not job.succeeded or job.get().shape != (720, 1280, 3):
            raise RuntimeError("Could not capture the bench frame")
        selected = bench_candidates(job.get(), reader) or [0]
        bare = [slot for slot in range(capacity) if slot not in selected]
    if not selected or any(type(slot) is not int or not 0 <= slot < capacity for slot in selected):
        raise ValueError("Expected valid zero-based screen hand slots")
    if len(set(selected)) != len(selected):
        raise ValueError("Hand slots must be unique")

    recognition = HandRecognition(reader)
    resource = tasker.resource
    if not resource.register_custom_recognition("ArkHand", recognition):
        raise RuntimeError("Could not register hand inspection")
    observations = []
    try:
        for position, slot in enumerate(selected):
            recognition.select_slot(slot)
            override = {"Hand.Inspect": {"next": ["Hand.Pick"]}} if position else {}
            job = tasker.post_task("Hand.Inspect", override).wait()
            detail = job.get()
            last = detail.nodes[-1].name if detail and detail.nodes else ""
            if not job.succeeded or last not in ("Hand.Empty", "Hand.Closed") or recognition.result is None:
                raise RuntimeError(f"Hand slot {slot} or return to preparation could not be verified")
            observations.append(recognition.result)
    finally:
        resource.unregister_custom_recognition("ArkHand")
    observations += [{"screenSlot": slot, "status": "empty", "name": None, "id": None, "baseId": None,
                      "candidateIds": []} for slot in bare]
    observations.sort(key=lambda row: row["screenSlot"])
    return {"stageId": recognition.stage, "round": recognition.hud[0], "funds": recognition.hud[1],
            "remaining": recognition.hud[2], "slots": observations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-slot", type=int, action="append", dest="indices")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    args = parser.parse_args()
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        reader = Reader(args.catalog, read_json(ROOT / "calib/slots.json"))
        tasker = load_ocr(connect(args.adb, args.address), args.ocr_resource)
        result = read_hand(tasker, reader, args.indices)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
