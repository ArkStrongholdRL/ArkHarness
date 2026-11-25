"""Verify hand-slot occupants through the MaaFramework preparation pipeline."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import add_session_args, session
from .native import Frame, completed_data
from .session import run as run_task, task_recognition
from .read import CATALOG, Reader, chess_identity, integer, screen_integer


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
        self.center = None
        self.anchor = None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if frame.scene() != "preparation" or not frame.hit("ArkRestText") or frame.hit("Shop.Expanded"):
            return None
        kind = json.loads(argv.custom_recognition_param)
        board_details = frame.hit("Withdraw.Icon")
        hand_details = frame.hit("HandSale.Button") or frame.hit("Screen.OperatorDetails")
        token = None
        if not board_details and not hand_details and kind in {"read", "empty", "closed", "reset"}:
            candidate = chess_identity(self.reader, frame)
            if candidate.get("kind") == "token":
                token = candidate
        details = board_details or hand_details or token is not None
        hit = (0, 0, 1, 1)
        if kind == "reset":
            return hit if details else None
        if kind == "read":
            if board_details or (not hand_details and token is None):
                return None
            return self.AnalyzeResult(hit, {"status": "occupied", **(token or chess_identity(self.reader, frame))})
        if details:
            return None
        hud = {
            "round": integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)"),
            "funds": integer(frame.text("Screen.Funds")),
            "remaining": screen_integer(frame, "Screen.DeployCollapsed"),
            "stageId": self.reader.stage_hint or frame.stage_id(),
        }
        if None in hud.values() or (self.anchor is not None and self.anchor != hud):
            return None
        if kind == "pick":
            return self.AnalyzeResult((*self.center, 1, 1), hud)
        if completed_data(argv.task_detail.nodes, "Hand.Pick") != hud:
            return None
        if kind == "empty":
            return self.AnalyzeResult(hit, {"status": "empty", "name": None, "id": None,
                                            "baseId": None, "candidateIds": []})
        if kind == "closed":
            result = completed_data(argv.task_detail.nodes, "Hand.ReadDetails")
            if result is not None:
                return self.AnalyzeResult(hit, result)
        return None


def read_hand(tasker: Tasker, reader: Reader, indices: list[int] | None = None, *,
              skip_bare: bool = False, known_summons: tuple[int, ...] = ()) -> dict:
    """Return hand slots from one preparation round and restore its anchor.

    skip_bare: slots whose tier badge is clearly absent are reported empty without a click (collapsed-shop view only).
    At least one slot is always clicked, which also reads the HUD (round, funds, remaining placements, stage).
    """
    centers = reader.slots["hand"]["centersByView"]["normal"]
    capacity = len(centers)
    selected = list(range(capacity)) if indices is None else indices
    bare = []
    if skip_bare and indices is None:
        job = tasker.controller.post_screencap().wait()
        if not job.succeeded or job.get().shape != (720, 1280, 3):
            raise RuntimeError("Could not capture the bench frame")
        selected = sorted(set(bench_candidates(job.get(), reader)).union(known_summons)) or [0]
        bare = [slot for slot in range(capacity) if slot not in selected]
    if not selected or any(type(slot) is not int or not 0 <= slot < capacity for slot in selected):
        raise ValueError("Expected valid zero-based screen hand slots")
    if len(set(selected)) != len(selected):
        raise ValueError("Hand slots must be unique")

    anchor = None
    observations = []
    recognition = HandRecognition(reader)
    with task_recognition(tasker, "ArkHand", recognition):
        for position, slot in enumerate(selected):
            recognition.center, recognition.anchor = centers[slot], anchor
            job = run_task(tasker, "Hand.InspectNext" if position else "Hand.Inspect")
            nodes = job.nodes
            result = completed_data(nodes, "Hand.Empty", "Hand.Closed")
            anchor = completed_data(nodes, "Hand.Pick")
            if not job.succeeded or result is None or anchor is None:
                raise RuntimeError(f"Hand slot {slot} or return to preparation could not be verified")
            observations.append({"screenSlot": slot, **result})
    observations += [{"screenSlot": slot, "status": "empty", "name": None, "id": None, "baseId": None,
                      "candidateIds": []} for slot in bare]
    observations.sort(key=lambda row: row["screenSlot"])
    return {**anchor, "slots": observations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-slot", type=int, action="append", dest="indices")
    add_session_args(parser, catalog=CATALOG)
    args = parser.parse_args()
    try:
        screen_session = session(args.adb, args.address, args.ocr_resource, catalog=args.catalog)
        tasker, reader = screen_session.tasker, screen_session.reader
        result = read_hand(tasker, reader, args.indices)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
