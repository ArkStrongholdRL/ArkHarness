"""Drag a named equipment item to a hand or board operator through MaaFramework."""

import argparse
import json
import re
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.define import MaaAdbInputMethodEnum
from maa.tasker import Tasker

from .board_read import read_board_tiles
from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .deploy import read_board, tile_point
from .equipment_read import read_equipment_slots
from .native import Frame
from .session import task_recognition
from .read import CATALOG, ROOT, Reader, clean, integer, read_json


def equipment_funds_match(gained: int, expected: int | None,
                          gain_range: tuple[int, int | None] | None) -> bool:
    if expected is not None and gained != expected:
        return False
    if gain_range is None:
        return True
    low, high = gain_range
    return gained >= low and (high is None or gained <= high)


class EquipRecognition(CustomRecognition):
    def __init__(self, expected_gain: int | None, *, expected_round: int | None = None,
                 stage: str | None = None, gain_range: tuple[int, int | None] | None = None):
        super().__init__()
        self.expected_gain = expected_gain
        self.gain_range = gain_range
        self.expected_round = expected_round
        self.stage = stage
        self.highlight_roi = None
        self.before = None
        self.after = None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        if kind == "boardHover":
            if frame.hit("Equip.TargetHighlight", roi=self.highlight_roi):
                return None
            return (0, 0, 1, 1)
        if frame.scene() != "preparation" or self.stage is not None and frame.stage_id() != self.stage:
            return None
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        funds = integer(frame.text("Screen.Funds"))
        if current_round is None or funds is None:
            return None
        if self.expected_round is not None and current_round != self.expected_round:
            return None
        state = {"round": current_round, "funds": funds}
        if kind == "before":
            if self.before is not None and state != self.before:
                return None
            self.before = state
        else:
            before = self.before
            if before is None or current_round != before["round"]:
                return None
            if not equipment_funds_match(funds - before["funds"], self.expected_gain, self.gain_range):
                return None
            self.after = state
        return (0, 0, 1, 1)


def equip_to_hand(tasker: Tasker, calibration: dict, item_slot: int, hand_slot: int,
                  name: str, *, expected_gain: int | None = None, reader: Reader | None = None,
                  expected_round: int | None = None, opens_choice: bool = False,
                  gain_range: tuple[int, int | None] | None = None) -> dict:
    targets = calibration["hand"]["centersByView"]["drag"]
    if type(hand_slot) is not int or not 0 <= hand_slot < len(targets):
        raise ValueError("A valid zero-based hand slot is required")
    result = drag_equipment(tasker, calibration, item_slot, name, targets[hand_slot],
                            EquipRecognition(expected_gain, expected_round=expected_round, gain_range=gain_range),
                            entry="Equip.Hand", opens_choice=opens_choice)
    if reader is not None and not opens_choice and not expected_gain:
        verify_rack_source(tasker, reader, result)
    return {"handSlot": hand_slot, **result}


def equip_to_board(tasker: Tasker, reader: Reader, board: dict, item_slot: int, name: str,
                   row: int, col: int, operator_name: str, *, expected_gain: int | None = None,
                   expected_round: int | None = None, opens_choice: bool = False,
                   gain_range: tuple[int, int | None] | None = None) -> dict:
    """Drag to the verified board occupant and return the observed release result."""
    validate_equipment(reader.slots, item_slot, name, expected_gain)
    operator_name = clean(operator_name)
    if not operator_name:
        raise ValueError("An exact displayed operator name is required")
    target = tile_point(board, row, col)
    tracked = expected_round is not None
    if not tracked:
        snapshot = read_board_tiles(tasker, reader, board, inspect_tiles={(row, col)})
        if len(snapshot["pieces"]) != 1 or snapshot["pieces"][0]["name"] != operator_name:
            raise RuntimeError("The expected board equipment recipient was not verified; no drag was sent")
        expected_round = snapshot["round"]
    recognition = EquipRecognition(expected_gain, expected_round=expected_round,
                                   stage=None if tracked else board["stageId"], gain_range=gain_range)
    result = drag_equipment(tasker, reader.slots, item_slot, name, target, recognition, entry="Equip.Board",
                            tracked=tracked, opens_choice=opens_choice)
    if not opens_choice and not expected_gain:
        verify_rack_source(tasker, reader, result)
    return {"stageId": board["stageId"], "row": row, "col": col,
            "operatorName": operator_name, "attachmentVerified": False, **result}


def verify_rack_source(tasker: Tasker, reader: Reader, result: dict) -> None:
    try:
        rack = read_equipment_slots(tasker, reader, [result["itemSlot"]])
        after = result["after"]
        slot = rack["slots"][0]
        cleared = slot["status"] == "empty" or (slot.get("id") and clean(slot.get("name") or "") != clean(result["name"]))
        if not cleared or (rack["round"], rack["funds"]) != (after["round"], after["funds"]):
            raise RuntimeError("The original item was not verified gone from its rack slot in the same state")
    except RuntimeError as error:
        raise RuntimeError(f"Equipment was released but rack clearance is unverified: {error}") from error
    result["sourceCleared"] = True
    result["slotAfter"] = slot


def destroy_item(tasker: Tasker, reader: Reader, item_slot: int, name: str, *,
                 expected_round: int, replacement_id: str | None = None) -> dict:
    """Drag one verified item to the visible destroy control and read its former slot."""
    if replacement_id is not None and any(kind == "item" and item_id == replacement_id
                                         for kind, item_id, _ in reader.shop_names.get(clean(name), [])):
        raise ValueError("The replacement and original item would be indistinguishable at this slot")
    recognition = EquipRecognition(0, expected_round=expected_round)
    result = drag_equipment(tasker, reader.slots, item_slot, name, (1188, 650), recognition,
                            entry="Equip.Destroy")
    try:
        rack = read_equipment_slots(tasker, reader, [item_slot])
        slot = rack["slots"][0]
        after = result["after"]
        expected = (slot["status"] == "empty" if replacement_id is None
                    else slot["status"] == "occupied" and replacement_id in slot["candidateIds"])
        if not expected or (rack["round"], rack["funds"]) != (after["round"], after["funds"]):
            raise RuntimeError("The rack slot did not match the expected outcome")
    except RuntimeError as error:
        raise RuntimeError(f"Item release was attempted; destruction is unverified: {error}") from error
    return {**result, "destroyed": True, "slotAfter": slot}


def validate_equipment(calibration: dict, item_slot: int, name: str, expected_gain: int | None) -> None:
    sources = calibration["equipment"]["centersByView"]["shopExpanded"]
    if type(item_slot) is not int or not 0 <= item_slot < len(sources) or not clean(name):
        raise ValueError("A valid zero-based equipment slot and displayed item name are required")
    if expected_gain is not None and (type(expected_gain) is not int or expected_gain < 0):
        raise ValueError("Expected funds gain must be a nonnegative integer")


def drag_equipment(tasker: Tasker, calibration: dict, item_slot: int, name: str, target: tuple[int, int],
                   recognition: EquipRecognition, *, entry: str, tracked: bool = False,
                   opens_choice: bool = False) -> dict:
    validate_equipment(calibration, item_slot, name, recognition.expected_gain)
    name = clean(name)
    equipment = calibration["equipment"]
    sources = equipment["centersByView"]["shopExpanded"]
    sx, sy = sources[item_slot]
    dx, dy = equipment["liftOffset"]
    tx, ty = target
    x, y, width, height = equipment["targetHighlightROI"]
    highlight_roi = [tx + x, ty + y, width, height]
    override = {
        "Equip.Pick": {"target": [sx, sy]},
        "Equip.Lift": {"target": [sx + dx, sy + dy]},
        "Equip.Hover": {"target": [tx, ty]},
        "Equip.ItemName": {"expected": "^" + re.escape(name) + "$"},
        "Equip.TargetHighlight": {"roi": highlight_roi},
        "Board.CancelMove": {"target": [35, 690]},
    }
    terminal = "Equip.DialogReleased" if opens_choice else "Equip.Returned"
    if tracked:
        override["Equip.Hover"]["next"] = ["Equip.DropTracked"]
        override["Equip.DropTracked"] = {"next": [terminal]}
    elif entry == "Equip.Board":
        recognition.highlight_roi = highlight_roi
        override["Equip.Hover"] = {"target": [tx, ty], "all_of": [
            "Equip.ItemName", "Equip.DragHint",
            {"recognition": "Custom", "custom_recognition": "ArkEquip", "custom_recognition_param": "boardHover"},
        ]}
    elif entry == "Equip.Destroy":
        override["Equip.Lift"]["next"] = ["Equip.DestroyHover"]
    if opens_choice and not tracked:
        override["Equip.Drop"] = {"next": [terminal]}
    with task_recognition(tasker, "ArkEquip", recognition):
        job = tasker.post_task(entry, override).wait()
        detail = job.get()
        nodes = [node.name for node in detail.nodes] if detail else []
    if not job.succeeded or not nodes or nodes[-1] != terminal or (not opens_choice and recognition.after is None):
        if {"Equip.Drop", "Equip.DropTracked", "Equip.DestroyDrop"}.intersection(nodes):
            raise RuntimeError("Equipment was released, but its expected postcondition was not verified")
        raise RuntimeError("Equip did not reach a verified item and highlighted target")
    return {
        "itemSlot": item_slot, "name": name,
        "before": recognition.before, "after": recognition.after,
        "fundsGainVerified": recognition.expected_gain is not None or recognition.gain_range is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-slot", required=True, type=int)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--hand-slot", type=int)
    target.add_argument("--tile", nargs=2, type=int, metavar=("ROW", "COL"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--operator", help="Exact displayed operator name for --tile")
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--expected-gain", type=int, help="Require this change in funds after equipping")
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    args = parser.parse_args()
    if args.tile is not None and not args.operator:
        parser.error("--tile requires --operator")
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        controller = connect(args.adb, args.address, input_methods=MaaAdbInputMethodEnum.Maatouch)
        tasker = load_ocr(controller, args.ocr_resource)
        slots = read_json(ROOT / "calib/slots.json")
        if args.tile is None:
            result = equip_to_hand(tasker, slots, args.item_slot, args.hand_slot, args.name,
                                   expected_gain=args.expected_gain, reader=Reader(args.catalog, slots))
        else:
            result = equip_to_board(tasker, Reader(args.catalog, slots), read_board(args.board),
                                    args.item_slot, args.name, *args.tile, args.operator,
                                    expected_gain=args.expected_gain)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
