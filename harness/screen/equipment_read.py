"""Inspect items in equipment slots through MaaFramework without using them."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.define import MaaAdbInputMethodEnum
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .native import Frame
from .read import CATALOG, ROOT, Reader, catalog_identity, integer, read_json
from .shop import set_shop_open


def hud_status(frame: Frame) -> tuple[int, int] | None:
    """Round and funds on the expanded preparation HUD. The rack does not depend on the shop level (whose heading OCR is unstable)."""
    if not frame.hit("ArkRestText") or len(frame.rows("Shop.RefreshText")) != 1:
        return None
    current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
    funds = integer(frame.text("Screen.Funds"))
    return None if current_round is None or funds is None else (current_round, funds)


class EquipmentReadRecognition(CustomRecognition):
    def __init__(self, reader: Reader):
        super().__init__()
        self.reader = reader
        self.status = None
        self.slot = None
        self.result = None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        hit = (0, 0, 1, 1)
        if kind == "name":
            self.result = {"screenSlot": self.slot, "status": "occupied",
                           **catalog_identity(self.reader, frame, "EquipmentRead.Name", "item")}
            return hit
        status = hud_status(frame)
        if status is None or self.status not in (None, status) or frame.hit("Equip.DragHint"):
            return None
        if kind == "before":
            self.status = status
            return hit
        if kind == "empty":
            self.result = {"screenSlot": self.slot, "status": "empty", "name": None, "id": None,
                           "baseId": None, "candidateIds": []}
        return hit if self.result is not None else None


def read_equipment_slots(tasker: Tasker, reader: Reader, indices: list[int] | None = None) -> dict:
    """Lift and cancel at the requested screen slots, then restore the original shop view."""
    rack = reader.slots["equipment"]
    centers = rack["centersByView"]["shopExpanded"]
    selected = list(range(len(centers))) if indices is None else indices
    if not selected or any(type(slot) is not int or not 0 <= slot < len(centers) for slot in selected):
        raise ValueError("Expected valid zero-based equipment slots")
    if len(set(selected)) != len(selected):
        raise ValueError("Equipment slots must be unique")

    recognition = EquipmentReadRecognition(reader)
    resource = tasker.resource
    if not resource.register_custom_recognition("ArkEquipmentRead", recognition):
        raise RuntimeError("Could not register equipment rack inspection")
    observations = []
    view = None
    try:
        view = set_shop_open(tasker, True)
        dx, dy = rack["liftOffset"]
        for slot in selected:
            recognition.slot, recognition.result = slot, None
            x, y = centers[slot]
            job = tasker.post_task("EquipmentRead.ScanSlot", {
                "EquipmentRead.Pick": {"target": [x, y]},
                "EquipmentRead.Lift": {"target": [x + dx, y + dy]},
            }).wait()
            detail = job.get()
            last = detail.nodes[-1].name if detail and detail.nodes else ""
            if not job.succeeded or last != "EquipmentRead.Restored" or recognition.result is None:
                raise RuntimeError(f"Equipment slot {slot} or return from the drag could not be verified")
            observations.append(recognition.result)
    finally:
        resource.unregister_custom_recognition("ArkEquipmentRead")
        if view is not None and view["clicked"]:
            set_shop_open(tasker, False)
    current_round, funds = recognition.status
    if (current_round, funds) != (view["before"]["round"], view["before"]["funds"]):
        raise RuntimeError("The preparation round or funds changed during rack inspection")
    return {"round": current_round, "funds": funds, "slots": observations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item-slot", type=int, action="append", dest="indices")
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
        controller = connect(args.adb, args.address, input_methods=MaaAdbInputMethodEnum.Maatouch)
        print(json.dumps(read_equipment_slots(load_ocr(controller, args.ocr_resource), reader, args.indices),
                         ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
