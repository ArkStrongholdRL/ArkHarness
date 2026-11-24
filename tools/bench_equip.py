import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from time import perf_counter
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, Reader, read_json
from harness.screen.equipment_read import read_equipment_slots
from harness.screen.shop import set_shop_open
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
reader = Reader(CATALOG, read_json(ROOT / "calib/slots.json"))
reader.stage_hint = "act3autochess_m04"
v = set_shop_open(tasker, False)
try:
    t = perf_counter(); r = read_equipment_slots(tasker, reader)
    print("OK", round((perf_counter() - t) * 1000), "ms", [s["status"] for s in r["slots"]])
except Exception as e:
    print("FAIL", e)
finally:
    set_shop_open(tasker, v["before"]["expanded"])
