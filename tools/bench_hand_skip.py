import sys, json
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, Reader, read_json
from harness.screen.hand import read_hand
from harness.screen.shop import set_shop_open
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
reader = Reader(CATALOG, read_json(ROOT / "calib/slots.json")); reader.stage_hint = "act3autochess_m04"
v = set_shop_open(tasker, False)
try:
    out = {}
    for name, kw in (("skip", {"skip_bare": True}), ("full", {})):
        s = perf_counter(); r = read_hand(tasker, reader, **kw); out[name] = r
        print(name, round((perf_counter() - s) * 1000), "ms", [(x["screenSlot"], x["status"], x.get("id")) for x in r["slots"] if x["status"] != "empty"])
    print("identical:", out["skip"]["slots"] == out["full"]["slots"], out["skip"]["remaining"], out["full"]["remaining"])
finally:
    set_shop_open(tasker, v["before"]["expanded"])
