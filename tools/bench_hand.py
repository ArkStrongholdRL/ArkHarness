import sys, json
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, Reader, read_json
from harness.screen import hand
from harness.screen.shop import set_shop_open
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
reader = Reader(CATALOG, read_json(ROOT / "calib/slots.json")); reader.stage_hint = "act3autochess_m04"
T0 = [0]; log = []
orig = hand.HandRecognition.analyze
def analyze(self, context, argv):
    s = perf_counter(); r = orig(self, context, argv); e = perf_counter()
    log.append((round((s - T0[0]) * 1000), round((e - s) * 1000), json.loads(argv.custom_recognition_param), r is not None)); return r
hand.HandRecognition.analyze = analyze
v = set_shop_open(tasker, False)
try:
    for idx in ([int(a) for a in sys.argv[1:]] or [0, 9]):
        log.clear(); T0[0] = perf_counter(); s = perf_counter()
        r = hand.read_hand(tasker, reader, [idx])
        print(f"slot {idx}: total {round((perf_counter()-s)*1000)} ms status {r['slots'][0]['status']}")
        for t, d, k, ok in log: print(f"   t+{t:5d} ms  analyze {d:4d} ms  {k:6s} {'hit' if ok else '-'}")
finally:
    set_shop_open(tasker, v["before"]["expanded"])
