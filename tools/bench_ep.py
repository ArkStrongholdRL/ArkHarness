"""OCR cost per inference provider (read-only)."""
import sys, json, statistics
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.resource import Resource
from maa.tasker import Tasker
from maa.pipeline import JOCR
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, recognize
from harness.screen.flow import observe_screen
Tasker.set_log_dir(SAMPLES / "_logs")
ep = sys.argv[1]
c = connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH)
res = Resource()
ok = {"cpu": res.use_cpu, "auto": res.use_auto_ep, "dml-auto": res.use_directml}.get(ep, lambda: res.use_directml(int(ep.split(":")[1])))()
print("provider", ep, "set", ok, flush=True)
assert res.post_bundle(OCR_RESOURCE).wait().succeeded
t = Tasker(); t.bind(res, c)
from harness.screen.capture import capture
img = capture(c)
recognize(t, img, JOCR())
def med(fn, n):
    xs = []
    for _ in range(n):
        s = perf_counter(); fn(); xs.append((perf_counter() - s) * 1000)
    return round(statistics.median(xs), 1)
print(json.dumps({"ep": ep, "ocrFull": med(lambda: recognize(t, img, JOCR()), 8),
                  "ocrRoi": med(lambda: recognize(t, img, JOCR(roi=(792, 21, 40, 27))), 12),
                  "observe": med(lambda: observe_screen(t), 5)}), flush=True)
