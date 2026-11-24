"""Per-node recognition cost inside one observe run (read-only)."""
import sys, collections
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.controller import AdbController
from maa.define import MaaAdbInputMethodEnum as I, MaaAdbScreencapMethodEnum as S
from maa.tasker import Tasker
from harness.screen.capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, load_ocr
from harness.screen import native
from harness.screen.flow import observe_screen

Tasker.set_log_dir(SAMPLES / "_logs")
c = AdbController(ADB_PATH, ADB_ADDRESS, screencap_methods=S.EmulatorExtras, input_methods=I.EmulatorExtras,
                  config={"extras": {"mumu": {"enable": True, "index": 0, "path": "D:/Program Files/Netease/MuMu"}}})
c.set_screenshot_use_raw_size(True); c.post_connection().wait()
tasker = load_ocr(c, OCR_RESOURCE)
cost = collections.defaultdict(list)
orig = native.Frame.node
def node(self, name, **o):
    cached = not o and name in self.results
    t = perf_counter(); r = orig(self, name, **o)
    if not cached: cost[name + ("*" if o else "")].append((perf_counter() - t) * 1000)
    return r
native.Frame.node = node
observe_screen(tasker)
cost.clear()
t = perf_counter()
for _ in range(3): s = observe_screen(tasker)
total = (perf_counter() - t) / 3 * 1000
print("screen", s, "observe ms", round(total))
tot = 0
for k, v in sorted(cost.items(), key=lambda kv: -sum(kv[1])):
    print(f"{k:32s} calls/obs {len(v)/3:.1f}  avg {sum(v)/len(v):6.1f} ms  per-obs {sum(v)/3:6.1f}")
    tot += sum(v) / 3
print("recognition total per observe", round(tot), "ms; remainder (framework/capture/etc.)", round(total - tot))
