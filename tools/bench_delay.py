"""Does observe's framework overhead come from MFW default delays? (read-only)"""
import sys
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.controller import AdbController
from maa.define import MaaAdbInputMethodEnum as I, MaaAdbScreencapMethodEnum as S
from maa.tasker import Tasker
from harness.screen.capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, load_ocr
from harness.screen import flow
Tasker.set_log_dir(SAMPLES / "_logs")
c = AdbController(ADB_PATH, ADB_ADDRESS, screencap_methods=S.EmulatorExtras, input_methods=I.EmulatorExtras,
                  config={"extras": {"mumu": {"enable": True, "index": 0, "path": "D:/Program Files/Netease/MuMu"}}})
c.set_screenshot_use_raw_size(True); c.post_connection().wait()
tasker = load_ocr(c, OCR_RESOURCE)
Z = {"pre_delay": 0, "post_delay": 0, "rate_limit": 0}
orig = flow.Tasker.post_task
def run(label, override):
    def post(self, entry, ov=None):
        return orig(self, entry, {**(ov or {}), **override})
    flow.Tasker.post_task = post
    flow.observe_screen(tasker)
    ts = []
    for _ in range(6):
        t = perf_counter(); flow.observe_screen(tasker); ts.append((perf_counter() - t) * 1000)
    ts.sort(); print(f"{label:34s} median {ts[3]:7.1f} ms  min {ts[0]:7.1f}")
    flow.Tasker.post_task = orig
run("default", {})
run("observe+snapshot delays 0", {"ArkObserve": Z, "ArkSnapshot": Z})
run("rate_limit only 0", {"ArkObserve": {"rate_limit": 0}, "ArkSnapshot": {"rate_limit": 0}})
