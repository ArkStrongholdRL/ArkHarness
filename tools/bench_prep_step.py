"""Wall time of the read steps in one live.prep() iteration (no game action)."""
import sys
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, Reader, read_json, read_screen
from harness.screen.flow import observe_screen, execute
from harness.screen.shop import set_shop_open, quote_slot
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
slots = read_json(ROOT / "calib/slots.json")
reader = Reader(CATALOG, slots); reader.stage_hint = "act3autochess_m04"
def t(label, fn, n=4):
    xs = []
    for _ in range(n):
        s = perf_counter(); r = fn(); xs.append((perf_counter() - s) * 1000)
    xs.sort(); print(f"{label:34s} median {xs[len(xs)//2]:7.0f} ms  min {xs[0]:7.0f}", flush=True); return r
t("observe_screen (run loop)", lambda: observe_screen(tasker))
t("set_shop_open(True), already open", lambda: set_shop_open(tasker, True))
t("read_screen (full ScreenState)", lambda: read_screen(tasker, reader))
t("flow.execute('observe') probe", lambda: execute(tasker, slots, "observe", CATALOG, None, 6000))
t("quote_slot(1) (no click)", lambda: quote_slot(tasker, reader, 1))
