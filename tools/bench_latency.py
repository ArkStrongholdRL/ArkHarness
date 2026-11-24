"""Phase 0 latency benchmark (read-only: screenshots, OCR and the observe task; no game input).

  .venv\\Scripts\\python -I tools/bench_latency.py [--frames 30] [--variant adb|mumu|both]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maa.controller import AdbController  # noqa: E402
from maa.define import MaaAdbInputMethodEnum as I, MaaAdbScreencapMethodEnum as S  # noqa: E402
from maa.pipeline import JOCR  # noqa: E402
from maa.tasker import Tasker  # noqa: E402

from harness.screen.capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, load_ocr, recognize  # noqa: E402

MUMU = {"extras": {"mumu": {"enable": True, "index": 0, "path": "D:/Program Files/Netease/MuMu"}}}
VARIANTS = {
    "adb": dict(screencap_methods=S.RawWithGzip, input_methods=I.Maatouch, config=None),
    "adb-default": dict(screencap_methods=S.Default, input_methods=I.Maatouch, config=None),
    "mumu": dict(screencap_methods=S.EmulatorExtras, input_methods=I.EmulatorExtras, config=MUMU),
}


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "median": round(statistics.median(xs), 1), "min": round(xs[0], 1),
            "p90": round(xs[int(len(xs) * 0.9) - 1], 1), "max": round(xs[-1], 1)}


def timed(fn, n):
    out = []
    for _ in range(n):
        t = perf_counter()
        fn()
        out.append((perf_counter() - t) * 1000)
    return out


def bench(name, kw, frames, adb, address):
    kw = {**kw, "config": kw["config"] or {}}
    c = AdbController(adb, address, **kw)
    c.set_screenshot_use_raw_size(True)
    t = perf_counter()
    if not c.post_connection().wait().succeeded:
        return {"variant": name, "error": "connect failed"}
    res = {"variant": name, "connectMs": round((perf_counter() - t) * 1000)}

    def shot():
        job = c.post_screencap().wait()
        assert job.succeeded
        return job.get()

    img = shot()
    res["shape"] = list(img.shape)
    res["screencapMs"] = stats(timed(shot, frames))
    tasker = load_ocr(c, OCR_RESOURCE)
    recognize(tasker, img, JOCR())  # warm up
    res["ocrFullMs"] = stats(timed(lambda: recognize(tasker, img, JOCR()), 8))
    res["ocrRoiMs"] = stats(timed(lambda: recognize(tasker, img, JOCR(roi=(792, 21, 40, 27))), 12))
    # the observe navigation task (what harness.live runs before every action)
    from harness.screen.flow import observe_screen
    obs = []
    for _ in range(6):
        t = perf_counter()
        screen = observe_screen(tasker)
        obs.append((perf_counter() - t) * 1000)
    res["observeScreenMs"], res["screen"] = stats(obs), screen
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--variant", default="both")
    ap.add_argument("--adb", type=Path, default=ADB_PATH)
    ap.add_argument("--address", default=ADB_ADDRESS)
    a = ap.parse_args()
    Tasker.set_log_dir(SAMPLES / "_logs")
    Tasker.set_save_draw(False)
    names = ["adb", "mumu"] if a.variant == "both" else a.variant.split(",")
    for name in names:
        try:
            print(json.dumps(bench(name, VARIANTS[name], a.frames, a.adb, a.address), ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001 — report and continue to the next variant
            print(json.dumps({"variant": name, "error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
