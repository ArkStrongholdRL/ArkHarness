"""Capture and index a 1280×720 sample from MuMu without sending game input."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from maa.controller import AdbController, Controller
from maa.define import MaaAdbInputMethodEnum, MaaAdbScreencapMethodEnum, OCRResult
from maa.pipeline import JOCR
from maa.resource import Resource
from maa.tasker import Tasker
from numpy import ndarray
from PIL import Image


ADB_PATH = Path(os.environ.get("ARK_ADB", "D:/ADB/platform-tools/adb.exe"))
ADB_ADDRESS = "127.0.0.1:16384"
SAMPLES = Path(__file__).resolve().parents[2] / "samples"
OCR_RESOURCE = SAMPLES.parent / "resource"
MUMU_PATH = Path(os.environ.get("ARK_MUMU", "D:/Program Files/Netease/MuMu"))


def connect(adb_path: Path, address: str, *,
            input_methods: int = MaaAdbInputMethodEnum.AdbShell,
            mumu: Path | None = None, mumu_index: int = 0) -> AdbController:
    """ADB controller; with `mumu` set, screenshots go through MuMu's extras (~8 ms instead of ~225 ms).

    Input keeps `input_methods` (the touch path verified on device); only the screenshot channel changes.
    """
    kwargs = {}
    if mumu is not None:
        kwargs = {"screencap_methods": MaaAdbScreencapMethodEnum.EmulatorExtras,
                  "config": {"extras": {"mumu": {"enable": True, "index": mumu_index, "path": str(mumu)}}}}
    controller = AdbController(adb_path, address, input_methods=input_methods, **kwargs)
    if not controller.set_screenshot_use_raw_size(True):
        raise RuntimeError("Could not enable raw screenshot size")
    if not controller.post_connection().wait().succeeded:
        raise RuntimeError(f"Could not connect to {address}")
    return controller


def capture(controller: AdbController) -> ndarray:
    job = controller.post_screencap().wait()
    if not job.succeeded:
        raise RuntimeError("MaaFramework screenshot failed")
    image = job.get()
    if image.shape != (720, 1280, 3):
        raise RuntimeError(f"Expected a 1280×720 BGR screenshot, got shape {image.shape}")
    return image


def load_ocr(controller: Controller, resource_path: Path) -> Tasker:
    model_dir = resource_path / "model" / "ocr"
    for name in ("det.onnx", "rec.onnx", "keys.txt"):
        if not (model_dir / name).is_file():
            raise RuntimeError(f"OCR file missing: {model_dir / name}")
    resource = Resource()
    # Almost every recognition is a small ROI: the auto/DirectML provider costs ~60 ms per call regardless of size, CPU ~15 ms
    # (full-screen OCR is the other way round: 215 ms vs 725 ms). ARK_OCR_EP=auto restores the framework default.
    if os.environ.get("ARK_OCR_EP", "cpu") == "cpu" and not resource.use_cpu():
        raise RuntimeError("Could not select the CPU inference provider")
    if not resource.post_bundle(resource_path).wait().succeeded:
        raise RuntimeError(f"Could not load OCR resources from {resource_path}")
    tasker = Tasker()
    if not tasker.bind(resource, controller):
        raise RuntimeError("Could not bind MaaFramework resources and controller")
    return tasker


def recognize(tasker: Tasker, image: ndarray, params: JOCR) -> list[OCRResult]:
    job = tasker.post_recognition("OCR", params, image).wait()
    detail = job.get()
    if detail is None or not detail.nodes or detail.nodes[0].recognition is None:
        raise RuntimeError("MaaFramework did not return OCR results")
    return detail.nodes[0].recognition.filtered_results


def save_sample(root: Path, screen: str, round_number: int | None, note: str, image: ndarray) -> dict:
    folder = root / screen
    folder.mkdir(parents=True, exist_ok=True)
    existing = [int(path.stem) for path in folder.glob("*.png") if path.stem.isdecimal()]
    number = max(existing, default=0) + 1
    while True:
        path = folder / f"{number:02d}.png"
        try:
            output = path.open("xb")
        except FileExistsError:
            number += 1
            continue
        break
    try:
        with output:
            Image.fromarray(image[:, :, ::-1]).save(output, format="PNG")
        record = {
            "file": path.relative_to(root).as_posix(),
            "screen": screen,
            "round": round_number,
            "note": note,
        }
        with (root / "index.jsonl").open("a", encoding="utf-8") as index:
            index.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True, help="Sample directory name")
    parser.add_argument("--round", type=int, help="Current round, if visible")
    parser.add_argument("--note", default="", help="Sample annotation")
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    parser.add_argument("--ocr", action="store_true", help="Print OCR results and runtime")
    args = parser.parse_args()
    if not args.screen or args.screen.endswith((".", " ")) or any(
        char in '<>:"/\\|?*' or ord(char) < 32 for char in args.screen
    ):
        parser.error("--screen must be a single directory name")
    log_dir = SAMPLES / "_logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        Tasker.set_log_dir(log_dir)
        controller = connect(args.adb, args.address)
        start = perf_counter()
        image = capture(controller)
        capture_ms = (perf_counter() - start) * 1000
        record = save_sample(SAMPLES, args.screen, args.round, args.note, image)
        print(json.dumps({**record, "captureMs": round(capture_ms, 1)}, ensure_ascii=False))
        if args.ocr:
            tasker = load_ocr(controller, args.ocr_resource)
            start = perf_counter()
            results = recognize(tasker, image, JOCR())
            ocr_ms = (perf_counter() - start) * 1000
            print(json.dumps({
                "ocrMs": round(ocr_ms, 1),
                "text": [{"box": row.box, "score": row.score, "text": row.text} for row in results],
            }, ensure_ascii=False))
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
