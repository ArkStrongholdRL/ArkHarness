"""Install the local MFW assets or rebuild image templates from key samples."""

import argparse
from pathlib import Path
from shutil import copyfile

from PIL import Image

from .capture import OCR_RESOURCE, SAMPLES


OCR_SOURCE = Path("E:/Program/MaaEnd/assets/resource/model/ocr")
CROPS = {
    "ready-checkbox": ("主准备界面/03.png", (1133, 100, 28, 29)),
    "bond-members-icon": ("盟约详情面板/01.png", (334, 203, 21, 23)),
    "enemy-header-icon": ("敌人预告面板/01.png", (769, 79, 25, 30)),
    "bond-ring-left": ("主准备界面/03.png", (533, 66, 16, 45)),
    "drag-sell-icon": ("拖拽视角/01.png", (1167, 552, 42, 43)),
    "stage-m04-left-crates": ("主准备界面-m04/01.png", (435, 131, 130, 74)),
    "stage-m04-right-crate": ("主准备界面-m04/01.png", (775, 131, 82, 73)),
    "stage-m06-expanded": ("主准备界面/01.png", (502, 192, 140, 135)),
    "stage-m06-normal": ("主准备界面/02.png", (509, 250, 145, 145)),
}


def install_assets(resource: Path, ocr_source: Path, rebuild_images: bool = False) -> None:
    model = resource / "model" / "ocr"
    names = ("det.onnx", "rec.onnx", "keys.txt")
    sources = [ocr_source / name for name in names]
    missing_images = [name for name in CROPS if not (resource / "image" / f"{name}.png").is_file()]
    if rebuild_images:
        sources.extend(SAMPLES / sample for sample, _ in CROPS.values())
    else:
        sources.extend(SAMPLES / "_templates" / f"{name}.png" for name in missing_images)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    model.mkdir(parents=True, exist_ok=True)
    for name in names:
        copyfile(ocr_source / name, model / name)
    images = resource / "image"
    images.mkdir(parents=True, exist_ok=True)
    for name, (sample, (x, y, width, height)) in CROPS.items():
        output = images / f"{name}.png"
        if rebuild_images:
            with Image.open(SAMPLES / sample) as png:
                if png.size != (1280, 720):
                    raise ValueError(f"Expected native 1280×720 sample: {sample}")
                png.convert("RGB").crop((x, y, x + width, y + height)).save(output)
        elif name in missing_images:
            copyfile(SAMPLES / "_templates" / output.name, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--install", action="store_true", help="Copy the retained small templates and local OCR models")
    mode.add_argument("--build", action="store_true", help="Rebuild the image crops from the source screenshots")
    parser.add_argument("--resource", type=Path, default=OCR_RESOURCE)
    parser.add_argument("--ocr-source", type=Path, default=OCR_SOURCE)
    args = parser.parse_args()
    try:
        install_assets(args.resource, args.ocr_source, rebuild_images=args.build)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
