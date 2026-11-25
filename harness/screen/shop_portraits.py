"""Collect local shop templates from calibrated, fully lit card regions."""

import re
from base64 import b64decode, b64encode
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from maa.define import OrRecognitionResult
from PIL import Image


def match_portrait(frame, roi: list[int], cards: list[str]) -> str | None:
    if not cards:
        return None
    left, top, width, height = roi
    search = [left - 5, top - 5, width + 10, height + 10]
    nodes = {f"Shop.Portrait.{key}": {"recognition": "TemplateMatch", "template": f"portrait/{key}.png",
                                     "roi": search, "threshold": 0.94} for key in cards}
    entry = "Shop.CardPortrait"
    nodes[entry] = {"recognition": "Or", "any_of": list(nodes)}
    result = frame.context.run_recognition(entry, frame.image, nodes)
    if result is not None and isinstance(result.best_result, OrRecognitionResult):
        return next((child.name.removeprefix("Shop.Portrait.") for child in result.best_result.sub_results
                     if child.hit), None)
    return None


def _crop(image: np.ndarray, roi: list[int] | None) -> np.ndarray | None:
    if not isinstance(roi, list) or len(roi) != 4 or any(type(value) is not int for value in roi):
        return None
    x, y, width, height = roi
    rows, cols = image.shape[:2]
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > cols or y + height > rows:
        return None
    return image[y:y + height, x:x + width]


def encode_portrait(image: np.ndarray, roi: list[int]) -> str | None:
    art = _crop(image, roi)
    if art is None:
        return None
    with BytesIO() as buffer:
        Image.fromarray(art[:, :, ::-1]).save(buffer, format="PNG")
        return b64encode(buffer.getvalue()).decode("ascii")


def save_portraits(folder: Path, portraits: dict[str, str]) -> None:
    """Publish each complete template once. Collection failures leave the read result usable."""
    for card_id, png in portraits.items():
        if re.fullmatch(r"[a-zA-Z0-9_]+", card_id) is None:
            continue
        destination = folder / f"{card_id}.png"
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=folder) as directory:
                pending = Path(directory) / destination.name
                pending.write_bytes(b64decode(png))
                destination.hardlink_to(pending)
        except OSError:
            continue
