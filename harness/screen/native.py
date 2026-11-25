"""MaaFramework recognition nodes evaluated on one captured frame."""

import json
import logging
import re
from pathlib import Path
from statistics import median

from maa.context import Context
from maa.controller import CustomController
from maa.define import AndRecognitionResult, CustomRecognitionResult, OCRResult, OrRecognitionResult
from numpy import ndarray


BOARD_PIPELINE = Path(__file__).resolve().parents[2] / "resource" / "pipeline" / "BOARD.json"
LOG = logging.getLogger(__name__)
STAGE_MARKERS = tuple((f"act3autochess_{node.removeprefix('Screen.Stage').lower()}", node)
                      for node in json.loads(BOARD_PIPELINE.read_text(encoding="utf-8"))
                      if re.fullmatch(r"Screen\.StageM[0-9]{2}", node))
MIN_NAME_SCORE = 0.70
MIN_NUMBER_SCORE = 0.60
CHECK_NUMBER_BELOW = 0.75
SCENES = {
    "Screen.BondPanel": "bondDetail",
    "Screen.EnemyPanel": "enemyPreview",
    "Screen.Drag": "drag",
    "Bounty.Screen": "bountySelection",
    "Supply.Screen": "equipmentSelection",
    "Tactic.Screen": "tacticSelection",
    "Screen.CommandDetails": "commandDetails",
    "Screen.ReadyIcon": "preparation",
    "Battle.Settlement": "settlement",
    "Battle.Active": "battle",
    "Battle.LeaderActive": "battle",
}


def recognition_data(detail):
    if detail is not None and detail.hit:
        result = detail.best_result
        if isinstance(result, CustomRecognitionResult) and isinstance(result.detail, dict):
            yield result.detail
        elif isinstance(result, (AndRecognitionResult, OrRecognitionResult)):
            for child in result.sub_results:
                yield from recognition_data(child)


def completed_data(nodes, *names):
    for node in reversed(nodes):
        if node.completed and node.name in names:
            for data in recognition_data(node.recognition):
                return data
    return None


class OfflineController(CustomController):
    def __init__(self, image: ndarray):
        super().__init__()
        self.image = image

    def connect(self) -> bool:
        return True

    def request_uuid(self) -> str:
        return "arkharness-offline"

    def screencap(self) -> ndarray:
        return self.image


class Frame:
    def __init__(self, context: Context, image: ndarray):
        self.context = context
        self.image = image
        self.results = {}

    def node(self, name: str, **override):
        if override:
            return self.context.run_recognition(name, self.image, {name: override})
        if name not in self.results:
            self.results[name] = self.context.run_recognition(name, self.image)
        return self.results[name]

    def hit(self, name: str, **override) -> bool:
        result = self.node(name, **override)
        return result is not None and result.hit

    def rows(self, name: str, *, fallback_unfiltered: bool = False, **override) -> list[OCRResult]:
        result = self.node(name, **override)
        if result is None:
            return []
        rows = result.filtered_results
        if not rows and fallback_unfiltered:
            rows = result.all_results
        return [row for row in rows if isinstance(row, OCRResult)]

    def text(self, name: str, *, min_score: float = 0.0,
             confirm_below: float = 0.0, fallback_unfiltered: bool = False, **override) -> str:
        rows = self.rows(name, fallback_unfiltered=True, **override) if fallback_unfiltered else self.rows(name, **override)
        if not rows or rows[0].score < min_score:
            return ""
        row = rows[0]
        text = row.text.strip()
        if row.score < confirm_below:
            x, y, width, height = row.box
            image_height, image_width = self.image.shape[:2]
            left, top = max(0, x - 2), max(0, y - 2)
            right, bottom = min(image_width, x + width + 2), min(image_height, y + height + 2)
            checked = self.rows(name, **{**override, "roi": [left, top, right - left, bottom - top]})
            if not checked or checked[0].score < min_score or checked[0].text.strip() != text:
                return ""
        return text

    def scene(self) -> str | None:
        return SCENES.get(self.first_hit("Screen.Scene"))

    def first_hit(self, group: str, *, overrides: dict | None = None) -> str | None:
        result = self.context.run_recognition(group, self.image, overrides) if overrides else self.node(group)
        if result is None or not isinstance(result.best_result, OrRecognitionResult):
            return None
        for child in result.best_result.sub_results:
            if not overrides:
                self.results.setdefault(child.name, child)
            if child.hit:
                return child.name
        return None

    def stage_id(self) -> str | None:
        matches = [key for key, node in STAGE_MARKERS if self.hit(node)]
        return matches[0] if len(matches) == 1 else None

    def bond_boxes(self, *, leader: bool = False) -> list[list[int]]:
        # Grey rings may miss the template, and clipped names may be off-centre.
        # Both lie on the same 80-pixel grid, whose start can shift with the screen layout.
        names = [row.box[0] + row.box[2] // 2 for row in self.rows("Screen.BondNameRow")
                 if row.score >= MIN_NAME_SCORE and 2 <= len(re.sub(r"\s+", "", row.text)) <= 6]
        ring = self.node("Leader.BondRing" if leader else "Screen.BondRing")
        rings = [item.box[0] + 27 for item in ring.filtered_results] if ring and ring.hit else []
        anchors = rings or names
        if not anchors:
            return []
        pitch = 80
        first = anchors[0]
        origin = round(median(x - pitch * round((x - first) / pitch) for x in anchors))
        centers = {origin + pitch * round((x - origin) / pitch) for x in rings + names}
        return [[center - 27, 64, 16, 45] for center in sorted(centers)]

    def visible_bonds(self, *, leader: bool = False) -> list[tuple[str, str | None, list[int]]]:
        prefix = "Leader" if leader else "Screen"
        bonds = []
        for box in self.bond_boxes(leader=leader):
            name = self.text(f"{prefix}.BondName", roi=box, min_score=MIN_NAME_SCORE)
            node = f"{prefix}.BondLayers"
            rows = self.rows(node, roi=box)
            layers = rows[0].text.strip() if rows else ""
            if not rows or rows[0].score < 0.90:
                checked = self.text(node, roi=[box[0] + 12, 100, 30, 15], roi_offset=[0, 0, 0, 0],
                                    min_score=MIN_NUMBER_SCORE)
                if checked != layers or (rows and rows[0].score < MIN_NUMBER_SCORE):
                    LOG.warning("Unverified bond layers: name=%r x=%d first=%r second=%r",
                                name, box[0] + 27, layers, checked)
                    layers = None
            bonds.append((name, layers, box))
        return bonds
