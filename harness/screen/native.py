"""MaaFramework recognition nodes evaluated on one captured frame."""

import json
import re
from pathlib import Path

from maa.context import Context
from maa.controller import CustomController
from maa.define import AndRecognitionResult, CustomRecognitionResult, OCRResult, OrRecognitionResult
from numpy import ndarray


BOARD_PIPELINE = Path(__file__).resolve().parents[2] / "resource" / "pipeline" / "BOARD.json"
STAGE_MARKERS = tuple((f"act3autochess_{node.removeprefix('Screen.Stage').lower()}", node)
                      for node in json.loads(BOARD_PIPELINE.read_text(encoding="utf-8"))
                      if re.fullmatch(r"Screen\.StageM[0-9]{2}", node))


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

    def rows(self, name: str, **override) -> list[OCRResult]:
        result = self.node(name, **override)
        return [row for row in result.filtered_results if isinstance(row, OCRResult)] if result else []

    def text(self, name: str, **override) -> str:
        rows = self.rows(name, **override)
        return rows[0].text.strip() if rows else ""

    def scene(self) -> str | None:
        if self.hit("Screen.BondPanel"):
            return "bondDetail"
        if self.hit("Screen.EnemyPanel"):
            return "enemyPreview"
        if self.hit("Screen.Drag"):
            return "drag"
        if self.hit("Bounty.Screen"):
            return "bountySelection"
        if self.hit("Supply.Screen"):
            return "equipmentSelection"
        if self.hit("Tactic.Screen"):
            return "tacticSelection"
        if self.hit("Screen.ReadyIcon"):
            if self.hit("Command.Panel"):
                return "commandDetails"
            return "preparation"
        if self.hit("Battle.Settlement"):
            return "settlement"
        if self.hit("Battle.Active") or self.hit("Battle.LeaderActive"):
            return "battle"
        return None

    def stage_id(self) -> str | None:
        matches = [key for key, node in STAGE_MARKERS if self.hit(node)]
        return matches[0] if len(matches) == 1 else None

    def bond_boxes(self) -> list[list[int]]:
        ring = self.node("Screen.BondRing")
        return [list(item.box) for item in ring.filtered_results] if ring and ring.hit else []

    def visible_bonds(self) -> list[tuple[str, str, list[int]]]:
        return [(self.text("Screen.BondName", roi=box), self.text("Screen.BondLayers", roi=box), box)
                for box in self.bond_boxes()]
