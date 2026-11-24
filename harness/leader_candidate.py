"""Offline evaluation of generated solo leader layouts against the live planner."""

import sys
from pathlib import Path

import numpy as np
import torch

from .layout import as_tile, plan_layout
from .track import Track


class LeaderCandidate:
    def __init__(self, directory: Path):
        # The supplied inference scripts use a sibling import when loaded as standalone modules.
        sys.path.insert(0, str(directory.resolve()))
        try:
            from layout_gen_infer import LayoutGen
            from layout_scorer_infer import LayoutScorer
        finally:
            sys.path.pop(0)

        self.gen = LayoutGen(torch.load(directory / "layout_gen_v1.pt", map_location="cpu", weights_only=True))
        self.scorer = LayoutScorer(torch.load(directory / "layout_scorer_v2.pt", map_location="cpu", weights_only=True))

    def choose(self, track: Track, state: dict, calibrated: set[tuple[int, int]], *,
               count: int = 64, seed: int = 0) -> dict:
        """Return the highest-scoring candidate whose whole layout can be executed."""
        if count < 1:
            raise ValueError("Leader candidate count must be positive")
        if state.get("side") != "L" or not state.get("bossTemplate") or state.get("bonds") is None:
            raise ValueError("Leader candidate needs the solo left field, boss template and complete bond input")
        if not state["pieces"]:
            raise ValueError("Leader candidate needs a nonempty lineup")
        if any(not (9 <= row <= 12 and 3 <= col <= 9) for row, col in map(as_tile, state["map"])):
            raise ValueError("Leader candidate requires canonical boss field coordinates")

        candidates = [self.gen.layout(state)] + self.gen.sample(state, k=count - 1, seed=seed)
        scores = self.scorer.score(state, candidates)
        for index in np.argsort(-scores, kind="stable"):
            if not np.isfinite(scores[index]):
                continue
            candidate = candidates[index]
            try:
                plan_layout(track, state, candidate, calibrated)
            except ValueError:
                continue
            return {**candidate, "index": int(index), "scores": scores.tolist()}
        raise ValueError("No generated leader candidate passed layout preflight")
