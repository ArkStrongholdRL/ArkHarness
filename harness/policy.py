"""P7 policy: CPU inference, greedy (argmax of the logits inside the mask), as GOAL-live section 2 requires.

The network class is the engine's own tools/rl/py/model.py (SPNet), imported by file path (the engine stays read only).
"""

import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(os.environ.get("ARK_ENGINE", "E:/ArkHarness-engine/sp-label"))
CKPT = Path(os.environ.get("ARK_P7", "E:/ArkHarness-data/P7-model-final.pt"))
DATA_ID = "act2autochess-1177430123fc"


def _model_module():
    spec = importlib.util.spec_from_file_location("p7_model", ENGINE / "tools/rl/py/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Policy:
    def __init__(self, ckpt: Path = CKPT, threads: int = 4):
        torch.set_num_threads(threads)
        self.net, ck = _model_module().load(str(ckpt), "cpu")
        self.checkpoint = ckpt
        self.net.eval()
        self.spec = self.net.spec
        if self.spec.get("dataId") != DATA_ID:
            raise ValueError(f"checkpoint dataId {self.spec.get('dataId')} != {DATA_ID}")
        self.dim, self.actions = self.spec["dim"], self.spec["actions"]

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """obs (B, 2570) float32, mask (B, 100) -> greedy actions (B,) int."""
        o = torch.from_numpy(np.ascontiguousarray(obs, np.float32).reshape(-1, self.dim))
        m = torch.from_numpy(np.ascontiguousarray(mask, np.uint8).reshape(-1, self.actions).astype(bool))
        logits, _ = self.net(o, m)
        return logits.argmax(1).numpy()

    def timing(self, n: int = 200) -> float:
        """Median ms of one single-row inference."""
        o = np.zeros((1, self.dim), np.float32)
        m = np.ones((1, self.actions), np.uint8)
        self.act(o, m)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.act(o, m)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))
