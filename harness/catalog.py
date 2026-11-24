"""The engine catalogue for the Python harness.

Ported from the engine (GPL, read only): server/match/gamedata.js (GameData accessors) on top of data/catalog.json,
which tools/export_catalog.mjs writes from the engine's own getData(). Order-sensitive derived lists (bondIds in
identifier order, bandIds) come from the export as the engine computed them.
"""

import json
import re
from copy import copy
from functools import cached_property, lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"
MODE_ID = "mode_single_20"
POOL_COPIES_DEFAULT = {1: 12, 2: 14, 3: 18, 4: 16, 5: 8, 6: 5}  # gamedata.js DEFAULTS.poolCopies


def _own(table, key):
    if not isinstance(table, dict) or not isinstance(key, str):
        return None
    v = table.get(key)
    return v if isinstance(v, dict) else None


class GameData:
    def __init__(self, path: Path = CATALOG):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `node tools/export_catalog.mjs` first")
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.raw = doc["raw"]
        self.derived = doc["derived"]
        self.default_version = doc["defaultVersion"]
        self.mode_id = doc["modeId"]
        self.mode = (self.raw.get("config") or {}).get("modes", {}).get(self.mode_id) or {}
        self.bond_ids = list(self.derived["bondIds"])  # gd.bondIds: identifier order
        self.mode_inactive_bonds = set(self.derived["modeInactiveBonds"])
        self.regulation_bond_draw = bool(self.derived["regulationBondDraw"])
        self.boss_round = self.derived["bossRound"]
        self.hidden_round = self.derived["hiddenRound"]
        self.last_round = self.derived["lastRound"]
        self.difficulty = self.derived["difficulty"]
        self._band_bonds: dict[str, tuple] = {}

    def for_mode(self, mode_id: str) -> "GameData":
        """Share the fixed policy catalog while selecting a different solo mode's rules."""
        if mode_id == self.mode_id:
            return self
        mode = self.raw["config"]["modes"].get(mode_id)
        if not mode or mode.get("type") != "SINGLE":
            raise ValueError(f"Unknown solo mode {mode_id}")
        result = copy(self)
        result.mode_id, result.mode = mode_id, mode
        result.mode_inactive_bonds = set(mode.get("inactiveBondIds") or [])
        result.boss_round = mode["bossRound"]
        result.hidden_round = mode["hiddenRound"]
        result.last_round = mode["lastRound"]
        result.difficulty = mode["difficulty"]
        return result

    # ---- ids (gamedata.js chess / item / bond / band / ...) ----
    def chess(self, i): return _own(self.raw.get("chess"), i)
    def item(self, i): return _own(self.raw.get("items"), i)
    def bond(self, i): return _own(self.raw.get("bonds"), i)
    def band(self, i): return _own(self.raw.get("bands"), i)
    def enemy(self, i): return _own(self.raw.get("enemies"), i)
    def boss(self, i): return _own(self.raw.get("bosses"), i)
    def token(self, i): return _own(self.raw.get("tokens"), i)
    def stage(self, i): return _own(self.raw.get("stages"), i)
    def garrison(self, i): return _own(self.raw.get("garrisons"), i)

    def base_id_of(self, i):
        c = self.chess(i)
        if not c:
            return re.sub(r"_b$", "_a", i) if isinstance(i, str) else None
        return c.get("baseId") or (re.sub(r"_b$", "_a", i) if c.get("isGolden") else i)

    def golden_id_of(self, i):
        c = self.chess(self.base_id_of(i))
        if c and c.get("goldenId") and self.chess(c["goldenId"]):
            return c["goldenId"]
        alt = re.sub(r"_a$", "_b", i) if isinstance(i, str) else None
        return alt if alt and self.chess(alt) else None

    def is_golden(self, i):
        c = self.chess(i) or self.item(i)
        return bool(c and c.get("isGolden"))

    def tier_of(self, i):
        c = self.chess(i) or self.item(i)
        t = c.get("tier") if c else None
        return t if isinstance(t, int) and not isinstance(t, bool) else 1

    def merge_count(self, i):
        c = self.chess(i)
        if not c or c.get("isGolden"):
            return 0
        n = c.get("upgradeNum")
        if isinstance(n, int) and n > 0:
            return n
        eco = (self.raw.get("config") or {}).get("economy") or {}
        ov = eco.get("mergeCountOverrides") or {}
        if isinstance(ov.get(i), int):
            return ov[i]
        n = eco.get("mergeCount")
        return n if isinstance(n, int) and n > 0 else 3

    # ---- bands ----
    def band_allowed(self, band_id):
        b = self.band(band_id)
        if not b:
            return False
        if isinstance(b.get("inactiveModeIds"), list) and self.mode_id in b["inactiveModeIds"]:
            return False
        lst = b.get("modeTypeList")
        return True if not isinstance(lst, list) else "SINGLE" in lst

    def band_ids(self):
        return list(self.derived["bandIds"])

    def pool_copies(self, base_id):
        """SharedPool copies of a base chess (gamedata.js poolCopies, solo: multiplier 1)."""
        econ = (self.raw.get("config") or {}).get("economy") or {}
        ov = econ.get("poolCopiesOverrides") or {}
        if isinstance(ov.get(base_id), int) and ov[base_id] >= 0:
            return ov[base_id]
        v = (econ.get("poolCopies") or {}).get(str(self.tier_of(base_id)))
        if isinstance(v, int) and v >= 0:
            return v
        return POOL_COPIES_DEFAULT.get(self.tier_of(base_id), 10)

    def start_lp(self, band_id):
        b = self.band(band_id)
        hp = b.get("totalHp") if b else None
        return hp if isinstance(hp, int) and hp > 0 else self.derived["defaultStartLp"]

    def band_bond_ids(self, band_id):
        if band_id not in self._band_bonds:
            listed = (self.band(band_id) or {}).get("bondIds")
            s = set(listed) if isinstance(listed, list) else set()
            self._band_bonds[band_id] = tuple(b for b in self.bond_ids if b in s)
        return self._band_bonds[band_id]


@lru_cache(maxsize=1)
def load() -> GameData:
    return GameData()
