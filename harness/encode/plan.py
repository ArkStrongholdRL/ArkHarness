"""The 59-dim plan block P7 was trained with (natPlan assign 'order').

Ported from the engine (GPL, read only): tools/rl/env.mjs _natBonds, _orderPlan, _planObs (slots / head off) and
server/match/match/infra.js bondLive / bondInPool / bondPlayable, server/match/gamedata.js bondsExclude.
"""

import numpy as np

from ..catalog import GameData

NAT_PLAN = {"assign": "order", "order": ["ursusShip", "lethaniaShip"], "bonus": 0, "sec": 0, "phiCount": 0, "phiLayers": 0}


class Plan:
    def __init__(self, gd: GameData, bond_ids_sorted: list, order=tuple(NAT_PLAN["order"])):
        self.gd = gd
        self.order = list(order)
        self.bond_ids = bond_ids_sorted  # encoder vocab order (sorted ids): the plan block's one-hot index
        self.index = {b: i for i, b in enumerate(bond_ids_sorted)}
        self.width = 1 + 2 * len(bond_ids_sorted) + 6
        raw = gd.raw.get("chess", {})
        self.visible = [i for i in sorted(raw) if raw[i].get("visible") and not raw[i].get("isGolden")
                        and not raw[i].get("isDiy") and not raw[i].get("isHidden") and isinstance(raw[i].get("tier"), int)]

    def _excluded(self, rec, off: set) -> bool:
        bonds = rec.get("bonds") if isinstance(rec.get("bonds"), list) else []
        if not self.gd.regulation_bond_draw:
            return bool(bonds) and all(b in off for b in bonds)
        core = [b for b in bonds if (self.gd.bond(b) or {}).get("isCore")]
        rel = core or bonds
        return bool(rel) and all(b in off for b in rel)

    def playable(self, bond_id: str, disabled: list) -> bool:
        gd = self.gd
        if bond_id in gd.mode_inactive_bonds:
            return False
        off = set(disabled) | gd.mode_inactive_bonds
        in_pool = any(bond_id in (gd.chess(i).get("bonds") or []) and not self._excluded(gd.chess(i), off) for i in self.visible)
        if not in_pool:
            return False
        return not (gd.regulation_bond_draw and (gd.bond(bond_id) or {}).get("isCore") and bond_id in disabled)

    def plan_of(self, s: dict):
        """{'P', 'S'} for ScreenState s, or None before the briefing group is picked / the band is set."""
        M = s["match"]
        bc = M.get("bondChoice")
        if bc and bc.get("picked") is None:
            return None
        if not s["self"].get("bandId"):
            return None
        disabled = M.get("disabledBonds") or []
        pl = [b for b in self.gd.bond_ids if (self.gd.bond(b) or {}).get("isCore")
              and (not self.gd.regulation_bond_draw or self.playable(b, disabled))]
        return {"P": next((b for b in self.order if b in pl), None), "S": None}

    def obs(self, s: dict, plan=None) -> np.ndarray:
        out = np.zeros(self.width, np.float32)
        plan = self.plan_of(s) if plan is None else plan
        if not plan or not plan.get("P"):
            return out
        nb = len(self.bond_ids)
        bonds = s["self"].get("bonds") or []

        def bv(i):
            b = next((x for x in bonds if x.get("bondId") == i), None)
            if not b:
                return {"tier": 0, "count": 0, "layers": 0, "active": False}
            return {"tier": (b.get("tier") or 0) if b.get("active") else 0, "count": b.get("count") or 0,
                    "layers": b.get("layers") or 0, "active": bool(b.get("active"))}

        out[0] = 1
        out[1 + self.index[plan["P"]]] = 1
        if plan.get("S"):
            out[1 + nb + self.index[plan["S"]]] = 1
        p = bv(plan["P"])
        sv = bv(plan["S"]) if plan.get("S") else {"tier": 0, "layers": 0, "active": False}
        q = 1 + 2 * nb
        out[q] = min(p["count"], 9) / 6
        out[q + 1] = p["tier"] / 2
        out[q + 2] = min(p["layers"], 999) / 100
        out[q + 3] = 1 if sv["active"] else 0
        out[q + 4] = min(sv["layers"], 999) / 100
        out[q + 5] = 1  # phase 'forced'
        return out
