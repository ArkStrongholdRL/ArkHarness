"""Legacy rules the harness applies itself (GOAL-live B4). Each function returns what to do; nothing here clicks.

Ported from the engine (GPL, read only), sources and functions:
  server/match/botBondGroups.js        botPickBondGroup                  -> Rules.bond_group
  server/match/bot.js                  botBandWeights, botPickBand,      -> Rules.band_weights, Rules.band_pick
                                       pagerPick, bandStrength, BAND_STRENGTH
  server/match/bot.js                  pagerUse, pagerPlan, pagerSeat,   -> Rules.pager, Rules.pager_plan, Rules.pager_seat
                                       pagerCarrier, pagerEligible, bondReach
  server/match/bot.js                  botUseCommand, commandView,       -> Rules.command_start, Rules.command_end,
  server/match/bandCommands.js         commandTargets, syncCommand,         Rules.command_view, Rules.sync_command
                                       activate, pickCommandDraft
  server/match/player/commands.js      autoPickCommands                  -> Rules.auto_pick_commands
  server/match/bot.js                  resolveTemp, sellWeakestHand,     -> Rules.resolve_temp
                                       makeHandRoom (item branch: approx.)
  server/match/player/pieces.js        _fillHandFromTemp, sell (refill)  -> _Sim
  server/match/bot.js                  applyPlanSteps (moves only)       -> Rules.apply_plan
  server/match/aiEconomy.js            planProtected (natPlan: bonds only) -> Rules.plan_protected
  server/match/aiPlan.js               planMain / planOf (natPlan P)     -> Rules.plan_main, Rules.legacy_plan_p

Inputs: a ScreenState (bridge/screen.mjs fromEngine shape; `self.command`, `self.commandDrafts`, `self.pagerSeat` are
track fields) and the shared pool `pool` (visible base id -> copies left). The pool is not on the screen: pass None and
the approximation `pool_approx` (catalogue copies minus the seat's own copies) is used.

Randomness: the engine's draws are not reproduced. `band_pick` takes our rng; `auto_pick_commands` takes the drawn card
index of each pick, so a recorded draw replays exactly.

Approximations (each one is named where it happens and measured by tools/rules_parity.py):
  - pool copies left (pool_approx), pieceValue (sell choice), makeHandRoom's item branch, planItemRoomFor's junk
    item, MAP_TILE commands (need the field's deploy map and enemy routes), gold-unlock commands (gold not on screen),
    selfPlan's main (stays in Node: planMain uses only the natPlan P here).
"""

from __future__ import annotations

import copy
import math
import re

from .catalog import GameData, load
from .encode.plan import NAT_PLAN, Plan

PAGER_BAND = "band_jesica"
PAGER_BOND = "kjeragShip"
PAGER_ITEM = "use_equip_reward_special_goods_char_chess"
BOT_MAX_LEVEL = 5
SOFT_BAND_RESERVE = 0.25  # unused in solo (no living teammate to reserve against); kept for the record
MATE_BURDEN = 0.1         # unused in solo
BAND_STRENGTH_SHRINK = 0.7
BAND_STRENGTH_POWER = 6
# bot.js BAND_STRENGTH: 终极 solo pool completion (%) per band at dev 17a075f1 (handoff/round57)
BAND_STRENGTH = {
    "band_amiya": 51.8, "band_bldsk": 51.2, "band_sciurus": 48.1, "band_quintus": 47.7, "band_ermengard": 47.2, "band_qalaisa": 47.2,
    "band_sarkazb": 47.1, "band_mlyss": 46.5, "band_cannot": 45.1, "band_amedic": 44.6, "band_ioleta": 43.0, "band_harold": 42.9,
    "band_chen": 42.3, "band_makiri": 42.2, "band_emperor": 41.6, "band_dusk": 40.5, "band_ducklord": 40.2, "band_kirara": 39.9,
    "band_yu": 39.9, "band_jesica": 39.3, "band_doberm": 39.2, "band_orchid": 39.2, "band_cathy": 38.8, "band_chiave": 38.2,
    "band_duyaoy": 35.8, "band_mberry": 35.6, "band_damaztic": 34.2, "band_paganini": 33.8, "band_humus": 32.0, "band_malkie": 31.9,
    "band_fang": 31.4, "band_narant": 31.4, "band_lisa": 31.3, "band_pepe": 30.6, "band_vodfox": 29.5, "band_lmlee": 29.2,
    "band_pith": 28.5, "band_justin": 28.3, "band_mlynar": 25.1, "band_clementia": 18.6,
}
BAND_STRENGTH_MEAN = sum(BAND_STRENGTH.values()) / len(BAND_STRENGTH)
# bot.js EFFECT_ALIAS (item buff key -> the key the item is known by)
EFFECT_ALIAS = {
    "use_equip_spe_goods_first_bond": "use_equip_reward_special_goods_char_chess",
    "use_equip_reward_same_tier": "use_equip_reward_char_chess",
    "use_equip_next_gold": "use_equip_gain_coin_when_next_round_start",
    "equip_tier_recr_gift": "use_equip_recruit_new_char_and_give_char_to_player_most_bond",
    "use_equip_bond_layer": "use_equip_reward_char_chess_bond_layer",
    "equip_destory_first_bond_char": "use_equip_reward_char_chess_with_same_bond",
    "equip_use_main_bond": "use_equip_reward_char_chess_with_same_bond",
    "prep_end_act_bond": "use_equip_reward_char_chess_bond_layer",
}
MUTATION_CELL = "char_chess_transformation_equip"
MARK_KEYS = {"skill_mark_char_inst_tag", "skill_mark_pos_grow_use_cnt", "skill_add_pos_effect_enemy",
             "skill_mark_shop_equip_tag", "skill_layer_limit_grow_round", "skill_garr_event_extra_exec"}
TEMP_SIZE = 5


def _params(b: dict) -> dict:
    return {**(b.get("bb") or {}), **(b.get("bbStr") or {})}


class Rules:
    """All rules for one catalogue (solo, mode_single_20, natPlan 'order' advisor)."""

    def __init__(self, gd: GameData | None = None):
        self.gd = gd or load()
        from .encode.encoder import Encoder  # the vocab's sorted bond order is the plan block's index
        self.plan = Plan(self.gd, Encoder(self.gd).vocab.bond_ids)
        self.visible = self.plan.visible
        self.skills: dict[str, dict] = {}
        for band in self.gd.raw["bands"].values():
            for row in band.get("skills") or []:
                self.skills[row["effectId"]] = row

    # ------------------------------------------------------------------ helpers
    def base_of(self, chess_id: str) -> str:
        return self.gd.base_id_of(chess_id)

    def bond_thresholds(self, bond: str) -> list:
        th = (self.gd.bond(bond) or {}).get("thresholds")
        return list(th) if isinstance(th, list) else []

    def top_threshold(self, bond: str) -> float:
        th = self.bond_thresholds(bond)
        return th[-1] if th else math.inf

    def bond_unavailable(self, bond: str, disabled: list) -> bool:
        gd = self.gd
        return bond in gd.mode_inactive_bonds or (gd.regulation_bond_draw and not self.plan.playable(bond, disabled))

    def banned_at_start(self, disabled: list) -> set:
        """bannedChessForBonds(gd, drawn, staticOff): visible chess whose bonds the draw (disabled + static) takes out."""
        off = set(disabled) | self.gd.mode_inactive_bonds
        return {i for i in self.visible if self.plan._excluded(self.gd.chess(i), off)}

    def pool_initial(self, disabled: list) -> dict:
        banned = self.banned_at_start(disabled)
        return {i: self.gd.pool_copies(i) for i in self.visible if i not in banned and self.gd.pool_copies(i) > 0}

    def held_copies(self, s: dict) -> dict:
        held: dict[str, int] = {}
        for p in self.all_chess(s):
            base = self.base_of(p["id"])
            n = self.gd.merge_count(base) if self.gd.is_golden(p["id"]) else 1
            held[base] = held.get(base, 0) + max(1, n)
        return held

    def pool_approx(self, s: dict) -> dict:
        """Copies left per visible base, from the catalogue minus the copies the seat holds (approximation: the shared
        pool is not on the screen; other sources of copies leaving or returning to it are not seen)."""
        pool = self.pool_initial(s["match"].get("disabledBonds") or [])
        held = self.held_copies(s)
        out = {}
        for i, c in pool.items():
            left = c - held.get(i, 0)
            if left > 0:
                out[i] = left
        return out

    def _pool(self, s: dict, pool: dict | None) -> dict:
        return pool if pool is not None else self.pool_approx(s)

    @staticmethod
    def board_pieces(s: dict) -> list:
        """The board in the engine's Map order (ScreenState board order)."""
        return [p for p in (s["self"].get("board") or []) if p]

    @staticmethod
    def board_order(s: dict) -> list:
        """boardOrder (board.js): rows bottom-up, then columns left to right; the order of allChess()."""
        return sorted(Rules.board_pieces(s), key=lambda p: (-(p.get("row") or 0), p.get("col") or 0))

    def all_chess(self, s: dict) -> list:
        out = [p for p in self.board_order(s) if p.get("kind") == "chess"]
        out += [p for p in (s["self"].get("hand") or []) if p and p.get("kind") == "chess"]
        out += [p for p in (s["self"].get("temp") or []) if p and p.get("kind") == "chess"]
        return out

    def owned_bonds(self, s: dict) -> dict:
        """ownedBonds: distinct owned bases per bond over board, hand and temp."""
        counts: dict[str, int] = {}
        bases: set[str] = set()
        for p in self.all_chess(s):
            base = self.base_of(p["id"])
            if base in bases:
                continue
            bases.add(base)
            for b in (self.gd.chess(p["id"]) or {}).get("bonds") or []:
                counts[b] = counts.get(b, 0) + 1
        return {"counts": counts, "bases": bases}

    # ------------------------------------------------------------------ plan main (legacy planFor under natPlan 'order')
    def legacy_plan_p(self, s: dict):
        """The P that legacy's planFor returns: null until the group is picked and round 1 starts, else the first
        natPlan.order bond that is a playable core bond (the same P as plan.py, which the env uses for the plan block)."""
        bc = s["match"].get("bondChoice")
        if bc and bc.get("picked") is None:
            return None
        if (s["match"].get("round") or 0) < 1:
            return None
        disabled = s["match"].get("disabledBonds") or []
        pl = [b for b in self.gd.bond_ids if (self.gd.bond(b) or {}).get("isCore")
              and (not self.gd.regulation_bond_draw or self.plan.playable(b, disabled))]
        return next((b for b in NAT_PLAN["order"] if b in pl), None)

    def plan_main(self, s: dict):
        """planMain: the plan's highest-priority core bond. selfPlan's own bonds (a Node-side plan) are not modelled."""
        p = self.legacy_plan_p(s)
        if p is None or not (self.gd.bond(p) or {}).get("isCore") or p in self.gd.mode_inactive_bonds:
            return None
        return p

    # ------------------------------------------------------------------ 1. core bond group (botPickBondGroup)
    def bond_group(self, s: dict) -> int:
        """Index of the briefing group holding the main core bond (first group that has it); 0 otherwise."""
        bc = s["match"].get("bondChoice")
        main = self.plan_main(s)
        if main is None and s["self"].get("bandId"):
            main = next((b for b in self.gd.band_bond_ids(s["self"]["bandId"]) if (self.gd.bond(b) or {}).get("isCore")), None)
        if not bc or main is None:
            return 0
        idx = next((i for i, g in enumerate(bc.get("groups") or []) if main in g), -1)
        return 0 if idx < 0 else idx

    # ------------------------------------------------------------------ 2. opening band (botPickBand / botBandWeights)
    def band_strength(self, band: str) -> float:
        if self.gd.difficulty != "ABYSS":
            return 1.0
        s = BAND_STRENGTH.get(band, BAND_STRENGTH_MEAN)
        return ((BAND_STRENGTH_MEAN + BAND_STRENGTH_SHRINK * (s - BAND_STRENGTH_MEAN)) / 50) ** BAND_STRENGTH_POWER

    def _bond_reach(self, bond: str, owned: dict, pool: dict) -> int:
        """bondReach: owned distinct members plus the unowned members of tier <= BOT_MAX_LEVEL with copies left."""
        n = owned["counts"].get(bond, 0)
        for cid, left in pool.items():
            if left <= 0 or self.gd.tier_of(cid) > BOT_MAX_LEVEL or cid in owned["bases"]:
                continue
            if bond in ((self.gd.chess(cid) or {}).get("bonds") or []):
                n += 1
        return n

    def band_weights(self, s: dict, pool: dict | None = None) -> dict:
        """{'forced': band | None, 'pairs': [(band, weight)]} at the band draft of a solo bot seat (no draw)."""
        gd = self.gd
        ids = gd.band_ids()
        disabled = s["match"].get("disabledBonds") or []
        inactive = {b: any(self.bond_unavailable(x, disabled) for x in gd.band_bond_ids(b)) for b in ids}
        # pagerPick: 杰西卡 is taken outright when 谢拉格 can still reach its top threshold (solo ABYSS, a bot seat)
        if self._is_solo_abyss(s) and not self.bond_unavailable(PAGER_BOND, disabled):
            pick = next((b for b in ids if b == PAGER_BAND and not inactive[b]), None)
            if pick and self._bond_reach(PAGER_BOND, {"counts": {}, "bases": set()}, self._pool(s, pool)) >= self.top_threshold(PAGER_BOND):
                return {"forced": pick, "pairs": []}
        pairs = []
        for b in ids:
            if inactive[b]:
                pairs.append((b, 0.0))
                continue
            w = max(1.0, (gd.start_lp(b) - 18) ** (3 if gd.difficulty == "ABYSS" else 2))
            if re.search("暂存", str((gd.band(b) or {}).get("desc") or "")) and self._is_solo(s):
                w *= 0.02  # lateFunds: a solo band that withholds the first rounds' funds is avoided
            w *= self.band_strength(b)
            pairs.append((b, w))
        return {"forced": None, "pairs": pairs}

    def _is_solo(self, s: dict) -> bool:
        return s["config"].get("modeId") == self.gd.mode_id

    def _is_solo_abyss(self, s: dict) -> bool:
        return self._is_solo(s) and self.gd.difficulty == "ABYSS"

    @staticmethod
    def band_pick(weights: dict, rng) -> str | None:
        """Our own draw over the engine's weights (botPickBand's cumulative order): forced, else weighted."""
        if weights["forced"]:
            return weights["forced"]
        total = sum(w for _, w in weights["pairs"])
        if not total > 0:
            return None
        r = rng.random() * total
        last = None
        for b, w in weights["pairs"]:
            if not w > 0:
                continue
            last = b
            r -= w
            if r < 0:
                return b
        return last

    # ------------------------------------------------------------------ 3. commands (bot.js, bandCommands.js)
    def _activate(self, cmd: dict, skill: dict, rnd: int) -> None:
        cmd["currentEffectId"] = skill["effectId"]
        recs = cmd["records"]
        if skill["effectId"] not in recs:
            recs[skill["effectId"]] = {"currentCost": skill["cost"], "used": 0, "round": rnd, "readyRound": 0, "totalUsed": 0}
        st = recs[skill["effectId"]]
        if st["round"] != rnd:
            st["round"] = rnd
            st["used"] = 0

    def _unlocked(self, s: dict, skill: dict) -> bool:
        a, b = (skill.get("unlockParam") or [None, None])[:2]
        t = skill.get("unlockType")
        if t == "band_skill_load_by_round":
            return (s["match"]["round"] or 0) >= float(a)
        if t == "band_skill_load_by_shop_lv":
            return s["self"]["shop"]["level"] >= float(a)
        if t == "band_skill_load_by_bond_act_count":
            bond = next((x for x in s["self"].get("bonds") or [] if x.get("bondId") == a), None)
            return bool(bond) and (bond.get("count") or 0) >= float(b)
        return False  # band_skill_load_by_coin_cost: gold is not on the screen (see the module note)

    def sync_command(self, s: dict, cmd: dict | None) -> dict | None:
        """syncCommand: the held command follows its band's chain; unlocks move it up the chain, never back."""
        band_id = s["self"].get("bandId")
        band = self.gd.band(band_id) if band_id else None
        if not band or not band.get("skills"):
            return None
        rnd = s["match"]["round"]
        cmd = copy.deepcopy(cmd) if cmd else None
        if not cmd or cmd["bandId"] != band_id:
            cmd = {"bandId": band_id, "currentEffectId": None, "records": {}, "swapped": False}
            self._activate(cmd, band["skills"][0], rnd)
        for _ in range(74):
            cur = self.skills.get(cmd["currentEffectId"]) if cmd["currentEffectId"] else None
            if not cur or not cur.get("next") or not self._unlocked(s, cur):
                break
            nxt = self.skills.get(cur["next"])
            if not nxt:
                break
            old = cmd["records"].get(cur["effectId"])
            self._activate(cmd, nxt, rnd)
            if old and old["round"] == rnd:
                cmd["records"][nxt["effectId"]]["used"] = old["used"]
        return cmd

    def auto_pick_commands(self, s: dict, drawn: list) -> dict:
        """autoPickCommands with the card index drawn for each pick (replayed). Returns the picks taken and the command
        state after them: {'picks': [(draft id, idx)], 'command': state, 'drafts_left': n}."""
        cmd = copy.deepcopy(s["self"].get("command"))
        drafts = [dict(d) for d in s["self"].get("commandDrafts") or []]
        rnd = s["match"]["round"]
        picks = []
        k = 0
        while drafts and len(picks) < 16 and k < len(drawn):
            d = drafts[0]
            idx = drawn[k]
            k += 1
            skill = self.skills.get(d["cards"][idx]) if 0 <= idx < len(d["cards"]) else None
            if skill is None or not cmd:
                break
            picks.append((d["id"], idx))
            self._activate(cmd, skill, rnd)
            cmd["swapped"] = True
            drafts.pop(0)
            cmd = self.sync_command(s, cmd)
        return {"picks": picks, "command": cmd, "drafts_left": len(drafts)}

    def command_view(self, s: dict) -> dict | None:
        cmd = s["self"].get("command")
        if not cmd or not cmd.get("currentEffectId"):
            return None
        skill = self.skills.get(cmd["currentEffectId"])
        if not skill:
            return None
        rnd = s["match"]["round"]
        st = cmd["records"].get(skill["effectId"]) or {"currentCost": skill["cost"], "used": 0, "round": rnd, "readyRound": 0, "totalUsed": 0}
        used = st["used"] if st["round"] == rnd else 0
        avail = skill.get("availType")
        if avail == "ALL_LIMIT":
            uses_left = max(0, int(skill["availCount"]) - st["totalUsed"])
        elif avail == "ROUND_RESET":
            uses_left = max(0, int(skill["availCount"]) - used)
        else:
            uses_left = None
        cooldown = max(0, st["readyRound"] - rnd)
        special = bool(skill.get("unusableAtSpecialRecruit")) and s["self"]["shop"].get("rewardOffer") is not None
        if skill["type"] == "NONE":
            reason = "locked"
        elif skill["type"] != "ACTIVE":
            reason = "passive"
        elif cooldown:
            reason = "cooldown"
        elif uses_left == 0:
            reason = "exhausted"
        elif special:
            reason = "specialRecruit"
        elif s["self"]["funds"] < st["currentCost"]:
            reason = "funds"
        else:
            reason = None
        targets = self.command_targets(s, skill) if skill["type"] == "ACTIVE" else []
        return {"effectId": skill["effectId"], "reason": reason, "usable": reason is None, "targets": targets,
                "currentCost": st["currentCost"], "skill": skill}

    def command_targets(self, s: dict, skill: dict):
        """commandTargets. COMMON_CHESS: board then hand chess; SHOP_SLOT: shop slots; NO_TARGET: [{}].
        MAP_TILE is not modelled (it needs the field's deploy map and the enemy routes): returns None."""
        ttype = skill.get("targetType")
        buffs = skill.get("buffs") or []
        keys = {b.get("key") for b in buffs}
        if ttype == "NO_TARGET":
            return [{}]
        if ttype == "COMMON_CHESS":
            out = []
            for p in self.board_pieces(s) + [q for q in s["self"].get("hand") or [] if q]:
                if p.get("kind") != "chess":
                    continue
                if "skill_upgrade_char" in keys or any(b.get("key") == "skill_mark_char_inst_tag" and _params(b).get("cond") == "gold" for b in buffs):
                    if self.gd.is_golden(p["id"]) or not self.gd.golden_id_of(p["id"]):
                        continue
                if "skill_trigger_garr_event" in keys or "skill_garr_event_extra_exec" in keys:
                    buff = next(b for b in buffs if b.get("key") in ("skill_trigger_garr_event", "skill_garr_event_extra_exec"))
                    ev = _params(buff).get("event")
                    rec = self.gd.chess(p["id"]) or {}
                    if not any((self.gd.garrison(g) or {}).get("eventType") == ev for g in rec.get("garrisonIds") or []):
                        continue
                out.append({"uid": p["uid"]})
            return out
        if ttype == "SHOP_SLOT":
            out = []
            want = "item" if skill.get("targetFilterType") == "EQUIP" else "chess"
            for i, sl in enumerate(s["self"]["shop"].get("slots") or []):
                if not sl or sl.get("sold") or sl.get("kind") != want:
                    continue
                if "skill_mark_shop_equip_tag" in keys:
                    item = self.gd.item(sl["id"])
                    if not item or item.get("itemType") != "EQUIP" or "装备时销毁" in str(item.get("desc") or ""):
                        continue
                out.append({"slot": i})
            return out
        if ttype == "MAP_TILE":
            # bandCommands.js commandTargets MAP_TILE: board rows 9-12 x cols 2-10 in reading order, on the field's stage
            # tile (board.js fieldTile: the boss field is 7 rows up, the left half in a solo match), passable 'ALL',
            # PASSABLE: not a start / end tile; PASSABLE_AND_BUILDABLE: deployable (approximated: buildable tile) and free.
            # Command marks of this round are not tracked here (each mark command is used once per round).
            stage = self.gd.stage(s["match"].get("stageId")) or {}
            rows, tiles = stage.get("rows") or [], stage.get("tiles") or {}
            boss = s["match"].get("round") in (self.gd.boss_round, self.gd.hidden_round)
            occupied = {(p["row"], p["col"]) for p in self.board_pieces(s)}
            filt = skill.get("targetFilterType")
            out = []
            for row in range(9, 13):
                for col in range(2, 11):
                    r = row - 7 if boss else row
                    ch = rows[r][col] if 0 <= r < len(rows) and 0 <= col < len(rows[r]) else None
                    tile = tiles.get(ch) if ch is not None else None
                    if not tile or tile.get("passable") != "ALL":
                        continue
                    if filt == "PASSABLE" and tile.get("tileKey") in ("tile_start", "tile_end"):
                        continue
                    if filt == "PASSABLE_AND_BUILDABLE" and (tile.get("buildable") in (None, "NONE") or (row, col) in occupied):
                        continue
                    out.append({"row": row, "col": col})
            return out
        return []

    def command_decide(self, s: dict, battle_marks: bool) -> dict | None:
        """botUseCommand(m, ps, battleMarks): {'effectId', 'target'} to use, None (nothing to do), or
        {'unsupported': why} when the decision needs a model that is not ported."""
        view = self.command_view(s)
        if not view or not view["usable"]:
            return None
        if not view["targets"]:
            return None
        skill = view["skill"]
        keys = {b.get("key") for b in skill.get("buffs") or []}
        if bool(keys & MARK_KEYS) != battle_marks:
            return None
        if "skill_drop_hp_to_low_gain_coin" in keys or "skill_sell_char_bond_layer" in keys:
            return None
        if "skill_next_equip_buy_discount" in keys and s["self"]["funds"] < view["currentCost"] + 1:
            return None
        targets = view["targets"]
        target = targets[0]
        if skill["targetType"] == "COMMON_CHESS":
            transform = "skill_transform_char_up_tier" in keys or "skill_remove_gain_bond_char" in keys
            on_board = {p["uid"] for p in self.board_pieces(s)}
            pieces = {p["uid"]: p for p in self.board_pieces(s) + [q for q in s["self"].get("hand") or [] if q]}

            def score(t):
                p = pieces.get(t["uid"])
                if not p:
                    return 0
                return self.gd.tier_of(p["id"]) * 10 + len(p.get("items") or []) + (100 if t["uid"] in on_board else 0)

            # JS sort is stable: ties keep the target order (board, then hand)
            ordered = sorted(targets, key=score if transform else (lambda t: -score(t)))
            target = ordered[0]
        if "skill_add_pos_effect_enemy" in keys:
            return {"unsupported": "route tile (skill_add_pos_effect_enemy)"}
        return {"effectId": skill["effectId"], "target": target}

    def command_start(self, s: dict) -> dict | None:
        """botUseCommand(m, ps, false) at the prep start (after autoPickCommands)."""
        return self.command_decide(s, battle_marks=False)

    def command_end(self, s: dict) -> dict | None:
        """botUseCommand(m, ps, true) in the END flow (after equipItems)."""
        return self.command_decide(s, battle_marks=True)

    # ------------------------------------------------------------------ 4. pager (pagerUse, the 杰西卡 seat)
    def effect_key(self, item_id: str) -> str:
        rec = self.gd.item(item_id) or {}
        buffs = rec.get("buffs") or []
        key = buffs[0].get("key") if buffs and isinstance(buffs[0], dict) and isinstance(buffs[0].get("key"), str) else ""
        return EFFECT_ALIAS.get(key, key)

    def pager_seat(self, s: dict, pool: dict | None = None) -> bool:
        """pagerSeat: the seat builds 谢拉格 from round 1 (decided at its first look, then cached: the track field)."""
        if not self._is_solo_abyss(s) or (s["match"].get("round") or 0) < 1:
            return False
        if s["self"].get("bandId") != PAGER_BAND or self.bond_unavailable(PAGER_BOND, s["match"].get("disabledBonds") or []):
            return False
        if (self.plan_main(s) or PAGER_BOND) != PAGER_BOND:
            return False
        if s["self"].get("pagerSeat") is not None:
            return bool(s["self"]["pagerSeat"])
        return self._bond_reach(PAGER_BOND, self.owned_bonds(s), self._pool(s, pool)) >= self.top_threshold(PAGER_BOND)

    def pager_plan(self, s: dict, pool: dict | None = None):
        if not self.pager_seat(s, pool):
            return None
        owned = self.owned_bonds(s)
        k = owned["counts"].get(PAGER_BOND, 0)
        top = self.top_threshold(PAGER_BOND)
        return {"bond": PAGER_BOND, "k": k, "top": top, "owned": owned} if k < top else None

    def _pager_carrier(self, s: dict, h: dict, pool: dict):
        """pagerCarrier: the owned member whose offer most likely shows a missing member (first max wins)."""
        best, best_p = None, -1.0
        shop_level = s["self"]["shop"]["level"]
        for p in self.all_chess(s):
            bonds = list((self.gd.chess(p["id"]) or {}).get("bonds") or [])
            if h["bond"] not in bonds:
                continue
            all_c = miss = 0
            for cid, left in pool.items():
                if left <= 0 or self.gd.tier_of(cid) > shop_level:
                    continue
                cb = (self.gd.chess(cid) or {}).get("bonds") or []
                if not any(b in bonds for b in cb):
                    continue
                all_c += left
                if h["bond"] in cb and cid not in h["owned"]["bases"]:
                    miss += left
            prob = 1 - (1 - miss / all_c) ** 3 if all_c > 0 else 0.0
            if prob > best_p + 1e-9:
                best_p, best = prob, p
        return best

    def pager(self, s: dict, pool: dict | None = None):
        """pagerUse at the prep start: the list of {'item', 'carrier'} equips (uids), in order (usually at most one)."""
        pl = self._pool(s, pool)
        items = [p for p in (s["self"].get("hand") or []) + (s["self"].get("temp") or []) if p and p.get("kind") == "item"]
        out = []
        for _ in range(4):
            h = self.pager_plan(s, pl)
            if not h:
                break
            item = next((q for q in items if self.effect_key(q["id"]) == PAGER_ITEM), None)
            carrier = self._pager_carrier(s, h, pl) if item else None
            if not carrier:
                break
            out.append({"item": item["uid"], "carrier": carrier["uid"]})
            items = [q for q in items if q is not item]
            s = copy.deepcopy(s)
            # the equipped item leaves the hand; the carrier's items change its bonds only through the item's own bonds
            s["self"]["hand"] = [None if (q and q["uid"] == item["uid"]) else q for q in s["self"].get("hand") or []]
            s["self"]["temp"] = [None if (q and q["uid"] == item["uid"]) else q for q in s["self"].get("temp") or []]
        return out

    # ------------------------------------------------------------------ 5. resolveTemp (the temp area)
    def plan_protected(self, s: dict, piece: dict) -> bool:
        """planProtected under the natPlan advisor (plan = bonds only): an owned member of the plan's bond whose removal
        lowers that bond's active tier. 'before' comes from the distinct owned members (the bond panel's count is not
        used: it is the engine's membership, not reproduced here)."""
        P = self.plan_main(s)
        if P is None or piece.get("kind") != "chess" or P not in ((self.gd.chess(piece["id"]) or {}).get("bonds") or []):
            return False
        # aiEconomy.js computeBonds: distinct members on the board, plus the hand for a bond counting it (BOARD_AND_DECK)
        S = s["self"]
        hand_counts = (self.gd.bond(P) or {}).get("countMode") == "BOARD_AND_DECK"
        pool = [p for p in (S.get("board") or []) if p] + ([p for p in (S.get("hand") or []) if p] if hand_counts else [])
        members = lambda ps: {self.base_of(p["id"]) for p in ps if p.get("kind") == "chess"  # noqa: E731
                              and P in ((self.gd.chess(p["id"]) or {}).get("bonds") or [])}
        before_count = len(members(pool))
        after_count = len(members([p for p in pool if p.get("uid") != piece.get("uid")]))
        tier = lambda c: sum(1 for t in self.bond_thresholds(P) if c >= t)  # noqa: E731
        return tier(before_count) > 0 and tier(after_count) < tier(before_count)

    def pager_keeps(self, s: dict, piece: dict) -> bool:
        if piece.get("kind") != "chess" or PAGER_BOND not in ((self.gd.chess(piece["id"]) or {}).get("bonds") or []):
            return False
        if not self.pager_seat(s):
            return False
        base = self.base_of(piece["id"])
        return not any(q["uid"] != piece["uid"] and self.base_of(q["id"]) == base for q in self.all_chess(s))

    def piece_value(self, piece: dict) -> float:
        """Approximation of pieceValue (legacy: the lineup's marginal value, which needs the layout model). Tier first,
        then the equipment and the elite form."""
        return self.gd.tier_of(piece["id"]) * 10 + len(piece.get("items") or []) * 2 + (5 if self.gd.is_golden(piece["id"]) else 0)

    def fa_keeps(self, p: dict) -> bool:
        """bot.js faKeeps: a piece with items, a member of a bond counting the hand, or one with a recurring layer 特质."""
        if p.get("items"):
            return True
        c = self.gd.chess(p.get("id"))
        if not c:
            return True
        if any((self.gd.bond(b) or {}).get("countMode") == "BOARD_AND_DECK" for b in c.get("bonds") or []):
            return True
        for gid in c.get("garrisonIds") or []:
            g = self.gd.garrison(gid) or {}
            if isinstance(g.get("effectKey"), str) and re.search("BOND|LAYER", g["effectKey"]) \
                    and g.get("eventType") in ("SERVER_PREP_START", "SERVER_PREP_FIN", "SERVER_REFRESH_SHOP"):
                return True
        return False

    def resolve_temp(self, s: dict, pool: dict | None = None) -> list:
        """resolveTemp under equipHand (config.version.rules): tempPieces() is the overflow queue temp[TEMP_SIZE:]; a
        freed slot takes a chess piece into the hand, an item into the equipment row temp[:TEMP_SIZE]. Ops, in order:
        ('move', uid, area, idx) | ('sell', uid) | ('destroy', uid)."""
        sim = _Sim(s)
        ops: list = []
        if s["match"].get("round") == self.gd.boss_round:
            # bot.js faItemHunt, first step (faHunts: the Final Assault prep): the bench is sold except faKeeps / planProtected
            # pieces. Its damage-item purchase and equipping need the boss field model and stay with the layout side.
            for p in [q for q in sim.hand if q]:
                if p.get("kind") != "chess" or self.fa_keeps(p) or self.plan_protected(sim.state(), p):
                    continue
                sim.sell(p["uid"])
                ops.append(("sell", p["uid"]))
        for p in [q for q in sim.temp[TEMP_SIZE:] if q]:
            if not sim.in_temp(p):
                continue
            idx = sim.free_for(p.get("kind"))
            if idx >= 0:
                sim.move(p, idx)
                ops.append(("move", p["uid"], "temp" if p.get("kind") == "item" else "hand", idx))
                continue
            if p.get("kind") == "chess":
                if self.plan_protected(sim.state(), p):
                    if self._sell_weakest(sim, handOnly=True, keep_pairs=True, ops=ops) or self._item_room(sim, ops):
                        if not sim.in_temp(p):
                            continue
                        j = sim.free_for("chess")
                        if j >= 0:
                            sim.move(p, j)
                            ops.append(("move", p["uid"], "hand", j))
                            continue
                sold = self._sell_weakest(sim, handOnly=False, keep_pairs=True, ops=ops)
                if not sold and not (self.pager_keeps(sim.state(), p) and self._sell_weakest(sim, handOnly=False, keep_pairs=False, ops=ops)):
                    sim.sell(p["uid"])
                    ops.append(("sell", p["uid"]))
            elif p.get("kind") == "item":
                sim.destroy(p["uid"])
                ops.append(("destroy", p["uid"]))
        for p in [q for q in sim.temp[TEMP_SIZE:] if q]:  # the final sweep: chess sold, items destroyed
            if not sim.in_temp(p):
                continue
            if p.get("kind") == "chess":
                sim.sell(p["uid"])
                ops.append(("sell", p["uid"]))
            elif p.get("kind") == "item":
                sim.destroy(p["uid"])
                ops.append(("destroy", p["uid"]))
        return ops

    def _sell_weakest(self, sim, *, handOnly: bool, keep_pairs: bool, ops: list) -> bool:
        """sellWeakestHand: sells the weakest eligible chess piece (temp + hand, or hand only). True when one was sold."""
        s = sim.state()
        copies: dict[str, int] = {}
        for p in self.all_chess(s):
            copies[self.base_of(p["id"])] = copies.get(self.base_of(p["id"]), 0) + 1
        cand = [p for p in sim.hand if p] if handOnly else [p for p in sim.temp[TEMP_SIZE:] if p] + [p for p in sim.hand if p]
        worst, worst_v = None, math.inf
        for p in cand:
            if p.get("kind") != "chess" or any(self.effect_key(i) == MUTATION_CELL for i in p.get("items") or []):
                continue
            if self.pager_keeps(s, p):
                continue
            base = self.base_of(p["id"])
            if keep_pairs and not self.gd.is_golden(p["id"]) and copies.get(base, 0) >= 2:
                continue
            if self.plan_protected(s, p):
                continue
            v = self.piece_value(p)
            if v < worst_v:
                worst, worst_v = p, v
        if worst is None:
            return False
        sim.sell(worst["uid"])
        ops.append(("sell", worst["uid"]))
        return True

    def _item_room(self, sim, ops: list) -> bool:
        """planItemRoomFor: destroys a cheap junk item from the hand (price <= 2, not kept by the plan); else False."""
        s = sim.state()
        cand = [q for q in sim.hand if q and q.get("kind") == "item" and self.effect_key(q["id"]) != MUTATION_CELL
                and (self.gd.item(q["id"]) or {}).get("price", math.inf) <= 2]
        if not cand:
            return False
        junk = sorted(cand, key=lambda q: ((self.gd.item(q["id"]) or {}).get("price", math.inf), q["uid"]))[0]
        sim.destroy(junk["uid"])
        ops.append(("destroy", junk["uid"]))
        return True

    # ------------------------------------------------------------------ 6. applyPlanSteps (moves onto the planned tiles)
    def apply_plan(self, pre: dict, post: dict) -> dict:
        """The moves that put the pre-END pieces on the post-END tiles (the post board is the layout's output).

        applyPlanSteps: tokens lift off the board first (to a hand slot), then up to three passes over the chosen pieces
        (post board order here), each moved to its target unless it is already there facing the planned way; a move onto
        an occupied tile swaps the occupant to the piece's old place. Returns {'board': [(uid, row, col, dir)],
        'hand': [(uid, idx)]} in the order made. Lineup swaps and off-board placement (the layout model) are not modelled."""
        board: dict = {}
        loc: dict = {}
        for p in self.board_pieces(pre):
            board[(p["row"], p["col"])] = p
            loc[p["uid"]] = ("board", (p["row"], p["col"]))
        hand = list(pre["self"].get("hand") or [])
        temp = list(pre["self"].get("temp") or [])
        for i, p in enumerate(hand):
            if p:
                loc[p["uid"]] = ("hand", i)
        for i, p in enumerate(temp):
            if p:
                loc[p["uid"]] = ("temp", i)
        by_uid = {p["uid"]: p for p in self.board_pieces(pre) + [q for q in hand + temp if q]}
        out_board: list = []
        out_hand: list = []
        for tok in [p for p in self.board_pieces(pre) if p.get("kind") == "token"]:
            free = next((i for i in range(len(hand) - 1, -1, -1) if hand[i] is None), -1)
            if free < 0:
                break
            board.pop((tok["row"], tok["col"]), None)
            hand[free] = tok
            loc[tok["uid"]] = ("hand", free)
            out_hand.append((tok["uid"], free))
        target = [(p["uid"], (p["row"], p["col"]), p.get("dir")) for p in self.board_order(post)]
        # pieces that leave the board (benched by the layout) take the last free hand slot first, freeing their tiles
        post_uids = {p["uid"] for p in self.board_pieces(post)}
        for p in self.board_order(pre):
            if p["uid"] in post_uids or loc.get(p["uid"], ("",))[0] != "board":
                continue
            free = next((i for i in range(len(hand) - 1, -1, -1) if hand[i] is None), -1)
            if free < 0:
                break
            board.pop((p["row"], p["col"]), None)
            hand[free] = p
            loc[p["uid"]] = ("hand", free)
            out_hand.append((p["uid"], free))

        def place(uid, key, dire):
            where = loc[uid]
            piece = dict(by_uid[uid], dir=dire)
            by_uid[uid] = piece
            if where[0] == "board":
                board.pop(where[1], None)
            elif where[0] == "hand":
                hand[where[1]] = None
            else:
                temp[where[1]] = None
            occ = board.get(key)
            if occ is not None and occ["uid"] != uid:
                board.pop(key, None)
                if where[0] == "board":
                    board[where[1]] = occ
                    loc[occ["uid"]] = ("board", where[1])
                else:
                    slot = where[1]
                    if where[0] == "hand":
                        hand[slot] = occ
                    else:
                        temp[slot] = occ
                    loc[occ["uid"]] = (where[0], slot)
            board[key] = piece
            loc[uid] = ("board", key)
            out_board.append((uid, key[0], key[1], dire))

        for _ in range(3):
            moved = False
            for uid, key, dire in target:
                if uid not in loc:
                    continue
                where = loc[uid]
                if where[0] == "board" and where[1] == key and by_uid[uid].get("dir") == dire:
                    continue
                if where[0] == "board" and where[1] == key:
                    by_uid[uid] = dict(by_uid[uid], dir=dire)  # a facing change on its own tile is a move as well
                    out_board.append((uid, key[0], key[1], dire))
                    continue
                place(uid, key, dire)
                moved = True
            if not moved:
                break
        final = {uid: (key[0], key[1]) for uid, (kind, key) in loc.items() if kind == "board"}
        return {"board": out_board, "hand": out_hand, "final": final}


class _Sim:
    """The hand (bench) and the temp areas of one seat under equipHand: temp[:TEMP_SIZE] is the equipment row (items), the
    overflow queue is temp[TEMP_SIZE:]. After every change the refill runs (PlayerState._fillSplitHand): each overflow piece
    takes the free slot of its kind (chess: the hand, last free index; item: the equipment row), arrival order. A sold
    piece's items go to the overflow queue (stow, allowTemp)."""

    def __init__(self, s: dict):
        self.base = s
        self.hand = list(s["self"].get("hand") or [])
        self.temp = list(s["self"].get("temp") or [])
        while len(self.temp) < TEMP_SIZE:
            self.temp.append(None)
        self.fill()

    def state(self) -> dict:
        out = copy.deepcopy(self.base)
        out["self"]["hand"] = copy.deepcopy(self.hand)
        out["self"]["temp"] = copy.deepcopy(self.temp)
        return out

    def in_temp(self, p) -> bool:
        return any(q and q["uid"] == p["uid"] for q in self.temp[TEMP_SIZE:])

    def free_hand(self) -> int:
        for i in range(len(self.hand) - 1, -1, -1):
            if self.hand[i] is None:
                return i
        return -1

    def free_item(self) -> int:
        for i in range(TEMP_SIZE - 1, -1, -1):
            if self.temp[i] is None:
                return i
        return -1

    def free_for(self, kind) -> int:
        return self.free_item() if kind == "item" else self.free_hand()

    def fill(self) -> None:
        for j in range(TEMP_SIZE, len(self.temp)):
            p = self.temp[j]
            if not p:
                continue
            i = self.free_for(p.get("kind"))
            if i < 0:
                continue
            self.temp[j] = None
            if p.get("kind") == "item":
                self.temp[i] = p
            else:
                self.hand[i] = p
        self.temp = self.temp[:TEMP_SIZE] + [x for x in self.temp[TEMP_SIZE:] if x]

    def move(self, p, idx: int) -> None:
        """A piece from the overflow queue to the hand (chess) or the equipment row (item) slot idx."""
        for j, q in enumerate(self.temp):
            if q and q["uid"] == p["uid"]:
                self.temp[j] = None
        if p.get("kind") == "item":
            self.temp[idx] = p
        else:
            self.hand[idx] = p
        self.fill()

    def _remove(self, uid):
        for area in (self.hand, self.temp):
            for j, q in enumerate(area):
                if q and q["uid"] == uid:
                    area[j] = None
                    return q
        return None

    def sell(self, uid) -> None:
        p = self._remove(uid)
        if p:
            for it in p.get("items") or []:
                self.temp.append({"uid": -1, "kind": "item", "id": it, "items": []})
        self.fill()

    def destroy(self, uid) -> None:
        self._remove(uid)
        self.fill()
