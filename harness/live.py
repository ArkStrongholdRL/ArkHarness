"""Main loop of the live harness (GOAL-live E): the policy plays 卫戍协议 solo regulation 10–20.

  .venv\\Scripts\\python -m harness.live --games 3 --mode auto        (F3: unattended)
  .venv\\Scripts\\python -m harness.live --mode confirm               (F2: acts, asks before every END flow)
  .venv\\Scripts\\python -m harness.live --mode observe               (F1, default: reads, decides and logs; sends no game action)

State machine on harness.screen.flow.observe_screen: briefing → squad (both through flow `enter-layout`: the bond group
by rules.bond_group, the commander by the band rule) → [bounty] → preparation (prep start: holdings scan + rules +
legacy focus; policy loop; END flow: battle-mark command, legacy layout / equipment via the Node sidecar, rules
resolveTemp, legacy's free hand slot, ready) → battle → settlement → next round / match end.

Safety, in three tiers. Hard guards stay hard: price / funds / expected card and operator are verified before any click, and
an `unknown` screen is retried briefly, then the loop stops clicking, saves its state and exits with a report; decisions the
harness cannot carry out yet raise Unsupported (no guessing). Post-conditions are soft: when a check after a click, a scan or
a bookkeeping comparison fails, the loop never repeats the same action blindly: it marks what the action may have touched
as dirty, observes, re-reads and decides again (at most RETRY_LIMIT times per round and kind, then it stops with the
reason). Bookkeeping that disagrees with a screen-verified action is corrected by re-reading, not by stopping.

Logs (not committed): logs/live/<game>/decisions.jsonl (one row per decision: time, screen, ScreenState, ctx, obs hash,
legal actions, action, executor receipts, ms), screenshots on stops, state.json (resume after a restart: the loop
continues from the current screen; holdings are refreshed for the areas affected by round events).
"""

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .catalog import CATALOG as CATALOG_JSON, load
from .checkpoint import GameState, load_checkpoint, restore_track, save_checkpoint
from .encode.encoder import ACT, HAND, OWNED, TEMP, Ctx, Encoder, action_name
from .encode.plan import Plan
from .execution import device_action, mask_unverified_prep_actions, recover_action
from .policy import CKPT, Policy
from .prep_holdings import ITEM_GRANTS, prep_read_plan, scheduled_rack_slot
from .rules import Rules
from .sidecar import Sidecar, SidecarError
from .track import Track

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "live"
TIMEOUT_MS = 180_000
UNKNOWN_RETRIES = 4
LIVE_MODES = {f"mode_single_{level}" for level in range(10, 21)}
INCOME_DOLL_EFFECTS = {"gain_coin_when_round_start", "use_equip_gain_coin_when_next_round_start"}
EQUIPMENT_REWARDS = {
    "use_equip_spe_goods_first_bond": "chess",
    "use_equip_spe_goods_equip": "item",
    "equip_tier_recr_gift": "chess",
    "equip_destory_skill_pool": "command",
}
EQUIPMENT_MUTATIONS = {
    "use_equip_bond_layer": (),
    "use_equip_take_shop": ("hand", "board", "temp"),
    "equip_destory_first_bond_char": ("hand", "board", "temp"),
    "use_equip_reward_same_tier": ("hand", "board", "temp"),
    "equip_use_main_bond": (),
    "equip_use_deploy": (),
}


class Stop(RuntimeError):
    """The loop must stop clicking (unknown screen, unverifiable state)."""


class Unsupported(Stop):
    """A decision whose executor is not wired yet."""


class Retry(RuntimeError):
    """A soft check failed in a way a fresh read can fix: the loop observes and decides again instead of stopping.

    Bounded per round (RETRY_LIMIT); beyond that the failure becomes a Stop with the original reason.
    """


RETRY_LIMIT = 2


class Live:
    def __init__(self, a):
        self.a = a
        self.mode = a.mode
        self.gd = load()
        self.enc = Encoder(self.gd)
        self.plan = Plan(self.gd, self.enc.vocab.bond_ids)
        self.log_dir = LOGS
        self.resume_doc = self.read_checkpoint() if a.resume else None
        requested = a.policy.resolve() if a.policy else None
        saved_policy = Path(self.resume_doc.get("policy", CKPT)).resolve() if self.resume_doc else None
        if requested is not None and saved_policy is not None and requested != saved_policy:
            raise Stop(f"Saved match uses {saved_policy}; omit --policy to resume with it")
        self.policy = Policy(ckpt=requested or saved_policy or CKPT.resolve(), threads=4)
        self.check_policy_width()
        self.rules = Rules(self.gd)
        self.track = Track(self.gd, legacy_equip_snapshot=False)
        self.rng = random.Random(a.seed)
        self.state = GameState()
        self.initial_holding_ids = dict(a.holding_id)
        self.force_holdings_scan = bool(getattr(a, "rescan_holdings", False) or self.initial_holding_ids)
        self._retries: dict = {}  # (round, key) -> soft failures so far
        self._shop_open = False  # the expanded shop was verified and nothing since may have collapsed it
        self._prep_chain = False  # the last iteration acted in preparation, so the next one may skip the observe
        self._connect()
        self.sc = Sidecar()

    # ---------- device ----------
    def _connect(self):
        from maa.define import MaaAdbInputMethodEnum
        from maa.tasker import Tasker

        from .screen.capture import ADB_ADDRESS, ADB_PATH, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
        from .screen.read import CATALOG, Reader, read_json
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        self.catalog_dir = CATALOG
        self.controller = connect(self.a.adb or ADB_PATH, self.a.adb_address or ADB_ADDRESS,
                                  input_methods=MaaAdbInputMethodEnum.Maatouch,
                                  mumu=None if self.a.no_mumu else MUMU_PATH)
        self.tasker = load_ocr(self.controller, OCR_RESOURCE)
        self.slots = read_json(ROOT / "calib/slots.json")
        self.reader = Reader(CATALOG, self.slots)
        self.stages = read_json(CATALOG / "stages.json")
        if self.a.stage:  # the terrain template is covered once operators stand on the board; name the map by hand
            if self.a.stage not in self.stages:
                raise Stop(f"unknown --stage {self.a.stage}")
            self.state.stage_id = self.reader.stage_hint = self.a.stage

    def board_calibration(self, stage_id):
        from .screen.deploy import read_board
        if not isinstance(stage_id, str) or stage_id not in self.stages:
            raise Stop(f"stage is not identified: {stage_id} (no unique terrain match; pass --stage if the stage is known)")
        if self.state.round is not None and self.state.round in (self.gd.boss_round, self.gd.hidden_round):
            board = self.leader_calibration(stage_id)
            if board is None:
                raise Stop(f"no solo leader preparation calibration for {stage_id}")
            return board
        directory = ROOT / "calib"
        for name in (f"board_{stage_id}.json", f"board.{stage_id.rsplit('_', 1)[-1]}.json", "board.json"):
            path = directory / name
            if path.is_file():
                board = read_board(path)
                if board.get("stageId") == stage_id:
                    return board
        raise Stop(f"no board calibration for {stage_id}")

    def leader_calibration(self, stage_id):
        from .screen.deploy import read_board

        if stage_id not in self.stages:
            return None
        path = ROOT / "calib" / f"board.{stage_id.rsplit('_', 1)[-1]}.bossLeft.json"
        if path.is_file():
            board = read_board(path)
            if board["stageId"] == stage_id and board.get("deployField") == "bossLeft":
                return board
        return None

    # ---------- logging / resume ----------
    def game_checkpoints(self):
        return sorted(((int(directory.name[4:]), directory) for directory in self.log_dir.glob("game*")
                       if directory.is_dir() and directory.name[4:].isdigit()), reverse=True)

    def game_dir(self):
        d = self.log_dir / f"game{self.state.game:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log(self, row: dict):
        row = {"t": time.time(), "mode": self.mode, **row}
        with open(self.game_dir() / "decisions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps({k: row[k] for k in row if k in ("screen", "round", "decision", "action", "result", "error")},
                         ensure_ascii=False, default=str), flush=True)

    def save(self):
        save_checkpoint(self.game_dir() / "state.json", self.state, self.track, self.policy.checkpoint)

    def read_checkpoint(self):
        for _, directory in self.game_checkpoints():
            path = directory / "state.json"
            if path.exists():
                return load_checkpoint(path)
        return None

    def resume(self, doc):
        saved = dict(doc["state"])
        if pager := saved.pop("pending_pager", None):
            saved.setdefault("pending_equipment_reward", {**pager, "kind": "chess", "effect": "use_equip_spe_goods_first_bond"})
        self.state = GameState(**saved)
        self.reader.stage_hint = self.state.stage_id
        self.reader.deploy_cap_floor = self.state.deploy_cap_floor
        kind = (self.state.pending_equipment_reward or {}).get("kind")
        self.reader.reward_kind_hint = None if kind == "command" else kind
        if self.state.mode_id is not None:
            self.use_mode(self.state.mode_id)
        if restore_track(self.track, doc):
            if self.state.holdings_round is None:
                self.mark_holdings_dirty("hand", "temp", "board")
        else:
            self.state.prep_round = None
            self.state.start_uids = []
        if action := recover_action(self.state):
            self.save()
            self.log({"screen": "resume", "round": self.state.round, "decision": "interrupted action", "result": action})
        print(f"resumed game {self.state.game} round {self.state.round}", flush=True)

    def screenshot(self, name):
        try:
            from PIL import Image
            from .screen.capture import capture
            img = capture(self.controller)
            Image.fromarray(img[:, :, ::-1]).save(self.game_dir() / f"{name}-{int(time.time())}.png")
        except Exception as e:  # noqa: BLE001 — a failed screenshot must not hide the stop reason
            print(f"screenshot failed: {e}", flush=True)

    # ---------- screen ----------
    def observe(self):
        from .screen.flow import observe_snapshot
        from .screen.state import LIVE_SCREENS
        for k in range(UNKNOWN_RETRIES):
            self._observation = observe_snapshot(self.tasker)
            if self._observation["screen"] in LIVE_SCREENS:
                return LIVE_SCREENS[self._observation["screen"]]
            time.sleep(1.5 * (k + 1))
        return "unknown"

    def read(self):
        from .screen.read import read_screen
        state = read_screen(self.tasker, self.reader)
        if stage := state["match"].get("stageId"):
            self.state.stage_id = self.reader.stage_hint = stage
        elif self.state.stage_id and state["match"].get("round") is not None:
            # The terrain template is covered by operators standing on it; the map cannot change inside a match.
            state["match"]["stageId"] = self.state.stage_id
        own = state["self"]
        if (self.state.tracked_deploy_count_round is not None
                and self.state.tracked_deploy_count_round == state["match"]["round"]
                and "board" not in self.state.dirty_areas and own["deployCount"] is not None):
            remaining = own["deployCap"] - own["deployCount"]
            own["deployCount"] = self.track.deployed_chess_count()
            own["deployCap"] = own["deployCount"] + remaining
        return state

    def flow(self, action, **kw):
        from .screen.flow import execute
        return execute(self.tasker, self.slots, action, self.catalog_dir, kw.pop("bond", None), TIMEOUT_MS, **kw)

    def acting(self) -> bool:
        return self.mode in ("confirm", "auto")

    def can_inspect_panels(self) -> bool:
        return self.acting() or self.a.inspect_panels

    def check_policy_width(self):
        if self.enc.dim + self.plan.width != self.policy.dim or self.enc.A != self.policy.actions:
            raise Stop("The policy checkpoint input or action width does not match the live encoder")

    def use_mode(self, mode_id):
        if mode_id not in LIVE_MODES:
            raise Stop(f"Harness supports regulation 10–20; screen reports {mode_id or 'unreadable mode'}")
        if mode_id != self.gd.mode_id:
            self.gd = self.gd.for_mode(mode_id)
            self.enc = Encoder(self.gd)
            self.plan = Plan(self.gd, self.enc.vocab.bond_ids)
            self.rules = Rules(self.gd)
            self.track.gd = self.gd
            self.check_policy_width()
        self.state.mode_id = mode_id

    def probe(self) -> dict:
        """Check the real device and available prerequisites with zero input."""
        snapshot = self.flow("observe")
        result = {"screen": snapshot, "supportedRegulation": [10, 20],
                  "policy": {"checkpoint": str(self.policy.checkpoint), "device": "cpu",
                             "dataId": self.policy.spec["dataId"], "obs": self.policy.dim,
                             "actions": self.policy.actions}}
        if snapshot["screen"] == "preparation":
            state = self.read()
            stage = state["match"].get("stageId")
            result.update({"mode": state["config"].get("modeId"), "stage": stage,
                           "funds": state["self"].get("funds")})
            try:
                result["calibration"] = self.board_calibration(stage)["stageId"]
            except Stop as error:
                result["calibrationError"] = str(error)
            result["modeSupported"] = result["mode"] in LIVE_MODES
        return result

    # ---------- state assembly ----------
    def screen_state(self, s: dict, *, include_overflow: bool = False) -> dict:
        """The ScreenState the policy sees: the screen read + bookkeeping (holdings, band, group) + rebuilt preview."""
        st = json.loads(json.dumps(s))
        S, M = st["self"], st["match"]
        if self.state.mode_id and not st["config"].get("modeId"):
            st["config"]["modeId"] = self.state.mode_id
        t = self.track.to_screen(include_overflow=include_overflow)["self"]
        S["hand"], S["temp"], S["board"] = t["hand"], t["temp"], t["board"]
        S["bandId"] = S.get("bandId") or self.state.band_id
        S["playerId"] = S.get("playerId") or "ai_0"
        if self.state.bond_choice:
            M["bondChoice"] = self.state.bond_choice
        M["bossId"] = M.get("bossId") or self.state.boss_id
        M["hiddenBossId"] = M.get("hiddenBossId") or self.state.hidden_boss_id
        M["spRound"] = bool(M.get("spRound"))
        S.setdefault("personalChoice", None)
        if S["shop"].get("refreshPrice") == 0 and self.state.tactic_free_refreshes:
            S["shop"]["freeRefreshes"] = self.state.tactic_free_refreshes
        if self.state.preview is not None and self.state.preview_round == M.get("round"):
            S["nextEnemies"] = [{k: e[k] for k in ("enemyKey", "count", "fly", "elite", "boss", "gate")} for e in self.state.preview]
        S["nextEnemies"] = S.get("nextEnemies") or []
        S["bounties"] = self.track.active_bounties(M["round"]) if M.get("round") else []
        S["botSelf"] = self.state.bot_self
        return st

    def preview(self, s: dict):
        """Once per round: the enemy kinds from the preview panel → the full preview rebuilt from the template."""
        rnd = s["match"]["round"]
        if self.state.preview_round == rnd:
            return
        leader = rnd in (self.gd.boss_round, self.gd.hidden_round)
        bounties = self.track.active_bounties(rnd)
        r = self.flow("inspect-enemies", scroll_preview=leader or bool(bounties))
        kinds = r.get("kinds") or []
        if r.get("ambiguousEnemyNames"):
            raise Stop(f"ambiguous enemy names in the preview: {r['ambiguousEnemyNames']}")
        panel_match = (r.get("inspection") or {}).get("match", {})
        if leader:
            hidden = rnd == self.gd.hidden_round
            key = "hiddenBossId" if hidden else "bossId"
            boss_id = panel_match.get(key) or (self.state.hidden_boss_id if hidden else self.state.boss_id)
            if not boss_id:
                raise Stop("The leader name was not verified in the enemy preview")
            if hidden:
                self.state.hidden_boss_id = boss_id
            else:
                self.state.boss_id = boss_id
        else:
            boss_id = self.state.boss_id or s["match"].get("bossId")
        state = {"config": {"modeId": self.gd.mode_id}}
        inputs = {"round": rnd, "bossId": boss_id, "bounties": bounties,
                  "factions": s["match"].get("factions")}
        try:
            p = self.sc.request("preview", state, kinds=kinds, **inputs)
        except SidecarError as error:
            if bounties or not str(error).startswith("preview: no template pick matches"):
                raise
            extra = self.flow("inspect-enemies", scroll_preview=True)
            kinds = sorted(set(kinds) | set(extra.get("kinds") or []))
            p = self.sc.request("preview", state, kinds=kinds, **inputs)
        if p.get("approx"):
            print("preview: approximate match (kinds outside the template; bounty bookkeeping?)", flush=True)
        self.state.preview, self.state.preview_round = p["preview"], rnd
        if leader:
            self.state.leader_template = p.get("templateId")

    def encode(self, st: dict, kind: str):
        rnd = st["match"]["round"]
        new = sorted({u for u in self._uids()} - set(self.state.start_uids)) if kind == "prep" else []
        ctx = Ctx(kind=kind, actions=self.state.actions, freezes=self.state.freezes, lp0=self.state.lp0 or 0,
                  new_uids=set(new), sold_bases=set(self.state.sold_bases))
        obs = np.zeros(self.enc.dim + self.plan.width, np.float32)
        for e, pe in zip(st["self"]["nextEnemies"], self.state.preview or []):
            e["t"] = pe.get("t", 0)  # the rebuilt spawn time (template)
        _, mask = self.enc.encode(st, ctx, obs[: self.enc.dim])
        obs[self.enc.dim:] = self.plan.obs(st)
        vals = {}
        if kind == "card":
            for i, c in enumerate((st["match"].get("sp") or {}).get("cards") or []):
                if c and c.get("kind") == "bounty":
                    vals[("card", i)] = self.sc.request("killChance", st, card=c)["p"]
        self.enc.fill_kill_chance(obs, kind, st, vals)
        del rnd
        return obs, mask, ctx

    def _uids(self):
        return {p.uid for p in [*self.track.hand, *self.track.temp, *self.track.board.values()] if p is not None}

    def decide(self, st, kind, *, only: slice | None = None):
        t0 = time.perf_counter()
        obs, mask, ctx = self.encode(st, kind)
        if kind == "prep" and only is None:
            mask_unverified_prep_actions(mask, st["self"], self.state.blocked_board_sales)
        if only is not None:
            allowed = np.zeros_like(mask)
            allowed[only] = mask[only]
            mask = allowed
            if not mask.any():
                raise Stop("No visible reward can be taken")
        a = int(self.policy.act(obs[None], mask[None])[0])
        legal = [action_name(i) for i in np.nonzero(mask)[0]]
        row = {"screen": kind, "round": st["match"]["round"], "decision": kind, "action": action_name(a),
               "legal": legal, "obsHash": hashlib.sha1(obs.tobytes()).hexdigest()[:12], "ctx": ctx.__dict__,
               "state": st, "inferMs": round((time.perf_counter() - t0) * 1000, 1)}
        return a, row

    # ---------- match start ----------
    def new_match(self, *, restart=False):
        """From the 卫戍协议 entrance: briefing → bond group (rules) → commander (band rule) → round 1 preparation."""
        if not self.acting():
            raise Stop("observe mode does not start a match; open one by hand")
        from .screen.briefing import inspect_briefing
        if restart:
            if self.flow("observe")["screen"] == "gameHome":
                self.flow("open-entrance")
            self.flow("interrupt-match", confirm_interrupt=True)
        self.flow("enter-layout", stop_at="briefing")
        b = inspect_briefing(self.tasker, self.reader)
        if b.get("modeId"):
            self.use_mode(b["modeId"])
        groups = (b.get("bondChoice") or {}).get("groups") or b.get("groups")
        if not groups or len(groups) != 2:
            raise Stop(f"briefing groups not read: {b}")
        cfg = {"modeId": self.gd.mode_id, "difficulty": self.gd.difficulty, "lastRound": self.gd.last_round,
               "bossRound": self.gd.boss_round, "hiddenRound": self.gd.hidden_round}
        s = {"config": cfg, "match": {"round": 0, "disabledBonds": b.get("disabledBonds") or [], "bossId": b.get("bossId"),
                                     "bondChoice": {"groups": groups, "picked": None}}, "self": {"hand": [], "temp": [], "board": [], "bonds": []}}
        group = self.rules.bond_group(s)
        s["match"]["bondChoice"]["picked"] = group
        outside_groups = set(s["match"]["disabledBonds"]) - set(groups[0]) - set(groups[1])
        s["match"]["disabledBonds"] = sorted(outside_groups | set(groups[1 - group]))
        bw = self.rules.band_weights(s)
        band = bw.get("forced") or self.rules.band_pick(bw, self.rng)
        weights = dict(bw["pairs"])
        passive = {}
        for key in sorted(self.gd.band_ids(), key=lambda key: weights.get(key, 1), reverse=True):
            info = self.gd.band(key)
            skills = info.get("skills") or []
            if weights.get(key, 1) > 0 and skills and all(skill.get("type") == "PASSIVE" for skill in skills):
                passive[info["name"]] = key
        name = (self.gd.band(band) or {}).get("name")
        if not name:
            name = next(iter(passive), None)
        if not name:
            raise Stop("No catalog commander is available")
        r = self.flow("enter-layout", bond_group=group, commander=name,
                      commander_fallbacks=tuple(passive) if self.mode == "auto" else ())
        selected = r["commander"]["commander"]
        if selected != name or not band:
            band = passive[selected]
        game = max((index for index, _ in self.game_checkpoints()), default=-1) + 1
        self.state = GameState(game=game, band_id=band, bond_choice={"groups": groups, "picked": group},
                               lp0=self.gd.start_lp(band), boss_id=b.get("bossId"), mode_id=self.gd.mode_id)
        self.reader.stage_hint = None
        self.reader.deploy_cap_floor = 0
        self.reader.reward_kind_hint = None
        self.track = Track(self.gd, legacy_equip_snapshot=False)
        self.log({"screen": "briefing", "decision": "match start", "action": {"group": group, "band": band, "weights": bw},
                  "result": r})
        self.save()

    # ---------- combat drafts ----------
    def draft(self, *, tactic=False):
        kind = "tactic" if tactic else "bounty"
        screen = "tacticSelection" if tactic else "specialSelection"
        s = self.read()
        if s["config"].get("modeId") is not None:
            self.use_mode(s["config"]["modeId"])
        if s["match"].get("round") is None:
            if self.state.round is None:
                raise Stop(f"The {kind} draft covers the round HUD and the previous round is unknown")
            s["match"]["round"] = self.state.round + 1
        s["match"]["stageId"] = self.state.stage_id
        st = self.screen_state(s)
        cards = (st["match"].get("sp") or {}).get("cards") or []
        if not any(cards) or (tactic and not all(cards)):
            self.soft_fail(kind, f"The {kind} cards could not be read")
        a, row = self.decide(st, "card")
        i = a - ACT["CARD"]
        if not self.acting():
            self.log({**row, "result": "observe: not picked"})
            return self.wait_change(screen)
        card = cards[i]
        buffs = self.gd.raw.get("effects", {}).get(card["id"], {}).get("buffs", []) if tactic else []
        keys = {buff["key"] for buff in buffs}
        areas = ()
        if "single_special_choice_gain_bond_chess" in keys:
            areas = ("hand", "board", "temp")
        elif "global_special_choice_gain_equip" in keys:
            areas = ("temp",)
        with device_action(self.state, self.save, kind, areas):
            rec = self.flow(f"select-{kind}", slot=card["idx"], expected_id=card["id"], expected_round=st["match"]["round"])
            if tactic:
                self.mark_holdings_dirty(*areas)
                for buff in buffs:
                    count = buff.get("bb", {}).get("count", 1)
                    if buff["key"] == "global_special_choice_refresh_free":
                        self.state.tactic_free_refreshes += count
                    elif buff["key"] == "single_special_choice_gloden_char_chess":
                        self.state.tactic_upgrades["chess"] = count
                    elif buff["key"] == "single_special_choice_gloden_equip_chess":
                        self.state.tactic_upgrades["item"] = count
            else:
                self.track.record_bounty(rec)
            self.state.round = rec["round"]
        self.log({**row, "decision": "tactic draft" if tactic else row["decision"], "result": rec})

    def supply(self):
        if not self.acting():
            return self.wait_change("equipmentSelection")
        s = self.read()
        offers = s.get("equipmentOffers") or []
        if not offers:
            self.soft_fail("supply", "No equipment offer could be identified")
        rnd = s["match"]["round"]
        if rnd is None and self.state.round is not None:
            rnd = self.state.round + 1
        if rnd is None:
            raise Stop("The supply draft covers the round HUD and no previous round is saved")
        offer = max(offers, key=lambda card: (
            not (card["kind"] or "").startswith("consume_on_equip"), card["tier"] or 0, -card["idx"],
        ))
        with device_action(self.state, self.save, "supply", ("temp",)):
            receipt = self.flow("select-equipment", slot=offer["idx"], expected_id=offer["id"], expected_round=rnd)
            self.state.round = receipt["round"]
            self.mark_holdings_dirty("temp")
        self.log({"screen": "equipmentSelection", "round": rnd, "decision": "supply draft", "result": receipt})

    # ---------- soft failures ----------
    def _spend_retry(self, key) -> int:
        slot = (self.state.round, key)
        self._retries[slot] = self._retries.get(slot, 0) + 1
        return self._retries[slot]

    def soft_fail(self, key, reason):
        """Verification could not be satisfied: re-read and decide again (bounded), instead of stopping the match."""
        used = self._spend_retry(key)
        if used > RETRY_LIMIT:
            raise Stop(f"{reason} (still failing after {RETRY_LIMIT} re-reads)")
        self._shop_open = self._prep_chain = False
        raise Retry(f"{reason} (re-read {used}/{RETRY_LIMIT})")

    def uncertain_receipt(self, receipt, action, st):
        """The click was verified on screen but bookkeeping disagrees: trust the screen, mark the area for a re-read."""
        self.note_uncertain_action(action, st)
        return {**receipt, "bookkeeping": "uncertain; affected holdings are re-read before the next decision"}

    # ---------- preparation ----------
    def mark_holdings_dirty(self, *areas):
        self.state.dirty_areas = sorted(set(self.state.dirty_areas).union(areas))

    def note_track_gifts(self):
        for _, effect, _ in self.track.unpredicted + self.track.guessed:
            self.mark_holdings_dirty("temp" if effect in ITEM_GRANTS else "hand")

    def chess_gain_areas(self, chess_id):
        areas = {"hand"}
        chess = self.gd.chess(chess_id) or {}
        effects = [self.gd.garrison(gid) or {} for gid in chess.get("garrisonIds") or []]
        if any(e.get("eventType") == "SERVER_GAIN"
               and (e.get("effectKey") or e.get("effectType")) in ITEM_GRANTS for e in effects):
            areas.add("temp")
        if self.track._completes_chess_merge(chess_id):
            for loc in self.track._chess_locs():
                if loc["piece"].id == chess_id:
                    if loc["area"] == "board":
                        areas.add("board")
                    if loc["piece"].items:
                        areas.add("temp")
        return tuple(sorted(areas))

    def note_uncertain_action(self, action, st):
        self.mark_holdings_dirty(*self.action_areas(action, st))

    def action_areas(self, action, st):
        if ACT["BUY"] <= action < ACT["SELL"]:
            slots = st["self"]["shop"].get("slots") or []
            index = action - ACT["BUY"]
            offer = slots[index] if index < len(slots) else None
            if not offer or not offer.get("id"):
                return ()
            if offer["kind"] == "item":
                return ("temp",)
            return self.chess_gain_areas(offer["id"])
        if ACT["SELL"] <= action < ACT["REWARD"]:
            piece, area, _ = self.owned(action - ACT["SELL"])
            if piece:
                return (area, "temp") if piece.get("items") and area != "temp" else (area,)
        return ()

    def apply_confirmed_gain(self, action, st, chess_id):
        sources = self.track.merge_board_sources(chess_id) if chess_id else {}
        if not self.track.apply(action, st, after_buy=self.apply_tactic_purchase):
            return False
        if len(sources) > 1:
            self.state.pending_board_merges.append({"id": self.gd.golden_id_of(chess_id), "sources": sources})
        return True

    def apply_tactic_purchase(self, piece):
        remaining = self.state.tactic_upgrades.get(piece.kind, 0)
        if not remaining or self.gd.is_golden(piece.id):
            return
        if piece.kind == "chess":
            upgraded = self.gd.golden_id_of(piece.id)
        else:
            item = self.gd.item(piece.id) or {}
            upgraded = item.get("upgradeChessId") or item.get("goldenId")
            if not self.gd.item(upgraded):
                return
            self.mark_holdings_dirty("temp")
        if upgraded:
            piece.id = upgraded
            self.track._recompute()
            self.state.tactic_upgrades[piece.kind] = remaining - 1

    def verify_board_merges(self, s):
        from .screen.board_read import find_board_piece
        from .screen.shop import set_shop_open

        own, match = s["self"], s["match"]
        fields = (match["round"], own["funds"], own["deployCap"], own["deployCount"])
        if any(type(value) is not int for value in fields):
            raise RuntimeError("Preparation HUD could not anchor a merged board operator")
        rnd, funds, cap, count = fields
        board = self.board_calibration(match["stageId"])
        set_shop_open(self.tasker, False)
        self._shop_open = False
        for pending in self.state.pending_board_merges[:]:
            sources, chess_id = pending["sources"], pending["id"]
            predicted = [key for key in sources if key in self.track.board and self.track.board[key].id == chess_id]
            if len(predicted) != 1 or any(key in self.track.board for key in sources if key != predicted[0]):
                raise RuntimeError(f"Cannot isolate the predicted merge of {chess_id}")
            old_key = predicted[0]
            keys = [old_key, *(key for key in sources if key != old_key)]
            tiles = [tuple(map(int, key.split(","))) for key in keys]
            tile = find_board_piece(self.tasker, self.reader, board, tiles, chess_id, (rnd, funds, cap - count))
            key = f"{tile[0]},{tile[1]}"
            piece = self.track.board[old_key]
            if key != old_key:
                self.track.move(piece.uid, tile, sources[key])
            self.state.pending_board_merges.remove(pending)
            self.save()
            self.log({"screen": "preparation", "round": rnd, "decision": "board merge",
                      "result": {"id": chess_id, "predicted": old_key, "observed": key}})
        set_shop_open(self.tasker, True)
        self._shop_open = True

    def board_ledger(self, rnd):
        layout = {tuple(map(int, tile.split(","))): (self.name(piece.id), piece.dir)
                  for tile, piece in self.track.board.items()}
        step = self.state.layout_step
        if step and step["round"] == rnd and step["target"] is not None:
            layout.setdefault(tuple(step["target"]), (self.name(step["id"]), None))
        return layout

    def refresh_holdings(self, s, board):
        """Carry the ledger into prep and read only areas affected by unknown changes."""
        from .screen.holdings import reconcile_holdings, scan_holdings

        rnd = s["match"]["round"]
        previous = self.state.holdings_round or self.state.preview_round
        plan = prep_read_plan(self.track, self.state.band_id, rnd, previous, s["self"].get("deployCount"),
                              force=self.force_holdings_scan)
        for area in self.state.dirty_areas:
            plan[area] = "uncertain change in this area"
        if previous == rnd - 1 and "board" not in plan:
            self.track.replenish_summons()
        scan_ms = 0
        rack_slots = None
        if plan:
            scheduled = None
            if plan.get("temp") == "scheduled fixed equipment":
                scheduled = scheduled_rack_slot(self.track, self.state.band_id, rnd)
                if scheduled is not None:
                    rack_slots = [scheduled[0]]
            layout = self.board_ledger(rnd) if "board" in plan else None
            known_summons = tuple(i for i, p in enumerate(self.track.hand) if p and p.kind == "token")
            scan = scan_holdings(self.tasker, self.reader, board, areas=set(plan), rack_slots=rack_slots,
                                 known_layout=layout, known_summons=known_summons)
            if scheduled is not None:
                observed = scan["temp"][0]
                if observed.get("id") != scheduled[1]:
                    self.mark_holdings_dirty("temp")
                    rack_slots = None
                    plan["temp"] = "fixed equipment did not match predicted slot"
                    rack = scan_holdings(self.tasker, self.reader, board, areas={"temp"})
                    if (rack["round"], rack["funds"]) != (scan["round"], scan["funds"]):
                        self.soft_fail("holdings", "preparation round or funds changed during rack refresh")
                    scan.update(temp=rack["temp"], rackSlots=None, totalMs=scan["totalMs"] + rack["totalMs"])
            if (scan["round"], scan["funds"]) != (rnd, s["self"]["funds"]):
                self.soft_fail("holdings", "preparation round or funds changed during holdings refresh")
            rec = reconcile_holdings(self.track, scan, deploy_cap=s["self"]["deployCap"],
                                     confirmed_ids=self.initial_holding_ids)
            if rec["issues"]:
                self.log({"screen": "preparation", "round": rnd, "decision": "calibration", "result": rec})
                self.soft_fail("reconcile", f"holdings did not reconcile: {rec['issues']}")
            if "board" in plan:
                self.state.layout_step = None
            scan_ms += scan["totalMs"]
        self.initial_holding_ids.clear()
        self.force_holdings_scan = False
        self.state.dirty_areas.clear()
        self.state.holdings_round = rnd
        self.track.guessed.clear()
        self.track.unpredicted.clear()
        return {"scanMs": scan_ms, "boardSource": "screen" if "board" in plan else "track", "facingSource": "track", "reads": plan,
                **({"rackSlots": rack_slots} if rack_slots is not None else {})}

    def prep_start(self, s):
        """Refresh holdings, run prep-start rules and determine the legacy focus."""
        rnd = s["match"]["round"]
        board = self.board_calibration(s["match"]["stageId"])
        holdings = self.refresh_holdings(s, board)
        self.state.actions, self.state.freezes, self.state.sold_bases = 0, 0, []
        self.state.blocked_board_sales.clear()
        self.state.start_uids = sorted(self._uids())
        self.save()
        self.preview(s)
        st = self.screen_state(s)
        cmd = self.rules.command_start(st)
        if cmd and "effectId" in cmd:
            raise Unsupported(f"prep-start command {cmd['effectId']} (active command executor not wired)")
        if self.acting() and not s["self"]["shop"].get("rewardOffer"):
            for operation in self.rules.pager(st):
                carrier = self.track._loc_by_uid(operation["carrier"])
                if carrier and carrier["area"] == "board":
                    self.apply_equipment({"item": operation["item"], "target": operation["carrier"]}, board, rnd)
                    self.finish_reward(self.read())
                    st = self.screen_state(self.read())
        self.state.bot_self = self.sc.request("selfPlan", st)["botSelf"]
        self.state.prep_round = rnd
        self.log({"screen": "preparation", "round": rnd, "decision": "prep start",
                  "result": {**holdings, "botSelf": self.state.bot_self}})
        self.save()

    def prep(self):
        if not self.can_inspect_panels():
            raise Stop("Full preparation input requires opening the holdings and enemy panels")
        from .screen.shop import set_shop_open
        chained, self._prep_chain = self._prep_chain, False
        shop_view = None
        if not self._shop_open:
            shop_view = set_shop_open(self.tasker, True)
            self._shop_open = True
        s = self.read()
        rnd = s["match"]["round"]
        if chained and (rnd is None or rnd != self.state.round or s["self"].get("funds") is None
                        or s["self"]["shop"].get("refreshPrice") is None):
            self._shop_open = False  # not provably the same expanded preparation: observe again, no input sent
            return
        if shop_view is not None and rnd != shop_view["after"]["round"]:
            self.soft_fail("round", "the preparation round changed while opening the shop")
        self.state.round = rnd
        self.use_mode(s["config"].get("modeId"))
        step = self.state.layout_step
        if step and "uid" in step:
            from .layout_recovery import recover_layout_step

            result = recover_layout_step(self.tasker, self.reader, self.board_calibration(s["match"]["stageId"]),
                                         self.track, step, s)
            self.state.layout_step = None
            self.save()
            self.log({"screen": "preparation", "round": rnd, "decision": "layout step recovery", "result": result})
        if getattr(self.a, "stop_at_leader_prep", False) and rnd in (self.gd.boss_round, self.gd.hidden_round):
            set_shop_open(self.tasker, False)
            self.save()
            self.screenshot("leader-prep")
            self.log({"screen": "preparation", "round": rnd, "decision": "leader calibration checkpoint"})
            return True
        if self.state.pending_equipment_reward and self.acting():
            self.finish_reward(s)
            s = self.read()
        if s["self"]["shop"].get("rewardOffer"):
            self.finish_reward(s)
            if not self.acting() and not self.a.once:
                return self.wait_change("preparation", rnd)
            return
        if self.state.pending_board_merges:
            self.verify_board_merges(s)
            s = self.read()
        if rnd in (self.gd.boss_round, self.gd.hidden_round):
            calibration = self.leader_calibration(s["match"]["stageId"])
            if not calibration or not calibration["views"].get("drag"):
                return self.prep_uncalibrated_leader(s)
        if self.state.lp0 is None:
            self.state.band_id = self.state.band_id or self.command_band()
            self.state.lp0 = self.gd.start_lp(self.state.band_id)
        if self.state.prep_round != rnd:
            self.prep_start(s)
            set_shop_open(self.tasker, True)
            self._shop_open = True
            s = self.read()
            if s["match"]["round"] != rnd:
                self.soft_fail("round", "the preparation round changed before the policy decision")
        elif self.force_holdings_scan or self.state.dirty_areas:
            holdings = self.refresh_holdings(s, self.board_calibration(s["match"]["stageId"]))
            self.log({"screen": "preparation", "round": rnd, "decision": "holdings refresh", "result": holdings})
            self.save()
        if self.acting() and self.equip_income_dolls(s):
            if self.state.dirty_areas:
                return
            s = self.read()
        st = self.screen_state(s)
        a, row = self.decide(st, "prep")
        if not self.acting():
            self.log({**row, "result": "observe: not executed"})
            if self.a.once:
                return
            return self.wait_change("preparation", rnd)
        if a == ACT["END"]:
            if self.a.stop_before_end:
                self.log({**row, "result": "waiting for END confirmation"})
                self.save()
                return True
            if self.mode == "confirm" and not self.a.stop_after_layout:
                input(f"round {rnd}: END flow next (layout, equipment, temp, ready). Enter to continue, Ctrl+C to stop: ")
            return self.end_flow(st, row)
        with device_action(self.state, self.save, action_name(a), self.action_areas(a, st)):
            receipt = self.execute_action(a, st)
            self.note_track_gifts()
            self.state.actions += 1
            if a == ACT["FREEZE"]:
                self.state.freezes += 1
        self.log({**row, "result": receipt})
        self._prep_chain = True

    def command_band(self):
        from .screen.command import inspect_command
        c = inspect_command(self.tasker, self.catalog_dir)
        if not c.get("bandId"):
            raise Stop("band not known and not readable from the command panel")
        return c["bandId"]

    def owned(self, i):
        """Owned row i in the encoder's order (hand 0-9, temp 10-14, board in reading order)."""
        t = self.track.to_screen()["self"]
        rows = (t["hand"] + [None] * HAND)[:HAND] + (t["temp"] + [None] * TEMP)[:TEMP] + t["board"]
        return rows[i] if i < len(rows) else None, ("hand" if i < HAND else "temp" if i < HAND + TEMP else "board"), i

    def name(self, pid):
        rec = self.gd.chess(pid) or self.gd.item(pid) or self.gd.token(pid) or {}
        return rec.get("name") or pid

    def sell_owned(self, p, area, index, stage_id, round_number):
        price = (self.gd.chess(p["id"]) or {}).get("sellPrice")
        if area == "hand":
            from .screen.hand_sale import sell_hand
            return sell_hand(self.tasker, self.slots, index, self.name(p["id"]), expected_price=price)
        if area == "board":
            from .screen.sell import sell_board
            return sell_board(self.tasker, self.slots, self.board_calibration(stage_id), self.name(p["id"]),
                              p["row"], p["col"], expected_price=price, expected_round=round_number,
                              stage_confirmed=True)
        raise Unsupported(f"operator {p['uid']} has no verified sale position")

    def resolve_visible(self, operations, round_number, stage_id):
        from .screen.equip import destroy_item

        receipts = []
        for operation, uid, *args in operations:
            loc = self.track._loc_by_uid(uid)
            if operation == "move":
                area, idx = args
                if loc is None or loc["area"] != area or loc.get("idx") != idx:
                    raise Unsupported(f"automatic overflow refill for UID {uid} did not reconcile")
                continue
            if loc is None:
                raise Unsupported(f"{operation} for UID {uid} requires a visible source")
            index = loc.get("idx")
            if operation == "destroy" and loc["area"] == "temp" and loc["piece"].kind == "item":
                replacement = next((p.id for p in self.track.temp[TEMP:] if p and p.kind == "item"), None)
                with device_action(self.state, self.save, "destroy", ("temp",)):
                    receipt = destroy_item(self.tasker, self.reader, index, self.name(loc["piece"].id),
                                           expected_round=round_number, replacement_id=replacement)
                    self.track.destroy(uid)
            elif operation == "sell" and loc["piece"].kind == "chess":
                if loc["area"] == "board":
                    board_index, p = next((i, p) for i, p in enumerate(self.track.to_screen()["self"]["board"])
                                          if p["uid"] == uid)
                    index = HAND + TEMP + board_index
                elif loc["area"] == "hand":
                    p = self.track.to_screen()["self"]["hand"][index]
                else:
                    raise Unsupported(f"sale for UID {uid} requires hand or board")
                area = loc["area"]
                with device_action(self.state, self.save, "sell", (area, "temp") if p.get("items") else (area,)):
                    receipt = self.sell_owned(p, area, index, stage_id, round_number)
                    if not self.track._sell(index):
                        raise Stop(f"sale completed but bookkeeping rejected UID {uid}")
                    if area == "board":
                        self.state.tracked_deploy_count_round = round_number
            else:
                raise Unsupported(f"unsupported visible overflow action {operation} for UID {uid}")
            receipts.append({"op": operation, "uid": uid, "receipt": receipt})
        return receipts

    def execute_action(self, a, st):
        from .screen import shop
        sh = st["self"]["shop"]
        rnd = st["match"]["round"]
        if ACT["BUY"] <= a < ACT["SELL"]:
            i = a - ACT["BUY"]
            slots = sh.get("slots") or []
            sl = slots[i] if i < len(slots) else None
            if not sl or not sl.get("id") or type(sl.get("price")) is not int:
                raise Stop(f"buy slot {i} does not have a verified ID and price")
            merge = sl.get("kind") == "chess" and self.track._completes_chess_merge(sl["id"])
            r = shop.buy_slot(self.tasker, self.reader, i, expected_id=sl.get("id"), expected_price=sl.get("price"),
                              expected_round=rnd, merge=merge)
            if not self.apply_confirmed_gain(a, st, sl["id"] if sl["kind"] == "chess" else None):
                r = self.uncertain_receipt(r, a, st)
            if sl["kind"] == "item" and ((self.gd.item(sl["id"]) or {}).get("kind") or "").startswith("consume_on_equip"):
                self.mark_holdings_dirty("temp")
            return r
        if ACT["SELL"] <= a < ACT["REWARD"]:
            from .screen.sell import SaleNotSent

            p, area, i = self.owned(a - ACT["SELL"])
            if not p or p["kind"] != "chess":
                raise Stop(f"sell row {a - ACT['SELL']} is not an operator in the bookkeeping")
            try:
                r = self.sell_owned(p, area, i, st["match"]["stageId"], rnd)
            except SaleNotSent:
                if area == "board":
                    self.state.blocked_board_sales.append([p["uid"], p["row"], p["col"]])
                raise
            self.state.sold_bases.append(self.gd.base_id_of(p["id"]))
            if not self.track.apply(a, st):
                r = self.uncertain_receipt(r, a, st)
            if area == "board":
                self.state.tracked_deploy_count_round = rnd
            return r
        if a in (ACT["REFRESH"], ACT["LEVEL"], ACT["FREEZE"]):
            r = shop.change_shop(self.tasker, self.reader, {ACT["REFRESH"]: "refresh", ACT["LEVEL"]: "level", ACT["FREEZE"]: "freeze"}[a],
                                 frozen=True if a == ACT["FREEZE"] else None)
            self.track.apply(a, st)
            if a == ACT["REFRESH"] and r["before"]["funds"] == r["after"]["funds"]:
                self.state.tactic_free_refreshes = max(self.state.tactic_free_refreshes - 1, 0)
            return r
        if ACT["REWARD"] <= a < ACT["PERSONAL"]:
            raise Unsupported("reward pick (read / select chain not wired)")
        if ACT["PERSONAL"] <= a < ACT["CARD"]:
            raise Unsupported("personal choice (read / select chain not wired)")
        raise Stop(f"unexpected prep action {action_name(a)}")

    def equip_income_dolls(self, s):
        """Equip verified economy dolls before the next policy decision."""
        items = [p for p in self.track.temp[:TEMP] if p and p.kind == "item"
                 and self.rules.effect_key(p.id) in INCOME_DOLL_EFFECTS]
        if not items:
            return False
        carrier = next((p for p in (*self.track.board.values(), *self.track.hand)
                        if p and p.kind == "chess" and len(p.items) < 2), None)
        if carrier is None:
            return False
        rnd = s["match"]["round"]
        board = self.board_calibration(s["match"]["stageId"])
        for item in items:
            receipt = self.apply_equipment({"item": item.uid, "target": carrier.uid}, board, rnd)
            self.log({"screen": "prep", "round": rnd, "decision": "equip income doll", "result": receipt})
            if self.state.dirty_areas:
                break
        return True

    def apply_equipment(self, operation, board, rnd):
        from .screen.equip import equipment_funds_match, equip_to_board, equip_to_hand, verify_rack_source
        from .screen.equip_choice import choose_equipment_dialog

        item_uid, target_uid = operation["item"], operation["target"]
        source, target = self.track._loc_by_uid(item_uid), self.track._loc_by_uid(target_uid)
        if (source is None or source["area"] != "temp" or source["idx"] >= TEMP
                or target is None or target["area"] not in {"board", "hand"}):
            raise Unsupported("equipment requires a visible rack source and operator")
        item, carrier = source["piece"], target["piece"]
        name = self.name(item.id)
        replace = None
        if len(carrier.items) >= 2:
            requested = operation.get("replace")
            replace = next((it for it in carrier.items if it.uid == requested), None) if requested is not None else carrier.items[0]
            if replace is None or sum(self.name(it.id) == self.name(replace.id) for it in carrier.items) != 1:
                raise Unsupported("The equipment to replace is not uniquely identified")
        record = self.gd.item(item.id) or {}
        kind = record.get("kind") or ""
        params = record.get("params") or {}
        effect = next((b.get("key") for b in record.get("buffs") or []), None)
        reward_kind = EQUIPMENT_REWARDS.get(effect)
        round_income = self.rules.effect_key(item.id) in INCOME_DOLL_EFFECTS
        gain_range = None
        if effect == "equip_destory_gain_random_coin":
            gain_range = (params["min"], params["max"])
        elif effect == "equip_destory_gain_coin":
            gain_range = (max(params.get("value", 0), 0), None)
        consumes = bool(reward_kind or round_income or gain_range or effect in EQUIPMENT_MUTATIONS)
        if kind.startswith("consume_on_equip") and not consumes:
            raise Unsupported(f"equipment effect for {name}")
        if effect == "equip_tier_recr_gift" and (self.gd.chess(carrier.id) or {}).get("chessType") == "DIY":
            raise Unsupported("Beacon cannot be equipped to a custom operator")
        dirty = EQUIPMENT_MUTATIONS.get(effect, ())
        expected_gain = 0 if consumes and not reward_kind and gain_range is None else None
        if gain_range and gain_range[0] == gain_range[1] and gain_range[0] > 0:
            expected_gain, gain_range = gain_range[0], None
        areas = tuple(sorted({"temp", target["area"], *dirty}))
        opens_dialog = bool(reward_kind or replace)
        with device_action(self.state, self.save, "equip", areas):
            if target["area"] == "board":
                row, col = map(int, target["key"].split(","))
                receipt = equip_to_board(self.tasker, self.reader, board, source["idx"], name,
                                         row, col, self.name(carrier.id), expected_round=rnd,
                                         opens_choice=opens_dialog, expected_gain=expected_gain, gain_range=gain_range)
            else:
                receipt = equip_to_hand(self.tasker, self.slots, source["idx"], target["idx"], name,
                                        reader=self.reader, expected_round=rnd, opens_choice=opens_dialog,
                                        expected_gain=expected_gain, gain_range=gain_range)
            if replace:
                choice = choose_equipment_dialog(self.tasker, ("替换装备", "装备替换", "请选择要替换的装备"),
                                                 {self.name(replace.id): replace.id}, rnd)
                receipt["replaced"] = replace.id
                if not reward_kind:
                    receipt["after"] = {key: choice[key] for key in ("round", "funds")}
                    if not equipment_funds_match(choice["funds"] - receipt["before"]["funds"], expected_gain, gain_range):
                        raise RuntimeError("Equipment replacement did not produce the expected funds change")
                    verify_rack_source(self.tasker, self.reader, receipt)
                if consumes:
                    carrier.items.remove(replace)
            if reward_kind:
                self.state.pending_equipment_reward = {"item": item_uid, "target": target_uid, "kind": reward_kind,
                                                       "effect": effect, "rackSlot": source["idx"]}
                self.reader.reward_kind_hint = None if reward_kind == "command" else reward_kind
            elif consumes:
                if not self.track.destroy(item_uid):
                    self.mark_holdings_dirty("temp")
                if effect == "equip_use_deploy":
                    self.state.deploy_cap_floor = self.reader.deploy_cap_floor = max(
                        self.state.deploy_cap_floor, params.get("count", 9))
                self.mark_holdings_dirty(*dirty)
                receipt["consumed"] = True
                receipt["replan"] = effect in EQUIPMENT_MUTATIONS
            elif not self.track.equip(item_uid, target_uid, replace_uid=replace.uid if replace else None):
                self.mark_holdings_dirty(*areas)
            if (receipt.get("slotAfter") or {}).get("status") == "occupied":
                self.mark_holdings_dirty("temp")
        return {"item": item_uid, "target": target_uid, **receipt}

    def finish_command_reward(self, pending, rnd):
        from .screen.command import command_choice_names, inspect_command
        from .screen.equip_choice import choose_equipment_dialog

        with device_action(self.state, self.save, "command reward", ("temp",)):
            options = command_choice_names(self.catalog_dir)
            receipt = choose_equipment_dialog(self.tasker, ("更换指令", "选择指令", "指令选择"), options, rnd)
            actual = inspect_command(self.tasker, self.catalog_dir)
            if actual["round"] != rnd or actual["effectId"] != receipt["id"]:
                raise RuntimeError("The selected command could not be verified in its details")
            if not self.track.destroy(pending["item"]):
                self.mark_holdings_dirty("temp")
            self.state.pending_equipment_reward = None
        self.log({"screen": "prep", "round": rnd, "decision": "command reward", "result": {**pending, **receipt}})

    def finish_reward(self, s):
        from .screen.holdings import reconcile_holdings, scan_holdings
        from .screen.read import shop_reward_choices
        from .screen.shop import pick_reward

        pending = self.state.pending_equipment_reward
        reward_kind = pending["kind"] if pending else "chess"
        rnd = s["match"]["round"] or self.state.round
        if reward_kind == "command":
            return self.finish_command_reward(pending, rnd)
        choices = shop_reward_choices(s["self"]["shop"].get("slots") or [], kind=reward_kind)
        if pending and not choices:
            # The cards slide into their calibrated slots after the pager is used.
            deadline = time.monotonic() + 3
            while not choices and time.monotonic() < deadline:
                time.sleep(0.15)
                s = self.read()
                current_round = s["match"]["round"]
                if current_round is not None and current_round != rnd:
                    raise Stop("The round changed while waiting for the pager reward")
                if current_round == rnd:
                    choices = shop_reward_choices(s["self"]["shop"].get("slots") or [], kind=reward_kind)
            if not choices and s["match"]["round"] == rnd:
                choices = shop_reward_choices(s["self"]["shop"].get("slots") or [], min_cards=1, kind=reward_kind)
        if not choices:
            paid = [card for card in s["self"]["shop"].get("slots") or []
                    if card and card.get("id") and (card.get("price") or 0) > 0]
            if pending and s["match"]["round"] == rnd and len(paid) >= 2:
                return self.reconcile_closed_equipment_reward(pending, s)
            raise RuntimeError(f"No {reward_kind} reward was verified in the expanded shop")
        areas = {"hand"} & set(self.state.dirty_areas)
        layout = None
        if (not pending and "board" in self.state.dirty_areas
                and s["self"]["deployCount"] == self.track.deployed_chess_count()):
            areas.add("board")
            layout = self.board_ledger(rnd)
        if not pending and areas:
            board = self.board_calibration(s["match"]["stageId"])
            scan = scan_holdings(self.tasker, self.reader, board, areas=areas, known_layout=layout)
            if (scan["round"], scan["funds"]) != (rnd, s["self"]["funds"]):
                raise Stop("Preparation changed while checking holdings for the reward")
            checked = reconcile_holdings(self.track, scan, deploy_cap=s["self"]["deployCap"])
            if checked["issues"]:
                raise Stop(f"Reward holdings did not reconcile: {checked['issues']}")
            self.state.dirty_areas = [area for area in self.state.dirty_areas if area not in areas]
        st = self.screen_state(s)
        st["self"]["shop"].update(slots=[], rewardOffer={"slots": [offer for _, offer in choices]})
        action, row = self.decide(st, "prep", only=slice(ACT["REWARD"], ACT["PERSONAL"]))
        slot, offer = choices[action - ACT["REWARD"]]
        effect = pending.get("effect") if pending else None
        decision = ("pager reward" if effect == "use_equip_spe_goods_first_bond" else
                    "equipment reward" if pending else "recruitment reward")
        if not self.acting():
            self.log({**row, "decision": decision, "result": "observe: not executed"})
            return
        gain_areas = self.chess_gain_areas(offer["id"]) if reward_kind == "chess" else ("temp",)
        beacon = effect == "equip_tier_recr_gift"
        areas = ("board", "hand", "temp") if beacon else (*gain_areas, "temp") if pending else gain_areas
        with device_action(self.state, self.save, decision, areas):
            receipt = pick_reward(self.tasker, self.reader, slot, expected_id=offer["id"], expected_round=rnd,
                                  kind=reward_kind)
            if pending and not self.track.destroy(pending["item"]):
                self.mark_holdings_dirty("temp")
            if beacon:
                self.track.destroy_chess(pending["target"])
                self.mark_holdings_dirty(*areas)
            elif (not pending and "board" in self.state.dirty_areas
                    or not self.apply_confirmed_gain(action, st, offer["id"] if reward_kind == "chess" else None)):
                self.mark_holdings_dirty(*gain_areas)
            if reward_kind == "item":
                self.mark_holdings_dirty("temp")
            self.state.pending_equipment_reward = None
            self.reader.reward_kind_hint = None
            self.note_track_gifts()
        self.log({**row, "decision": decision, "result": {**(pending or {}), "receipt": receipt}})

    def reconcile_closed_equipment_reward(self, pending, s):
        from .screen.equip import verify_rack_source

        loc = self.track._loc_by_uid(pending["item"])
        if loc is None or loc["area"] != "temp":
            raise RuntimeError("The pending reward item has no known rack slot")
        rnd, funds = s["match"]["round"], s["self"]["funds"]
        receipt = {"itemSlot": pending.get("rackSlot", loc["idx"]), "name": self.name(loc["piece"].id),
                   "after": {"round": rnd, "funds": funds}}
        verify_rack_source(self.tasker, self.reader, receipt)
        self.track.destroy(pending["item"])
        if pending.get("effect") == "equip_tier_recr_gift":
            self.track.destroy_chess(pending["target"])
        areas = ("temp",) if pending["kind"] == "item" else ("board", "hand", "temp")
        self.mark_holdings_dirty(*areas)
        self.state.pending_equipment_reward = None
        self.reader.reward_kind_hint = None
        self.save()
        self.log({"screen": "prep", "round": rnd, "decision": "equipment reward already closed", "result": receipt})

    def prep_uncalibrated_leader(self, s):
        rnd = s["match"]["round"]
        own = s["self"]
        count = own["deployCount"]
        if not own["canReady"] or not count or count != self.track.deployed_chess_count() or "board" in self.state.dirty_areas:
            raise Stop("Could not verify the deployed formation for the uncalibrated leader round")
        row = {"screen": "preparation", "round": rnd, "decision": "leader round", "deployed": count}
        if self.a.stop_before_end or self.a.stop_after_layout:
            self.log(row)
            return True
        if not self.acting():
            self.log(row)
            return self.a.once or self.wait_change("preparation", rnd)
        if self.mode == "confirm":
            input(f"round {rnd}: start battle with the deployed formation. Enter to continue, Ctrl+C to stop: ")
        self.log({**row, "result": self.ready_battle()})

    def ready_battle(self):
        with device_action(self.state, self.save, "ready"):
            receipt = self.flow("ready-battle")
            self.state.prep_round = None
        return receipt

    def plan_end(self, st, receipts):
        """Make room for the sidecar's chosen formation using verified visible operations."""
        match = st["match"]
        stage, rnd = match["stageId"], match["round"]
        for _ in range(len(self._uids()) + 1):
            plan = self.sc.request("end", st)
            room = next((op for op in plan["ops"] if op["op"] in {"sell", "destroy"}), None)
            if room is None:
                return st, plan
            actions = self.resolve_visible([(room["op"], room["uid"])], rnd, stage)
            receipts.setdefault("makeRoom", []).extend(actions)
            self.log({"screen": "prep", "round": rnd, "decision": "make room", "result": actions})
            observed = self.read()
            if (observed["match"]["round"], observed["match"]["stageId"]) != (rnd, stage):
                raise RuntimeError("Preparation changed while making room for the layout")
            st = self.screen_state(observed)
        raise RuntimeError("The layout continued to request room after all visible sources were considered")

    def end_flow(self, st, row):
        """END: battle-mark command, legacy layout + equipment (sidecar), resolveTemp (rules), free hand slot, ready."""
        from .layout import build_layout_state
        from .screen.layout import execute_layout
        rnd = st["match"]["round"]
        cmd = self.rules.command_end(st)
        if cmd and "effectId" in cmd:
            raise Unsupported(f"battle-mark command {cmd['effectId']} (active command executor not wired)")
        receipts = {}
        st, r = self.plan_end(st, receipts)
        receipts["sidecar"] = {"ops": r["ops"], "ms": r.get("ms")}
        board = self.board_calibration(st["match"]["stageId"])
        final = [p for p in r["board"] if p["kind"] == "chess"]
        result = {"board": [{"uid": p["uid"], "to": [p["row"], p["col"]], "dir": p["dir"]} for p in final]}
        # The sidecar has already chosen the layout. The executor validates pieces and tiles.
        state = build_layout_state(self.track, st, [p["uid"] for p in final],
                                   boss_template=self.state.leader_template, require_bonds=False)

        def remember_step(step):
            self.state.layout_step = {"round": rnd, "stageId": board["stageId"], "uid": step.uid,
                                      "id": self.track._loc_by_uid(step.uid)["piece"].id,
                                      "source": step.source, "target": step.target, "handSlot": step.hand_slot}
            self.save()

        def commit_step():
            self.state.layout_step = None
            self.save()

        with device_action(self.state, self.save, "layout"):
            receipts["layout"] = execute_layout(self.tasker, self.reader, board, self.track, st, state, result,
                                                operator_names={p["uid"]: self.name(p["id"]) for p in final},
                                                before_step=remember_step, on_step=commit_step)
        for operation in r["ops"]:
            if operation["op"] == "equip":
                try:
                    receipt = self.apply_equipment(operation, board, rnd)
                except Unsupported as error:
                    if self.mode != "auto" and not getattr(self.a, "stop_at_leader_prep", False):
                        raise
                    receipts.setdefault("skippedEquipment", []).append({**operation, "reason": str(error)})
                    continue
                receipts.setdefault("equipment", []).append(receipt)
                if self.state.pending_equipment_reward:
                    self.log({**row, "result": receipts})
                    self.finish_reward(self.read())
                    return  # Replan with the newly acquired operator.
                if receipt.get("replan") or self.state.dirty_areas:
                    self.log({**row, "result": receipts})
                    return
        self.place_summons(board, [p for p in r["board"] if p["kind"] == "token"], rnd, receipts)
        if self.a.stop_after_layout:
            anchor = self.flow("observe")
            if anchor["screen"] != "preparation" or anchor["round"] != rnd:
                raise Stop("Layout finished without returning to the current preparation anchor")
            self.log({**row, "result": {**receipts, "after": anchor}})
            return True
        st2 = self.screen_state(self.read(), include_overflow=True)
        res = self.rules.resolve_temp(st2)
        if res:
            receipts["temp"] = self.resolve_visible(res, rnd, st["match"]["stageId"])
            st2 = self.screen_state(self.read(), include_overflow=True)
        tail = self.sc.request("tail", st2, ops=[])
        if tail["ops"]:
            receipts["tail"] = self.resolve_visible([(o["op"], o["uid"]) for o in tail["ops"]],
                                                    rnd, st["match"]["stageId"])
        receipts["ready"] = self.ready_battle()
        self.log({**row, "result": receipts})

    def place_summons(self, board, planned, rnd, receipts):
        if not planned:
            return
        from .layout import as_tile
        from .layout_recovery import matches_piece, recover_layout_step
        from .screen.board_read import inspect_board_positions
        from .screen.deploy import deploy, deployment_tiles
        from .screen.shop import set_shop_open

        field = board.get("deployField", "normal")
        legal_tiles = deployment_tiles(self.reader.stages[board["stageId"]], field, include_changed=False)
        calibrated_tiles = set.intersection(*({tuple(cell[:2]) for cell in board["views"][view]["groundCenters"]}
                                              for view in ("normal", "drag")))
        for target in planned:
            tile = (target["row"], target["col"])
            owner_uid = target.get("ownerUid")
            owner = self.track._loc_by_uid(owner_uid)
            stack = next(((i, p) for i, p in enumerate(self.track.hand)
                          if p and p.kind == "token" and p.id == target["id"] and p.owner == owner_uid), None)
            if tile not in legal_tiles or tile not in calibrated_tiles:
                receipts.setdefault("skippedSummons", []).append({**target, "reason": "Destination is not calibrated"})
                continue
            if not owner or owner["area"] != "board" or stack is None:
                receipts.setdefault("skippedSummons", []).append({**target, "reason": "No matching deployed owner or hand stack"})
                continue
            slot, summon = stack
            s = self.read()
            self._shop_open = False
            own = s["self"]
            funds, cap, count = (own.get(key) for key in ("funds", "deployCap", "deployCount"))
            if (s["match"]["round"] != rnd or s["match"]["stageId"] != board["stageId"]
                    or any(type(value) is not int for value in (funds, cap, count))
                    or count != self.track.deployed_chess_count()):
                raise RuntimeError("The preparation HUD did not match the summon placement")
            legal = self.sc.request("validateSummon", self.screen_state(s), uid=summon.uid,
                                    ownerUid=owner_uid, to=list(tile), dir=target["dir"])
            if not legal["legal"]:
                receipts.setdefault("skippedSummons", []).append({**target, "reason": legal["reason"]})
                continue
            set_shop_open(self.tasker, False)
            owner_tile = as_tile(owner["key"])
            visible = inspect_board_positions(self.tasker, self.reader, board, [owner_tile, tile],
                                              (rnd, funds, cap - count))
            if not matches_piece(visible[owner_tile], owner["piece"]) or visible[tile] is not None:
                raise RuntimeError("The summon owner or destination did not match the board")
            step = {"round": rnd, "stageId": board["stageId"], "uid": summon.uid, "id": summon.id,
                    "source": None, "target": tile, "handSlot": slot}
            with device_action(self.state, self.save, "deploy summon"):
                self.state.layout_step = step
                self.save()
                receipt = deploy(self.tasker, self.reader.slots, board, self.reader.stages, slot, self.name(summon.id),
                                 *tile, target["dir"], expected_round=rnd, stage_confirmed=True, summon=True)
                step["verifiedFacing"] = receipt["facing"]
                self.save()
                checked = recover_layout_step(self.tasker, self.reader, board, self.track, step, s)
                if not checked["applied"]:
                    raise RuntimeError("The summon was still in its original hand slot after release")
                self.state.layout_step = None
            receipts.setdefault("summons", []).append({**receipt, "ownerUid": owner_uid, "verified": checked})

    # ---------- waiting ----------
    def wait_change(self, screen, rnd=None, timeout=600):
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(3)
            s = self.observe()
            if s != screen:
                return
            if rnd is not None and screen == "preparation":
                try:
                    if self.read()["match"]["round"] != rnd:
                        return
                except RuntimeError:
                    pass
        raise Stop(f"screen {screen} did not change in {timeout} s")

    # ---------- main ----------
    def recover_match(self, screen):
        # Title and event navigation are only entered after this match reported a timeout.
        if not self.acting() or not self.state.round or (screen != "disconnect" and not self.state.reconnecting):
            raise Stop(f"Cannot match {screen} to an interrupted saved match")
        if self._spend_retry("reconnect") > RETRY_LIMIT:
            raise Stop("Reconnect retry limit reached")
        self.state.reconnecting = True
        self.mark_holdings_dirty("hand", "temp", "board")
        self.save()
        try:
            receipt = self.flow("recover-match", expected_round=self.state.round)
        except RuntimeError as error:
            raise Retry(f"Reconnection did not return to the match: {error}") from error
        self.state.reconnecting = False
        self.save()
        self.log({"screen": screen, "round": self.state.round, "decision": "reconnect", "result": receipt})

    def run(self, games):
        done = 0
        if self.a.resume and self.resume_doc is not None:
            self.resume(self.resume_doc)
        while done < games:
            pending = self.state.pending_equipment_reward or {}
            if pending.get("kind") == "command" and self.acting():
                screen = "equipmentReward"
            elif self._prep_chain:
                screen = "preparation"  # prep() re-reads the screen and falls back to an observe if it is not the same round
            else:
                screen = self.observe()
                self._shop_open = False
            try:
                if screen == "unknown":
                    raise Stop("screen not recognized")
                if screen == "preparation":
                    if self.prep() or self.a.once:
                        return 0
                elif screen == "equipmentReward":
                    self.finish_command_reward(pending, self.state.round)
                elif screen in ("disconnect", "title", "gameHome", "entrance"):
                    self.recover_match(screen)
                elif screen in {"enemyPreview", "matchInfo", "commandDetails", "bondDetail", "directionPicker"} and self.can_inspect_panels():
                    self.flow("close-panel")
                elif screen == "operatorDetails" and self.can_inspect_panels():
                    from .screen.shop import set_shop_open
                    set_shop_open(self.tasker, True)
                elif screen == "specialSelection":
                    self.draft()
                elif screen == "tacticSelection":
                    self.draft(tactic=True)
                elif screen == "equipmentSelection":
                    self.supply()
                elif screen in ("battle", "settlement"):
                    if self.acting():
                        observed_round = getattr(self, "_observation", {}).get("round")
                        if observed_round is not None:
                            self.state.round = observed_round
                        self.flow("wait-battle", battle_round=self.state.round)
                    else:
                        self.wait_change(screen)
                elif screen in ("briefing", "squadSelection"):
                    raise Stop(f"{screen} reached outside a match start (started by hand?)")
                elif screen == "matchEnded":
                    outcome = self.flow("observe").get("result")
                    self.screenshot("match-end")
                    self.log({"screen": screen, "round": self.state.round, "decision": "match end", "result": outcome})
                    done += 1
                    if done < games:
                        self.new_match(restart=True)
                else:
                    raise Unsupported(f"screen {screen}")
            except Retry as e:
                self.log({"screen": screen, "decision": "retry", "error": str(e)})
                print(f"RETRY: {e}", flush=True)
                continue
            except Stop as e:
                self.screenshot("stop")
                self.save()
                self.log({"screen": screen, "decision": "stop", "error": str(e)})
                print(f"STOP: {e}", flush=True)
                return 2
            except (RuntimeError, ValueError) as e:
                # An executor's verification failed. Never repeat the same action blindly: the next iteration observes,
                # re-reads (areas touched by an unverified action are marked dirty) and decides again. Bounded per round.
                if screen in ("preparation", "specialSelection", "tacticSelection", "equipmentSelection", "equipmentReward") and self._spend_retry("executor") <= RETRY_LIMIT:
                    self.screenshot("retry")
                    self.log({"screen": screen, "decision": "executor retry", "error": f"{type(e).__name__}: {e}"})
                    print(f"RETRY after executor error: {e}", flush=True)
                    self._shop_open = self._prep_chain = False
                    continue
                self.screenshot("error")
                self.save()
                self.log({"screen": screen, "decision": "executor error", "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc(limit=4)})
                print(f"STOP after executor error: {e}", flush=True)
                return 3
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--policy", type=Path, help="policy checkpoint. Resume uses the saved checkpoint by default")
    ap.add_argument("--mode", choices=["observe", "confirm", "auto"], default="observe")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--new", action="store_true", help="start a new match from the 卫戍协议 entrance first")
    start.add_argument("--restart", action="store_true", help="settle the saved match before starting a new one")
    start.add_argument("--resume", action="store_true", help="continue from logs/live/<last game>/state.json")
    ap.add_argument("--seed", type=int, default=0, help="our own RNG (the band draw's weighted part)")
    ap.add_argument("--probe", action="store_true", help="read the live screen and report prerequisites without input")
    ap.add_argument("--adb", type=Path, help="ADB executable for the live device")
    ap.add_argument("--adb-address", help="ADB serial or endpoint for the live device")
    ap.add_argument("--stage", help="stage id (e.g. act3autochess_m04) when the map cannot be recognised from a board with operators on it")
    ap.add_argument("--no-mumu", action="store_true", help="screenshot through plain ADB instead of MuMu extras")
    ap.add_argument("--inspect-panels", action="store_true",
                    help="in observe mode, allow UI navigation to collect the decision inputs")
    ap.add_argument("--once", action="store_true", help="stop after reporting one preparation decision")
    ap.add_argument("--holding-id", action="append", type=lambda value: value.split("=", 1), default=[],
                    metavar="SLOT=ID", help="confirmed ID for the first holdings scan, e.g. hand:9=chess_char_1_19_a")
    ap.add_argument("--rescan-holdings", action="store_true",
                    help="force a holdings scan after manual game changes or an uncertain receipt")
    stop = ap.add_mutually_exclusive_group()
    stop.add_argument("--stop-before-end", action="store_true",
                      help="save and stop when the policy first requests END")
    stop.add_argument("--stop-after-layout", action="store_true",
                      help="execute and verify the policy's END layout, then stop in preparation")
    stop.add_argument("--stop-at-leader-prep", action="store_true",
                      help="save the first leader preparation screenshot and stop before any leader-round decision")
    a = ap.parse_args()
    if any(len(entry) != 2 or not all(entry) for entry in a.holding_id):
        ap.error("--holding-id requires SLOT=ID")
    if len(dict(a.holding_id)) != len(a.holding_id):
        ap.error("--holding-id must identify each slot at most once")
    if a.probe and (a.mode != "observe" or a.new or a.restart or a.resume or a.inspect_panels or a.once
                    or a.holding_id or a.rescan_holdings or a.stop_before_end or a.stop_after_layout
                    or a.stop_at_leader_prep):
        ap.error("--probe reads the current screen only")
    if (a.new or a.restart) and a.mode == "observe":
        ap.error("--new and --restart require an acting mode")
    if (a.once or a.inspect_panels) and a.mode != "observe":
        ap.error("--once and --inspect-panels are for observe mode")
    if a.once and not a.inspect_panels:
        ap.error("--once requires --inspect-panels to collect a decision")
    if a.stop_before_end and a.mode != "confirm":
        ap.error("--stop-before-end requires confirm mode")
    if a.stop_after_layout and a.mode != "confirm":
        ap.error("--stop-after-layout requires confirm mode")
    if a.stop_at_leader_prep and a.mode == "observe":
        ap.error("--stop-at-leader-prep requires an acting mode")
    if not CATALOG_JSON.exists():
        sys.exit("data/catalog.json missing: run `node tools/export_catalog.mjs` first")
    try:
        live = Live(a)
    except (OSError, RuntimeError, ValueError) as error:
        ap.exit(3, f"harness startup: {error}\n")
    try:
        if a.probe:
            print(json.dumps(live.probe(), ensure_ascii=False))
            return
        if a.new or a.restart:
            live.new_match(restart=a.restart)
        sys.exit(live.run(a.games))
    except Stop as error:
        ap.exit(2, f"STOP: {error}\n")
    finally:
        live.sc.close()


if __name__ == "__main__":
    main()
