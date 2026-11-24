"""B6 / B7 offline loop: the engine plays solo:ABYSS@20 matches (tools/sim_server.mjs) and Python makes every decision.

Observation sources (--obs):
  env   the env's own obs/mask (P7 as trained): W_ref
  full  Python encoder + plan block from the ScreenState, with the engine values the screen lacks filled in (spawn
        times, hidden leader, the engine's bounty kill chance) — must reproduce `env` decision for decision (B3)
  info  Python encoder as live: the fields in ZEROED stay 0 (not on screen), bounty kill chance computed live
        (bridge/bounty.mjs from the screen): W_info; with --rebuild the preview is rebuilt from the screen's enemy kinds
--legacy-off (B7, with --obs info): legacy runs no step for the policy seat. Python decides the prep start (B4 commands,
pager; the sidecar's selfPlan focus) and the END flow (B4 battle-mark command, the sidecar's legacy layout + equipment,
B4 resolveTemp); the engine executes them by position. Hand, temp and board of the prep observations come from the B5
tracker, re-read only where the live harness re-reads (the prep start after its own start steps, after a reward pick).

  .venv\\Scripts\\python -m harness.sim --obs env --matches 200 --seed 9200000 --procs 10 --out logs/sim_env.json
"""

import argparse
import base64
import copy
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .catalog import load
from .encode.encoder import ACT, CARDS, FC, FK, PERSONAL, Ctx, Encoder
from .encode.plan import Plan
from .policy import CKPT, ENGINE, Policy
from .rules import Rules
from .sidecar import Sidecar
from .track import Track, screen_snapshot

ROOT = Path(__file__).resolve().parent.parent
# fields the live screen cannot give; in 'info' mode they stay 0 (extend with whatever phase A cannot read)
ZEROED = ["enemies.t", "extra.hiddenBossPreReveal"]
# preview fields phase A may not read (REPORT-win-screen.md): zeroable as "enemies.count", "enemies.gate", "enemies.boss",
# "enemies.elite", "enemies.fly" (count 0, gate unknown = not upper, flags false); also blanked for the live kill chance
PREVIEW_FIELDS = ("count", "gate", "boss", "elite", "fly")


class Driver:
    def __init__(self, mode: str, zeroed=ZEROED, kc: str = "live", rebuild: bool = False, legacy_off: bool = False,
                 truth_holdings: bool = False, engine_rng: bool = False, extra=(), *, checkpoint: Path = CKPT):
        self.extra = list(extra)  # extra sim_server switches (ablations)
        # B7 ablations: holdings read from the engine at every decision instead of the tracker; the layout's tie-breaks
        # drawn from the engine's own rngBots stream instead of the sidecar's
        self.truth_holdings, self.engine_rng = truth_holdings, engine_rng
        self.rebuild = rebuild  # info mode: the preview is rebuilt from the screen's kinds (tools/sim_server.mjs --rebuild)
        self.legacy_off = legacy_off
        self.mode = mode
        self.kc = kc  # 'live' (bridge/bounty.mjs from the screen) or 'engine' (the env's own value): info mode only
        self.zeroed = set(zeroed)
        self.gd = load()
        self.enc = Encoder(self.gd)
        self.plan = Plan(self.gd, self.enc.vocab.bond_ids)
        self.policy = Policy(ckpt=checkpoint.resolve(), threads=2)
        if (self.policy.dim, self.policy.actions) != (self.enc.dim + self.plan.width, self.enc.A):
            raise ValueError("The policy checkpoint input or action width does not match the simulation encoder")
        self.rules = Rules(self.gd) if legacy_off else None
        self.lock = threading.Lock()
        self.diff_rows = 0
        self.rows = 0
        self.act_diff = 0
        self.lo = {"recalibrations": 0, "prepRows": 0, "trackerRowsDiffering": 0, "commandStart": 0, "pager": 0,
                   "commandEnd": 0, "sidecarErrors": 0, "resolveOps": 0, "unsupported": 0, "freeHand": 0}

    def observe(self, row):
        """obs (2570) and mask (100) of one decision row."""
        ref = np.frombuffer(base64.b64decode(row["obs"]), np.float32)
        refm = np.frombuffer(base64.b64decode(row["mask"]), np.uint8)
        if self.mode == "env":
            return ref, refm
        s = row["screen"]
        if self.mode == "full" or "enemies.t" not in self.zeroed or "extra.hiddenBossPreReveal" not in self.zeroed:
            s = copy.deepcopy(s)
            if self.mode == "full" or "enemies.t" not in self.zeroed:
                for e, t in zip(s["self"]["nextEnemies"], row["full"]["t"]):
                    e["t"] = t
            if self.mode == "full" or "extra.hiddenBossPreReveal" not in self.zeroed:
                s["match"]["hiddenBossId"] = row["full"]["hiddenBossId"]
        blank = [f for f in PREVIEW_FIELDS if f"enemies.{f}" in self.zeroed]
        if blank:
            s = copy.deepcopy(s)
            for e in s["self"]["nextEnemies"]:
                for f in blank:
                    e[f] = 0 if f == "count" else None if f == "gate" else False
        ctx = Ctx.of(row["ctx"])
        obs = np.zeros(self.enc.dim + self.plan.width, np.float32)
        _, mask = self.enc.encode(s, ctx, obs[: self.enc.dim])
        obs[self.enc.dim:] = self.plan.obs(s)
        L = self.enc.blocks
        if self.mode == "full" or self.kc == "engine":  # the engine's own kill chance
            for i in range(CARDS):
                obs[L["cards"][0] + i * FC + 10] = ref[L["cards"][0] + i * FC + 10]
            for i in range(PERSONAL):
                obs[L["personal"][0] + i * FK + 7] = ref[L["personal"][0] + i * FK + 7]
        else:
            vals = {("card", int(i)): p for i, p in row["live"]["card"].items()}
            vals.update({("personal", int(i)): p for i, p in row["live"]["personal"].items()})
            self.enc.fill_kill_chance(obs, ctx.kind, s, vals)
        with self.lock:
            self.rows += 1
            if np.abs(obs - ref).max() > 1e-6 or (mask != refm).any():
                self.diff_rows += 1
        return obs, mask

    # ---- B7 (legacy off): the seat's own bookkeeping, as the live harness keeps it ----
    @staticmethod
    def _merged(screen, track):
        s = copy.deepcopy(screen)
        t = track.to_screen()["self"]
        s["self"]["hand"], s["self"]["temp"], s["self"]["board"] = t["hand"], t["temp"], t["board"]
        return s

    @staticmethod
    def _positions(track):
        """Where the tracker believes each of its pieces is (the executor clicks positions, not uids)."""
        pos = {}
        for i, p in enumerate(track.hand):
            if p is not None:
                pos[p.uid] = {"area": "hand", "idx": i}
        for i, p in enumerate(track.temp):
            if p is not None:
                pos[p.uid] = {"area": "temp", "idx": i}
        for k, p in track.board.items():
            r, c = map(int, k.split(","))
            pos[p.uid] = {"area": "board", "row": r, "col": c}
        return pos

    @staticmethod
    def recalibrate(track, screen):
        """Re-read holdings, keeping the uid of every piece found at the same place with the same id (a re-read must not
        make old pieces look gained this prep: the observation's newUids field)."""
        old = {}
        for i, p in enumerate(track.hand):
            if p is not None:
                old[("hand", i, p.id)] = p.uid
        for i, p in enumerate(track.temp):
            if p is not None:
                old[("temp", i, p.id)] = p.uid
        for k, p in track.board.items():
            old[("board", k, p.id)] = p.uid
        bought = {}
        for p in [*track.hand, *track.temp, *track.board.values()]:
            if p is not None:
                bought[p.uid] = p.bought
        track.calibrate(screen)
        remap = {}
        for i, p in enumerate(track.hand):
            if p is not None and ("hand", i, p.id) in old:
                remap[p.uid] = old[("hand", i, p.id)]
        for i, p in enumerate(track.temp):
            if p is not None and ("temp", i, p.id) in old:
                remap[p.uid] = old[("temp", i, p.id)]
        for k, p in track.board.items():
            if ("board", k, p.id) in old:
                remap[p.uid] = old[("board", k, p.id)]
        fresh = max([track._uid, *old.values()], default=0)
        for p in [*track.hand, *track.temp, *track.board.values()]:
            if p is None:
                continue
            if p.uid in remap:
                p.uid = remap[p.uid]
                p.bought = bought.get(p.uid, 0)
            else:
                fresh += 1
                p.uid = fresh
                p.bought = (screen.get("match") or {}).get("round") or 0
        for p in [*track.hand, *track.board.values()]:  # summons follow their owner's uid
            if p is not None and p.kind == "token" and p.owner in remap:
                p.owner = remap[p.owner]
        track._uid = max(track._uid, fresh)

    @staticmethod
    def _uids(track):
        return {p.uid for p in [*track.hand, *track.temp, *track.board.values()] if p is not None}

    def _count(self, key, n=1):
        with self.lock:
            self.lo[key] += n

    def _prep_start(self, seat, msg, sc):
        R, s0, tr = self.rules, msg["screen"], seat["track"]
        self.recalibrate(tr, s0)
        st = self._merged(s0, tr)
        pos = self._positions(tr)
        ops = []
        cmd = R.command_start(st)
        if cmd and "effectId" in cmd:
            ops.append({"op": "command", "effectId": cmd["effectId"], "target": cmd["target"]})
            self._count("commandStart")
        elif cmd:
            self._count("unsupported")
        for e in R.pager(st):
            ops.append({"op": "equip", "item": e["item"], "target": e["carrier"]})
            self._count("pager")
        try:
            seat["botSelf"] = sc.request("selfPlan", st)["botSelf"]
        except Exception:  # noqa: BLE001 — counted; the END flow then runs without legacy's own focus
            seat["botSelf"] = None
            self._count("sidecarErrors")
        seat["recal"] = True  # the prep-start read happens after the harness's own start steps
        return {"ops": ops, "pos": pos}

    def _end(self, seat, msg, sc):
        R, tr = self.rules, seat["track"]
        if self.truth_holdings:
            self.recalibrate(tr, msg["screen"])
        st = self._merged(msg["screen"], tr)
        rng = {"rngState": msg["screen"].get("rngState")} if self.engine_rng else {}
        st["self"]["botSelf"] = seat.get("botSelf")
        if seat.get("seen") is not None:
            st["self"]["seenUids"] = sorted(seat["seen"])
        pos = self._positions(tr)
        cmd = R.command_end(st)
        cmd = cmd if cmd and "effectId" in cmd else None
        if cmd:
            self._count("commandEnd")
        try:
            r = sc.request("end", st, command=cmd, **rng)
        except Exception:  # noqa: BLE001
            self._count("sidecarErrors")
            return {"ops": [], "pos": pos}
        ops = list(r["ops"])
        # legacy's rememberOwned at the end of the prep: the operators held after the flow (ours; flow-made ones are new)
        seat["seen"] = {p["uid"] for p in [*r["hand"], *r["temp"], *r["board"]] if p and p.get("kind") == "chess" and p["uid"] < 1_000_000}
        st2 = copy.deepcopy(st)
        st2["self"]["hand"], st2["self"]["temp"], st2["self"]["board"] = r["hand"], r["temp"], r["board"]
        res = []
        for o in R.resolve_temp(st2):
            self._count("resolveOps")
            res.append({"op": "stow", "uid": o[1], "area": o[2], "idx": o[3]} if o[0] == "move" else {"op": o[0], "uid": o[1]})
        ops += res
        try:  # legacy prepEnd: a full hand frees a slot (freeHandSlot) after the temp is resolved
            t = sc.request("tail", st2, ops=res, **rng)
            ops += t["ops"]
            if t["ops"]:
                self._count("freeHand")
        except Exception:  # noqa: BLE001
            self._count("sidecarErrors")
        # what the sidecar's flow ends with (the engine compares its board after executing the ops: translation check)
        expect = [{k: q.get(k) for k in ("kind", "id", "row", "col", "dir", "items")} for q in r["board"]]
        return {"ops": ops, "pos": pos, "expect": expect}

    def _prep_row(self, seat, row):
        """The prep row as the live harness sees it: holdings from the tracker (newUids in its own uids)."""
        s0, tr = row["screen"], seat["track"]
        new_prep = seat.get("round") != row["round"]
        if seat.get("recal") or new_prep or self.truth_holdings:
            self.recalibrate(tr, s0)
            self._count("recalibrations")
            if new_prep:
                seat["start"] = self._uids(tr)
            seat["recal"] = False
        seat["round"] = row["round"]
        st = self._merged(s0, tr)
        self._count("prepRows")
        if screen_snapshot(st) != screen_snapshot(s0):
            self._count("trackerRowsDiffering")
        return dict(row, screen=st, ctx=dict(row["ctx"], newUids=sorted(self._uids(tr) - seat.get("start", set())))), st

    def run_chunk(self, seed: int, matches: int, envs: int = 8):
        blank = [f for f in PREVIEW_FIELDS if f"enemies.{f}" in self.zeroed]
        p = subprocess.Popen(["node", str(ROOT / "tools/sim_server.mjs"), str(ENGINE), "--seed", str(seed),
                              "--matches", str(matches), "--envs", str(envs), "--blank", ",".join(blank)]
                             + (["--rebuild"] if self.rebuild else []) + (["--legacy-off"] if self.legacy_off else []) + self.extra,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        seats: dict = {}
        sc = Sidecar() if self.legacy_off else None

        def seat_of(sd):
            return seats.setdefault(sd, {"track": Track(self.gd)})

        def reply(o):
            p.stdin.write(json.dumps(o) + "\n")
            p.stdin.flush()

        try:
            for line in p.stdout:
                msg = json.loads(line)
                if msg["t"] == "done":
                    if sc:
                        msg["stats"]["sidecarMs"] = {k: float(np.median(v)) for k, v in sc.ms.items()}
                    return msg
                if msg["t"] == "prepStart":
                    reply(self._prep_start(seat_of(msg["seed"]), msg, sc))
                    continue
                if msg["t"] == "end":
                    reply(self._end(seat_of(msg["seed"]), msg, sc))
                    continue
                rows = list(msg["rows"])
                reads = [None] * len(rows)
                if self.legacy_off:
                    for k, r in enumerate(rows):
                        if r["ctx"]["kind"] == "prep":
                            rows[k], reads[k] = self._prep_row(seat_of(r["seed"]), r)
                pairs = [self.observe(r) for r in rows]
                obs = np.stack([o for o, _ in pairs])
                mask = np.stack([m for _, m in pairs])
                acts = self.policy.act(obs, mask)
                if self.mode == "full":  # B3: the same decision as on the env's own obs
                    ref = np.stack([np.frombuffer(base64.b64decode(r["obs"]), np.float32) for r in rows])
                    refm = np.stack([np.frombuffer(base64.b64decode(r["mask"]), np.uint8) for r in rows])
                    with self.lock:
                        self.act_diff += int((self.policy.act(ref, refm) != acts).sum())
                for k, r in enumerate(rows):  # B7: the tracker follows our own actions
                    if reads[k] is None:
                        continue
                    seat, a = seat_of(r["seed"]), int(acts[k])
                    if a != ACT["END"]:
                        before = self._uids(seat["track"])
                        seat["track"].apply(a, reads[k])
                        for q in [*seat["track"].hand, *seat["track"].temp, *seat["track"].board.values()]:
                            if q is not None and q.uid not in before:
                                q.bought = r["round"]
                    if ACT["REWARD"] <= a < ACT["PERSONAL"]:
                        seat["recal"] = True  # the harness re-reads after a reward pick
                reply({"actions": [int(a) for a in acts]})
            raise RuntimeError(f"sim_server seed {seed} ended without results (exit {p.wait()})")
        finally:
            p.kill()
            if sc:
                sc.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["env", "full", "info"], default="env")
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=9200000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--policy", type=Path, default=CKPT)
    ap.add_argument("--zeroed", default=",".join(ZEROED), help="info mode: comma list of ZEROED fields kept 0")
    ap.add_argument("--kc", choices=["live", "engine"], default="live", help="info mode: bounty kill chance source")
    ap.add_argument("--rebuild", action="store_true", help="info mode: preview rebuilt from the screen's enemy kinds (bridge/preview.mjs)")
    ap.add_argument("--legacy-off", action="store_true", help="B7: Python rules + sidecar + tracker replace legacy for the policy seat")
    ap.add_argument("--b7-truth-holdings", action="store_true", help="B7 ablation: holdings read from the engine at every decision")
    ap.add_argument("--b7-engine-rng", action="store_true", help="B7 ablation: the layout's tie-breaks from the engine's rngBots")
    ap.add_argument("--b7-legacy-end", action="store_true", help="B7 ablation: the engine's own legacy runs the END flow")
    ap.add_argument("--b7-legacy-start", action="store_true", help="B7 ablation: the engine's own legacy runs the start of prep")
    ap.add_argument("--b7-end-compare", action="store_true", help="with --b7-legacy-end: the sidecar's END on the exact engine state vs legacy's")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Driver(a.obs, [z for z in a.zeroed.split(",") if z], a.kc, a.rebuild, a.legacy_off, a.b7_truth_holdings, a.b7_engine_rng,
               (["--legacy-end"] if a.b7_legacy_end else []) + (["--legacy-start"] if a.b7_legacy_start else [])
               + (["--end-compare"] if a.b7_end_compare else []), checkpoint=a.policy)
    per = -(-a.matches // a.procs)
    chunks = [(a.seed + k * per, min(per, a.matches - k * per)) for k in range(a.procs) if a.matches - k * per > 0]
    t0 = time.time()
    with ThreadPoolExecutor(len(chunks)) as ex:
        outs = list(ex.map(lambda c: d.run_chunk(*c), chunks))
    results = sorted((r for o in outs for r in o["results"]), key=lambda r: r["seed"])
    errors = sum(o["stats"].get("errors", 0) + o["stats"].get("failed", 0) for o in outs)
    rejected = sum(o["stats"].get("rejected", 0) for o in outs)
    n = len(results)
    rb = [o["stats"].get("rebuild") or {} for o in outs]
    summary = {"rebuild": {k: sum(x.get(k, 0) for x in rb) for k in ("n", "miss", "ambiguous")} if a.rebuild else None,
               "policy": str(d.policy.checkpoint), "obs": a.obs, "zeroed": sorted(d.zeroed), "kc": d.kc,
               "matches": n, "win": sum(r["win"] for r in results) / max(1, n),
               "survival": float(np.mean([r["survivalRound"] for r in results])), "lp": float(np.mean([r["lp"] for r in results])),
               "errors": errors, "rejected": rejected, "rows": d.rows, "rowsDifferingFromEnv": d.diff_rows,
               "actionsDifferingFromEnvObs": d.act_diff if a.obs == "full" else None, "seconds": round(time.time() - t0, 1),
               "errorSamples": [e for o in outs for e in o["stats"].get("errorSamples", [])][:5]}
    if a.legacy_off:
        eng = [o["stats"].get("legacyOff") or {} for o in outs]
        agg = {k: sum(x.get(k, 0) for x in eng) for k in ("prepStarts", "ends", "ops", "applied", "refused", "unmapped", "learned", "readyFallback", "tempLeft")}
        by: dict = {}
        for x in eng:
            for k, v in (x.get("byOp") or {}).items():
                by.setdefault(k, [0, 0])
                by[k][0] += v["n"]
                by[k][1] += v["ok"]
        keys = ("endBoards", "endSame", "cmpN", "cmpSame", "cmpErrors")
        summary["endChecks"] = {k: sum((o["stats"].get("legacyOff") or {}).get(k, 0) for o in outs) for k in keys}
        summary["endDiffs"] = [d for o in outs for d in (o["stats"].get("legacyOff") or {}).get("endDiffs", [])][:3]
        summary["cmpDiffs"] = [d for o in outs for d in (o["stats"].get("legacyOff") or {}).get("cmpDiffs", [])][:3]
        summary["cmpErrSample"] = [d for o in outs for d in (o["stats"].get("legacyOff") or {}).get("cmpErrSample", [])][:3]
        summary["legacyOff"] = {"unmappedBy": [o["stats"]["legacyOff"].get("unmappedBy") for o in outs], "unmappedSample": outs[0]["stats"]["legacyOff"].get("unmappedSample"), "engine": agg, "byOp": by, "python": d.lo, "sidecarMs": outs[0]["stats"].get("sidecarMs")}
    print(json.dumps(summary, ensure_ascii=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
