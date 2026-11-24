"""Acceptance tool for B1 / B2: re-encode every decision of tools/dump_decisions.mjs with the Python encoder and plan
block and compare with the engine's obs / mask.

  .venv\\Scripts\\python tools/encode_parity.py logs/decisions.jsonl
Exempt fields (screen has no such value): enemies[*].7, extra.2, cards[*].10, personal[*].7.
"""
import base64
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.catalog import load  # noqa: E402
from harness.encode.encoder import Ctx, Encoder, action_name, where  # noqa: E402
from harness.encode.plan import Plan  # noqa: E402

EXEMPT = {"enemies[*].7", "extra.2", "cards[*].10", "personal[*].7"}


def main(path):
    gd = load()
    enc = Encoder(gd)
    plan = Plan(gd, enc.vocab.bond_ids)
    print(f"vocab hash {enc.vocab.hash} dataId {enc.vocab.data_id} dim {enc.dim} + plan {plan.width}")
    n = clean = 0
    worst = defaultdict(float)
    fields = Counter()
    exempt = Counter()
    mask_bad = Counter()
    plan_bad = 0
    kinds = Counter()
    cover = Counter()
    example = {}
    matches = set()
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        n += 1
        matches.add(d["seed"])
        kinds[d["ctx"]["kind"]] += 1
        ref = np.frombuffer(base64.b64decode(d["obs"]), np.float32)
        refm = np.frombuffer(base64.b64decode(d["mask"]), np.uint8)
        s = d["screen"]
        S = s["self"]
        allp = [p for p in (S["hand"] + S["temp"] + S["board"]) if p]
        cover["golden piece"] += any(p["kind"] == "chess" and gd.is_golden(p["id"]) for p in allp)
        cover["equipped piece"] += any(p.get("items") for p in allp)
        cover["item owned"] += any(p["kind"] == "item" for p in allp)
        cover["reward offer"] += bool(S["shop"].get("rewardOffer"))
        cover["personal choice"] += bool(S.get("personalChoice"))
        cover["bond layers > 0"] += any((b.get("layers") or 0) > 0 for b in S["bonds"])
        cover["boss round"] += s["match"]["round"] == s["config"]["bossRound"]
        cover["frozen shop"] += bool(S["shop"].get("frozen"))
        cover["free refresh"] += (S["shop"].get("freeRefreshes") or 0) > 0
        cover["plan P set"] += bool(ref[enc.dim])
        obs, mask = enc.encode(s, Ctx.of(d["ctx"]))
        diff = np.abs(obs.astype(np.float64) - ref[:enc.dim].astype(np.float64))
        same = True
        seen = set()
        for i in np.nonzero(diff > 1e-6)[0]:
            k = where(enc.blocks, int(i))
            if k in EXEMPT:
                if k not in seen:
                    exempt[k] += 1
                seen.add(k)
                continue
            same = False
            worst[k] = max(worst[k], float(diff[i]))
            if k not in seen:
                fields[k] += 1
                example.setdefault(k, f"{d['ctx']['kind']} r{d['round']} seed {d['seed']}: engine {ref[i]:.5f} py {obs[i]:.5f}")
            seen.add(k)
        for a in np.nonzero(mask != refm[:enc.A])[0]:
            same = False
            mask_bad[action_name(int(a)).rstrip("0123456789")] += 1
        if len(ref) > enc.dim:
            pd = np.abs(plan.obs(s) - ref[enc.dim:]).max()
            if pd > 1e-6:
                plan_bad += 1
                same = False
                example.setdefault("plan", f"r{d['round']} seed {d['seed']}: max |d| {pd:.4f}")
        clean += same
    print(f"matches {len(matches)}, decisions {n} ({dict(kinds)}), identical apart from exempt fields: {clean} ({100 * clean / max(1, n):.2f}%)")
    print("coverage", dict(cover))
    print("exempt fields differing (decisions):", dict(exempt))
    print("non-exempt fields differing:", {k: (v, round(worst[k], 6), example[k]) for k, v in fields.most_common()} or "none")
    print("mask differences:", dict(mask_bad) or "none")
    print(f"plan block differing: {plan_bad}" + (f" e.g. {example['plan']}" if plan_bad else ""))


if __name__ == "__main__":
    main(sys.argv[1])
