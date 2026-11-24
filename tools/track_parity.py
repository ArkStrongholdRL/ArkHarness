"""Acceptance tool for B5: ordered self-tracking (harness/track.py) against the engine, per prep.

Input: tools/track_dump.mjs output (one line per prep: the engine's ScreenState at every decision and the executed actions).
Per prep: calibrate at the first decision's ScreenState, then predict from the recorded actions only (the shop, reward offer
and bonds are read from the ScreenState before each action, as the live harness would); after every action the ordered hand
(10), visible temp (5) and board (reading order) are compared with the engine's state.

Two modes:
  strict  - no reseeding: a miss carries into the rest of the prep (what the live harness would see without a re-read).
  reseed  - a miss is counted once and the tracker is reseeded from the engine (the Mac multiset method, for comparison).

  .venv\\Scripts\\python tools/track_parity.py logs/track_dump.jsonl [--show 8]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.catalog import load  # noqa: E402
from harness.encode.encoder import ACT  # noqa: E402
from harness.track import Track, screen_snapshot  # noqa: E402


def label_of(code, read):
    sh = read["self"]["shop"]
    if ACT["BUY"] <= code < ACT["SELL"]:
        s = (sh.get("slots") or [None] * 10)[code - ACT["BUY"]]
        return "buy " + (s["kind"] if s else "?")
    if ACT["SELL"] <= code < ACT["REWARD"]:
        return "sell"
    if ACT["REWARD"] <= code < ACT["PERSONAL"]:
        ro = (sh.get("rewardOffer") or {}).get("slots") or []
        i = code - ACT["REWARD"]
        return "reward " + (ro[i]["kind"] if i < len(ro) and ro[i] else "?")
    return {0: "end", 1: "refresh", 2: "level", 3: "freeze"}.get(code, "other")


def gift_items(gd):
    out = set()
    for g in gd.raw.get("garrisons", {}).values():
        if g.get("eventType") == "SERVER_GAIN" and g.get("effectType") == "SERVER_GAIN_EQUIP":
            out.add((g.get("bbStr") or {}).get("chess"))
    return out


def pieces(snap):
    hand, temp, board = snap
    out = Counter()
    for p in list(hand) + list(temp):
        if p is not None:
            out[(p[0], p[1], p[2])] += 1
    for r, c, *p in board:
        out[(p[0], p[1], p[2])] += 1
    return out


def classify(T, eng, unp_delta, stash, gifts, gd, guess_delta=0):
    tm, em = pieces(T.snapshot()), pieces(screen_snapshot(eng))
    added = em - tm
    removed = tm - em
    if not added and not removed:
        return "slot or order only"
    keys = list(added) + list(removed)
    if unp_delta:
        return "random gift (" + ",".join(sorted({u[1] for u in T.unpredicted[-unp_delta:]})) + ")"
    if guess_delta:
        return "pool gift taken as modal item (roll differed)"
    if any(k[0] == "token" for k in keys):
        return "summon grant or removal"
    if any(k[0] == "item" and k[1] in gifts for k in added):
        return "gift item (SERVER_GAIN_EQUIP) not predicted"
    if any(k[0] == "item" and (gd.is_golden(k[1]) or gd.item(k[1]) and gd.item(k[1]).get("canFuse")) for k in keys):
        return "item merge or fusion"
    if any(k[0] == "item" for k in keys):
        return "item (other)"
    if stash:
        return "hidden overflow queue"
    if any(k[0] == "chess" for k in keys):
        return "chess merge or placement"
    return "other"


def main(path, show=8, debug=0):
    debug_out = []
    gd = load()
    gifts = gift_items(gd)
    T_by_type = defaultdict(lambda: [0, 0, 0])  # n, strict ok, reseed ok
    prep_strict = prep_reseed = preps = 0
    first_miss = Counter()
    strict_causes = Counter()
    causes_reseed = Counter()
    drift = 0
    apply_fail = 0
    examples = []
    actions_total = 0
    stash_preps = 0
    apply_examples = []
    guess_total = unp_total = 0
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        ev = d["ev"]
        if not ev or "s" not in ev[0]:
            continue
        preps += 1
        T = Track(gd)
        R = Track(gd)
        T.calibrate(ev[0]["s"])
        R.calibrate(ev[0]["s"])
        last = ev[0]["s"]
        stash_last = ev[0].get("stash", 0)
        if stash_last:
            stash_preps += 1
        pend = None
        all_strict = all_reseed = True
        missed_once = False
        for e in ev[1:]:
            if "a" in e:
                if pend is not None:
                    continue
                actions_total += 1
                lab = label_of(e["a"], last)
                unp0 = (len(T.unpredicted), len(T.guessed))
                unpR0 = (len(R.unpredicted), len(R.guessed))
                if not T.apply(e["a"], last):
                    apply_fail += 1
                    if len(apply_examples) < 3:
                        apply_examples.append(f"seed {d['seed']} r{d['round']} {label_of(e['a'], last)} code {e['a']}")
                R.apply(e["a"], last)
                pend = (lab, unp0, stash_last, e["a"], unpR0[0], unpR0[1])
                continue
            s = e["s"]
            eng = screen_snapshot(s)
            if pend is None:
                # no action in between (refused call or drift): the state must not have moved
                if T.snapshot() != eng:
                    drift += 1
            else:
                lab, unp0, st, code, unpR0, guessR0 = pend
                ok_s = T.snapshot() == eng
                ok_r = R.snapshot() == eng
                row = T_by_type[lab]
                row[0] += 1
                row[1] += ok_s
                row[2] += ok_r
                if not ok_s:
                    all_strict = False
                    cause = classify(T, s, len(T.unpredicted) - unp0[0], st, gifts, gd, len(T.guessed) - unp0[1])
                    strict_causes[f"{lab}: {cause}"] += 1
                    if not missed_once:
                        missed_once = True
                        first_miss[f"{lab}: {cause}"] += 1
                        if len(examples) < show:
                            examples.append(f"seed {d['seed']} r{d['round']} {lab}: {cause}")
                        if debug and len(debug_out) < debug:
                            tm, em = pieces(T.snapshot()), pieces(screen_snapshot(s))
                            debug_out.append(f"seed {d['seed']} r{d['round']} {lab} code {code}: tracker-only "
                                             f"{dict(tm - em)} engine-only {dict(em - tm)}")
                            debug_out.append(f"    before hand {[x and x['id'] for x in last['self']['hand']]} temp "
                                             f"{[x and x['id'] for x in last['self']['temp']]}")
                            debug_out.append(f"    after  hand {[x and x['id'] for x in s['self']['hand']]} temp "
                                             f"{[x and x['id'] for x in s['self']['temp']]} stash {e.get('stash')}")
                            debug_out.append(f"    tracker hand {[x and x.id for x in T.hand]} temp "
                                             f"{[x and x.id for x in T.temp[:5]]} hidden {len(T.temp) - 5}")
                if not ok_r:
                    all_reseed = False
                    cause_r = classify(R, s, len(R.unpredicted) - unpR0, st, gifts, gd, len(R.guessed) - guessR0)
                    causes_reseed[f"{lab}: {cause_r}"] += 1
                    R.calibrate(s)
                pend = None
            last = s
            stash_last = e.get("stash", 0)
        guess_total += len(T.guessed)
        unp_total += len(T.unpredicted)
        prep_strict += all_strict
        prep_reseed += all_reseed

    def pct(a, b):
        return f"{100 * a / max(1, b):.1f}%"

    n_act = sum(v[0] for v in T_by_type.values())
    s_ok = sum(v[1] for v in T_by_type.values())
    r_ok = sum(v[2] for v in T_by_type.values())
    print(f"preps {preps} (with hidden overflow at prep start: {stash_preps}), actions compared {n_act}, apply failures {apply_fail}, "
          f"state drift without an action {drift}")
    print(f"{'action':14s} {'n':>6s} {'strict ok':>10s} {'reseed ok':>10s}")
    for k in sorted(T_by_type, key=lambda x: -T_by_type[x][0]):
        n, a, b = T_by_type[k]
        print(f"{k:14s} {n:6d} {pct(a, n):>10s} {pct(b, n):>10s}")
    print(f"{'ALL':14s} {n_act:6d} {pct(s_ok, n_act):>10s} {pct(r_ok, n_act):>10s}")
    print(f"preps with every action correct: strict {prep_strict}/{preps} ({pct(prep_strict, preps)}), "
          f"reseed {prep_reseed}/{preps} ({pct(prep_reseed, preps)})")
    print(f"gifts: pool gifts taken as their modal item {guess_total} (the roll itself is random); unpredicted random gifts "
          f"{unp_total} (sell with SERVER_MOST_BOND: a random chess)")
    print("\nfirst miss of each prep, strict (cause):")
    for k, v in first_miss.most_common(20):
        print(f"  {v:5d}  {k}")
    for line_ in debug_out:
        print(line_)
    print("\nall strict misses (cascades included), cause:")
    for k, v in strict_causes.most_common(20):
        print(f"  {v:5d}  {k}")
    print("\nmisses, reseed mode (cause):")
    for k, v in causes_reseed.most_common(20):
        print(f"  {v:5d}  {k}")
    for e in examples:
        print("  e.g.", e)
    for e in apply_examples:
        print("  apply failed:", e)


if __name__ == "__main__":
    args = sys.argv[1:]
    show = int(args[args.index("--show") + 1]) if "--show" in args else 8
    debug = int(args[args.index("--debug") + 1]) if "--debug" in args else 0
    main(args[0], show, debug)
