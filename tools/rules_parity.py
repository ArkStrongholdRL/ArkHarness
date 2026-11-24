"""Acceptance tool for B4: recompute every legacy decision recorded by tools/rules_dump.mjs with harness/rules.py from the
same input (ScreenState + tracked fields) and compare.

  .venv\\Scripts\\python tools/rules_parity.py logs/rules.jsonl [--pool approx|exact] [--show 5]

--pool exact passes the engine's shared pool (diagnostic: the dump records it); approx (default) passes None, so the
rule uses its own approximation of the pool, which is what the live harness would have.
Categories per rule: match / mismatch (with the example), unsupported (a model the port does not have, counted apart),
and for the END/resolve flows the window cases (legacy did nothing, or the decision needs a state not captured).
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.rules import PAGER_ITEM, Rules  # noqa: E402


def load_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def main(path, pool_mode="approx", show=5):
    R = Rules()
    rows = load_rows(path)
    stats = defaultdict(Counter)
    ex = defaultdict(list)

    def note(rule, cat, detail=None):
        stats[rule][cat] += 1
        if detail is not None and len(ex[(rule, cat)]) < show:
            ex[(rule, cat)].append(detail)

    def pool_of(r):
        return r.get("pool") if pool_mode == "exact" else None

    for r in rows:
        s = r["screen"]
        rule = r["rule"]
        if rule == "group":
            got = R.bond_group(s)
            note(rule, "match" if got == r["legacy"] else "mismatch", (r["seed"], r["round"], got, r["legacy"]))

        elif rule == "band":
            w = R.band_weights(s, pool_of(r))
            leg = r["weights"]
            if leg["forced"] is not None:
                ok = w["forced"] == leg["forced"] and r["legacy"] == leg["forced"]
                note(rule, "forced_match" if ok else "forced_mismatch", (r["seed"], w["forced"], leg["forced"], r["legacy"]))
            else:
                if w["forced"] is not None:
                    note(rule, "forced_mismatch", (r["seed"], "py forced", w["forced"], "legacy weighted"))
                    continue
                py = [(b, round(x, 12)) for b, x in w["pairs"]]
                lg = [(b, round(x, 12)) for b, x in leg["pairs"]]
                same_ids = [b for b, _ in py] == [b for b, _ in lg]
                worst = max((abs(a[1] - b[1]) / max(1e-12, abs(b[1])) for a, b in zip(py, lg)), default=0)
                legacy_in = any(b == r["legacy"] and wt > 0 for b, wt in lg)
                if same_ids and worst <= 1e-9 and legacy_in:
                    note(rule, "weighted_match")
                else:
                    note(rule, "weighted_mismatch", (r["seed"], r["legacy"], same_ids, worst, legacy_in))
                note(rule, "weighted_draws")  # every weighted decision (sum of these = the band draws)

        elif rule == "autopick":
            drawn = [pk["idx"] for pk in r["legacy"]["picks"]]
            got = R.auto_pick_commands(s, drawn)
            after = got["command"]["currentEffectId"] if got["command"] else None
            want_after = r["legacy"]["after"]
            ok = len(got["picks"]) == len(drawn) and after == want_after
            note(rule, "match" if ok else "mismatch", (r["seed"], r["round"], drawn, after, want_after, len(got["picks"]), len(drawn)))

        elif rule == "cmd_start":
            got = R.command_start(s)
            note_cmd(rule, got, r, note)

        elif rule == "cmd_end":
            got = R.command_end(s)
            cat_prefix = "" if r.get("exact") else "approx_"
            note_cmd(rule, got, r, note, prefix=cat_prefix)

        elif rule == "pager":
            got = R.pager(s, pool_of(r))
            leg = r["legacy"]
            leg_pager = leg if (leg and R.effect_key(leg.get("itemId") or "") == PAGER_ITEM) else None
            if leg and not leg_pager:
                note(rule, "legacy_other_equip", (r["seed"], r["round"], leg.get("itemId")))
            py_first = got[0] if got else None
            if py_first is None and leg_pager is None:
                note(rule, "match_none")
            elif py_first and leg_pager and py_first["item"] == leg_pager["itemUid"] and py_first["carrier"] == leg_pager["targetUid"]:
                note(rule, "match")
            else:
                note(rule, "mismatch", (r["seed"], r["round"], py_first, leg_pager and (leg_pager["itemUid"], leg_pager["targetUid"])))

        elif rule == "end":
            resolve_check(r, R, note)
            # END flow: legacy layouts (applyPlanSteps) interleave with equipItems, battle marks and resolveTemp.
            # A window is compared when it holds one layout: each board move is made once and lands on the post-END tile.
            evs = r["events"]
            post = {p["uid"]: (p["row"], p["col"], p.get("dir")) for p in (r["post"]["self"].get("board") or []) if p}
            bmoves = [e for e in evs if e["type"] == "move" and e["ok"] and e["to"]["area"] == "board"]
            if not bmoves:
                note(rule, "no_board_move")
                continue
            # functional check on every window: from the state before the window, do the Python moves give the post board?
            if evs[0].get("endPre") is not None:
                g0 = R.apply_plan(evs[0]["endPre"], r["post"])
                fin = {u: (v[0], v[1]) for u, v in post.items()}
                ok = set(g0["final"]) == set(fin) and all(g0["final"][u] == fin[u] for u in fin)
                note("plan_final", "match" if ok else "mismatch", None if ok else (r["seed"], r["round"]))
            uids = [m["uid"] for m in bmoves]
            single = len(set(uids)) == len(uids) and all(
                post.get(m["uid"]) == (m["to"]["row"], m["to"]["col"], m["dir"]) for m in bmoves)
            if not single:
                note(rule, "multi_layout_window")
                continue
            # the layout's input is the state before the window's first event (its lifts and benches are part of it)
            first_pre = evs[0].get("endPre")
            if first_pre is None:
                note(rule, "no_pre_state")
                continue
            got = R.apply_plan(first_pre, r["post"])
            leg_board = [(m["uid"], m["to"]["row"], m["to"]["col"], m["dir"]) for m in bmoves]
            last_board = evs.index(bmoves[-1])
            leg_hand = [(m["uid"], m["to"]["idx"]) for m in evs[:last_board] if m["type"] == "move" and m["ok"] and m["to"]["area"] == "hand"]
            want_final = {uid: (row, col) for uid, row, col, _ in leg_board}
            want_final.update({uid: (post[uid][0], post[uid][1]) for uid in post if uid not in want_final and uid in {x for x in got["final"]}})
            functional = all(got["final"].get(uid) == (row, col) for uid, (row, col, _) in post.items()) and set(got["final"]) == set(post)
            board_set_ok = sorted(got["board"]) == sorted(leg_board)
            hand_uids_ok = sorted(u for u, _ in got["hand"]) == sorted(u for u, _ in leg_hand)
            note(rule, "board_moves_set_match" if board_set_ok else "board_moves_set_differ")
            note(rule, "hand_moves_uid_match" if hand_uids_ok else "hand_moves_uid_differ")
            if got["board"] == leg_board and got["hand"] == leg_hand:
                note(rule, "match_seq")
            elif sorted(got["board"]) == sorted(leg_board) and sorted(got["hand"]) == sorted(leg_hand):
                note(rule, "match_set_order_differs", (r["seed"], r["round"], len(leg_board)))
            else:
                note(rule, "mismatch", (r["seed"], r["round"], "py", got["board"], got["hand"], "legacy", leg_board, leg_hand))
            note(rule, "py_final_is_post" if functional else "py_final_differs", (r["seed"], r["round"]))

        elif rule == "resolve":
            pass  # resolveTemp runs inside END (before setReady): compared from the END events below

    report(stats, ex, pool_mode)


def resolve_check(r, R, note):
    """resolveTemp inside the END window: the ops after the last board move, from the state before the first of them."""
    evs = r["events"]
    last_board = max((i for i, e in enumerate(evs) if e["type"] == "move" and e["ok"] and e["to"]["area"] == "board"), default=-1)
    cand = [e for e in evs[last_board + 1:] if (e["type"] in ("sell", "destroy") and e["ok"])
            or (e["type"] == "move" and e["ok"] and e["to"]["area"] in ("hand", "temp"))]
    if not cand:
        note("resolve", "match_none")
        return
    pre = cand[0].get("endPre")
    leg = []
    for e in cand:
        if e["type"] == "move":
            leg.append(("move", e["uid"], e["to"]["area"], e["to"]["idx"]))
        else:
            leg.append((e["type"], e["uid"]))
    if pre is None:
        note("resolve", "no_pre_state")
        return
    got = R.resolve_temp(pre)
    if got == leg:
        note("resolve", "match")
    else:
        note("resolve", "mismatch", (r["seed"], r["round"], "py", got, "legacy", leg))


def note_cmd(rule, got, r, note, prefix=""):
    leg = r["legacy"]
    if got and "unsupported" in got:
        note(rule, prefix + "unsupported", got["unsupported"])
        return
    if got is None and leg is None:
        note(rule, prefix + "match_none")
        return
    if got is not None and leg is not None and got["effectId"] == leg["effectId"] and got["target"] == leg["target"]:
        note(rule, prefix + ("match" if leg["ok"] else "match_but_use_refused"))
        return
    note(rule, prefix + "mismatch", (r["seed"], r["round"], got, leg))


def report(stats, ex, pool_mode):
    print(f"pool: {pool_mode}")
    for rule in ["group", "band", "autopick", "cmd_start", "cmd_end", "pager", "end", "plan_final", "resolve"]:
        if rule not in stats:
            continue
        c = stats[rule]
        total = sum(v for k, v in c.items() if k not in ("weighted_draws",))
        print(f"{rule}: {dict(c)}  (records {total})")
        for (rl, cat), items in ex.items():
            if rl == rule and cat not in ("match", "match_none", "match_seq", "forced_match", "weighted_match", "board_moves_set_match", "hand_moves_uid_match"):
                for it in items:
                    print(f"    {cat}: {it}")


if __name__ == "__main__":
    args = sys.argv[1:]
    pool = args[args.index("--pool") + 1] if "--pool" in args else "approx"
    show = int(args[args.index("--show") + 1]) if "--show" in args else 5
    main(args[0], pool, show)
