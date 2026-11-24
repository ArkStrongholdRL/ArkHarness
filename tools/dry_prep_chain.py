"""Dry run of live.prep()'s chained fast path and its fallback in observe mode (sends no game action)."""
import sys, argparse
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.live import Live
a = argparse.Namespace(mode="observe", seed=0, holding_id=[], adb=None, adb_address=None, no_mumu=False,
                       stage="act3autochess_m04", once=True, inspect_panels=True, stop_before_end=False,
                       stop_after_layout=False, resume=False)
live = Live(a)
s = live.read(); rnd = s["match"]["round"]; print("round", rnd, flush=True)
live.state.lp0, live.state.band_id, live.state.prep_round = 20, "band_none", rnd
live.use_mode(s["config"]["modeId"])
def run(label, chain, state_round):
    live.state.round = state_round; live._prep_chain = chain
    t = perf_counter()
    try:
        r = live.prep()
        print(f"{label:38s} {round((perf_counter()-t)*1000):5d} ms  returned {r!r}  shop_open={live._shop_open} chain={live._prep_chain}", flush=True)
    except Exception as e:
        print(f"{label:38s} {round((perf_counter()-t)*1000):5d} ms  {type(e).__name__}: {str(e)[:120]}", flush=True)
live._shop_open = False
run("cold (set_shop_open + read + decide)", False, rnd)
run("chained, shop known open", True, rnd)
run("chained, round mismatch -> fallback", True, rnd - 1)
live.sc.close()
