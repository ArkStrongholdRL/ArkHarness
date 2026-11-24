"""Layout move experiment: read the board, then (with --run) move one operator to an empty tile and back under delay overrides.
The board is restored after every trial. Needs the preparation screen. Sends real game input with --run only.
  python -I tools/bench_move.py                       # read-only: list operators and free tiles
  python -I tools/bench_move.py --run --name X --to R,C [--delays pick,lift,move,release,facing,swipe]
"""
import argparse, json, sys
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, Reader, read_json
from harness.screen.board_read import read_board_tiles
from harness.screen.deploy import read_board, move_board
from harness.screen.shop import set_shop_open
ap = argparse.ArgumentParser()
ap.add_argument("--run", action="store_true"); ap.add_argument("--at", help="row,col of the operator to move"); ap.add_argument("--to")
ap.add_argument("--delays", help="pick,lift,move,release,facingSwipe(ms duration),facingPost")
ap.add_argument("--trials", type=int, default=1)
a = ap.parse_args()
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
slots = read_json(ROOT / "calib/slots.json"); reader = Reader(CATALOG, slots); reader.stage_hint = "act3autochess_m04"
board = read_board(ROOT / "calib/board.m04.json")
view = set_shop_open(tasker, False)
cache = ROOT / "logs/board_snap.json"
if cache.exists() and a.run:
    snap = json.loads(cache.read_text(encoding="utf-8")); print("board snapshot from cache (round", snap["round"], ")", flush=True)
else:
    s = perf_counter(); snap = read_board_tiles(tasker, reader, board); print("board read", round((perf_counter() - s) * 1000), "ms", flush=True)
    cache.write_text(json.dumps(snap, ensure_ascii=False), encoding="utf-8")
print("round", snap["round"], "remaining", snap["remaining"])
for p in snap["pieces"]: print(" piece", p["row"], p["col"], p.get("facing"), p.get("name"), p.get("id"))
stage = reader.stages[board["stageId"]]["deployTiles"]["normal"]
occupied = {(p["row"], p["col"]) for p in snap["pieces"]}
free = [t for t in stage["melee"] + stage["rangedOnly"] if tuple(t) not in occupied and any(tuple(t) == (r, c) for r, c, *_ in board["views"]["drag"]["groundCenters"])]
print("free calibrated tiles", free[:12])
if not a.run:
    set_shop_open(tasker, view["before"]["expanded"]); sys.exit()
src = tuple(int(v) for v in a.at.split(",")); dst = tuple(int(v) for v in a.to.split(","))
piece = next(p for p in snap["pieces"] if (p["row"], p["col"]) == src); facing = piece["facing"]; a.name = piece["name"]
kind = lambda t: "melee" if list(t) in stage["melee"] else "ranged" if list(t) in stage["rangedOnly"] else None
assert dst not in occupied and kind(src) == kind(dst) is not None, f"tile kinds {kind(src)} -> {kind(dst)} / occupied {dst in occupied}"
print("moving", src, kind(src), "->", dst, kind(dst), "facing", facing)
# delay overrides are passed through post_task by wrapping Tasker.post_task for the Board.Deploy entry
from harness.screen import deploy
if a.delays:
    d = [int(v) for v in a.delays.split(",")]
    ov = {"Board.PickHand": {"post_delay": d[0]}, "Board.LiftHand": {"post_delay": d[1]}, "Board.MoveToTile": {"post_delay": d[2]},
          "Board.ReleaseTarget": {"post_delay": d[3]}, "Board.ChooseFacing": {"duration": d[4], "post_delay": d[5]}}
    orig = deploy.Tasker.post_task
    def post(self, entry, overrides=None):
        if entry == "Board.Deploy":
            overrides = dict(overrides or {})
            for node, settings in ov.items():
                overrides[node] = {**overrides.get(node, {}), **settings}
        return orig(self, entry, overrides)
    deploy.Tasker.post_task = post
print("trial delays", a.delays or "default", flush=True)
for i in range(a.trials):
    for frm, to in ((src, dst), (dst, src)):
        t = perf_counter()
        try:
            r = move_board(tasker, board, reader.stages, a.name, *frm, *to, facing, expected_round=snap["round"], stage_confirmed=True)
            print(f"  move {frm}->{to}: OK {round((perf_counter()-t)*1000)} ms", flush=True)
        except Exception as e:
            print(f"  move {frm}->{to}: FAIL {round((perf_counter()-t)*1000)} ms {e}", flush=True); sys.exit(2)
