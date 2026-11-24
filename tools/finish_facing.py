"""Recovery helper: complete an open direction picker at a tile with the standard 420 ms swipe (sends real input)."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, connect
from harness.screen.deploy import read_board, tile_point, facing_override
from maa.define import MaaAdbInputMethodEnum as I
row, col, facing = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
c = connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH)
d = facing_override(tile_point(read_board(ROOT / "calib/board.m04.json"), row, col), facing)
print("swipe", d, c.post_swipe(*d["begin"], *d["end"], 420).wait().succeeded)
