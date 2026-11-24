"""Time the holdings scan phases (UI selection clicks only, no game action)."""
import sys
from pathlib import Path
from time import perf_counter
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from harness.screen.capture import ADB_ADDRESS, ADB_PATH, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, Reader, read_json
from harness.screen.shop import set_shop_open
from harness.screen.hand import read_hand
from harness.screen.equipment_read import read_equipment_slots
from harness.screen.board_read import read_board_tiles
from harness.screen.deploy import read_board
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
reader = Reader(CATALOG, read_json(ROOT / "calib/slots.json"))
board = read_board(ROOT / sys.argv[1])
reader.stage_hint = board["stageId"]  # the terrain template is covered once operators are placed
def t(label, fn):
    s = perf_counter(); r = fn(); print(f"{label:18s} {(perf_counter()-s)*1000:8.0f} ms", flush=True); return r
v = t("close shop", lambda: set_shop_open(tasker, False))
try:
    t("read_hand (10)", lambda: read_hand(tasker, reader, skip_bare=True))
    t("read_board_tiles", lambda: read_board_tiles(tasker, reader, board))
    t("read_equipment", lambda: read_equipment_slots(tasker, reader))
finally:
    t("restore shop", lambda: set_shop_open(tasker, v["before"]["expanded"]))
