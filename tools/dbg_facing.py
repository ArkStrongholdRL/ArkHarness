import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from maa.define import MaaAdbInputMethodEnum as I
from maa.tasker import Tasker
from harness.screen.capture import ADB_PATH, ADB_ADDRESS, MUMU_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from harness.screen.read import CATALOG, read_json
from harness.screen.deploy import read_board, tile_point, facing_override, DeployRecognition, completed_facing_swipes
Tasker.set_log_dir(SAMPLES / "_logs")
tasker = load_ocr(connect(ADB_PATH, ADB_ADDRESS, input_methods=I.Maatouch, mumu=MUMU_PATH), OCR_RESOURCE)
row, col, facing, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], json.load(open(ROOT / "logs/board_snap.json", encoding="utf-8"))["pieces"][0]["name"]
snap = json.load(open(ROOT / "logs/board_snap.json", encoding="utf-8"))
name = next(p["name"] for p in snap["pieces"] if (p["row"], p["col"]) == (int(sys.argv[4]), int(sys.argv[5])))
board = read_board(ROOT / "calib/board.m04.json")
x, y = tile_point(board, row, col)
rec = DeployRecognition("act3autochess_m04", (0, 0), name, (x, y), tile_point(board, row, col, view_name="normal"), facing, from_board=True, stage_confirmed=True)
tasker.resource.register_custom_recognition("ArkDeploy", rec)
job = tasker.post_task("Board.ChooseFacing", {"Board.ChooseFacing": facing_override((x, y), facing)}).wait()
d = job.get(); nodes = [n.name for n in d.nodes] if d else []
print("succeeded", job.succeeded, "nodes", nodes, "swipes", completed_facing_swipes(d.nodes if d else []), "reason", rec.reason)
