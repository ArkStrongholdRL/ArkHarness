"""Fit and draw board homographies from measured tile centers."""

import argparse
import json
from pathlib import Path
from time import sleep

import numpy as np
from PIL import Image, ImageDraw

from .capture import ADB_ADDRESS, ADB_PATH, MUMU_PATH, OCR_RESOURCE, SAMPLES, capture, connect, load_ocr, save_sample
from .deploy import deployment_tiles, read_board, tile_point
from .read import CATALOG, ROOT, Reader, read_json, read_screen


def project(matrix: np.ndarray, grid: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((grid, np.ones(len(grid)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def fit(points: list[list[int]]) -> tuple[np.ndarray, float, float]:
    samples = np.asarray(points, dtype=float)
    if len(samples) < 4:
        raise ValueError("At least four distinct ground points are required")
    grid = samples[:, [1, 0]]
    targets = samples[:, 2:]
    x, y = grid.T
    u, v = targets.T
    zeros = np.zeros((len(samples), 3))
    affine = np.column_stack((x, y, np.ones(len(samples))))
    linear = np.vstack((
        np.column_stack((affine, zeros, -x * u, -y * u)),
        np.column_stack((zeros, affine, -x * v, -y * v)),
    ))
    parameters, _, rank, _ = np.linalg.lstsq(linear, np.r_[u, v], rcond=None)
    if rank < 8:
        raise ValueError("The measured points do not determine a homography")
    matrix = np.r_[parameters, 1].reshape(3, 3)
    error = np.linalg.norm(project(matrix, grid) - targets, axis=1)
    return matrix, float(np.sqrt(np.mean(error**2))), float(error.max())


def build(board: dict) -> None:
    rows = range(board["region"]["rows"][0], board["region"]["rows"][1] + 1)
    cols = range(board["region"]["cols"][0], board["region"]["cols"][1] + 1)
    grid = np.array([(col, row) for row in rows for col in cols])
    for view in board["views"].values():
        if view is None:
            continue
        matrix, rms, max_error = fit(view["measuredGround"])
        centers = project(matrix, grid).round(1).tolist()
        view["homography"] = matrix.round(10).tolist()
        view["residualPx"] = {"rms": round(rms, 2), "max": round(max_error, 2)}
        view["groundCenters"] = [[int(row), int(col), x, y] for (col, row), (x, y) in zip(grid, centers)]
        high = np.asarray(view.get("measuredHigh", []), dtype=float).reshape(-1, 4)
        offsets = high[:, 2:] - project(matrix, high[:, [1, 0]])
        view["heightCorrection"] = [
            {"row": int(row), "col": int(col), "dx": round(float(dx), 1), "dy": round(float(dy), 1)}
            for (row, col, _, _), (dx, dy) in zip(high, offsets)
        ]


def overlay(view: dict, slots: dict, output: Path, *, board: dict | None = None,
            view_name: str = "shopExpanded", catalog: Path = CATALOG) -> None:
    with Image.open(SAMPLES / view["sample"]) as sample:
        image = sample.convert("RGB")
    draw = ImageDraw.Draw(image)
    points = view["groundCenters"]
    if board is not None and (field := board.get("deployField")):
        stage = read_json(catalog / "stages.json")[board["stageId"]]
        tiles = deployment_tiles(stage, field)
        points = [(row, col, *tile_point(board, row, col, view_name=view_name))
                  for row, col, _, _ in points if (row, col) in tiles]
    for row, col, x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#11e6ca", outline="#071819")
        draw.text((x + 5, y - 12), f"{row},{col}", fill="white", stroke_width=1, stroke_fill="black")
    for _, _, x, y in view.get("measuredHigh", []):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#ff8be1", outline="black")
    controls = [(name, data["center"]) for name, data in slots["buttons"].items()
                if data.get("screen", "preparation") == "preparation"]
    if view_name == "shopExpanded":
        controls.extend((f"shop {row['screenIndex']}", row["center"]) for row in slots["shop"]["layouts"]["1"])
    hand_view = "shopExpanded" if view_name == "shopExpanded" else "normal"
    controls.extend((f"hand {index}", center)
                    for index, center in enumerate(slots["hand"]["centersByView"][hand_view]))
    for name, (x, y) in controls:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#ffba39", outline="black")
        draw.text((x + 7, y - 12), name, fill="white", stroke_width=1, stroke_fill="black")
    image.save(output)


def capture_leader_drag(adb: Path, address: str, board: dict, catalog: Path, slot: int) -> dict:
    """Lift a named hand operator, capture the highlighted board, then return it to the same slot."""
    from maa.define import MaaAdbInputMethodEnum
    from maa.tasker import Tasker

    from .flow import execute
    from .hand import read_hand
    from .shop import set_shop_open

    if board.get("deployField") != "bossLeft" or not board["views"].get("normal"):
        raise ValueError("Select a solo leader board with preparation calibration")
    slots = read_json(ROOT / "calib/slots.json")
    if not 0 <= slot < slots["hand"]["capacity"]:
        raise ValueError("Expected a valid zero-based hand slot")
    Tasker.set_log_dir(SAMPLES / "_logs")
    Tasker.set_save_draw(False)
    Tasker.set_save_on_error(False)
    controller = connect(adb, address, mumu=MUMU_PATH, input_methods=MaaAdbInputMethodEnum.Maatouch)
    tasker = load_ocr(controller, OCR_RESOURCE)
    reader = Reader(catalog, slots)
    reader.stage_hint = board["stageId"]
    before = read_screen(tasker, reader)
    round_number = before["match"]["round"]
    config = before["config"]
    if (not before["self"]["canReady"] or round_number is None
            or round_number not in (config["bossRound"], config["hiddenRound"])):
        raise RuntimeError("Leader preparation was not verified")
    set_shop_open(tasker, False)
    hand = read_hand(tasker, reader, indices=[slot])
    operator = hand["slots"][0]["name"]
    if not operator:
        raise RuntimeError("No named operator was found in the selected hand slot")
    x, y = slots["hand"]["centersByView"]["normal"][slot]
    try:
        if not controller.post_touch_down(x, y).wait().succeeded:
            raise RuntimeError("Could not touch the verified hand slot")
        sleep(0.30)
        if not controller.post_touch_move(x, y - 80).wait().succeeded:
            raise RuntimeError("Could not lift the operator")
        sleep(0.40)
        if execute(tasker, slots, "observe", catalog, None, 3000)["screen"] != "drag":
            raise RuntimeError("The operator drag view was not verified")
        sample = capture(controller)
    finally:
        try:
            returned = controller.post_touch_move(x, y).wait().succeeded
        finally:
            released = controller.post_touch_up().wait().succeeded
        if not returned or not released:
            raise RuntimeError("Could not return the touch to the original hand slot")
    after = read_hand(tasker, reader, indices=[slot])
    if any(after[key] != hand[key] for key in ("round", "funds", "remaining", "slots")):
        raise RuntimeError("The original hand slot and preparation HUD did not match after the drag")
    return save_sample(SAMPLES, "领袖拖拽视角", round_number, f"{board['stageId']} {operator}", sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json")
    parser.add_argument("--view", default="shopExpanded")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--capture-leader-drag", type=int, metavar="HAND_SLOT",
                        help="capture a leader drag from a zero-based hand slot and cancel the drag")
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overlay", type=Path, default=SAMPLES / "_calib_overlay.png")
    args = parser.parse_args()
    path = args.board
    if args.capture_leader_drag is not None:
        try:
            sample = capture_leader_drag(args.adb, args.address, read_board(path), args.catalog, args.capture_leader_drag)
        except (OSError, RuntimeError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        print(json.dumps(sample, ensure_ascii=False))
        return

    board = read_json(path)
    if "groundGeometry" in board:
        if args.write:
            parser.error("--write requires a board file with its own measured geometry")
        board = read_board(path)
    else:
        build(board)
        if args.write:
            path.write_text(json.dumps(board, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    view = board["views"].get(args.view)
    if not view or not view.get("sample"):
        parser.error(f"No calibration screenshot for {args.view}")
    slots = read_json(ROOT / "calib/slots.json")
    overlay(view, slots, args.overlay, board=board, view_name=args.view, catalog=args.catalog)
    for name, view in board["views"].items():
        if view is not None:
            if "residualPx" in view:
                print(name, view["residualPx"])


if __name__ == "__main__":
    main()
