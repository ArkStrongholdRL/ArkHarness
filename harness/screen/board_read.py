"""Read calibrated board positions and operator details through MaaFramework."""

import argparse
import json
from pathlib import Path

from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from .capture import ADB_ADDRESS, ADB_PATH, OCR_RESOURCE, SAMPLES, connect, load_ocr
from .deploy import FACING_MARKERS, deployment_tiles, matches_facing, read_board, tile_point
from .native import Frame
from .session import task_recognition
from .read import CATALOG, ROOT, Reader, chess_identity, integer, read_json


class BoardReadRecognition(CustomRecognition):
    def __init__(self, reader: Reader, stage: str, points: dict, expected_facings: dict | None = None,
                 verified_up: set | None = None, expected_remaining: int | None = None,
                 deploy_cap: int | None = None, known_layout: dict | None = None, field: str = "normal"):
        super().__init__()
        self.reader = reader
        self.stage = stage
        self.points = points
        self.expected_facings = expected_facings
        self.verified_up = verified_up or set()
        self.expected_remaining = expected_remaining
        self.deploy_cap = deploy_cap
        self.known_layout = known_layout
        self.allow_missing = known_layout is not None
        self.field = field
        self.leader_round = None
        self.hud = None
        self.tile = None
        self.markers = {}
        self.identity = None
        self.scan_reason = "The preparation board was not visible"

    def directions(self, frame: Frame, tile: tuple[int, int], *, partial: bool = True) -> tuple[str, ...]:
        point = self.points[tile]
        visible = tuple(direction for direction in FACING_MARKERS if matches_facing(frame, point, direction))
        if visible or not partial:
            return visible
        # A sprite can hide the main arrow while leaving its lower wing or upper tip visible.
        x, y = point
        fragments = (
            ("RIGHT", [x + 24, y + 8, 19, 24], 50),
            ("UP", [x - 16, y - 33, 22, 21], 10),
        )
        return tuple(direction for direction, roi, count in fragments
                     if frame.hit("Board.FacingMarker", roi=roi, count=count))

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        if frame.scene() != "preparation" or not frame.hit("ArkRestText") or frame.hit("Shop.Expanded"):
            return None
        kind = json.loads(argv.custom_recognition_param)
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
        if self.field != "normal" and self.leader_round != current_round:
            mode = self.reader.read_mode(frame)
            if not mode or current_round is None or current_round not in (mode.get("bossRound"), mode.get("hiddenRound")):
                self.scan_reason = "The configured leader field is outside a leader preparation round"
                return None
            self.leader_round = current_round
        board_details = frame.hit("Withdraw.Icon")
        details = board_details or frame.hit("Screen.OperatorDetails") or frame.hit("HandSale.Button")
        hit = (0, 0, 1, 1)
        if kind == "reset":
            return hit if details else None
        hud = (
            current_round,
            integer(frame.text("Screen.Funds")),
            integer(frame.text("Screen.DeployCollapsed"), r"剩余可放置角色:([0-9]+)"),
        )
        if None in hud or (self.hud is not None and hud != self.hud):
            if kind == "scan":
                self.scan_reason = f"Preparation HUD read round, funds and remaining as {hud}"
            return None
        if kind == "read":
            if not board_details:
                candidate = chess_identity(self.reader, frame)
                if candidate.get("kind") != "token":
                    return None
                self.identity = candidate
            else:
                self.identity = chess_identity(self.reader, frame)
            return hit
        if kind == "closed" and self.identity and self.identity.get("kind") == "token":
            displayed = chess_identity(self.reader, frame)
            if displayed.get("kind") == "token" and set(displayed["candidateIds"]).intersection(self.identity["candidateIds"]):
                return None
        if details or (self.reader.stage_hint or frame.stage_id()) != self.stage:
            if kind == "scan":
                self.scan_reason = "Operator details were still open" if details else "Stage was not recognized"
            return None
        if kind == "scan":
            if self.expected_remaining is not None and hud[2] != self.expected_remaining:
                self.scan_reason = f"Read {hud[2]} remaining deployments, expected {self.expected_remaining}"
                return None
            self.hud = hud
            if self.known_layout is not None:
                known_chess = sum(name not in self.reader.tokens for name, _ in self.known_layout.values())
                if self.deploy_cap - hud[2] > known_chess:
                    self.scan_reason = (f"HUD reports {self.deploy_cap - hud[2]} operators, but only "
                                        f"{known_chess} operator positions are recorded")
                    return None
                self.markers = {tile: (facing,) for tile, (_, facing) in self.known_layout.items()}
                return hit
            directions = {tile: self.directions(frame, tile) for tile in self.points}
            if self.deploy_cap is not None:
                # Idle animations can cover an arrow entirely for part of a cycle.
                self.markers = {tile: tuple(sorted(set(seen).union(self.markers.get(tile, ()))))
                                for tile, seen in directions.items()}
                occupied = [seen for seen in self.markers.values() if seen]
                if len(occupied) != self.deploy_cap - hud[2] or any(len(seen) != 1 for seen in occupied):
                    ambiguous = [tile for tile, seen in self.markers.items() if len(seen) > 1]
                    self.scan_reason = (f"Found {len(occupied)} marked tiles for {self.deploy_cap - hud[2]} "
                                        f"operators, ambiguous arrows at {ambiguous}")
                    return None
            else:
                self.markers = directions
            if self.expected_facings is not None:
                seen = {tile: ds[0] if len(ds) == 1 else None for tile, ds in self.markers.items() if ds}
                missing = self.expected_facings.keys() - seen.keys()
                wrong = any(tile not in self.expected_facings or self.expected_facings[tile] != direction
                            for tile, direction in seen.items())
                hidden = {tile for tile in self.verified_up if self.expected_facings.get(tile) == "UP"}
                if wrong or missing and (not missing <= hidden or self.expected_remaining is None):
                    self.scan_reason = f"Visible arrows disagree with the planned facings: {seen}"
                    return None
            return hit
        if kind == "closed":
            return hit
        if kind == "empty":
            return hit if self.allow_missing and self.identity is None else None
        if kind == "pick":
            if self.tile is None:
                return None
            if self.known_layout is None:
                visible = self.directions(frame, self.tile)
                if visible and visible != self.markers[self.tile]:
                    return None
            return (*self.points[self.tile], 1, 1)
        return None


def require_terminal(tasker: Tasker, entry: str, terminal: str, *, allow_empty: bool = False) -> str:
    override = {"BoardRead.Pick": {"next": ["BoardRead.ReadDetails", "BoardRead.Empty"], "post_delay": 600}} if allow_empty else {}
    job = tasker.post_task(entry, override).wait() if override else tasker.post_task(entry).wait()
    detail = job.get()
    last = detail.nodes[-1].name if detail and detail.nodes else ""
    if not job.succeeded or (last != terminal and not (allow_empty and last == "BoardRead.Empty")):
        raise RuntimeError(f"MFW did not verify {terminal} for {entry}")
    return last


def find_board_piece(tasker: Tasker, reader: Reader, board: dict, tiles: list[tuple[int, int]],
                     expected_id: str, expected_hud: tuple[int, int, int]) -> tuple[int, int]:
    """Read a newly merged elite at one of the tiles occupied by its consumed copies."""
    points = {tile: tile_point(board, *tile, view_name="normal") for tile in tiles}
    recognition = BoardReadRecognition(reader, board["stageId"], points, known_layout={},
                                       field=board.get("deployField", "normal"))
    recognition.hud = expected_hud
    with task_recognition(tasker, "ArkBoardRead", recognition):
        for tile in tiles:
            recognition.tile = tile
            recognition.identity = None
            try:
                require_terminal(tasker, "BoardRead.Inspect", "BoardRead.Closed")
            except RuntimeError:
                if recognition.identity is not None:
                    raise
                continue
            identity = recognition.identity
            if identity and expected_id in identity["candidateIds"]:
                return tile
    raise RuntimeError(f"Merged operator {expected_id} was not found at any consumed tile: {tiles}")


def inspect_board_positions(tasker: Tasker, reader: Reader, board: dict, tiles: list[tuple[int, int]],
                            hud: tuple[int, int, int]) -> dict:
    """Probe only the affected tiles while the preparation HUD matches the transaction."""
    field = board.get("deployField", "normal")
    legal = deployment_tiles(reader.stages[board["stageId"]], field)
    if not set(tiles) <= legal:
        raise ValueError("Layout recovery requested a tile outside the deployable field")
    points = {tile: tile_point(board, *tile, view_name="normal") for tile in tiles}
    recognition = BoardReadRecognition(reader, board["stageId"], points, known_layout={}, field=field)
    recognition.hud = hud
    observed = {}
    with task_recognition(tasker, "ArkBoardRead", recognition):
        for tile in tiles:
            recognition.tile, recognition.identity = tile, None
            require_terminal(tasker, "BoardRead.Inspect", "BoardRead.Closed", allow_empty=True)
            observed[tile] = recognition.identity
    return observed


def read_board_tiles(tasker: Tasker, reader: Reader, board: dict, *,
                     inspect_tiles: set[tuple[int, int]] | None = None,
                     expected_facings: dict[tuple[int, int], str] | None = None,
                     verified_up: set[tuple[int, int]] | None = None,
                     expected_remaining: int | None = None,
                     known_layout: dict[tuple[int, int], tuple[str, str | None]] | None = None) -> dict:
    """With a live ledger, verify occupant identities and carry its facing values forward."""
    stage = board["stageId"]
    field = board.get("deployField", "normal")
    tiles = deployment_tiles(reader.stages[stage], field)
    calibrated = {tuple(point[:2]) for point in board["views"]["normal"]["groundCenters"]}
    points = {tile: tile_point(board, *tile, view_name="normal") for tile in sorted(tiles & calibrated)}
    if not points:
        raise ValueError("No normal deployment tiles have calibration")
    deploy_cap = (reader.stages[stage]["options"].get("characterLimit")
                  if inspect_tiles is None and expected_facings is None else None)
    if deploy_cap is not None:
        deploy_cap = max(deploy_cap, reader.deploy_cap_floor)
    if known_layout is not None and (deploy_cap is None or not known_layout.keys() <= points.keys()):
        raise ValueError("The saved layout must use calibrated board tiles")
    recognition = BoardReadRecognition(reader, stage, points, expected_facings, verified_up, expected_remaining,
                                       deploy_cap, known_layout, field)
    pieces = {}

    def inspect(tile):
        recognition.tile, recognition.identity = tile, None
        if require_terminal(tasker, "BoardRead.Inspect", "BoardRead.Closed",
                            allow_empty=recognition.allow_missing) == "BoardRead.Empty":
            return
        row, col = tile
        if known_layout is not None and recognition.identity["name"] != known_layout[tile][0]:
            raise RuntimeError(f"The operator at ({row}, {col}) did not match the saved board")
        directions = recognition.markers[tile]
        pieces[tile] = {"row": row, "col": col, "facing": directions[0] if len(directions) == 1 else None,
                        **recognition.identity}

    with task_recognition(tasker, "ArkBoardRead", recognition):
        try:
            require_terminal(tasker, "BoardRead.Scan", "BoardRead.Anchor")
        except RuntimeError as error:
            raise RuntimeError(f"{error}: {recognition.scan_reason}") from error
        candidates = [tile for tile, directions in recognition.markers.items()
                      if directions and (inspect_tiles is None or tile in inspect_tiles)]
        for tile in candidates:
            inspect(tile)
        if recognition.allow_missing:
            expected = deploy_cap - recognition.hud[2]
            missing = [tile for tile in candidates if tile not in pieces]
            chess_count = sum(p.get("kind", "chess") == "chess" for p in pieces.values())
            if chess_count < expected:
                for tile in missing:
                    inspect(tile)
                chess_count = sum(p.get("kind", "chess") == "chess" for p in pieces.values())
            if chess_count != expected:
                unread = [tile for tile in missing if tile not in pieces]
                raise RuntimeError(f"Read {chess_count} saved board operators but HUD reports {expected}; "
                                   f"unread saved tiles: {unread}")
    ordered = [pieces[tile] for tile in candidates if tile in pieces]
    scanned_markers = ({(p["row"], p["col"]): (p["facing"],) for p in ordered}
                       if recognition.allow_missing else recognition.markers)
    return {"stageId": stage, "round": recognition.hud[0], "funds": recognition.hud[1],
            "remaining": recognition.hud[2], "pieces": ordered,
            "facingSource": "track" if known_layout is not None else "markers",
            "markedTiles": [tile for tile, directions in scanned_markers.items() if directions],
            "facings": [{"row": row, "col": col, "facing": directions[0] if len(directions) == 1 else None}
                        for (row, col), directions in scanned_markers.items() if directions],
            "scannedTiles": list(points), "unreadTiles": sorted(tiles - calibrated)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--adb", type=Path, default=ADB_PATH)
    parser.add_argument("--address", default=ADB_ADDRESS)
    parser.add_argument("--ocr-resource", type=Path, default=OCR_RESOURCE)
    args = parser.parse_args()
    try:
        Tasker.set_log_dir(SAMPLES / "_logs")
        Tasker.set_save_draw(False)
        Tasker.set_save_on_error(False)
        reader = Reader(args.catalog, read_json(ROOT / "calib/slots.json"))
        tasker = load_ocr(connect(args.adb, args.address), args.ocr_resource)
        result = read_board_tiles(tasker, reader, read_board(args.board))
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
