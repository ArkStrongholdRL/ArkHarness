"""Swap two calibrated board occupants using verified MaaFramework moves."""

import argparse
import json
from pathlib import Path

from maa.define import MaaAdbInputMethodEnum
from maa.tasker import Tasker

from .board_read import read_board_tiles
from .capture import add_session_args, session
from .deploy import FACING_OFFSETS, move_board, read_board, tile_point
from .read import CATALOG, ROOT, Reader, clean


class SwapUnverified(RuntimeError):
    def __init__(self, message: str, moves: list[dict]):
        super().__init__(message)
        self.moves = tuple(moves)


def accepts_ranged_tile(reader: Reader, piece: dict) -> bool:
    ids = set(piece["candidateIds"])
    positions = {row.get("position") for kind, entity_id, row in reader.shop_names[piece["name"]]
                 if kind == "chess" and entity_id in ids}
    return positions == {"RANGED"}


def swap_board(tasker: Tasker, reader: Reader, board: dict,
               source: tuple[int, int], target: tuple[int, int],
               source_name: str, target_name: str, facing: str) -> dict:
    """Move source to target and preserve the displaced unit's original facing."""
    source_name, target_name = clean(source_name), clean(target_name)
    if source == target or not source_name or not target_name or facing not in FACING_OFFSETS:
        raise ValueError("Two distinct tiles, their displayed names and a valid facing are required")
    legal = reader.stages[board["stageId"]]["deployTiles"]["normal"]
    melee = set(map(tuple, legal["melee"]))
    ranged = set(map(tuple, legal["rangedOnly"]))
    if source not in melee | ranged or target not in melee | ranged:
        raise ValueError("Both swap endpoints must be normal deployment tiles")
    for tile in (source, target):
        tile_point(board, *tile)

    endpoints = {source, target}
    before = read_board_tiles(tasker, reader, board, inspect_tiles=endpoints)
    occupants = {(row["row"], row["col"]): row for row in before["pieces"]}
    for tile, name in ((source, source_name), (target, target_name)):
        occupant = occupants.get(tile)
        if occupant is None or occupant["name"] != name or occupant["facing"] is None:
            raise RuntimeError(f"The occupant, name or facing of {tile} did not match")
    first, second = occupants[source], occupants[target]
    first_ranged = accepts_ranged_tile(reader, first)
    second_ranged = accepts_ranged_tile(reader, second)
    if source in ranged and not second_ranged:
        raise RuntimeError("The displaced unit was not verified for the source tile's terrain")
    if target in ranged and not first_ranged:
        raise RuntimeError("The moving unit was not verified for the destination tile's terrain")

    drag_tiles = {tuple(tile[:2]) for tile in board["views"]["drag"]["groundCenters"]}
    free = (set(before["scannedTiles"]) & drag_tiles & (melee | ranged)) - set(before["markedTiles"])
    first_staging = free & (melee | ranged if first_ranged else melee)
    second_staging = free & (melee | ranged if second_ranged else melee)
    if second_staging:
        staging = min(second_staging)
        sequence = ((target_name, target, staging, second["facing"]),
                    (source_name, source, target, facing),
                    (target_name, staging, source, second["facing"]))
    elif first_staging:
        staging = min(first_staging)
        sequence = ((source_name, source, staging, facing),
                    (target_name, target, source, second["facing"]),
                    (source_name, staging, target, facing))
    else:
        raise RuntimeError("No compatible calibrated staging tile is free")

    moves = []
    try:
        for name, start, end, direction in sequence:
            receipt = move_board(tasker, board, reader.stages, name, *start, *end, direction,
                                 expected_round=before["round"])
            moves.append(receipt)
            if receipt["round"] != before["round"]:
                raise RuntimeError("The preparation round changed during the swap")
        after = read_board_tiles(tasker, reader, board, inspect_tiles=endpoints | {staging})
        observed = {(row["row"], row["col"]): (row["name"], row["facing"]) for row in after["pieces"]}
        expected = {source: (target_name, second["facing"]), target: (source_name, facing)}
        if after["round"] != before["round"] or observed != expected or staging in after["markedTiles"]:
            raise RuntimeError("Final identities, facings and staging clearance did not verify")
    except RuntimeError as error:
        raise SwapUnverified(f"Swap unverified after {len(moves)} checked move(s): {error}", moves) from error
    return {"stageId": before["stageId"], "round": before["round"], "source": source, "target": target,
            "staging": staging, "moves": moves, "pieces": after["pieces"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, nargs=2, required=True)
    parser.add_argument("--target", type=int, nargs=2, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--facing", required=True, choices=FACING_OFFSETS)
    parser.add_argument("--board", type=Path, default=ROOT / "calib/board.json")
    add_session_args(parser, catalog=CATALOG)
    args = parser.parse_args()
    try:
        screen_session = session(args.adb, args.address, args.ocr_resource,
                                 input_methods=MaaAdbInputMethodEnum.Maatouch, catalog=args.catalog)
        tasker, reader = screen_session.tasker, screen_session.reader
        result = swap_board(tasker, reader, read_board(args.board), tuple(args.source), tuple(args.target),
                            args.source_name, args.target_name, args.facing)
        print(json.dumps(result, ensure_ascii=False))
    except SwapUnverified as error:
        parser.exit(1, f"{error}\n{json.dumps(error.moves, ensure_ascii=False)}\n")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
