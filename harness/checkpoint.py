"""The live match checkpoint and its atomic on-disk representation."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .execution import PendingAction
from .track import Piece, Track


@dataclass
class GameState:
    """State carried across decisions in one match."""
    game: int = 0
    round: int | None = None
    band_id: str | None = None
    mode_id: str | None = None
    stage_id: str | None = None
    deploy_cap_floor: int = 0
    bond_choice: dict | None = None
    lp0: int | None = None
    prep_round: int | None = None  # the round whose prep start ran
    actions: int = 0
    freezes: int = 0
    sold_bases: list = field(default_factory=list)
    blocked_board_sales: list[list[int]] = field(default_factory=list)
    start_uids: list = field(default_factory=list)
    bot_self: dict | None = None
    preview: list | None = None
    preview_round: int | None = None
    boss_id: str | None = None
    hidden_boss_id: str | None = None
    leader_template: str | None = None
    pending_equipment_reward: dict | None = None
    pending_board_merges: list[dict] = field(default_factory=list)
    pending_action: PendingAction | None = None
    layout_step: dict | None = None
    holdings_round: int | None = None
    tracked_deploy_count_round: int | None = None
    dirty_areas: list[str] = field(default_factory=list)
    reconnecting: bool = False


def load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, state: GameState, track: Track, policy: Path) -> None:
    def piece(p):
        return asdict(p) if p else None

    payload = {"state": asdict(state), "track": {
        "nextUidBase": track._uid,
        "hand": [piece(p) for p in track.hand],
        "temp": [piece(p) for p in track.temp],
        "board": {key: piece(p) for key, p in track.board.items()},
    }, "bounties": [[*key, card] for key, card in track._bounty_picks.items()], "policy": str(policy)}
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    pending.replace(path)


def restore_track(track: Track, checkpoint: dict) -> bool:
    def piece(p):
        return Piece(**{**p, "items": [piece(item) for item in p["items"]]}) if p else None

    for rnd, selection, card in checkpoint.get("bounties", []):
        track._bounty_picks[(rnd, selection)] = card
    if "track" not in checkpoint:
        return False
    holdings = checkpoint["track"]
    track._uid = holdings["nextUidBase"]
    track.hand = [piece(p) for p in holdings["hand"]]
    track.temp = [piece(p) for p in holdings["temp"]]
    track.board = {key: piece(p) for key, p in holdings["board"].items()}
    return True
