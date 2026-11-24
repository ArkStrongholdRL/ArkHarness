"""Persist the intent to mutate the device and reconcile interrupted receipts."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypedDict

from .encode.encoder import ACT, HAND, TEMP

if TYPE_CHECKING:
    from .checkpoint import GameState


class PendingAction(TypedDict):
    action: str
    round: int | None
    areas: list[str]


def mask_unverified_prep_actions(mask, own: dict, blocked_sales: list[list[int]]) -> None:
    """Restrict the policy to preparation targets the live executor can identify."""
    slots = own["shop"].get("slots") or []
    for index in range(ACT["SELL"] - ACT["BUY"]):
        offer = slots[index] if index < len(slots) else None
        if not offer or not offer.get("id") or type(offer.get("price")) is not int:
            mask[ACT["BUY"] + index] = 0
    for index, piece in enumerate(own["board"]):
        if piece and [piece["uid"], piece["row"], piece["col"]] in blocked_sales:
            mask[ACT["SELL"] + HAND + TEMP + index] = 0


def recover_action(state: "GameState") -> PendingAction | None:
    """Transfer an unfinished receipt to the normal holdings recovery path."""
    action = state.pending_action
    if action is not None:
        state.dirty_areas = sorted(set(state.dirty_areas).union(action["areas"]))
        state.pending_action = None
    return action


@contextmanager
def device_action(state: "GameState", save: Callable[[], None], action: str,
                  areas: tuple[str, ...] = ()) -> Iterator[PendingAction]:
    """Save before input and after the receipt and its bookkeeping are applied."""
    if state.pending_action is not None:
        raise RuntimeError("An earlier device action still needs recovery")
    intent: PendingAction = {"action": action, "round": state.round, "areas": list(areas)}
    state.pending_action = intent
    save()
    try:
        yield intent
    except Exception:
        recover_action(state)
        save()
        raise
    else:
        state.pending_action = None
        save()
