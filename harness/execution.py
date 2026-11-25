"""Persist the intent to mutate the device and reconcile interrupted receipts."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, TypedDict

from .encode.encoder import ACT, HAND, TEMP, action_name

if TYPE_CHECKING:
    from .checkpoint import GameState


class PendingAction(TypedDict):
    action: str
    round: int | None
    areas: list[str]


def mask_unverified_prep_actions(mask, own: dict, blocked_sales: list[list[int]]) -> list[dict]:
    """Restrict the policy to preparation targets the live executor can identify.

    Returns the legal actions it removed, each with its reason, for the decision log.
    """
    masked: list[dict] = []

    def hide(index: int, reason: str) -> None:
        if mask[index]:
            mask[index] = 0
            masked.append({"action": action_name(index), "reason": reason})

    slots = own["shop"].get("slots") or []
    for index in range(ACT["SELL"] - ACT["BUY"]):
        offer = slots[index] if index < len(slots) else None
        if not offer or not offer.get("id"):
            hide(ACT["BUY"] + index, "card unread")
        elif type(offer.get("price")) is not int:
            hide(ACT["BUY"] + index, "price unread")
    for index, piece in enumerate(own["board"]):
        if piece and [piece["uid"], piece["row"], piece["col"]] in blocked_sales:
            hide(ACT["SELL"] + HAND + TEMP + index, "blocked board sale")
    return masked


def recover_action(state: "GameState") -> PendingAction | None:
    """Transfer an unfinished receipt to the normal holdings recovery path."""
    action = state.pending_action
    if action is not None:
        state.dirty_areas = sorted(set(state.dirty_areas).union(action["areas"]))
        if action["action"] == "level":
            state.shop_level_uncertain = True
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
