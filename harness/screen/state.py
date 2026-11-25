"""State contracts for tool calls made through the live screen entry point."""

from typing import Literal

from .native import Frame
from .read import integer


PANELS = {"matchInfo", "enemyPreview", "bondDetail", "commandDetails", "directionPicker"}
INSPECTIONS = {"inspect-info", "inspect-enemies", "inspect-bond"}
BRIEFING_ACTIONS = ("inspect-briefing", "select-bond-group")
BATTLE_ACTIONS = {"start-battle", "ready-battle", "wait-battle"}
DRAFT_SCREENS = {"bountySelection", "equipmentSelection", "tacticSelection"}
INTERRUPT_SCREENS = {"preparation", "leaveConfirm", "entrance", "abandonConfirm", "matchInterrupted"}
ENTRY_STATES = {
    "Recovery.Notice": "disconnect",
    "Lifecycle.Title": "title",
    "Lifecycle.GameHome": "gameHome",
    "Lifecycle.Entrance": "entrance",
    "Lifecycle.Difficulty": "difficultySelection",
    "Lifecycle.Briefing": "briefing",
    "Lifecycle.Strategy": "strategySelection",
    "Lifecycle.Squad": "squadSelection",
    "Lifecycle.LeaveConfirm": "leaveConfirm",
    "Lifecycle.AbandonConfirm": "abandonConfirm",
    "Lifecycle.Interrupted": "matchInterrupted",
}

LiveScreen = Literal[
    "briefing", "squadSelection", "specialSelection", "preparation", "enemyPreview", "rewardSelection",
    "personalSelection", "equipmentSelection", "tacticSelection", "battle", "settlement", "matchEnded", "disconnect",
    "title", "gameHome", "entrance", "operatorDetails", "matchInfo", "commandDetails", "bondDetail", "directionPicker", "unknown",
]
LIVE_SCREENS: dict[str, LiveScreen] = {
    "briefing": "briefing",
    "squadSelection": "squadSelection",
    "bountySelection": "specialSelection",
    "equipmentSelection": "equipmentSelection",
    "tacticSelection": "tacticSelection",
    "preparation": "preparation",
    "enemyPreview": "enemyPreview",
    "operatorDetails": "operatorDetails",
    "matchInfo": "matchInfo",
    "commandDetails": "commandDetails",
    "directionPicker": "directionPicker",
    "bondDetail": "bondDetail",
    "battle": "battle",
    "settlement": "settlement",
    "matchInterrupted": "matchEnded",
    "disconnect": "disconnect",
    "title": "title",
    "gameHome": "gameHome",
    "entrance": "entrance",
}


def screen_snapshot(frame: Frame) -> dict:
    screen = frame.scene()
    if ((screen == "drag" and frame.hit("Board.FacingCancel"))
            or (screen in (None, "preparation") and frame.hit("Board.FacingCancelIcon") and frame.hit("ArkRestText"))):
        screen = "directionPicker"
    anchored = screen == "preparation" and frame.hit("ArkRestText")
    if screen in {None, "preparation"}:
        if frame.hit("Screen.MatchInfoPanel"):
            screen = "matchInfo"
            anchored = False
        elif anchored and frame.hit("Screen.OperatorDetailsVisible"):
            screen = "operatorDetails"
            anchored = False
    if screen == "preparation" and not anchored:
        screen = None
    if screen is None:
        screen = ENTRY_STATES.get(frame.first_hit("Screen.Lifecycle"))
    current_round = None
    if screen is None or screen in {"preparation", "enemyPreview", "bondDetail", "commandDetails", "drag", "directionPicker",
                                   *DRAFT_SCREENS, "battle", "settlement"}:
        current_round = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
    snapshot = {
        "screen": screen or ("roundHud" if current_round is not None else "unknown"),
        "round": current_round,
        "canReady": anchored and frame.hit("Screen.ReadyAvailable"),
    }
    if screen in {"battle", "settlement"} and current_round is None and frame.hit("Battle.LeaderActive"):
        snapshot["leaderBattle"] = True
    if screen == "entrance":
        snapshot["savedRound"] = integer(frame.text("Entry.ResumeSoloText"), r"当前回合([0-9]+)")
    if screen == "matchInterrupted":
        snapshot["result"] = {"outcome": "interrupted", "passedRounds": integer(frame.text("Result.PassedRounds"))}
    return snapshot


def require_action(action: str, before: dict) -> None:
    screen = before["screen"]
    if action == "recover-match" and screen in {"disconnect", "title", "gameHome", "entrance"}:
        return
    if action == "open-entrance" and screen in {"gameHome", "entrance"}:
        return
    if action == "leave-match" and screen in {"preparation", "leaveConfirm", "entrance"}:
        return
    if action in {"interrupt-match", "restart-match"} and screen in INTERRUPT_SCREENS:
        return
    if action == "wait-battle" and screen in DRAFT_SCREENS | {
        "battle", "settlement", "preparation", "matchInterrupted", "disconnect", "roundHud", "unknown",
    }:
        return
    if action in BRIEFING_ACTIONS and screen == "briefing":
        return
    if action == "select-bounty" and screen == "bountySelection":
        return
    if action == "select-equipment" and screen == "equipmentSelection":
        return
    if action == "select-tactic" and screen == "tacticSelection":
        return
    if action == "enter-layout" and screen in {
        "gameHome", "entrance", "difficultySelection", "briefing", "strategySelection", "squadSelection", "preparation",
    }:
        return
    if action == "close-panel" and (screen in PANELS or screen == "preparation"):
        return
    if screen == "preparation" and before["round"] is not None:
        if action in INSPECTIONS or (action in {"start-battle", "ready-battle"} and before["canReady"]):
            return
    raise RuntimeError(f"{action} is unavailable on {screen}; no input was sent")


def require_return(action: str, before: dict, after: dict) -> None:
    if action in BATTLE_ACTIONS:
        started = before["round"]
        current = after["round"]
        if after["screen"] in {"matchInterrupted", "disconnect"}:
            return
        if after["screen"] in DRAFT_SCREENS and current is None:
            return
        if after["screen"] in DRAFT_SCREENS | {"preparation"} and current is not None and current > started:
            return
        if action == "ready-battle" and after["screen"] in {"battle", "settlement"} and current == started:
            return
        raise RuntimeError(f"Could not verify battle transition from round {started}; "
                           f"observed {after['screen']} in round {current}")
    if after["screen"] != "preparation":
        raise RuntimeError(f"{action} did not return to the preparation anchor")
    initial_round = before["round"]
    next_round = after["round"]
    if action == "enter-layout" and next_round is None:
        raise RuntimeError("The preparation anchor was detected but its round was unreadable")
    if initial_round is not None and next_round != initial_round:
        raise RuntimeError(f"{action} returned to preparation in a different or unreadable round")
