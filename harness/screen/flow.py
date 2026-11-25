"""Run the anchored MaaFramework navigation pipeline."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from maa.context import Context
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker

from ..wave import match_preview
from .bounty import SLOTS as BOUNTY_SLOTS
from .briefing import choose_bond_group, inspect_briefing
from .capture import OCR_RESOURCE, add_session_args, session
from .commander import select as select_commander
from .native import MIN_NAME_SCORE, Frame, recognition_data
from .names import match_name
from .read import CATALOG, ROOT, Reader, clean, integer, read_json, read_screen
from .session import run as run_task, task_recognition
from .state import (
    BATTLE_ACTIONS, BRIEFING_ACTIONS, DRAFT_SCREENS, INSPECTIONS, LIVE_SCREENS, LiveScreen, require_action, require_return,
    screen_snapshot,
)


ENTRIES = {
    "observe": ("ArkObserve", "ArkSnapshot"),
    "enter-layout": ("Entry.EnterLayout", "ArkReturnedRest"),
    "open-entrance": ("Entry.OpenEvent", "Entry.EventEntrance"),
    "close-panel": ("ArkClosePanel", "ArkReturnedRest"),
    "inspect-info": ("ArkInspectMatchInfo", "ArkReturnedRest"),
    "inspect-enemies": ("ArkInspectEnemies", "ArkReturnedRest"),
    "inspect-bond": ("ArkInspectBond", "ArkReturnedRest"),
    "start-battle": ("ArkStartBattle", "ArkRestAfterBattle"),
    "ready-battle": ("ArkReadyBattle", "ArkBattleStarted"),
    "wait-battle": ("ArkWaitBattle", "ArkRestAfterBattle"),
    "leave-match": ("Interrupt.Leave", "Interrupt.SavedEntrance"),
    "interrupt-match": ("Interrupt.End", "Interrupt.NewEntrance"),
    "select-bounty": ("Bounty.Select", "ArkReturnedRest"),
    "select-equipment": ("Supply.Select", "ArkReturnedRest"),
    "select-tactic": ("Tactic.Select", "ArkReturnedRest"),
    "recover-match": ("Recovery.Route", "Recovery.Returned"),
}
DRAFT_ACTIONS = {"select-bounty", "select-equipment", "select-tactic"}
ERRORS = {
    "Entry.SavedMatch": "A saved match is displayed at the entrance; no opening input was sent",
    "ArkUnexpectedScreen": "Expected the preparation anchor or a supported panel for close-panel",
    "ArkPanelNotFound": "Panel did not open after one click",
    "ArkRestNotRecovered": "Panel close clicked once; the preparation anchor was not recovered",
    "ArkBattleNotVerified": "The requested battle or post-battle phase was not verified",
}
ENTRY_CHECKPOINTS = {"briefing": "Entry.BriefingCheckpoint", "squadSelection": "Entry.SquadCheckpoint"}


class ReadyAction(CustomAction):
    def __init__(self, navigation):
        super().__init__()
        self.navigation = navigation

    def run(self, context, argv):
        round_number = next((data["readyRound"] for data in recognition_data(argv.reco_detail)
                             if "readyRound" in data), None)
        if round_number is None:
            return False
        click = context.run_action("ArkClickReady", argv.box)
        if click is None or not click.success:
            return False
        self.navigation.started_round = round_number
        return True


class NavigationRecognition(CustomRecognition):
    def __init__(self, buttons: dict, action: str, bond: str | None, reader: Reader | None,
                 expected_round: int | None = None, *, bounty: tuple[int, str] | None = None):
        super().__init__()
        self.buttons = buttons
        self.action = action
        self.bond = bond
        self.reader = reader
        self.expected_round = expected_round
        self.bounty = bounty
        self.started_round = None
        self.snapshot = None
        self.stage_id = None
        self.inspection = None
        self.inspection_error = None
        self.steps = []

    def point(self, name: str) -> tuple[int, int, int, int]:
        x, y = self.buttons[name]["center"]
        return x, y, 1, 1

    def panel_result(self, frame: Frame, action: str, target: tuple):
        if self.action == action and self.reader is not None:
            try:
                return self.AnalyzeResult(target, {"inspection": self.reader.read(frame)})
            except (OSError, RuntimeError, ValueError) as error:
                return self.AnalyzeResult(target, {"inspectionError": str(error)})
        return target

    def collect_results(self, nodes):
        self.snapshot = self.stage_id = self.inspection = self.inspection_error = None
        for node in nodes:
            if not node.completed:
                continue
            for data in recognition_data(node.recognition):
                if "snapshot" in data:
                    self.snapshot = data["snapshot"]
                    self.stage_id = data.get("stageId")
                if "inspection" in data:
                    self.inspection = data["inspection"]
                if "inspectionError" in data:
                    self.inspection_error = data["inspectionError"]
                if "scrolledEnemies" in data and self.inspection is not None:
                    kinds, ambiguous = data["scrolledEnemies"]
                    self.inspection["enemyKinds"] = sorted(set(self.inspection["enemyKinds"]) | set(kinds))
                    self.inspection["ambiguousEnemyNames"].update(ambiguous)

    @staticmethod
    def selected_bond(frame: Frame) -> tuple[int, int, int, int] | None:
        result = frame.node("Screen.BondUnderline")
        boxes = [row.box for row in result.filtered_results if 45 <= row.box[2] <= 70] if result else []
        if len(boxes) != 1:
            return None
        x, _, width, _ = boxes[0]
        return x + width // 2, 89, 1, 1

    def named_bond(self, frame: Frame) -> tuple | None:
        if self.reader is None:
            return None
        candidates = [box[0] + 27 for box in frame.bond_boxes()
                      if match_name(clean(frame.text("Screen.BondName", roi=box, min_score=MIN_NAME_SCORE)),
                                    self.reader.bonds) == self.bond]
        return (candidates[0], 89, 1, 1) if len(candidates) == 1 else None

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg) -> CustomRecognition.AnalyzeResult | tuple | None:
        if argv.image.shape != (720, 1280, 3):
            return None
        frame = Frame(context, argv.image)
        kind = json.loads(argv.custom_recognition_param)
        hit = (0, 0, 1, 1)
        if kind == "savedRound":
            current = integer(frame.text("Entry.ResumeSoloText"), r"当前回合([0-9]+)")
            return hit if current is not None and self.expected_round in (None, current) else None
        if kind == "recoverySavedRound":
            saved = integer(frame.text("Entry.ResumeSoloText"), r"当前回合([0-9]+)")
            return hit if saved in (self.expected_round, self.expected_round + 1) else None
        if kind == "recoveryReturned":
            snapshot = screen_snapshot(frame)
            current = snapshot["round"]
            screen = snapshot["screen"]
            same_match = (
                screen in {"preparation", "battle", "settlement"}
                and (current in (self.expected_round, self.expected_round + 1) or snapshot.get("leaderBattle"))
            )
            next_draft = screen in DRAFT_SCREENS and current in (None, self.expected_round + 1)
            return self.AnalyzeResult(hit, {"snapshot": snapshot}) if same_match or next_draft else None
        if kind in {"bountyTarget", "bountyConfirm"} and self.reader is not None and self.bounty is not None:
            slot, card_id = self.bounty
            equipment = self.action == "select-equipment"
            if equipment:
                offer = self.reader.equipment_offer(frame, slot)
            elif self.action == "select-tactic":
                offer = self.reader.tactic_offer(frame, slot)
            else:
                offer = self.reader.bounties.offer(frame, slot)
            if offer is None or offer[0]["id" if equipment else "effectId"] != card_id:
                return None
            box = offer[1]
            if kind == "bountyConfirm":
                labels = frame.rows(f"Bounty.Confirm{BOUNTY_SLOTS[slot]}")
                if len(labels) != 1:
                    return None
                box = tuple(labels[0].box)
            return box
        if kind == "matchInfo":
            return self.point(kind) if frame.hit("Screen.MatchInfoButton") else None
        if kind == "enemyPreviewButton":
            return self.point("enemyPreview") if frame.hit("Screen.EnemyPreviewButton") else None
        if kind == "closeMatchInfo":
            if not frame.hit("Screen.MatchInfoCross"):
                return None
            return self.panel_result(frame, "inspect-info", self.point(kind))
        if kind in ("expectedRound", "rememberRound", "laterRound", "startedRound"):
            current = integer(frame.text("ArkRoundNumber"), r"回合([0-9]+)")
            if kind == "laterRound":
                advancing = current is not None and self.started_round is not None and current > self.started_round
                return hit if advancing else None
            if kind == "startedRound":
                if self.started_round is not None and (
                    current == self.started_round or current is None and frame.hit("Battle.LeaderActive")
                ):
                    snapshot = {"screen": "battle", "round": self.started_round, "canReady": False}
                    if current is None:
                        snapshot["leaderBattle"] = True
                    return self.AnalyzeResult(hit, {"snapshot": snapshot})
                return None
            if current is None or self.expected_round is not None and current != self.expected_round:
                return None
            if kind == "rememberRound":
                return self.AnalyzeResult(hit, {"readyRound": current})
            return hit

        if kind == "enemyPreviewScrolled":
            first_page_read = any(
                "inspection" in data
                for node in argv.task_detail.nodes if node.name == "ArkScrollEnemyPreview" and node.completed
                for data in recognition_data(node.recognition)
            )
            if first_page_read and self.reader is not None:
                return self.AnalyzeResult(self.point("enemyPreview"),
                                          {"scrolledEnemies": self.reader.read_enemy_kinds(frame)})
            return None

        if kind == "namedBond":
            return self.named_bond(frame)
        if kind == "snapshot":
            snapshot = screen_snapshot(frame)
            if snapshot.get("leaderBattle") and self.started_round is not None:
                snapshot["round"] = self.started_round
            stage = frame.stage_id() if snapshot["screen"] == "preparation" else None
            return self.AnalyzeResult(hit, {"snapshot": snapshot, "stageId": stage})
        screen = frame.scene()
        if kind == "preparation":
            return hit if screen == "preparation" else None
        if kind == "ready":
            return self.point("ready") if screen == "preparation" and frame.hit("Screen.ReadyColor") else None
        if kind == "enemyPreview":
            return self.panel_result(frame, "inspect-enemies", self.point(kind)) if screen == kind else None
        if kind == "bondDetail":
            target = self.selected_bond(frame) if screen == kind else None
            return self.panel_result(frame, "inspect-bond", target) if target else None
        return None


def run_navigation(tasker: Tasker, recognition: NavigationRecognition, timeout_ms: int, *,
                   checkpoint: str | None = None, scroll_preview: bool = False) -> None:
    entry, terminal = ENTRIES[recognition.action]
    if recognition.action == "inspect-enemies" and scroll_preview:
        entry = "ArkInspectEnemiesScrolled"
    if recognition.action == "enter-layout" and checkpoint is not None:
        entry = {"briefing": "Entry.EnterBriefing", "squadSelection": "Entry.EnterSquad"}[checkpoint]
    override = {
        "ArkReadyOnce": {"timeout": timeout_ms},
        "ArkConfirmReady": {"timeout": timeout_ms},
        "ArkWaitBattle": {"timeout": timeout_ms},
    }
    ready_action = ReadyAction(recognition) if recognition.action in {"start-battle", "ready-battle"} else None
    with task_recognition(tasker, "ArkNavigation", recognition, ready_action=ready_action):
        job = run_task(tasker, entry, override)
        steps = job.nodes
    recognition.collect_results(steps)
    last = job.last
    recognition.steps = job.names
    message = job.failure(ERRORS)
    if message:
        if last == "ArkReturnedRest":
            message += "; returned to the preparation anchor"
        raise RuntimeError(message)
    opening_stop = (
        recognition.action == "enter-layout"
        and recognition.snapshot is not None
        and last == {"strategySelection": "Entry.CommanderRequired", **ENTRY_CHECKPOINTS}.get(
            recognition.snapshot["screen"]
        )
    )
    battle_stop = recognition.action in BATTLE_ACTIONS and last in {
        "ArkRestAfterBattle", "ArkBountyAfterBattle", "ArkMatchEnded", "ArkDisconnected",
    }
    if not job.succeeded or (last != terminal and not opening_stop and not battle_stop) or recognition.snapshot is None:
        if recognition.action in DRAFT_ACTIONS:
            if any(node.name.endswith((".ClickSlot", ".ConfirmSlot")) and node.action is not None for node in steps):
                raise RuntimeError("Card selection did not return to the preparation anchor")
            raise RuntimeError("The card slot and ID did not match again; no input was sent")
        raise RuntimeError(f"Navigation stopped at {last or entry}")
    if (recognition.action not in BATTLE_ACTIONS | {"observe", "open-entrance", "leave-match", "interrupt-match", "recover-match"}
            and not opening_stop
            and recognition.snapshot["screen"] != "preparation"):
        raise RuntimeError("Navigation did not finish at the preparation anchor")


def observe_snapshot(tasker: Tasker) -> dict:
    """Read one live-loop screen snapshot without sending device input."""
    recognition = NavigationRecognition({}, "observe", None, None)
    try:
        run_navigation(tasker, recognition, 6000)
    except RuntimeError:
        return {"screen": "unknown", "round": None, "canReady": False}
    return recognition.snapshot


# Regions the waits watch. CENTER_ROI avoids the countdown (top right) and the latency readout (top left). SHOP_CARDS_ROI is
# the union of every card box in calib/slots.json shop layouts (the reward cards slide into this row).
CENTER_ROI = (200, 120, 880, 480)
SHOP_CARDS_ROI = (135, 520, 951, 184)


def wait_stable(tasker: Tasker, roi, *, target=None, time_ms: int = 150, timeout_ms: int = 3000) -> float:
    """Wait until the `target` box has not changed for `time_ms`, or until `timeout_ms`. No input is sent.

    The MFW node Wait.Stable only carries post_wait_freezes, so this returns as soon as the box is still and never later
    than timeout_ms. `target` must be explicit: MFW skips a wait_freezes that has no target box. Returns elapsed ms.
    """
    if target is None:
        raise ValueError("wait_stable requires an explicit target box")
    if time_ms <= 0 or timeout_ms <= 0:
        raise ValueError("wait_stable times must be positive")
    override = {"Wait.Stable": {
        "roi": list(roi), "timeout": int(timeout_ms),
        "post_wait_freezes": {"target": list(target), "time": int(time_ms), "rate_limit": 50, "threshold": 0.99,
                              "timeout": int(timeout_ms)},
    }}
    started = time.perf_counter()
    if not run_task(tasker, "Wait.Stable", override).succeeded:
        raise RuntimeError("Wait.Stable did not run")
    return (time.perf_counter() - started) * 1000


def observe_screen(tasker: Tasker) -> LiveScreen:
    return LIVE_SCREENS.get(observe_snapshot(tasker)["screen"], "unknown")


def enter_layout(tasker: Tasker, slots: dict, catalog: Path, before: dict, timeout_ms: int, *,
                 commander: str | None, resource_path: Path, stop_at: str | None,
                 bond_group: int | None, commander_fallbacks: tuple[str, ...] = (),
                 expected_groups: list[list[str]] | None = None) -> dict:
    screen = before["screen"]
    if stop_at is not None and stop_at not in ENTRY_CHECKPOINTS:
        raise ValueError("Opening checkpoint must be briefing or squadSelection")
    if bond_group is not None and (type(bond_group) is not int or bond_group not in (0, 1)):
        raise ValueError("Bond group must be 0 or 1")
    if stop_at == "briefing" and bond_group is not None:
        raise ValueError("Stopping before briefing confirmation conflicts with choosing a bond group")
    if (stop_at == "briefing" or bond_group is not None) and screen not in {
        "gameHome", "entrance", "difficultySelection", "briefing",
    }:
        raise RuntimeError(f"Briefing has already been passed on {screen}; no input was sent")
    if stop_at == "squadSelection" and screen == "preparation":
        raise RuntimeError("Squad selection has already been passed; no input was sent")

    result = {"before": before, "after": before, "steps": []}
    if screen == "preparation":
        require_return("enter-layout", before, before)
        return result

    recognition = NavigationRecognition(slots["buttons"], "enter-layout", None, None)
    checkpoint = "briefing" if bond_group is not None else stop_at
    run_navigation(tasker, recognition, timeout_ms, checkpoint=checkpoint)
    result["steps"] = recognition.steps.copy()
    screen = recognition.snapshot["screen"]
    if bond_group is not None:
        if screen != "briefing":
            raise RuntimeError(f"Opening did not reach briefing; stopped on {screen}")
        choice = choose_bond_group(tasker, Reader(catalog, slots), bond_group, expected_groups=expected_groups)
        result["bondChoice"] = choice["bondChoice"]
        result["steps"].extend(choice["steps"])
        screen = choice["after"]
    if screen == "strategySelection" and stop_at != "briefing":
        if not commander:
            raise RuntimeError("At commander selection; pass --commander with the displayed name to continue")
        result["commander"] = select_commander(tasker, slots["commander"], commander, resource_path=resource_path,
                                               fallback_names=commander_fallbacks, catalog=catalog)
        run_navigation(tasker, recognition, timeout_ms, checkpoint=stop_at)
        result["steps"].extend(recognition.steps)
        screen = recognition.snapshot["screen"]

    result["after"] = recognition.snapshot
    if stop_at is not None:
        if screen != stop_at:
            raise RuntimeError(f"Opening did not reach {stop_at}; stopped on {screen}")
    else:
        require_return("enter-layout", before, result["after"])
    return result


def execute(tasker: Tasker, slots: dict, action: str, catalog: Path, bond: str | None,
            timeout_ms: int, *, commander: str | None = None, resource_path: Path = OCR_RESOURCE,
            slot: int | None = None, bond_group: int | None = None, stop_at: str | None = None,
            battle_round: int | None = None, confirm_interrupt: bool = False,
            expected_round: int | None = None, scroll_preview: bool = False, expected_id: str | None = None,
            commander_fallbacks: tuple[str, ...] = (), expected_groups: list[list[str]] | None = None) -> dict:
    buttons = slots["buttons"]
    probe = NavigationRecognition(buttons, "observe", None, None)
    run_navigation(tasker, probe, timeout_ms)
    before = probe.snapshot
    if action == "observe":
        return before

    require_action(action, before)
    if action == "recover-match":
        if type(expected_round) is not int or expected_round < 1:
            raise ValueError("Reconnection requires the last known round")
        recognition = NavigationRecognition(buttons, action, None, None, expected_round)
        run_navigation(tasker, recognition, timeout_ms)
        return {"before": before, "after": recognition.snapshot, "steps": recognition.steps}
    if action in {"leave-match", "interrupt-match", "restart-match"}:
        if action != "leave-match" and confirm_interrupt is not True:
            raise ValueError("Pass confirm_interrupt=True to settle the current saved match")
        if action == "restart-match":
            if stop_at not in (None, *ENTRY_CHECKPOINTS):
                raise ValueError("Opening checkpoint must be briefing or squadSelection")
            if stop_at == "briefing" and bond_group is not None:
                raise ValueError("Stopping at briefing and confirming a bond group are mutually exclusive")
            if stop_at != "briefing" and not clean(commander or ""):
                raise ValueError("Supply the new commander name or stop at briefing")
            if bond_group is not None and (type(bond_group) is not int or bond_group not in (0, 1)):
                raise ValueError("Bond group must be 0 or 1")
        nav_action = "leave-match" if action == "leave-match" else "interrupt-match"
        recognition = NavigationRecognition(buttons, nav_action, None, None, before["round"])
        run_navigation(tasker, recognition, timeout_ms)
        interruption = {"before": before, "after": recognition.snapshot, "steps": recognition.steps}
        if action != "restart-match":
            return interruption
        try:
            opening = enter_layout(tasker, slots, catalog, recognition.snapshot, timeout_ms,
                                   commander=commander, resource_path=resource_path, stop_at=stop_at,
                                   bond_group=bond_group)
        except RuntimeError as error:
            raise RuntimeError(f"The fresh entrance was verified; continue opening with enter-layout: {error}") from error
        if stop_at is None and opening["after"]["round"] != 1:
            raise RuntimeError("Restart did not reach round 1")
        return {"interruption": interruption, "opening": opening, "after": opening["after"]}
    if action in BRIEFING_ACTIONS:
        reader = Reader(catalog, slots)
        if action == "inspect-briefing":
            return inspect_briefing(tasker, reader)
        if bond_group is None:
            raise ValueError("select-bond-group requires group 0 or 1")
        return choose_bond_group(tasker, reader, bond_group)
    if action == "close-panel" and before["screen"] == "preparation":
        return before
    if action == "enter-layout":
        return enter_layout(tasker, slots, catalog, before, timeout_ms, commander=commander,
                            resource_path=resource_path, stop_at=stop_at, bond_group=bond_group,
                            commander_fallbacks=commander_fallbacks, expected_groups=expected_groups)
    if action == "wait-battle":
        if type(battle_round) is not int or battle_round < 1:
            raise ValueError("wait-battle requires a positive battle round")
        if before["screen"] in DRAFT_SCREENS | {"preparation", "matchInterrupted", "disconnect"}:
            require_return(action, {"round": battle_round}, before)
            return {"startedRound": battle_round, "after": before}
        transition = before["screen"] in {"roundHud", "unknown"} and before["round"] in (
            None, battle_round, battle_round + 1)
        if not (before["round"] == battle_round or before.get("leaderBattle") or transition):
            raise RuntimeError("The current battle round does not match the requested round")
        recognition = NavigationRecognition(buttons, action, None, None)
        recognition.started_round = battle_round
        run_navigation(tasker, recognition, timeout_ms)
        require_return(action, {"round": battle_round}, recognition.snapshot)
        return {"startedRound": battle_round, "after": recognition.snapshot}

    if action in DRAFT_ACTIONS:
        if type(slot) is not int or not 0 <= slot < len(BOUNTY_SLOTS):
            raise ValueError("Card selection requires a zero-based slot from left to right")
        if expected_round is not None:
            if before["round"] not in (None, expected_round):
                raise RuntimeError("The card draft round differs from the live match")
            before = {**before, "round": expected_round}
        reader = Reader(catalog, slots)
        state = read_screen(tasker, reader)
        if action == "select-equipment":
            card = next((card for card in state.get("equipmentOffers", []) if card["idx"] == slot), None)
            if not expected_id or not card or card["id"] != expected_id:
                raise RuntimeError("The equipment offer changed or could not be read; no input was sent")
        else:
            draft = state["match"]["sp"]
            cards = (draft or {}).get("cards") or []
            card = cards[slot] if slot < len(cards) else None
            if not card or expected_id is not None and card["id"] != expected_id:
                raise RuntimeError("The requested draft offer changed or could not be read; no input was sent")
        card_id = card["id"]
        recognition = NavigationRecognition(buttons, action, None, reader, before["round"], bounty=(slot, card_id))
        run_navigation(tasker, recognition, timeout_ms)
        after = recognition.snapshot
        require_return(action, before, after)
        if after["round"] is None:
            raise RuntimeError("Card confirmation returned to preparation, but the round was unreadable")
        return {"slot": slot, "cardId": card_id, "card": card, "round": after["round"],
                "before": before, "after": after, "steps": recognition.steps}

    reader = Reader(catalog, slots) if action in INSPECTIONS else None
    recognition = NavigationRecognition(buttons, action, bond, reader, before["round"])
    run_navigation(tasker, recognition, timeout_ms, scroll_preview=action == "inspect-enemies" and scroll_preview)
    after = recognition.snapshot
    if action == "open-entrance":
        if after["screen"] != "entrance":
            raise RuntimeError("The event entrance was not verified")
        return {"before": before, "after": after, "steps": recognition.steps}
    require_return(action, before, after)

    if reader is not None:
        inspection = recognition.inspection
        if inspection is None:
            raise RuntimeError(recognition.inspection_error or "Inspection frame was not recognized")
        if action == "inspect-bond":
            ids = {key for key, _ in reader.bonds.get(bond, [])}
            actual = {row["bondId"] for row in inspection["self"]["bonds"] or []}
            if not ids.intersection(actual):
                raise RuntimeError("Bond detail did not match the requested bond; returned to preparation")
        result = {"inspection": inspection, "after": after}
        if action == "inspect-enemies":
            result["kinds"] = inspection.pop("enemyKinds")
            result["ambiguousEnemyNames"] = inspection.pop("ambiguousEnemyNames")
            result["wave"] = match_preview(inspection, probe.stage_id, before["round"])
        return result
    if action in {"start-battle", "ready-battle"}:
        return {"startedRound": recognition.started_round, "after": after}
    return after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(*ENTRIES, *BRIEFING_ACTIONS, "restart-match"))
    parser.add_argument("--bond", help="Displayed bond name for inspect-bond")
    parser.add_argument("--commander", help="Displayed commander name for a new run with enter-layout")
    parser.add_argument("--slot", type=int, help="Zero-based draft slot from left to right")
    parser.add_argument("--expected-id", help="Verified offer ID for card selection")
    parser.add_argument("--expected-round", type=int, help="Last known round for recover-match")
    parser.add_argument("--bond-group", type=int, choices=(0, 1), help="Briefing group from left to right")
    parser.add_argument("--stop-at", choices=ENTRY_CHECKPOINTS, help="Stop before confirming an opening page")
    parser.add_argument("--battle-round", type=int, help="Original round for wait-battle")
    parser.add_argument("--confirm-interrupt", action="store_true",
                        help="Allow the current saved simulation to settle and be removed")
    parser.add_argument("--timeout", type=float, default=95, help="Seconds to verify a later preparation round")
    add_session_args(parser, catalog=CATALOG)
    args = parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    bond = clean(args.bond) if args.bond else None
    commander = clean(args.commander) if args.commander else None
    if args.action == "inspect-bond" and not bond:
        parser.error("inspect-bond requires --bond")
    if args.commander is not None and (args.action not in {"enter-layout", "restart-match"} or not commander):
        parser.error("--commander requires enter-layout or restart-match and a displayed name")
    card_action = args.action in DRAFT_ACTIONS
    if card_action and args.slot not in range(len(BOUNTY_SLOTS)):
        parser.error("Card selection requires --slot 0, 1 or 2")
    if not card_action and args.slot is not None:
        parser.error("--slot requires a card selection action")
    if args.action in {"select-equipment", "select-tactic"} and not args.expected_id:
        parser.error("--expected-id is required for select-equipment and select-tactic")
    if args.expected_id and not card_action:
        parser.error("--expected-id requires a card selection action")
    if (args.action == "recover-match") != (args.expected_round is not None):
        parser.error("--expected-round is required for recover-match")
    if args.action == "select-bond-group" and args.bond_group is None:
        parser.error("--bond-group is required for select-bond-group")
    if args.bond_group is not None and args.action not in {"select-bond-group", "enter-layout", "restart-match"}:
        parser.error("--bond-group requires select-bond-group, enter-layout or restart-match")
    if args.stop_at is not None and args.action not in {"enter-layout", "restart-match"}:
        parser.error("--stop-at requires enter-layout or restart-match")
    if args.confirm_interrupt != (args.action in {"interrupt-match", "restart-match"}):
        parser.error("--confirm-interrupt is required for interrupt-match or restart-match")
    if (args.action == "wait-battle") != (args.battle_round is not None):
        parser.error("--battle-round is required for wait-battle")
    try:
        tasker = session(args.adb, args.address, args.ocr_resource).tasker
        slots = read_json(ROOT / "calib/slots.json")
        result = execute(tasker, slots, args.action, args.catalog, bond, max(1, round(args.timeout * 1000)),
                         commander=commander, resource_path=args.ocr_resource, slot=args.slot,
                         bond_group=args.bond_group, stop_at=args.stop_at, battle_round=args.battle_round,
                         confirm_interrupt=args.confirm_interrupt, expected_id=args.expected_id,
                         expected_round=args.expected_round)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
