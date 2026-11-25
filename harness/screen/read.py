"""Read one 1280×720 ArkHarness ScreenState without sending game input."""

import argparse
import json
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.tasker import Tasker
from PIL import Image

from .bounty import SLOTS as BOUNTY_SLOTS, BountyCatalog
from .capture import add_session_args, session
from .native import CHECK_NUMBER_BELOW, MIN_NAME_SCORE, MIN_NUMBER_SCORE, Frame, OfflineController, completed_data
from .names import match_name
from .session import run as run_task, task_recognition
from .shop_portraits import encode_portrait, match_portrait, save_portraits


ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
SHOP_TIER_BADGES = {f"Shop.CardTier.{number}": number for number in range(1, 7)}
SHOP_LEVEL_BADGES = {f"Shop.LevelBadge.{number}": number for number in range(1, 7)}
SHOP_PORTRAITS = ROOT / "resource/image/portrait"
CATALOG = Path(os.environ.get("ARK_CATALOG", "E:/ArkHarness-data/act3autochess-v3_1"))
CONFIG_FIELDS = ("modeId", "difficulty", "lastRound", "bossRound", "hiddenRound")
MATCH_FIELDS = ("round", "stageId", "bossId", "hiddenBossId", "disabledBonds", "bondChoice", "spRound", "sp")
SELF_FIELDS = ("playerId", "lp", "funds", "bandId", "canReady", "deployCap", "deployCount",
               "personalChoice", "hand", "temp", "board", "bonds", "nextEnemies")
SHOP_FIELDS = ("level", "maxLevel", "upgradePrice", "refreshPrice", "freeRefreshes", "frozen",
               "slots", "rewardOffer")
CARD_FIELDS = ("idx", "kind", "id", "tier", "coin", "payout", "rounds", "enemyKey", "count",
               "soldOut", "team")
ROMAN = dict(zip(("I", "II", "III", "IV", "V", "VI"), range(1, 7)))
SHOP_UPGRADE_CENTERS = (63, 225, 387)


def clean(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("：", ":")


def shop_frozen(frame: Frame) -> bool | None:
    return {"冻结": False, "解冻": True, "解除冻结": True}.get(clean(frame.text("Shop.FreezeText")))


def shop_reward_phase(frame: Frame) -> bool | None:
    """Classify the freeze button on an already verified expanded shop."""
    if frame.hit("Shop.FreezeDisabledColor"):
        return True
    if frame.hit("Shop.FreezeEnabledColor"):
        return False
    return None


def shop_reward_choices(slots: list[dict | None], *, min_cards: int = 3,
                        kind: str = "chess", reward_phase: bool = False) -> list[tuple[int, dict]]:
    """Identify free rewards of the expected kind and their engine slots."""
    if not reward_phase and any(slot is None for slot in slots):
        return []
    choices = [(index, slot) for index, slot in enumerate(slots) if slot is not None and not slot["sold"]]
    if reward_phase:
        if any((card["price"] or 0) > 0 or (card["kind"] == kind and card["price"] is None)
               for _, card in choices):
            return []
        choices = [(index, card) for index, card in choices if card["kind"] == kind and card["price"] == 0]
    if not min_cards <= len(choices) <= 3 or any(card["kind"] != kind or card["price"] != 0
                                                 for _, card in choices):
        return []
    identified = [(index, card) for index, card in choices if card["id"]]
    return identified if len({card["id"] for _, card in identified}) == len(identified) else []


def shop_confirm_box(frame: Frame, position: dict) -> tuple | None:
    x, y, width, _ = position["nameROI"]
    labels = frame.rows("Shop.ConfirmBuyText", roi=[x - 8, y - 24, width + 24, 57])
    return tuple(labels[0].box) if len(labels) == 1 else None


def integer(text: str, pattern: str = r"([0-9]+)") -> int | None:
    match = re.fullmatch(pattern, clean(text))
    return int(match[1]) if match else None


def screen_integer(frame: Frame, node: str, pattern: str = r"([0-9]+)", **override) -> int | None:
    text = frame.text(node, min_score=MIN_NUMBER_SCORE, confirm_below=CHECK_NUMBER_BELOW, **override)
    if not text and node == "Screen.Funds" and not override and frame.hit("Screen.FundsZero"):
        text = "0"
    return integer(text, pattern)


def tier(text: str) -> int | None:
    label = clean(unicodedata.normalize("NFKC", text)).upper()
    number = integer(label)
    return number if number in ROMAN.values() else ROMAN.get(label)


def shop_card_tier(frame: Frame, roi: list[int]) -> int | None:
    x, y, width, _ = roi
    search = [x - 2, y + 2, width + 4, 24]
    badge = frame.first_hit("Shop.CardTierBadge", overrides={name: {"roi": search} for name in SHOP_TIER_BADGES})
    return SHOP_TIER_BADGES.get(badge) or tier(frame.text("Screen.ShopCardTier", roi=roi))


def shop_upgrade_center(frame: Frame) -> int | None:
    headings = frame.rows("Shop.LevelHeading")
    if len(headings) == 1:
        x, _, width, _ = headings[0].box
        heading_center = x + width // 2
        for center in SHOP_UPGRADE_CENTERS:
            if abs(heading_center - center) <= 28:
                return center
    # The upgrade card moves left by one slot as the shop grows from four to six cards.
    centers = [x for x in SHOP_UPGRADE_CENTERS if frame.hit(
        "Shop.LevelHeading", roi=[x - 26, 567, 52, 18], only_rec=True, expected="^LEVEL$",
    )]
    return centers[0] if len(centers) == 1 else None


def expanded_shop_level(frame: Frame) -> int | None:
    if level := SHOP_LEVEL_BADGES.get(frame.first_hit("Shop.LevelBadge")):
        return level
    center = shop_upgrade_center(frame)
    if center is None:
        return None
    # II and III share the same position. A narrow center strip distinguishes their printed strokes.
    if 190 <= center <= 260 and all(
        frame.hit("Screen.ShopLevelStroke", roi=[center + offset, 590, 16, 29], count=105)
        for offset in (-22, 6)
    ):
        return 3 if frame.hit("Screen.ShopLevelStroke", roi=[center - 2, 590, 5, 29], count=85) else 2
    half_width = 30 if center == SHOP_UPGRADE_CENTERS[0] else 22
    level = tier(frame.text("Screen.ShopLevelExpanded", roi=[center - half_width, 584, 2 * half_width, 43]))
    if level is None:
        # The wider crop can pick up neighboring strokes on the level III badge.
        level = tier(frame.text("Screen.ShopLevelExpanded", roi=[center - 19, 585, 38, 43]))
    return level


def shop_upgrade_price(frame: Frame, *, level: int | None = None) -> int | None:
    center = (SHOP_UPGRADE_CENTERS[2 if level == 1 else 1 if level <= 3 else 0]
              if level in ROMAN.values() else shop_upgrade_center(frame))
    if center is None:
        return None
    return screen_integer(frame, "Screen.UpgradePrice", roi=[center - 13, 522, 26, 26])


def shop_refresh_price(frame: Frame) -> int | None:
    price = screen_integer(frame, "Screen.RefreshPrice")
    return 0 if price is None and frame.hit("Shop.FreeRefresh") else price


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def index_names(rows: dict) -> dict:
    names = defaultdict(list)
    for key, row in rows.items():
        if row.get("name"):
            names[clean(row["name"])].append((key, row))
    return names


def expected_names(names) -> str:
    labels = [r"\s*".join(re.escape(c) if c != ":" else "[:：]" for c in name) for name in sorted(names)]
    return r"^\s*(?:" + "|".join(labels) + r")\s*$" if labels else r"(?!)"


def chess_identity(reader: "Reader", frame: Frame) -> dict:
    identity = catalog_identity(reader, frame, "Board.CarriedName", "chess")
    if not identity["candidateIds"]:
        tokens = reader.tokens.get(identity["name"], [])
        if tokens:
            ids = [key for key, _ in tokens]
            token_id = ids[0] if len(ids) == 1 else None
            return {**identity, "kind": "token", "id": token_id, "baseId": token_id,
                    "candidateIds": ids}
    return identity


def normalize_item_description(text: str) -> str:
    return clean(unicodedata.normalize("NFKC", text)).replace(",", "").replace("。", "")


def catalog_identity(reader: "Reader", frame: Frame, node: str, kind: str) -> dict:
    if kind == "chess":
        rows = frame.rows(node, expected=reader.chess_detail_expected)
        names = {clean(row.text) for row in rows if row.score >= MIN_NAME_SCORE}
        names.intersection_update(reader.chess_detail_names)
    else:
        names = {clean(row.text) for row in frame.rows(node)} - {""}
    entries = {
        name: [(entity_id, row)
               for entry_kind, entity_id, row in reader.shop_names.get(name, []) if entry_kind == kind]
        for name in names
    }
    matched = {name for name, rows in entries.items() if rows}
    choices = matched or names
    name = next(iter(choices)) if len(choices) == 1 else None
    candidates = entries.get(name, [])
    if len(candidates) > 1:
        if kind == "chess":
            levels = frame.rows("Board.OperatorLevel")
            level = integer(levels[0].text) if len(levels) == 1 and levels[0].score >= 0.9 else None
            exact = [entry for entry in candidates if level is not None and entry[1].get("status", {}).get("level") == level]
        else:
            lines = sorted(frame.rows("EquipmentRead.Description"), key=lambda row: (row.box[1], row.box[0]))
            description = normalize_item_description("".join(row.text for row in lines))
            exact = [entry for entry in candidates if description and
                     normalize_item_description(entry[1].get("desc") or "") == description]
        if len(exact) == 1:
            candidates = exact
    ids = sorted({entity_id for entity_id, _ in candidates})
    bases = {row.get("baseId") or entity_id for entity_id, row in candidates}
    return {"name": name, "id": ids[0] if len(ids) == 1 else None,
            "baseId": next(iter(bases)) if len(bases) == 1 else None, "candidateIds": ids}


def unread_fields(value: Any, path: str = "") -> set[str]:
    if value is None:
        return {path}
    if isinstance(value, dict):
        return {missing for key, item in value.items()
                for missing in unread_fields(item, f"{path}.{key}" if path else key)}
    if isinstance(value, list):
        return {missing for index, item in enumerate(value)
                for missing in unread_fields(item, f"{path}[{index}]")}
    return set()


class Reader:
    stage_hint: str | None = None
    shop_level_hint: int | None = None
    deploy_cap_floor: int = 0
    reward_kind_hint: str | None = None

    def __init__(self, catalog: Path, slots: dict):
        self.slots = slots
        self.portrait_rois = slots.get("shop", {}).get("portraits", {})
        self.modes = index_names(read_json(catalog / "config.json")["modes"])
        self.stages = read_json(catalog / "stages.json")
        self.bonds = index_names(read_json(catalog / "bonds.json"))
        self.core_bonds = {name: [(key, row) for key, row in entries if row.get("isCore")]
                           for name, entries in self.bonds.items() if any(row.get("isCore") for _, row in entries)}
        self.bosses = index_names(read_json(catalog / "bosses.json"))
        self.tokens = index_names(read_json(catalog / "tokens.json"))
        enemies = read_json(catalog / "enemies.json")
        self.enemies = index_names(enemies)
        self.shop_names = {}
        self.shop_base_names = {"chess": {}, "item": {}}
        self.shop_by_tier_price = defaultdict(dict)
        saved_portraits = {path.stem for path in SHOP_PORTRAITS.glob("*.png")} if self.portrait_rois else set()
        self.shop_portrait_ids = defaultdict(list)
        for kind, file in (("chess", "chess.json"), ("item", "items.json")):
            by_name = index_names(read_json(catalog / file))
            if kind == "chess":
                self.chess_detail_names = by_name.keys() | self.tokens.keys()
                self.chess_detail_expected = expected_names(self.chess_detail_names)
            for name, entries in by_name.items():
                self.shop_names.setdefault(name, []).extend((kind, key, row) for key, row in entries)
                base = [(key, row) for key, row in entries if row.get("isGolden") is not True]
                if base:
                    self.shop_base_names[kind][name] = base
                for key, row in base:
                    if key in saved_portraits:
                        self.shop_portrait_ids[kind, row.get("tier")].append(key)
                    for price in {0, row.get("price")} - {None}:
                        bucket = self.shop_by_tier_price[kind, row.get("tier"), price]
                        bucket.setdefault(name, []).append((key, row))
        self.shop_name_expected = {kind: expected_names(names) for kind, names in self.shop_base_names.items()}
        raw_cards = read_json(catalog / "choices.json")["cards"]
        self.bounties = BountyCatalog(raw_cards.get("bounty", []))
        self.tactics = index_names({row["effectId"]: row for row in raw_cards.get("tactic", [])})
        self.cards = index_names({row["effectId"]: row for group in raw_cards.values()
                                  for row in group if row.get("effectId")})

    def read_mode(self, frame: Frame) -> dict:
        label = clean(frame.text("Screen.Mode"))
        rows = self.modes.get(label, [])
        if not rows and (match := re.fullmatch(r".{1,2}(规程等级[0-9]+)", label)):
            rows = self.modes.get(match[1], [])
        if not rows:
            # A neighboring regulation number represents a different mode.
            rows = self.modes.get(match_name(label, self.modes, approximate=not any(map(str.isdigit, label))), [])
        return rows[0][1] if len(rows) == 1 else {}

    def read(self, frame: Frame) -> dict:
        config = dict.fromkeys(CONFIG_FIELDS)
        match = dict.fromkeys(MATCH_FIELDS)
        own = dict.fromkeys(SELF_FIELDS)
        shop = dict.fromkeys(SHOP_FIELDS)
        own["shop"] = shop
        state = {"config": config, "match": match, "self": own}
        scene = frame.scene()
        known_absent = set()
        incomplete = set()

        mode = self.read_mode(frame)
        for key in config:
            config[key] = mode.get(key)
        if scene in ("preparation", "bondDetail", "drag"):
            match["stageId"] = self.stage_hint or frame.stage_id()

        if scene in ("preparation", "bondDetail", "enemyPreview", "drag"):
            match["round"] = screen_integer(frame, "ArkRoundNumber", r"回合([0-9]+)")
            own["lp"] = integer(frame.text("Screen.Life"))
            schedule = mode.get("rounds", {}).get(str(match["round"]))
            if schedule is not None:
                match["spRound"] = schedule.get("isSpPrepare")
                if match["spRound"] is False:
                    known_absent.add("match.sp")
            if match["round"] is not None and config["hiddenRound"] is not None:
                if match["round"] < config["hiddenRound"]:
                    known_absent.add("match.hiddenBossId")

        if scene in ("preparation", "bondDetail"):
            if frame.hit("Screen.ReadyIcon") and frame.hit("Screen.ReadyColor"):
                own["canReady"] = True
            own["funds"] = screen_integer(frame, "Screen.Funds")
            expanded = self.read_shop(frame, shop, mode, incomplete)
            if scene == "preparation":
                self.read_deployment(frame, own, match["stageId"], expanded)
            leader = match["round"] is not None and match["round"] in (config["bossRound"], config["hiddenRound"])
            own["bonds"] = self.read_bonds(frame, scene, leader=leader)
            if own["bonds"] is not None:
                incomplete.add("self.bonds")
        elif scene == "enemyPreview":
            state["enemyKinds"], state["ambiguousEnemyNames"] = self.read_enemy_kinds(frame)
            current_round = match["round"]
            if current_round is not None and current_round in (config["bossRound"], config["hiddenRound"]):
                hidden = current_round == config["hiddenRound"]
                names = {clean(row.text) for row in frame.rows("Screen.LeaderName") if row.score >= MIN_NAME_SCORE}
                leaders = {name: [(key, row) for key, row in candidates if row.get("hidden") is hidden]
                           for name, candidates in self.bosses.items()
                           if any(row.get("hidden") is hidden for _, row in candidates)}
                bosses = {key: data for name in names for key, data in leaders.get(match_name(name, leaders), [])}
                if len(bosses) == 1:
                    key, boss = bosses.popitem()
                    match["hiddenBossId" if hidden else "bossId"] = key
                    if enemy_key := boss.get("enemyKey"):
                        state["enemyKinds"] = sorted({*state["enemyKinds"], enemy_key})
        elif scene == "equipmentSelection":
            state["equipmentOffers"] = [offer[0] for slot in range(len(BOUNTY_SLOTS))
                                        if (offer := self.equipment_offer(frame, slot))]
        elif scene in ("bountySelection", "tacticSelection"):
            match["round"] = screen_integer(frame, "ArkRoundNumber", r"回合([0-9]+)")
            match["spRound"] = True
            kind = "tactic" if scene == "tacticSelection" else "bounty"
            read_offer = self.tactic_offer if kind == "tactic" else self.bounties.offer
            offers = [read_offer(frame, slot) for slot in range(len(BOUNTY_SLOTS))]
            if any(offers):
                cards = [None] * len(offers)
                for slot, offer in enumerate(offers):
                    if offer:
                        data, _ = offer
                        card = {field: data.get(field) for field in CARD_FIELDS}
                        cards[slot] = {**card, "idx": slot, "kind": kind, "id": data["effectId"], "soldOut": False}
                match["sp"] = {"cards": cards, "taken": None}
            if not all(offers):
                incomplete.add("match.sp.cards")
        elif scene is None:
            self.read_panels(frame, match, incomplete)

        state["unread"] = sorted((unread_fields(state) | incomplete) - known_absent)
        return state

    def tactic_offer(self, frame: Frame, slot: int) -> tuple[dict, tuple] | None:
        suffix = BOUNTY_SLOTS[slot]
        rows = sorted(frame.rows(f"Bounty.Name{suffix}"), key=lambda row: (row.box[1], row.box[0]))
        name = clean("".join(row.text for row in rows))
        candidates = self.tactics.get(name, [])
        if len(candidates) != 1:
            return None
        return candidates[0][1], tuple(rows[0].box)

    def equipment_offer(self, frame: Frame, slot: int) -> tuple[dict, tuple] | None:
        suffix = BOUNTY_SLOTS[slot]
        rows = sorted(frame.rows(f"Bounty.Name{suffix}"), key=lambda row: (row.box[1], row.box[0]))
        if not rows:
            return None
        name = clean("".join(row.text for row in rows))
        items = [(key, data) for kind, key, data in self.shop_names.get(name, ())
                 if kind == "item" and not data.get("isGolden")]
        if len(items) != 1:
            return None
        key, item = items[0]
        return {"idx": slot, "id": key, "name": name, "tier": item.get("tier"), "kind": item.get("kind")}, tuple(rows[0].box)

    def read_bonds(self, frame: Frame, scene: str, *, leader: bool = False) -> list[dict] | None:
        if scene == "bondDetail":
            records = [(frame.text("Screen.BondDetailName", min_score=MIN_NAME_SCORE),
                        integer(frame.text("Screen.BondDetailLayers"), r"层数([0-9]+)"),
                        frame.text("Screen.BondMembers"), False)]
        else:
            records = [(name, integer(value or ""), "", value is None)
                       for name, value, _ in frame.visible_bonds(leader=leader)]
        bonds = []
        for name, layers, members, unverified in records:
            entries = self.bonds.get(match_name(clean(name), self.bonds), [])
            member = re.match(r"(\d+)/(\d+)(尚未激活|已激活)", clean(members))
            active = member[3] == "已激活" if member else None
            bonds.append({
                "bondId": entries[0][0] if len(entries) == 1 else None,
                "count": int(member[1]) if member else None,
                "active": active,
                "tier": 0 if active is False else None,
                "layers": layers,
            })
            if unverified and bonds[-1]["bondId"] is not None:
                bonds[-1]["layerReadFailed"] = True
        return bonds or None

    def read_deployment(self, frame: Frame, own: dict, stage_id: str | None, expanded: bool) -> None:
        stage = self.stages.get(stage_id, {})
        cap = stage.get("options", {}).get("characterLimit")
        if cap is None:
            return
        cap = max(cap, self.deploy_cap_floor)
        # The limit is the stage's catalog value, so it does not depend on the remaining-count OCR succeeding.
        own["deployCap"] = cap
        remaining = screen_integer(frame, "Screen.DeployExpanded" if expanded else "Screen.DeployCollapsed")
        if remaining is not None and 0 <= remaining <= cap:
            own["deployCount"] = cap - remaining

    def read_enemy_kinds(self, frame: Frame) -> tuple[list[str], dict[str, list[str]]]:
        kinds = set()
        ambiguous = {}
        for row in frame.rows("Screen.EnemyNames"):
            if row.score < MIN_NAME_SCORE:
                continue
            label = clean(row.text).lstrip("•·∙-")
            label = match_name(label, self.enemies)
            candidates = self.enemies.get(label, [])
            if len(candidates) == 1:
                kinds.add(candidates[0][0])
            elif candidates:
                ambiguous[label] = sorted(key for key, _ in candidates)
        return sorted(kinds), ambiguous

    @staticmethod
    def panel_rows(frame: Frame) -> list:
        """Text rows of a panel. A caller that knows the layout (the briefing page) sets `frame.text_rois` and pays for
        those areas only; everything else keeps the single full-screen pass."""
        rois = getattr(frame, "text_rois", None)
        if not rois:
            return frame.rows("Screen.FullText")
        return [row for roi in rois for row in frame.rows("Screen.FullText", roi=roi)]

    def read_panels(self, frame: Frame, match: dict, incomplete: set[str]) -> None:
        rows = self.panel_rows(frame)
        briefing = any("确认本局信息" in clean(row.text) for row in rows)
        summary = briefing or any("附加盟约" == clean(row.text) for row in rows)
        if summary:
            for row in rows:
                x, y, _, _ = row.box
                if row.score < MIN_NAME_SCORE or not (x >= 920 and 300 <= y <= 430):
                    continue
                candidates = self.bosses.get(match_name(clean(row.text), self.bosses), [])
                ordinary = [key for key, data in candidates if data.get("hidden") is False]
                if len(ordinary) == 1:
                    match["bossId"] = ordinary[0]
            detected = []
            enabled = set()
            groups = [[], []]
            for row in rows:
                x, y, width, _ = row.box
                if row.score < MIN_NAME_SCORE or x >= 890 or not 250 <= y <= 465:
                    continue
                candidates = self.core_bonds.get(match_name(clean(row.text), self.core_bonds), [])
                if len(candidates) != 1:
                    continue
                key = candidates[0][0]
                detected.append(key)
                center = x + width // 2
                if briefing:
                    # Row first, then left to right: OCR box heights jitter by a pixel or two within one printed row.
                    groups[int(center >= 445)].append((0 if y < 360 else 1, center, key))
                if frame.hit("Screen.EnabledBond", roi=[max(0, center - 29), y - 92, 58, 68]):
                    enabled.add(key)
            if len(set(detected)) == len(detected) == 10:
                if briefing and all(len(group) == 5 for group in groups):
                    group_ids = [[key for _, _, key in sorted(group)] for group in groups]
                    match["bondChoice"] = {"groups": group_ids, "picked": None}
                    if any(set(group) == enabled for group in group_ids):
                        match["disabledBonds"] = [key for key in detected if key not in enabled]
                elif not briefing and len(enabled) == 5:
                    match["disabledBonds"] = [key for key in detected if key not in enabled]
            return

        if not any("机变" in row.text for row in rows):
            return
        cards = []
        for row in sorted(rows, key=lambda item: (item.box[0], item.box[1])):
            entry = self.cards.get(clean(row.text), [])
            if len(entry) != 1:
                continue
            key, data = entry[0]
            card = {field: data.get(field) for field in CARD_FIELDS}
            card["id"] = key
            cards.append(card)
        if cards:
            match["spRound"] = True
            match["sp"] = {"cards": cards, "taken": None}
            incomplete.add("match.sp.cards")

    def read_shop(self, frame: Frame, shop: dict, mode: dict, incomplete: set[str]) -> bool:
        expanded = frame.hit("Shop.Expanded")
        shop["level"] = self.shop_level_hint
        if shop["level"] is None:
            shop["level"] = expanded_shop_level(frame) if expanded else tier(frame.text("Screen.ShopLevelCollapsed"))
        shop["maxLevel"] = mode.get("maxShopLevel")
        if not expanded:
            return False
        if frame.hit("Shop.ConfirmLevelText"):
            shop["levelPreview"] = True
            return True
        shop["upgradePrice"] = shop_upgrade_price(frame, level=shop["level"])
        shop["refreshPrice"] = shop_refresh_price(frame)
        shop["frozen"] = shop_frozen(frame)
        reward_phase = shop_reward_phase(frame) is True
        if reward_phase:
            shop["rewardPending"] = True
        positions = self.shop_positions(mode, shop["level"])
        if not positions:
            return True
        output = self.read_shop_slots(frame, positions, reward_kind=self.reward_kind_hint, reward_phase=reward_phase)
        if any(value is None for value in output):
            incomplete.add("self.shop.slots")
        shop["slots"] = output
        choices = shop_reward_choices(output, min_cards=1 if reward_phase else 3,
                                      kind=self.reward_kind_hint or "chess", reward_phase=reward_phase)
        if choices:
            shop["rewardOffer"] = {"slots": [card for _, card in choices]}
        return True

    def read_shop_slots(self, frame: Frame, positions: list[dict], *, reward_kind: str | None = None,
                        reward_phase: bool = False) -> list[dict | None]:
        output = [None] * len(positions)
        for position in positions:
            output[position["engineIndex"]] = self.read_shop_slot(frame, position, reward_kind=reward_kind,
                                                                reward_phase=reward_phase)
        return output

    def shop_positions(self, mode: dict, level: int | None) -> list[dict]:
        shop = self.slots.get("shop", {})
        level_key = str(level)
        layout_key = shop.get("layoutAliases", {}).get(level_key, level_key)
        positions = shop.get("layouts", {}).get(layout_key, [])
        expected = mode.get("shopSlots", {}).get(level_key)
        if not expected or Counter(row["kind"] for row in positions) != Counter(expected):
            return []
        indices = {row["engineIndex"] for row in positions}
        return positions if indices == set(range(len(positions))) else []

    def portrait_roi(self, position: dict, kind: str) -> list[int]:
        left, top, width, height = self.portrait_rois[kind]
        return [position["center"][0] + left, top, width, height]

    def shop_portraits(self, frame: Frame, shop: dict, known: frozenset[str]) -> dict[str, str]:
        slots = shop.get("slots")
        if (not slots or frame.scene() != "preparation"
                or frame.hit("Shop.ConfirmLevelText")):
            return {}
        phase = shop_reward_phase(frame)
        if phase is None or (phase is False and (shop.get("rewardPending") or shop.get("rewardOffer"))):
            return {}
        positions = self.shop_positions(self.read_mode(frame), shop["level"])
        if phase and any(shop_confirm_box(frame, position) for position in positions):
            return {}
        portraits = {}
        for position in positions:
            card = slots[position["engineIndex"]]
            if not card or card["sold"] or (key := card["id"]) is None:
                continue
            if key in known or key in portraits or (not phase and shop_confirm_box(frame, position)):
                continue
            if phase and (card["kind"] != "chess" or card["price"] != 0):
                continue
            badge = "Screen.ShopRewardBadge" if phase else "Shop.PortraitLit"
            if not frame.hit(badge, roi=position["priceROI"]):
                continue
            name = clean(frame.text("Screen.ShopCardName", roi=position["nameROI"], min_score=MIN_NAME_SCORE,
                                    expected=self.shop_name_expected[card["kind"]]))
            exact = self.shop_base_names.get(card["kind"], {}).get(name, ())
            if len(exact) == 1 and exact[0][0] == key:
                if png := encode_portrait(frame.image, self.portrait_roi(position, card["kind"])):
                    portraits[key] = png
        return portraits

    def read_shop_slot(self, frame: Frame, position: dict, *, reward_kind: str | None = None,
                       reward_phase: bool = False) -> dict | None:
        kind = position["kind"]
        price_roi = position["priceROI"]
        if reward_phase:
            price = 0 if frame.hit("Screen.ShopRewardBadge", roi=price_roi) else None
            if price is None and position.get("soldROI") and frame.hit("Screen.ShopSold", roi=position["soldROI"]):
                return {"kind": kind, "id": None, "price": None, "sold": True, "frozen": None}
            if price is None:
                price = screen_integer(frame, "Screen.ShopCardPrice", roi=price_roi)
            kind = reward_kind or "chess"
        else:
            if position.get("soldROI") and frame.hit("Screen.ShopSold", roi=position["soldROI"]):
                return {"kind": kind, "id": None, "price": None, "sold": True, "frozen": None}
            price = screen_integer(frame, "Screen.ShopCardPrice", roi=price_roi)
            if price == 0 and (kind == "chess" or reward_kind is not None):
                if frame.hit("Screen.ShopRewardBadge", roi=price_roi):
                    kind = reward_kind or kind
                else:
                    price = None
        use_portrait = not reward_phase and price != 0 and kind in self.portrait_rois
        level = None
        if use_portrait:
            level = shop_card_tier(frame, position["tierROI"])
            portraits = self.shop_portrait_ids.get((kind, level), [])
            if portraits and (key := match_portrait(frame, self.portrait_roi(position, kind), portraits)):
                return {"kind": kind, "id": key, "price": price, "sold": False, "frozen": None}
        name = clean(frame.text("Screen.ShopCardName", roi=position["nameROI"], min_score=MIN_NAME_SCORE,
                                expected=self.shop_name_expected[kind], fallback_unfiltered=True))
        if use_portrait:
            LOGGER.warning("Shop portrait miss: screen slot=%s kind=%s tier=%s templates=%d OCR name=%r",
                           position["screenIndex"], kind, level, len(portraits), name)
        if price == 0 and (kind == "chess" or reward_kind is not None) and shop_confirm_box(frame, position):
            name = catalog_identity(self, frame, "Board.CarriedName", kind)["name"] or name
        names = self.shop_base_names.get(kind, {})
        candidates = names.get(match_name(name, names, approximate=False), [])
        if not candidates and use_portrait and kind == "chess" and level is not None and price:
            items = self.shop_by_tier_price.get(("item", level, price), {})
            if len(items.get(name, ())) == 1:
                return {"kind": "item", "id": items[name][0][0], "price": price, "sold": False, "frozen": None}
        fuzzy = not candidates and price is not None and bool(name)
        if not use_portrait and (fuzzy or len({data.get("tier") for _, data in candidates}) > 1):
            level = tier(frame.text("Screen.ShopCardTier", roi=position["tierROI"]))
        if fuzzy:
            if level is not None:
                constrained = self.shop_by_tier_price.get((kind, level, price), {})
                candidates = constrained.get(match_name(name, constrained), [])
        if len({data.get("tier") for _, data in candidates}) > 1:
            if level is None:
                return None
            candidates = [(key, data) for key, data in candidates if data.get("tier") == level]
        if not candidates and price != 0:
            return None
        key = candidates[0][0] if len(candidates) == 1 else None
        return {"kind": kind, "id": key, "price": price, "sold": False, "frozen": None}


class ScreenRecognition(CustomRecognition):
    def __init__(self, reader: Reader, known_portraits: frozenset[str] | None = None):
        super().__init__()
        self.reader = reader
        self.known_portraits = known_portraits

    def analyze(self, context: Context, argv: CustomRecognition.AnalyzeArg):
        hit = (0, 0, 1, 1)
        if argv.image.shape != (720, 1280, 3):
            return self.AnalyzeResult(hit, {"error": f"Expected a 1280×720 BGR screenshot, got {argv.image.shape}"})
        try:
            frame = Frame(context, argv.image)
            state = self.reader.read(frame)
            result = {"state": state}
            if self.known_portraits is not None:
                portraits = self.reader.shop_portraits(frame, state["self"]["shop"], self.known_portraits)
                if portraits:
                    result["portraits"] = portraits
            return self.AnalyzeResult(hit, result)
        except (OSError, RuntimeError, ValueError) as error:
            return self.AnalyzeResult(hit, {"error": str(error)})


def read_screen(tasker: Tasker, reader: Reader) -> dict:
    known = None
    if reader.portrait_rois:
        known = frozenset(path.stem for path in SHOP_PORTRAITS.glob("*.png"))
    recognition = ScreenRecognition(reader, known)
    with task_recognition(tasker, "ArkReadScreen", recognition):
        job = run_task(tasker, "ReadScreen")
    result = completed_data(job.nodes, "Screen.Read")
    if not job.succeeded or result is None or result.get("state") is None:
        raise RuntimeError((result or {}).get("error") or "MaaFramework ReadScreen did not return a ScreenState")
    if portraits := result.get("portraits"):
        save_portraits(SHOP_PORTRAITS, portraits)
    return result["state"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--png", type=Path, help="Read a saved screenshot without connecting to MuMu")
    source.add_argument("--live", action="store_true", help="Read one screenshot from MuMu in memory")
    add_session_args(parser, catalog=CATALOG)
    args = parser.parse_args()
    try:
        controller = None
        if not args.live:
            with Image.open(args.png) as image:
                if image.size != (1280, 720):
                    raise ValueError(f"Expected 1280×720 PNG, got {image.width}×{image.height}")
                frame = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
            controller = OfflineController(frame)
            if not controller.set_screenshot_use_raw_size(True) or not controller.post_connection().wait().succeeded:
                raise RuntimeError("Could not initialize offline screenshot controller")
        screen_session = session(args.adb, args.address, args.ocr_resource, controller=controller, catalog=args.catalog)
        tasker, reader = screen_session.tasker, screen_session.reader
        print(json.dumps(read_screen(tasker, reader), ensure_ascii=False))
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
