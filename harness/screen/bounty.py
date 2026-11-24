"""Resolve an on-screen bounty by its displayed name and enemy description."""

import re
from unicodedata import normalize

from .native import Frame


SLOTS = ("Left", "Center", "Right")
TITLE = r"^(悬赏·[一-鿿]+)[I|]*$"
DESCRIPTION = (
    r"为自身(下场|接下来两场)(?:作战|战斗)添加([0-9]+)[只个](.+?)[,，](?=将其|若|但不)"
)


def text_key(text: str) -> str:
    return re.sub(r"\s+", "", normalize("NFKC", text))


def family(text: str) -> str | None:
    name = text_key(text).replace("・", "·").replace("•", "·").replace("賞", "赏")
    match = re.fullmatch(TITLE, name)
    if match:
        return match[1]
    return name if name.endswith(("·悬赏", "·特训")) else None


def enemy_description(text: str) -> tuple[str, int, int] | None:
    match = re.search(DESCRIPTION, text_key(text))
    if match is None:
        return None
    duration = 1 if match[1] == "下场" else 2
    # Wrapped names can lose the dot before a suffix such as α or β in OCR.
    name = re.sub(r"[·・•∙]", "", match[3])
    return name, duration, int(match[2])


class BountyCatalog:
    def __init__(self, cards: list[dict]):
        self.cards = {}
        for card in cards:
            name = family(card["name"])
            enemy = enemy_description(card.get("desc", ""))
            if name and enemy and enemy[1:] == (card.get("rounds"), card.get("count")):
                key = (name, *enemy)
                self.cards.setdefault(key, []).append(card)

    def offer(self, frame: Frame, slot: int) -> tuple[dict, tuple] | None:
        suffix = SLOTS[slot]
        names = sorted(frame.rows(f"Bounty.Name{suffix}"), key=lambda row: (row.box[1], row.box[0]))
        if not names or not (name := family("".join(row.text for row in names))):
            return None
        rows = frame.rows(f"Bounty.Description{suffix}")
        description = "".join(text_key(row.text) for row in sorted(rows, key=lambda row: (row.box[1], row.box[0])))
        enemy = enemy_description(description)
        if enemy is None:
            return None
        candidates = self.cards.get((name, *enemy), [])
        if len(candidates) > 1 and (reward := re.search(r"获得([0-9]+)资金", description)):
            candidates = [card for card in candidates if card.get("coin") == int(reward[1])]
        return (candidates[0], tuple(names[0].box)) if len(candidates) == 1 else None
