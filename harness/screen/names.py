"""Resolve short OCR strings against a constrained catalogue."""

import re
import unicodedata
from collections.abc import Collection
from functools import lru_cache


@lru_cache(maxsize=4096)
def normalized_name(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", text).rstrip("。.·")


def one_edit_apart(observed: str, candidate: str) -> bool:
    if len(candidate) <= 2:
        # Short names cannot safely take a substitution. Allow one leading icon glyph.
        return len(candidate) == 2 and len(observed) == 3 and observed[1:] == candidate
    if len(observed) == len(candidate):
        return sum(left != right for left, right in zip(observed, candidate)) == 1
    if abs(len(observed) - len(candidate)) != 1:
        return False
    short, long = sorted((observed, candidate), key=len)
    index = next((i for i, (left, right) in enumerate(zip(short, long)) if left != right), len(short))
    return short[index:] == long[index + 1:]


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        row = [i]
        for j, b in enumerate(right, start=1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = row
    return previous[-1]


def match_name(text: str, vocab: Collection[str], *, approximate: bool = True) -> str | None:
    """Require a unique nearest name, with at least one edit of margin over the runner-up."""
    if text in vocab:
        return text
    observed = normalized_name(text)
    if not observed:
        return None
    labels = [(name, normalized_name(name)) for name in vocab]
    exact = [name for name, candidate in labels if observed == candidate]
    if exact:
        return exact[0] if len(exact) == 1 else None
    if not approximate or not observed[-1].isalnum():
        return None
    close = [name for name, candidate in labels if one_edit_apart(observed, candidate)]
    if close or len(observed) < 6:
        return close[0] if len(close) == 1 else None
    # Two edits require an intact family prefix. Enemy labels can share a two-character suffix.
    close = [name for name, candidate in labels if len(candidate) >= 6 and candidate[:4] == observed[:4]
             and abs(len(observed) - len(candidate)) <= 2 and edit_distance(observed, candidate) == 2]
    return close[0] if len(close) == 1 else None
