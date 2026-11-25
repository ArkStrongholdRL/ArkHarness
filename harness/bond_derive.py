"""Bond observations the screen does not show, derived from the opening choice and the holdings.

Mirrors the engine's bond arithmetic (server/match/bondsMeta.js computeBonds / bondList, server/match/match/phases.js
pickBondGroup) for a solo seat. Only the solo case is derived here; a multiplayer match has no picked group to read the
disabled set from.
"""

from .catalog import GameData

HARMONY_BOND = "maniShip"
DEPUTY_BOND = "deputShip"


def disabled_bonds_from_choice(bond_choice: dict | None) -> list[str] | None:
    """Solo only: the core bonds of the group that was not picked are disabled (phases.js pickBondGroup).

    Returns None when the choice is not made yet. Note that the engine also disables the add-on bonds drawn at match
    start (pool.js drawDisabledBonds, regulation solo); those are not on the briefing screen and are not derived here.
    """
    if not bond_choice:
        return None
    picked = bond_choice.get("picked")
    if picked not in (0, 1):
        return None
    return list(bond_choice["groups"][1 - picked])


def disabled_addon_bonds(enabled, gd: GameData) -> list[str] | None:
    """Solo only: add-on bonds drawn off at match start (pool.js drawDisabledBonds, eligible = weight > 0 and not
    mode-inactive). The briefing's add-on row lists the enabled ones; an eligible add-on missing from it is disabled.

    `enabled` is None when the row was not read. Returns None unless the row was read completely: every weight 0 add-on
    (never drawn, always listed) must be present, and nothing outside the add-on set may be.
    """
    if enabled is None:
        return None
    addon = [b for b in gd.bond_ids if not (gd.bond(b) or {}).get("isCore")]
    drawable = {b for b in addon if _weight(gd.bond(b)) > 0 and b not in gd.mode_inactive_bonds}
    listed = set(enabled)
    always = {b for b in addon if _weight(gd.bond(b)) <= 0}  # weight 0 add-ons are never drawn, always listed
    if not always <= listed or not listed <= set(addon):
        return None
    return sorted(drawable - listed)


def merge_disabled_bonds(core, addon) -> list[str]:
    """Solo only: the screen or choice core-disabled set, then the add-on set from disabled_addon_bonds (no duplicates)."""
    return list(dict.fromkeys([*(core or []), *(addon or [])]))


def can_derive_bonds(board, hand, gd: GameData) -> bool:
    """False when a chess piece on the board or in the hand cannot be derived: its catalog `bonds` is empty or unknown.

    The only such operators are the DIY slots (chess_char_5/6_diy*): their bonds come from a hidden choice in the player
    lobby, which the screen does not show. derive_bonds would undercount them, so the caller keeps the screen's reading.
    Temp (备战区) holds items only; pass it in `hand` to check it too.
    """
    for p in list(board) + list(hand):
        if not p or p.get("kind") != "chess":
            continue
        if not (gd.chess(p["id"]) or {}).get("bonds"):
            return False
    return True


def fill_unread_bonds(bonds, derived) -> list | None:
    """The screen's bonds[] with unread count / active / tier filled from `derived` (derive_bonds output), by bondId.

    A value the screen read is never overwritten. A bond the screen lists but derive_bonds omits has no members and no
    activation (bondList omits those), so its unread fields become 0 / False / 0. layers is never touched (it accumulates
    through effects). A None bondId (not resolved to the catalog) is left as read.

    The ring on screen lists only some bonds, and in most preparation frames none at all (the block is None). The engine's
    bonds[] holds every bond with members, so any derived bond the screen did not list is added, with layers None (unknown).
    """
    by_id = {d["bondId"]: d for d in derived}
    filled = []
    for entry in bonds or []:
        entry = dict(entry)
        if entry.get("bondId") is not None:
            d = by_id.get(entry["bondId"], {"count": 0, "active": False, "tier": 0})
            for key in ("count", "active", "tier"):
                if entry.get(key) is None:
                    entry[key] = d[key]
        filled.append(entry)
    listed = {entry.get("bondId") for entry in filled}
    filled += [{"bondId": d["bondId"], "count": d["count"], "active": d["active"], "tier": d["tier"], "layers": None}
               for d in derived if d["bondId"] not in listed]
    return filled


def _weight(bond: dict) -> float:
    w = bond.get("weight")
    return float(w) if isinstance(w, (int, float)) and not isinstance(w, bool) else 0.0


def _thresholds(bond: dict) -> list:
    """bondsMeta.js thresholdsOf: ascending, finite; falls back to activeCount, then 2."""
    th = bond.get("thresholds")
    if isinstance(th, list) and th:
        return sorted(n for n in th if isinstance(n, (int, float)) and not isinstance(n, bool))
    active = bond.get("activeCount")
    if isinstance(active, (int, float)) and not isinstance(active, bool):
        return [active]
    return [2]


def _tier(bond: dict, count: int) -> int:
    """bondsMeta.js tierFor (downward bonds: 1 while 1 <= count <= maxCount)."""
    th = _thresholds(bond)
    if bond.get("thresholdTemplate") == "count_threshold_downward":
        mx = bond.get("maxCount")
        top = mx if isinstance(mx, (int, float)) and not isinstance(mx, bool) else th[0]
        return 1 if th[0] <= count <= top else 0
    return sum(1 for n in th if count >= n)


def _granted_by_morph(gd: GameData, items: list) -> list[str]:
    """bondsMeta.js pieceBonds with the official 变形同构体 pairing (fewer than two items grants nothing)."""
    if len(items) < 2:
        return []
    recs = [gd.item(i) for i in items]
    recs = [r for r in recs if isinstance(r, dict)]
    if not any(r.get("canGiveBond") for r in recs):
        return []
    out: list[str] = []
    for r in recs:
        if r.get("canGiveBond"):
            continue
        bond = r.get("giveBondId")
        if isinstance(bond, str) and bond and gd.bond(bond) and bond not in out:
            out.append(bond)
    return out


def derive_bonds(board, gd: GameData, hand=()) -> list[dict]:
    """The engine's bonds[] (count / active / tier) for a solo seat, from the chess on the board and in the hand.

    `board` and `hand` are ScreenState-style pieces ({kind, id, items}); tokens and empty slots are skipped, as the
    engine does. `layers` is not derived: it accumulates through effects. Bonds with no members and no activation are
    omitted, as the engine's bondList omits them. Mode-inactive bonds are reported with active False, as bondList does.
    """
    on_board: dict[str, set] = {}
    on_variant: dict[str, set] = {}
    in_hand: dict[str, set] = {}
    golden_on_board = 0
    for p in board:
        if not p or p.get("kind") != "chess":
            continue
        base, golden = gd.base_id_of(p["id"]), bool(gd.is_golden(p["id"]))
        golden_on_board += golden
        for bond in list((gd.chess(p["id"]) or {}).get("bonds") or []) + _granted_by_morph(gd, p.get("items") or []):
            on_board.setdefault(bond, set()).add(base)
            on_variant.setdefault(bond, set()).add(f"{base}|{int(golden)}")
    for p in hand:
        if not p or p.get("kind") != "chess":
            continue
        base = gd.base_id_of(p["id"])
        for bond in list((gd.chess(p["id"]) or {}).get("bonds") or []) + _granted_by_morph(gd, p.get("items") or []):
            in_hand.setdefault(bond, set()).add(base)

    raw: dict[str, int] = {}
    off: dict[str, int] = {}  # mode-inactive bonds: counted for the views, never activated (bondsMeta offBondCounts)
    for bond_id in gd.bond_ids:
        bond = gd.bond(bond_id) or {}
        if bond.get("countMode") == "BOARD_ALL_CHESS" or bond.get("thresholdTemplate") == "count_threshold_upward_golden":
            count = golden_on_board
        elif bond.get("countMode") == "BOARD_AND_DECK":
            count = len(on_board.get(bond_id, set()) | in_hand.get(bond_id, set()))
        else:
            count = len(on_board.get(bond_id, set()))
        if bond_id in gd.mode_inactive_bonds:
            off[bond_id] = count
        else:
            raw[bond_id] = count

    # maniShip (harmony): while active, every core bond with a board member gets one more
    harmony = gd.bond(HARMONY_BOND)
    harmony_active = harmony is not None and HARMONY_BOND in raw and _tier(harmony, raw[HARMONY_BOND]) >= 1

    out = []
    for bond_id, count in raw.items():
        bond = gd.bond(bond_id) or {}
        if harmony_active and bond.get("isCore") and on_board.get(bond_id):
            count += 1
        tier = _tier(bond, count)
        if bond_id == DEPUTY_BOND and tier >= 1:
            # upper tiers count operators that differ in name or elite state
            variants = len(on_variant.get(bond_id, set()))
            th = _thresholds(bond)
            tier = 1
            for i in range(1, len(th)):
                if variants >= th[i]:
                    tier = i + 1
        if count > 0 or tier >= 1:
            out.append({"bondId": bond_id, "count": count, "active": tier >= 1, "tier": tier})
    out += [{"bondId": bond_id, "count": count, "active": False, "tier": 0} for bond_id, count in off.items() if count > 0]
    return out

