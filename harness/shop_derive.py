"""Shop prices the screen can leave unread, derived from the mode's price table and the match's own level history.

Mirrors the engine: upgradePrice starts at upgradeBase(1), drops by one at each round start after the first
(player/round.js startRound) and resets to upgradeBase(level) when the shop is upgraded (player/economy.js levelUp);
refreshPrice is the economy's refreshPrice, or 0 while a free refresh is held (player/views.js privateView). Band and
command discounts (bandCommandEffects.js onGain, the upgrade discount) are not derived.
"""

from .catalog import GameData

DEFAULT_MAX_SHOP_LEVEL = 6  # gamedata.js DEFAULTS.maxShopLevel
DEFAULT_REFRESH_PRICE = 1  # gamedata.js DEFAULTS.refreshPrice


def upgrade_price(gd: GameData, level: int | None, level_round: int | None, round_no: int | None) -> int | None:
    """Upgrade price at `round_no` of a shop at `level`, reached in `level_round`.

    None when any input is unknown, or the mode has no price table. 0 at the maximum level (the engine view's value).
    """
    if level is None or level_round is None or round_no is None:
        return None
    if type(level) is not int or type(level_round) is not int or type(round_no) is not int:
        return None
    if level >= gd.mode.get("maxShopLevel", DEFAULT_MAX_SHOP_LEVEL):
        return 0
    table = gd.mode.get("upgradePrices")
    if not isinstance(table, list) or not 1 <= level <= len(table):
        return None
    base = table[level - 1]
    if not isinstance(base, int) or isinstance(base, bool):
        return None
    return max(0, base - max(0, round_no - level_round))


def refresh_price(gd: GameData, free_refreshes: int = 0) -> int:
    """The refresh price: 0 while a free refresh is held, else the economy's price."""
    if free_refreshes > 0:
        return 0
    economy = (gd.raw.get("config") or {}).get("economy") or {}
    price = economy.get("refreshPrice", DEFAULT_REFRESH_PRICE)
    return price if isinstance(price, int) and not isinstance(price, bool) and price >= 0 else DEFAULT_REFRESH_PRICE
