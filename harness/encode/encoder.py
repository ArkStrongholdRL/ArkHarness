"""Observation encoder of P7 (obsVersion 3) from a ScreenState.

Ported from the engine (GPL, read only): tools/rl/spec.mjs — buildVocab (with its hash), layout, Encoder.encode for
version 3 without ext / capBonds / v4, _card, and the helpers it calls (bucket, mainCoreBondOf, humanBonds, humanPairs,
bandConflicts, chessInBonds). The engine views are rebuilt from the ScreenState as bridge/screen.mjs toViews / stubs
do (a solo seat: no teammates, no humans, m.players empty, bandTaken false).

The bounty kill chance of draft rows (cards[*].10) and personal rows (personal[*].7) is left 0 here: the caller fills
it from the Node sidecar (bridge/bounty.mjs), as the engine's own value needs engine wave internals.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from ..catalog import GameData

SHOP, HAND, TEMP, BOARD = 10, 10, 5, 12
OWNED = HAND + TEMP + BOARD
REWARDS, PERSONAL, CARDS, ENEMIES, MATES = 6, 5, 8, 8, 3
CARD_BUCKETS, STAGE_BUCKETS, KIND_BUCKETS = 1024, 64, 16
MAX_PREP_ACTIONS = 40
BANDS = 40
ACT = {"END": 0, "REFRESH": 1, "LEVEL": 2, "FREEZE": 3, "BUY": 4, "SELL": 4 + SHOP, "REWARD": 4 + SHOP + OWNED,
       "PERSONAL": 4 + SHOP + OWNED + REWARDS, "CARD": 4 + SHOP + OWNED + REWARDS + PERSONAL,
       "BAND": 4 + SHOP + OWNED + REWARDS + PERSONAL + CARDS}
NUM_ACTIONS = ACT["BAND"] + BANDS
DIFFS = ["FUNNY", "NORMAL", "HARD", "ABYSS"]
G, FS, FP, FB, FE, FR, FK, FC, FM, FX, FBD = 38, 13, 13, 5, 8, 8, 9, 12, 11, 8, 9


def action_name(a: int) -> str:
    if a == 0: return "end"
    if a == 1: return "refresh"
    if a == 2: return "level"
    if a == 3: return "freeze"
    if a < ACT["SELL"]: return f"buy{a - ACT['BUY']}"
    if a < ACT["REWARD"]: return f"sell{a - ACT['SELL']}"
    if a < ACT["PERSONAL"]: return f"reward{a - ACT['REWARD']}"
    if a < ACT["CARD"]: return f"personal{a - ACT['PERSONAL']}"
    if a < ACT["BAND"]: return f"card{a - ACT['CARD']}"
    return f"band{a - ACT['BAND']}"


def fnv(s: str) -> int:
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def bucket(s, n: int) -> int:
    return 1 + (fnv(str(s)) % (n - 1)) if s else 0


def num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0


@dataclass
class Vocab:
    piece: dict
    enemy: dict
    band: dict
    band_list: list
    bond: dict
    boss: dict
    bond_ids: list
    hash: str
    data_id: str


def build_vocab(gd: GameData) -> Vocab:
    raw = gd.raw
    pieces = sorted({gd.base_id_of(i) for i in raw.get("chess", {})})
    items = sorted(raw.get("items", {}))
    tokens = sorted(raw.get("tokens", {}))
    piece: dict = {}
    for i, k in enumerate(pieces + items + tokens):
        piece[k] = i + 1  # a repeated key keeps its first position and takes the later code (JS Map semantics)
    enemy = {k: i + 1 for i, k in enumerate(sorted(raw.get("enemies", {})))}
    band_list = sorted(raw.get("bands", {}))[:BANDS]
    band = {k: i + 1 for i, k in enumerate(band_list)}
    bond_ids = sorted(raw.get("bonds", {}))
    boss = {k: i + 1 for i, k in enumerate(sorted(raw.get("bosses", {})))}
    bond = {k: i for i, k in enumerate(bond_ids)}
    payload = [gd.default_version, list(piece), list(enemy), list(band), bond_ids, list(boss), "obs3"]
    h = hashlib.sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return Vocab(piece, enemy, band, band_list, bond, boss, bond_ids, h, f"{gd.default_version}-{h}")


def layout(vocab: Vocab) -> dict:
    nb = len(vocab.bond_ids)
    blocks = [("global", 1, G), ("shop", SHOP, FS), ("owned", OWNED, FP), ("bonds", nb, FB), ("enemies", ENEMIES, FE),
              ("rewards", REWARDS, FR), ("personal", PERSONAL, FK), ("cards", CARDS, FC), ("mates", MATES, FM),
              ("extra", 1, FX), ("mateBonds", MATES, 1 + 2 * nb + 3), ("bands", BANDS, FBD + nb)]
    out, off = {}, 0
    for name, n, w in blocks:
        out[name] = (off, n, w)
        off += n * w
    return {"dim": off, "blocks": out}


def where(blocks: dict, i: int) -> str:
    for name, (off, n, w) in blocks.items():
        if off <= i < off + n * w:
            k = i - off
            return f"{name}[*].{k % w}" if n > 1 else f"{name}.{k}"
    return f"?{i}"


@dataclass
class Ctx:
    """Per-decision context (spec.mjs Encoder.encode ctx): env bookkeeping, never hidden game state."""
    kind: str = "prep"  # 'prep' | 'card' | 'band'
    actions: int = 0
    freezes: int = 0
    lp0: float = 0
    new_uids: set = field(default_factory=set)
    sold_bases: set = field(default_factory=set)
    strict_preview: bool = False
    visible_only: bool = False

    @classmethod
    def of(cls, d: dict) -> "Ctx":
        return cls(kind=d.get("kind", "prep"), actions=d.get("actions", 0) or 0, freezes=d.get("freezes", 0) or 0,
                   lp0=d.get("lp0", 0) or 0, new_uids=set(d.get("newUids") or []), sold_bases=set(d.get("soldBases") or []),
                   strict_preview=bool(d.get("strictPreview")), visible_only=bool(d.get("visibleOnly")))


class Encoder:
    def __init__(self, gd: GameData):
        self.gd = gd
        self.vocab = build_vocab(gd)
        lay = layout(self.vocab)
        self.dim, self.blocks = lay["dim"], lay["blocks"]
        self.A = NUM_ACTIONS
        self.max_actions = MAX_PREP_ACTIONS

    def piece_code(self, i) -> int:
        if not i:
            return 0
        v = self.vocab.piece.get(self.gd.base_id_of(i))
        if v is None:
            v = self.vocab.piece.get(i)
        return v or 0

    # ---- views (bridge/screen.mjs toViews) ----
    def views(self, s: dict):
        gd, S, M, C = self.gd, s["self"], s["match"], s.get("config") or {}

        def piece(p):
            if not p:
                return None
            tok = p.get("kind") == "token"
            return {**p, "tier": 0 if tok else gd.tier_of(p.get("id")), "golden": (not tok) and gd.is_golden(p.get("id"))}

        board = [piece(p) for p in S.get("board") or []]
        pub = {**C, "round": M.get("round"), "stageId": M.get("stageId"), "bossId": M.get("bossId"),
               "hiddenBossId": M.get("hiddenBossId"), "disabledBonds": M.get("disabledBonds") or [], "spRound": M.get("spRound"),
               "sp": M.get("sp"),
               "players": [{"playerId": S.get("playerId"), "isBot": True, "alive": True, "lp": S.get("lp"), "bandId": S.get("bandId"),
                            "shopLevel": (S.get("shop") or {}).get("level"), "boardCount": sum(1 for p in board if p),
                            "bonds": S.get("bonds") or []}]}
        own = {"playerId": S.get("playerId"), "lp": S.get("lp"), "funds": S.get("funds"), "bandId": S.get("bandId"),
               "canReady": S.get("canReady"), "deployCap": S.get("deployCap"), "deployCount": S.get("deployCount"),
               "personalChoice": S.get("personalChoice"), "shop": dict(S.get("shop") or {}),
               "hand": [piece(p) for p in S.get("hand") or []], "temp": [piece(p) for p in S.get("temp") or []], "board": board,
               "bonds": S.get("bonds") or [], "nextEnemies": S.get("nextEnemies") or []}
        return pub, own

    @staticmethod
    def owned(own) -> list:
        out = [None] * OWNED
        hand, temp, board = own.get("hand") or [], own.get("temp") or [], own.get("board") or []
        for i in range(HAND):
            out[i] = hand[i] if i < len(hand) and hand[i] else None
        for i in range(TEMP):
            out[HAND + i] = temp[i] if i < len(temp) and temp[i] else None
        for i in range(BOARD):
            out[HAND + TEMP + i] = board[i] if i < len(board) and board[i] else None
        return out

    # ---- public-view helpers (spec.mjs) ----
    def main_core_bond_of(self, row):
        counts = {b["bondId"]: b.get("count") or 0 for b in (row or {}).get("bonds") or []}
        top, top_n = None, 1
        for i in self.gd.bond_ids:
            n = counts.get(i, 0)
            if n > top_n and (self.gd.bond(i) or {}).get("isCore"):
                top_n, top = n, i
        return top

    @staticmethod
    def pool_humans(pub, self_id):
        return [p for p in pub.get("players") or [] if p.get("playerId") != self_id and p.get("alive") and p.get("isBot") is False]

    def human_bonds(self, pub, self_id) -> set:
        out = set()
        for p in self.pool_humans(pub, self_id):
            main = self.main_core_bond_of(p)
            if main:
                out.add(main)
            out.update(self.gd.band_bond_ids(p.get("bandId")))
        return out

    def chess_in_bonds(self, i, bonds) -> bool:
        if not bonds:
            return False
        c = self.gd.chess(i)
        return bool(c) and isinstance(c.get("bonds"), list) and any(b in bonds for b in c["bonds"])

    # ---- encode ----
    def encode(self, s: dict, ctx: Ctx, obs: np.ndarray | None = None, mask: np.ndarray | None = None):
        """obs (dim float32) and mask (A uint8) for ScreenState `s` at a decision of context `ctx`."""
        gd, V, L = self.gd, self.vocab, self.blocks
        obs = np.zeros(self.dim, np.float32) if obs is None else obs
        mask = np.zeros(self.A, np.uint8) if mask is None else mask
        obs.fill(0)
        mask.fill(0)
        pub, own = self.views(s)
        self_id = own.get("playerId")
        shop = own.get("shop") or {}
        funds = own.get("funds") or 0
        owned = self.owned(own)
        self_row = next((p for p in pub["players"] if p.get("playerId") == self_id), None)
        self_human = bool(self_row) and self_row.get("isBot") is False
        bound = not self_human
        protect = self.human_bonds(pub, self_id)
        pairs: set = set()  # humanPairs: no human seat in a solo match (m.players of the stub is empty)
        main_on = merge_on = bound  # rules default on
        slots = shop.get("slots") or []

        def blocked(i):
            return bool(i) and bool(gd.chess(i)) and ((main_on and self.chess_in_bonds(i, protect)) or (merge_on and gd.base_id_of(i) in pairs))

        hand_free = sum(1 for p in own.get("hand") or [] if not p)
        copies: dict = {}
        for p in owned:
            if p and p.get("kind") == "chess":
                b = gd.base_id_of(p["id"])
                copies[b] = copies.get(b, 0) + (3 if p.get("golden") else 1)
        g = L["global"][0]
        rnd = pub.get("round") or 0
        obs[g] = rnd / 15
        obs[g + 1 + (1 if ctx.kind == "card" else 0)] = 1
        if pub.get("difficulty") in DIFFS:
            obs[g + 4 + DIFFS.index(pub["difficulty"])] = 1
        mates = [p for p in pub["players"] if p.get("playerId") != self_id]
        obs[g + 8] = 1 if mates else 0
        obs[g + 9] = (own.get("lp") or 0) / ctx.lp0 if ctx.lp0 else 0
        obs[g + 10] = funds / 30
        obs[g + 11] = (shop.get("level") or 0) / 6
        obs[g + 12] = (shop.get("upgradePrice") or 0) / 20
        obs[g + 13] = (shop.get("refreshPrice") or 0) / 5
        obs[g + 14] = (shop.get("freeRefreshes") or 0) / 3
        obs[g + 15] = 1 if shop.get("frozen") else 0
        obs[g + 16] = (own.get("deployCap") or 0) / 10
        obs[g + 17] = (own.get("deployCount") or 0) / 10
        obs[g + 18] = (ctx.actions or 0) / self.max_actions
        obs[g + 19] = 1 if ctx.freezes else 0
        obs[g + 20] = 1 if pub.get("round") == pub.get("bossRound") else 0
        obs[g + 21] = 1 if pub.get("round") == pub.get("hiddenRound") else 0
        obs[g + 22] = 1 if pub.get("spRound") else 0
        obs[g + 23] = sum(1 for p in mates if p.get("alive")) / 3
        obs[g + 24] = hand_free / HAND
        obs[g + 25] = sum(1 for p in own.get("temp") or [] if p) / TEMP
        obs[g + 26] = V.band.get(own.get("bandId"), 0)
        obs[g + 27] = bucket(pub.get("stageId"), STAGE_BUCKETS)
        obs[g + 28] = (own.get("lp") or 0) / 50
        obs[g + 29] = 1 if own.get("personalChoice") else 0
        obs[g + 30] = 1 if own.get("canReady") else 0
        obs[g + 31] = (pub.get("lastRound") or 14) / 15
        obs[g + 32] = 1 if self_human else 0
        obs[g + 33] = 0  # living human mates / 3
        obs[g + 34] = 1 if main_on else 0
        obs[g + 35] = 1 if merge_on else 0
        obs[g + 37] = 0  # pool groups / disableSharedPool: not in a solo match
        # shop
        for i in range(min(SHOP, len(slots))):
            sl = slots[i]
            if not sl:
                continue
            o = L["shop"][0] + i * FS
            chess = sl.get("kind") == "chess"
            obs[o] = 1
            obs[o + 1] = 1 if chess else 0
            obs[o + 2] = 1 if sl.get("kind") == "item" else 0
            obs[o + 3] = self.piece_code(sl.get("id"))
            obs[o + 4] = 1 if gd.is_golden(sl.get("id")) else 0
            obs[o + 5] = gd.tier_of(sl.get("id")) / 6
            obs[o + 6] = (sl.get("price") or 0) / 10
            obs[o + 7] = 1 if sl.get("sold") else 0
            obs[o + 8] = 1 if sl.get("frozen") else 0
            obs[o + 9] = copies.get(gd.base_id_of(sl["id"]), 0) / 3 if chess else 0
            obs[o + 10] = 1 if chess and self.chess_in_bonds(sl["id"], protect) else 0
            obs[o + 11] = 1 if chess and gd.base_id_of(sl["id"]) in pairs else 0
            obs[o + 12] = 1 if chess and gd.base_id_of(sl["id"]) in ctx.sold_bases else 0
        # owned
        for i, p in enumerate(owned):
            if not p:
                continue
            o = L["owned"][0] + i * FP
            obs[o] = 1
            obs[o + 1] = 1 if p.get("kind") == "chess" else 0
            obs[o + 2] = 1 if p.get("kind") == "item" else 0
            obs[o + 3] = 1 if p.get("kind") == "token" else 0
            obs[o + 4] = self.piece_code(p.get("id"))
            obs[o + 5] = 1 if p.get("golden") else 0
            obs[o + 6] = (p.get("tier") or 0) / 6
            obs[o + 7] = len(p.get("items") or []) / 3
            obs[o + (8 if i < HAND else 9 if i < HAND + TEMP else 10)] = 1
            obs[o + 11] = copies.get(gd.base_id_of(p["id"]), 0) / 3 if p.get("kind") == "chess" else 0
            obs[o + 12] = 1 if p.get("uid") in ctx.new_uids else 0
        # bonds
        bo = L["bonds"][0]
        for b in own.get("bonds") or []:
            k = V.bond.get(b.get("bondId"))
            if k is None:
                continue
            o = bo + k * FB
            obs[o] = (b.get("count") or 0) / 6
            obs[o + 1] = 1 if b.get("active") else 0
            obs[o + 2] = (b.get("tier") or 0) / 3
            obs[o + 3] = (b.get("layers") or 0) / 50
        for i in set(pub.get("disabledBonds") or []):
            k = V.bond.get(i)
            if k is not None:
                obs[bo + k * FB + 4] = 1
        # enemies
        en = own.get("nextEnemies") or []
        for i in range(min(ENEMIES, len(en))):
            e = en[i]
            o = L["enemies"][0] + i * FE
            obs[o] = 1
            obs[o + 1] = V.enemy.get(e.get("enemyKey"), 0)
            obs[o + 2] = (e.get("count") or 0) / 10
            obs[o + 3] = 1 if e.get("fly") else 0
            obs[o + 4] = 1 if e.get("elite") else 0
            obs[o + 5] = 1 if e.get("boss") else 0
            obs[o + 6] = 1 if e.get("gate") == "upper" else 0
            obs[o + 7] = 0 if ctx.strict_preview else (e.get("t") or 0) / 60  # the spawn time: not on screen (0 live)
        # reward offer
        rw = ((shop.get("rewardOffer") or {}).get("slots")) or []
        for i in range(min(REWARDS, len(rw))):
            sl = rw[i]
            if not sl:
                continue
            o = L["rewards"][0] + i * FR
            chess = sl.get("kind") == "chess"
            obs[o] = 1
            obs[o + 1] = 1 if chess else 0
            obs[o + 2] = self.piece_code(sl.get("id"))
            obs[o + 3] = 1 if gd.is_golden(sl.get("id")) else 0
            obs[o + 4] = (sl.get("price") or 0) / 10
            obs[o + 5] = 1 if sl.get("sold") else 0
            obs[o + 6] = 1 if chess and self.chess_in_bonds(sl["id"], protect) else 0
            obs[o + 7] = 1 if chess and gd.base_id_of(sl["id"]) in pairs else 0
        # personal choice cards
        pc = ((own.get("personalChoice") or {}).get("cards")) or []
        for i in range(min(PERSONAL, len(pc))):
            self._card(obs, L["personal"][0] + i * FK, pc[i], None)
        # draft cards
        sp = pub.get("sp")
        if sp and ctx.kind == "card":
            cards = sp.get("cards") or []
            for i in range(min(CARDS, len(cards))):
                self._card(obs, L["cards"][0] + i * FC, cards[i], sp)
        # mates: none in a solo match
        # extra (S5)
        o = L["extra"][0]
        obs[o] = 1
        obs[o + 1] = V.boss.get(pub.get("bossId"), 0)
        revealed = bool(pub.get("hiddenRound")) and rnd >= pub["hiddenRound"]
        obs[o + 2] = 0 if ctx.visible_only and not revealed else V.boss.get(pub.get("hiddenBossId"), 0)
        obs[o + 3] = max(0, pub["bossRound"] - rnd) / 15 if pub.get("bossRound") else 0
        obs[o + 4] = 0
        obs[o + 5] = len(mates) / 3
        obs[o + 6] = 1 if ctx.kind == "band" else 0
        obs[o + 7] = 0  # pub.draft?.concurrent: the ScreenState carries no draft (solo)
        if ctx.kind == "band":
            for j, i in enumerate(V.band_list):
                o = L["bands"][0] + j * L["bands"][2]
                rec, tied = gd.band(i) or {}, gd.band_bond_ids(i)
                obs[o] = 1 if gd.band_allowed(i) else 0
                obs[o + 1] = V.band.get(i, 0)
                obs[o + 2] = num(float(rec["totalHp"]) if _numlike(rec.get("totalHp")) else 0) / 50
                obs[o + 3] = 1 if any(b in gd.mode_inactive_bonds or b in (pub.get("disabledBonds") or []) for b in tied) else 0
                for b in tied:
                    k = V.bond.get(b)
                    if k is not None:
                        obs[o + FBD + k] = 1
        # ---- mask ----
        if ctx.kind == "band":
            for j, i in enumerate(V.band_list):
                if gd.band_allowed(i) and i in gd.band_ids():
                    mask[ACT["BAND"] + j] = 1
            return obs, mask
        if ctx.kind == "card":
            if sp:
                cards = sp.get("cards") or []
                taken = sp.get("taken") or {}
                for i in range(min(CARDS, len(cards))):
                    c = cards[i]
                    if c and not c.get("soldOut") and taken.get(str(c.get("idx"))) is None:
                        mask[ACT["CARD"] + i] = 1
            return obs, mask
        mask[ACT["END"]] = 1
        if (ctx.actions or 0) >= self.max_actions:
            return obs, mask
        refresh = shop.get("refreshPrice")
        if (shop.get("freeRefreshes") or 0) > 0 or (refresh is not None and funds >= refresh):
            mask[ACT["REFRESH"]] = 1
        up = shop.get("upgradePrice")
        if (shop.get("level") or 0) < (shop.get("maxLevel") or 0) and up is not None and funds >= up:
            mask[ACT["LEVEL"]] = 1
        if not ctx.freezes and any(sl and not sl.get("sold") for sl in slots):
            mask[ACT["FREEZE"]] = 1
        if hand_free > 0:
            for i in range(min(SHOP, len(slots))):
                sl = slots[i]
                if sl and not sl.get("sold") and (sl.get("price") or 0) <= funds and not blocked(sl.get("id")):
                    mask[ACT["BUY"] + i] = 1
        for i, p in enumerate(owned):
            if p and p.get("kind") == "chess":
                mask[ACT["SELL"] + i] = 1
        if hand_free > 0:
            for i in range(min(REWARDS, len(rw))):
                sl = rw[i]
                if sl and not sl.get("sold") and (sl.get("price") or 0) <= funds and not blocked(sl.get("id")):
                    mask[ACT["REWARD"] + i] = 1
        for i in range(min(PERSONAL, len(pc))):
            mask[ACT["PERSONAL"] + i] = 1
        return obs, mask

    def _card(self, obs, o, c, sp):
        if not c:
            return
        obs[o] = 1
        obs[o + 1] = bucket(c.get("kind"), KIND_BUCKETS)
        obs[o + 2] = bucket(c.get("id"), CARD_BUCKETS)
        if sp is not None:
            taken = sp.get("taken") or {}
            obs[o + 1] = 1 if not c.get("soldOut") and taken.get(str(c.get("idx"))) is None else 0
            obs[o + 2] = bucket(c.get("kind"), KIND_BUCKETS)
            obs[o + 3] = bucket(c.get("id"), CARD_BUCKETS)
            obs[o + 4] = num(c.get("tier")) / 6
            obs[o + 5] = num(c.get("coin")) / 10
            obs[o + 6] = 1 if c.get("payout") == "perfect" else 0
            obs[o + 7] = num(c.get("rounds")) / 5
            obs[o + 8] = self.vocab.enemy.get(c.get("enemyKey"), 0)
            obs[o + 9] = num(c.get("count")) / 10
            obs[o + 10] = 0  # bounty kill chance: filled by the caller
            obs[o + 11] = 1 if c.get("team") else 0
            return
        obs[o + 3] = num(c.get("tier")) / 6
        obs[o + 4] = num(c.get("coin")) / 10
        obs[o + 5] = 1 if c.get("payout") == "perfect" else 0
        obs[o + 6] = num(c.get("rounds")) / 5
        obs[o + 7] = 0  # bounty kill chance: filled by the caller
        obs[o + 8] = self.vocab.enemy.get(c.get("enemyKey"), 0)

    def fill_kill_chance(self, obs, kind: str, s: dict, values: dict, strict_preview=False):
        """Write bounty kill chances: values = {('card', i): p, ('personal', i): p} (bridge/bounty.mjs fillBounty)."""
        L = self.blocks
        for (blk, i), p in values.items():
            if blk == "card" and kind == "card" and not strict_preview and i < CARDS:
                obs[L["cards"][0] + i * FC + 10] = p
            elif blk == "personal" and i < PERSONAL:
                obs[L["personal"][0] + i * FK + 7] = p


def _numlike(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
