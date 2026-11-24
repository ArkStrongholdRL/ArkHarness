"""Self-maintained holdings with order (B5): hand (10), temp row (5, equipment), hidden temp queue, board.

Ported from the engine (GPL, read only), regulation rules (config.version.rules.equipHand = true):
  server/match/player/pieces.js      stow / _stowSplit, _fillSplitHand, find, allChess, _chessLocations, _detach, removeTokensOf,
                                     grantTokensFor (a board owner's summons go to the hand)
  server/match/player/acquire.js     acquireChess, _mergeChess (consumed copies: incoming, temp, hand, board; the elite takes a
                                     consumed board copy's tile (mergeTile) else is stowed), acquireItem, _mergeItem, _fuseItem,
                                     checkItemMerges
  server/match/player/economy.js     buy, sell (equipment returned by stow, then checkItemMerges)
  server/match/player/prep.js        pickReward
  server/match/board.js              freeSlot (rightmost free), mergeTile, boardOrder (reading order)
  server/match/effectsMeta.js        garrison gifts on gain (SERVER_GAIN -> SERVER_GAIN_EQUIP, repeated by 投资人)
  server/match/player/views.js       privateView: hand, temp (5 visible), board in reading order

API: Track(gd); calibrate(screen) resets from a read ScreenState (prep start; after a card or reward pick the harness re-reads);
apply(action, read) runs one executed prep action (encoder ACT numbering; read = the ScreenState the policy saw before it, for
the shop, reward offer and bonds); to_screen() gives the holdings in ScreenState form; snapshot() is the comparable view.

Pieces carry our own uids (calibrated pieces get new ones: the screen shows none live); parity compares by content and
position, never by uid.
The board is a dict keyed 'row,col' (insertion order = JS Map order); reads use reading order, as privateView does.
Not modelled: the hidden overflow queue at prep start (not on the ScreenState; only the visible row is read), deferred
merges, pool stock, bond effects, random gifts (reported in `unpredicted`).
"""

from dataclasses import dataclass, field

from .catalog import GameData
from .encode.encoder import ACT, BOARD, HAND, SHOP, TEMP

TOKEN_MAX = 9


@dataclass(eq=False)
class Piece:
    uid: int
    kind: str  # 'chess' | 'item' | 'token'
    id: str
    items: list = field(default_factory=list)  # equipped items (chess)
    owner: object = None  # token: owning chess uid
    count: int = 1  # token stack size (not on the ScreenState)
    dir: str | None = "RIGHT"
    bought: int = 0  # the round we got it (legacy's keepFresh guard reads boughtRound); 0 = unknown / earlier


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _pos(v, d):
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else d


def rightmost_free(arr) -> int:
    """engine freeSlot: the highest index that is empty, or -1."""
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] is None:
            return i
    return -1


def _consumed_on_gain(rec) -> bool:
    """Approximation for legacy snapshots taken after the bot could equip a purchase."""
    return rec.get("kind") == "consume_on_equip" and rec.get("mergeable") is True


class Track:
    def __init__(self, gd: GameData, *, legacy_equip_snapshot: bool = True):
        self.gd = gd
        self.legacy_equip_snapshot = legacy_equip_snapshot
        self.hand = [None] * HAND
        self.temp = [None] * TEMP  # temp[:5] is the visible equipment row; temp[5:] the hidden queue
        self.board: dict = {}  # 'r,c' -> Piece; dict order = JS Map order
        self._uid = 0
        self.unpredicted: list = []  # gifts with a random roll the tracker cannot see
        self.guessed: list = []  # pool gifts taken as their modal item
        self._bounty_picks: dict[tuple[int, str], dict] = {}
        eco = (gd.raw.get("config") or {}).get("economy") or {}
        self.item_merge_count = _pos(eco.get("itemMergeCount"), 2)

    def record_bounty(self, receipt: dict, *, selection_id: str = "bounty-draft") -> None:
        """Record a confirmed pick. Personal choices should supply their own selection ID."""
        round_number = receipt.get("round")
        after = receipt.get("after") or {}
        card = receipt.get("card") or {}
        enemy, count, rounds = (card.get(key) for key in ("enemyKey", "count", "rounds"))
        card_id = card.get("id")
        if (type(round_number) is not int or round_number < 1 or after.get("screen") != "preparation"
                or after.get("round") != round_number or card.get("kind") != "bounty"
                or not isinstance(card_id, str) or not card_id or not self.gd.enemy(enemy)
                or type(count) is not int or not 1 <= count <= 20
                or type(rounds) is not int or not 1 <= rounds <= 99
                or not isinstance(selection_id, str) or not selection_id):
            raise ValueError("A confirmed bounty pick with an enemy, count and duration is required")
        key = (round_number, selection_id)
        picked = {"id": card_id, "enemyKey": enemy, "count": count, "rounds": rounds}
        if key in self._bounty_picks and self._bounty_picks[key] != picked:
            raise ValueError("A different bounty has already been recorded for this selection")
        self._bounty_picks[key] = picked

    def active_bounties(self, round_number: int) -> list[dict]:
        """Return the active cards in previewFromKinds input shape and pick order."""
        if type(round_number) is not int or round_number < 1:
            raise ValueError("A positive battle round is required")
        return [{"card": {"enemyKey": card["enemyKey"], "count": card["count"]}}
                for (start, _), card in self._bounty_picks.items()
                if start <= round_number < start + card["rounds"]]

    # ---------- basics ----------
    def _new(self, kind, pid, **kw) -> Piece:
        self._uid += 1
        return Piece(self._uid, kind, pid, **kw)

    def _board_sorted(self):
        """reading order (privateView / boardOrder): row desc, col asc."""
        out = []
        for k, p in self.board.items():
            r, c = (int(x) for x in k.split(","))
            out.append((r, c, k, p))
        out.sort(key=lambda t: (-t[0], t[1]))
        return out

    def _all_chess(self):
        """engine allChess: board (reading order), hand, temp."""
        out = [p for _, _, _, p in self._board_sorted() if p.kind == "chess"]
        out += [p for p in self.hand if p is not None and p.kind == "chess"]
        out += [p for p in self.temp if p is not None and p.kind == "chess"]
        return out

    def _chess_locs(self):
        """engine _chessLocations: temp (all indices), hand, board (reading order)."""
        out = []
        for i, p in enumerate(self.temp):
            if p is not None and p.kind == "chess":
                out.append({"area": "temp", "idx": i, "piece": p})
        for i, p in enumerate(self.hand):
            if p is not None and p.kind == "chess":
                out.append({"area": "hand", "idx": i, "piece": p})
        for _, _, k, p in self._board_sorted():
            if p.kind == "chess":
                out.append({"area": "board", "key": k, "piece": p})
        return out

    def _item_locs(self, pid):
        """engine _itemLocations: temp (all), hand, equipped (holders in allChess order)."""
        out = []
        for i, p in enumerate(self.temp):
            if p is not None and p.kind == "item" and p.id == pid:
                out.append({"area": "temp", "idx": i, "piece": p})
        for i, p in enumerate(self.hand):
            if p is not None and p.kind == "item" and p.id == pid:
                out.append({"area": "hand", "idx": i, "piece": p})
        for h in self._all_chess():
            for it in h.items:
                if it.id == pid:
                    out.append({"area": "equipped", "holder": h, "piece": it})
        return out

    def _detach(self, loc):
        a = loc["area"]
        if a == "hand":
            self.hand[loc["idx"]] = None
        elif a == "temp":
            self.temp[loc["idx"]] = None
        elif a == "board":
            self.board.pop(loc["key"], None)
        elif a == "equipped":
            hs = loc["holder"].items
            for i, x in enumerate(hs):
                if x is loc["piece"]:
                    del hs[i]
                    break

    def _handslot(self, kind) -> int:
        """equipHand: items fill the visible temp row, chess and summons the hand (both rightmost-free)."""
        if kind == "item":
            return rightmost_free(self.temp[:TEMP])
        return rightmost_free(self.hand)

    def _stow(self, p):
        """engine _stowSplit with allowTemp: the row's free slot, else the hidden queue (appended)."""
        i = self._handslot(p.kind)
        if i >= 0:
            if p.kind == "item":
                self.temp[i] = p
            else:
                self.hand[i] = p
            return
        self.temp.append(p)  # max(TEMP, len(temp)) == len(temp) here

    def _recompute(self):
        """engine recompute -> _fillSplitHand: hidden pieces move into free slots of their row, in arrival order."""
        for j in range(TEMP, len(self.temp)):
            p = self.temp[j]
            if p is None:
                continue
            i = self._handslot(p.kind)
            if i < 0:
                continue
            self.temp[j] = None
            if p.kind == "item":
                self.temp[i] = p
            else:
                self.hand[i] = p
        self.temp = self.temp[:TEMP] + [x for x in self.temp[TEMP:] if x is not None]

    # ---------- summons ----------
    def _placeable_tokens(self, cid):
        """engine gamedata.placeableTokens (loadout = default: the variant's own sources)."""
        c = self.gd.chess(cid)
        if not c or not isinstance(c.get("tokens"), list):
            return []
        out = []
        for tid in c["tokens"]:
            t = self.gd.token(tid)
            if not t or t.get("kind") != "summon" or t.get("placeable") is not True:
                continue
            vs = t.get("variants") if isinstance(t.get("variants"), dict) else {}
            alt = cid[:-2] + "_a" if cid.endswith("_b") else cid
            v = vs.get(cid) or vs.get(alt) or None
            if v:
                src = v.get("sources") if isinstance(v.get("sources"), list) else []
                if "talent" not in src and "skill" not in src:
                    continue
            dl = ((v or {}).get("stats") or {}).get("deployLimit")
            count = _pos(dl, _pos(t.get("deployLimit"), 1))
            out.append((tid, min(count, TOKEN_MAX)))
        return out

    def _token_count(self, owner_uid, tid) -> int:
        n = 0
        for p in list(self.board.values()) + self.hand + self.temp:
            if p is not None and p.kind == "token" and p.owner == owner_uid and p.id == tid:
                n += p.count
        return n

    def _remove_tokens_of(self, uid):
        for k in [k for k, p in self.board.items() if p.kind == "token" and p.owner == uid]:
            del self.board[k]
        for row in (self.hand, self.temp):
            for i, p in enumerate(row):
                if p is not None and p.kind == "token" and p.owner == uid:
                    row[i] = None

    def _grant_tokens_for(self, owner):
        for tid, count in self._placeable_tokens(owner.id):
            missing = count - self._token_count(owner.uid, tid)
            if missing <= 0:
                continue
            stack = next((p for p in self.hand + self.temp if p is not None and p.kind == "token"
                          and p.owner == owner.uid and p.id == tid), None)
            if stack is not None:
                stack.count += missing
                continue
            self._stow(self._new("token", tid, owner=owner.uid, count=missing))

    def replenish_summons(self):
        """Apply the engine's round-start top-up for currently deployed owners."""
        for piece in self.board.values():
            if piece.kind == "chess":
                self._grant_tokens_for(piece)
        self._recompute()

    # ---------- chess gains and merges ----------
    def _completes_chess_merge(self, cid) -> bool:
        rec = self.gd.chess(cid)
        if not rec or rec.get("isGolden"):
            return False
        need = self.gd.merge_count(cid)
        if not (need > 1) or not self.gd.golden_id_of(cid):
            return False
        base = self.gd.base_id_of(cid)
        copies = sum(1 for loc in self._chess_locs()
                     if not self.gd.is_golden(loc["piece"].id) and self.gd.base_id_of(loc["piece"].id) == base)
        return copies + 1 >= need

    def merge_board_sources(self, cid: str) -> dict[str, str]:
        if not self._completes_chess_merge(cid):
            return {}
        base = self.gd.base_id_of(cid)
        copies = [loc for loc in self._chess_locs() if not self.gd.is_golden(loc["piece"].id)
                  and self.gd.base_id_of(loc["piece"].id) == base]
        return {loc["key"]: loc["piece"].dir for loc in copies[:self.gd.merge_count(cid) - 1]
                if loc["area"] == "board"}

    def _merge_chess(self, base, incoming):
        need = self.gd.merge_count(base)
        golden = self.gd.golden_id_of(base)
        if not (need > 1) or not golden:
            return None
        locs = [l for l in self._chess_locs()
                if not self.gd.is_golden(l["piece"].id) and self.gd.base_id_of(l["piece"].id) == base]
        consumed = [{"area": "new", "piece": incoming}]
        for l in locs:
            if len(consumed) >= need:
                break
            consumed.append(l)
        if len(consumed) < need:
            return None
        items = []
        for l in consumed:
            if l["area"] != "new":
                self._detach(l)
            self._remove_tokens_of(l["piece"].uid)
            items += l["piece"].items
            l["piece"].items = []
        elite = self._new("chess", golden)
        deployed = [l for l in consumed if l["area"] == "board"]
        if deployed:
            # mergeTile: column ascending, then row descending; the copies are all legal on their tiles
            tile = min(deployed, key=lambda l: (int(l["key"].split(",")[1]), -int(l["key"].split(",")[0])))
            elite.dir = tile["piece"].dir
            self.board[tile["key"]] = elite
            where = "board"
        else:
            self._stow(elite)
            where = "hand"
        for it in items:
            self._stow(it)
        self._check_item_merges()
        if where == "board":
            self._grant_tokens_for(elite)
        return elite

    def _acquire_chess(self, cid, source, read):
        rec = self.gd.chess(cid)
        if not rec:
            return None
        base = self.gd.base_id_of(cid)
        piece = self._new("chess", cid)
        owned = None
        if not rec.get("isGolden") and self._completes_chess_merge(cid):
            owned = self._merge_chess(base, piece)
            if owned is None:
                return None
        else:
            self._stow(piece)
        self._recompute()
        self._gain_gifts(owned or piece, read)
        self._recompute()
        return owned or piece

    def _invest_repeat(self, read) -> int:
        """effectsMeta investRepeat: 投资人 doubles (triples from 100 layers) every gain's garrisons."""
        bonds = ((read or {}).get("self") or {}).get("bonds") or []
        b = next((x for x in bonds if x.get("bondId") == "investShip"), None)
        if not b or not b.get("active"):
            return 1
        bond = self.gd.bond("investShip") or {}
        bb = bond.get("bb") if isinstance(bond.get("bb"), dict) else {}
        base = bb["count"] if isinstance(bb.get("count"), int) and bb["count"] > 1 else 2
        ms = next((x for x in (bond.get("layerMilestones") or []) if isinstance(x, dict) and _num(x.get("layer")) > 0), None)
        hi = int(ms["layer"]) if ms else (bb["layer"] if isinstance(bb.get("layer"), int) and bb["layer"] > 0 else 100)
        return base + 1 if (b.get("layers") or 0) >= hi else base

    _RANDOM_GAIN = {"SERVER_POOL_EQUIP", "SERVER_MOST_BOND", "SERVER_POOL_CHAR", "SERVER_GAIN_RANDOM_EQUIP_CHESS_IN_POOL"}

    def _gain_gifts(self, piece, read):
        """onGain of an owned chess piece: its SERVER_GAIN garrisons (effectsMeta triggerGarrisons)."""
        rec = self.gd.chess(piece.id)
        if not rec:
            return
        repeat = self._invest_repeat(read)
        for gid in rec.get("garrisonIds") or []:
            g = self.gd.garrison(gid)
            if not g or g.get("eventType") != "SERVER_GAIN":
                continue
            et = g.get("effectType")
            if et == "SERVER_GAIN_EQUIP":
                item = (g.get("bbStr") or {}).get("chess")
                count = min(10, _pos((g.get("bb") or {}).get("count"), 1))
                for _ in range(repeat):
                    for _ in range(count):
                        self._acquire_item(item, requirePool=False)
            elif et == "SERVER_POOL_EQUIP":
                golden = str(gid).endswith("_b")
                count = min(10, _pos((g.get("bb") or {}).get("count"), 1))
                for _ in range(repeat):
                    for _ in range(count):
                        modal = self._modal_pool_item((g.get("bbStr") or {}).get("pool"), golden)
                        if modal is None:
                            self.unpredicted.append(("gain", et, gid))
                        else:
                            self.guessed.append(("gain", et, gid))
                            self._acquire_item(modal)
            elif et in self._RANDOM_GAIN:
                self.unpredicted.append(("gain", et, gid))

    def _modal_pool_item(self, pool_id, golden):
        """the equipment a pool roll gives most often (engine rollEquip weights); None when no item has half the weight."""
        pool = ((self.gd.raw.get("choices") or {}).get("pools") or {}).get(pool_id) or {}
        w = pool.get("weighted") or []
        if golden and isinstance(pool.get("goldenWeights"), list) and len(pool["goldenWeights"]) == len(w):
            w = [[x[0], g] for x, g in zip(w, pool["goldenWeights"])]
        pairs = [(x[0], _num(x[1])) for x in w if isinstance(x, list) and len(x) >= 2 and self.gd.item(x[0])]
        tot = sum(v for _, v in pairs)
        if not pairs or tot <= 0:
            return None
        best, bv = max(pairs, key=lambda t: t[1])
        return best if bv / tot >= 0.5 else None

    # ---------- items ----------
    def _item_mergeable(self, pid) -> bool:
        rec = self.gd.item(pid)
        if not rec or rec.get("isGolden") or rec.get("itemType") != "EQUIP" or rec.get("mergeable") is not True:
            return False
        n = self._need_item(pid)
        if not (1 < n < 100):
            return False
        gid = rec.get("upgradeChessId") or rec.get("goldenId")
        return bool(gid and self.gd.item(gid))

    def _need_item(self, pid) -> int:
        rec = self.gd.item(pid) or {}
        return rec["upgradeNum"] if isinstance(rec.get("upgradeNum"), int) else self.item_merge_count

    def _completes_item_merge(self, pid) -> bool:
        if not self._item_mergeable(pid):
            return False
        return len(self._item_locs(pid)) + 1 >= self._need_item(pid)

    def _merge_item(self, pid, incoming):
        rec = self.gd.item(pid)
        need = self._need_item(pid)
        consumed = [{"area": "new", "piece": incoming}] if incoming is not None else []
        for l in self._item_locs(pid):
            if len(consumed) >= need:
                break
            consumed.append(l)
        if len(consumed) < need:
            return None
        for l in consumed:
            if l["area"] != "new":
                self._detach(l)
        golden = self._new("item", rec.get("upgradeChessId") or rec.get("goldenId"))
        self._stow(golden)
        return golden

    def _item_fusion_partner(self, pid):
        rec = self.gd.item(pid) or {}
        if rec.get("canFuse") is not True or not isinstance(rec.get("fuseWith"), str) or not isinstance(rec.get("fuseResult"), str):
            return None
        res = self.gd.item(rec["fuseResult"])
        if not res or res.get("itemType") != "EQUIP":
            return None
        locs = self._item_locs(rec["fuseWith"])
        return locs[0] if locs else None

    def _acquire_item(self, pid, requirePool=True):
        rec = self.gd.item(pid)
        if not rec:
            return None
        piece = self._new("item", pid)
        fusion = self._item_fusion_partner(pid)
        if fusion is None and self._completes_item_merge(pid):
            piece = self._merge_item(pid, piece)
        elif fusion is None:
            if self.legacy_equip_snapshot and _consumed_on_gain(rec):
                return None
            self._stow(piece)
        else:
            self._detach(fusion)
            piece = self._new("item", self.gd.item(pid)["fuseResult"])
            self._stow(piece)
        self._recompute()
        return piece

    def _check_item_merges(self):
        """engine checkItemMerges: scan hand, temp, then equipped; the first id with enough copies merges; repeat."""
        for _ in range(200):  # loop-cap
            counts: dict = {}
            for p in self.hand + self.temp:
                if p is not None and p.kind == "item" and self._item_mergeable(p.id):
                    counts[p.id] = counts.get(p.id, 0) + 1
            for h in self._all_chess():
                for it in h.items:
                    if self._item_mergeable(it.id):
                        counts[it.id] = counts.get(it.id, 0) + 1
            did = False
            for pid, n in counts.items():
                if n >= self._need_item(pid):
                    did = self._merge_item(pid, None) is not None
                    break
            if not did:
                return

    # ---------- intents ----------
    def _owned(self):
        """Encoder.owned order: hand (10), visible temp (5), board in reading order (12)."""
        out = list(self.hand) + list(self.temp[:TEMP])
        board = [p for _, _, _, p in self._board_sorted()][:BOARD]
        return out + board + [None] * (BOARD - len(board))

    def _find_loc(self, piece):
        for i, p in enumerate(self.hand):
            if p is piece:
                return {"area": "hand", "idx": i, "piece": p}
        for i, p in enumerate(self.temp):
            if p is piece:
                return {"area": "temp", "idx": i, "piece": p}
        for k, p in self.board.items():
            if p is piece:
                return {"area": "board", "key": k, "piece": p}
        return None

    def _sell(self, j):
        piece = self._owned()[j]
        if piece is None or piece.kind != "chess":
            return False
        loc = self._find_loc(piece)
        if loc is None:
            return False
        self._detach(loc)
        self._remove_tokens_of(piece.uid)
        for it in piece.items:
            self._stow(it)
        piece.items = []
        for gid in (self.gd.chess(piece.id) or {}).get("garrisonIds") or []:
            g = self.gd.garrison(gid) or {}
            if g.get("eventType") == "SERVER_CHESS_SOLD" and g.get("effectType") == "SERVER_MOST_BOND":
                self.unpredicted.append(("sell", "SERVER_MOST_BOND", piece.id))
        self._check_item_merges()
        self._recompute()
        return True

    def apply(self, a: int, read: dict, *, after_buy=None) -> bool:
        """One executed prep action (encoder ACT numbering). `read` is the ScreenState the policy saw before it: the shop,
        the reward offer and the bonds are screen reads; hand, temp and board are the tracker's own."""
        sh = ((read or {}).get("self") or {}).get("shop") or {}
        if ACT["BUY"] <= a < ACT["SELL"]:
            s = (sh.get("slots") or [None] * SHOP)[a - ACT["BUY"]]
            if not s:
                return False
            piece = self._acquire_chess(s["id"], "buy", read) if s["kind"] == "chess" else self._acquire_item(s["id"])
            if after_buy is not None and piece is not None and piece.id == s["id"]:
                after_buy(piece)
            return piece is not None or s["kind"] != "chess"
        if ACT["SELL"] <= a < ACT["REWARD"]:
            return self._sell(a - ACT["SELL"])
        if ACT["REWARD"] <= a < ACT["PERSONAL"]:
            ro = ((sh.get("rewardOffer") or {}).get("slots")) or []
            i = a - ACT["REWARD"]
            if i >= len(ro) or not ro[i]:
                return False
            if ro[i]["kind"] == "chess":
                return self._acquire_chess(ro[i]["id"], "reward", read) is not None
            self._acquire_item(ro[i]["id"])
            return True
        return True  # END / refresh / level / freeze: no holding changes

    # ---------- layout operations (sidecar output; UNVERIFIED: the teacher prep does not run them) ----------
    def _loc_by_uid(self, uid):
        for area, row in (("hand", self.hand), ("temp", self.temp[:TEMP])):
            for i, p in enumerate(row):
                if p is not None and p.uid == uid:
                    return {"area": area, "idx": i, "piece": p}
        for k, p in self.board.items():
            if p.uid == uid:
                return {"area": "board", "key": k, "piece": p}
        return None

    def deployed_chess_count(self) -> int:
        return sum(piece.kind == "chess" for piece in self.board.values())

    def move(self, uid, to, dir_="RIGHT") -> bool:
        """sidecar `move {uid, to:[row,col], dir}`: a board tile (empty, or a swap with its occupant), from hand, temp or board."""
        loc = self._loc_by_uid(uid)
        if loc is None:
            return False
        key = f"{to[0]},{to[1]}"
        p = loc["piece"]
        occ = self.board.get(key)
        if loc["area"] == "board" and loc["key"] == key:  # re-orientation on its own tile
            p.dir = dir_
            return True
        if p.kind == "token":
            owner = self._loc_by_uid(p.owner)
            if occ is not None or not owner or owner["area"] != "board" or owner["piece"].kind != "chess":
                return False
            if loc["area"] != "board" and p.count > 1:
                p.count -= 1
                self.board[key] = self._new("token", p.id, owner=p.owner, dir=dir_)
                self._recompute()
                return True
        self._detach(loc)
        if occ is not None:
            self.board.pop(key)
            if loc["area"] == "board":
                self.board[loc["key"]] = occ
            else:
                self._stow(occ)  # the occupant leaves the board into the row the piece came from (first free slot)
        self.board[key] = p
        p.dir = dir_
        if loc["area"] != "board" and p.kind == "chess":
            self._grant_tokens_for(p)
        self._recompute()
        return True

    def withdraw_slot(self, uid: int) -> int:
        """The rightmost slot available once the withdrawn owner's summons leave."""
        loc = self._loc_by_uid(uid)
        if loc and loc["piece"].kind == "token":
            token = loc["piece"]
            return next((i for i, p in enumerate(self.hand)
                         if p and p.kind == "token" and p.id == token.id and p.owner == token.owner),
                        rightmost_free(self.hand))
        return next((i for i in range(len(self.hand) - 1, -1, -1)
                     if self.hand[i] is None or (self.hand[i].kind == "token" and self.hand[i].owner == uid)), -1)

    def bench(self, uid) -> bool:
        """sidecar `bench {uid}`: a board piece back to the hand."""
        loc = self._loc_by_uid(uid)
        if loc is None or loc["area"] != "board":
            return False
        piece = loc["piece"]
        if piece.kind == "token":
            slot = self.withdraw_slot(uid)
            if slot < 0:
                return False
            self._detach(loc)
            if self.hand[slot] is None:
                piece.dir = None
                self.hand[slot] = piece
            else:
                self.hand[slot].count += piece.count
            self._recompute()
            return True
        self._detach(loc)
        self._remove_tokens_of(uid)
        self._stow(loc["piece"])
        self._recompute()
        return True

    def destroy(self, uid) -> bool:
        """Remove a visible loose item after the device has verified destruction."""
        loc = self._loc_by_uid(uid)
        if loc is None or loc["area"] not in {"hand", "temp"} or loc["piece"].kind != "item":
            return False
        self._detach(loc)
        self._recompute()
        return True

    def destroy_chess(self, uid: int) -> bool:
        loc = self._loc_by_uid(uid)
        if loc is None or loc["piece"].kind != "chess":
            return False
        self._detach(loc)
        self._remove_tokens_of(uid)
        for item in loc["piece"].items:
            self._stow(item)
        self._check_item_merges()
        self._recompute()
        return True

    def equip(self, item_uid, target_uid, replace_uid=None) -> bool:
        """sidecar `equip {item, target}`: a loose item onto a chess (at most 2; a full carrier replaces replace_uid or the oldest)."""
        loc = self._loc_by_uid(item_uid)
        tgt = self._loc_by_uid(target_uid)
        if loc is None or tgt is None or loc["piece"].kind != "item" or tgt["piece"].kind != "chess" or loc["area"] == "board":
            return False
        item = loc["piece"]
        self._detach(loc)
        holder = tgt["piece"]
        if len(holder.items) >= 2:
            j = next((i for i, x in enumerate(holder.items) if x.uid == replace_uid), 0)
            holder.items.pop(j)  # the replaced item is destroyed
        holder.items.append(item)
        self._recompute()
        return True

    # ---------- calibration and views ----------
    def calibrate(self, screen: dict):
        """Reset from a read ScreenState (prep start, and after a card or reward pick the harness re-reads)."""
        S = screen["self"]
        self.hand = [None] * HAND
        self.temp = [None] * TEMP
        self.board = {}
        for i, p in enumerate((S.get("hand") or [])[:HAND]):
            if p:
                self.hand[i] = self._from_screen(p)
        for i, p in enumerate((S.get("temp") or [])[:TEMP]):
            if p:
                self.temp[i] = self._from_screen(p)
        for p in S.get("board") or []:
            if p:
                self.board[f"{p['row']},{p['col']}"] = self._from_screen(p, dir_=p.get("dir"))
        # a summon's owner: the one chess piece on the board or in the hand whose token list names it (unique, else none)
        chess = self._all_chess()
        for p in self.hand + list(self.board.values()):
            if p is None or p.kind != "token":
                continue
            names = (self.gd.token(p.id) or {}).get("owners") or []
            cand = [o for o in chess if o.id in names]
            p.owner = cand[0].uid if len(cand) == 1 else None

    def _from_screen(self, p, dir_=None):
        kind = p["kind"]
        items = [self._new("item", x) for x in (p.get("items") or [])] if kind == "chess" else []
        q = self._new(kind, p["id"], items=items)
        if dir_:
            q.dir = dir_
        return q

    def to_screen(self, *, include_overflow: bool = False) -> dict:
        """The tracker's holdings in ScreenState form (self.hand / self.temp / self.board), uids are the harness's own
        (the screen shows none). The hidden overflow queue is not on the screen: its size is reported as hidden_temp."""
        def pc(p, row=None, col=None, dir_=None):
            if p is None:
                return None
            d = {"uid": p.uid, "kind": p.kind, "id": p.id, "items": [it.id for it in p.items] if p.kind == "chess" else []}
            if p.kind == "token" and p.owner is not None:
                d["ownerUid"] = p.owner
                d["count"] = p.count
            if p.bought:
                d["boughtRound"] = p.bought
            if row is not None:
                d.update({"row": row, "col": col, "dir": dir_ or p.dir})
            return d
        return {"self": {"hand": [pc(p) for p in self.hand],
                         "temp": [pc(p) for p in (self.temp if include_overflow else self.temp[:TEMP])],
                         "board": [pc(p, r, c, p.dir) for r, c, _, p in self._board_sorted()]},
                "hidden_temp": max(0, len(self.temp) - TEMP)}

    def snapshot(self):
        """Comparable view: hand (10), visible temp (5), board in reading order; each piece (kind, id, items)."""
        def pc(p):
            return None if p is None else (p.kind, p.id, tuple(it.id for it in p.items))
        hand = tuple(pc(p) for p in self.hand)
        temp = tuple(pc(p) for p in self.temp[:TEMP])
        board = tuple((r, c, *pc(p)) for r, c, _, p in self._board_sorted())
        return hand, temp, board


def screen_snapshot(screen: dict):
    """The same view of a ScreenState (engine truth, as the screen shows it)."""
    S = screen["self"]

    def pc(p):
        return None if not p else (p["kind"], p["id"], tuple(p.get("items") or ()))
    hand = tuple(pc(p) for p in ((S.get("hand") or []) + [None] * HAND)[:HAND])
    temp = tuple(pc(p) for p in ((S.get("temp") or []) + [None] * TEMP)[:TEMP])
    board = tuple((p["row"], p["col"], *pc(p)) for p in (S.get("board") or []) if p)
    return hand, temp, board
