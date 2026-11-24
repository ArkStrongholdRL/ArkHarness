"""Offline check of the soft-failure tier in harness.live (no device, no game input)."""
import sys, types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.catalog import load
from harness.checkpoint import load_checkpoint, restore_track, save_checkpoint
from harness.encode.encoder import ACT, HAND, TEMP
from harness.execution import device_action, mask_unverified_prep_actions, recover_action
from harness.live import Live, GameState, Retry, Stop, RETRY_LIMIT
from harness.rules import Rules
from harness.layout import plan_layout
from harness.layout_recovery import recover_layout_step
from harness.screen.flow import NavigationRecognition, execute
from harness.screen.deploy import DeployRecognition, read_board
from harness.screen.board_read import read_board_tiles
from harness.screen.equip import equipment_funds_match
from harness.screen.holdings import reconcile_holdings
from harness.screen.read import CATALOG, Reader, expanded_shop_level, read_json, shop_reward_choices
from harness.screen.state import require_action, require_return
from harness.screen.withdraw import WithdrawRecognition
from harness.track import Track

def make(prep_behaviour):
    live = object.__new__(Live)
    live.a = types.SimpleNamespace(resume=False, once=False)
    live.state = GameState(round=5)
    live._retries, live._shop_open, live._prep_chain = {}, True, True
    live.logs = []
    live.log = lambda row: live.logs.append(row)
    live.save = lambda: None
    live.screenshot = lambda name: None
    live.observe = lambda: "preparation"
    calls = {"n": 0}
    def prep():
        calls["n"] += 1
        return prep_behaviour(calls["n"])
    live.prep = prep
    return live, calls

# 1. soft_fail spends the budget, then becomes a Stop with the original reason
live, _ = make(lambda n: None)
for i in range(RETRY_LIMIT):
    try: live.soft_fail("x", "mismatch"); raise SystemExit("no Retry raised")
    except Retry as e: assert f"{i+1}/{RETRY_LIMIT}" in str(e), e
    assert live._shop_open is False and live._prep_chain is False
try: live.soft_fail("x", "mismatch"); raise SystemExit("no Stop raised")
except Stop as e: assert "mismatch" in str(e) and "still failing" in str(e), e
live.state.round = 6  # a new round has a fresh budget
try: live.soft_fail("x", "mismatch")
except Retry: pass

# 2. run(): an executor error twice, then success -> the loop re-observes and finishes; nothing stops
def flaky(n):
    if n <= 2: raise RuntimeError("A release was attempted; placement is unverified")
    return True
live, calls = make(flaky)
assert live.run(1) == 0 and calls["n"] == 3, (calls, live.logs)
assert [r["decision"] for r in live.logs].count("executor retry") == 2

# 3. run(): a Retry from a soft check is logged and the loop continues
def soft(n):
    if n == 1: live2.soft_fail("holdings", "round or funds changed")
    return True
live2, calls = make(soft)
assert live2.run(1) == 0 and calls["n"] == 2 and live2.logs[0]["decision"] == "retry"

# 4. run(): persistent failure stops after the budget (exit 3), it does not loop forever
live3, calls = make(lambda n: (_ for _ in ()).throw(RuntimeError("always")))
assert live3.run(1) == 3 and calls["n"] == RETRY_LIMIT + 1, calls

# 5. a hard Stop is never retried
live4, calls = make(lambda n: (_ for _ in ()).throw(Stop("hard")))
assert live4.run(1) == 2 and calls["n"] == 1

# 6. screens other than preparation / specialSelection do not get the executor retry
live5, calls = make(lambda n: None); live5.observe = lambda: "battle"
live5.flow = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("battle executor"))
live5.acting = lambda: True; live5._prep_chain = False
assert live5.run(1) == 3 and calls["n"] == 0

# 7. An interrupted input leaves its intent on disk. Resume reads the affected holdings once.
with TemporaryDirectory() as directory:
    path = Path(directory) / "state.json"
    policy = Path("P7-model-final.pt")
    track = Track(load())
    state = GameState(round=5, dirty_areas=["temp"])

    def save():
        save_checkpoint(path, state, track, policy)

    try:
        with device_action(state, save, "sell22", ("board",)):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        checkpoint = load_checkpoint(path)
        state = GameState(**checkpoint["state"])
        assert restore_track(track, checkpoint)
        assert recover_action(state) == {"action": "sell22", "round": 5, "areas": ["board"]}
        assert state.dirty_areas == ["board", "temp"] and recover_action(state) is None

    # A verified receipt commits its bookkeeping without adding recovery work.
    with device_action(state, save, "refresh"):
        state.actions += 1
    assert load_checkpoint(path)["state"]["actions"] == 1
    assert load_checkpoint(path)["state"]["pending_action"] is None

    # An unverified layout preserves its verified prefix and narrows recovery to the failed step.
    try:
        with device_action(state, save, "layout", ("board", "hand")) as attempt:
            state.actions += 1
            save()
            attempt["areas"] = ["board"]
            raise RuntimeError("unverified move")
    except RuntimeError:
        saved = load_checkpoint(path)["state"]
        assert saved["dirty_areas"] == ["board", "temp"]
        assert saved["actions"] == 2 and saved["pending_action"] is None

# 8. Make room once, then build the layout from the observed balance and holdings.
live6, _ = make(lambda n: None)
before = {"match": {"round": 5, "stageId": "act3autochess_m09"}, "self": {"funds": 0}}
after = {"match": before["match"], "self": {"funds": 1}}
plans = iter([{"ops": [{"op": "sell", "uid": 7}]}, {"ops": []}])
requests = []
sales = []

def plan(operation, state):
    requests.append((operation, state))
    return next(plans)

def sell(operations, rnd, stage):
    sales.append((operations, rnd, stage))
    return [{"op": "sell", "uid": 7}]

live6._uids = lambda: {7}
live6.sc = types.SimpleNamespace(request=plan)
live6.resolve_visible = sell
live6.read = lambda: after
live6.screen_state = lambda screen: screen
receipts = {}
assert live6.plan_end(before, receipts) == (after, {"ops": []})
assert receipts == {"makeRoom": [{"op": "sell", "uid": 7}]}
assert sales == [([("sell", 7)], 5, "act3autochess_m09")]
assert requests == [("end", before), ("end", after)]

# 9. A confirmed battle marker supplies the transition snapshot without running scene OCR again.
nav = NavigationRecognition({}, "ready-battle", None, None)
nav.started_round = 12
args = types.SimpleNamespace(image=np.zeros((720, 1280, 3), dtype=np.uint8),
                             custom_recognition_param='"startedRound"')
with patch("harness.screen.flow.Frame", return_value=types.SimpleNamespace(text=lambda name: "回合12")):
    assert nav.analyze(None, args) is not None
require_return("ready-battle", {"round": 12}, nav.snapshot)

# A deployment guard reads the starting HUD even when touch input has not begun.
deployment = DeployRecognition("act3autochess_m02", (940, 610), "角峰", (636, 232), (636, 231), "RIGHT",
                               expected_round=9, stage_confirmed=True)
hud = types.SimpleNamespace(scene=lambda: "preparation", hit=lambda name, **_: name == "ArkRestText",
                            text=lambda name: {"ArkRoundNumber": "回合9", "Screen.DeployCollapsed": "剩余可放置角色:1"}[name])
with patch("harness.screen.deploy.Frame", return_value=hud):
    assert deployment.analyze(None, types.SimpleNamespace(image=args.image, custom_recognition_param='"pick"')) == (
        940, 610, 1, 1)
assert (deployment.round, deployment.remaining) == (9, 1)

# A transient empty hit on a saved tile retries only that tile when the HUD still counts its occupant.
layout = {(9, 3): ("赫德雷", "RIGHT"), (9, 4): ("佩佩", "LEFT")}
inspected = []
board_tasker = types.SimpleNamespace(resource=types.SimpleNamespace())


def set_board_recognition(_, recognition):
    board_tasker.recognition = recognition
    return True


def inspect_saved_tile(_, entry, terminal, **kwargs):
    recognition = board_tasker.recognition
    if entry == "BoardRead.Scan":
        recognition.hud = (14, 11, 6)
        recognition.markers = {tile: (facing,) for tile, (_, facing) in layout.items()}
        return terminal
    tile = recognition.tile
    inspected.append(tile)
    if tile == (9, 4) and inspected.count(tile) == 1:
        return "BoardRead.Empty"
    recognition.identity = {"name": layout[tile][0]}
    return terminal


board_tasker.resource.register_custom_recognition = set_board_recognition
board_tasker.resource.unregister_custom_recognition = lambda _: True
root = Path(__file__).resolve().parent.parent
board = read_board(root / "calib/board.m02.bossLeft.json")
reader = Reader(CATALOG, read_json(root / "calib/slots.json"))
with patch("harness.screen.board_read.require_terminal", side_effect=inspect_saved_tile):
    scanned = read_board_tiles(board_tasker, reader, board, known_layout=layout)
assert inspected == [(9, 3), (9, 4), (9, 4)]
assert [(p["name"], p["facing"]) for p in scanned["pieces"]] == list(layout.values())

# A tactical draft with its round HUD covered is a valid post-battle destination.
tactic_screen = {"screen": "tacticSelection", "round": None, "canReady": False}
require_return("wait-battle", {"round": 10}, tactic_screen)
require_action("select-tactic", tactic_screen)

# An accepted elite tactic applies once to the next normal operator gained by buying.
live, _ = make(lambda _: None)
live.gd = load()
live.track = Track(live.gd, legacy_equip_snapshot=False)
live.state.tactic_upgrades["chess"] = 1
base_id = "chess_char_3_18_a"
shop_state = {"self": {"shop": {"slots": [{"kind": "chess", "id": base_id}]}}}
assert live.apply_confirmed_gain(ACT["BUY"], shop_state, base_id)
assert {piece.id for piece in live.track.hand if piece} == {live.gd.golden_id_of(base_id)}
assert live.state.tactic_upgrades["chess"] == 0
assert live.apply_confirmed_gain(ACT["BUY"], shop_state, base_id)
assert {piece.id for piece in live.track.hand if piece} == {base_id, live.gd.golden_id_of(base_id)}
assert nav.snapshot == {"screen": "battle", "round": 12, "canReady": False}

# A transition animation can hide the battle badge during the next passive observation.
def battle_frames(tasker, recognition, timeout):
    recognition.snapshot = ({"screen": "roundHud", "round": 12, "canReady": False}
                            if recognition.action == "observe" else
                            {"screen": "preparation", "round": 13, "canReady": True})

with patch("harness.screen.flow.run_navigation", side_effect=battle_frames):
    receipt = execute(None, {"buttons": {}}, "wait-battle", Path(), None, 1000, battle_round=12)
assert receipt["after"]["round"] == 13

# 10. OCR kept the price of one offer and lost its identity. The policy can still choose a verified offer.
mask = np.ones(100, dtype=np.uint8)
mask_unverified_prep_actions(mask, {
    "shop": {"slots": [{"id": "chess_char_1_01_a", "price": 2}, {"id": None, "price": 2}]},
    "board": [{"uid": 7, "row": 9, "col": 4}, {"uid": 8, "row": 9, "col": 5}],
}, [[7, 9, 4]])
assert mask[ACT["BUY"]] == 1 and not mask[ACT["BUY"] + 1:ACT["SELL"]].any()
assert mask[ACT["SELL"] + HAND + TEMP:ACT["SELL"] + HAND + TEMP + 2].tolist() == [0, 1]

# 11. Verified purchases of both economy dolls stay in the live rack until equipped before policy inference.
live7, _ = make(lambda n: None)
gd = load()
live7.gd = gd
live7.track = Track(gd, legacy_equip_snapshot=False)
live7.rules = Rules(gd)
ids = ("chess_item_2_05_e_a", "chess_item_2_07_e_a", "chess_item_1_04_e_a")
offers = {"self": {"shop": {"slots": [{"kind": "item", "id": i} for i in ids]}}}
for index in range(len(ids)):
    assert live7.track.apply(ACT["BUY"] + index, offers)
assert {p.id for p in live7.track.temp if p} == set(ids)
carrier = live7.track._new("chess", "chess_char_1_05_a")
live7.track.board["9,4"] = carrier
live7._prep_chain = False
live7.force_holdings_scan = False
live7.state.prep_round, live7.state.lp0 = 5, 40
live7.a.stop_before_end = True
live7.acting = live7.can_inspect_panels = lambda: True
live7.use_mode = lambda value: None
stage = "act3autochess_m05"
snapshot = {"match": {"round": 5, "stageId": stage}, "config": {}, "self": {"shop": {}}}
live7.read = lambda: snapshot
live7.board_calibration = lambda actual: actual
live7.screen_state = lambda s: s
equipped = []

def equip(operation, board, round_number):
    assert operation["target"] == carrier.uid and board == stage and round_number == 5
    piece = live7.track._loc_by_uid(operation["item"])["piece"]
    equipped.append(piece.id)
    assert live7.track.destroy(piece.uid)
    return {"name": piece.id, "sourceCleared": True}

def decide_after_dolls(s, kind):
    assert set(equipped) == set(ids[:2])
    assert [p.id for p in live7.track.temp if p] == [ids[2]]
    return ACT["END"], {"decision": kind}

live7.apply_equipment = equip
live7.decide = decide_after_dolls
assert Live.prep(live7) is True

# 12. The same drag hands off each effect and replans only when its result changes policy inputs.
def equipment_live(item_id, *, worn=()):
    live, _ = make(lambda n: None)
    live.gd, live.rules = gd, Rules(gd)
    live.track = Track(gd, legacy_equip_snapshot=False)
    live.reader, live.tasker = types.SimpleNamespace(), None
    item = live.track._acquire_item(item_id)
    carrier = live.track._new("chess", "chess_char_1_05_a",
                              items=[live.track._new("item", key) for key in worn])
    live.track.board["9,4"] = carrier
    return live, {"item": item.uid, "target": carrier.uid}, carrier

board = {"stageId": stage}
direct = {
    "chess_item_1_04_e_a": ((), None),
    "chess_item_2_02_e_a": (("board", "hand", "temp"), None),
    "chess_item_2_06_e_a": (("board", "hand", "temp"), None),
    "chess_item_3_11_e_a": ((), (1, 6)),
    "chess_item_3_15_e_a": ((), (0, None)),
    "chess_item_5_03_e_a": (("board", "hand", "temp"), None),
    "chess_item_5_09_e_a": ((), None),
    "chess_item_6_02_e_a": ((), None),
}
for item_id, (dirty, gain_range) in direct.items():
    live, operation, carrier = equipment_live(item_id)
    with patch("harness.screen.equip.equip_to_board", return_value={"sourceCleared": True}) as drag:
        receipt = live.apply_equipment(operation, board, 5)
    assert drag.call_args.kwargs["gain_range"] == gain_range
    assert live.track._loc_by_uid(operation["item"]) is None and receipt["consumed"]
    assert tuple(live.state.dirty_areas) == dirty and not carrier.items
    if item_id == "chess_item_6_02_e_a":
        assert live.state.deploy_cap_floor == live.reader.deploy_cap_floor == 9

for item_id, kind in (("chess_item_4_15_e_a", "item"), ("chess_item_5_02_e_a", "chess"),
                      ("chess_item_5_06_e_a", "command")):
    live, operation, _ = equipment_live(item_id)
    with patch("harness.screen.equip.equip_to_board", return_value={}) as drag:
        live.apply_equipment(operation, board, 5)
    assert drag.call_args.kwargs["opens_choice"] and live.state.pending_equipment_reward["kind"] == kind
    assert live.track._loc_by_uid(operation["item"])

assert equipment_funds_match(3, None, (1, 6)) and not equipment_funds_match(7, None, (1, 6))
assert equipment_funds_match(0, None, (0, None)) and not equipment_funds_match(-1, None, (0, None))
dr = [{"kind": "chess", "sold": True}, {"kind": "item", "id": "offer", "price": 0, "sold": False}]
assert shop_reward_choices(dr, min_cards=1, kind="item") == [(1, dr[1])]
assert not shop_reward_choices(dr, min_cards=1, kind="chess")

live, operation, carrier = equipment_live("chess_item_1_05_e_a", worn=("chess_item_1_01_e_a", "chess_item_1_02_e_a"))
with (patch("harness.screen.equip.equip_to_board", return_value={"before": {"funds": 4}}),
      patch("harness.screen.equip_choice.choose_equipment_dialog", return_value={"round": 5, "funds": 4}) as choice,
      patch("harness.screen.equip.verify_rack_source") as verify):
    receipt = live.apply_equipment(operation, board, 5)
assert choice.call_args.args[2] == {"维式重锤": "chess_item_1_01_e_a"}
assert verify.called and receipt["replaced"] == "chess_item_1_01_e_a"
assert [p.id for p in carrier.items] == ["chess_item_1_02_e_a", "chess_item_1_05_e_a"]

live, operation, _ = equipment_live("chess_item_4_15_e_a")
with patch("harness.screen.equip.equip_to_board", return_value={}):
    live.apply_equipment(operation, board, 5)
free_item = {"kind": "item", "id": "chess_item_1_05_e_a", "price": 0, "sold": False}
offers = [dr[0], free_item, {**free_item, "id": "chess_item_1_01_e_a"}, {**free_item, "id": "chess_item_1_02_e_a"}]
snapshot = {"match": {"round": 5}, "self": {"shop": {"slots": offers}}}
live.screen_state = lambda s: {"self": {"shop": {}}}
live.decide = lambda st, kind, only: (ACT["REWARD"], {"decision": kind})
live.acting = lambda: True
with patch("harness.screen.shop.pick_reward", return_value={"rewardClaimed": True}) as pick:
    live.finish_reward(snapshot)
assert pick.call_args.args[2] == 1 and pick.call_args.kwargs["kind"] == "item"
assert live.state.pending_equipment_reward is None and live.reader.reward_kind_hint is None
assert live.track._loc_by_uid(operation["item"]) is None and live.state.dirty_areas == ["temp"]

# 13. An interrupted placement reveals the occupant but supplies no facing receipt.
live, operation, carrier = equipment_live("chess_item_1_05_e_a")
live.state.layout_step = {"round": 5, "id": carrier.id, "target": [9, 5]}
assert live.board_ledger(5) == {(9, 4): (live.name(carrier.id), carrier.dir),
                               (9, 5): (live.name(carrier.id), None)}
scan = {"board": [{"row": 9, "col": 5, "facing": None, "id": carrier.id, "candidateIds": [carrier.id]}],
        "remaining": 7, "unreadTiles": [], "hand": None, "temp": None, "facingSource": "track"}
assert reconcile_holdings(live.track, scan, deploy_cap=8)["applied"]
assert live.track.board["9,5"] is carrier and carrier.dir is None
profile = {"uid": carrier.uid, "id": carrier.id, "key": "9,5", "dir": None, "cls": "melee"}
state = {"pieces": [profile], "map": {"9,5": "melee"}, "deployCap": 8}
target = {"board": [{"uid": carrier.uid, "to": [9, 5], "dir": "RIGHT"}]}
steps = plan_layout(live.track, state, target, {(9, 5)})
assert [(step.source, step.target, step.direction) for step in steps] == [
    ((9, 5), None, None), (None, (9, 5), "RIGHT"),
]

# 14. Recover one interrupted withdrawal by inspecting just its source tile and receiving hand slot.
for applied in (True, False):
    tr = Track(gd, legacy_equip_snapshot=False)
    operator = tr._new("chess", "chess_char_1_11_a", dir="LEFT")
    other = tr._new("chess", "chess_char_1_05_a")
    tr.board["10,4"], tr.hand[0] = operator, other
    step = {"round": 5, "stageId": stage, "uid": operator.uid, "id": operator.id,
            "source": [10, 4], "target": None, "handSlot": 9}
    screen = {"match": {"round": 5}, "self": {"funds": 1, "deployCap": 8, "deployCount": int(not applied)}}
    hud = {"round": 5, "funds": 1, "expanded": True}
    hand = {"stageId": stage, "round": 5, "funds": 1, "remaining": 8 - int(not applied),
            "slots": [{"screenSlot": 9, "status": "occupied" if applied else "empty", "id": operator.id}]}
    with (patch("harness.layout_recovery.set_shop_open", return_value={"before": hud, "after": hud}),
          patch("harness.layout_recovery.read_hand", return_value=hand) as read_slots,
          patch("harness.layout_recovery.read_equipment_slots") as read_rack,
          patch("harness.layout_recovery.inspect_board_positions",
                return_value={(10, 4): None if applied else {"id": operator.id}}) as read_tiles):
        result = recover_layout_step(None, None, {"stageId": stage}, tr, step, screen)
    assert result == {"applied": applied, "handSlots": [9], "rackSlots": [], "tiles": [(10, 4)]}
    assert read_slots.call_args.args[2] == [9] and read_tiles.call_args.args[3] == [(10, 4)]
    assert not read_rack.called and tr.hand[0].uid == other.uid
    assert (tr.hand[9].uid if applied else tr.board["10,4"].uid) == operator.uid

tr = Track(gd, legacy_equip_snapshot=False)
operator = tr._new("chess", "chess_char_1_11_a", dir="LEFT")
tr.board["10,4"] = operator
step = {"round": 5, "stageId": stage, "uid": operator.uid, "id": operator.id,
        "source": [10, 4], "target": [10, 5], "handSlot": None}
with (patch("harness.layout_recovery.set_shop_open", return_value={"before": hud, "after": hud}),
      patch("harness.layout_recovery.read_hand") as read_slots,
      patch("harness.layout_recovery.inspect_board_positions",
            return_value={(10, 4): None, (10, 5): None}) as read_tiles):
    try:
        recover_layout_step(None, None, {"stageId": stage}, tr, step, screen)
    except RuntimeError as e:
        assert "Could not determine" in str(e)
    else:
        raise AssertionError("An inconclusive probe must not update the ledger")
assert not read_slots.called and read_tiles.call_args.args[3] == [(10, 4), (10, 5)]
assert tr.board == {"10,4": operator} and operator.dir == "LEFT"

# A drifting LEVEL text box must leave the calibrated IV digit crop fixed.
for heading_box in ((18, 562, 76, 22), (28, 557, 94, 31)):
    frame = types.SimpleNamespace(
        rows=lambda _: [types.SimpleNamespace(box=heading_box)],
        text=lambda _, *, roi: "IV" if roi == [33, 584, 60, 43] else "V",
    )
    assert expanded_shop_level(frame) == 4

# Recover observed shop OCR variants without guessing unrelated card names.
reader = Reader(CATALOG, read_json(Path(__file__).resolve().parent.parent / "calib/slots.json"))
position = reader.slots["shop"]["layouts"]["4"][1]
for label, expected in (("伊桑", "chess_char_3_18_a"), ("伊桑。", "chess_char_3_18_a"),
                        ("伊桑·", "chess_char_3_18_a"), ("w", "chess_char_4_17_a"),
                        ("罗比拉塔", "chess_char_3_19_a"), ("伊来", None), ("贝洛内+", None)):
    frame = types.SimpleNamespace(hit=lambda *a, **kw: False,
                                  text=lambda name, **kw: label if name == "Screen.ShopCardName" else "3")
    card = reader.read_shop_slot(frame, position)
    assert (card["id"] if card else None) == expected, label
    assert reader.read_shop_slot(frame, {**position, "kind": "item"}) is None, label

# 15. A withdrawal receipt requires the matching hand detail and a subsequent closed preparation view.
withdraw = WithdrawRecognition(stage, (556, 231), (539, 609), "夏栎", 12, True)
withdraw.before = {"round": 12, "funds": 0, "remaining": 1}
panel = {"ArkRestText"}
text_nodes = {"ArkRoundNumber": "回合12", "Screen.Funds": "0", "Screen.DeployCollapsed": "剩余可放置角色:2",
              "Board.CarriedName": "夏栎"}
frame = types.SimpleNamespace(scene=lambda: "preparation", hit=lambda name: name in panel,
                              text=lambda name: text_nodes.get(name, ""))
with patch("harness.screen.withdraw.Frame", return_value=frame):
    args.custom_recognition_param = '"verified"'
    assert withdraw.analyze(None, args) is None
    panel.add("HandSale.Button")
    args.custom_recognition_param = '"hand"'
    assert withdraw.analyze(None, args) is not None
    args.custom_recognition_param = '"verified"'
    assert withdraw.analyze(None, args) is None
    panel.remove("HandSale.Button")
    assert withdraw.analyze(None, args) is not None and withdraw.result["sourceCleared"]
print("soft retry checks passed")
