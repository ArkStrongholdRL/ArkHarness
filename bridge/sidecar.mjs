// Node sidecar of the live harness (GOAL-live C): legacy layout and equipment on a shadow engine match built from a
// ScreenState, until our own layout / equipment models replace the `layout` and `equip` requests.
//
// stdio, one JSON object per line; every response echoes `id`; errors are {"id", "ok": false, "error"}.
//   {"id", "op": "layout", "state", "final"?: bool}   bot.js arrangeSteps (its selfPlan inside; final: the last arrangement)
//   {"id", "op": "equip",  "state"}                   bot.js equipItems only (not the botUseCommand it wraps in production)
//   {"id", "op": "end",    "state", "command"?}       the P7 END flow (env.mjs LAYOUT_ONLY → botPrep.js): arrange, equip,
//                                                    the battle-mark command Python decided (B4: {effectId, target}),
//                                                    final arrange. resolveTemp and Ready stay with Python (they run after it).
//   {"id", "op": "selfPlan", "state"}                 bot.js selfPlan at the START of a prep (env.mjs startOfPrepOnce): legacy's
//                                                    own focus bond → {"ok", "botSelf": {round, focus} | null}; the harness keeps
//                                                    it and sends it back as `self.botSelf` with the END requests of that prep
//   {"id", "op": "tail", "state", "ops"}             after Python's resolveTemp (`ops`, applied on the shadow): legacy's
//                                                    prepEnd step `if no free hand slot: freeHandSlot` (bot.js; sells the weakest)
//   {"id", "op": "preview", "round", "bossId", "kinds", "bounties"?, "factions"?}
//                                                    bridge/preview.mjs previewFromKinds (match 'set'): the round's full preview
//                                                    {"ok", "preview": [{enemyKey, count, gate, fly, elite, boss, t, source}], "approx"}
//   {"id", "op": "killChance", "state", "card"}       bridge/bounty.mjs
// layout / equip / end answer {"ok": true, "ops": [...], "board": [...], "hand": [...], "temp": [...]}:
//   ops in the order made, the executor's operations: {op:'move', uid, from, to:[row,col], dir} | {op:'bench', uid, idx}
//   | {op:'equip', item, target} | {op:'sell', uid} | {op:'destroy', uid} | {op:'command', effectId, target};
//   board = the final board in the engine's Map order [{uid, kind, id, row, col, dir, items}].
//
// The state is a ScreenState (bridge/screen.mjs fromEngine) plus the self-tracked `self.bounties` ([{card:{enemyKey,
// count}}]) and `self.commandMarks`, and the account's 自选编队 picks `self.diy` (config); summons carry `ownerUid`. The wave is rebuilt from the preview's enemy kinds
// (bridge/preview.mjs). P7's training conditions: solo mode_single_20, botRehearsal 0, natPlan 'order' advisor plan.
//
//   node bridge/sidecar.mjs [engine-root]
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { previewFromKinds } from './preview.mjs';
import { killChance } from './bounty.mjs';

const NAT_ORDER = ['ursusShip', 'lethaniaShip'];

/** Load the engine modules the shadow needs (act3 data version first). */
export async function loadEngine(root = 'E:/ArkHarness-engine/sp-label') {
  const imp = (p) => import(pathToFileURL(`${resolve(root)}/${p}`).href);
  await imp('tools/rl/act3.mjs');
  const [{ Match }, { VirtualScheduler }, { getData }, { defineBotStrategy }, bot, waves, { computeBonds }, pool] = await Promise.all([
    imp('server/match/Match.js'), imp('server/match/scheduler.js'), imp('server/data.js'), imp('server/match/botStrategyContract.js'),
    imp('server/match/bot.js'), imp('server/match/waves.js'), imp('server/match/bondsMeta.js'), imp('server/match/pool.js')]);
  const data = getData({ log: { warn() {}, error() {}, info() {} } });
  const { createRng } = await imp('server/sim/rng.js');
  const { GameData } = await imp('server/match/gamedata.js');
  const tables = JSON.parse(readFileSync(new URL('../data/wave_tables.json', import.meta.url)));
  return { Match, VirtualScheduler, data, defineBotStrategy, bot, waves, tables, computeBonds, pool, createRng, GameData };
}

function soloMode(E, id = 'mode_single_20') {
  const mode = E.data.config.modes[id];
  if (!mode || mode.type !== 'SINGLE') throw new Error(`unknown solo mode ${id}`);
  return mode;
}

/** The natPlan advisor plan P7 trained with (env.mjs ~822): P = first of NAT_ORDER the picked group makes playable. */
export function installPlan(m) {
  m.externalAI = { close() {}, observe() {}, receive() {}, journalConfig() { return {}; }, teamFor() { return null; }, battleFor() { return null; },
    inflight: new Set(), waitForIdle: async () => {},
    planFor: () => {
      if ((m.bondChoice && m.bondChoice.picked === null) || (m.round || 0) < 1) return null;
      const gd = m.gd, pl = gd.bondIds.filter((b) => gd.bond(b)?.isCore && (!gd.regulationBondDraw || m.bondPlayable(b)));
      const P = NAT_ORDER.find((b) => pl.includes(b));
      return P ? { bonds: [{ id: P, priority: 5 }] } : null;
    } };
}

/** A fresh solo match at its first prep, with a seat whose prep hooks do nothing (the shadow writes everything itself). */
function baseMatch(E, seed, modeId) {
  const sched = new E.VirtualScheduler();
  const idle = E.defineBotStrategy({ id: 'shadow', *prepBegin() { return null; }, *prepEnd() {} });
  const mode = soloMode(E, modeId);
  const m = new E.Match({ roomCode: 'SHDW', mode: 'solo', difficulty: mode.difficulty, modeId: mode.modeId,
    seats: [{ seat: 0, playerId: 'ai_0', name: 'shadow', isBot: true, connected: true }], seed, data: E.data, scheduler: sched,
    botRehearsal: 0, botStrategy: idle, log: { info() {}, debug() {}, warn() {}, error() {} }, send: () => true, broadcast: () => {}, onEnd: () => {} });
  m.start();
  for (let g = 0; g < 1e5 && !(m.phase === 'PREP' && m.round >= 1); g++) if (!sched.runNext()) break;
  if (m.phase !== 'PREP') throw new Error(`shadow match did not reach a prep (${m.phase})`);
  return m;
}

/**
 * Shadow match + seat for ScreenState `s`. opts.wave / opts.bossWave: the wave object to use (parity: the real one);
 * default: rebuilt from the preview kinds. opts.rng: an rngBots stream to use (parity: a clone of the real one).
 */
export function buildShadow(E, s, opts = {}) {
  const M = s.match, S = s.self;
  const m = baseMatch(E, opts.seed ?? 1, s.config?.modeId ?? undefined);
  const ps = m.players.get('ai_0');
  const gd = m.gd;
  m.round = M.round;
  m.stageId = M.stageId; m.stage = gd.stage(M.stageId);
  m.bossId = M.bossId ?? m.bossId; if (M.hiddenBossId) m.hiddenBossId = M.hiddenBossId;
  m.disabledBonds = [...(M.disabledBonds || [])];
  // the match's pool follows its bans (Match.js: bannedChessForBonds over the drawn + static bonds)
  m.bannedChess = E.pool.bannedChessForBonds(gd, m.disabledBonds, m.staticInactiveBonds || []);
  m.pool = new E.pool.SharedPool(gd, { banned: m.bannedChess });
  if (Array.isArray(M.factions) && M.factions.length) m.factions = [...M.factions]; // 本局敌人 (match info panel)
  if (M.bondChoice) m.bondChoice = { groups: M.bondChoice.groups.map((g) => [...g]), picked: M.bondChoice.picked };
  m.sp = null;
  const boss = M.round === gd.bossRound || M.round === gd.hiddenRound;
  let wave = opts.wave ?? null;
  if (!wave) {
    const got = previewFromKinds(E.waves, gd, { round: M.round, bossId: M.round === gd.hiddenRound ? (M.hiddenBossId || m.hiddenBossId) : M.bossId,
      kinds: (S.nextEnemies || []).map((e) => e.enemyKey), bounties: S.bounties || null, match: 'set' });
    wave = got.wave;
    if (!wave) throw new Error(`wave not rebuilt for round ${M.round} (${(S.nextEnemies || []).map((e) => e.enemyKey).join(',')})`);
  }
  if (boss) { m.wave = null; m.bossWaves = [{ players: [ps.playerId], wave }]; } else { m.wave = wave; m.bossWaves = null; }
  // seat; 自选编队 first (setDiy makes ps.gd the seat's own data view): the account's picks, config (GOAL-live §6 item 2)
  if (S.diy && Object.keys(S.diy).length) ps.setDiy(S.diy, { allowBot: true });
  ps.lp = S.lp ?? ps.lp; ps.funds = S.funds ?? 0; ps.bandId = S.bandId ?? ps.bandId;
  ps.shop.level = S.shop?.level ?? ps.shop.level;
  if (Array.isArray(S.shop?.slots)) ps.shop.slots = S.shop.slots.map((sl) => (sl ? { kind: sl.kind, id: sl.id, basePrice: sl.price, frozen: !!sl.frozen, sold: !!sl.sold } : null));
  if (S.shop) { ps.shop.frozen = !!S.shop.frozen; ps.shop.freeRefreshes = S.shop.freeRefreshes || 0; }
  ps.deployCapBonus = (S.deployCap ?? gd.deployCap) - gd.deployCap; ps.deployCapMin = 0;
  ps.layers = Object.fromEntries((S.bonds || []).filter((b) => b.layers).map((b) => [b.bondId, b.layers]));
  ps.bounties = (S.bounties || []).map((b, i) => ({ id: `bounty:s${i}`, card: { ...b.card }, roundsLeft: b.roundsLeft ?? 1, chooser: ps.playerId }));
  if (S.command !== undefined) ps.command = S.command ? structuredClone(S.command) : null;
  if (S.commandMarks) ps.commandMarks = structuredClone(S.commandMarks);
  if (S.pagerSeat !== undefined && S.pagerSeat !== null) ps._botPager = S.pagerSeat;
  if (S.botSelf && S.botSelf.focus) ps._botSelf = { round: S.botSelf.round, plan: Object.freeze({ bonds: Object.freeze([Object.freeze({ id: S.botSelf.focus, priority: 5 })]) }) };
  if (Array.isArray(S.seenUids)) ps._botSeenUids = new Set(S.seenUids); // track: the pieces held when our last END ran
  let maxUid = 0;
  const itemPiece = (id) => ({ uid: ++maxUid + 1e6, kind: 'item', id, poolCopies: 0, boughtRound: m.round, meta: {} });
  const piece = (p) => {
    if (!p) return null;
    maxUid = Math.max(maxUid, p.uid);
    // boughtRound (track: the round we got it; unknown = an earlier round, never "fresh")
    const q = { uid: p.uid, kind: p.kind, id: p.id, poolCopies: 0, boughtRound: Number.isInteger(p.boughtRound) ? p.boughtRound : 0, meta: {} };
    if (p.kind === 'chess') q.items = [];
    if (p.kind === 'token') { q.count = 1; q.ownerUid = p.ownerUid; }
    if (p.dir) q.dir = p.dir;
    q._items = p.items || [];
    return q;
  };
  ps.hand = Array.from({ length: ps.hand.length }, (_, i) => piece(S.hand?.[i] || null));
  const tempLen = Math.max(ps.temp.length, (S.temp || []).length);
  ps.temp = Array.from({ length: tempLen }, (_, i) => piece(S.temp?.[i] || null));
  ps.board = new Map();
  for (const p of S.board || []) if (p) ps.board.set(`${p.row},${p.col}`, piece(p));
  for (const q of [...ps.hand, ...ps.temp, ...ps.board.values()]) if (q) { if (q.kind === 'chess') q.items = q._items.map(itemPiece); delete q._items; }
  m.uidSeq = Math.max(m.uidSeq, maxUid + 1e6 + 1000);
  if (Number.isInteger(opts.uidSeq)) m.uidSeq = opts.uidSeq; // parity: new pieces get the uids the real match gives
  ps._deployMap = null; ps._deployField = undefined;
  // solo pool: copies left = the initial stock − the copies we hold (an elite holds its merge count); shop cards take none
  for (const q of [...ps.hand, ...ps.temp, ...ps.board.values()]) {
    if (!q || q.kind !== 'chess') continue;
    const base = gd.baseIdOf(q.id);
    m.pool.take(base, gd.isGolden(q.id) ? gd.mergeCount(base) || 3 : 1, { overdraw: true });
  }
  // count bonuses written by effects: the screen's member count above what the pieces themselves count
  ps.bondCountBonus = {};
  const own = E.computeBonds(ps.gd, ps);
  for (const b of S.bonds || []) { const d = (b.count || 0) - (own[b.bondId]?.count || 0); if (d > 0) ps.bondCountBonus[b.bondId] = d; }
  installPlan(m);
  if (opts.rng) m.rngBots = opts.rng;
  ps.recompute();
  return { m, ps };
}

/** Record the seat's successful mutations (the executor's operations) while `fn` runs. */
function recording(ps, fn) {
  const ops = [];
  const where = (uid) => { const l = ps.find(uid); return l ? (l.area === 'board' ? { area: 'board', key: l.key } : { area: l.area, idx: l.idx }) : null; };
  const wrap = (name, make) => { const f = ps[name]; ps[name] = function (...a) { const before = make.pre?.(...a); const r = f.apply(this, a); if (r && r.ok !== false && !r.error) { const op = make.op(before, ...a); if (op) ops.push(op); } return r; }; return f; };
  const orig = {
    move: wrap('move', { pre: (uid) => where(uid), op: (from, uid, to, dir) => (to?.area === 'board'
      ? { op: 'move', uid, from, to: [to.row, to.col], dir: ps.find(uid)?.piece?.dir ?? dir ?? null }
      : { op: 'bench', uid, from, idx: to?.idx ?? null }) }),
    equip: wrap('equip', { pre: (item, target) => ({ itemFrom: where(item), targetFrom: where(target) }),
      op: (at, item, target, replace) => ({ op: 'equip', item, target, ...at, ...(replace != null ? { replace } : {}) }) }),
    sell: wrap('sell', { pre: (uid) => where(uid), op: (from, uid) => ({ op: 'sell', uid, from }) }),
    destroy: wrap('destroy', { pre: (uid) => where(uid), op: (from, uid) => ({ op: 'destroy', uid, from }) }),
    useCommand: wrap('useCommand', { op: (_, effectId, target) => ({ op: 'command', effectId, target }) }),
  };
  try { fn(); } finally { for (const [k, f] of Object.entries(orig)) ps[k] = f; }
  return ops;
}

export function seatResult(ps) {
  const pc = (p) => (p ? { uid: p.uid, kind: p.kind, id: p.id, items: (p.items || []).map((x) => x.id) } : null);
  return {
    board: [...ps.board.entries()].map(([k, p]) => { const [row, col] = k.split(',').map(Number); return { ...pc(p), row, col, dir: p.dir ?? 'RIGHT' }; }),
    hand: ps.hand.map(pc), temp: ps.temp.map(pc),
  };
}

/** Python's own operations (resolveTemp) on the shadow, by uid; not recorded as ours. */
function applyOps(ps, ops) {
  for (const o of ops) {
    try {
      if (o.op === 'sell') ps.sell(o.uid); else if (o.op === 'destroy') ps.destroy(o.uid);
      else if (o.op === 'stow') ps.move(o.uid, { area: o.area, idx: o.idx });
    } catch { /* a refused op: the engine refuses it live too */ }
  }
}

/** Run one request's legacy steps on a built shadow. */
export function runOp(E, { m, ps }, op, req = {}) {
  const { bot } = E;
  if (op === 'tail') applyOps(ps, req.ops || []); // Python's resolveTemp, before the recording: not ours to return
  const ops = recording(ps, () => {
    if (op === 'layout') bot.runSteps(bot.arrangeSteps(m, ps, req.final ? { final: true } : {}));
    else if (op === 'equip') bot.equipItems(m, ps);
    else if (op === 'end') {
      bot.runSteps(bot.arrangeSteps(m, ps));
      bot.equipItems(m, ps);
      if (req.legacyCommand) { if (ps.command) bot.botUseCommand(m, ps, true); } // parity only: legacy decides the battle mark
      else if (req.command?.effectId) ps.useCommand(req.command.effectId, req.command.target);
      bot.runSteps(bot.arrangeSteps(m, ps, { final: true }));
    } else if (op === 'tail') {
      if (!ps.hand.some((p) => p == null)) bot.freeHandSlot(m, ps);
    } else throw new Error(`unknown op ${op}`);
  });
  return { ok: true, ops, ...seatResult(ps) };
}

async function main() {
  const E = await loadEngine(process.argv[2]);
  const out = (o) => process.stdout.write(JSON.stringify(o) + '\n');
  for await (const line of createInterface({ input: process.stdin })) {
    if (!line.trim()) continue;
    let req;
    try { req = JSON.parse(line); } catch (e) { out({ id: null, ok: false, error: `bad JSON: ${e.message}` }); continue; }
    try {
      if (req.op === 'preview') {
        const gd = new E.GameData(E.data, soloMode(E, req.state?.config?.modeId ?? undefined).modeId);
        const got = previewFromKinds(E.waves, gd, { round: req.round, bossId: req.bossId ?? null, kinds: req.kinds || [], bounties: req.bounties || null, factions: req.factions || null, match: 'set' });
        if (!got.preview) { out({ id: req.id, ok: false, error: `no template pick matches round ${req.round} kinds ${(req.kinds || []).join(',')}` }); continue; }
        out({ id: req.id, ok: true, preview: got.preview.map(({ tag, start, ...e }) => e), approx: !!got.approx,
          distinct: got.distinct, templateId: got.templateId });
        continue;
      }
      if (req.op === 'killChance') {
        const gd = new E.GameData(E.data, soloMode(E, req.state?.config?.modeId ?? undefined).modeId);
        const p = killChance(E.bot.bountyKillChance, gd, E.tables, req.state, req.card);
        out({ id: req.id, ok: true, p });
        continue;
      }
      const t0 = performance.now();
      if (req.op === 'selfPlan') {
        const sh = buildShadow(E, req.state);
        E.bot.selfPlan(sh.m, sh.ps);
        const f = sh.ps._botSelf;
        out({ id: req.id, ok: true, botSelf: f ? { round: f.round, focus: f.plan.bonds[0].id } : null, ms: Math.round(performance.now() - t0) });
        try { sh.m.dispose(); } catch { /* */ }
        continue;
      }
      const sh = buildShadow(E, req.state, Number.isInteger(req.rngState) ? { rng: E.createRng(req.rngState) } : {});
      const r = runOp(E, sh, req.op, req);
      try { sh.m.dispose(); } catch { /* */ }
      out({ id: req.id, ...r, ms: Math.round(performance.now() - t0) });
    } catch (e) {
      out({ id: req.id ?? null, ok: false, error: String(e?.message || e).slice(0, 500) });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] || '').href) main();
