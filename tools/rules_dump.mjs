// Acceptance tool for B4: run teacher matches in the engine and record, at every legacy call point, the state the call saw
// (fromEngine ScreenState + the diagnostic pool) and what legacy did. harness/rules.py recomputes each decision from the
// same state (tools/rules_parity.py compares). Nothing in the engine is modified: the call points are wrapped on the
// exported classes (Match, PlayerState, Encoder) and bot.js's botBandWeights is called read-only.
//
// Windows (the engine's own order, see the trace notes in REPORT-live.md):
//   group  : Match.pickBotBondGroup (INFO_CHECK, before any band / round)
//   band   : Match._applyBand for a bot seat (its weights are read before the draw takes effect)
//   autopick / cmd_start / pager : PlayerState.autoPickCommands, then the events until the next policy decision (Encoder.encode)
//   end / cmd_end / apply_plan   : the events after the last policy decision until PlayerState.setReady (the END flow)
//   resolve : the events after setReady until the next autopick / decision (legacy resolveTemp)
//
//   node tools/rules_dump.mjs <engine-root> [--matches 24] [--seed 9300000] [--out logs/rules.jsonl]
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { createWriteStream, mkdirSync } from 'node:fs';
import { fromEngine } from '../bridge/screen.mjs';

const root = resolve(process.argv[2] || 'E:/ArkHarness-engine/sp-label');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 24)), seed = Number(arg('seed', 9300000)), out = arg('out', 'logs/rules.jsonl');
// --only group,band: the cheap run (the bond group and the band draft only; no per-decision snapshots)
const ONLY = new Set(String(arg('only', '')).split(',').filter(Boolean));
const want = (k) => ONLY.size === 0 || ONLY.has(k);
const NAT_PLAN = { assign: 'order', order: ['ursusShip', 'lethaniaShip'], bonus: 0, sec: 0, phiCount: 0, phiLayers: 0 };
const imp = (p) => import(pathToFileURL(`${root}/${p}`).href);
const { SPEnv } = await imp('tools/rl/env.mjs');
const { Encoder } = await imp('tools/rl/spec.mjs');
const { Match } = await imp('server/match/Match.js');
const { PlayerState } = await imp('server/match/PlayerState.js');
const { botBandWeights } = await imp('server/match/bot.js');

mkdirSync(dirname(resolve(out)), { recursive: true });
const ws = createWriteStream(out);
const counts = {};
const emit = (m, o) => {
  counts[o.rule] = (counts[o.rule] || 0) + 1;
  ws.write(JSON.stringify({ seed: m.seed, round: m.round || 0, ...o }) + '\n');
};
const snap = (m, ps) => {
  const pool = {};
  for (const [id, e] of m.pool?.entries || []) if (e.left > 0) pool[id] = e.left; // diagnostic only (not in ScreenState)
  return { screen: fromEngine(m, ps), pool };
};

const seats = new Map(); // ps -> window state
const S = (ps) => { if (!seats.has(ps)) seats.set(ps, { win: null, evs: [], lastEnc: null, resolve: null }); return seats.get(ps); };
const addEv = (s, ev) => (s.resolve ? s.resolve.evs : s.evs).push(ev);
// the state before an END-flow event (legacy END: the layouts, equipItems + battle marks, resolveTemp, takeOffers)
const preEnd = (s, m, ps) => (s.lastEnc && !s.resolve ? snap(m, ps).screen : null);

function closeWin(ps, m) {
  const s = S(ps), w = s.win;
  if (!w) return;
  s.win = null;
  const first = w.cmd[0] ?? null;
  emit(m, { rule: 'cmd_start', screen: first ? first.pre : w.pre, pool: w.poolPre, legacy: first ? { effectId: first.effectId, target: first.target, ok: first.ok } : null, nCmd: w.cmd.length });
  const ps0 = w.pagerSnap ?? snap(m, ps);
  emit(m, { rule: 'pager', screen: ps0.screen, pool: ps0.pool, legacy: w.equips[0] ?? null, nEquip: w.equips.length });
}

function closeResolve(ps, m) {
  const s = S(ps), r = s.resolve;
  if (!r) return;
  s.resolve = null;
  emit(m, { rule: 'resolve', screen: r.screen, pool: r.pool, legacy: r.evs.filter((e) => e.type !== 'equip' && e.type !== 'cmd'), nEv: r.evs.length });
}

// ---- call points ----
const oGroup = Match.prototype.pickBotBondGroup;
if (want('group')) Match.prototype.pickBotBondGroup = function (ps) {
  const pre = this.bondChoice?.picked === null ? snap(this, ps) : null;
  const r = oGroup.call(this, ps);
  if (pre) emit(this, { rule: 'group', screen: pre.screen, pool: pre.pool, legacy: this.bondChoice.picked });
  return r;
};

const oBand = Match.prototype._applyBand;
if (want('band')) Match.prototype._applyBand = function (ps, bandId, opts) {
  if (!ps?.isBot) return oBand.call(this, ps, bandId, opts);
  const pre = snap(this, ps);
  const w = botBandWeights(this, ps);
  const r = oBand.call(this, ps, bandId, opts);
  emit(this, { rule: 'band', screen: pre.screen, pool: pre.pool, legacy: bandId, weights: { forced: w.forced, pairs: w.pairs } });
  return r;
};

let pickCapture = null;
const oPick = PlayerState.prototype.pickCommand;
if (!ONLY.size) PlayerState.prototype.pickCommand = function (id, idx) {
  const r = oPick.call(this, id, idx);
  if (pickCapture) pickCapture.push({ id, idx, ok: !!r?.ok });
  return r;
};

const oAuto = PlayerState.prototype.autoPickCommands;
if (!ONLY.size) PlayerState.prototype.autoPickCommands = function () {
  const m = this.m, s = S(this);
  closeWin(this, m);
  closeResolve(this, m);
  const pre = snap(m, this);
  pickCapture = [];
  const r = oAuto.call(this);
  const picks = pickCapture; pickCapture = null;
  emit(m, { rule: 'autopick', screen: pre.screen, pool: pre.pool, legacy: { picks, after: this.command?.currentEffectId ?? null } });
  const post = snap(m, this);
  s.win = { pre: post.screen, poolPre: post.pool, cmd: [], equips: [], pagerSnap: null };
  return r;
};

const oUse = PlayerState.prototype.useCommand;
if (!ONLY.size) PlayerState.prototype.useCommand = function (effectId, target) {
  const m = this.m, s = S(this);
  const pre = snap(m, this);
  const r = oUse.call(this, effectId, target);
  const ev = { type: 'cmd', pre: pre.screen, pool: pre.pool, effectId, target, ok: !!r?.ok, endPre: preEnd(s, m, this) };
  if (s.win) s.win.cmd.push(ev);
  addEv(s, ev);
  return r;
};

const oEquip = PlayerState.prototype.equip;
if (!ONLY.size) PlayerState.prototype.equip = function (itemUid, targetUid, replaceUid) {
  const m = this.m, s = S(this);
  if (s.win && !s.win.pagerSnap && s.win.equips.length === 0) s.win.pagerSnap = snap(m, this);
  const itemId = this.find(itemUid)?.piece?.id ?? null;
  const endPre = preEnd(s, m, this);
  const r = oEquip.call(this, itemUid, targetUid, replaceUid);
  const ev = { type: 'equip', itemUid, targetUid, itemId, ok: !!r?.ok, endPre };
  if (s.win) s.win.equips.push(ev);
  addEv(s, ev);
  return r;
};

const wrapMut = (name, type) => {
  const o = PlayerState.prototype[name];
  PlayerState.prototype[name] = function (...a) {
    const st = S(this), endPre = preEnd(st, this.m, this);
    const r = o.apply(this, a);
    const ok = !!r?.ok;
    if (name === 'move') addEv(st, { type, uid: a[0], to: a[1], dir: a[2] ?? null, ok: ok || r === undefined, endPre });
    else addEv(st, { type, uid: a[0], ok: ok || r === undefined, endPre });
    return r;
  };
};
if (!ONLY.size) { wrapMut('move', 'move'); wrapMut('sell', 'sell'); wrapMut('destroy', 'destroy'); }

const oReady = PlayerState.prototype.setReady;
if (!ONLY.size) PlayerState.prototype.setReady = function (v) {
  const m = this.m, s = S(this);
  if (!v || !s.lastEnc || s.resolve) return oReady.call(this, v);
  closeWin(this, m);
  const post = snap(m, this);
  const r = oReady.call(this, v);
  const moves = s.evs.filter((e) => e.type === 'move');
  const cmd = s.evs.find((e) => e.type === 'cmd') ?? null;
  emit(m, { rule: 'end', screen: s.lastEnc.screen, pool: s.lastEnc.pool, post: post.screen, moves, events: s.evs,
    equips: s.evs.filter((e) => e.type === 'equip'), sells: s.evs.filter((e) => e.type === 'sell').length, nEv: s.evs.length });
  emit(m, { rule: 'cmd_end', screen: cmd ? cmd.pre : s.lastEnc.screen, pool: cmd ? cmd.pool : s.lastEnc.pool,
    legacy: cmd ? { effectId: cmd.effectId, target: cmd.target, ok: cmd.ok } : null, exact: !!cmd });
  s.evs = []; s.lastEnc = null;
  s.resolve = { screen: post.screen, pool: post.pool, evs: [] };
  return r;
};

const oEnc = Encoder.prototype.encode;
if (!ONLY.size) Encoder.prototype.encode = function (m, ps, ctx, ...rest) {
  const s = S(ps);
  closeWin(ps, m);
  closeResolve(ps, m);
  s.lastEnc = snap(m, ps);
  s.evs = [];
  return oEnc.call(this, m, ps, ctx, ...rest);
};

// ---- run ----
const env = new SPEnv({ envs: Math.min(4, N), configs: ['solo:ABYSS@20'], seed, maxMatches: N, mode: 'teacher', enforce: true,
  rehearsal: 0, natPlan: NAT_PLAN, recorder: { sample() {}, term() {} } });
let b = env.reset();
while (b.n > 0) b = env.step([]);
for (const [ps] of seats) { const m = ps.m; closeWin(ps, m); closeResolve(ps, m); }
ws.end();
console.log(`matches ${N} seed ${seed}: records ${JSON.stringify(counts)} -> ${out}; env errors ${env.stats?.errors ?? 0}`);
