// B6 / B7 offline loop, engine side: runs solo:ABYSS@20 matches in the engine (SPEnv policy mode, P7's training options:
// natPlan order, rehearsal 0) and hands every policy decision to the Python driver (harness/sim.py) over stdio.
// Default (B6): legacy still runs inside the engine (prep start steps, END layout); this checks the decision chain only.
// --legacy-off (B7): the engine runs no legacy step for the policy seat; Python decides them (harness/rules.py, the Node
// sidecar bridge/sidecar.mjs) and the engine executes them as intents: the start of every prep (commands, pager; the
// command draft pick stays the engine's random draw, as in legacy) and the END flow (layout, equipment, battle marks,
// temp). The operations name Python's own uids; they are mapped to engine pieces by position, as the live clicks are.
//
// Protocol (one JSON object per line; the engine side reads stdin synchronously, so hooks can call Python mid-step):
//   out  {"t":"batch","rows":[{seed, round, screen, ctx, full, live, obs, mask}...]}   obs/mask: base64 f32 / u8 (env's own)
//        screen = bridge/screen.mjs fromEngine(m, ps) at encode time; ctx = the encoder ctx;
//        full = engine values the screen lacks: {t:[spawn time per preview row], hiddenBossId}
//        live = bounty kill chances computed live (bridge/bounty.mjs killChance from the screen): {card:{i:p}, personal:{i:p}}
//   in   {"actions":[a0, a1, ...]}  (one per row)
//   out  {"t":"prepStart"|"end", seed, round, screen}       (--legacy-off) screen + self.bounties / commandMarks / diy
//   in   {"ops":[...], "pos":{uid: {area, idx} | {area:'board', row, col}}}   ops as bridge/sidecar.mjs returns them
//   out  {"t":"done","results":[{seed, win, survivalRound, lp}...], "stats":{...}}
//
//   node tools/sim_server.mjs <engine-root> --seed 9200000 --matches 50 [--envs 8] [--rebuild] [--legacy-off]
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { readFileSync, readSync, writeSync } from 'node:fs';
import { fromEngine } from '../bridge/screen.mjs';
import { killChance } from '../bridge/bounty.mjs';
import { previewFromKinds } from '../bridge/preview.mjs';
import { loadEngine, buildShadow, runOp, seatResult } from '../bridge/sidecar.mjs';

const root = resolve(process.argv[2] || 'E:/ArkHarness-engine/sp-label');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 8)), seed = Number(arg('seed', 9200000)), envs = Number(arg('envs', 8));
// --blank count,gate,boss,elite,fly: preview fields the live screen lacks, blanked before the live kill chance (as live)
const BLANK = String(arg('blank', '')).split(',').filter(Boolean);
const blanked = (scr) => (BLANK.length ? { ...scr, self: { ...scr.self, nextEnemies: scr.self.nextEnemies.map((e) => { const x = { ...e }; for (const k of BLANK) x[k] = k === 'count' ? 0 : k === 'gate' ? null : false; return x; }) } } : scr);
const NAT_PLAN = { assign: 'order', order: ['ursusShip', 'lethaniaShip'], bonus: 0, sec: 0, phiCount: 0, phiLayers: 0 };
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);
const { bountyKillChance } = await import(pathToFileURL(`${root}/server/match/bot.js`).href);
const waves = await import(pathToFileURL(`${root}/server/match/waves.js`).href);
// --rebuild: the preview the live harness has: rebuilt from the screen's enemy kinds + our own bounty cards (bridge/preview.mjs)
const REBUILD = process.argv.includes('--rebuild');
const LEGACY_OFF = process.argv.includes('--legacy-off');
// ablations: hand one half back to the engine's own legacy (--legacy-end: END flow; --legacy-start: start of prep)
const LEGACY_END = process.argv.includes('--legacy-end'), LEGACY_START = process.argv.includes('--legacy-start');
const tables = JSON.parse(readFileSync(new URL('../data/wave_tables.json', import.meta.url)));

// ---- synchronous line I/O ----
const rbuf = Buffer.alloc(1 << 16);
let pending = '';
function readLine() {
  for (;;) {
    const i = pending.indexOf('\n');
    if (i >= 0) { const l = pending.slice(0, i); pending = pending.slice(i + 1); return l; }
    let n;
    try { n = readSync(0, rbuf, 0, rbuf.length, null); } catch (e) { if (e.code === 'EAGAIN') continue; if (e.code === 'EOF') return null; throw e; }
    if (n === 0) return null;
    pending += rbuf.toString('utf8', 0, n);
  }
}
const send = (o) => { const s = JSON.stringify(o) + '\n'; let off = 0; const b = Buffer.from(s); while (off < b.length) off += writeSync(1, b, off); };
const call = (o) => { send(o); const l = readLine(); if (l == null) process.exit(1); return JSON.parse(l); };

let captured = [];
const rebuildStats = { n: 0, miss: 0, ambiguous: 0 };
const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = orig.call(this, m, ps, ctx, obs, mask);
  const screen = fromEngine(m, ps), own = ps.privateView(), pub = m.publicView();
  let times = (own.nextEnemies || []).map((e) => e.t || 0);
  if (REBUILD && (own.nextEnemies || []).length) {
    const got = previewFromKinds(waves, m.gd, { round: m.round, bossId: m.round === m.gd.hiddenRound ? m.hiddenBossId : m.bossId,
      kinds: [...new Set(own.nextEnemies.map((e) => e.enemyKey))], match: 'set',
      bounties: (ps.bounties || []).map((b) => ({ card: { enemyKey: b.card?.enemyKey, count: b.card?.count } })) });
    rebuildStats.n++; if (!got.preview) rebuildStats.miss++; else if (got.distinct > 1) rebuildStats.ambiguous++;
    if (got.preview) {
      screen.self.nextEnemies = got.preview.map((e) => ({ enemyKey: e.enemyKey, count: e.count, fly: e.fly, elite: e.elite, boss: e.boss, gate: e.gate }));
      times = got.preview.map((e) => e.t || 0);
    } else { screen.self.nextEnemies = own.nextEnemies.map((e) => ({ enemyKey: e.enemyKey, count: 0, fly: false, elite: false, boss: false, gate: null })); times = times.map(() => 0); }
  }
  const live = { card: {}, personal: {} };
  if (ctx.kind === 'card') (screen.match.sp?.cards || []).forEach((c, i) => { if (c?.kind === 'bounty') live.card[i] = killChance(bountyKillChance, m.gd, tables, blanked(screen), c); });
  (screen.self.personalChoice?.cards || []).forEach((c, i) => { if (c?.kind === 'bounty') live.personal[i] = killChance(bountyKillChance, m.gd, tables, blanked(screen), c); });
  captured.push({ seed: m.seed, round: m.round, screen, live,
    ctx: { kind: ctx.kind, actions: ctx.actions || 0, freezes: ctx.freezes || 0, lp0: ctx.lp0 || 0,
      newUids: [...(ctx.newUids || [])], soldBases: [...(ctx.soldBases || [])] },
    full: { t: times, hiddenBossId: pub.hiddenBossId ?? null } });
  return r;
};

const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
const env = new SPEnv({ envs: Math.min(envs, N), configs: ['solo:ABYSS@20'], seed, maxMatches: N, mode: 'policy', rehearsal: 0, natPlan: NAT_PLAN, check: true });

// ---- B7: legacy off ----
const fieldOf = (board) => board.map((p) => `${p.id}[${(p.items || []).slice().sort().join('+')}]@${p.row},${p.col}${p.dir}`).sort().join(' ');
// --end-compare (with --legacy-end): before the engine's own legacy END, the sidecar's END on the exact engine state
const END_COMPARE = process.argv.includes('--end-compare');
const E = END_COMPARE ? await loadEngine(root) : null;
const { LAYOUT_ONLY } = END_COMPARE ? await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href) : {};
const off = { unmappedBy: {}, unmappedSample: [], prepStarts: 0, ends: 0, ops: 0, applied: 0, refused: 0, unmapped: 0, readyFallback: 0, tempLeft: 0, byOp: {} };
function snap(m, ps) {
  const s = fromEngine(m, ps);
  s.self.bounties = (ps.bounties || []).map((b) => ({ card: { enemyKey: b.card?.enemyKey, count: b.card?.count }, roundsLeft: b.roundsLeft }));
  s.self.commandMarks = structuredClone(ps.commandMarks || []);
  s.self.diy = structuredClone(ps.diy || {}); // the account's 自选 picks: config
  s.rngState = m.rngBots.state(); // ablation only (--b7-engine-rng): the engine's tie-break stream for the layout
  return s;
}
/** Execute Python's operations: its uids → engine pieces by the position Python believes each piece is at. */
function execute(ps, ops, pos) {
  const map = new Map();
  for (const [uid, l] of Object.entries(pos || {})) {
    const p = l.area === 'board' ? ps.board.get(`${l.row},${l.col}`) : l.area === 'hand' ? ps.hand[l.idx] : l.area === 'temp' ? ps.temp[l.idx] : null;
    if (p) map.set(Number(uid), p.uid);
  }
  // a piece the flow itself created (a summon sent to the hand, a merge) is unknown at the start: it is found where the
  // operation says it is at that moment (as the live executor clicks a position), and learned for the later operations
  const at = (l) => (!l ? null : l.area === 'board' ? ps.board.get(l.key ?? `${l.row},${l.col}`) : l.area === 'hand' ? ps.hand[l.idx] : l.area === 'temp' ? ps.temp[l.idx] : null);
  let curFrom = null;
  const eng0 = (u) => {
    if (u == null) return null;
    if (map.has(Number(u))) return map.get(Number(u));
    const p = at(curFrom);
    if (p) { map.set(Number(u), p.uid); off.learned = (off.learned || 0) + 1; return p.uid; }
    return null;
  };
  let curOp = null;
  const eng = (u) => { const e = eng0(u); if (e == null && curOp) { off.unmappedBy[curOp.op] = (off.unmappedBy[curOp.op] || 0) + 1; if (off.unmappedSample.length < 12) off.unmappedSample.push({ op: curOp, known: Object.prototype.hasOwnProperty.call(pos || {}, String(u)) }); } return e; };
  for (const op of ops || []) {
    off.ops++; curOp = op; curFrom = op.from ?? null;
    const tally = (ok) => { const k = off.byOp[op.op] ||= { n: 0, ok: 0 }; k.n++; if (ok) { k.ok++; off.applied++; } else off.refused++; };
    let r = null;
    try {
      if (op.op === 'move') { const u = eng(op.uid); if (u == null) { off.unmapped++; continue; } r = ps.move(u, { area: 'board', row: op.to[0], col: op.to[1] }, op.dir || 'RIGHT'); }
      else if (op.op === 'bench') { const u = eng(op.uid); if (u == null) { off.unmapped++; continue; } let idx = Number.isInteger(op.idx) && !ps.hand[op.idx] ? op.idx : -1; if (idx < 0) for (let i = ps.hand.length - 1; i >= 0; i--) if (!ps.hand[i]) { idx = i; break; } r = ps.move(u, { area: 'hand', idx: Math.max(0, idx) }); }
      else if (op.op === 'equip') { curFrom = op.itemFrom ?? null; const a = eng(op.item); curFrom = op.targetFrom ?? null; const b = eng(op.target); if (a == null || b == null) { off.unmapped++; continue; } r = ps.equip(a, b, op.replace != null ? eng(op.replace) : null); }
      else if (op.op === 'stow') { const u = eng(op.uid); if (u == null) { off.unmapped++; continue; } r = ps.move(u, { area: op.area, idx: op.idx }); }
      else if (op.op === 'sell' || op.op === 'destroy') { const u = eng(op.uid); if (u == null) { off.unmapped++; continue; } r = ps[op.op](u); }
      else if (op.op === 'command') { const t = op.target && op.target.uid != null ? { ...op.target, uid: eng(op.target.uid) } : op.target; r = ps.useCommand(op.effectId, t); }
      else { tally(false); continue; }
    } catch (e) { r = { error: String(e?.message || e) }; }
    tally(!!r && r.ok !== false && !r.error);
  }
}
if (LEGACY_OFF) {
  if (!LEGACY_END) env.readyHook = () => {}; // the env then takes the policy seat's END steps from _hookSteps (replaced below)
  const steps = {
    *prepBegin(m, ps) { // the END flow, decided by Python
      if (!ps.alive || ps.ready) return null;
      off.ends++;
      const r = call({ t: 'end', seed: m.seed, round: m.round, screen: snap(m, ps) });
      execute(ps, r.ops, r.pos);
      if (r.expect) { // translation check: the engine's board after the ops vs the board the sidecar's flow ended with
        const got = fieldOf(seatResult(ps).board), want = fieldOf(r.expect);
        off.endBoards = (off.endBoards || 0) + 1;
        if (got === want) off.endSame = (off.endSame || 0) + 1;
        else if ((off.endDiffs ||= []).length < 6) off.endDiffs.push({ round: m.round, engine: got, sidecar: want });
      }
      return null;
    },
    *prepEnd(m, ps) {
      if (!ps.alive || ps.ready) return;
      m.autoPickPersonalChoice(ps, 'bot');
      if (!ps.tempEmpty) { off.tempLeft++; ps.resolveTemp(); }
      if (!ps.setReady(true)?.ok && !ps.ready) off.readyFallback++;
    },
  };
  if (!LEGACY_END) env._hookSteps = (slot) => (slot.hookSteps ||= { layout: steps, legacy: steps });
  if (LEGACY_END && END_COMPARE) {
    env.readyHook = () => {};
    const cmp = {
      *prepBegin(m, ps, o) {
        let want = null;
        try {
          const s = snap(m, ps);
          const bought = new Map(ps.allChess().map((p) => [p.uid, p.boughtRound]));
          for (const a of ['hand', 'temp', 'board']) for (const p of s.self[a]) if (p && bought.has(p.uid)) p.boughtRound = bought.get(p.uid);
          if (ps._botSeenUids) s.self.seenUids = [...ps._botSeenUids];
          if (ps._botSelf) s.self.botSelf = { round: ps._botSelf.round, focus: ps._botSelf.plan?.bonds?.[0]?.id };
          const boss = m.round === m.gd.bossRound || m.round === m.gd.hiddenRound;
          const sh = buildShadow(E, s, { wave: boss ? m.bossGroupOf(ps)?.wave : m.wave, rng: m.rngBots.clone(), uidSeq: m.uidSeq });
          const r = runOp(E, sh, 'end', { legacyCommand: true });
          want = fieldOf(r.board);
          try { sh.m.dispose(); } catch { /* */ }
        } catch (e) { off.cmpErrors = (off.cmpErrors || 0) + 1; if ((off.cmpErrSample ||= []).length < 3) off.cmpErrSample.push(String(e?.message || e).slice(0, 200)); }
        const res = yield* LAYOUT_ONLY.prepBegin(m, ps, o);
        if (want != null) {
          const got = fieldOf(seatResult(ps).board);
          off.cmpN = (off.cmpN || 0) + 1;
          if (got === want) off.cmpSame = (off.cmpSame || 0) + 1;
          else if ((off.cmpDiffs ||= []).length < 6) off.cmpDiffs.push({ round: m.round, engine: got, sidecar: want });
        }
        return res;
      },
      prepEnd(m, ps, job) { return LAYOUT_ONLY.prepEnd(m, ps, job); },
    };
    env._hookSteps = (slot) => (slot.hookSteps ||= { layout: cmp, legacy: cmp });
  }
  const wrap = (slot) => {
    const m = slot?.m;
    if (!m || m._b7 || LEGACY_START) return slot;
    m._b7 = true;
    const ep = m.enterPrep;
    m.enterPrep = function (...args) {
      const seats = [...slot.rl].map((pid) => this.players.get(pid)).filter((ps) => ps && ps.alive);
      for (const ps of seats) ps._act3Start = this.round; // env.mjs startOfPrepOnce: legacy's start-of-prep steps are skipped
      const res = ep.apply(this, args);
      for (const ps of seats) {
        off.prepStarts++;
        ps.autoPickCommands(); // the command draft pick: a random draw in legacy too (rngMeta)
        const r = call({ t: 'prepStart', seed: this.seed, round: this.round, screen: snap(this, ps) });
        execute(ps, r.ops, r.pos);
      }
      return res;
    };
    return slot;
  };
  for (const slot of env.slots || []) wrap(slot); // the constructor already opened the first matches
  const open = env._open.bind(env);
  env._open = (...a) => wrap(open(...a));
}

captured = [];
let b = env.reset();
while (b.n > 0) {
  if (captured.length !== b.n) throw new Error(`captured ${captured.length} encodes for ${b.n} decisions`);
  const rows = captured.map((c, k) => ({ ...c, obs: b64(b.obs.subarray(k * b.D, (k + 1) * b.D)), mask: b64(b.mask.subarray(k * b.A, (k + 1) * b.A)) }));
  captured = [];
  const { actions } = call({ t: 'batch', rows });
  b = env.step(Int32Array.from(actions));
}
const results = env.results.map((r) => { const s = r.seats.find((x) => x.rl) || r.seats[0];
  return { seed: r.seed, win: s.win ? 1 : 0, survivalRound: s.survivalRound, lp: s.lp, band: s.band, failed: !!r.failed }; });
send({ t: 'done', results, stats: { ...env.stats, errorSamples: env.errorSamples, rebuild: rebuildStats, legacyOff: LEGACY_OFF ? off : null } });
