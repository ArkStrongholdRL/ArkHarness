// Acceptance tool for bridge/sidecar.mjs (GOAL-live C): solo:ABYSS@20 matches with legacy prep (P7's prep-start steps
// and natPlan advisor plan, botRehearsal 0); every legacy arrangeSteps / equipItems call is first replayed on a shadow
// match built from the ScreenState at that moment, then run for real, and the results are compared:
//   restore  shadow with the REAL wave (m.wave / the boss group's wave) and a clone of the real rngBots: state restoration
//   live     shadow with the wave rebuilt from the preview kinds (bridge/preview.mjs) and its own rng: live conditions
// Compared: the final board (uid, tile, facing), its Map order, equipment per uid, the hand.
//
//   node tools/layout_parity.mjs <engine-root> [--matches 24] [--seed 9200000]
import { performance } from 'node:perf_hooks';
import { fromEngine } from '../bridge/screen.mjs';
import { loadEngine, buildShadow, runOp, seatResult, installPlan } from '../bridge/sidecar.mjs';

const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 24)), seed0 = Number(arg('seed', 9200000));
const E = await loadEngine(process.argv[2]);
const { bot } = E;

const stats = {};
const examples = [];
const times = { restore: [], liveRng: [], live: [] };
const bump = (k, v) => { stats[k] = stats[k] || {}; stats[k][v] = (stats[k][v] || 0) + 1; };
const state = (ps) => {
  const S = seatResult(ps);
  // a summon placed from its stack gets a fresh engine uid: compared by its id (the harness never knows engine uids)
  const who = (p) => (p.kind === 'token' ? `T:${p.id}` : p.uid);
  // copies of one operator with the same items are interchangeable on the field: also compared by (id, items) per tile
  const what = (p) => `${p.id}[${(p.items || []).slice().sort().join('+')}]`;
  return { tiles: S.board.map((p) => `${who(p)}@${p.row},${p.col}${p.dir}`).sort().join(' '), order: S.board.map(who).join(' '),
    field: S.board.map((p) => `${what(p)}@${p.row},${p.col}${p.dir}`).sort().join(' '),
    items: S.board.concat(S.hand.filter(Boolean)).filter((p) => p.kind === 'chess').map((p) => `${p.uid}:${(p.items || []).slice().sort().join('+')}`).sort().join(' '),
    hand: S.hand.map((p) => (p ? who(p) : '-')).join(' ') };
};

function snapshot(m, ps) {
  const s = fromEngine(m, ps);
  s.self.bounties = (ps.bounties || []).map((b) => ({ card: { enemyKey: b.card?.enemyKey, count: b.card?.count }, roundsLeft: b.roundsLeft }));
  s.self.commandMarks = structuredClone(ps.commandMarks || []);
  s.self.diy = structuredClone(ps.diy || {});
  // self-tracked: the round each piece came, the pieces held at our last END
  const bought = new Map(ps.allChess().map((p) => [p.uid, p.boughtRound]));
  for (const a of ['hand', 'temp', 'board']) for (const p of s.self[a]) if (p && bought.has(p.uid)) p.boughtRound = bought.get(p.uid);
  if (ps._botSeenUids) s.self.seenUids = [...ps._botSeenUids];
  if (ps._botSelf) s.self.botSelf = { round: ps._botSelf.round, focus: ps._botSelf.plan?.bonds?.[0]?.id };
  return s;
}

function compare(kind, m, ps, s, req, run) {
  const boss = m.round === m.gd.bossRound || m.round === m.gd.hiddenRound;
  const realWave = boss ? m.bossGroupOf(ps)?.wave : m.wave;
  const shadows = {};
  for (const variant of ['restore', 'liveRng', 'live']) {
    try {
      const t0 = performance.now();
      const sh = buildShadow(E, s, variant === 'restore' ? { wave: realWave, rng: m.rngBots.clone(), uidSeq: m.uidSeq } : variant === 'liveRng' ? { rng: m.rngBots.clone(), uidSeq: m.uidSeq } : {});
      runOp(E, sh, kind, req);
      times[variant].push(performance.now() - t0);
      shadows[variant] = state(sh.ps);
      try { sh.m.dispose(); } catch { /* */ }
    } catch (e) { shadows[variant] = { error: String(e?.message || e).slice(0, 160) }; }
  }
  const r = run();
  const real = state(ps);
  for (const [variant, sh] of Object.entries(shadows)) {
    const key = `${kind}${req.final ? ':final' : ''}|${variant}|${boss ? 'boss' : 'normal'}`;
    if (sh.error) { bump(key, 'error'); if (examples.length < 12) examples.push(`${key} r${m.round}: ${sh.error}`); continue; }
    bump(key, 'n');
    if (sh.tiles === real.tiles) bump(key, 'board'); else if (false) examples.push(`${key} seed ${m.seed} r${m.round} board\n    real   ${real.tiles}\n    shadow ${sh.tiles}`);
    if (sh.order === real.order) bump(key, 'order');
    if (sh.items === real.items) bump(key, 'items'); else if (examples.length < 40 && variant === 'restore' && boss && kind === 'equip') examples.push(`${key} seed ${m.seed} r${m.round} items\n    real   ${real.items}\n    shadow ${sh.items}`);
    if (sh.hand === real.hand) bump(key, 'hand');
    if (sh.tiles === real.tiles && sh.items === real.items) bump(key, 'both');
    if (sh.field === real.field) bump(key, 'field'); else if (examples.length < 12 && variant === 'restore') examples.push(`${key} seed ${m.seed} r${m.round} field
    real   ${real.field}
    shadow ${sh.field}`);
  }
  return r;
}

const steps = bot.createBotPrepSteps({
  *arrangeSteps(m, ps, o = {}) {
    const s = snapshot(m, ps);
    let gen;
    return yield* (function* () {
      const out = compare('layout', m, ps, s, { final: !!o.final }, () => { gen = bot.arrangeSteps(m, ps, { ...o, defer: false }); return bot.runSteps(gen); });
      return out;
    })();
  },
  equipItems(m, ps) {
    const s = snapshot(m, ps);
    compare('equip', m, ps, s, {}, () => bot.equipItems(m, ps));
    if (ps.command) bot.botUseCommand(m, ps, true);
  },
});
const strategy = E.defineBotStrategy({ id: 'parity',
  *prepBegin(m, ps, o) { // P7's start of prep (env.mjs startOfPrepOnce) before the shared prep
    if (ps._act3Start !== m.round) { ps._act3Start = m.round; ps.autoPickCommands(); bot.botUseCommand(m, ps, false); bot.selfPlan(m, ps); bot.pagerUse(m, ps); }
    return yield* steps.prepBegin(m, ps, o);
  },
  prepEnd(m, ps, job) { return steps.prepEnd(m, ps, job); } });

const t0 = performance.now();
for (let k = 0; k < N; k++) {
  const sched = new E.VirtualScheduler();
  const mode = E.data.config.modes.mode_single_20;
  const m = new E.Match({ roomCode: 'PAR', mode: 'solo', difficulty: mode?.difficulty || 'ABYSS', modeId: 'mode_single_20',
    seats: [{ seat: 0, playerId: 'ai_0', name: 'p', isBot: true, connected: true }], seed: seed0 + k, data: E.data, scheduler: sched,
    botRehearsal: 0, botStrategy: strategy, log: { info() {}, debug() {}, warn() {}, error() {} }, send: () => true, broadcast: () => {},
    onEnd: () => { m._ended = true; } });
  installPlan(m);
  m.start();
  for (let g = 0; g < 5e6 && !m._ended; g++) if (!sched.runNext()) break;
  try { m.dispose(); } catch { /* */ }
}
const med = (a) => { const b = [...a].sort((x, y) => x - y); return b.length ? b[Math.floor(b.length / 2)].toFixed(0) : '-'; };
console.log(`matches ${N} seed ${seed0}, ${((performance.now() - t0) / 1000).toFixed(0)} s; shadow build+run median ms: restore ${med(times.restore)}, live ${med(times.live)}`);
console.log('field = final board by (operator id, items, tile, facing); board(uid) = the same by uid; restore: real wave + real rng; liveRng: rebuilt wave + real rng; live: rebuilt wave + own rng');
for (const [k, v] of Object.entries(stats).sort()) {
  const n = v.n || 0, p = (x) => `${x || 0}/${n} (${(100 * (x || 0) / Math.max(1, n)).toFixed(1)}%)`;
  console.log(`${k.padEnd(28)} field ${p(v.field)}  board(uid) ${p(v.board)}  equip ${p(v.items)}  hand ${p(v.hand)}${v.error ? `  errors ${v.error}` : ''}`);
}
for (const e of examples) console.log('  ' + e);
