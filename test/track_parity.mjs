// Self-tracking parity: can the harness keep its holdings by bookkeeping its own actions?
//
// Policy: a merge-seeking rule with some noise — take rewards; buy copies of owned operators first, then the cheapest
// operator; sell the cheapest non-copy operator when the hand is nearly full; level up or refresh now and then; first
// legal card / band. One Holdings tracker (bridge/tracker.mjs) per seat is seeded from the engine at the first decision
// of each prep (= one screen read at prep start) and from then on advanced by our own actions only. A mismatch is counted
// once and the tracker is reseeded (so one miss does not cascade).
// Across a battle it reports what changed between our END and the next prep, and which code removed operators
// (a hook on PlayerState.sell, attributed to the first legacy bot function on the stack).
//
//   node test/track_parity.mjs <engine-root> [--matches 24] [--seed 9500000] [--config solo:ABYSS@20]
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { Holdings, minus } from '../bridge/tracker.mjs';

const root = resolve(process.argv[2] || '');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 24)), seed = Number(arg('seed', 9500000)), config = arg('config', 'solo:ABYSS@20');
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder, ACT, SHOP, REWARDS, actionName } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);
const { PlayerState } = await import(pathToFileURL(`${root}/server/match/PlayerState.js`).href);

const bump = (m, k, n = 1) => m.set(k, (m.get(k) || 0) + n);
const stat = { within: new Map(), across: new Map(), sells: new Map(), acrossN: 0, examples: [] };

// who sells: our policy (through the env) or a legacy step
let sellPhase = 'policy step';
const os = PlayerState.prototype.sell;
PlayerState.prototype.sell = function (uid, ...rest) {
  const area = this.find?.(uid)?.area ?? '?';
  const frames = String(new Error().stack).split('\n').slice(2);
  const bot = frames.find((f) => /bot\w*\.js/.test(f));
  const who = bot ? (bot.match(/at (?:Object\.|Generator\.)?([\w$.]+) /)?.[1] || 'bot') : frames.some((f) => f.includes('env.mjs')) ? 'env' : 'other';
  bump(stat.sells, `${sellPhase} | ${who} | from ${area}`);
  return os.call(this, uid, ...rest);
};

const seats = new Map(); // ps -> { round, trk, last }
const kindOf = (gd, id) => (gd.chess?.(id) ? 'operator' : gd.item?.(id) ? 'item' : 'token');
const same = (a, b) => !minus(a.pieces, b.pieces).length && !minus(b.pieces, a.pieces).length
  && !minus(a.equipped, b.equipped).length && !minus(b.equipped, a.equipped).length;
let current = null;

const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = orig.call(this, m, ps, ctx, obs, mask);
  const gd = m.gd, own = ps.privateView(), now = Holdings.fromView(gd, own);
  let s = seats.get(ps);
  if (s && s.last?.action != null) {
    const a = s.last.action, name = actionName(a).replace(/\d+$/, '');
    if (s.round === m.round && name !== 'end') {
      if (a >= ACT.BUY && a < ACT.SELL) { const x = s.last.shop[a - ACT.BUY]; if (x) s.trk.acquire(x.kind, x.id); }
      else if (a >= ACT.SELL && a < ACT.REWARD) { const p = s.last.owned[a - ACT.SELL]; if (p) s.trk.remove(p.id); } // slot -> id: assumed known
      else if (a >= ACT.REWARD && a < ACT.PERSONAL) { const x = s.last.reward[a - ACT.REWARD]; if (x) s.trk.acquire(x.kind, x.id); }
      const ok = same(s.trk.key(), now.key());
      bump(stat.within, `${name}: ${ok ? 'ok' : 'MISS'}`);
      if (!ok) {
        const w = s.trk.key(), e = now.key();
        if (stat.examples.length < 10) stat.examples.push(`${name} r${m.round}: tracker-only [${minus(w.pieces, e.pieces)}] engine-only [${minus(e.pieces, w.pieces)}]; equip tracker-only [${minus(w.equipped, e.equipped)}] engine-only [${minus(e.equipped, w.equipped)}]`);
        s.trk = Holdings.fromView(gd, own); // reseed
      }
    } else if (s.round !== m.round) {
      stat.acrossN++;
      const before = s.last.key, after = now.key();
      for (const x of minus(after.pieces, before.pieces)) bump(stat.across, `gained ${kindOf(gd, x)}`);
      for (const x of minus(before.pieces, after.pieces)) bump(stat.across, `lost ${kindOf(gd, x)}`);
      if (minus(after.equipped, before.equipped).length || minus(before.equipped, after.equipped).length) bump(stat.across, 'equipment changed');
    }
  }
  if (!s || s.round !== m.round) { s = { round: m.round, trk: Holdings.fromView(gd, own) }; seats.set(ps, s); }
  s.last = { key: now.key(), owned: Encoder.owned(own), shop: own.shop?.slots || [], reward: own.shop?.rewardOffer?.slots || [], action: null,
    funds: own.funds || 0, handFree: (own.hand || []).filter((p) => !p).length, upgradePrice: own.shop?.upgradePrice ?? 99 };
  current = s;
  return r;
};

let a = seed >>> 0; const rng = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const pick = (b, k) => {
  const A = b.A, mk = (x) => b.mask[k * A + x] === 1, legal = []; for (let x = 0; x < A; x++) if (mk(x)) legal.push(x);
  if (!legal.length) return 0;
  const L = current?.last;
  if (!L || !legal.includes(ACT.END)) return legal[0]; // card / band decision
  for (let i = 0; i < REWARDS; i++) if (mk(ACT.REWARD + i)) return ACT.REWARD + i;
  if (mk(ACT.LEVEL) && L.funds >= L.upgradePrice + 4 && rng() < 0.5) return ACT.LEVEL;
  const have = new Map(); for (const p of L.owned) if (p && p.kind === 'chess') have.set(p.id, (have.get(p.id) || 0) + 1);
  if (L.handFree <= 2) { // sell the lowest-tier operator without a copy
    let best = -1, bt = Infinity;
    L.owned.forEach((p, i) => { if (p && p.kind === 'chess' && mk(ACT.SELL + i) && (have.get(p.id) || 0) < 2 && (p.tier || 9) < bt) { bt = p.tier || 9; best = ACT.SELL + i; } });
    if (best >= 0 && rng() < 0.7) return best;
  }
  let best = -1, bp = Infinity;
  for (let i = 0; i < SHOP; i++) if (mk(ACT.BUY + i)) {
    const x = L.shop[i]; if (!x) continue;
    const score = (x.kind === 'chess' && have.has(x.id) ? -100 : 0) + (x.kind === 'item' ? 50 : 0) + (x.price || 0);
    if (score < bp) { bp = score; best = ACT.BUY + i; }
  }
  if (best >= 0 && (bp < 0 || rng() < 0.8)) return best;
  if (mk(ACT.REFRESH) && rng() < 0.4) return ACT.REFRESH;
  return ACT.END;
};

const env = new SPEnv({ envs: 1, configs: [config], seed, maxMatches: N });
let b = env.reset();
while (b.n > 0) {
  const acts = new Int32Array(b.n);
  for (let k = 0; k < b.n; k++) { acts[k] = pick(b, k); if (current?.last) current.last.action = acts[k]; }
  sellPhase = acts[0] === ACT.END ? 'after END' : 'policy step';
  b = env.step(acts);
}

console.log(`config ${config}, matches ${N}, seed ${seed}`);
console.log('\nwithin a prep (tracker seeded once at prep start, then advanced by our actions only):');
for (const [k, v] of [...stat.within].sort()) console.log(`  ${k.padEnd(16)} ${v}`);
for (const e of stat.examples) console.log(`    e.g. ${e}`);
console.log(`\nacross a battle (${stat.acrossN} round boundaries; our END -> next prep's first decision):`);
for (const [k, v] of [...stat.across].sort((x, y) => y[1] - x[1])) console.log(`  ${k.padEnd(20)} ${v}`);
console.log('\nsells (env.step phase | caller | area):');
for (const [k, v] of [...stat.sells].sort((x, y) => y[1] - x[1])) console.log(`  ${k.padEnd(48)} ${v}`);
