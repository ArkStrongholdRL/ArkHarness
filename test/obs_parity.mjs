// Observation parity: at every policy decision, encode the seat twice — from the engine's own views, and from the
// ScreenState a perfect harness would produce (bridge/screen.mjs) — and count, per block field, how often they differ.
// The fields that differ are what the live policy cannot see (or what the bridge gets wrong).
//
//   node test/obs_parity.mjs <engine-root> [--matches 4] [--seed 9200000] [--config solo:ABYSS@20] [--mode random|teacher]
// <engine-root>: a Stronghold-Protocol tree with tools/rl (e.g. ~/ark-mac/sp-label).
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { fromEngine, stubs } from '../bridge/screen.mjs';
import { fillBounty } from '../bridge/bounty.mjs';
import { readFileSync } from 'node:fs';

const root = resolve(process.argv[2] || '');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 4)), seed = Number(arg('seed', 9200000)), config = arg('config', 'solo:ABYSS@20'), mode = arg('mode', 'random');
const { SPEnv, randomActions } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder, actionName } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);
const { bountyKillChance } = await import(pathToFileURL(`${root}/server/match/bot.js`).href);
// the live kill chance needs the wave tables (tools/build_wave_tables.mjs; build them on other seeds than this test's)
const tables = JSON.parse(readFileSync(new URL('../data/wave_tables.json', import.meta.url)));
let bountyRows = 0;
const close = new Map(); // obs field -> [n, sum |d|] for the fields the bridge estimates rather than reads

const where = (L, i) => {
  for (const [name, b] of Object.entries(L)) {
    const end = b.offset + b.count * b.width;
    if (i >= b.offset && i < end) { const k = i - b.offset; return b.count > 1 ? `${name}[*].${k % b.width}` : `${name}.${k}`; }
  }
  return `?${i}`;
};
const obsDiff = new Map(), maskDiff = new Map(), byKind = new Map(), errors = new Map();
let decisions = 0, clean = 0;
const cover = new Map(); // non-vacuity: how many decisions showed each situation
const bump = (map, key, ex) => { const r = map.get(key) || { n: 0, ex }; r.n++; map.set(key, r); };

const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = orig.call(this, m, ps, ctx, obs, mask);
  if (this._parityInner) return r;
  decisions++; byKind.set(ctx.kind, (byKind.get(ctx.kind) || 0) + 1);
  { const pv = ps.privateView(), pb = m.publicView(), all = [...(pv.hand || []), ...(pv.temp || []), ...(pv.board || [])].filter(Boolean);
    const c = (k, on) => { if (on) cover.set(k, (cover.get(k) || 0) + 1); };
    c('golden piece', all.some((p) => p.kind === 'chess' && m.gd.isGolden?.(p.id))); c('equipped piece', all.some((p) => p.items?.length));
    c('item in hand/temp', all.some((p) => p.kind === 'item')); c('reward offer', pv.shop?.rewardOffer); c('personal choice', pv.personalChoice);
    c('bond layers > 0', (pv.bonds || []).some((x) => x.layers > 0)); c('boss round', pb.round === pb.bossRound); c('round >= hidden round', pb.hiddenRound && pb.round >= pb.hiddenRound);
    c('frozen shop', pv.shop?.frozen); c('free refresh', pv.shop?.freeRefreshes > 0); }
  const o2 = new Float32Array(obs.length), k2 = new Uint8Array(mask.length);
  try {
    const scr = fromEngine(m, ps), st = stubs(scr, m.gd);
    this._parityInner = true;
    orig.call(this, st.m, st.ps, { ...ctx, cache: {} }, o2, k2);
    fillBounty(o2, this.lay.blocks, scr, ctx.kind, { bountyKillChance, gd: m.gd, tables, strictPreview: !!ctx.strictPreview });
    if (ctx.kind === 'card' && scr.match.sp) bountyRows += scr.match.sp.cards.filter((c) => c?.kind === 'bounty').length;
    bountyRows += (scr.self.personalChoice?.cards || []).filter((c) => c?.kind === 'bounty').length;
  } catch (e) { bump(errors, String(e?.message || e).slice(0, 160)); return r; } finally { this._parityInner = false; }
  const seen = new Set(); let same = true;
  for (let i = 0; i < obs.length; i++) {
    if (Math.abs(obs[i] - o2[i]) <= 1e-6) continue;
    same = false; const key = where(this.lay.blocks, i);
    if (key === 'cards[*].10' || key === 'personal[*].7') { const c = close.get(key) || [0, 0]; c[0]++; c[1] += Math.abs(obs[i] - o2[i]); close.set(key, c); }
    if (!seen.has(key)) { seen.add(key); bump(obsDiff, key, `${ctx.kind} r${m.round}: engine ${obs[i].toFixed(4)} vs screen ${o2[i].toFixed(4)}`); }
  }
  const mseen = new Set();
  for (let a = 0; a < mask.length; a++) if (mask[a] !== k2[a]) {
    same = false; const key = actionName(a).replace(/\d+$/, '');
    if (!mseen.has(key)) { mseen.add(key); bump(maskDiff, key, `${ctx.kind} r${m.round}: engine ${mask[a]} vs screen ${k2[a]}`); }
  }
  if (same) clean++;
  return r;
};

let a = seed >>> 0; const rng = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const env = new SPEnv({ envs: Math.min(4, N), configs: [config], seed, maxMatches: N, ...(mode === 'teacher' ? { mode: 'teacher' } : {}) });
let b = env.reset();
if (mode !== 'teacher') while (b.n > 0) b = env.step(randomActions(b, rng));

const pct = (n) => `${((100 * n) / Math.max(1, decisions)).toFixed(1)}%`;
console.log(`config ${config}, mode ${mode}, matches ${N}, seed ${seed}`);
console.log(`decisions ${decisions} (${[...byKind].map(([k, v]) => `${k} ${v}`).join(', ')}), identical obs+mask ${clean} (${pct(clean)})`);
const table = (title, map) => {
  console.log(`\n${title}: ${map.size} keys`);
  for (const [k, v] of [...map].sort((x, y) => y[1].n - x[1].n)) console.log(`  ${k.padEnd(22)} ${String(v.n).padStart(6)} ${pct(v.n).padStart(6)}  e.g. ${v.ex ?? ''}`);
};
console.log('coverage', JSON.stringify(Object.fromEntries(cover)));
table('obs fields that differ (decisions affected)', obsDiff);
table('mask actions that differ', maskDiff);
if (errors.size) table('bridge errors', errors);
console.log(`bounty card values compared: ${bountyRows}`);
for (const [k, [n, d]] of close) console.log(`estimated field ${k}: ${n} differing values, mean |d| over those ${(d / n).toFixed(4)}`);
