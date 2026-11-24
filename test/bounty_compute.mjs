// Bounty kill chance, computed live instead of looked up. The policy's bounty rows carry the legacy bot's
// bountyKillChance (server/match/bot.js): an exposure model of the own board (pieces, tiles, facings) against the bounty
// enemy's route, taken from the round's wave (m.wave: template routes + spawns). The board is self-tracked, enemy stats
// and the round scale are catalogue data; only the wave is not on screen. This test rebuilds the wave from what is on
// screen (stage, round, the preview) plus tables learnt on other matches, calls bountyKillChance with it, and compares
// with the value the Encoder wrote.
//   tables (even seeds):  stage|round|preview -> templateId;  stage|templateId -> routes;  stage|templateId|i -> routeIndex
//   check  (odd seeds):   every bounty card row (draft cards[*].10, personal[*].7)
//
//   node test/bounty_compute.mjs <engine-root> [--matches 200] [--seed 9700000] [--config solo:ABYSS@20]
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const root = resolve(process.argv[2] || '');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 200)), seed = Number(arg('seed', 9700000)), config = arg('config', 'solo:ABYSS@20');
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);
const { bountyKillChance } = await import(pathToFileURL(`${root}/server/match/bot.js`).href);

const sigOf = (pv) => (pv || []).map((e) => `${e.enemyKey}x${e.count}@${e.gate}${e.fly ? 'F' : ''}${e.elite ? 'E' : ''}${e.boss ? 'B' : ''}`).join(',');
// counts per key; the rebuild takes the most frequent value (route choice is random per match, so exact tables conflict)
const C = { tpl: new Map(), routes: new Map(), ri: new Map() }, conflicts = { routes: 0 };
const count = (map, k, v) => { let c = map.get(k); if (!c) map.set(k, (c = new Map())); c.set(v, (c.get(v) || 0) + 1); };
const top = (map, k) => { const c = map.get(k); if (!c) return undefined; let b, n = -1; for (const [v, x] of c) if (x > n) { n = x; b = v; } return b; };
const kills = new Map(); // enemy|round -> [sum, n]: the plain lookup table, for comparison
const games = []; // odd-seed bounty rows, checked after all tables are learnt

const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = orig.call(this, m, ps, ctx, obs, mask);
  const pub = m.publicView(), own = ps.privateView(), w = m.wave, odd = (m.seed ?? 0) % 2 === 1;
  const stage = pub.stageId, pv = own.nextEnemies || [];
  if (!odd && w) {
    count(C.tpl, `${stage}|${pub.round}`, w.templateId);
    count(C.routes, `${stage}|${w.templateId}`, JSON.stringify(w.routes));
    (w.spawns || []).forEach((s) => count(C.ri, `${stage}|${w.templateId}|${s.preview?.gate}|${s.preview?.fly ? 'F' : ''}`, s.routeIndex));
    const L = this.lay.blocks;
    if (ctx.kind === 'card' && pub.sp) pub.sp.cards.forEach((c, i) => { if (c.kind === 'bounty' && i < L.cards.count) { const k = `${c.enemyKey}|${pub.round}`, a = kills.get(k) || [0, 0]; a[0] += obs[L.cards.offset + i * L.cards.width + 10]; a[1]++; kills.set(k, a); } });
  }
  if (odd) {
    const L = this.lay.blocks, cards = [];
    if (ctx.kind === 'card' && pub.sp) pub.sp.cards.forEach((c, i) => { if (c.kind === 'bounty' && i < L.cards.count) cards.push({ c, v: obs[L.cards.offset + i * L.cards.width + 10] }); });
    (own.personalChoice?.cards || []).forEach((c, i) => { if (c.kind === 'bounty' && i < L.personal.count) cards.push({ c, v: obs[L.personal.offset + i * L.personal.width + 7] }); });
    // the screen side, frozen now: the stub is evaluated later against the final tables
    if (cards.length) games.push({ m, ps, round: pub.round, stage, pv: pv.map((e) => ({ ...e })), cards,
      board: new Map(ps.board), stageObj: m.stage, hasWave: !!w });
  }
  return r;
};
const env = new SPEnv({ envs: 4, configs: [config], seed, maxMatches: N, mode: 'teacher' });
env.reset();

for (const [k, c] of C.routes) if (c.size > 1) conflicts.routes++;
let n = 0, exact = 0, ae = 0, aeTable = 0, aeConst = 0, noWave = 0, unknown = 0; const bad = [];
let mean = 0, mn = 0; for (const [a, b] of kills.values()) { mean += a; mn += b; } mean /= Math.max(1, mn);
for (const g of games) {
  const tpl = top(C.tpl, `${g.stage}|${g.round}`), routes = tpl && top(C.routes, `${g.stage}|${tpl}`);
  let wave = null;
  if (tpl && routes) {
    const spawns = g.pv.map((e, i) => ({ enemyKey: e.enemyKey, count: e.count, routeIndex: top(C.ri, `${g.stage}|${tpl}|${e.gate}|${e.fly ? 'F' : ''}`) ?? 0,
      tag: e.boss ? 'boss' : null, mods: {}, preview: { gate: e.gate, fly: e.fly, elite: e.elite, boss: e.boss } }));
    wave = { templateId: tpl, routes: JSON.parse(routes), spawns };
  }
  for (const { c, v } of g.cards) {
    n++;
    if (!g.hasWave) noWave++;
    const kt = kills.get(`${c.enemyKey}|${g.round}`), tv = kt ? kt[0] / kt[1] : mean;
    aeTable += Math.abs(tv - v); aeConst += Math.abs(mean - v);
    if (!wave) { unknown++; ae += Math.abs(tv - v); continue; } // fall back to the table
    // stand-ins: the match as the screen knows it (gd, stage, round, rebuilt wave), the seat as self-tracked (its board)
    const sm = { gd: g.m.gd, round: g.round, stageId: g.stage, stage: g.stageObj, wave, isSolo: true, players: new Map() };
    const sps = Object.create(Object.getPrototypeOf(g.ps)); Object.assign(sps, g.ps, { board: g.board });
    let q; try { q = bountyKillChance(sm, sps, c); } catch (e) { q = NaN; if (bad.length < 5) bad.push(String(e.message).slice(0, 120)); }
    const d = Math.abs(q - v); ae += Number.isFinite(d) ? d : 1; if (d < 1e-4) exact++;
    else if (bad.length < 8) bad.push(`r${g.round} ${c.enemyKey}: engine ${v.toFixed(4)} computed ${Number(q).toFixed(4)}`);
  }
}
console.log(`config ${config}, matches ${N}: keys tpl ${C.tpl.size} routes ${C.routes.size} routeIndex ${C.ri.size}; stage|template with >1 route sets: ${conflicts.routes}`);
console.log(`stage|round -> template: ${[...C.tpl.values()].filter((c) => c.size > 1).length}/${C.tpl.size} keys ambiguous`);
console.log(`bounty rows on odd seeds ${n} (mean value ${mean.toFixed(3)}); no rebuild (table fallback) ${unknown}; engine had no wave ${noWave}`);
console.log(`  MAE constant ${(aeConst / Math.max(1, n)).toFixed(4)} | table enemy|round ${(aeTable / Math.max(1, n)).toFixed(4)} | computed ${(ae / Math.max(1, n)).toFixed(4)} (exact ${exact})`);
for (const b of bad) console.log(`  e.g. ${b}`);
