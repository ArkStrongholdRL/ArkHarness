// Acceptance tool for bridge/preview.mjs: in teacher matches (solo:ABYSS@20), at the first decision of every prep, rebuild
// the preview from (round, leader, the preview's enemy kinds only) and compare with the engine's nextEnemies, entry by
// entry and in order: enemyKey, count, gate, fly, elite, boss, t. Bounty-card spawns (source 'bounty') are compared apart.
//
//   node tools/preview_parity.mjs <engine-root> [--matches 48] [--seed 9300000]
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { previewFromKinds } from '../bridge/preview.mjs';

const root = resolve(process.argv[2] || 'E:/ArkHarness-engine/sp-label');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 48)), seed = Number(arg('seed', 9300000));
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);
const waves = await import(pathToFileURL(`${root}/server/match/waves.js`).href);

const done = new Set();
const st = { preps: 0, exact: 0, noMatch: 0, ambiguous: 0, ambiguousButFirstRight: 0, withBounty: 0, boss: 0, bossExact: 0, fieldMiss: {} };
const examples = [];
const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = orig.call(this, m, ps, ctx, obs, mask);
  const key = `${m.seed}|${m.round}`;
  if (ctx.kind !== 'prep' || done.has(key)) return r;
  done.add(key);
  const all = ps.privateView().nextEnemies || [];
  const withB = arg('bounties', 'on') === 'on';
  const wave = withB ? all : all.filter((e) => e.source !== 'bounty');
  if (!wave.length) return r;
  st.preps++;
  if (wave.some((e) => e.source === 'bounty')) st.withBounty++;
  const boss = m.round === m.gd.bossRound || m.round === m.gd.hiddenRound;
  if (boss) st.boss++;
  const bossId = m.round === m.gd.hiddenRound ? m.hiddenBossId : m.bossId;
  const match = arg('match', 'multiset');
  const names = wave.map((e) => e.enemyKey);
  const kinds = match === 'set' ? [...new Set(names)] : names;
  const got = previewFromKinds(waves, m.gd, { round: m.round, bossId, kinds, factions: arg('factions', 'on') === 'on' ? [...m.factions] : null, match,
    bounties: withB ? (ps.bounties || []).map((b) => ({ card: { enemyKey: b.card?.enemyKey, count: b.card?.count } })) : null });
  if (!got.preview) { st.noMatch++; if (examples.length < 8) examples.push(`no match r${m.round} ${got.templateId}: ${wave.map((e) => e.enemyKey).join(',')}`); return r; }
  if (got.distinct > 1) st.ambiguous++;
  const F = ['enemyKey', 'count', 'gate', 'fly', 'elite', 'boss', 't', 'source'];
  let ok = got.preview.length === wave.length;
  for (let i = 0; ok && i < wave.length; i++) for (const f of F) {
    const a = f === 't' ? Math.round(wave[i][f] * 1000) : wave[i][f], b = f === 't' ? Math.round(got.preview[i][f] * 1000) : got.preview[i][f];
    if (a !== b) { ok = false; st.fieldMiss[f] = (st.fieldMiss[f] || 0) + 1; if (examples.length < 8) examples.push(`r${m.round} #${i} ${f}: engine ${a} rebuilt ${b}`); break; }
  }
  if (ok) { st.exact++; if (boss) st.bossExact++; if (got.distinct > 1) st.ambiguousButFirstRight++; }
  return r;
};

const env = new SPEnv({ envs: Math.min(8, N), configs: ['solo:ABYSS@20'], seed, maxMatches: N, mode: 'teacher', enforce: true, rehearsal: 0,
  natPlan: { assign: 'order', order: ['ursusShip', 'lethaniaShip'], bonus: 0, sec: 0, phiCount: 0, phiLayers: 0 } });
let b = env.reset();
while (b.n > 0) b = env.step([]);
console.log(`matches ${N} seed ${seed}`);
console.log(JSON.stringify(st));
console.log(`exact full preview (kind, count, gate, fly, elite, boss, t, order): ${st.exact}/${st.preps} (${(100 * st.exact / Math.max(1, st.preps)).toFixed(1)}%); boss rounds ${st.bossExact}/${st.boss}`);
for (const e of examples) console.log('  ' + e);
