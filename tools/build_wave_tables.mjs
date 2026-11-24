// Wave tables for the live bounty kill chance (bridge/bounty.mjs). From teacher matches on the engine, per stage:
//   template: stage|round -> templateId (deterministic in the engine)
//   routes:   stage|templateId -> the template's routes (deterministic)
//   route:    stage|templateId|gate|kind -> the most frequent routeIndex (random per match in the engine; live we take the
//             most frequent one as fixed)
//
//   node tools/build_wave_tables.mjs <engine-root> [--matches 400] [--seed 9800000] [--config solo:ABYSS@20] [--out data/wave_tables.json]
import { pathToFileURL, fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
import { mkdirSync, writeFileSync } from 'node:fs';

const root = resolve(process.argv[2] || '');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 400)), seed = Number(arg('seed', 9800000)), config = arg('config', 'solo:ABYSS@20');
const out = resolve(arg('out', resolve(dirname(fileURLToPath(import.meta.url)), '../data/wave_tables.json')));
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);

const C = { template: new Map(), routes: new Map(), route: new Map() };
const count = (map, k, v) => { let c = map.get(k); if (!c) map.set(k, (c = new Map())); c.set(v, (c.get(v) || 0) + 1); };
const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ...r) {
  const w = m.wave;
  if (w) {
    count(C.template, `${m.stageId}|${m.round}`, w.templateId);
    count(C.routes, `${m.stageId}|${w.templateId}`, JSON.stringify(w.routes));
    for (const s of w.spawns || []) count(C.route, `${m.stageId}|${w.templateId}|${s.preview?.gate}|${s.preview?.fly ? 'fly' : 'ground'}`, s.routeIndex);
  }
  return orig.call(this, m, ps, ...r);
};
new SPEnv({ envs: 4, configs: [config], seed, maxMatches: N, mode: 'teacher' }).reset();

const top = (c) => { let b, n = -1; for (const [v, x] of c) if (x > n) { n = x; b = v; } return b; };
const amb = (map) => [...map.values()].filter((c) => c.size > 1).length;
const table = (map, parse = (v) => v) => Object.fromEntries([...map].sort().map(([k, c]) => [k, parse(top(c))]));
const res = {
  source: { config, matches: N, seed, engine: root },
  ambiguous: { template: amb(C.template), routes: amb(C.routes), route: amb(C.route) },
  template: table(C.template), routes: table(C.routes, JSON.parse), route: table(C.route),
};
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(res));
console.log(`wrote ${out}: template ${C.template.size} keys, routes ${C.routes.size}, route ${C.route.size}; ambiguous ${JSON.stringify(res.ambiguous)}`);
