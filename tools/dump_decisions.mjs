// Acceptance tool for B1 / B2: run teacher matches in the engine and write one line per policy decision:
//   {seed, round, screen: fromEngine(m, ps), ctx, obs: base64 f32 (encoder dim + plan block), mask: base64 u8}
// obs is what the env hands the policy (P7 natPlan on, so the 59-dim plan block is appended); harness/encode compares
// its own encoding of `screen` + `ctx` against it (tools/encode_parity.py).
//
//   node tools/dump_decisions.mjs <engine-root> [--matches 24] [--seed 9300000] [--out logs/decisions.jsonl]
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { createWriteStream, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { fromEngine } from '../bridge/screen.mjs';

const root = resolve(process.argv[2] || 'E:/ArkHarness-engine/sp-label');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 24)), seed = Number(arg('seed', 9300000)), out = arg('out', 'logs/decisions.jsonl');
export const NAT_PLAN = { assign: 'order', order: ['ursusShip', 'lethaniaShip'], bonus: 0, sec: 0, phiCount: 0, phiLayers: 0 };
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);

mkdirSync(dirname(resolve(out)), { recursive: true });
const ws = createWriteStream(out);
let last = null, rows = 0, encodes = 0;
const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = orig.call(this, m, ps, ctx, obs, mask);
  encodes++;
  last = { seed: m.seed, round: m.round, screen: fromEngine(m, ps),
    ctx: { kind: ctx.kind, actions: ctx.actions || 0, freezes: ctx.freezes || 0, lp0: ctx.lp0 || 0,
      newUids: [...(ctx.newUids || [])], soldBases: [...(ctx.soldBases || [])] } };
  return r;
};
const b64 = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString('base64');
const recorder = {
  sample(kind, traj, obs, mask, action) {
    if (!last) return;
    ws.write(JSON.stringify({ ...last, action, obs: b64(Float32Array.from(obs)), mask: b64(Uint8Array.from(mask)) }) + '\n');
    rows++;
  },
  term() {},
};
const env = new SPEnv({ envs: Math.min(4, N), configs: ['solo:ABYSS@20'], seed, maxMatches: N, mode: 'teacher', enforce: true,
  rehearsal: 0, natPlan: NAT_PLAN, recorder });
let b = env.reset();
while (b.n > 0) b = env.step([]);
ws.end();
console.log(`matches ${N} seed ${seed}: ${rows} decisions written to ${out} (${encodes} encode calls)`);
