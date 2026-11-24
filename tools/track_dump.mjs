// Acceptance tool for B5 (harness/track.py): record, for every prep of the teacher seat, the engine's ScreenState at each
// decision (before the first action, before every executed action, and at END) and the executed actions in between.
//   { seed, round, pid, ev: [ {s: ScreenState, legal: [action ids], stash: hidden temp count}, {a: action, n: name}, ... ] }
// Every executed legacy call of the teacher's prep goes through PlayerState (buy / sell / refresh / freeze / levelUp /
// pickReward); a refused call (error) is not an action. The state after an action is the next {s} event.
//
//   node tools/track_dump.mjs <engine-root> [--matches 24] [--seed 9400000] [--out logs/track_dump.jsonl]
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
import { createWriteStream, mkdirSync } from 'node:fs';
import { fromEngine } from '../bridge/screen.mjs';

const root = resolve(process.argv[2] || 'E:/ArkHarness-engine/sp-label');
const arg = (k, d) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : d; };
const N = Number(arg('matches', 24)), seed = Number(arg('seed', 9400000)), out = arg('out', 'logs/track_dump.jsonl');
export const NAT_PLAN = { assign: 'order', order: ['ursusShip', 'lethaniaShip'], bonus: 0, sec: 0, phiCount: 0, phiLayers: 0 };
const { SPEnv } = await import(pathToFileURL(`${root}/tools/rl/env.mjs`).href);
const { Encoder, ACT } = await import(pathToFileURL(`${root}/tools/rl/spec.mjs`).href);
const { PlayerState } = await import(pathToFileURL(`${root}/server/match/PlayerState.js`).href);

mkdirSync(dirname(resolve(out)), { recursive: true });
const ws = createWriteStream(out);
let cur = null, nPreps = 0, nActs = 0;
const legalOf = (mask) => { const l = []; for (let a = 0; a < mask.length; a++) if (mask[a]) l.push(a); return l; };

// states: every encode of the prep (each is the state before a call, incl. refused ones and END)
const origEncode = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ctx, obs, mask) {
  const r = origEncode.call(this, m, ps, ctx, obs, mask);
  if (cur && ctx.kind === 'prep' && ps.playerId === cur.pid) {
    cur.ev.push({ s: fromEngine(m, ps), legal: legalOf(mask), stash: ps.privateView().stashCount ?? 0 });
  }
  return r;
};

// executed actions of the teacher seat
const codeOf = {
  buy: (ps, a) => ACT.BUY + a[0],
  refresh: () => ACT.REFRESH,
  levelUp: () => ACT.LEVEL,
  freeze: () => ACT.FREEZE,
  pickReward: (ps, a) => ACT.REWARD + a[0],
  sell: (ps, a) => { const owned = Encoder.owned(ps.privateView()); const j = owned.findIndex((p) => p && p.uid === a[0]); return j >= 0 ? ACT.SELL + j : -1; },
};
for (const name of Object.keys(codeOf)) {
  const orig = PlayerState.prototype[name];
  PlayerState.prototype[name] = function (...args) {
    const code = cur && this.playerId === cur.pid ? codeOf[name](this, args) : null;
    const r = orig.apply(this, args);
    if (code != null && !(r && r.error)) { cur.ev.push({ a: code, n: name, legalCall: code >= 0 }); nActs++; }
    return r;
  };
}

const origTeach = SPEnv.prototype._teach;
SPEnv.prototype._teach = function (slot, ps, d) {
  if (d.kind !== 'prep') return origTeach.call(this, slot, ps, d);
  cur = { seed: slot.m.seed, round: slot.m.round, pid: ps.playerId, ev: [] };
  try { return origTeach.call(this, slot, ps, d); }
  finally { ws.write(JSON.stringify(cur) + '\n'); nPreps++; cur = null; }
};

const env = new SPEnv({ envs: Math.min(4, N), configs: ['solo:ABYSS@20'], seed, maxMatches: N, mode: 'teacher', enforce: true,
  rehearsal: 0, natPlan: NAT_PLAN, recorder: { sample() {}, term() {} } });
let b = env.reset();
while (b.n > 0) b = env.step([]);
ws.end();
console.log(`matches ${N} seed ${seed}: ${nPreps} preps, ${nActs} executed actions written to ${out}`);
