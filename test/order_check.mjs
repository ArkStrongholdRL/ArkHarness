// Order checks: board row order vs reading order, wave-preview length (cut at 8). node test/order_check.mjs <engine-root>
import { resolve } from "node:path"; const E = resolve(process.argv[2] || "");
const { SPEnv } = await import(`${E}/tools/rl/env.mjs`);
const { Encoder } = await import(`${E}/tools/rl/spec.mjs`);
const st = { n: 0, boardSorted: 0, boardN: 0, enemyLen: new Map(), handGaps: 0, tempGaps: 0 };
const orig = Encoder.prototype.encode;
Encoder.prototype.encode = function (m, ps, ...r) {
  const own = ps.privateView(); st.n++;
  const b = (own.board || []).filter(Boolean);
  if (b.length > 1) { st.boardN++; const s = [...b].sort((x, y) => x.row - y.row || x.col - y.col); if (s.every((p, i) => p === b[i])) st.boardSorted++; }
  const L = (own.nextEnemies || []).length; st.enemyLen.set(L, (st.enemyLen.get(L) || 0) + 1);
  const gap = (arr) => { const a = arr || []; const last = a.map(Boolean).lastIndexOf(true); return a.slice(0, last).some((x) => !x); };
  if (gap(own.hand)) st.handGaps++; if (gap(own.temp)) st.tempGaps++;
  return orig.call(this, m, ps, ...r);
};
const env = new SPEnv({ envs: 4, configs: ['solo:ABYSS@20'], seed: 9300000, maxMatches: 48, mode: 'teacher' }); env.reset();
console.log(JSON.stringify({ ...st, enemyLen: Object.fromEntries([...st.enemyLen].sort((a, b) => a[0] - b[0])) }));
