// node tools/replay_boss.mjs logs/live/game12 14 [seed]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { buildShadow, loadEngine } from '../bridge/sidecar.mjs';

const [, , directory, roundArg, seedArg = '1'] = process.argv;
const round = Number(roundArg);
const seed = Number(seedArg);
if (!directory || !Number.isInteger(round) || round < 1 || !Number.isInteger(seed)) {
  throw new Error('Usage: node tools/replay_boss.mjs <game-log-dir> <round> [seed]');
}

const rows = readFileSync(resolve(directory, 'decisions.jsonl'), 'utf8').trim().split(/\r?\n/).map(line => JSON.parse(line));
const captured = rows.findLast(row => row.round === round && row.decision === 'boss replay input');
const historical = rows.findLast(row => row.round === round && row.result?.ready && row.result?.layout?.board);
const state = captured?.state ?? historical?.state;
if (!state) throw new Error('No verified leader layout or boss replay input was logged');

if (!captured) {
  // Older logs have pre-layout holdings and a verified final placement receipt.
  const own = state.self;
  const pieces = new Map([...own.board, ...own.hand, ...own.temp].filter(Boolean).map(piece => [piece.uid, piece]));
  const result = historical.result;
  if (result.equipment?.length || result.summons?.length) {
    throw new Error('This older battle needs a post-layout snapshot to replay equipment or summons accurately');
  }
  own.board = result.layout.board.map(({ uid, to: [row, col], dir }) => {
    const piece = pieces.get(uid);
    if (!piece) throw new Error('Final layout UID missing from starting holdings: ' + uid);
    return { ...piece, row, col, dir };
  });
  const consumed = new Set(own.board.map(piece => piece.uid));
  for (const action of [...(result.makeRoom || []), ...(result.temp || []), ...(result.tail || [])]) {
    consumed.add(action.uid);
    if (action.receipt?.after) own.funds = action.receipt.after.funds;
  }
  own.hand = own.hand.map(piece => consumed.has(piece?.uid) ? null : piece);
  own.temp = own.temp.map(piece => consumed.has(piece?.uid) ? null : piece);
  own.deployCount = own.board.length;
}

const engine = await loadEngine(process.env.ARK_ENGINE);
const { m, ps } = buildShadow(engine, state, { seed });
if (![m.gd.bossRound, m.gd.hiddenRound].includes(round)) throw new Error('The requested round is not a boss round');
const hidden = round === m.gd.hiddenRound;
const template = m.bossWaves[0].wave.templateId;
m.startFinalAssault(hidden);
const battle = m.fields[0]?.battle;
if (!battle) throw new Error('The offline boss field did not start');
try {
  for (let steps = 0; steps < 100000 && m.runner; steps++) {
    if (!m.sched.runNext()) break;
  }
  if (m.runner) throw new Error('The offline boss field did not finish');
  const result = battle.result();
  const cleared = m.bossPool.hp <= 0;
  console.log(JSON.stringify({
    round, modeId: state.config.modeId, template, bossId: hidden ? m.hiddenBossId : m.bossId,
    source: captured ? 'opening-state' : 'historical-layout', seed,
    bossMaxHp: m.bossPool.maxHp, outcome: cleared ? 'cleared' : 'failed',
    killGameSeconds: cleared ? result.time : null, endGameSeconds: result.time,
    remainingBossHp: m.bossPool.hp, endTargetLife: m.teamLp,
    leaks: result.perPlayer[ps.playerId]?.leaked?.length ?? null,
  }));
} finally {
  m.dispose?.();
}
