// Rebuild the round's full wave preview (count, gate, fly, elite, boss, spawn time t, in engine order) from what the
// screen shows: the stage-independent round template (roundCfg(round).template; the leader round: bossTemplates[bossId]
// in its solo `_s` form) plus the enemy kinds of the preview list. The only per-match unknown of a template is the
// round pick (server/match/waves.js pickRoundEntry: a factions entry of the right half, one of its normal and one of its
// elite keys); every candidate pick is expanded with the engine's own buildNormalWave / buildBossWave + previewOf, and the
// candidates whose enemy kinds equal the screen's are kept.
//
// Live OCR supplies unique enemy keys. Pass `match: 'set'` with the round and active tracked bounties:
// `previewFromKinds(waves, gd, { round, bossId, kinds, factions, bounties, match: 'set' })`
// → { preview, candidates, distinct, templateId }
//   factions  optional: the match's special enemy types as the match-info panel lists them (「本局敌人」: type ids such as
//             'FLY' or display names such as 特训敌人·飞行 / 特训敌人：飞行); a round's pick is one of them or SPECIAL
//   preview   the engine-shaped preview (null when no candidate matches); with `distinct` > 1 the matching candidates
//             disagree somewhere and the first one is returned
//   bounties  optional: the bounty cards we took that still come this round ([{ card: { enemyKey, count } }], self-tracked:
//             spDraft.js pushes { card, roundsLeft }); their spawns are added as the engine's preview does
//             (views.js nextEnemiesFor: previewOf([...wave.spawns, ...bountySpawns(...)])), and `kinds` is then the whole
//             screen list, bounty rows included. Without it the bounty rows must be left out of `kinds`.

const kindsKey = (list) => list.map((e) => e.enemyKey).sort().join('|');
const full = (e) => [e.enemyKey, e.count, e.gate, !!e.fly, !!e.elite, !!e.boss, e.t].join(',');

/** Every candidate pick of round `round` (pickRoundEntry without the draw). */
export function candidatePicks(gd, round) {
  const g = gd.factions?.generation || {};
  const max = Number.isInteger(g.maxLevelCnt) ? g.maxLevelCnt : 15;
  const half = round <= (Number.isInteger(g.firstHalfMaxRound) ? g.firstHalfMaxRound : Math.floor(max / 2));
  const keys = (list) => (Array.isArray(list) ? list : []).map((x) => (typeof x === 'string' ? x : x?.key))
    .filter((k) => typeof k === 'string' && gd.enemy(k) && k !== 'enemy_9012_acloon');
  const out = [];
  for (const e of Object.values(gd.factions?.entries || {})) {
    if (!e || typeof e.key !== 'string' || !!e.firstHalf !== half || !gd.enemy(e.key)) continue;
    const N = keys(e.N), E = keys(e.E);
    for (const normal of N.length ? N : [null]) for (const elite of E.length ? E : [null]) {
      out.push({ round, type: e.type, key: e.key, normal, elite, fly: typeof e.fly === 'boolean' ? e.fly : null, firstHalf: half });
    }
  }
  return out;
}

/** Type ids of the match's special enemy types, from ids or display names (factions.json types[].name). */
export function factionTypes(gd, names) {
  if (!Array.isArray(names) || !names.length) return null;
  const types = gd.factions?.types || {};
  const norm = (x) => String(x).replace(/[：:]/g, '·').trim();
  const out = new Set(['SPECIAL']);
  for (const n of names) {
    const t = types[n] ? n : Object.keys(types).find((k) => norm(types[k].name) === norm(n) || norm(types[k].name).endsWith(norm(n)));
    if (t) out.add(t);
  }
  return out;
}

// match 'multiset' (default): kinds lists every preview row; 'set': only which kinds appear (a screen list that shows a
// kind once, or is cut off after a few rows but names every kind)
export function previewFromKinds(waves, gd, { round, bossId = null, kinds, factions = null, bounties = null, match = 'multiset' }) {
  const allowed = factionTypes(gd, factions);
  const boss = round === gd.bossRound || (gd.hiddenRound && round === gd.hiddenRound);
  const keyOf = match === 'set' ? (list) => [...new Set(list.map((e) => e.enemyKey))].sort().join('|') : kindsKey;
  const want = keyOf(kinds.map((k) => ({ enemyKey: k })));
  const seen = new Map();
  let templateId = null;
  for (const pick of candidatePicks(gd, round)) {
    if (allowed && !allowed.has(pick.type)) continue;
    if (pick.fly === null) pick.fly = !!gd.enemy(pick.key)?.motion && gd.enemy(pick.key).motion === 'FLY';
    const picks = []; picks[round] = pick;
    const w = boss ? waves.buildBossWave(gd, null, { picks }, round, { bossId, solo: true }) : waves.buildNormalWave(gd, null, { picks }, round);
    templateId = w.templateId;
    const extra = bounties && bounties.length ? waves.bountySpawns(gd, round, w, bounties, 'self', { solo: true }) : [];
    const pv = waves.previewOf([...w.spawns, ...extra]);
    if (keyOf(pv) !== want) continue;
    const k = pv.map(full).join(';');
    if (!seen.has(k)) seen.set(k, { pv, w });
  }
  let all = [...seen.values()], approx = false;
  if (!all.length && match === 'set') {
    // no exact match: the wave kinds must be a subset of the screen's (the rest e.g. bounty rows our bookkeeping lacks);
    // the candidate covering the most screen kinds wins
    const screen = new Set(kinds);
    let best = -1;
    for (const pick of candidatePicks(gd, round)) {
      if (allowed && !allowed.has(pick.type)) continue;
      const picks = []; picks[round] = pick;
      const w = boss ? waves.buildBossWave(gd, null, { picks }, round, { bossId, solo: true }) : waves.buildNormalWave(gd, null, { picks }, round);
      const pv = waves.previewOf(w.spawns);
      const ks = new Set(pv.map((e) => e.enemyKey));
      if (![...ks].every((k) => screen.has(k))) continue;
      if (ks.size > best) { best = ks.size; all = [{ pv, w }]; } else if (ks.size === best && all.length < 2 && all[0].pv.map(full).join(';') !== pv.map(full).join(';')) all.push({ pv, w });
    }
    approx = all.length > 0;
  }
  return { preview: all[0]?.pv ?? null, wave: all[0]?.w ?? null, candidates: all.length, distinct: all.length, templateId, approx };
}
