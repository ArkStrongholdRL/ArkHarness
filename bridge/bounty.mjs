// Live bounty kill chance: the legacy bot's bountyKillChance (server/match/bot.js), the value the policy saw in training
// (draft rows cards[*].10, personal rows personal[*].7), computed from what the harness has:
//   board          self-tracked pieces with tile and facing (ScreenState self.board)
//   enemy, round   catalogue stats and the round's scale (gd)
//   wave           rebuilt from the screen's preview + data/wave_tables.json: template by stage|round, its routes, and
//                  each preview entry's route as the most frequent one for its stage|template|gate|kind
// The engine picks the route per match at random, so the rebuilt wave is an estimate. In test/obs_parity.mjs (tables
// from other seeds) 95-97% of bounty values come out identical to the engine's; an enemy|round lookup is off by 0.062.

/** The round's wave rebuilt from the preview, or null when the tables do not know the stage / round. */
export function rebuildWave(tables, stageId, round, preview) {
  const tpl = tables.template[`${stageId}|${round}`];
  const routes = tpl && tables.routes[`${stageId}|${tpl}`];
  if (!tpl || !routes) return null;
  const spawns = (preview || []).map((e) => ({
    enemyKey: e.enemyKey, count: e.count, tag: e.boss ? 'boss' : null, mods: {},
    routeIndex: tables.route[`${stageId}|${tpl}|${e.gate}|${e.fly ? 'fly' : 'ground'}`] ?? 0,
    preview: { gate: e.gate, fly: !!e.fly, elite: !!e.elite, boss: !!e.boss },
  }));
  return { templateId: tpl, routes, spawns };
}

/**
 * Kill chance of a bounty card for the seat described by ScreenState `s`.
 * `bountyKillChance` is the engine's function (import it from server/match/bot.js); `gd` the catalogue.
 */
export function killChance(bountyKillChance, gd, tables, s, card) {
  const wave = rebuildWave(tables, s.match.stageId, s.match.round, s.self.nextEnemies);
  const board = new Map();
  for (const p of s.self.board || []) if (p && Number.isInteger(p.row) && Number.isInteger(p.col)) board.set(`${p.row},${p.col}`, p);
  const m = { gd, round: s.match.round, stageId: s.match.stageId, stage: gd.stage?.(s.match.stageId) ?? null, wave,
    isSolo: true, players: new Map() };
  const ps = { gd, board, playerId: s.self.playerId };
  try { const p = bountyKillChance(m, ps, card); return Number.isFinite(p) ? p : 0; } catch { return 0; }
}

/**
 * Overwrite the bounty kill-chance fields of an encoded observation (the Encoder computes them from the engine match,
 * which the live side does not have). `L` = encoder.lay.blocks; `strictPreview` as in the policy's training env.
 */
export function fillBounty(obs, L, s, kind, { bountyKillChance, gd, tables, strictPreview = false }) {
  const kc = (c) => killChance(bountyKillChance, gd, tables, s, c);
  const sp = s.match.sp;
  if (kind === 'card' && sp && !strictPreview) sp.cards.forEach((c, i) => {
    if (i < L.cards.count) obs[L.cards.offset + i * L.cards.width + 10] = c?.kind === 'bounty' ? kc(c) : 0;
  });
  (s.self.personalChoice?.cards || []).forEach((c, i) => { // the personal row is not gated by strictPreview in the encoder
    if (i < L.personal.count) obs[L.personal.offset + i * L.personal.width + 7] = c?.kind === 'bounty' ? kc(c) : 0;
  });
}
