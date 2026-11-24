// ScreenState: everything the live harness hands the policy at one decision, and the bridge from it to the
// engine-shaped views (`m.publicView()`, `ps.privateView()`) that the policy's Encoder (tools/rl/spec.mjs) reads.
//
// Every field has a source:
//   read   - read from the screen (OCR / templates / pixel probes)
//   track  - maintained by the harness itself (we bought, sold, dragged, equipped, chose it)
//   config - fixed by the match we started (regulation level, mode constants)
// `fromEngine` builds a ScreenState out of a live engine seat; it stands for a perfect harness and exists only for the
// parity test. Whatever the engine knows but the ScreenState does not carry is, by construction, not available live.

export const SOURCES = Object.freeze({
  // match constants
  modeId: 'config', difficulty: 'config', lastRound: 'config', bossRound: 'config', hiddenRound: 'config',
  // public match state
  round: 'read', stageId: 'read', bossId: 'read', hiddenBossId: 'read', disabledBonds: 'read', spRound: 'read',
  factions: 'read', // match info 「本局敌人」: the special enemy types (特训敌人·飞行 …)
  bondChoice: 'read', // briefing: the two core-bond groups and which one was picked
  sp: 'read', // 机变 / 悬赏 draft cards on screen; sp.taken is our own pick (track)
  // own seat
  lp: 'read', funds: 'read', bandId: 'track', canReady: 'read', deployCap: 'read', deployCount: 'read',
  shop: 'read', rewardOffer: 'read', personalChoice: 'read',
  hand: 'track', temp: 'track', board: 'track', // pieces: id, golden (in the id), equipment, tile, facing
  bonds: 'read', // count / tier / active / layers from the bond panel
  nextEnemies: 'read', // the wave preview: key, count, fly, elite, boss, gate (no spawn time)
  // command (规程 band skill) state: which command is held, its uses / cooldown / cost, and the drafts we have to answer
  command: 'track', commandDrafts: 'track', pagerSeat: 'track', // pagerSeat: the 杰西卡 seat's first-look cache (null: not yet)
});

const pick = (o, keys) => { if (!o) return o ?? null; const r = {}; for (const k of keys) if (o[k] !== undefined) r[k] = o[k]; return r; };
const PIECE = ['uid', 'kind', 'id', 'items', 'row', 'col', 'dir', 'ownerUid']; // ownerUid: a summon's operator (track: we placed it)
const SLOT = ['kind', 'id', 'price', 'sold', 'frozen'];
const CARD = ['idx', 'kind', 'id', 'tier', 'coin', 'payout', 'rounds', 'enemyKey', 'count', 'soldOut', 'team'];
const ENEMY = ['enemyKey', 'count', 'fly', 'elite', 'boss', 'gate'];
const BOND = ['bondId', 'count', 'active', 'tier', 'layers'];
const COMMAND_STATE = ['currentCost', 'used', 'round', 'readyRound', 'totalUsed'];

/** Test only: the ScreenState a perfect harness would produce for seat `ps` of engine match `m`. */
export function fromEngine(m, ps) {
  const pub = m.publicView(), own = ps.privateView();
  const round = pub.round || 0;
  const piece = (p) => (p ? { ...pick(p, PIECE), items: (p.items || []).map((x) => (typeof x === 'string' ? x : x?.id)).filter(Boolean) } : null);
  const shop = own.shop || {};
  return {
    config: pick(pub, ['modeId', 'difficulty', 'lastRound', 'bossRound', 'hiddenRound']),
    match: {
      round, stageId: pub.stageId ?? null, bossId: pub.bossId ?? null,
      // the hidden leader is on screen only from its own round on
      hiddenBossId: pub.hiddenRound && round >= pub.hiddenRound ? pub.hiddenBossId ?? null : null,
      disabledBonds: [...(pub.disabledBonds || [])], spRound: !!pub.spRound,
      factions: Array.isArray(m.factions) ? [...m.factions] : null,
      bondChoice: pub.bondChoice ? { groups: pub.bondChoice.groups.map((g) => [...g]), picked: pub.bondChoice.picked ?? null } : null,
      sp: pub.sp ? { cards: (pub.sp.cards || []).map((c) => pick(c, CARD)), taken: { ...(pub.sp.taken || {}) } } : null,
    },
    self: {
      playerId: ps.playerId, lp: own.lp ?? 0, funds: own.funds ?? 0, bandId: own.bandId ?? null, canReady: !!own.canReady,
      deployCap: own.deployCap ?? 0, deployCount: own.deployCount ?? 0,
      shop: {
        ...pick(shop, ['level', 'maxLevel', 'upgradePrice', 'refreshPrice', 'freeRefreshes', 'frozen']),
        slots: (shop.slots || []).map((s) => (s ? pick(s, SLOT) : null)),
        rewardOffer: shop.rewardOffer ? { slots: (shop.rewardOffer.slots || []).map((s) => (s ? pick(s, SLOT) : null)) } : null,
      },
      personalChoice: own.personalChoice ? { cards: (own.personalChoice.cards || []).map((c) => pick(c, CARD)) } : null,
      hand: (own.hand || []).map(piece), temp: (own.temp || []).map(piece), board: (own.board || []).map(piece),
      bonds: (own.bonds || []).map((b) => pick(b, BOND)),
      nextEnemies: (own.nextEnemies || []).map((e) => pick(e, ENEMY)),
      command: ps.command ? {
        bandId: ps.command.bandId, currentEffectId: ps.command.currentEffectId ?? null, swapped: !!ps.command.swapped,
        records: Object.fromEntries(Object.entries(ps.command.records || {}).map(([k, r]) => [k, pick(r, COMMAND_STATE)])),
      } : null,
      commandDrafts: (ps.commandDrafts || []).map((d) => ({ id: d.id, choiceId: d.choiceId, cards: [...d.cards] })),
      pagerSeat: ps._botPager ?? null,
    },
  };
}

/** The engine-shaped views the Encoder reads, built from a ScreenState only (solo seat; `gd` = the catalogue). */
export function toViews(s, gd) {
  const S = s.self, M = s.match, C = s.config;
  // tier and golden follow from the id (operators and items both have golden `_b` forms); summoned tokens carry neither
  const piece = (p) => (p ? { ...p, tier: p.kind === 'token' ? 0 : gd.tierOf?.(p.id) || 0, golden: p.kind !== 'token' && !!gd.isGolden?.(p.id) } : null);
  const board = S.board.map(piece);
  const pub = {
    ...C, round: M.round, stageId: M.stageId, bossId: M.bossId, hiddenBossId: M.hiddenBossId,
    disabledBonds: M.disabledBonds, spRound: M.spRound, sp: M.sp, bondChoice: M.bondChoice ?? undefined,
    players: [{ playerId: S.playerId, isBot: true, alive: true, lp: S.lp, bandId: S.bandId, shopLevel: S.shop.level,
      boardCount: board.filter(Boolean).length, bonds: S.bonds }],
  };
  const own = {
    playerId: S.playerId, alive: true, lp: S.lp, funds: S.funds, bandId: S.bandId, canReady: S.canReady,
    deployCap: S.deployCap, deployCount: S.deployCount, personalChoice: S.personalChoice,
    shop: { ...S.shop }, hand: S.hand.map(piece), temp: S.temp.map(piece), board,
    bonds: S.bonds, nextEnemies: S.nextEnemies,
  };
  return { pub, own };
}

/** Minimal stand-ins for the engine match and seat, enough for Encoder.encode on a solo seat. */
export function stubs(s, gd) {
  const { pub, own } = toViews(s, gd);
  const m = { gd, players: new Map(), publicView: () => pub, bandTaken: () => false };
  const ps = { playerId: s.self.playerId, privateView: () => own };
  return { m, ps };
}
