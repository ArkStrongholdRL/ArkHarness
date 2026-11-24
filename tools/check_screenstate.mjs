// Acceptance check for the screen reader's output: does a ScreenState JSON match the contract of bridge/screen.mjs
// (the shape fromEngine returns), and which fields are still unread? No engine needed.
//
//   node tools/check_screenstate.mjs <state.json> [more.json ...]
// Prints per file: shape errors (wrong type / missing key), and the fields that are null or listed in `unread`.
import { readFileSync } from 'node:fs';

const isInt = (v) => Number.isInteger(v), isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const isStr = (v) => typeof v === 'string' && v.length > 0, isBool = (v) => typeof v === 'boolean';
const PIECE = { ownerUid: (v) => v === undefined || isInt(v), uid: isInt, kind: (v) => ['chess', 'item', 'token'].includes(v), id: isStr, items: Array.isArray,
  row: (v) => v === undefined || isInt(v), col: (v) => v === undefined || isInt(v), dir: (v) => v === undefined || ['UP', 'RIGHT', 'DOWN', 'LEFT'].includes(v) };
const SLOT = { kind: (v) => ['chess', 'item'].includes(v), id: isStr, price: isNum, sold: isBool, frozen: (v) => v === undefined || isBool(v) };
const CARD = { kind: isStr, id: isStr };
const ENEMY = { enemyKey: isStr, count: isInt, fly: isBool, elite: isBool, boss: isBool, gate: (v) => ['upper', 'lower'].includes(v) };
const BOND = { bondId: isStr, count: isInt, active: isBool, tier: isInt, layers: isInt };

for (const file of process.argv.slice(2)) {
  const s = JSON.parse(readFileSync(file, 'utf8'));
  const errors = [], nulls = [];
  // a check that accepts undefined marks an optional key
  const field = (path, v, ok) => { if (v === null) nulls.push(path); else if (v === undefined) { if (!ok(undefined)) errors.push(`${path}: missing`); } else if (!ok(v)) errors.push(`${path}: bad value ${JSON.stringify(v).slice(0, 60)}`); };
  const rows = (path, list, spec, { holes = false } = {}) => {
    if (list === null) { nulls.push(path); return; }
    if (!Array.isArray(list)) { errors.push(`${path}: not an array`); return; }
    list.forEach((x, i) => {
      if (x === null) { if (!holes) errors.push(`${path}[${i}]: null row`); return; }
      for (const [k, ok] of Object.entries(spec)) field(`${path}[${i}].${k}`, x[k], ok);
    });
  };
  for (const top of ['config', 'match', 'self']) if (!s[top] || typeof s[top] !== 'object') errors.push(`${top}: missing`);
  if (!errors.length) {
    const { config: C, match: M, self: S } = s;
    field('config.modeId', C.modeId, isStr);
    for (const k of ['round']) field(`match.${k}`, M[k], isInt);
    field('match.stageId', M.stageId, isStr); field('match.bossId', M.bossId, isStr);
    field('match.disabledBonds', M.disabledBonds, Array.isArray); field('match.spRound', M.spRound, isBool);
    field('match.bondChoice', M.bondChoice, (v) => v === undefined || (Array.isArray(v?.groups) && v.groups.length === 2 && v.groups.every((g) => Array.isArray(g) && g.every(isStr)) && (v.picked === null || v.picked === 0 || v.picked === 1)));
    field('match.factions', M.factions, (v) => v === undefined || (Array.isArray(v) && v.every(isStr)));
    if (M.sp) rows('match.sp.cards', M.sp.cards, CARD);
    for (const k of ['lp', 'funds', 'deployCap', 'deployCount']) field(`self.${k}`, S[k], isInt);
    field('self.canReady', S.canReady, isBool);
    if (!S.shop) errors.push('self.shop: missing');
    else {
      for (const k of ['level', 'maxLevel', 'upgradePrice', 'refreshPrice', 'freeRefreshes']) field(`self.shop.${k}`, S.shop[k], isInt);
      field('self.shop.frozen', S.shop.frozen, isBool);
      rows('self.shop.slots', S.shop.slots, SLOT, { holes: true });
      if (S.shop.rewardOffer) rows('self.shop.rewardOffer.slots', S.shop.rewardOffer.slots, SLOT, { holes: true });
    }
    if (S.personalChoice) rows('self.personalChoice.cards', S.personalChoice.cards, CARD);
    for (const a of ['hand', 'temp']) rows(`self.${a}`, S[a], PIECE, { holes: true });
    rows('self.board', S.board, { ...PIECE, row: isInt, col: isInt, dir: PIECE.dir });
    rows('self.bonds', S.bonds, BOND);
    rows('self.nextEnemies', S.nextEnemies, ENEMY);
    // command state (track): null without a band skill; drafts are the pending command choices
    if (S.command !== undefined && S.command !== null) {
      field('self.command.bandId', S.command.bandId, isStr);
      field('self.command.currentEffectId', S.command.currentEffectId, (v) => v === null || isStr(v));
      field('self.command.records', S.command.records, (v) => v && typeof v === 'object' && !Array.isArray(v));
    }
    if (S.commandDrafts !== undefined) rows('self.commandDrafts', S.commandDrafts, { id: isStr, choiceId: isStr, cards: Array.isArray });
    field('self.pagerSeat', S.pagerSeat, (v) => v === undefined || v === null || isBool(v));
  }
  const unread = Array.isArray(s.unread) ? s.unread : [];
  console.log(`${file}: ${errors.length ? 'SHAPE ERRORS' : 'shape ok'}; null ${nulls.length}; unread ${unread.length}`);
  for (const e of errors.slice(0, 30)) console.log(`  ! ${e}`);
  if (nulls.length) console.log(`  null: ${nulls.slice(0, 40).join(', ')}${nulls.length > 40 ? ' …' : ''}`);
  if (unread.length) console.log(`  unread: ${unread.join(', ')}`);
}
