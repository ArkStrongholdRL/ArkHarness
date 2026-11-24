// Export the engine's catalogue (server/data.js getData(), act3autochess@V3_1) for the Python harness: data/catalog.json.
// The Python side (harness/catalog.py) ports the GameData accessors it needs on top of this; the derived lists whose
// order matters (gd.bondIds, gd.bandIds()) and the mode constants are exported as the engine computes them.
// Local only (gitignored): it is the official catalogue.
//
//   node tools/export_catalog.mjs [engine-root]   (default E:/ArkHarness-engine/sp-label)
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { writeFileSync, mkdirSync } from 'node:fs';

const root = resolve(process.argv[2] || process.env.ARK_ENGINE || 'E:/ArkHarness-engine/sp-label');
const imp = (p) => import(pathToFileURL(`${root}/${p}`).href);
await imp('tools/rl/act3.mjs');
const { getData } = await imp('server/data.js');
const { GameData } = await imp('server/match/gamedata.js');
const { DEFAULT_VERSION } = await imp('shared/versions.js');
const { buildVocab } = await imp('tools/rl/spec.mjs');

const data = getData({ log: { warn() {}, error() {}, info() {} } });
const modeId = 'mode_single_20';
const gd = new GameData(data, modeId);
const raw = Object.fromEntries(Object.entries(data).filter(([k]) => k !== 'backups'));
const vocab = buildVocab(gd, 3);
const out = {
  defaultVersion: DEFAULT_VERSION, modeId,
  derived: {
    bondIds: gd.bondIds, bandIds: gd.bandIds(), modeInactiveBonds: [...gd.modeInactiveBonds],
    regulationBondDraw: gd.regulationBondDraw, isSolo: gd.isSolo, difficulty: gd.difficulty,
    lastRound: gd.lastRound, bossRound: gd.bossRound, hiddenRound: gd.hiddenRound, maxShopLevel: gd.maxShopLevel,
    benchSize: gd.benchSize, tempSize: gd.tempSize, deployCap: gd.deployCap, defaultStartLp: gd.defaultStartLp,
    spRounds: gd.spRounds(), vocabHash: vocab.hash, dataId: vocab.dataId,
  },
  raw,
};
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
const file = new URL('../data/catalog.json', import.meta.url);
writeFileSync(file, JSON.stringify(out));
console.log(`wrote data/catalog.json: ${Object.keys(raw).join(', ')}; dataId ${vocab.dataId}`);
