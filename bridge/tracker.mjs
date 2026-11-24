// Self-maintained holdings: what the harness predicts we own after its own actions, from the catalogue alone.
// Holdings are a multiset of piece ids (operators, items, tokens) plus the equipment on them; slot positions are the
// screen layer's job and are not modelled here. Merge rules follow the engine (server/match/player/acquire.js):
//   operator: the copy that completes gd.mergeCount(base) normal copies merges at once into gd.goldenIdOf(base);
//             the consumed copies' equipment goes back to the hand (and may merge in turn);
//   item:     a mergeable EQUIP item: rec.upgradeNum identical copies (loose or worn) merge into rec.upgradeChessId.
// Not modelled (the parity test measures them): item fusion recipes, deferred merges, pool stock limits, effects
// triggered by a gain (特质 / band grants).

export class Holdings {
  constructor(gd, pieces = []) {
    this.gd = gd;
    const itemId = (x) => (typeof x === 'string' ? x : x?.id); // engine views carry equipment as objects
    this.pieces = pieces.map((p) => ({ kind: p.kind, id: p.id, items: (p.items || []).map(itemId).filter(Boolean) }));
  }

  static fromView(gd, own) {
    return new Holdings(gd, [...(own.hand || []), ...(own.temp || []), ...(own.board || [])].filter(Boolean));
  }

  /** Sorted multiset keys: pieces, then equipment carried. */
  key() {
    return {
      pieces: this.pieces.map((p) => p.id).sort(),
      equipped: this.pieces.flatMap((p) => p.items).sort(),
    };
  }

  acquire(kind, id) {
    if (kind === 'chess') return this._chess(id);
    if (kind === 'item') return this._item(id);
    this.pieces.push({ kind, id, items: [] });
  }

  remove(id) {
    const i = this.pieces.findIndex((p) => p.id === id);
    if (i < 0) return false;
    const [p] = this.pieces.splice(i, 1);
    for (const it of p.items) this._item(it); // [ASSUMED] equipment returns to the hand on a sale (measured by the test)
    return true;
  }

  _chess(id) {
    const gd = this.gd;
    if (gd.isGolden(id)) { this.pieces.push({ kind: 'chess', id, items: [] }); return; }
    const base = gd.baseIdOf(id), need = gd.mergeCount(base), golden = gd.goldenIdOf(base);
    const same = this.pieces.filter((p) => p.kind === 'chess' && !gd.isGolden(p.id) && gd.baseIdOf(p.id) === base);
    if (need > 1 && golden && same.length + 1 >= need) {
      const used = same.slice(0, need - 1), back = used.flatMap((p) => p.items);
      this.pieces = this.pieces.filter((p) => !used.includes(p));
      this.pieces.push({ kind: 'chess', id: golden, items: [] });
      for (const it of back) this._item(it);
      return;
    }
    this.pieces.push({ kind: 'chess', id, items: [] });
  }

  _item(id) {
    // mergeable as in the engine's _itemMergeable: a normal EQUIP item marked mergeable, with an upgrade target
    const gd = this.gd, rec = gd.item?.(id);
    const golden = rec && !rec.isGolden && rec.itemType === 'EQUIP' && rec.mergeable ? rec.upgradeChessId || rec.goldenId : null;
    const need = Number.isInteger(rec?.upgradeNum) ? rec.upgradeNum : gd.itemMergeCount;
    // copies count loose and equipped alike; an equipped copy is taken off its operator, the golden item goes to the hand
    const loose = this.pieces.filter((p) => p.kind === 'item' && p.id === id);
    const worn = this.pieces.filter((p) => p.items.includes(id));
    if (golden && golden !== id && need > 1 && loose.length + worn.reduce((n, p) => n + p.items.filter((x) => x === id).length, 0) + 1 >= need) {
      let left = need - 1;
      const used = loose.slice(0, left); left -= used.length;
      this.pieces = this.pieces.filter((p) => !used.includes(p));
      for (const p of worn) while (left > 0 && p.items.includes(id)) { p.items.splice(p.items.indexOf(id), 1); left--; }
      return this._item(golden); // a golden item may complete a merge in turn
    }
    this.pieces.push({ kind: 'item', id, items: [] });
  }
}

/** Multiset difference a - b (sorted arrays of ids). */
export function minus(a, b) {
  const n = new Map(); for (const x of b) n.set(x, (n.get(x) || 0) + 1);
  const out = []; for (const x of a) { const k = n.get(x) || 0; if (k) n.set(x, k - 1); else out.push(x); }
  return out;
}
