// Robustness: rerun the 40-day replay on 20 independently seeded clusters (seeds 1..20).
// node scripts/seeds.mjs  -> data/seeds.json (+ data/seeds.js)
import * as O from '../core/optics.mjs';
import { writeFileSync } from 'node:fs';
const display = O.CFG.seed, rows = [];
for (let s = 1; s <= 20; s++) {
  O.CFG.seed = s;
  const w = O.makeWorld(), r = {};
  for (const p of ['reactive', 'ber', 'flag', 'planner']) r[p] = +O.runPolicy(w, p).gpuHoursLost.toFixed(1);
  rows.push({ seed: s, ...r, vsFlag: +(1 - r.planner / r.flag).toFixed(4), vsReactive: +(1 - r.planner / r.reactive).toFixed(4), vsBer: +(1 - r.planner / r.ber).toFixed(4) });
  console.log(JSON.stringify(rows.at(-1)));
}
O.CFG.seed = display;
const med = a => { const b = [...a].sort((x, y) => x - y); return (b[(b.length - 1) >> 1] + b[b.length >> 1]) / 2; };
const stat = k => ({ median: +med(rows.map(r => r[k])).toFixed(4), min: Math.min(...rows.map(r => r[k])), max: Math.max(...rows.map(r => r[k])), wins: rows.filter(r => r[k] > 0).length });
const out = { seeds: rows, vsFlag: stat('vsFlag'), vsReactive: stat('vsReactive'), vsBer: stat('vsBer') };
writeFileSync(new URL('../data/seeds.json', import.meta.url), JSON.stringify(out, null, 1));
writeFileSync(new URL('../data/seeds.js', import.meta.url), 'window.SEEDS=' + JSON.stringify(out) + ';\n');
console.log(JSON.stringify({ vsFlag: out.vsFlag, vsReactive: out.vsReactive, vsBer: out.vsBer }));
