import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as O from '../core/optics.mjs';

const link400 = { id: 0, kind: 'host', host: 10, speed: 400, cwps: 400e9 / 5140 };

test('poisson tail matches a brute-force sum', () => {
  for (const lam of [1e-3, 0.05, 0.7, 2.5, 6, 20]) {
    let lp = -lam, s = 0;
    for (let k = 1; k <= 300; k++) { lp += Math.log(lam) - Math.log(k); if (k >= 16) s += Math.exp(lp); }
    const got = O.poisTail(lam);
    assert.ok(Math.abs(got - s) <= 1e-9 * Math.max(s, 1e-300) + 1e-300, `lam=${lam}: ${got} vs ${s}`);
  }
});

test('codeword distribution sums to one', () => {
  for (const st of [{ lam: 1e-3, q: 1e-9, L: 1.5 }, { lam: 0.3, q: 1e-6, L: 2.5 }, { lam: 2, q: 1e-4, L: 3.2 }]) {
    const { P, ptail } = O.binProbs(st);
    const tot = P.reduce((a, b) => a + b, 0) + ptail;
    assert.ok(Math.abs(tot - 1) < 1e-9, `sum=${tot}`);
  }
});

test('two-component fit recovers lambda and the uncorrectable probability (burst-dominated tail)', () => {
  const st = { lam: 1.2e-3, q: 2e-7, L: 2.4 };
  const { P, ptail } = O.binProbs(st);
  const N = link400.cwps * 86400;
  const r = O.mulberry32(42);
  const h = { N, counts: P.map(p => O.poisson(N * p, r)), ucw: 0 };
  const f = O.fitHistogram(h);
  assert.ok(Math.abs(f.lam / st.lam - 1) < 0.05, `lam ${f.lam}`);
  const truth = ptail, est = O.poisTail(f.lam) + f.b;
  assert.ok(Math.abs(Math.log10(est) - Math.log10(truth)) < 1.0, `est ${est} truth ${truth}`);
});

test('margin falls monotonically as random errors grow (noise-type degradation)', () => {
  let prev = Infinity;
  for (const lam of [0.05, 0.2, 0.5, 1, 1.5, 2, 3]) {
    const m = O.marginOf(lam, 1e-18, link400);
    assert.ok(m < prev, `lam=${lam} m=${m}`); prev = m;
  }
  assert.ok(O.marginOf(3, 1e-18, link400) < 0, 'lambda=3 is past the flap line at 1 UCW/s');
});

test('Theil-Sen is exact on a line and ignores one outlier', () => {
  const xs = [0, 1, 2, 3, 4, 5, 6, 7], ys = xs.map(x => 3 - 0.4 * x);
  const e = O.theilSen(xs, ys);
  assert.ok(Math.abs(e.slope + 0.4) < 1e-12 && Math.abs(e.icpt - 3) < 1e-12);
  ys[3] = 50;
  const r = O.theilSen(xs, ys);
  assert.ok(Math.abs(r.slope + 0.4) < 1e-9, `slope ${r.slope}`);
});

test('trend projects the burst driver to the analytic flap-line crossing', () => {
  const thresh = Math.log10(O.CFG.flapUcwPerSec / link400.cwps); // log10 per-codeword flap line
  const lat = {};
  for (let t = 0; t < 10; t++) { const x2 = thresh - 4 + 0.25 * t; lat[t] = { x1: -3, x2, m: O.marginOf(1e-3, 10 ** x2, link400) }; }
  const tr = O.trend(lat, 10, link400);
  // x2 at day 9 is thresh - 1.75 -> reaches the line 7 days after day 9, i.e. 6 days after the start of day 10
  assert.ok(Math.abs(tr.tauHat - 6) <= 0.26, `tauHat ${tr.tauHat}`);
});

test('Hungarian assignment equals brute force on random matrices', () => {
  const r = O.mulberry32(7);
  const perms = (arr, k) => k === 0 ? [[]] : arr.flatMap((x, i) => perms([...arr.slice(0, i), ...arr.slice(i + 1)], k - 1).map(p => [x, ...p]));
  for (let trial = 0; trial < 25; trial++) {
    const n = 4, m = 6, C = Array.from({ length: n }, () => Array.from({ length: m }, () => Math.round(r() * 100)));
    const a = O.hungarian(C), got = a.reduce((s, j, i) => s + C[i][j], 0);
    assert.equal(new Set(a).size, n);
    let best = Infinity;
    for (const p of perms([0, 1, 2, 3, 4, 5], n)) best = Math.min(best, p.reduce((s, j, i) => s + C[i][j], 0));
    assert.equal(got, best);
  }
});

test('planner waits for a cheap window when risk is low, swaps now when risk is high, and respects capacity', () => {
  const host = { id: 1, kind: 'host', host: 70, speed: 400, name: 'x', cwps: link400.cwps }; // moe-sft-v7: windows on odd days
  const mk = (rate) => ({ cdf: t => 1 - Math.exp(-rate * t) });
  const low = O.planSwaps([{ link: host, pred: mk(0.03) }], 90, 4, 14);
  assert.equal(low.length, 1); assert.equal(low[0].window, true); assert.equal(low[0].day, 91);
  const high = O.planSwaps([{ link: host, pred: mk(3) }], 90, 4, 14);
  assert.equal(high[0].day, 90);
  const many = Array.from({ length: 9 }, (_, k) => ({ link: { ...host, id: 10 + k }, pred: mk(3) }));
  const plan = O.planSwaps(many, 90, 2, 14);
  const perDay = {}; for (const p of plan) perDay[p.day] = (perDay[p.day] || 0) + 1;
  assert.ok(Object.values(perDay).every(c => c <= 2), JSON.stringify(perDay));
});

test('calibrator gives a monotone CDF in [0,1] that tracks its neighbours', () => {
  const cal = new O.Calibrator();
  for (let k = 0; k < 200; k++) cal.add(2 + (k % 5) * 0.1, 2.5);      // short tauHat -> flap at 2.5 d
  for (let k = 0; k < 200; k++) cal.add(30 + k * 0.05, Infinity);      // long tauHat -> no flap
  for (let k = 0; k < 2000; k++) cal.add(Infinity, k < 4 ? 5 : Infinity);
  cal.finalize();
  const near = cal.predict(2.2), far = cal.predict(35);
  let prev = -1; for (let t = 0; t <= 14; t += 0.5) { const v = near.cdf(t); assert.ok(v >= prev && v >= 0 && v <= 1); prev = v; }
  assert.ok(near.cdf(3) > 0.85 && near.cdf(2) < 0.01, `near ${near.cdf(3)}`);
  assert.ok(far.cdf(14) < 0.05, `far ${far.cdf(14)}`);
});

test('backtest ledger is internally consistent and the committed results reproduce', () => {
  const world = O.makeWorld();
  const runs = {};
  for (const p of ['reactive', 'flag', 'planner']) runs[p] = O.runPolicy(world, p);
  for (const r of Object.values(runs)) {
    const bt = r.events.filter(e => e.day >= O.CFG.backtestStart);
    assert.ok(Math.abs(bt.reduce((s, e) => s + e.cost, 0) - r.gpuHoursLost) < 1e-6);
  }
  assert.ok(runs.planner.gpuHoursLost < runs.flag.gpuHoursLost && runs.flag.gpuHoursLost < runs.reactive.gpuHoursLost);
  const committed = JSON.parse(readFileSync(new URL('../data/results.json', import.meta.url)));
  for (const p of ['reactive', 'flag', 'planner']) assert.equal(committed.policies[p].gpuHoursLost, +runs[p].gpuHoursLost.toFixed(1));
});

test('a stable but high-BER optic is not mistaken for a burst failure (regression)', () => {
  const N = link400.cwps * 86400;
  for (let day = 0; day < 30; day++) {
    const st = { lam: 0.08 + 0.04 * (day % 3), q: 1.5e-9, L: 1.5 };
    const { P } = O.binProbs(st), r = O.mulberry32(1000 + day);
    const h = { N, counts: P.map(p => O.poisson(N * p, r)), ucw: 0 };
    const m = O.tailMargin(h, link400).margin;
    assert.ok(m > 6, `day ${day}: margin ${m}`);
  }
});

test('the generator keeps flap risk at very high error rates (no lost Poisson mass)', () => {
  const { ptail } = O.binProbs({ lam: 500, q: 0, L: 1.5 });
  assert.ok(ptail > 0.999, `ptail ${ptail}`);
});
