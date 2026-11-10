// Runs the full pipeline and writes data/results.json (+ data/results.js for file:// use).
// node scripts/run.mjs
import * as O from '../core/optics.mjs';
import { writeFileSync } from 'node:fs';

const DAY0 = Date.UTC(2026, 7, 12); // day 0 = 2026-08-12, day 90 = 2026-11-10
const world = O.makeWorld();
const runs = {};
for (const p of ['reactive', 'ber', 'flag', 'planner']) runs[p] = O.runPolicy(world, p);
const rel = O.reliability(world, runs.reactive, 7);
const td = O.today(world, runs.planner);
const D = td.day;

const summary = r => ({ gpuHoursLost: +r.gpuHoursLost.toFixed(1), flapCost: +r.flapCost.toFixed(1), swapCost: +r.swapCost.toFixed(1),
  flaps: r.flaps, precursorFlaps: r.precursorFlaps, swaps: r.swaps, healthySwaps: r.healthySwaps });
const GRID = []; for (let t = 0; t <= O.CFG.horizon + 1e-9; t += 0.25) GRID.push(+t.toFixed(2));
const planIds = new Map(td.plan.map(p => [p.link.id, p]));
const lat = (i, from) => { const out = []; for (let t = from; t < D; t++) { const v = O.telemetryOf(world, runs.planner.st, i, t); out.push({ day: t, m: +v.m.toFixed(3), x1: +v.x1.toFixed(3), x2: +v.x2.toFixed(3), ber: v.ber }); } return out; };
const fin = x => Number.isFinite(x) ? +x.toFixed(2) : null;
const cands = td.cands.slice(0, 8).map(c => {
  const i = c.link.id, l = c.link, p = planIds.get(i), since = O.sinceOf(runs.planner.st, i, D);
  const job = l.kind === 'host' ? O.jobOfHost(l.host) : null;
  const h = O.telemetryOf(world, runs.planner.st, i, D - 1).h;
  return {
    id: i, name: l.name, peer: l.peer, kind: l.kind, speed: l.speed, job: job ? job.id : 'fabric (ECMP/ARS reroute)', gpus: job ? job.gpus : null,
    since: Number.isFinite(since) ? since : null,
    p7: +c.pred.cdf(7).toFixed(4), p14: +c.pred.cdf(14).toFixed(4), q10: fin(c.pred.q10), q50: fin(c.pred.q50), q90: fin(c.pred.q90), n: c.pred.n,
    cdf: GRID.map(t => +c.pred.cdf(t).toFixed(4)),
    trend: { s1: +c.tr.s1.toFixed(4), s2: +c.tr.s2.toFixed(4), level: +c.tr.level.toFixed(3), tauHat: fin(c.tr.tauHat),
      proj: GRID.map(t => +c.tr.at(t).toFixed(3)) },
    costs: Array.from({ length: O.CFG.horizon }, (_, k) => O.linkCost(l, D + k)),
    plan: p ? { day: p.day, at: +p.at.toFixed(3), window: p.window, eCost: +p.eCost.toFixed(2), eNone: +p.eNone.toFixed(2), saving: +p.saving.toFixed(2) } : null,
    hist: { N: h.N, counts: h.counts, ucw: h.ucw },
    lat: lat(i, Math.max(D - 30, Number.isFinite(since) ? since : 0)),
  };
});
// fleet margin distribution today
const fleetM = world.links.map((l, i) => O.telemetryOf(world, runs.planner.st, i, D - 1).m);
const bucket = [[-99, 1], [1, 2], [2, 3], [3, 4], [4, 6], [6, 99]].map(([lo, hi]) => ({ lo, hi, n: fleetM.filter(m => m >= lo && m < hi).length }));
const bt = e => ({ link: world.links[e.i].name, day: e.day, t: +e.t.toFixed(3), kind: e.kind, cost: +e.cost.toFixed(1), type: e.type });
const out = {
  generatedBy: 'scripts/run.mjs', seed: O.CFG.seed, cfg: O.CFG, day0: DAY0, today: D,
  fleet: { links: world.links.length, hostLinks: world.links.filter(l => l.kind === 'host').length, fabricLinks: world.links.filter(l => l.kind === 'fabric').length, gpus: 1024, leaves: 16, spines: 8, marginBuckets: bucket },
  jobs: O.JOBS,
  policies: Object.fromEntries(Object.entries(runs).map(([k, r]) => [k, summary(r)])),
  plannerEvents: runs.planner.events.filter(e => e.day >= O.CFG.backtestStart).map(bt),
  flagEvents: runs.flag.events.filter(e => e.day >= O.CFG.backtestStart).map(bt),
  reliability: rel,
  grid: GRID,
  candidates: cands,
  plan: td.plan.map(p => ({ id: p.link.id, day: p.day, at: +p.at.toFixed(3) })),
};
writeFileSync(new URL('../data/results.json', import.meta.url), JSON.stringify(out));
writeFileSync(new URL('../data/results.js', import.meta.url), 'window.RESULTS=' + JSON.stringify(out) + ';\n');
console.log(JSON.stringify({ policies: out.policies, rel: { coverage80: rel.coverage80, n: rel.coverageN }, cands: cands.map(c => [c.name, c.job, c.p14, c.plan && c.plan.day]), buckets: bucket }, null, 1));
