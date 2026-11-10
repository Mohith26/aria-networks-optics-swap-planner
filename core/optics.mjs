// Optics Swap Planner core.
// Pure ES module (runs in the browser and under `node --test`). No dependencies.
//
// Pipeline
//   1. fleet()           deterministic synthetic fabric: 1,024 host links + 512 leaf-spine links,
//                        per-link optic physics (random symbol errors + DFE-style error bursts)
//   2. telemetry()       daily RS(544,514) FEC codeword histograms (bins 0..15 + uncorrectable)
//   3. tailMargin()      fits the histogram tail and extrapolates the uncorrectable-codeword
//                        probability, reported as "FEC margin" in decades below the flap line
//   4. trend()           Theil-Sen slope of margin -> naive days-to-zero (tauHat)
//   5. Calibrator        empirical nearest-neighbour distribution of the ACTUAL time-to-flap
//                        for past link-days with a similar tauHat (calibrated, horizon-limited)
//   6. planSwaps()       expected-cost assignment of swaps to (day, technician slot) with job
//                        checkpoint/maintenance windows, solved exactly with the Hungarian method
//   7. runPolicy()       rolling day-by-day backtest: reactive vs pre-FEC BER alarm vs planner

export const CFG = {
  days: 90,              // simulated history (day 0 .. 89)
  backtestStart: 50,     // rolling backtest covers days 50 .. 89
  calStart: 10,          // calibration records start here (needs a trend window first)
  window: 10,            // trend window (days)
  horizon: 14,           // planning / follow-up horizon (days)
  K: 40,                 // nearest neighbours for the time-to-flap distribution
  maxTau: 45,            // trends that would need more than this many days to reach the flap line are ignored
  prior: 4,              // pseudo-count shrinking each neighbourhood toward the fleet base rate
  capacity: 4,           // optic swaps a technician team can do per day
  flapUcwPerSec: 1.0,    // uncorrectable codewords/s at which a link flaps with p = 63%/day
  berAlarm: 1e-5,        // incumbent rule: swap when pre-FEC BER exceeds this
  flagAt: 0.5,           // 'swap when flagged' baseline: P(flap within horizon) at which it swaps
  suddenPerDay: 0.8e-4,  // failures with no precursor (laser death, mechanical)
  swapOverhead: 20,      // GPU-hour equivalent of a technician visit + new optic + reseat risk
  seed: 71,
};

const SYMBOLS = 544, DATA_BITS = 5140, CW_BITS = 5440;

// ---------- deterministic randomness ----------
export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function hash(...xs) {
  let h = 0x811c9dc5 ^ CFG.seed;
  for (const x of xs) {
    let v = Math.floor(x * 1000) | 0;
    h = Math.imul(h ^ v, 0x01000193);
    h ^= h >>> 13; h = Math.imul(h, 0x5bd1e995); h ^= h >>> 15;
  }
  return h >>> 0;
}
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
export function poisson(mean, r) {
  if (mean <= 0) return 0;
  if (mean < 30) { const L = Math.exp(-mean); let k = 0, p = 1; do { k++; p *= r(); } while (p > L); return k - 1; }
  return Math.max(0, Math.round(mean + Math.sqrt(mean) * gauss(r)));
}

// ---------- topology and jobs ----------
export const JOBS = [
  { id: 'llama-4-pretrain', hosts: [1, 64], gpus: 512, ckptMin: 60, windowEvery: 3, windowPhase: 1, windowHour: 2, kind: 'train' },
  { id: 'moe-sft-v7', hosts: [65, 96], gpus: 256, ckptMin: 30, windowEvery: 2, windowPhase: 1, windowHour: 6, kind: 'train' },
  { id: 'rl-rollouts-b', hosts: [97, 112], gpus: 128, ckptMin: 15, windowEvery: 1, windowPhase: 0, windowHour: 4, kind: 'train' },
  { id: 'inference-decode', hosts: [113, 124], gpus: 96, kind: 'serve' },
  { id: 'spare', hosts: [125, 128], gpus: 32, kind: 'idle' },
];
const TRAIN_GPUS = JOBS.filter(j => j.kind === 'train').reduce((s, j) => s + j.gpus, 0);
export const jobOfHost = h => JOBS.find(j => h >= j.hosts[0] && h <= j.hosts[1]);

export function linkCost(link, day) {
  // Returns GPU-hours for an unplanned flap, and for a planned swap at the cheapest moment of `day`
  // (with the time-of-day offset in days when that swap happens).
  if (link.kind === 'fabric') {
    return { flap: +(0.03 * TRAIN_GPUS).toFixed(2), swap: 0.5 + CFG.swapOverhead, swapAt: 0.5, window: false };
  }
  const j = jobOfHost(link.host);
  if (j.kind === 'idle') return { flap: 0, swap: CFG.swapOverhead, swapAt: 0.5, window: false };
  if (j.kind === 'serve') return { flap: 8 * 0.5, swap: 8 * 0.25 + CFG.swapOverhead, swapAt: 0.5, window: false };
  const flap = j.gpus * (j.ckptMin / 120 + 0.25 + 10 / 60); // lost work since checkpoint + restart + NCCL timeout
  const isWindow = (day - j.windowPhase) % j.windowEvery === 0;
  if (isWindow) return { flap, swap: j.gpus * 0.02 + CFG.swapOverhead, swapAt: j.windowHour / 24, window: true };
  return { flap, swap: j.gpus * 0.35 + CFG.swapOverhead, swapAt: 0.5, window: false }; // stop at a checkpoint, swap, restart
}

export function fleet() {
  const links = [];
  for (let h = 1; h <= 128; h++) for (let nic = 0; nic < 8; nic++) {
    const leaf = nic + (h > 64 ? 8 : 0), local = (h - 1) % 64;
    links.push({ id: links.length, kind: 'host', host: h, nic, leaf, speed: 400,
      name: `leaf${leaf} / Eth1/${Math.floor(local / 2) + 1}/${(local % 2) + 1}`,
      peer: `host-${String(h).padStart(3, '0')} / nic${nic}` });
  }
  for (let leaf = 0; leaf < 16; leaf++) for (let u = 0; u < 32; u++) {
    const spine = Math.floor(u / 4);
    links.push({ id: links.length, kind: 'fabric', leaf, spine, speed: 800,
      name: `leaf${leaf} / Eth1/${33 + u}`, peer: `spine${spine} / Eth1/${leaf * 4 + (u % 4) + 1}` });
  }
  for (const l of links) { l.cwps = l.speed * 1e9 / DATA_BITS; l.base = opticParams(l, -1); }
  return links;
}

// Optic physics for one installed optic (segment). start = install day (-1 = original install).
export function opticParams(link, start) {
  const r = mulberry32(hash(link.id, start, 7));
  const p = { start, type: 'healthy',
    lam0: Math.pow(10, -4 + 1.1 * r()),     // mean random symbol errors per codeword (pre-FEC BER ~ 1e-8 .. 1e-7)
    q0: Math.pow(10, -10 + 1.5 * r()),      // fraction of codewords hit by an error burst
    L0: 1.3 + 0.4 * r(),                    // mean burst length in symbols
    onset: Infinity, rate: 0, accel: 0 };
  if (start >= 0) return p;                 // replacement optics start healthy
  const u = r();
  if (u < 0.018) {                          // gradual noise growth: dirty connector, laser aging
    p.type = 'noise'; p.onset = -40 + 140 * r(); p.rate = 0.06 + 0.12 * r(); p.accel = 0.002 * r();
  } else if (u < 0.036) {                   // burst growth: CDR / DFE error propagation, reflections
    p.type = 'burst'; p.onset = -15 + 110 * r(); p.rate = 0.10 + 0.22 * r(); p.accel = 0.004 * r();
  } else if (u < 0.066) {                   // stable but high BER (long or lossy path): never fails
    p.type = 'stable-high'; p.lam0 = Math.pow(10, -1.7 + 0.8 * r());
  }
  return p;
}

// True state of an optic on a given (fractional) day, with day-to-day jitter.
export function trueState(link, p, day) {
  const j = mulberry32(hash(link.id, p.start, day, 11));
  const dt = Math.max(0, day - p.onset), grow = p.rate * dt + p.accel * dt * dt;
  let lam = p.lam0, q = p.q0, L = p.L0;
  if (p.type === 'noise') lam = p.lam0 * Math.pow(10, grow);
  if (p.type === 'burst') { q = p.q0 * Math.pow(10, grow); L = Math.min(3.4, p.L0 + 0.09 * dt); }
  const jit = Math.pow(10, 0.12 * gauss(j));
  return { lam: lam * jit, q: Math.min(0.5, q * jit), L };
}

// Codeword symbol-error distribution: mixture of Poisson(lam) and geometric bursts of mean L.
export function binProbs({ lam, q, L }) {
  const P = new Array(16).fill(0);
  let lp = -lam; // log pmf at k = 0
  const pois = [];
  for (let k = 0; k <= 80; k++) { if (k > 0) lp += Math.log(lam) - Math.log(k); pois.push(Math.exp(lp)); }
  const g = k => k === 0 ? 0 : Math.pow(1 - 1 / L, k - 1) / L;
  for (let k = 0; k < 16; k++) P[k] = (1 - q) * pois[k] + q * g(k);
  const ptail = (1 - q) * poisTail(lam) + q * Math.pow(1 - 1 / L, 15);
  return { P, ptail };
}

export function flapHazardPerDay(link, st) {
  const u = link.cwps * binProbs(st).ptail; // true uncorrectable codewords per second
  return 1 - Math.exp(-u / CFG.flapUcwPerSec);
}

// One day of FEC counters: counts per bin 0..15 and uncorrectable count.
export function histogram(link, p, day) {
  const st = trueState(link, p, day);
  const { P, ptail } = binProbs(st);
  const N = link.cwps * 86400;
  const r = mulberry32(hash(link.id, p.start, day, 13));
  const counts = P.map(pk => poisson(N * pk, r));
  return { N, counts, ucw: poisson(N * ptail, r) };
}

export function preFecBer(h) {
  let errs = 0; h.counts.forEach((c, k) => { errs += k * c; }); errs += 17 * h.ucw;
  return errs / (h.N * CW_BITS);
}

// ---------- 3. FEC tail margin ----------
// Two-component fit of one day's codeword histogram:
//   random errors: Poisson(lam), lam from the zero bin (lam = -ln(c0/N));
//   bursts: whatever the Poisson part cannot explain in the high bins, fitted as a geometric tail.
// The uncorrectable probability is the Poisson tail beyond 15 plus the burst tail beyond 15.
// Directly observed uncorrectables (>= 3) override the extrapolation.
const LNFACT = [0]; for (let k = 1; k <= 200; k++) LNFACT[k] = LNFACT[k - 1] + Math.log(k);
export function poisTail(lam, from = 16) {
  if (lam <= 0) return 0;
  const ll = Math.log(lam);
  if (lam > from) { // mostly beyond: complement
    let s = 0; for (let k = 0; k < from; k++) s += Math.exp(-lam + k * ll - LNFACT[k]); return Math.max(0, 1 - s);
  }
  let sum = 0;
  for (let k = from; k <= 200; k++) { const t = Math.exp(-lam + k * ll - LNFACT[k]); sum += t; if (t < sum * 1e-12) break; }
  return sum;
}
export function fitHistogram(h) {
  const N = h.N, lam = Math.max(1e-12, -Math.log(Math.max(1e-300, h.counts[0] / N)));
  const pts = [];
  let lp = -lam;
  for (let k = 1; k < 16; k++) {
    lp += Math.log(lam) - Math.log(k);
    const exp = N * Math.exp(lp), ex = h.counts[k] - exp;
    // a burst bin must beat the Poisson expectation both statistically and materially (>25%);
    // bin 1 is never used: it is dominated by random errors and by how well c0 pins lambda down
    if (k >= 2 && ex > 3 * Math.sqrt(exp) + 4 && ex > 0.25 * exp) pts.push([k, Math.log10(ex), ex]);
  }
  let burstSlope = Math.log10(1 - 1 / 1.5), bIcpt = null;
  if (pts.length >= 2) {
    const tail = pts.slice(-4);
    let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const [x, y, c] of tail) { const w = Math.min(c, 50); sw += w; sx += w * x; sy += w * y; sxx += w * x * x; sxy += w * x * y; }
    const den = sw * sxx - sx * sx, prior = burstSlope;
    let fitSlope = den > 0 ? Math.min(-0.03, Math.max(-1.5, (sw * sxy - sx * sy) / den)) : prior;
    // shrink toward the prior burst shape (mean length 1.5 symbols) unless the tail is well populated
    const nTail = tail.reduce((a, t) => a + t[2], 0), w0 = 400;
    burstSlope = (nTail * fitSlope + w0 * prior) / (nTail + w0);
    bIcpt = (sy - burstSlope * sx) / sw;
  } else if (pts.length === 1 && pts[0][0] >= 4) bIcpt = pts[0][1] - burstSlope * pts[0][0];
  // burst tail mass beyond 15 symbols, as a probability per codeword (floored at the detection limit)
  const ps = Math.log10(1 - 1 / 1.5); // detection limit: 5 burst codewords at k = 5 with the prior shape
  const floor = (5 / N) * Math.pow(10, ps * 11) / (1 - Math.pow(10, ps));
  let b = bIcpt === null ? 0 : Math.pow(10, bIcpt + burstSlope * 16) / (1 - Math.pow(10, burstSlope)) / N;
  b = Math.max(b, floor);
  return { lam, b, burstSlope, bIcpt, pts };
}
export function marginOf(lam, b, link) {
  const thresh = CFG.flapUcwPerSec / link.cwps;
  const p = poisTail(lam) + b;
  return Math.max(-3, Math.min(12, Math.log10(thresh) - Math.log10(Math.max(p, 1e-60))));
}
export function tailMargin(h, link) {
  const f = fitHistogram(h);
  let margin = marginOf(f.lam, f.b, link);
  if (h.ucw >= 3) margin = Math.max(-3, Math.min(12, Math.log10(CFG.flapUcwPerSec / link.cwps) - Math.log10(h.ucw / h.N)));
  return { margin, ...f };
}

// ---------- 4. trend ----------
export function theilSen(xs, ys) {
  const s = [];
  for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) s.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
  s.sort((a, b) => a - b);
  const slope = s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0;
  const ints = ys.map((y, i) => y - slope * xs[i]).sort((a, b) => a - b);
  const icpt = ints.length % 2 ? ints[(ints.length - 1) / 2] : (ints[ints.length / 2 - 1] + ints[ints.length / 2]) / 2;
  return { slope, icpt };
}
// lat[t] = {x1: log10 lam, x2: log10 burst tail, m: margin} for days t < d.
// Robust trends of both latent drivers are projected forward; tauHat = first time the projected
// margin reaches 0 (days from the start of day d), or Infinity if not within CFG.maxTau.
export function trend(lat, d, link, since = -Infinity) {
  const xs = [], y1 = [], y2 = [];
  for (let t = Math.max(d - CFG.window, since); t < d; t++) if (lat[t]) { xs.push(t); y1.push(lat[t].x1); y2.push(lat[t].x2); }
  const last = lat[d - 1];
  if (xs.length < 6) return { s1: 0, s2: 0, level: last ? last.m : 12, tauHat: Infinity, proj: null };
  const a = theilSen(xs, y1), c = theilSen(xs, y2);
  const s1 = a.slope > 0.02 ? a.slope : 0, s2 = c.slope > 0.08 ? c.slope : 0;
  const x1 = a.icpt + a.slope * (d - 1), x2 = c.icpt + c.slope * (d - 1);
  const at = tau => marginOf(Math.pow(10, x1 + s1 * (tau + 1)), Math.pow(10, x2 + s2 * (tau + 1)), link);
  const level = at(-1);
  let tauHat = Infinity;
  if (level <= 0) tauHat = 0.05;
  else if (s1 > 0 || s2 > 0) for (let tau = 0; tau <= CFG.maxTau; tau += 0.25) if (at(tau) <= 0) { tauHat = Math.max(0.05, tau); break; }
  return { s1, s2, x1, x2, level, tauHat, at };
}

// ---------- 5. calibrated time-to-flap ----------
export class Calibrator {
  constructor() { this.recs = []; this.baseN = 0; this.baseFlaps = []; }
  add(tauHat, ttf) { // ttf = actual time to flap from the forecast moment (Infinity if none within horizon)
    if (Number.isFinite(tauHat)) this.recs.push({ lt: Math.log(tauHat + 0.05), ttf });
    else { this.baseN++; if (Number.isFinite(ttf)) this.baseFlaps.push(ttf); }
  }
  finalize() { this.recs.sort((a, b) => a.lt - b.lt); return this; }
  neighbours(tauHat) {
    const lt = Math.log(tauHat + 0.05), R = this.recs, K = Math.min(CFG.K, R.length);
    let lo = 0, hi = R.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (R[m].lt < lt) lo = m + 1; else hi = m; }
    let i = lo - 1, j = lo; const out = [];
    while (out.length < K && (i >= 0 || j < R.length)) {
      if (j >= R.length || (i >= 0 && lt - R[i].lt <= R[j].lt - lt)) out.push(R[i--]); else out.push(R[j++]);
    }
    return out;
  }
  // Returns cdf(t) for t in [0, horizon], plus 10/50/90% quantiles (Infinity = beyond horizon).
  predict(tauHat) {
    if (!Number.isFinite(tauHat) || this.recs.length < CFG.K) {
      const lam = this.baseFlaps.length / Math.max(1, this.baseN * CFG.horizon);
      const cdf = t => 1 - Math.exp(-lam * t);
      return { cdf, q10: Infinity, q50: Infinity, q90: Infinity, n: this.baseN, base: true };
    }
    const nb = this.neighbours(tauHat), ts = nb.map(r => r.ttf).sort((a, b) => a - b);
    const lam = this.baseFlaps.length / Math.max(1, this.baseN * CFG.horizon), a = CFG.prior;
    const cdf = t => (ts.filter(x => x <= t).length + a * (1 - Math.exp(-lam * t))) / (ts.length + a);
    const q = p => ts[Math.min(ts.length - 1, Math.floor(p * ts.length))];
    return { cdf, q10: q(0.1), q50: q(0.5), q90: q(0.9), n: ts.length, base: false };
  }
}

// ---------- 6. assignment ----------
// Hungarian algorithm (minimisation) for an n x m cost matrix with n <= m. Returns column per row.
export function hungarian(C) {
  const n = C.length, m = C[0].length, INF = 1e18;
  const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0), p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0; const minv = new Array(m + 1).fill(INF), used = new Array(m + 1).fill(false);
    do {
      used[j0] = true; const i0 = p[j0]; let delta = INF, j1 = 0;
      for (let j = 1; j <= m; j++) if (!used[j]) {
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const res = new Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]) res[p[j] - 1] = j - 1;
  return res;
}

// Expected GPU-hours for link i if swapped at absolute time s (day + offset), measured from `now`.
export function expectedSwapCost(pred, cost, now, s) {
  const F = pred.cdf(Math.max(0, s - now));
  return F * cost.flap + (1 - F) * cost.swap;
}

// cands: [{link, pred}] ; returns [{link, pred, day, at, eCost, eNone, saving}] for chosen swaps.
export function planSwaps(cands, now, capacity = CFG.capacity, horizon = CFG.horizon) {
  const rows = cands.map(c => {
    const opts = [];
    for (let d = now; d < now + horizon; d++) {
      const cost = linkCost(c.link, d), at = d + cost.swapAt;
      opts.push({ day: d, at, e: expectedSwapCost(c.pred, cost, now, at), window: cost.window, swapCost: cost.swap, flapCost: cost.flap });
    }
    const flap = linkCost(c.link, now).flap;
    return { ...c, opts, eNone: c.pred.cdf(horizon) * flap, flapCost: flap };
  }).filter(r => r.opts.some(o => o.e < r.eNone - 1e-9));
  if (!rows.length) return [];
  const slots = []; for (let d = 0; d < horizon; d++) for (let k = 0; k < capacity; k++) slots.push(d);
  const n = rows.length, m = slots.length + n, BIG = 1e12;
  const C = rows.map((r, i) => {
    const row = slots.map(d => r.opts[d].e);
    for (let k = 0; k < n; k++) row.push(k === i ? r.eNone : BIG);
    return row;
  });
  const asg = hungarian(C), out = [];
  rows.forEach((r, i) => {
    const j = asg[i];
    if (j < slots.length) {
      const o = r.opts[slots[j]];
      out.push({ link: r.link, pred: r.pred, day: o.day, at: o.at, window: o.window, eCost: o.e, eNone: r.eNone, saving: r.eNone - o.e, swapCost: o.swapCost, flapCost: r.flapCost });
    }
  });
  return out.sort((a, b) => a.at - b.at || b.saving - a.saving);
}

// ---------- 7. world + rolling policies ----------
export function makeWorld() {
  const links = fleet();
  return { links, cache: new Map() };
}
function segAt(world, st, i, day) { // current optic params for link i on day under policy state st
  const starts = st.installs[i]; let s = -1;
  for (const x of starts) if (x <= day) s = x;
  const key = i + '|' + s;
  if (!world.cache.has(key)) world.cache.set(key, s < 0 ? world.links[i].base : opticParams(world.links[i], s));
  return world.cache.get(key);
}
export function telemetryOf(world, st, i, day) {
  const p = segAt(world, st, i, day), key = 'h' + i + '|' + p.start + '|' + day;
  let v = world.cache.get(key);
  if (!v) {
    const h = histogram(world.links[i], p, day);
    const tm = tailMargin(h, world.links[i]);
    v = { h, ber: preFecBer(h), m: tm.margin, x1: Math.log10(tm.lam), x2: Math.log10(tm.b) };
    world.cache.set(key, v);
  }
  return v;
}
// Latent flap time of link i during `day` under its current optic (or Infinity).
function flapTimeOn(world, st, i, day) {
  const link = world.links[i], p = segAt(world, st, i, day);
  const r = mulberry32(hash(i, day, 17));
  if (r() < CFG.suddenPerDay) return day + r();
  const r2 = mulberry32(hash(i, p.start, day, 19));
  if (r2() < flapHazardPerDay(link, trueState(link, p, day + 0.5))) return day + r2();
  return Infinity;
}

export function sinceOf(st, i, d) { let s = -Infinity; for (const x of st.installs[i]) if (x <= d) s = x; return s; }

// policy: 'reactive' | 'ber' | 'flag' | 'planner'. Returns full ledger.
export function runPolicy(world, policy, opts = {}) {
  const capacity = opts.capacity ?? CFG.capacity, D = CFG.days, nL = world.links.length;
  const st = { installs: Array.from({ length: nL }, () => []) };
  const events = [];   // {i, day, t, kind:'flap'|'swap', cost, type, precursor}
  const margins = Array.from({ length: nL }, () => ({}));
  const forecasts = []; // {i, day, tauHat, flapAt} for calibration (reactive run only uses them)
  let cal = null;
  const pendingFlap = new Map(); // i -> true time of next flap (for calibration labels)
  const trends = {};
  for (let d = 0; d < D; d++) {
    // telemetry of yesterday is known at the start of day d
    if (d > 0) for (let i = 0; i < nL; i++) margins[i][d - 1] = telemetryOf(world, st, i, d - 1);
    const decide = d >= CFG.backtestStart && policy !== 'reactive';
    let swaps = [];
    if (d >= CFG.calStart && policy !== 'ber') {
      for (let i = 0; i < nL; i++) {
        const tr = trend(margins[i], d, world.links[i], sinceOf(st, i, d));
        trends[i] = tr;
        forecasts.push({ i, day: d, tauHat: tr.tauHat, installs: st.installs[i].length });
      }
    }
    if (decide && policy === 'ber') {
      const alarmed = [];
      for (let i = 0; i < nL; i++) {
        const y = telemetryOf(world, st, i, d - 1);
        if (y.ber > CFG.berAlarm && !(st.installs[i].length && st.installs[i].at(-1) >= d - 1)) alarmed.push([i, y.ber]);
      }
      alarmed.sort((a, b) => b[1] - a[1]);
      swaps = alarmed.slice(0, capacity).map(([i]) => { const c = linkCost(world.links[i], d); return { i, at: d + c.swapAt, cost: c.swap }; });
    }
    if (decide && policy === 'flag') { // same detector, but swap as soon as risk is flagged (no timing/cost model)
      cal = buildCalibrator(world, forecasts, events, d, st);
      const flagged = [];
      for (let i = 0; i < nL; i++) {
        const tr = trends[i];
        if (!Number.isFinite(tr.tauHat)) continue;
        const p14 = cal.predict(tr.tauHat).cdf(CFG.horizon);
        if (p14 >= CFG.flagAt) flagged.push([i, p14]);
      }
      flagged.sort((a, b) => b[1] - a[1]);
      swaps = flagged.slice(0, capacity).map(([i]) => { const c = linkCost(world.links[i], d); return { i, at: d + c.swapAt, cost: c.swap }; });
    }
    if (decide && policy === 'planner') {
      cal = buildCalibrator(world, forecasts, events, d, st);
      const cands = [];
      for (let i = 0; i < nL; i++) {
        const tr = trends[i];
        if (!Number.isFinite(tr.tauHat)) continue;
        cands.push({ link: world.links[i], pred: cal.predict(tr.tauHat), tr });
      }
      cands.sort((a, b) => b.pred.cdf(CFG.horizon) * linkCost(b.link, d).flap - a.pred.cdf(CFG.horizon) * linkCost(a.link, d).flap);
      const plan = planSwaps(cands.slice(0, 60), d, capacity);
      swaps = plan.filter(p => p.day === d).map(p => ({ i: p.link.id, at: p.at, cost: linkCost(p.link, d).swap }));
    }
    // simulate the day
    const swapAt = new Map(swaps.map(s => [s.i, s]));
    for (let i = 0; i < nL; i++) {
      const p = segAt(world, st, i, d);
      const tf = flapTimeOn(world, st, i, d);
      const sw = swapAt.get(i);
      if (Number.isFinite(tf) && (!sw || tf < sw.at)) {
        const c = linkCost(world.links[i], d);
        events.push({ i, day: d, t: tf, kind: 'flap', cost: d >= CFG.backtestStart ? c.flap : 0, type: p.type, precursor: p.type === 'noise' || p.type === 'burst' });
        st.installs[i].push(d + 1); // reactive replacement next day
      } else if (sw) {
        events.push({ i, day: d, t: sw.at, kind: 'swap', cost: sw.cost, type: p.type, degrading: p.type === 'noise' || p.type === 'burst' });
        st.installs[i].push(d + 1);
      }
    }
  }
  const bt = events.filter(e => e.day >= CFG.backtestStart);
  const sum = (arr) => arr.reduce((s, e) => s + e.cost, 0);
  return {
    policy, events, forecasts, margins, st,
    gpuHoursLost: sum(bt),
    flapCost: sum(bt.filter(e => e.kind === 'flap')), swapCost: sum(bt.filter(e => e.kind === 'swap')),
    flaps: bt.filter(e => e.kind === 'flap').length,
    precursorFlaps: bt.filter(e => e.kind === 'flap' && e.precursor).length,
    swaps: bt.filter(e => e.kind === 'swap').length,
    healthySwaps: bt.filter(e => e.kind === 'swap' && !e.degrading).length,
    lastCal: cal,
  };
}

// Calibration records: forecasts made on days [calStart, d - horizon] with complete follow-up.
// Label = time from forecast moment to the link's next flap (Infinity if none within the horizon).
// Records whose follow-up was cut short by a planned swap are dropped (censoring).
export function buildCalibrator(world, forecasts, events, d, st) {
  const nextFlap = new Map(), nextSwap = new Map();
  for (const e of events) {
    const m = e.kind === 'flap' ? nextFlap : nextSwap;
    if (!m.has(e.i)) m.set(e.i, []);
    m.get(e.i).push(e.t);
  }
  const first = (arr, t0) => { if (!arr) return Infinity; for (const t of arr) if (t >= t0) return t; return Infinity; };
  const cal = new Calibrator();
  for (const f of forecasts) {
    if (f.day > d - CFG.horizon) continue;
    const tf = first(nextFlap.get(f.i), f.day), ts = first(nextSwap.get(f.i), f.day);
    if (ts < f.day + CFG.horizon && ts < tf) continue;
    const ttf = tf - f.day <= CFG.horizon ? tf - f.day : Infinity;
    cal.add(f.tauHat, ttf);
  }
  return cal.finalize();
}

// Out-of-sample calibration of P(flap within h days), using the reactive run (no swaps censor outcomes).
export function reliability(world, reactive, h = 7) {
  const flaps = new Map();
  for (const e of reactive.events) if (e.kind === 'flap') { if (!flaps.has(e.i)) flaps.set(e.i, []); flaps.get(e.i).push(e.t); }
  const first = (arr, t0) => { if (!arr) return Infinity; for (const t of arr) if (t >= t0) return t; return Infinity; };
  const bins = [[0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1.01]].map(([lo, hi]) => ({ lo, hi, n: 0, pSum: 0, hits: 0 }));
  let covN = 0, covHit = 0;
  const byDay = new Map();
  for (const f of reactive.forecasts) { if (!byDay.has(f.day)) byDay.set(f.day, []); byDay.get(f.day).push(f); }
  for (let d = CFG.backtestStart; d <= CFG.days - h; d++) {
    const cal = buildCalibrator(world, reactive.forecasts, reactive.events, d, reactive.st);
    for (const f of byDay.get(d) || []) {
      if (!Number.isFinite(f.tauHat)) continue;
      const pr = cal.predict(f.tauHat), p = pr.cdf(h);
      const tf = first(flaps.get(f.i), d), hit = tf - d <= h;
      const b = bins.find(b => p >= b.lo && p < b.hi); b.n++; b.pSum += p; b.hits += hit ? 1 : 0;
      if (Number.isFinite(tf) && tf - d <= CFG.horizon && d <= CFG.days - CFG.horizon) {
        covN++; if (tf - d >= pr.q10 && tf - d <= pr.q90) covHit++;
      }
    }
  }
  return { h, bins: bins.map(b => ({ lo: b.lo, hi: b.hi, n: b.n, meanP: b.n ? b.pSum / b.n : null, observed: b.n ? b.hits / b.n : null })), coverage80: covN ? covHit / covN : null, coverageN: covN };
}

// Today's view: forecasts and plan for the day after the simulated history.
export function today(world, run, capacity = CFG.capacity) {
  const d = CFG.days, nL = world.links.length;
  for (let i = 0; i < nL; i++) run.margins[i][d - 1] = telemetryOf(world, run.st, i, d - 1);
  const cal = buildCalibrator(world, run.forecasts, run.events, d, run.st);
  const cands = [];
  for (let i = 0; i < nL; i++) {
    const tr = trend(run.margins[i], d, world.links[i], sinceOf(run.st, i, d));
    if (!Number.isFinite(tr.tauHat)) continue;
    cands.push({ link: world.links[i], pred: cal.predict(tr.tauHat), tr });
  }
  cands.sort((a, b) => b.pred.cdf(CFG.horizon) * linkCost(b.link, d).flap - a.pred.cdf(CFG.horizon) * linkCost(a.link, d).flap);
  const plan = planSwaps(cands.slice(0, 60), d, capacity);
  return { day: d, cal, cands, plan };
}
