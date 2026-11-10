// Aria Console concept: Optics swap plan. Renders data/results.json (produced by scripts/run.mjs from
// core/optics.mjs) and recomputes the plan, FEC tail fit and per-day expected costs live with the same core.
import * as O from './core/optics.mjs';

const R = window.RESULTS;
const $ = id => document.getElementById(id);
const DAY = 864e5, TODAY = R.today;
const fmt = (n, d = 0) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const dateOf = day => new Date(R.day0 + Math.floor(day) * DAY);
const dayLbl = day => dateOf(day).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', timeZone: 'UTC' });
const md = day => dateOf(day).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
const hhmm = at => { const f = at - Math.floor(at); const m = Math.round(f * 1440); return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };
const when = at => (Math.floor(at) === TODAY ? 'today' : Math.floor(at) === TODAY + 1 ? 'tomorrow' : dayLbl(at)) + ' ' + hhmm(at);
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (parent) parent.appendChild(e); return e; };
const tip = $('tip');
const showTip = (e, html) => { tip.innerHTML = html; tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; };
const hideTip = () => { tip.style.display = 'none'; };

// ---------- candidates -> core objects ----------
const cands = R.candidates.map(c => {
  const host = c.kind === 'host' ? parseInt(c.peer.slice(5, 8), 10) : undefined;
  const link = { id: c.id, kind: c.kind, host, speed: c.speed, name: c.name, cwps: c.speed * 1e9 / 5140 };
  const cdf = t => { const g = R.grid, x = Math.max(0, Math.min(g[g.length - 1], t)), k = Math.min(g.length - 2, Math.floor(x / 0.25)); const f = (x - g[k]) / 0.25; return c.cdf[k] * (1 - f) + c.cdf[k + 1] * f; };
  return { ...c, link, pred: { cdf, q10: c.q10, q50: c.q50, q90: c.q90 } };
});
let capacity = R.cfg.capacity, plan = [], sel = null;

function solve() {
  plan = O.planSwaps(cands.map(c => ({ link: c.link, pred: c.pred })), TODAY, capacity, R.cfg.horizon);
  if (capacity === R.cfg.capacity) { // consistency check against the pipeline output
    const a = plan.map(p => p.link.id + '@' + p.day).join(), b = R.plan.map(p => p.id + '@' + p.day).join();
    if (a !== b) console.error('plan mismatch', a, b);
  }
}
const planOf = c => plan.find(p => p.link.id === c.id);

// ---------- header, prompt, queries ----------
$('clock').textContent = dateOf(TODAY).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' }) + ' · 09:00 UTC';
$('prompt').innerHTML = 'Which optics are going to take down a job on <code>demo1</code> in the next two weeks, and when should we swap them without stopping training?';
const trainJobs = R.jobs.filter(j => j.kind === 'train');
const leaves = [...new Set(cands.map(c => c.name.split(' ')[0]))];
$('q-head').textContent = '4 queries complete';
$('qlines').innerHTML = [
  `FEC codeword histograms (bins 0-15 + uncorrectable) on all ${fmt(R.fleet.links)} links, ${md(0)} - ${md(TODAY - 1)} ...`,
  `pre-FEC BER, corrected and uncorrectable codewords on ${R.fleet.leaves} leaves and ${R.fleet.spines} spines ...`,
  `Slurm jobs, checkpoint cadence and maintenance windows for ${trainJobs.map(j => `<span class="or">${j.id}</span>`).join(', ')} ...`,
  `optic swap and link flap history on ${leaves.map(l => `<span class="hl">${l}</span>`).join(', ')} ...`,
].map(s => `<div>${s}</div>`).join('');

// ---------- answer ----------
function holdWhy(c) {
  if (c.kind === 'fabric') return 'ARS reroutes around a flap, so a visit costs more than the expected loss';
  const j = R.jobs.find(j => j.id === c.job);
  if (j.kind === 'serve') return 'a flap only drains one 8-GPU replica';
  if (j.kind === 'idle') return 'spare host, swap at the next routine visit';
  return `too early to stop ${j.id}; re-checked tomorrow`;
}
function renderAnswer() {
  const rows = [...plan].sort((a, b) => a.at - b.at);
  const holds = cands.filter(c => !planOf(c));
  const lines = rows.map(p => {
    const c = cands.find(x => x.id === p.link.id);
    const how = c.kind === 'fabric' ? 'drain first, ARS reroutes' : p.window ? `${c.job} maintenance window` : `stop ${c.job} at a checkpoint`;
    return `<span class="or">${c.name}</span><span>${Math.round(c.p14 * 100)}% flap risk in 14 days · swap ${when(p.at)}, ${how}</span>`;
  }).concat(holds.map(c => `<span class="or">${c.name}</span><span>${Math.round(c.p14 * 100)}% flap risk in 14 days · hold, ${holdWhy(c)}</span>`));
  const w = rows.filter(p => p.window && p.day > TODAY).sort((a, b) => b.saving - a.saving)[0];
  let tail = '';
  if (w) {
    const c = cands.find(x => x.id === w.link.id);
    const nowCost = O.linkCost(c.link, TODAY), eNow = O.expectedSwapCost(c.pred, nowCost, TODAY, TODAY + nowCost.swapAt);
    tail = ` Waiting for ${c.job}'s window ${when(w.at)} instead of stopping the job today saves <b>${fmt(eNow - w.eCost)}</b> GPU-hours in expectation; ${(() => { const pf = c.pred.cdf(w.at - TODAY); return pf < 0.01 ? 'similar links almost never flapped this early' : `that includes the ${Math.round(pf * 100)}% chance it flaps first`; })()}.`;
  }
  const nNow = rows.filter(p => p.day === TODAY).length;
  $('answer').innerHTML = `<p>Fitted the FEC tail on every link and projected it to the flap line. ${cands.length} links are heading toward it; ${fmt(R.fleet.links - cands.length)} are not.</p>
    <div class="runs">${lines.join('')}</div>
    <p>${nNow ? `Swap ${nNow} today.` : ''}${tail}</p>`;
}

// ---------- plan table ----------
function renderCap() {
  $('cap').innerHTML = '<span>SWAPS/DAY</span>' + [2, 4, 6].map(k => `<button class="${k === capacity ? 'on' : ''}" data-k="${k}">${k}</button>`).join('');
  $('cap').querySelectorAll('button').forEach(b => b.onclick = () => { capacity = +b.dataset.k; solve(); renderAll(); });
}
function windowText(c) {
  if (c.q10 === null) return 'after ' + md(TODAY + R.cfg.horizon - 1);
  if (c.q90 === null) return md(TODAY + c.q10) + ' or later';
  const a = md(TODAY + c.q10), b = md(TODAY + c.q90);
  return a === b ? a : `${a} - ${b}`;
}
function workload(c) {
  if (c.kind === 'fabric') return 'fabric uplink<div class="sub">ECMP/ARS reroutes on drain</div>';
  const j = R.jobs.find(j => j.id === c.job);
  if (j.kind === 'train') return `${j.id}<div class="sub">${j.gpus} GPUs · ckpt ${j.ckptMin} min</div>`;
  if (j.kind === 'serve') return `${j.id}<div class="sub">drain one 8-GPU replica</div>`;
  return `${j.id}<div class="sub">idle host</div>`;
}
function renderPlan() {
  const rows = [...cands].sort((a, b) => { const pa = planOf(a), pb = planOf(b); if (pa && pb) return pa.at - pb.at; if (pa) return -1; if (pb) return 1; return b.p14 - a.p14; });
  let h = `<tr><th>Link</th><th>Workload</th><th>P(flap ≤ 14d)</th><th>80% flap window</th><th>Action</th><th class="num">GPU-h saved</th></tr>`;
  for (const c of rows) {
    const p = planOf(c);
    const act = !p ? `<span class="tag hold">Hold${c.kind === 'fabric' || R.jobs.find(j => j.id === c.job)?.kind !== 'train' ? ' · low impact' : ' · re-check tomorrow'}</span>`
      : p.day === TODAY ? `<span class="tag now">Swap today ${hhmm(p.at)}${c.kind === 'fabric' ? ' · drain' : p.window ? ' · window' : ''}</span>`
      : `<span class="tag ${p.window ? 'win' : ''}">Swap ${dateOf(p.at).toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' })} ${hhmm(p.at)}${p.window ? ' · window' : ''}</span>`;
    h += `<tr class="row ${sel && sel.id === c.id ? 'sel' : ''}" data-id="${c.id}"><td>${c.name}<div class="sub">→ ${c.peer}</div></td><td>${workload(c)}</td>
      <td><span class="risk"><i style="width:${Math.max(2, c.p14 * 56)}px"></i>${Math.round(c.p14 * 100)}%</span></td><td>${windowText(c)}</td><td>${act}</td><td class="num">${p ? fmt(p.saving, 1) : '–'}</td></tr>`;
  }
  $('plan-tbl').innerHTML = h;
  $('plan-tbl').querySelectorAll('tr.row').forEach(tr => tr.onclick = () => { sel = cands.find(c => c.id === +tr.dataset.id); renderAll(); });
  const tot = plan.reduce((s, p) => s + p.saving, 0);
  $('plan-note').innerHTML = `${plan.length} swaps scheduled, ${cands.length - plan.length} held · expected saving ${fmt(tot, 1)} GPU-hours vs. waiting for the flaps · solved exactly (Hungarian) over ${R.cfg.horizon} days × ${capacity} technician slots/day`;
}

// ---------- FEC histogram ----------
function renderHist() {
  const c = sel, h = c.hist, f = O.fitHistogram(h), cw = c.link.cwps;
  $('hist-ttl').textContent = `FEC Codeword Histogram - ${c.name}`;
  $('hist-pill').textContent = md(TODAY - 1);
  const W = 500, H = 250, L = 48, B = 26, T = 8, Rr = 8, yMin = -22, yMax = 0;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H + 16}` });
  const x = k => L + (k + 0.5) * (W - L - Rr) / 17, bw = (W - L - Rr) / 17 * 0.56;
  const y = v => T + (yMax - Math.max(yMin, Math.min(yMax, v))) / (yMax - yMin) * (H - T - B);
  for (let v = 0; v >= yMin; v -= 4) el('text', { x: L - 8, y: y(v) + 3.5, 'text-anchor': 'end' }, svg).textContent = '1e' + (v === 0 ? '-0' : v);
  el('line', { x1: L, x2: L, y1: T, y2: H - B, stroke: '#4a4a4a' }, svg);
  el('line', { x1: L, x2: W - Rr, y1: H - B, y2: H - B, stroke: '#4a4a4a' }, svg);
  for (let k = 0; k <= 16; k++) el('text', { x: x(k), y: H - B + 16, 'text-anchor': 'middle' }, svg).textContent = k === 16 ? '16+' : k;
  // observed bars
  h.counts.forEach((cnt, k) => {
    if (!cnt) return;
    const v = Math.log10(cnt / h.N), r = el('rect', { x: x(k) - bw / 2, y: y(v), width: bw, height: y(yMin) - y(v), fill: '#e8141c' }, svg);
    r.onmousemove = e => showTip(e, `${k} symbol errors<br>${fmt(cnt)} codewords · ${(cnt / h.N).toExponential(2)}`); r.onmouseleave = hideTip;
  });
  // model: random (Poisson) and burst components
  const pois = []; let lp = -f.lam;
  for (let k = 0; k <= 16; k++) { if (k > 0) lp += Math.log(f.lam) - Math.log(k); const v = lp / Math.LN10; if (v > yMin) pois.push([x(k), y(v)]); }
  el('polyline', { points: pois.map(p => p.join(',')).join(' '), fill: 'none', stroke: '#9e9e9e', 'stroke-width': 1.5 }, svg);
  if (f.bIcpt !== null) {
    const pts = []; const k0 = Math.max(1, f.pts.length ? f.pts[0][0] : 1);
    for (let k = k0; k <= 16; k++) { const v = f.bIcpt + f.burstSlope * k - Math.log10(h.N); if (v > yMin) pts.push([x(k), y(v)]); }
    el('polyline', { points: pts.map(p => p.join(',')).join(' '), fill: 'none', stroke: '#f39a4d', 'stroke-width': 1.6, 'stroke-dasharray': '4 3' }, svg);
  }
  // uncorrectable estimate and flap line
  const est = Math.log10(O.poisTail(f.lam) + f.b), thr = Math.log10(R.cfg.flapUcwPerSec / cw);
  const obs = h.ucw >= 3 ? Math.log10(h.ucw / h.N) : null, v16 = obs ?? est;
  el('rect', { x: x(16) - bw / 2, y: y(v16), width: bw, height: y(yMin) - y(v16), fill: obs !== null ? '#e8141c' : 'none', stroke: '#e8141c', 'stroke-width': 1.4, 'stroke-dasharray': obs !== null ? '' : '3 2' }, svg);
  el('line', { x1: L, x2: W - Rr, y1: y(thr), y2: y(thr), stroke: '#e5e3e1', 'stroke-dasharray': '2 4', 'stroke-width': 1.2 }, svg);
  el('text', { x: x(13), y: y(thr) - 7, 'text-anchor': 'middle', style: 'fill:#e5e3e1' }, svg).textContent = 'flap line · 1 uncorrectable/s';
  const m = Math.max(-3, thr - v16);
  const ax = x(16) - bw / 2 - 10;
  if (m > 0.15) { el('line', { x1: ax, x2: ax, y1: y(thr), y2: y(v16), stroke: '#b9eeae', 'stroke-width': 1.2 }, svg); el('text', { x: ax - 6, y: (y(thr) + y(v16)) / 2 + 4, 'text-anchor': 'end', style: 'fill:#b9eeae' }, svg).textContent = `margin ${m.toFixed(1)} decades`; }
  $('hist').replaceChildren(svg);
  $('hist-leg').innerHTML = `<span><b style="background:#e8141c;border-color:#e8141c;border-radius:2px"></b>observed (24 h)</span><span><b style="border-color:#9e9e9e"></b>random errors · λ ${f.lam.toExponential(1)}/codeword</span><span><b style="border-color:#f39a4d"></b>burst tail fit</span><span><b style="border-color:#e8141c;border-style:dashed;border-radius:2px"></b>uncorrectable (${obs !== null ? 'observed' : 'extrapolated'})</span>`;
}

// ---------- margin forecast ----------
function renderMargin() {
  const c = sel, p = planOf(c), lat = c.lat;
  $('margin-ttl').textContent = `FEC margin to the flap line - ${c.name}`;
  const W = 500, H = 250, L = 34, B = 26, T = 10, Rr = 10;
  const d0 = lat[0].day, d1 = TODAY + R.cfg.horizon, yTop = 10, yBot = -1.5;
  const x = d => L + (d - d0) / (d1 - d0) * (W - L - Rr), y = v => T + (yTop - Math.max(yBot, Math.min(yTop, v))) / (yTop - yBot) * (H - T - B);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H + 16}` });
  for (let v = 0; v <= 10; v += 2) { el('line', { x1: L, x2: W - Rr, y1: y(v), y2: y(v), stroke: v === 0 ? '#5a5a5a' : '#363636' }, svg); el('text', { x: L - 8, y: y(v) + 3.5, 'text-anchor': 'end' }, svg).textContent = v; }
  const step = Math.ceil((d1 - d0) / 6);
  for (let d = d0; d <= d1; d += step) el('text', { x: x(d), y: H - B + 16, 'text-anchor': 'middle' }, svg).textContent = md(d);
  // flap window band
  if (c.q10 !== null) {
    const a = x(TODAY + c.q10), b = x(TODAY + (c.q90 ?? R.cfg.horizon));
    el('rect', { x: a, y: T, width: Math.max(2, b - a), height: H - T - B, fill: '#e8141c', opacity: 0.16 }, svg);
    el('text', { x: (a + b) / 2, y: T + 12, 'text-anchor': 'middle', style: 'fill:#ef8e86' }, svg).textContent = '80% flap window';
  }
  el('line', { x1: x(TODAY), x2: x(TODAY), y1: T, y2: H - B, stroke: '#5a5a5a', 'stroke-dasharray': '2 3' }, svg);
  el('text', { x: x(TODAY) + 4, y: H - B - 6 }, svg).textContent = 'now';
  // observed + projection
  // what the model saw each day: robust trend level of the two error drivers (Theil-Sen over the previous 10 days)
  const latObj = Object.fromEntries(lat.map(o => [o.day, o]));
  const lvl = [];
  for (let d = d0 + 6; d <= TODAY; d++) { const tr = O.trend(latObj, d, c.link, c.since ?? -Infinity); lvl.push([d - 0.5, tr.level]); }
  lat.forEach(o => { const ci = el('circle', { cx: x(o.day + 0.5), cy: y(o.m), r: 2.6, fill: '#e8141c', opacity: 0.45 }, svg); ci.onmousemove = e => showTip(e, `${md(o.day)} · daily margin ${o.m.toFixed(2)} decades<br>pre-FEC BER ${o.ber.toExponential(1)} · λ ${Math.pow(10, o.x1).toExponential(1)}`); ci.onmouseleave = hideTip; });
  el('polyline', { points: lvl.map(([d, v]) => `${x(d + 0.5)},${y(v)}`).join(' '), fill: 'none', stroke: '#e8141c', 'stroke-width': 2.2 }, svg);
  const pr = R.grid.map((t, k) => [x(TODAY + t), y(c.trend.proj[k])]).filter((_, k) => c.trend.proj[k] > yBot - 1);
  el('polyline', { points: pr.map(q => q.join(',')).join(' '), fill: 'none', stroke: '#ef8e86', 'stroke-width': 1.6, 'stroke-dasharray': '5 4' }, svg);
  el('text', { x: W - Rr, y: y(0) - 6, 'text-anchor': 'end', style: 'fill:#e5e3e1' }, svg).textContent = 'flap line';
  if (p) {
    el('line', { x1: x(p.at), x2: x(p.at), y1: T, y2: H - B, stroke: '#b9eeae', 'stroke-width': 1.5 }, svg);
    el('text', { x: x(p.at) + 5, y: T + 28, style: 'fill:#b9eeae' }, svg).textContent = `swap ${hhmm(p.at)} ${md(p.at)}`;
  }
  $('margin').replaceChildren(svg);
  $('margin-leg').innerHTML = `<span><b style="background:#e8141c77;border-color:transparent"></b>daily</span><span><b></b>trend level (decades)</span><span><b style="border-color:#ef8e86"></b>trend projection · τ̂ ${c.trend.tauHat === null ? '> ' + R.cfg.maxTau : c.trend.tauHat.toFixed(1)} d</span><span><b style="border-color:#ef8e86;background:#e8141c44;border-radius:2px"></b>calibrated 80% window (${c.n} similar link-days)</span>`;
}

// ---------- expected cost by swap day ----------
function renderCost() {
  const c = sel, p = planOf(c);
  const opts = c.costs.map((k, i) => { const at = TODAY + i + k.swapAt; return { i, at, window: k.window, e: O.expectedSwapCost(c.pred, k, TODAY, at), swap: k.swap }; });
  const flap = c.costs[0].flap, eNone = c.pred.cdf(R.cfg.horizon) * flap, best = opts.reduce((a, b) => b.e < a.e ? b : a);
  $('cost-ttl').textContent = `Expected cost by swap day - ${c.name}`;
  const today = opts[0];
  if (p) {
    $('cost-big').textContent = fmt(eNone - p.eCost);
    $('cost-lbl').innerHTML = `expected GPU-hours saved by swapping<br><em>${when(p.at)}</em>${!p.window || p.day === TODAY ? '' : ` (vs ${fmt(today.e - p.eCost)} by swapping today)`}`;
  } else {
    $('cost-big').textContent = fmt(Math.max(0, best.e - eNone), 1);
    $('cost-lbl').innerHTML = `more GPU-hours if swapped now than<br><em>holding and re-checking tomorrow</em>`;
  }
  const W = 500, H = 190, L = 40, B = 34, T = 8, Rr = 6;
  const mx = Math.max(eNone, ...opts.map(o => o.e)) * 1.1 || 1;
  const x = i => L + (i + 0.5) * (W - L - Rr) / opts.length, bw = (W - L - Rr) / opts.length * 0.58, y = v => T + (1 - v / mx) * (H - T - B);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}` });
  const tick = mx > 300 ? 100 : mx > 120 ? 50 : mx > 40 ? 20 : 5;
  for (let v = 0; v <= mx; v += tick) { el('line', { x1: L, x2: W - Rr, y1: y(v), y2: y(v), stroke: '#363636' }, svg); el('text', { x: L - 8, y: y(v) + 3.5, 'text-anchor': 'end' }, svg).textContent = v; }
  opts.forEach(o => {
    const chosen = p && Math.floor(p.at) === Math.floor(o.at);
    const r = el('rect', { x: x(o.i) - bw / 2, y: y(o.e), width: bw, height: Math.max(1, y(0) - y(o.e)), fill: chosen ? '#b9eeae' : o.window ? '#f0646b' : '#e8141c', rx: 1.5 }, svg);
    r.onmousemove = e => showTip(e, `swap ${when(o.at)}${o.window ? ' (window)' : ''}<br>E[cost] ${fmt(o.e, 1)} GPU-h = P(flap first) × ${fmt(flap)} + rest × ${fmt(o.swap, 1)}`); r.onmouseleave = hideTip;
    el('text', { x: x(o.i), y: H - B + 15, 'text-anchor': 'middle' }, svg).textContent = dateOf(TODAY + o.i).getUTCDate();
    if (o.window) el('circle', { cx: x(o.i), cy: H - B + 25, r: 2.6, fill: '#b9eeae' }, svg);
  });
  el('line', { x1: L, x2: W - Rr, y1: y(eNone), y2: y(eNone), stroke: '#e5e3e1', 'stroke-dasharray': '2 4' }, svg);
  el('text', { x: L + 4, y: y(eNone) - 5, style: 'fill:#e5e3e1' }, svg).textContent = `no swap: ${fmt(eNone)}`;
  $('cost').replaceChildren(svg);
  $('cost-leg').innerHTML = `<span><b style="background:#e8141c;border-color:#e8141c;border-radius:2px"></b>stop at checkpoint</span><span><b style="background:#f0646b;border-color:#f0646b;border-radius:2px"></b>maintenance window</span><span><b style="background:#b9eeae;border-color:#b9eeae;border-radius:2px"></b>planned</span><span>${md(TODAY)} - ${md(TODAY + R.cfg.horizon - 1)}</span>`;
}

// ---------- backtest ----------
function renderBacktest() {
  const P = R.policies, me = P.planner.gpuHoursLost;
  $('bt-big').textContent = Math.round((1 - me / P.flag.gpuHoursLost) * 100) + '%';
  $('bt-lbl').innerHTML = `fewer GPU-hours lost than<br><em>swapping each optic as soon as it is flagged</em>`;
  $('bt-pill').textContent = `${R.cfg.capacity} SWAPS/DAY`;
  const rows = [['Replace after flap', P.reactive], ['Pre-FEC BER > 1e-5', P.ber], ['Swap when flagged', P.flag], ['Swap planner', P.planner, true]];
  const mx = Math.max(...rows.map(r => r[1].gpuHoursLost));
  $('bt-bars').className = 'bars';
  $('bt-bars').innerHTML = rows.map(([n, r, m]) => `<div class="b ${m ? 'me' : ''}" data-n="${n}"><span>${n}</span><div class="track"><div class="fill" style="width:${r.gpuHoursLost / mx * 100}%"></div></div><span class="v">${fmt(r.gpuHoursLost)}</span></div>`).join('');
  $('bt-bars').querySelectorAll('.b').forEach((b, k) => { const r = rows[k][1]; b.onmousemove = e => showTip(e, `${rows[k][0]}<br>${r.flaps} flaps (${fmt(r.flapCost)} GPU-h) · ${r.swaps} swaps (${fmt(r.swapCost)} GPU-h)<br>${r.healthySwaps} swaps of optics that were not degrading`); b.onmouseleave = hideTip; });
  const sudden = P.planner.flaps - P.planner.precursorFlaps;
  const S = window.SEEDS;
  $('bt-note').innerHTML = `${md(R.cfg.backtestStart)} - ${md(R.cfg.days - 1)} · rolling, each day uses only past telemetry · ${P.reactive.precursorFlaps} optics degraded to a flap with no intervention · planner: ${P.planner.swaps} swaps, ${P.planner.flaps} flaps (${sudden} sudden, no precursor)`
    + (S ? `<br>Across ${S.seeds.length} independently seeded clusters: median <span class="hl">${Math.round(S.vsFlag.median * 100)}%</span> fewer (range ${Math.round(S.vsFlag.min * 100)}-${Math.round(S.vsFlag.max * 100)}%), better in ${S.vsFlag.wins}/${S.seeds.length}` : '');
}

// ---------- calibration ----------
function renderCal() {
  const rel = R.reliability, W = 500, H = 236, L = 40, B = 30, T = 8, Rr = 12;
  const x = v => L + v * (W - L - Rr), y = v => T + (1 - v) * (H - T - B);
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}` });
  for (let v = 0; v <= 1.0001; v += 0.25) {
    el('line', { x1: L, x2: W - Rr, y1: y(v), y2: y(v), stroke: '#363636' }, svg); el('text', { x: L - 8, y: y(v) + 3.5, 'text-anchor': 'end' }, svg).textContent = Math.round(v * 100) + '%';
    el('text', { x: x(v), y: H - B + 16, 'text-anchor': 'middle' }, svg).textContent = Math.round(v * 100) + '%';
  }
  el('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: '#6a6a6a', 'stroke-dasharray': '3 4' }, svg);
  el('text', { x: x(0.98), y: y(0.98) + 16, 'text-anchor': 'end' }, svg).textContent = 'perfect calibration';
  const pts = rel.bins.filter(b => b.n > 0);
  el('polyline', { points: pts.map(b => `${x(b.meanP)},${y(b.observed)}`).join(' '), fill: 'none', stroke: '#e8141c', 'stroke-width': 1.6 }, svg);
  pts.forEach(b => { const ci = el('circle', { cx: x(b.meanP), cy: y(b.observed), r: 3 + Math.sqrt(b.n) / 2.2, fill: '#242424', stroke: '#e8141c', 'stroke-width': 2 }, svg); ci.onmousemove = e => showTip(e, `predicted ${Math.round(b.lo * 100)}-${Math.round(Math.min(1, b.hi) * 100)}% · mean ${(b.meanP * 100).toFixed(1)}%<br>observed ${(b.observed * 100).toFixed(1)}% of ${fmt(b.n)} link-days`); ci.onmouseleave = hideTip; });
  el('text', { x: L + 4, y: T + 12, style: 'fill:#9e9e9e' }, svg).textContent = 'observed';
  el('text', { x: W - Rr, y: H - B - 6, 'text-anchor': 'end', style: 'fill:#9e9e9e' }, svg).textContent = 'predicted';
  $('cal').replaceChildren(svg);
  $('cal-leg').innerHTML = `<span><b></b>${rel.bins.reduce((s, b) => s + b.n, 0).toLocaleString()} link-days with a downward trend, ${md(R.cfg.backtestStart)} - ${md(R.cfg.days - rel.h)}</span><span><b style="border-color:#b9eeae"></b>80% flap windows held ${Math.round(rel.coverage80 * 100)}% of ${rel.coverageN} actual flaps</span>`;
}

// ---------- guided workflow ----------
let approved = false;
const CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="#0d0d0d" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
function renderFlow() {
  const c = sel, p = planOf(c);
  const [sw, port] = c.name.split(' / '), [peer, peerPort] = c.peer.split(' / ');
  const a = `${sw}:${port}`, b = `${peer}:${peerPort}`;
  const steps = [];
  if (p && p.window && p.day > TODAY) steps.push(['Wait for window', [`${c.job} maintenance window · ${dayLbl(p.at)} ${hhmm(p.at)}`]]);
  if (c.kind === 'fabric') steps.push(['Drain', [`shift traffic off ${a} → ${b} (ARS)`, `verify no RoCE flows on ${a}`]]);
  else steps.push(['Drain', [`${p && p.window ? 'hold' : 'checkpoint'} ${c.job}, cordon ${peer}`]]);
  steps.push(['Replace optic', [`${a} ⇄ ${b} · ${c.speed}G`]]);
  steps.push(['PRBS link test', [`${a} → ${b}`, `${b} → ${a}`]]);
  steps.push(['Restore', [c.kind === 'fabric' ? `return ${a} to ECMP/ARS group` : `uncordon ${peer}, resume ${c.job}`]]);
  const ok = approved ? 'ok' : '';
  $('flow').innerHTML = `<div class="iface"><span class="c"></span><span class="n">${sw.replace('leaf', 'Leaf ')} ${port}</span><span class="k">INTERFACE</span></div>
    <div class="stem"></div>
    <div class="suite"><div class="hd"><span class="ck ${approved ? '' : 'pend'}">${approved ? CHECK : ''}</span><span class="n">${p ? 'Optic swap' : 'Optic swap (not scheduled)'}</span><span class="k mono" style="font-size:11.5px;letter-spacing:.12em;color:#9e9e9e">WORKFLOW</span></div>
      <div class="steps">${steps.map(([t, ls]) => `<div class="st">${t}</div>${ls.map(l => `<div class="ln"><span class="mini ${ok}"></span>${l.replace(/(leaf\d+|spine\d+|host-\d+)/g, '<b>$1</b>')}</div>`).join('')}`).join('')}</div>
      <div class="act"><button class="btn primary" id="approve" ${approved || !plan.length ? 'disabled' : ''}>${approved ? 'Approved' : `Approve ${plan.length} swaps`}</button><span class="mono">${approved ? `scheduled · first runs ${when(Math.min(...plan.map(q => q.at)))}` : p ? `runs ${when(p.at)}` : 'held · re-checked daily'}</span></div></div>
    <div class="stem"></div>
    <div class="iface"><span class="c"></span><span class="n">${peer.replace('spine', 'Spine ')} ${peerPort}</span><span class="k">${c.kind === 'fabric' ? 'INTERFACE' : 'NIC'}</span></div>`;
  const btn = $('approve'); if (btn) btn.onclick = () => { approved = true; renderFlow(); };
}

function renderAll() { renderCap(); renderAnswer(); renderPlan(); renderHist(); renderMargin(); renderCost(); renderBacktest(); renderCal(); renderFlow(); }
solve();
sel = [...plan].sort((a, b) => b.saving - a.saving).map(p => cands.find(c => c.id === p.link.id))[0] || cands[0];
renderAll();
