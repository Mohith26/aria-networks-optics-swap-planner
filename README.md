# Optics Swap Planner for Aria Console (concept)

**Live demo:** https://mohith26.github.io/aria-networks-optics-swap-planner/

An unofficial concept feature for [Aria Networks](https://arianetworks.com)' Aria Console. Aria already captures FEC codeword histograms from every optic and catches transceivers that are going bad. This answers the next question an operator of a training cluster asks: **when should each degrading optic be swapped, given what is running over it?**

For every link it fits the FEC codeword histogram, projects it to the point where the link starts flapping, turns that into a calibrated time-to-flap distribution, and then schedules swaps into the cheapest moments (a job's maintenance window, a checkpoint boundary, or a drain for fabric links) by solving an assignment problem over the next 14 days.

Synthetic data. Not affiliated with Aria Networks.

## Why

On a training cluster a host-link flap does not just drop packets: NCCL stalls, the job dies and restarts from its last checkpoint, so one optic can cost hundreds of GPU-hours. Swapping early is not free either: on a host link it means stopping the job. "Swap when flagged" therefore wastes GPU-hours, and "wait for it to fail" wastes more. The right answer depends on three things at once: how soon the optic will fail (with honest uncertainty), how many GPUs are behind it, and when the job next pauses anyway.

## How it works (`core/optics.mjs`)

1. **Two-component FEC tail fit.** Each day's RS(544,514) codeword histogram (bins 0-15 plus uncorrectable) is split into random symbol errors (Poisson, rate taken from the zero bin) and error bursts (whatever the Poisson part cannot explain in the high bins, fitted as a geometric tail with a shrinkage prior on its shape). The probability of more than 15 symbol errors per codeword is the Poisson tail plus the burst tail. This matters because the two failure modes look different: rising random noise shows up in pre-FEC BER, but a growing burst tail can reach the flap line while pre-FEC BER stays near 1e-7.
2. **FEC margin.** Decades between the estimated uncorrectable-codeword probability and the flap line (1 uncorrectable codeword/s).
3. **Trend.** Theil-Sen slopes of both drivers (log random-error rate, log burst tail) over the last 10 days, projected forward to the first time the margin reaches zero: a naive days-to-flap τ̂.
4. **Calibration.** τ̂ is not trusted directly. For each link, the 40 most similar past link-days (by log τ̂) give an empirical distribution of the *actual* time to the next flap, shrunk toward the fleet base rate. That yields P(flap within t) and an 80% flap window.
5. **Plan.** For every candidate link and day, expected cost = P(flap before the swap) × flap cost + P(no flap) × swap cost, where costs come from the workload: training jobs lose the work since their last checkpoint plus restart time; a swap in a scheduled maintenance window is nearly free; a fabric link is drained and rerouted. Swaps are assigned to (day, technician slot) pairs with the Hungarian algorithm, so the plan is optimal for the stated costs and capacity.

## Results (all computed by `scripts/run.mjs` and `scripts/seeds.mjs`)

Simulated cluster: 1,024 GPUs, 128 hosts, 16 leaves, 8 spines, 1,536 optical links, 90 days. Four policies are replayed day by day over the last 40 days, each using only past telemetry, with the same 4 swaps/day technician capacity:

| Policy | GPU-hours lost (seed 71, shown in the demo) |
|---|---|
| Replace after flap | 6,458 |
| Pre-FEC BER > 1e-5 alarm | 8,948 |
| Swap when flagged (same detector, P(flap ≤ 14 d) ≥ 50%) | 4,060 |
| **Swap planner** | **2,845** |

Across 20 independently seeded clusters (seeds 1-20), the planner lost a **median 37% fewer GPU-hours than swap-when-flagged** (range 13-59%, better in 20/20), a median 61% fewer than replacing after flaps, and 72% fewer than the BER alarm. The BER alarm does badly because it swaps stable-but-noisy optics that never fail and misses burst-type failures.

Out-of-sample calibration of P(flap within 7 days) (reactive replay, so swaps don't censor outcomes): the 80% flap windows contained 89% of actual flaps (163). Calibration is uneven: the 10-30% band is over-confident (predicted 16%, observed 6%), the 50-70% band under-confident (61% vs 92%, only 12 link-days), and the top band under-confident (91% vs 100%).

## Tests

```
node --test tests/optics.test.mjs   # 12 tests
```
Poisson tail vs brute force; histogram probabilities sum to one; the fit recovers λ and the uncorrectable probability within a decade; margin is monotone in noise; Theil-Sen exactness and robustness; trend crossing matches the analytic answer; Hungarian equals brute force on random matrices; planner waits for a cheap window at low risk, swaps now at high risk and respects capacity; calibrator monotonicity; backtest ledger consistency and that committed results reproduce; plus two regression tests for defects found while building (a stable high-BER optic misread as a burst, and lost Poisson mass at very high error rates in the generator).

## Honest limitations

- **Synthetic physics.** The optic model (Poisson random errors plus geometric bursts, daily jitter, linear-in-log degradation with mild acceleration, a smooth flap hazard at 1 uncorrectable/s) is a plausible stand-in, not fitted to real Aria telemetry. Real optics have per-lane histograms, temperature and laser-bias signals, connector contamination that cleaning fixes, and correlated failures; none of that is modeled.
- **Daily counters only.** Aria's telemetry is microsecond-resolution. This uses one histogram per link per day; finer data would make the trend sharper and the flap model richer.
- **Cost model is assumed.** Flap cost = GPUs × (half a checkpoint interval + restart + NCCL timeout); a swap visit is charged 20 GPU-hour equivalents for labor, the optic and reseat risk; maintenance windows are synthetic. Change these and the plan changes; that is the point, but the numbers are only as good as the inputs.
- **Calibration is simple.** One feature (τ̂) with nearest neighbours and a base-rate prior. It is roughly calibrated but uneven across bands (see above) and has little data for rare regimes; a second feature (current margin) or isotonic recalibration would be the next step. Sudden failures with no precursor (6 of 8 planner flaps in the shown replay) are not predictable by design.
- **Seed choice.** Seed 71 is shown because its current-day state has a mix of decisions (swap now, wait for tomorrow's window, hold). Its backtest result (30% vs swap-when-flagged) is below the 20-seed median (37%).
- **UI.** Aria Console's full app shell is not public. The investigation thread, cards, charts, workflow card and composer are rebuilt from Aria's official demo videos and product images; the thin top bar is an assumption.

## Novelty note

Aria publicly describes catching degrading transceivers and guiding remediation, and its CEO has written about a model that predicts transceiver failure. NVIDIA UFM Cyber-AI predicts InfiniBand link failures 1-24 hours ahead, and its Autonomous Link Maintenance plugin isolates links it expects to fail. I could not find any product or Aria post that schedules optic swaps against the training jobs' checkpoint and maintenance windows with a calibrated time-to-flap distribution, or that separates the random-error and burst components of the FEC histogram to extrapolate the uncorrectable rate. That scheduling layer is the contribution here.

## Run it

Static site, no build step. `node scripts/run.mjs` regenerates `data/results.json`; `node scripts/seeds.mjs` regenerates the 20-seed robustness table. The page renders the results and recomputes the plan, FEC fit and per-day expected costs live with the same core module.

Built by Mohith Gajjela (mohithgajjela.com).
