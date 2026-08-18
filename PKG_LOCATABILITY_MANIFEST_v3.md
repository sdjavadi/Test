# PKG Counterparty Locatability Manifest — v3

**Payment Knowledge Graph (PKG) — inferring position for nodes with no address**
*PNC Bank · Treasury Management · Data Science*
*Tracks `pkg_geo_locatability.ipynb`*

> **v2 → v3 — the guard worked; four refinements and the scoring function.**
>
> The v2 run made the bandwidth/error relationship monotone across all ten
> valid deciles and lifted gate AUC 0.810 → 0.888. **Decile 0 — mean width
> 9.8 km — gives 14.7 km median error and 67.6% within 50 km on 8.5% of the
> census.** That signal was smeared across three deciles in v1. The hub
> coefficient flipped −1.25 → +1.69, confirming the v1 negative was the ladder
> definition tagging isolated small nodes.
>
> Two findings changed the design:
>
> - **Uniform averaging is worthless.** On a fixed cohort it is flat at ~190 km
>   from k=1 to k=20 and gets *worse* from one neighbour to two, while
>   inverse-variance falls to 19 km. With heavy-tailed widths, precision
>   weighting is not a refinement — it is the mechanism. One national
>   counterparty drags a good local one to the middle of nowhere.
> - **Selection beats combination.** `tightest_first` sits flat at ~75% hit@50
>   from k=1 while random pooling climbs from 22% to 70% over twenty
>   neighbours. The product picks the narrowest kernel; it does not average.
>
> And the curve **does not flatten by k≈5** — it keeps paying to k=20, which
> strengthens the case for acquiring counterparty edges.
>
> | # | Refinement | Reason |
> |---|---|---|
> | A | Band rule scans **empirical bandwidth quantiles**, and selects **max coverage subject to a precision floor** | the round-number grid shipped 4% coverage while decile 0 gave 8.5% at the same precision; max precision×coverage drifts to wide bands |
> | B | `COLOC_MIN_CLUSTER` 25 → **5** | only 9,080 parties caught vs Block B's ~910k, and `sh_bw_zero` stayed at 0.3% — exact duplicates surviving in small clusters |
> | C | **Per-node coverage table** added | the deliverable is per node, not per pair: a target with ten neighbours gets ten chances at a tight kernel |
> | D | **No-estimate class split** by degree and hub flag | median 714 km but hit@50 22% — it was averaging tiny local nodes with large mostly-unlocated ones |
>
> Also relabelled: **`bw_ok` is reliability, not tightness.** A neighbour with
> ≥5 counterparties skews larger and more national, and the row gave zero
> accuracy lift. The quality filter is `TIGHT_BW_KM`, fitted in §7.
>
> **New in v3: `locate()` (§13)** — the scoring function, in two modes.
>
> **v1 → v2 — five defects found by the first full run (5.49M pairs) and fixed.**
>
> | # | Defect | Fix |
> |---|---|---|
> | 1 | **`bandwidth = 0` degeneracy.** A leave-one-out that leaves a neighbour with one counterparty gives `R̄'=1` exactly, so the width collapsed to 0 — maximally *confident*, ~300 km wrong, across **1.17M pairs (21% of the census)**, sitting at the good end of the feature | `n_cp_loo` emitted; width **NaN below 2**, untrusted below 5. Mirrors Block F's own `MIN_CP_FOR_SPREAD` |
> | 2 | **Gate misspecified.** `log(bw)` entered linearly, but the relationship is non-monotone — tight is good, *zero* is meaningless — so a linear logit could not use it. 0.70 precision at 2.2% coverage, while bandwidth decile 2 alone gave 0.73 at 8.3% | bandwidth **binned**, plus an explicit no-estimate class, benchmarked against a **one-line band rule** |
> | 3 | **Placeholder co-location.** `bw = 0.000` *and* `err = 0.000` are shared-coordinate clusters (Block B: 909,820 parties in 16,824 clusters; one Pittsburgh point holds 29,780), manufacturing perfect scores | clusters detected and excluded from **both** truth and clouds; population cost reported |
> | 4 | **`is_hub` was wrong.** V0-minus-P99_9 = 1,206,594 nodes — hubs *plus everything that only reached the graph through one*. Only ~2,961 were real, so the flag largely tagged isolated small nodes | hubs by **median-multiple degree**; ladder set kept as a diagnostic |
> | 5 | **k-curve confounded.** `n` fell 1.9M → 10k across k, so every k was a different population and the rise past k=5 was composition | **fixed cohort** with ≥ `K_MAX` located neighbours |
>
> **Promoted to a standing rule:** accuracy is always reported **conditional on
> `geo_registered_vs_flow_km`**. The v1 pooled figure mixed estimator error with
> ground-truth error at ~6× — 293 km pooled versus **46 km** where the pin is
> representative, at **51% within 50 km from a single neighbour** against a
> footprint prior of 0.014%. The pooled number was the one that would have been
> briefed, and it understated the capability by an order of magnitude.

---

## 1. Problem statement

Counterparty nodes have no address. The goal is to infer one from the PNC
customers they transact with. There is no ground truth on counterparties, so
the entire study runs on **customers**, whose address is known: hide it,
predict it back from neighbours, measure the error.

**The deliverable is not a centroid for every node.** At degree 1 the error is
bimodal by construction — a single neighbourhood retailer pins a target within
kilometres; a single card processor returns the population prior. No estimator
repairs the second case, and a point estimate that cannot distinguish them is
worse than no estimate, because nobody downstream can tell which they hold.

The three outputs:

| # | Output | Consumer |
|---|---|---|
| 1 | **Error-vs-evidence curve** — error as a function of located neighbours, from k=1 | sizing / feasibility |
| 2 | **Locatability gate** — calibrated P(target within R km) | the product |
| 3 | **Neighbour informativeness ranking** — which sectors, sizes and hubs carry location | the hub registry, and the LOB work after it |

---

## 2. Method

### 2.1 Each neighbour is a kernel, not a point

Neighbour *j* induces a distribution over where *its* counterparties live. The
target is one draw from it.

- **Centre** = *j*'s **counterparty centroid**, not its registered pin. The
  target is a counterparty of *j*, so the cloud is the relevant distribution.
- **Width** = *j*'s own angular dispersion. A processor gets a near-flat kernel
  and contributes nothing **automatically** — hub muting falls out of the
  estimator instead of needing an exclusion list.

### 2.2 Mean resultant length, and why the leave-one-out is exact

For counterparties with unit vectors `u` and weights `w`:

```
S = Σ w        V = Σ w·u        R̄ = |V| / S
```

Removing target *i* is a subtraction, not a recomputation:

```
S' = S − w_i        V' = V − w_i·u_i        R̄' = |V'| / S'
centre' = V' / |V'|
width'  = R_EARTH · sqrt( 2 (1 − R̄') )   if n_cp_loo ≥ 2,  else NaN
```

**The guard is not optional.** With one counterparty remaining, `R̄' = 1`
identically and the width is exactly 0 — which reads as *maximum confidence*
rather than *no information*. In v1 that was 1.17M pairs at the confident end
of the feature and it is what broke the gate. `n_cp_loo` is emitted as a
column; below `MIN_CP_BW` the width is NaN, and below `MIN_CP_BW_TRUST` it
exists but `bw_ok = 0`.

Three consequences:

- **Exact LOO for both centre and width**, at one subtraction per pair.
  Without it the neighbour's cloud statistic contains the node being
  predicted; at low neighbour degree that is circular, and it is the single
  largest source of optimism in this class of analysis.
- **A neighbour whose only located counterparty is the target collapses to
  `S'=0` and drops out.** Correct — it carries no independent information.
- `R̄` is the same quantity as Block F's `geo_R`, so the bandwidth is
  consistent with the published metric rather than a parallel invention.

### 2.3 Weighting is an open question, deliberately

Three schemes accumulate in the same pass and are compared head to head:

| Scheme | Weight | Rationale |
|---|---|---|
| uniform | 1 | every neighbour is one observation |
| amount | edge dollars | Block F's convention — **carried here to be falsified** |
| inverse-variance | 1 / width² | correct combination; hubs self-mute |

Block F weights by dollars because it describes a customer's own economic
footprint. For *locating an unknown node*, dollars are close to
**anti-correlated** with informativeness: the largest edge is
disproportionately a processor. If `med_amt > med_uni`, that is the finding.

### 2.4 Selection vs combination

`tightest_first` ignores the random draw and takes the *k* narrowest kernels.
If it beats random pooling at low k, **evidence selection matters more than
evidence combination**, and the product should choose which neighbour to trust
rather than averaging. That changes the design, so it is tested, not assumed.

---

## 3. Configuration switches

### 3.1 `WINDOW` — `last3` vs `all`

Registered location is **current-state applied to all history**. Pooling 23
months adds edges (good) and stales the attribution for anyone who moved (bad).

**The difference between the runs is the NET of those two effects, not the
staleness cost alone.** If `all` wins, the extra evidence more than paid for
the movers. Neither number may be reported as "the cost of stale addresses".

### 3.2 `VERSION` — `V0` vs `P99_9`

Run **V0 first**. The question is not whether to exclude hubs but **which hubs
locate**. Hub membership is derived from the ladder itself — present in V0,
absent from P99_9 — and the rung is applied by **ablating hub-touching edges**,
because the rung is a property of the graph, not of the dimension.

Aggregate performance can be worse at V0 while §9 shows specific hubs are
excellent. Not a contradiction — it is the argument for a graded registry over
a flat exclusion list.

---

## 4. Standing traps

| Trap | Rule |
|---|---|
| **Join dtype** | `source`/`dest` and `mdm_id` must both be strings. The 2026-07 incident was an int64/str mismatch that typed every counterparty `unknown` and produced plausible output. A match-rate assert runs before anything else |
| **Leakage** | Neighbour statistics must be leave-one-out. Using the metric table's `geo_reach_p50_km` directly is circular |
| **`ntile` over a global window** | Sorts the whole pair table into one partition. Use `approxQuantile` + bucket map. At real scale this is a driver collapse, not a slowdown |
| **float16** | Never downcast coordinates. `lat 40.44052 → 40.4375` is 0.3 km of silent displacement on every node |
| **Direction** | in- and out-locality differ (`home_state_share_in` ≠ `_out`); keep `amt_in` / `amt_out` separate and let the harness decide |
| **Months are not independent evidence** | A neighbour seen 23 times is one spatial constraint. Cadence belongs in the confidence model and in LOB, **not** in the location likelihood |
| **Ground truth is itself noisy** | Never quote a pooled accuracy number. Always split on `geo_registered_vs_flow_km`; in v1 the pooled figure was 6× worse than the representative-pin figure |
| **Placeholder co-location** | Shared-coordinate clusters give `bw=0` and `err=0`. Exclude from truth *and* from clouds; report the population cost |
| **Hub definition** | Median-multiple degree, never a percentile (which flags a fixed fraction by construction) and never ladder exclusion (which captures hub-dependent nodes too) |
| **Cohort drift across k** | Fix the cohort before reading an error-vs-k curve, or composition masquerades as an effect |

---

## 5. Evaluation

**Metrics.** Median and p90 error in km, plus hit-rate at 25 / 50 / 250 km.
Never the mean — the distribution has a national tail.

**Baselines, ordered by how embarrassing it would be to lose to them:**

1. **PNC deposit-footprint prior** — population centroid of located customers.
   Much apparent skill is just "PNC customers are in PA/OH". An estimator that
   does not beat this decisively has produced nothing.
2. Single random neighbour (the degree-1 census), uniform centroid,
   amount-weighted centroid.
3. `tightest_first` at k→`K_MAX`, as the practical ceiling.

**Truth-set sensitivity.** Re-scored on `geo_registered_vs_flow_km ≤ 25 km` —
targets whose pin is representative. If accuracy improves sharply there, the
headline numbers are **pessimistic**, and the clean-set figure is the honest
one to quote, with the caveat that counterparties resemble the full set more.

**Footprint bias.** Every observed neighbour is a PNC customer, so every
prediction is pulled toward the PNC footprint. Targets registered outside the
core states are the closest available analogue to a counterparty that banks
elsewhere — which is most of them. The in-core vs out-of-core gap **bounds the
optimism and must be quoted alongside every accuracy figure.**

---

## 6. Outputs

```
../metrics/locatability/{VERSION}_{WINDOW}/
    summary.json                  headline numbers for the config comparison
    gate_operating_points.csv     precision / recall / flagged share
    error_vs_k.csv                full curve, all schemes, all replicates
    hub_locating_registry.csv     per-hub locating class   (V0 only)
    truth_sensitivity.csv         accuracy by ground-truth quality
    footprint_bias.csv            in-core vs out-of-core
../metrics/locatability/config_comparison.csv
```

### 6.1 Hub locating registry

Replaces the flat exclusion list with a graded one:

In the v1 run **over half of real hubs located to metro or better** — PIN 383,
METRO 1,165, REGION 883, NONE 530. Utilities (22) led at 28 km median / 69%
hit@50, then Education (61) and Public administration (92); Manufacturing (32)
and Finance (52) were worst. PNC's own book-transfer accounts landed correctly
in the NONE tail. A flat exclusion list discards most of this.

| class | median degree-1 error | use |
|---|---|---|
| `PIN` | ≤ 25 km | full-weight evidence |
| `METRO` | ≤ 100 km | metro-level evidence |
| `REGION` | ≤ 400 km | state-level only |
| `NONE` | > 400 km | exclude from location inference |

Hypothesis under test: government, utility and municipal accounts land in
`PIN`/`METRO` despite very high degree, because their counterparties are bound
to the geography they serve; processors land in `NONE`. **Degree does not
determine locating power — bandwidth does.**

---

## 7. Reading the results

| Result | Consequence |
|---|---|
| bandwidth-only AUC ≈ full AUC | ship the one-feature gate — auditable line by line, which matters because this feeds prospecting |
| sector effect vanishes conditional on bandwidth | NAICS belongs in the **prior on kernel width**, not as its own term; sparse-reach neighbours shrink toward it |
| `med_amt` > `med_uni` | dollars do not carry location |
| `tightest_first` > `random` at low k | selection beats combination; the product chooses evidence rather than averaging |
| curve flat beyond k ≈ 3–5 | the marginal neighbour stops paying early; effort goes to the gate, not to acquiring edges |

**The number that decides the programme** is `share_of_pairs_flagged` at the
chosen precision. If it is 15%, then 15% of counterparties get a CBSA and 85%
get an honest `unknown` — a usable asset. A point estimate for 100% is not.

---

## 8. Known limits

- **Distribution shift is real and one-directional.** Counterparties bank
  elsewhere, which correlates with sitting outside the PNC footprint, and
  their observed neighbour sets are a more biased sample of their true
  counterparty base. **Every number here is optimistic.** The out-of-footprint
  split bounds it; it does not remove it.
- **`scope = on_us_c2c`.** A counterparty's real cloud is mostly invisible; we
  infer position from the PNC-visible slice only.
- **No external ground truth yet.** The FI pinning registry (FDIC Summary of
  Deposits, NCUA) is the one available set of counterparties with known
  locations. Small and skewed to financial institutions, but drawn from the
  *right population* — a precondition for briefing any accuracy figure outside
  the team.
- **Compliance.** Inferring locations for non-customers to support prospecting
  needs the permissible-use read already flagged as a governance precondition,
  plus a fair-lending view on any geography-routed queue.

---

## 9. Sequencing

1. Run all four configurations (`V0`/`P99_9` × `last3`/`all`); settle window
   and rung before tuning anything.
2. Freeze the gate at a precision target. Emit **`(cell, radius, p)` — never a
   bare lat/lon.**
3. Validate against the FI registry; report the customer-to-counterparty gap.
4. Only then consider learned models. The estimator here is **one round of
   message passing with hand-set weights**, which makes it the ablation
   baseline any GNN must beat. A learned model will not help at degree 1 —
   architecture does not create data — but 2-hop co-location through shared
   customers is genuine headroom. If pursued, predict over a **discrete cell
   grid**, not raw lat/lon: it yields a real posterior (the confidence output,
   for free) and handles multimodality, which a Gaussian head cannot.

---

## 9a. `locate()` — the scoring function

Two modes, one estimator:

| mode | call | use |
|---|---|---|
| by node id | `locate(ids=[...])` | node already in the graph; neighbours looked up |
| by neighbour list | `locate(neighbours={"CPTY_1": [...]})` | **external counterparty** — no `mdm_id`, but its PNC counterparties are known |

Returns `est_lat`, `est_lon`, `pred_width_km`, `r50_km`, `r90_km`,
`p_within_50km`, `p_within_250km`, `k_used`, `tightest_bw_km`,
`evidence_class`. **It never returns a bare coordinate** — a point with no
radius cannot be used responsibly, because nothing distinguishes a 15 km
estimate from a 900 km one.

Design carried from the analysis:

- **tightest-*k* selection**, then inverse-variance weighting among the chosen
- **leave-one-out when the target is itself located.** `exclude_self` defaults
  True for `ids` (a customer sits inside its neighbours' clouds) and False for
  `neighbours` (an external counterparty does not). Getting this wrong is
  silent circularity — passing a customer id through counterparty mode leaks
  the answer.
- **co-located and unguarded neighbours dropped**, matching the harness
- **radii from `radius_calibration.csv`** — empirical quantiles of realised
  error at each predicted width, never a Gaussian assumption. The error
  distribution is heavy-tailed and partly bimodal; a parametric radius is
  badly optimistic at the top end.

**Self-scoring is built in (§13.1).** Known customers are scored through the
same function and checked two ways: realised coverage of `r50`/`r90` must land
near 50%/90%, and the median error must not be *better* than §8 on comparable
k — if it is, `exclude_self` is not applying.

`NO_EVIDENCE` is a real answer. Do not backfill it with a footprint centroid;
the entire point of the gate is that located and unlocated stay
distinguishable.

---

## 10. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08 | Initial. Exact-LOO kernel estimator, degree-1 census, locatability gate, hub locating registry, window and rung switches |
| 3.0 | 2026-08 | Precision-floored band selection on empirical quantiles; co-location threshold 5; per-node coverage; no-estimate class split; **`locate()` scoring function with empirical radius calibration** |
| 2.0 | 2026-08 | Post-first-run. Bandwidth degeneracy guard (`n_cp_loo`), binned gate + band-rule benchmark, co-location exclusion, degree-based hub definition, fixed-cohort k-curve. Gap-conditional reporting made a standing rule |
