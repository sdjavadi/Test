# PKG Counterparty Locatability Manifest — v1

**Payment Knowledge Graph (PKG) — inferring position for nodes with no address**
*PNC Bank · Treasury Management · Data Science*
*Tracks `pkg_geo_locatability.ipynb`*

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
width'  = R_EARTH · sqrt( 2 (1 − R̄') )        [angular dispersion, km]
```

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
| **Ground truth is itself noisy** | Error is measured against the registered pin, which the geo work showed is not always where the business is |

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

## 10. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08 | Initial. Exact-LOO kernel estimator, degree-1 census, locatability gate, hub locating registry, window and rung switches |
