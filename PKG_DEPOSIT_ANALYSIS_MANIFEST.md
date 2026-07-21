# Deposit Attrition × Payment Knowledge Graph — Analysis Manifest
**PNC Bank · Treasury Management | Data Science**
*Tiers 0–2 · July 2026*

---

## 1. Objective

Quantify the relationship between customer-to-customer (C2C) payment-network behavior
and deposit balance attrition, and determine whether PKG-derived metrics carry
**early-warning signal** for deposit decline beyond what deposit history alone provides.

Data window design: deposits run Jan 2024 – Jun 2026; PKG node metrics run
Jan 2024 – Nov 2025. The 7-month deposit-only tail (Dec 2025 – Jun 2026) is reserved
as a **zero-leakage out-of-time test**: features can be frozen at Nov 2025 and scored
against outcomes the network data has structurally never seen.

Known limitation, by design: the PKG currently contains only C2C (intra-PNC) edges.
Customers transacting mostly with outside-PNC counterparties are under- or un-measured.
All results are therefore **lower bounds** — classical measurement-error attenuation —
and the counterparty ingestion milestone is the natural before/after experiment.

---

## 2. Tier 0 — Panel Assembly (`deposit_panel_builder.py`)

### Inputs
| Source | Content | Grain |
|---|---|---|
| `pl43379_dsi_liq_attrition_base` | wide monthly balances `bal_2024_01…bal_2026_06`, account counts | 1 row / `cust_pwr_id` (verified: 66,947 rows = 66,947 ids) |
| PKG node metrics (Hive) | 57-column monthly node metrics, ablation version **P99_9** | (`cust_pwr_id`, month) — verified unique |

### Outputs (Hive, prefix `cust_deposit_pkg_metrics_`)
| Table | Grain | Content |
|---|---|---|
| `deposit_panel` | cust × month | balance, activity status, rolling features, decline flags, event markers |
| `joint_panel` | cust × month | deposit panel ⟕ `pkg_*` metrics; post-Nov-2025 months retained with null metrics |
| `customer_dim` | cust | activity bounds, first event month, visibility index + tier, closed-account ratio |
| `coverage` | month | join coverage both directions |

### Key definitions
- **Decline flag (recomputed monthly)**: `recent_avg` = mean(bal, t−2…t), `prior_avg` =
  mean(bal, t−8…t−3); flag = 1 if pct change ≤ −30%. Eligibility requires **full**
  windows (3 + 6 non-null) and `prior_avg ≥ $10K`; ineligible rows get **null** flags,
  not zeros, so they drop out of event analyses rather than diluting them.
- **Event start (t0)**: first 0→1 transition of the flag. Basis for the event study
  and the Tier 2 onset target.
- **Visibility tier**: composite percent-rank of graph presence rate and median
  strength/balance intensity, terciled; never-in-graph customers = `invisible`.

### What the build run established
- **Null semantics resolved**: balance null share declines monotonically 24% → 0.0 by
  2025-10 ⇒ nulls are onboarding ramp (`pre_first_activity`), not data quality issues.
- **Zero semantics**: zero share rises 1.5% → 7.2% across 2026 ⇒ account closures
  accumulating in the deposit-only tail. The decline flag catches balance→0 attrition
  in exactly the freeze-test window.
- **Mapping scale**: 25.5M metric rows (P99_9), 1.47M rows (5.7%) carry a
  `cust_pwr_id`; **159,224 distinct mapped ids**. Guard match rate mapped→deposit =
  5.3%: the deposit table is a **narrow book** (67K customers) relative to the
  mapping — findings generalize to this book, not the full customer base.
  *(Open item: confirm the table's population filter with its owner.)*
- **Analytical sample**: ~8.2K deposit customers ever graph-visible (~12% of book);
  ~4,700–5,000 joined per month; dep_coverage ≈ 9.9% (2024) → 7.1% (late 2025).
  This coverage number is the headline argument for counterparty ingestion.
- **Base rate**: 21.5% of the 1.06M eligible customer-months are in decline state.

---

## 3. Tier 1 — Correlation Geometry (`pkg_deposit_tier1.py`)

Sample: graph-visible customers, 190,095 customer-months, 4,723 customers with
sufficient history for within-customer statistics.

### 3.1 Methods
| Analysis | Question | Method |
|---|---|---|
| Cross-sectional | Which metrics separate customers on size / future outcomes? | per-month Spearman vs log balance, forward 3m Δlog, forward event flag |
| Within-customer | When THIS customer's network moves, do THEIR deposits move? | pooled Pearson on customer-demeaned pairs (≥8 obs/cust) |
| Lead–lag | Which side moves first? | demeaned corr( feature(t−k), Δlog deposit(t) ), k ∈ −6…+6; **k>0 ⇒ metric leads** |
| Event study | Do decliner trajectories separate from peers BEFORE t0? | per-month percentile-rank trajectories, t−6…t+3; controls matched on (month, NAICS2, size decile), event-free ±3m |

Stocks (strength, degree, centrality, concentration) are differenced (Δlog, consecutive
months only); turnover/contagion metrics enter as levels (they are already rates).

### 3.2 Results
**Within-customer co-movement (top |r|, all p ≪ 0.001, n ≈ 88K):**

| feature | r | reading |
|---|---|---|
| Δ strength | **+0.071** | total C2C throughput moves with deposits, same month |
| Δ n_neighbors | +0.046 | counterparty breadth moves with deposits |
| Δ avg_in_ticket | +0.041 | ticket-size growth ↔ balance growth |
| nbr_strength_trend | +0.039 | **contagion**: payer base shrinking ↔ deposits falling |
| Δ out_strength / out_degree | +0.038 / +0.034 | outbound activity tracks balances too |
| lost_payer_amount_share | **−0.030** | revenue walking out ↔ balances down |
| recurring_payer_amount_share | −0.025 | complement of new-payer money; new inflows drive deposit jumps |
| inflow_from_shrinking_share | −0.023 | inflow dependence on shrinking payers ↔ weaker deposits |

Every sign is economically coherent; nothing requires explaining away. Structure-only
metrics (hub shares, jaccards, clustering, pagerank) show ≈0 monthly co-movement —
they are slow state variables, not monthly co-movers (they may still matter as
*levels* in Tier 2).

**Lead–lag:** the relationship is predominantly **contemporaneous** (best lag 0 for
most features). Genuine but small leads: `flow_ratio` and `net_flow` at k=+1
(r ≈ +0.034) — net-flow improvement precedes deposit growth by one month. A
reversal pattern in `in_strength`/`in_volume`/`in_degree` at k=−1 (r ≈ −0.032):
deposit jumps are followed by payment pull-backs — transient inflow spikes that
mean-revert. Spike-then-fade vs. sustained growth is itself a usable feature shape.

**Event study:** 6,221 matched decline-onset events (4,708 customers) vs 30,539
matched control rows. Read `event_study.html` as: separation of the pink (event)
curve from the blue (control) curve **left of t0** = early-warning content; separation
only at/after t0 = coincident confirmation. Control curves sitting flat at ≈0.5
percentile is the expected null behavior and confirms the matching is unbiased.

### 3.3 Interpretation
1. The graph **co-moves with deposits in real time** — highly significant, coherent
   signs, across ~4.7K customers. The C2C window, despite covering a fraction of
   payment activity, sees deposit-relevant flow.
2. Monthly-resolution *leading* correlation is weak in univariate linear terms. Three
   reasons this does not close the early-warning question: (a) counterparty-gap
   attenuation biases every r toward zero; (b) 1-month deltas are the wrong timescale
   against a smooth 3v6 target — Tier 2 uses 3-month windows; (c) pooled univariate r
   is the weakest lens — multivariate lift is the real test.
3. Magnitudes (|r| 0.02–0.07) are normal for monthly financial panels; judge Tier 2 by
   classification lift, not by these correlations.

---

## 4. Tier 2 — Predictive Models (`pkg_deposit_tier2.py`)

### 4.1 Design
**Target**: decline **onset** within the next 3 months (`fwd_event_3m`), predicted from
rows that are currently eligible and *not* in decline — the model predicts the turn,
not the persistence of an existing decline.

**The central experiment is incremental value.** Three feature sets, identical rows:
| Set | Features | Role |
|---|---|---|
| **A — deposit-only** | balance level/changes, volatility, drawdown, pct-change trend | baseline: what deposit history alone predicts |
| **B — graph-only** | PKG levels, 3m changes, 3m means, deterioration persistence | what the network alone predicts |
| **C — combined** | A ∪ B | the product candidate |

**AUC(C) − AUC(A) is the deliverable.** If the graph adds nothing beyond deposit
history, C ≈ A and the story is "coincident indicator." If C > A, the network carries
information deposit trends don't yet show — the early-warning claim, quantified.

**Models**: logistic regression (standardized, class-weighted — the interpretable
floor) and `HistGradientBoostingClassifier` (native NaN handling — important given
intermittent graph presence; no imputation distortion).

**Validation**: rolling-origin folds inside the graph window (train ≤ cutoff, test the
next 3 feature-months), then the **freeze test**: final model trained ≤ 2025-08,
scored on feature months 2025-09…2025-11 whose outcomes extend into Dec 2025 – Feb 2026
— outcomes structurally outside the network data.

**Metrics**: ROC-AUC, PR-AUC (compare against the onset base rate, not 0.5),
lift@top-decile (the operational number: "flag the top 10%, catch X× their share of
declines"), plus permutation importances on the held-out folds.

### 4.2 How to read the results
- Base-rate anchor: PR-AUC must beat the printed onset rate to mean anything.
- AUC expectations: 0.60–0.65 for set C with a positive C−A gap of ≥0.02 would be a
  strong result at this coverage level; even C−A ≈ 0.01 with consistent sign across
  folds is evidence the graph leads.
- Importances: if turnover/contagion features (lost_payer, nbr_strength_trend,
  inflow_from_shrinking) rank high in set C, the mechanism story writes itself:
  *deposit attrition is preceded by the customer's payer ecosystem weakening.*
- Freeze test = the stakeholder slide: predictions made with data ending Nov 2025,
  evaluated on real 2026 outcomes.

---

## 5. Reproduction order
1. `deposit_panel_builder.run()` — Spark; writes 4 Hive tables (~min).
2. `pkg_deposit_tier1.run(spark)` — Spark pull → pandas; CSVs + HTML figs in `tier1_out/`.
3. `pkg_deposit_tier2.run(spark)` — Spark pull → pandas/sklearn; CSVs + figs in `tier2_out/`.

## 6. Open items
- Deposit table population filter (owner question) — defines generalization scope.
- `min_prior_avg = $10K` floor: calibrate against the book's balance distribution.
- `graph_presence_rate` denominator fix in `build_customer_dim` (cosmetic; numerator
  can count non-active months).
- Counterparty ingestion: rerun Tiers 1–2 unchanged when PAYS_CPTY lands; the delta
  in every statistic is the measurement-error experiment.

*Document prepared for internal use — PNC Treasury Management, Data Science*
