# PKG Edge Rhythm — EDA & Calibration Spec (v0.2 — settled)

**Purpose:** One-time retrospective EDA pass over all 23 monthly snapshots (Jan 2024 = month 0 … Nov 2025 = month 22) to (a) characterize edge periodicity, (b) calibrate classification thresholds against real distributions, and (c) decide whether the regular-backbone layer is worth building. Feeds a future `pkg_edge_rhythm.py` pipeline module (**as-of-month**, minimum 6-month lookback; this EDA is **full-window retrospective** — calibration only, never shipped as historical labels).

**Compute:** cuDF on the GPU server *(settled — decision 5)*. Est. peak < 15 GB, well within 60 GB.

**Companion notebook:** `pkg_edge_rhythm_eda.ipynb` / `.py` (py:percent source of truth).

---

## v0.1 → v0.2 changes (decision record)

| # | Question | Resolution |
|---|---|---|
| 1 | Modal band width | **Empirical.** Section 4 of the notebook pools \|amount − median\| / median across active months of eligible edges (months_present ≥ 6 — shorter histories bias rel_dev toward 0), plots count- and dollar-weighted distributions with a log-y axis, reports the exact-zero spike separately, and suggests a valley. `BAND` is a parameter set **after inspection**; the grid {0.01, 0.025, 0.05, 0.10} is computed regardless and drives the sensitivity table. |
| 2 | Distinct-amount rounding | **Nearest dollar** (locked). |
| 3 | Minimum history | **6 months** (locked). |
| 4 | Hub check inputs | **Reuse existing 23-snapshot aggregate ablation node lists** (P99, P99.9) verbatim (locked). |
| 5 | Engine | **cuDF** (locked). PySpark only if a future run must colocate with Hive. |

---

## Stage 0 — Stacked edge-history frame (`edge_month`)

Build once, cache as parquet. Every later stage reads this.

| Column | Dtype | Notes |
|---|---|---|
| `edge_key` | int64 | `src_id · N_NODES + dst_id`; recover endpoints by divmod. Factorize strings early — never groupby on string IDs on GPU. |
| `month_idx` | int8 | 0–22. Jan 2024 = 0. **Chronological file ordering is asserted and printed for eyeball verification.** |
| `amount` | float64 | Monthly aggregate dollars. |
| `volume` | int32 | Monthly transaction count. |

Node dimension `node_dim(node, node_id)` cached alongside; `N_NODES` in `stage0_meta.json`. NAICS attached only at analysis time via a second lightweight CSV pass, taking the **most recent non-null** value per node.

**Defensive aggregation:** duplicate (edge, month) rows summed and logged. **Asserts:** amounts > 0, volume ≥ 1, month_idx ∈ [0, 22], 23 files found, no unmatched node ids after factorization.

---

## Stage 1 — Per-edge aggregates (`edge_rhythm_agg`)

One row per edge. Plain groupby pass + window pass (global sort → diff → edge-boundary mask → cumsum run IDs; same RLE pattern as community lifecycle logic; zero iterrows; `vol_mode` and phase maxima via global sort + drop_duplicates, never groupby.nlargest).

**Presence & timing:** months_present, first/last_month, lifespan, `support_lifespan` (regularity while alive), `support_window` (maturity), months_since_last.

**Gaps & streaks:** max/mean/std inter-arrival, n_runs, longest_streak, current_streak.

**Amount:** amt_total/mean/median/std/min/max, MAD, `amt_cv`, `amt_rcv` = MAD/median (primary stability measure), `modal_amt_share` at chosen BAND + grid columns, `n_distinct_amt` (nearest-dollar), cv_h1/cv_h2 (step-up fingerprint).

**Volume:** vol_mode + vol_mode_share, vol_mean/std, share_vol1.

**Phase (k = 2, 3):** phase_share_k; interpretation gate months_present ≥ 6 ∧ support_lifespan ≤ 0.75. No spectral methods — 23 points.

---

## Stage 2 — Provisional labels (plot coloring; thresholds are the EDA output)

`ended_flag` (months_since_last ≥ 3) and `broken_established` (∧ longest_streak ≥ 6) kept **orthogonal** to rhythm class. `rhythm_class` priority: TOO_YOUNG → ONE_SHOT → FIXED_RECURRING (support ≥ 0.9 ∧ modal ≥ 0.7) → VARIABLE_RECURRING (support ≥ 0.8) → PERIODIC_2/3 (phase ≥ 0.9 within gate) → INTERMITTENT. Implemented as ascending-priority mask overwrites on an int8 code column.

---

## Stage 3 — EDA deliverables

1. **Scale facts** — distinct edges/nodes, rows, dollars, class counts & dollar shares → `scale_facts.json`.
2. **Rel-dev distribution & band selection** *(new in v0.2)* — decides BAND.
3. **Survival curves** — edges% and dollars% vs support ≥ t, both support definitions, eligible vs all overlaid. Target claim to test: "~20% of edges, ~50%+ of dollars."
4. **Joint distribution** — support_lifespan × modal_amt_share heatmaps, count- and dollar-weighted. Do FIXED/VARIABLE separate or smear?
5. **Hub check** — survival with vs without P99 / P99.9 ablation; % of recurring dollars kept; top-20 node concentration in the high-support mass. Backbone-viability verdict.
6. **NAICS conditioning** — recurring dollar share by payer naics2.
7. **Volume fingerprints** — vol_mode distribution inside FIXED candidates (sub-monthly rhythm is *inferred*, never proven, at monthly granularity).
8. **Phase scan** — PERIODIC_2/3 counts and dollar mass; keep-or-drop call.
9. **Threshold sensitivity table** — support grid × band grid (exact per-band modal shares, MODAL_CUT = 0.7 held fixed) → `threshold_sensitivity.csv`.

Screenshots of 2–5 go into the manifest as the calibration record.

## Exit checklist

1. BAND set from deliverable 2; Sections 5–13 re-run.
2. FIXED/VARIABLE support thresholds chosen from deliverables 3–4.
3. Backbone verdict from deliverable 5.
4. PERIODIC keep/drop from deliverable 8.
5. Manifest updated; thresholds carried into `pkg_edge_rhythm.py` (as-of-month + 6-month lookback).
