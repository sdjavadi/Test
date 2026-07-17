# PKG Customer Roles & Peer Ranking Manifest
**Payment Knowledge Graph (PKG) — Role Taxonomies, Stability, and NAICS Peer Percentiles**
*PNC Bank · Treasury Management · Data Science · companion to `PKG_MONTHLY_METRICS_MANIFEST.md`*

---

## 0. Purpose & Position in the Stack

`pkg_roles.py` is a **post-processing layer**: it consumes the per-month node panels produced by `pkg_pipeline.py` and assigns every customer a set of **named, interpretable roles** plus **industry peer percentiles**. It does not touch raw edges.

Why named roles instead of (or before) clustering: a rule-based role is auditable ("Conduit because |flow_ratio| ≤ 0.15 and throughflow ≥ P90"), stable in meaning across months, and communicates instantly to TM sales, risk, and compliance. Data-driven clusters (cuML KMeans / HDBSCAN) are a later layer *on top*; the cross-tab between clusters and these roles is how clusters earn names.

```
pkg_pipeline.py  →  ../metrics/node/node_{YYYY-MM}.parquet
                          │
pkg_roles.py  ────────────┤  streams months chronologically,
                          ▼  small per-version state only
../metrics/roles/roles_{YYYY-MM}.parquet    one row per (version, node)
../metrics/roles/transitions.csv            stable-role transition counts
```

Run order matters: months must be processed chronologically (the module sorts the files itself). Roles are computed **per graph version** independently — a customer can be a Conduit on V0 and Peripheral on P99_9; both are true statements about different graphs (see §7).

---

## 1. Design Principles

1. **Percentile calibration.** Every threshold that depends on scale (throughflow, core_number, trophic level, payer_jaccard) is a quantile computed *within that month × version*. Fixed cutoffs are used only for quantities that are already bounded ratios (flow_ratio, shares). Consequence: roles survive the ablation ladder, dollar inflation, and book growth without recalibration.
2. **Orthogonal axes.** Five taxonomies, five separate columns. Each answers a different question; the cross-product is the segmentation. One mega-label would hide which axis moved.
3. **Hysteresis.** The `_stable` variant of each role switches only after the same new raw role is observed **two consecutive months** — threshold flicker is suppressed, real changes pass with one month's lag.
4. **NaN over nonsense.** Customers with unknown NAICS get NaN peer percentiles (never ranked against "unknown"); first-month customers get NaN switch counts; missing inputs degrade a role to its residual class, never to a wrong confident label.

---

## 2. The Five Role Taxonomies

Each taxonomy produces three columns: `{tax}` (raw, this month), `{tax}_stable` (hysteresis-filtered), `{tax}_months_in` (months in current stable role). Rules are evaluated top-to-bottom; first match wins.

### 2.1 `flow_role` — what does this customer *do* with money?

| Role | Rule | Interpretation & questions it answers |
|---|---|---|
| `terminal_payee` | in_strength > 0, out_strength ≈ 0 | Endpoint of visible flow. **In the PKG this usually means the outflow side lives outside PNC** — a prospect-intelligence flag once counterparty data lands, not necessarily a hoarder |
| `terminal_payer` | out_strength > 0, in_strength ≈ 0 | Mirror case: revenue side invisible to us |
| `conduit` | \|flow_ratio\| ≤ 0.15 AND throughflow ≥ P90 | High-volume pass-through: treasury-like, settlement-like. AML-relevant (funnel candidate); deposit balance likely low relative to flow |
| `collector` | flow_ratio > +0.5 | Net accumulator — deposits likely growing; liquidity-product target |
| `distributor` | flow_ratio < −0.5 | Net payer — payroll-like; funded from somewhere else (often outside PNC) |
| `balanced_trader` | \|flow_ratio\| ≤ 0.15 AND node reciprocity ≥ median | Genuine two-way commerce with matched dollars |
| `mixed` | residual | No dominant flow pattern |

### 2.2 `hierarchy_role` — where in the supply chain?

Terciles of `trophic_level` within month × version: `upstream_supplier` (bottom third — money flows *to* them early in chains), `midstream`, `downstream_buyer` (top third), `unknown` (trophic NaN). Question: is the customer a producer, an intermediary, or an end-buyer in the flow hierarchy — and does that position *shift* over months (a drift signal no single flow metric captures)?

### 2.3 `dependence_role` — how fragile is the revenue/spend base?

| Role | Rule | Interpretation |
|---|---|---|
| `single_relationship` | degree < 3 | **Thin-file gate, checked first.** With 1–2 counterparts, top-1 share is structurally ≈ 1.0 — concentration rules would measure arithmetic, not behavior. Production 2025-04: ~73% of the V0 book |
| `infra_dependent` | hub_in_share > 0.7 | Revenue arrives via registry hubs (processors/payroll/settlement). Checked before customer-dependence so the two risk stories never blur |
| `anchor_dependent` | top1_in_share > 0.7 | One *customer* is the revenue base; anchor loss = attrition precursor |
| `captive_payer` | top1_out_share > 0.7 | Spend concentrated on one payee; supply-chain fragility |
| `diversified` | hhi_in ≤ median AND hhi_out ≤ median (medians over degree ≥ 3 population) | Balanced book both directions |
| `moderate` | residual | — |

### 2.4 `embeddedness_role` — position in the network fabric?

| Role | Rule | Interpretation |
|---|---|---|
| `peripheral` | degree < 3 OR core_number ≤ P25 — **checked first** | Edge of the visible economy (often: most of their graph is outside PNC). Must precede embedded_local: a degree-1 node has frac_intra = 1.0 trivially and previously flooded that role (74% of the book) |
| `connector` | participation_coef ≥ 0.62 | Edges spread across communities (GA-consistent cutoff); cross-segment broker |
| `local_anchor` | core_number **>** P90 (strict, computed over degree ≥ 3) AND frac_intra_edges_w ≥ 0.7 | Deep in the dense core *and* dollar-committed to its own community. Strict inequality prevents integer-tie inflation (18% qualified under ≥ at production) |
| `embedded_local` | frac_intra_edges_w ≥ 0.7 | Dollar-committed to its community without being a hub of it |
| `intermediate` | residual | — |

Complements `ga_role` (already in the node panel): GA is degree-topological, this is flow-weighted and community-dollar based.

### 2.5 `dynamics_role` — what is *happening* to this customer?

| Role | Rule | Interpretation |
|---|---|---|
| `newcomer` | months_since_first_seen ≤ 2 | Too young to judge; excluded from drift alerts |
| `intermittent` | activity_gap > 1 | Reappearing after absence; sporadic biller pattern |
| `bleeding` | (prior-month in_degree ≥ 3) AND [lost_payer_amount_share ≥ 0.4 OR (strength_mom ≤ −0.3 AND payer_jaccard ≤ P25)] | **Revenue walking out** — the role most directly predictive of churn; a TM call list, not just a feature |
| `expanding` | (in_degree ≥ 3) AND new_payer_amount_share ≥ 0.4 AND strength_mom > 0.1 | Growing book from new relationships |
| `steady` | payer_jaccard ≥ median AND \|strength_mom\| < 0.35 | Stable counterpart set, stable dollars |
| `variable` | residual | Moving but not classifiable |

`strength_mom` = log((strength_t + 1)/(strength_{t−1} + 1)), computed by the module and included in the output.

---

## 3. Stability Metrics (per row)

| Column | Definition | Use |
|---|---|---|
| `role_stability_score` | Mean of the five `_months_in` counters | Average role tenure |
| `role_stability_norm` | Score ÷ months_active, clipped [0, 1] | Tenure-fair stability: a 2-month customer can't hold a 10-month role. Near 1 = behaviorally locked in; low despite long tenure = chronic flux |
| `n_role_switches` | # taxonomies whose *stable* role changed this month (0–5); NaN for first-ever appearance and month 1 | Instantaneous drift intensity. Because it counts *stabilized* switches, ≥2 means multiple sustained behavioral changes at once — a far stronger drift event than any single metric moving. Natural trigger for Behavioral Drift (roadmap #15) |

Recommended churn features: `role_stability_norm` level + trailing 3–6-month rolling sum of `n_role_switches`. Instability tends to *precede* the dollar decline that level-based features catch later. Customers with chronically high switch counts are a distinct risk population worth profiling before even asking *which* roles they cycle through.

---

## 4. NAICS Peer Percentiles

**Adaptive granularity**: each customer is ranked within the finest NAICS level — naics4 → naics3 → naics2 — whose peer group has ≥ `MIN_PEERS` (default 30) members that month. `peer_level` (4/3/2, 0 = none adequate) and `peer_size` record the group actually used. Unknown-NAICS customers get NaN throughout.

| Column | Question it answers |
|---|---|
| `strength_pctl_naics`, `in_strength_pctl_naics`, `degree_pctl_naics` | How big / connected **for its industry**? Raw values are incomparable across sectors; percentiles are comparable everywhere |
| `net_flow_pctl_naics` | Collector-ness relative to industry norm — mild accumulation is unremarkable in retail, a flag in a normally net-paying sector |
| `hhi_in_pctl_naics` | Concentration vs. industry norm — the same HHI means opposite things in trucking vs. restaurants |
| `payer_jaccard_pctl_naics` | Retention vs. peers — "is this churn normal for the business they're in?" |
| `strength_mom_pctl_naics` | **Momentum vs. peers: is the customer shrinking, or is its whole industry shrinking?** Losing share within a healthy sector = competitive problem (churn signal); shrinking with the sector = macro. The single most decision-relevant peer feature |
| `strength_vs_peer_median` | log1p gap to peer median strength — interpretable magnitude alongside the rank |

The `naics2/3/4` ID columns are carried through so any *other* node metric can be peer-ranked later without re-joining the node panel.

---

## 5. Transitions Output

`transitions.csv`: `(version, taxonomy, from_role, to_role, n)` accumulated over the full run, on **stabilized** roles. This is the drift engine and the direct input for a Sankey panel in the Streamlit app.

High-signal cells to watch:

| Transition | Reading |
|---|---|
| steady → bleeding | The churn alarm; expect this cohort to dominate attrition within 1–2 quarters |
| balanced_trader → conduit | Flow pattern shifted from commerce to pass-through — AML review candidate |
| anchor_dependent → diversified | De-risking success story (or anchor already lost — check `bleeding`) |
| peripheral → local_anchor / embedded_local | Growth account deepening into an ecosystem; TM cross-sell moment |
| any → intermittent | Early disengagement pattern |

---

## 6. Output Schema Summary

`roles_{YYYY-MM}.parquet` — one row per (version, node), 36 columns: `time_key`, `version`, `node`; 5 × (`{tax}`, `{tax}_stable`, `{tax}_months_in`); `role_stability_score`, `role_stability_norm`, `n_role_switches`; `strength_mom`; `naics2/3/4`; `peer_level`, `peer_size`, 8 × `*_pctl_naics`, `strength_vs_peer_median`. Strings as `string` dtype, floats as float32, consistent schema every month (peer columns exist as NaN even when no group qualifies).

---

## 7. Caveats & Interpretation Rules

1. **Burn-in.** `dynamics_role` needs ~3 months of history (everyone is `newcomer` before that); `n_role_switches` starts month 2; stability scores mature over ~6 months. Treat the first quarter of the panel as warm-up, and exclude it from transition-rate baselines.
2. **Resume caveat** (inherited from the pipeline): tracker state isn't persisted, so a resumed run restarts momentum/turnover-dependent roles with one NaN month.
3. **Thresholds are starting points, not truth.** The 0.15 conduit band, 0.7 dependence cutoffs, 0.4 bleeding trigger are defensible defaults. After the first production run, check each role's population share per version in the output; a role holding >50% or <0.1% of customers isn't segmenting and should be recalibrated. Any recalibration must be **documented here** — role definitions are part of the audit trail.
4. **Version semantics.** Roles on V0 describe the customer *including* infrastructure flow; on P99_9 they describe the *discretionary* relationship book. `terminal_*` and `peripheral` on de-hubbed versions are partly artifacts of hub removal — prefer V0 for flow/dependence roles, de-hubbed versions for embeddedness and dynamics. When in doubt, report the version next to the role.
5. **Roles are per-graph statements, not identities.** "Conduit on V0, peripheral on P99" is not a contradiction — it says the customer's volume is infrastructure-mediated. The *disagreement between versions* is itself a feature.
6. **Terminal roles will shrink dramatically** once PAYS_CPTY/counterparty data lands — most `terminal_*` customers are terminal only within PNC's visibility perimeter. Expect a large re-labeling event at that data milestone; version the role panel accordingly.


---

## 8. Recalibration Log

| Date | Change | Production evidence |
|---|---|---|
| 2026-07 (post first prod run, 2025-04 V0, ~2.23M customers) | Added thin-file gate (`THIN_DEGREE = 3`): dependence → `single_relationship`; dynamics bleeding/expanding require ≥3 payers (prior/current month); percentile thresholds computed over the degree ≥ 3 population | `diversified` had collapsed to 0.7% while captive/anchor/infra absorbed 94% — top-1 share is structurally ≈1.0 for 1–2-counterpart customers |
| 2026-07 | Embeddedness rule order: `peripheral` first; `local_anchor` core cutoff changed from ≥P90 to strict >P90 over eligible population | `embedded_local` at 74.7% (degree-1 nodes have frac_intra = 1.0 trivially); `local_anchor` at 18% vs. the ≤10% the P90 rule intends (integer ties) |
| 2026-07 | `steady` momentum band widened from \|mom\| < 0.2 to < 0.35 | `variable` residual at 45% — the band was tighter than real monthly payment volatility |
| — (documented, unchanged) | `flow_role` terminal shares (76% at prod) are structural, not a threshold artifact — most customers' other side lives outside PNC; benchmark for the counterparty-data milestone. `hierarchy_role` terciles are flat by construction: population shares carry no signal, only per-customer transitions do; trophic solve succeeded for all 2.23M nodes | — |
