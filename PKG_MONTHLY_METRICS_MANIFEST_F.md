# PKG Monthly Snapshot Metrics Manifest — v2
**Payment Knowledge Graph (PKG) — Customer-to-Customer Monthly Network Analysis**
*PNC Bank · Treasury Management · Data Science*

---

## 0. Scope & Conventions

- **Input**: monthly snapshots `../data/cust_YYYY-MM.csv` with schema
  `source, source_name, source_naics, amount, volume, dest, dest_name, dest_naics`
- **Edge weight = `amount` only.** `volume` is descriptive, never a weight.
- **Directed**: `source → dest` = source *paid* dest.
- **SCC / WCC excluded** (memory budget).
- **Weight-scale policy** (per metric, chosen by application):
  - **log1p(amount)** for spectral/iterative metrics — HITS, `pagerank_logw`, trophic levels, Louvain. Rationale: raw dollar tails span 8+ orders of magnitude; without the log transform, whale edges dominate the spectrum and the metric degenerates into "distance from the biggest payment."
  - **raw amount** for flow/accounting metrics — strengths, net flow, HHI, top-k shares, reciprocity, rich-club, Gini/Hill. Rationale: these must reflect real dollar magnitudes.
  - **both** where the comparison is itself informative: `pagerank_raw` vs `pagerank_logw`. Nodes ranked high on raw but not log are pure whale beneficiaries; high on log but not raw are broadly-connected but dollar-modest. The rank divergence is a metric in its own right.
- **Scale envelope**: 3–5M nodes/edges per snapshot; single GPU ≤ 60 GB.

### Deliverable focus (this version)
1. **Node-level metrics** (incl. community-related metrics *at the node level* and relationship-dynamics metrics for churn/deposit modeling) — per snapshot × version.
2. **Graph-level metrics** — per snapshot × version.

**Output is streamed per month** — each month is computed, written, and freed (no in-memory accumulation, no aggregated-graph pass; both caused OOM at production scale):

```
../metrics/node/node_{YYYY-MM}.parquet   # ONE file per month, all versions stacked (uniform 57-col schema, float32)
../metrics/graph/graph_{YYYY-MM}.csv               # one row per version
../metrics/ladder_thresholds.csv                   # audit: the four cutoffs
../metrics/ladder_exclusions.csv                   # hub registry (P99 list + per-tier flags)
```

Combine at the end with `pd.concat(map(pd.read_parquet, glob("../metrics/node/*.parquet")))`. The run is **resumable**: existing (month, version) node files are skipped. Note that after a resume, cross-month metrics (turnover, tenure, NMI-vs-prev) restart NaN on the first processed month since tracker state isn't persisted.

Community-level tables, NAICS-group tables, and community/NAICS supergraph construction move to separate modules (planned next).

---

## 1. Ablation Ladder — Percentile Tiers (data-driven)

Hub/whale nodes dominate nearly every metric, so each metric suite runs on the raw graph plus three exclusion tiers. Exclusion sets are computed **once from the union graph aggregated over all snapshots**, so membership is stable across time.

**Why the union of degree and strength criteria?** In the 2024–2025 PKG aggregate, the P99.9 degree-hub set (~5.3K nodes) and P99.9 strength-whale set (~5.3K nodes) overlap only ~60% (union ≈ 8.7K). Degree-only exclusion leaves ~3.4K whales in place; strength-only leaves ~3.4K hubs. A union criterion removes both failure modes with one list.

| Version | Definition | Intent |
|---|---|---|
| **V0** | Full snapshot | Ground truth incl. dominance |
| **P99** | Exclude aggregate degree > P99 **OR** strength > P99 (~1% of nodes) | Aggressive de-hubbing: the "small-customer economy"; sensitivity bound |
| **P99_9** | Same at P99.9 | Primary analytical graph; also serves as the hub registry for hub-exposure metrics |
| **P99_99** | Same at P99.99 | Removes only extreme dominators; keeps ordinary hubs |

Tiers are configurable via `build_ladder(paths, percentiles=(...))`; version labels are derived from the percentiles.

The pipeline writes `ladder_thresholds.csv` (the six dollar/degree cutoffs, for audit) and `ladder_exclusions.csv` (the P99 node list with `in_P99_9` / `in_P99_99` flags marking the stricter subsets).

**How to read the ladder**: V0→P99_99 deltas isolate mega-dominators; P99_99→P99_9→P99 deltas peel off successively more ordinary hubs. A metric stable across all four is a genuinely distributed property; a metric that collapses already at V0→P99_99 was carried by a handful of nodes.

---

## 2. Node-Level Metrics

Output: `../metrics/node/node_{YYYY-MM}.parquet` — one file per month containing ALL versions stacked; one row per (version, node).

### 2.1 Flow magnitude & balance (raw amount)

| Metric | Definition | Example | Questions it answers |
|---|---|---|---|
| `in_degree`, `out_degree`, `degree` | # distinct payers / payees / both | Paid by 40, pays 12 | Relationship breadth; payer-count drops signal churn |
| `in_strength`, `out_strength`, `strength` | Σ amount received / sent / both | $8.2M in, $7.9M out | Dollar throughput; TM sizing; deposit relevance |
| `log_strength` | log1p(strength) | — | Scale-stabilized companion for modeling/plotting |
| `net_flow` | in − out strength | +$300K/mo | Net **source**, **sink**, or **conduit**? |
| `flow_ratio` | net_flow / strength ∈ [−1,1] | 0.02 | Near-0 with high strength = pass-through/funnel candidate |
| `throughflow` | min(in_strength, out_strength) | $7.9M | Money *flowing through* the node — key funnel signal |
| `in_volume`, `out_volume`, `avg_in_ticket`, `avg_out_ticket` | Transaction counts; strength/volume | Avg incoming $45K | Ticket-size drift precedes behavior change |

### 2.2 Concentration (raw amount)

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `hhi_out`, `hhi_in` | Herfindahl of amounts across payees / payers | 0.93 → one payee gets nearly all | Beneficiary/customer-base concentration; layering leg |
| `top1_out_share`, `top3_out_share` (and `_in_`) | Largest 1 / 3 counterpart shares | 0.97 | Interpretable companion to HHI for business audiences |

### 2.3 Spectral / centrality

| Metric | Weight | Definition | Questions |
|---|---|---|---|
| `pagerank_raw` | raw | Weighted PageRank, damping 0.85 | Who receives the biggest dollar flow-mass? |
| `pagerank_logw` | log1p | Same on log weights | Who is important by breadth-adjusted flow? **Rank gap vs raw = whale-dependence of importance** |
| `hits_hub_w` | log1p | Weighted HITS hub: h ← W·a (custom power iteration) | Who *funds* the important receivers? Distribution-side importance |
| `hits_auth_w` | log1p | Weighted HITS authority: a ← Wᵀ·h | Who is *funded by* important senders? Collection-side importance |
| `betweenness_approx` | log1p | Approx betweenness, k=128 sampled sources | Brokerage; layering-chain middlemen |
| `core_number` | unweighted | k-core index | Peripheral vs. dense-core economy membership |
| `trophic_level` | log1p | MacKay–Johnson–Rodgers (2020) position in the upstream→downstream hierarchy (CG solve) | Where does the customer sit in the supply chain? Level *shifts* over months are drift signals |

### 2.4 Local structure

| Metric | Weight | Definition | Questions |
|---|---|---|---|
| `reciprocity_node_w` | raw | Σⱼ min(wᵢⱼ,wⱼᵢ) / Σⱼ(wᵢⱼ+wⱼᵢ) | Mutual trading vs. one-way relationships; round-trip patterns |

### 2.5 Community-related metrics at the node level (Louvain on log1p)

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `community_id` | Louvain label | — | Segment membership; membership *changes* MoM are drift events |
| `within_module_z` | z-score of within-community degree vs. community peers | z=4.2 | Local hub of its own segment? |
| `participation_coef` | 1 − Σ_c (k_i,c/k_i)² over communities touched | 0.72 | Connector vs. provincial node; cross-segment brokerage |
| `ga_role` | Guimerà–Amaral role R1–R7 from the (z, P) plane | R6 connector hub | Canonical role taxonomy; **role transitions** are strong behavioral-drift flags |
| `frac_intra_edges_uw` | Share of the node's edge *count* staying inside its community | 0.64 | Link-wise dependence on the local ecosystem |
| `frac_intra_edges_w` | Share of the node's *amount* staying inside its community | 0.85 | Dollar-wise dependence; divergence from unweighted version shows whether the big money stays home or crosses out |
| `naics_participation` | Participation coefficient with the NAICS partition swapped in | 0.9 | Industry diversification of the payment book |

### 2.6 Relationship dynamics (cross-month; NaN in each version's first month)

The state metrics above describe the graph; churn lives in the *flux*. These compare each node's counterpart sets and flows against the previous month — the primary feature family for churn/deposit prediction.

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `n_payer_new / _lost / _retained` | Counterpart-set churn on the inbound side vs. prev month (`payee` variants for outbound) | Lost 3 of 5 payers | Relationship churn ahead of dollar churn |
| `payer_jaccard`, `payee_jaccard` | retained / (new + lost + retained) | 0.33 | Stability of the counterpart book; low + falling = disengagement |
| `lost_payer_amount_share` | Prev-month amount from now-lost payers / prev in_strength | 0.98 → nearly all revenue sources gone | **Revenue walking out the door** — expected top churn predictor |
| `new_payer_amount_share` | Current amount from first-time payers / current in_strength | 0.4 | Book renewal vs. dependence on legacy payers |
| `top_payer_same` | 1 if this month's #1 payer is last month's #1 | 0 | Anchor-payer loss — classic attrition precursor |
| `top_payer_share_delta` | Δ top-1 payer share MoM | −0.3 | Anchor relationship weakening even before it disappears |
| `months_since_first_seen`, `months_active`, `activity_gap` | Tenure; months present; months since previously active (1 = consecutive) | gap = 3 | Recency/tenure — trivial to compute, disproportionately predictive |
| `nbr_strength_trend` | Inflow-weighted mean log MoM strength ratio of the node's payers | −0.4 | **Contagion**: is the customer's revenue base itself shrinking? Graph-native — invisible to tabular systems |
| `inflow_from_shrinking_share` | Share of inflow from payers whose own strength fell >20% MoM | 0.6 | Distress exposure through the payment network |

### 2.7 Hub exposure (computed on raw edges, attached to every version)

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `hub_in_share`, `hub_out_share` | Share of the node's raw in/out amount exchanged with ladder-registry (P99_9) nodes | 0.95 | Structural fragility: one mega-hub feeds this customer; hub relationship changes propagate directly |

---

## 3. Graph-Level Metrics

Output: `../metrics/graph/graph_{YYYY-MM}.csv`, one row per version per month.

### 3.1 Scale & flow

| Metric | Definition | Questions |
|---|---|---|
| `n_nodes`, `n_edges`, `total_amount`, `total_volume`, `avg_ticket`, `density` | Basic scale | Book growth; macro trend; sparsification |
| `reciprocity_uw`, `reciprocity_w` | Edge-reverse fraction; dyad-min amount-matched share (raw) | Bilaterality of the client economy |

### 3.2 Heterogeneity & tails (raw amount)

| Metric | Definition | Questions |
|---|---|---|
| `gini_strength`, `gini_degree` | Gini of strength / degree | Concentration; **V0−P99_9 gap = hub contribution**, trended monthly |
| `hill_alpha_strength` | Hill tail index, top 5% | Fat-tail severity; falling α = growing whale dominance |
| `top_0.1pct_amount_share` | Amount share of top 0.1% nodes | Board-friendly concentration statement |
| `rich_club_w_0.01`, `rich_club_w_0.001` | Weighted rich-club at top 1% / 0.1% strength ranks | Do the biggest players reserve their strongest ties for each other? |

### 3.3 Mixing, hierarchy, segmentation

| Metric | Weight | Definition | Questions |
|---|---|---|---|
| `assort_in_in/in_out/out_in/out_out` | raw (log10 strengths) | Four-way directed weighted assortativity | Do big senders pay big receivers? Regime-shift fingerprint |
| `trophic_incoherence_F0` | log1p | Σw(hⱼ−hᵢ−1)²/Σw | Hierarchical (F0→0, feed-forward supply chains) vs. loopy (F0→1, circular flows) |
| `modularity_Q`, `n_communities`, `community_size_gini` | log1p | Louvain partition shape | Segmentation strength; fragmentation vs. consolidation |
| `nmi_vs_prev`, `ari_vs_prev` | — | Partition similarity vs. previous month (same version) | Is the segmentation stable enough to trend? Gates all community analytics |

### 3.4 Temporal panel usage

Long-format panels keyed by `(time_key, version, ...)` enable:
- MoM/YoY deltas, per-node rolling z-scores → drift flags feeding Rail Shift Monitor / Behavioral Drift (roadmap #2, #15)
- Rank stability (Spearman of consecutive top-k) for `pagerank_*`, `hits_*`, `strength`
- Version-gap series, e.g. `gini_strength(V0) − gini_strength(P99_9)` → hub-contribution trend
- Peer-relative and seasonality features (YoY ratios, 12-month autocorrelation) are derived downstream in the feature builder from the combined per-month files

---

## 4. Implementation Inventory

| Component | Source | Weight | Notes |
|---|---|---|---|
| PageRank (raw + log1p), Louvain, core number, k-sample betweenness | **cuGraph** | per §0 | `np.log1p(series)` for cuDF compatibility |
| Weighted HITS (power iteration) | `pkg_custom_metrics.py` | log1p | cuGraph HITS is unweighted |
| Trophic levels + incoherence (MJR 2020, CG) | `pkg_custom_metrics.py` | log1p | max_iter warning benign |
| Dyad-min weighted reciprocity (graph & node) | `pkg_custom_metrics.py` | raw | hash-join on reversed edges |
| Participation, within-module z, GA roles | `pkg_custom_metrics.py` | unweighted degrees | z-divide warning benign (single-member comms) |
|  Intra-community fractions; turnover/tenure/contagion (TemporalTracker); hub exposure | `pkg_pipeline.py` | raw + unweighted | new in v2 |
| 4-way assortativity, Gini/Hill/top-share, rich-club, NMI/ARI | `pkg_custom_metrics.py` | raw | |
| 2-version ladder builder + exclusion export | `pkg_custom_metrics.py` | — | union of degree/strength criteria per tier: P99, P99.9, P99.99 |
| Orchestration | `pkg_pipeline.py` | — | safe() isolation; NaN frames on per-metric failure |

**Deferred to upcoming modules**: community-level and NAICS-group metric tables; community-as-node / NAICS-as-node supergraph construction (`pkg_supergraph.py` retained for that work).

**Performance notes**: per-group `nlargest` is banned (use global sort + `groupby.head`); betweenness uses k=128 sampled sources (`BETWEENNESS_K=0` disables); trophic CG max_iter, z-score divide, and cuGraph `store_transposed` warnings are confirmed benign.
