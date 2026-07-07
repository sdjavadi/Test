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
1. **Node-level metrics** (incl. community-related metrics *at the node level*) — per snapshot × version.
2. **Graph-level metrics** — per snapshot × version **and** for the all-snapshot **aggregated graph** (`time_key = 'AGG'`) × version.

Community-level tables, NAICS-group tables, and community/NAICS supergraph construction move to separate modules (planned next).

---

## 1. Ablation Ladder — Two Versions (data-driven)

Hub/whale nodes dominate nearly every metric, so each metric suite runs on three graphs: raw plus two exclusion versions. Exclusion sets are computed **once from the union graph aggregated over all snapshots**, so membership is stable across time.

**Why the union of degree and strength criteria?** In the 2024–2025 PKG aggregate, the P99.9 degree-hub set (~5.3K nodes) and P99.9 strength-whale set (~5.3K nodes) overlap only ~60% (union ≈ 8.7K). Degree-only exclusion leaves ~3.4K whales in place; strength-only leaves ~3.4K hubs. A union criterion removes both failure modes with one list.

| Version | Definition | Intent |
|---|---|---|
| **V0 — raw** | Full snapshot | Ground truth incl. dominance |
| **V1 — de-hubbed** | Exclude nodes with aggregate degree > P99.9 **OR** aggregate strength > P99.9 | Primary analytical graph: structure of the "ordinary" economy |
| **V2 — mega-only** | Exclude nodes with aggregate degree > P99.99 **OR** aggregate strength > P99.99 | Removes only extreme dominators (~10× smaller list); keeps ordinary hubs |

The pipeline writes `ladder_thresholds.csv` (the four dollar/degree cutoffs, for audit) and `ladder_exclusions.csv` (V1 node list with a flag marking the V2 subset).

**How to read the ladder**: V0→V2 deltas isolate the effect of mega-dominators; V2→V1 deltas isolate ordinary hubs. A metric stable across all three is a genuinely distributed property; a metric that collapses V0→V2 was carried by a handful of nodes.

---

## 2. Node-Level Metrics

Output: `node_panel.parquet`, one row per `(time_key, version, node)`; `time_key='AGG'` rows carry the same metrics computed on the aggregated graph.

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

---

## 3. Graph-Level Metrics

Output: `graph_panel.parquet`, one row per `(time_key, version)` — **all monthly snapshots plus `time_key='AGG'`** (the multi-year aggregated graph), each at V0/V1/V2.

**Why the aggregate too?** Monthly snapshots answer "what changed"; the aggregate answers "what is structurally true of the client economy overall" — stable communities, persistent hierarchy, and the reference against which monthly deviations are anomalies. Snapshot-vs-AGG gaps are themselves diagnostics (e.g. monthly reciprocity ≪ AGG reciprocity ⇒ bilateral relationships exist but alternate direction across months).

### 3.1 Scale & flow

| Metric | Definition | Questions |
|---|---|---|
| `n_nodes`, `n_edges`, `total_amount`, `total_volume`, `avg_ticket`, `density` | Basic scale | Book growth; macro trend; sparsification |
| `reciprocity_uw`, `reciprocity_w` | Edge-reverse fraction; dyad-min amount-matched share (raw) | Bilaterality of the client economy |

### 3.2 Heterogeneity & tails (raw amount)

| Metric | Definition | Questions |
|---|---|---|
| `gini_strength`, `gini_degree` | Gini of strength / degree | Concentration; **V0−V1 gap = hub contribution**, trended monthly |
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
- Version-gap series, e.g. `gini_strength(V0) − gini_strength(V1)` → hub-contribution trend
- Snapshot-vs-AGG comparisons for every graph metric

---

## 4. Implementation Inventory

| Component | Source | Weight | Notes |
|---|---|---|---|
| PageRank (raw + log1p), Louvain, core number, k-sample betweenness | **cuGraph** | per §0 | `np.log1p(series)` for cuDF compatibility |
| Weighted HITS (power iteration) | `pkg_custom_metrics.py` | log1p | cuGraph HITS is unweighted |
| Trophic levels + incoherence (MJR 2020, CG) | `pkg_custom_metrics.py` | log1p | max_iter warning benign |
| Dyad-min weighted reciprocity (graph & node) | `pkg_custom_metrics.py` | raw | hash-join on reversed edges |
| Participation, within-module z, GA roles | `pkg_custom_metrics.py` | unweighted degrees | z-divide warning benign (single-member comms) |
| Intra-community flow fractions (node) | `pkg_pipeline.py` | raw + unweighted | new in v2 |
| 4-way assortativity, Gini/Hill/top-share, rich-club, NMI/ARI | `pkg_custom_metrics.py` | raw | |
| 2-version ladder builder + exclusion export | `pkg_custom_metrics.py` | — | union of degree/strength criteria, P99.9 & P99.99 |
| Aggregated-graph build + metrics | `pkg_pipeline.py` | — | streamed groupby-sum over snapshots |
| Orchestration | `pkg_pipeline.py` | — | safe() isolation; NaN frames on per-metric failure |

**Deferred to upcoming modules**: community-level and NAICS-group metric tables; community-as-node / NAICS-as-node supergraph construction (`pkg_supergraph.py` retained for that work).

**Performance notes**: per-group `nlargest` is banned (use global sort + `groupby.head`); betweenness uses k=128 sampled sources (`BETWEENNESS_K=0` disables); trophic CG max_iter, z-score divide, and cuGraph `store_transposed` warnings are confirmed benign.
