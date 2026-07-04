# PKG Monthly Snapshot Network Metrics Manifest

**Payment Knowledge Graph (PKG) — PNC Treasury Management, Data Science**
Scope: monthly customer-to-customer payment snapshots.
Edge schema: `source, source_name, source_naics, amount, volume, dest, dest_name, dest_naics`.
Edge weight: **amount only** (`volume` is descriptive, never a weight).
Scale assumption: 3–5M nodes and edges per snapshot; single GPU, ≤60 GB memory.
Excluded by design: SCC/WCC-based metrics (flow hierarchy is measured via trophic analysis instead).

---

## 0. Conventions

- **G** = directed multigraph collapsed to a simple directed weighted graph per snapshot: parallel edges `(u,v)` are summed on `amount` and `volume`, with `edge_count` retained.
- **w(u,v)** = total amount from u to v in the month.
- **s_in(v), s_out(v)** = weighted in/out strength (sum of amounts).
- **d_in(v), d_out(v)** = distinct in/out counterparties (pair-aggregated degree). Raw edge count is kept separately as `tx_edge_count` and used only for hub-exclusion thresholds.
- **Weight transform policy**:
  - *Flow & structural metrics* (net flow, HHI, mixing, conductance, reciprocity): **raw amount**.
  - *Spectral / iterative metrics* (PageRank, HITS, eigenvector, Katz): **log1p(amount)** to tame heavy tails; raw-amount variants optionally computed for sensitivity checks.
- **Community detection**: Louvain on the amount-weighted, **symmetrized** graph (w_sym(u,v) = w(u,v) + w(v,u)), log1p-transformed weights, fixed random seed per snapshot for cross-month comparability, then partition-matched across months (Section 5).
- **Self-loops**: removed before all metrics; self-payment totals stored as node attributes (`self_amount`, `self_volume`) feeding the self-payment detection pipeline.

---

## 1. Snapshot Version Ladder (Exclusion Ablation)

Every metric below is computed on **five versions** of each snapshot. The ladder is an ablation sequence: comparing V0→V4 attributes each metric shift to *connectivity hubs* vs. *amount whales*.

Thresholds are **data-driven per snapshot**, from the empirical distributions:

| Threshold | Definition | Typical anchor |
|---|---|---|
| `T_deg_broad` | total degree (raw edge count) at P99.9 | catches payroll processors, platforms |
| `T_deg_strict` | total degree at P99.99 | catches only extreme hubs |
| `T_str_broad` | total strength (s_in + s_out) at P99.9 | catches amount whales |
| `T_str_strict` | total strength at P99.99 | catches only mega-whales |

Percentiles are anchors, not law: inspect the log-log rank plot each month and snap the threshold to the visible knee if it deviates from the percentile. Persist chosen thresholds in the snapshot metadata table so months are auditable.

| Version | Removal rule | Interprets |
|---|---|---|
| **V0** | none (raw) | ground truth incl. hubs/whales |
| **V1** | degree > `T_deg_strict` | effect of extreme connectivity hubs |
| **V2** | strength > `T_str_strict` | effect of extreme amount whales |
| **V3** | degree > `T_deg_strict` OR strength > `T_str_strict` | combined-strict core |
| **V4** | degree > `T_deg_broad` OR strength > `T_str_broad` | combined-broad "mid-market" graph |

**Cross-version diagnostics** (per snapshot):
- Spearman rank stability of PageRank / HITS / strength between versions (which rankings are hub-driven artifacts?).
- Node/edge/amount retention per version.
- Metric deltas: e.g. if modularity jumps V0→V1 but not V0→V2, community structure was smeared by connectivity hubs, not whales.

*Question answered:* "Is this signal real economic structure, or an artifact of a few payroll processors / clearing-like whales?"

---

## 2. Node-Level Metrics

### 2.1 Flow magnitude & direction

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `s_in`, `s_out` | sum of incoming / outgoing amounts | s_in=$4.2M, s_out=$3.9M | How big is this customer's payment footprint? |
| `net_flow` | s_in − s_out | +$300K | Net accumulator (sink) or distributor (source)? |
| `flow_ratio` | s_in / (s_in + s_out) ∈ [0,1] | 0.52 | Direction of the business: 1≈pure collector, 0≈pure payer, 0.5≈balanced pass-through/operating. |
| `throughflow` | min(s_in, s_out) | $3.9M | How much money genuinely *transits* the node — the core layering-relevance quantity. |
| `pass_through_index` | throughflow / max(s_in, s_out) | 0.93 | High + high throughflow = conduit; funnels and layering way-stations score near 1. |
| `d_in`, `d_out` | distinct counterparties in/out | 480 in, 3 out | Breadth of the ecosystem on each side. |
| `tx_edge_count` | raw monthly edge count | 12,400 | Activity intensity; hub-threshold input only. |
| `avg_ticket_in/out` | s / distinct-pair volume | $8,750 | Retail-like (many small) vs wholesale-like (few large)? |

### 2.2 Concentration

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `hhi_in`, `hhi_out` | Herfindahl over amount shares across in/out counterparties | hhi_out=0.97 | Is the money going to essentially one place? Funnel signature = high d_in, hhi_out→1. |
| `top1_share_in/out` | largest single counterparty's amount share | 0.98 | Same, more interpretable for analysts. |
| `in_out_asym` | d_in vs d_out imbalance: (d_in−d_out)/(d_in+d_out) | +0.985 | N-to-1 collection vs 1-to-N distribution shape. |

### 2.3 Centrality (spectral, log1p weights)

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `pagerank` | weighted PageRank on directed G | rank 212 of 4.1M | Where does money *land* if it random-walks the network? Structural importance as a receiver. |
| `pagerank_rev` | PageRank on reversed G | — | Importance as an *originator* — whose outflows feed the network? |
| `hits_hub`, `hits_auth` | **weighted HITS** (custom, Section 6.1) | hub=0.004, auth=1e-7 | Hubs = pay many strong authorities (disbursers, payroll-like); Authorities = receive from many strong hubs (collectors, platforms). The hub/auth split is the natural directed complement to PageRank and separates "big payer to important receivers" from "big receiver from important payers." |
| `eigenvector_sym` | eigenvector centrality on symmetrized graph | — | Embeddedness in the dense weighted core, direction-agnostic. |
| `betweenness_approx` | k-sampled betweenness (k≈2–5K pivots) | — | Broker positions bridging otherwise separate money circuits. Expensive; compute on V3/V4 only. |
| `katz` *(optional)* | Katz with small α | — | Multi-hop reach when PageRank's damping is too aggressive; sensitivity companion. |

### 2.4 Flow hierarchy (trophic analysis — SCC-free)

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `trophic_level` | MacKay–Johnson–Rogers (2020) levels: solve Λh = v with weighted Laplacian (Section 6.2) | h=0.4 (near source) vs h=3.2 (deep sink) | Where does this node sit in the *direction of money*? Sources (payers-in-chief), mid-chain conduits, terminal sinks. |
| `trophic_incoherence_local` | node's contribution to F₀: amount-weighted variance of (h_dest − h_src − 1) over incident edges | — | Does money around this node flow "downhill" cleanly or churn in loops? Loop-heavy neighborhoods = circulation / potential round-tripping. |

### 2.5 Reciprocity & local structure

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `recip_w` | weighted node reciprocity: Σ_v min(w(u,v), w(v,u)) / Σ_v max(w(u,v), w(v,u)) | 0.31 | How bidirectional are its relationships? Trading partners vs one-way payer/payee. |
| `recip_u` | unweighted: fraction of neighbors with edges both ways | 0.12 | Same, structural. |
| `clustering_sym` | local clustering coefficient on symmetrized graph | 0.04 | Do its counterparties also pay each other? Supply-chain cliquishness vs star topology. |
| `core_number` | k-core index (symmetrized, unweighted) | 14 | Depth in the connectivity core; robust "importance floor" immune to single-whale distortion. |

### 2.6 Community membership metrics (node × partition)

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `community_id` | matched Louvain label (Section 5) | C_1042 | Which economic ecosystem does it live in? |
| `intra_strength_ratio_w` | (amount to/from own community) / (total amount) | 0.88 | Weighted embeddedness — is its money local to the community? |
| `intra_degree_ratio_u` | (distinct neighbors in own community) / (all distinct neighbors) | 0.75 | Unweighted embeddedness. |
| `within_module_z` | z-score of intra-community strength vs community peers (Guimerà–Amaral) | z=6.1 | Is it a local hub of its community? |
| `participation_coef` | 1 − Σ_c (s_c / s_total)² over communities c | 0.05 vs 0.71 | Provincial node (money stays home) vs connector spanning ecosystems. |
| `ga_role` | Guimerà–Amaral role class from (z, P): R1 ultra-peripheral … R7 kinless hub | R5 "connector hub" | Compact role taxonomy for analyst dashboards. |
| `boundary_flag` | has ≥1 inter-community edge above amount floor | true | Bridge inventory for community supergraph edges. |

### 2.7 Temporal deltas (node, month-over-month)

| Metric | Definition | Questions answered |
|---|---|---|
| `Δs_in, Δs_out, Δnet_flow` | absolute and % change | Volume growth/shrink; rail-shift companion signal. |
| `Δrank_pagerank`, `Δrank_hits_*` | rank movement | Rising/falling structural importance — sales triggers and drift alerts. |
| `new_cpty_in/out`, `lost_cpty_in/out` | counterparty set difference vs prior month | Ecosystem expansion or churn; sudden new out-degree = behavioral drift. |
| `edge_turnover_node` | Jaccard distance of incident edge sets | Stability of relationships; near-1 monthly turnover is anomalous for most NAICS. |
| `community_switch` | matched community changed? | Migration between ecosystems; mass switches indicate real market shifts (or partition instability — check NMI first). |
| `role_transition` | ga_role changed (e.g., R2→R6) | Peripheral node becoming a connector hub is a strong drift/AML review trigger. |

---

## 3. Community-Level Metrics (per community, per snapshot)

| Metric | Definition | Example | Questions answered |
|---|---|---|---|
| `n_nodes`, `n_internal_edges` | size, internal simple-edge count | 8,412 nodes | Ecosystem scale. |
| `internal_amount`, `internal_volume` | sums over intra-community edges | $310M | Economic mass of the ecosystem. |
| `ext_out_amount`, `ext_in_amount` | to / from other communities | $42M out | Openness of the ecosystem. |
| `mixing_ratio_w` | internal_amount / (internal + external amount) | 0.82 | Weighted self-containment. Autarkic supply chain vs open marketplace. |
| `mixing_ratio_u` | internal edges / all incident edges | 0.64 | Structural self-containment (compare with weighted: divergence = few large external pipes). |
| `density_directed` | internal edges / (n(n−1)) | 3.1e-4 | Cohesion normalized for size. |
| `weighted_density` | internal_amount / (n(n−1)) | — | Amount-normalized cohesion; comparable across community sizes. |
| `conductance` | cut amount / min(vol(C), vol(V∖C)) | 0.11 | Quality of the community as a "money basin" — low conductance = money pools inside. |
| `internal_reciprocity_w` | weighted reciprocity restricted to internal edges | 0.27 | Trading community (bidirectional) vs distribution tree (one-way). |
| `flow_character` | net internal trophic span + share of throughflow nodes | "cascade" / "circulatory" | Cascade = money enters top, exits bottom (distribution ecosystems); circulatory = loops (local economies, or round-trip risk). |
| `internal_incoherence` | trophic incoherence F₀ on the induced subgraph | 0.9 | Loopiness of the ecosystem's internal flow. |
| `strength_hhi_internal` | HHI of node strengths within community | 0.41 | Is the community one anchor + satellites, or distributed? Anchor-dependent ecosystems are fragility/risk flags. |
| `top_node_share` | largest node's share of internal amount | 0.55 | Analyst-friendly version of the above. |
| `naics_entropy`, `naics_top_share`, `naics_top_code` | Shannon entropy / dominant share of member NAICS | entropy 1.2, top=722 (58%) | Is the community an industry vertical, a geographic economy, or a mixed platform ecosystem? |
| `n_boundary_nodes`, `bridge_amount_top_pairs` | boundary inventory and largest inter-community pipes | — | Which specific nodes carry the community's external relationships? |
| **Lifecycle (cross-month)** | matched via Jaccard on membership: `born`, `died`, `merged(from)`, `split(into)`, `continued`; `membership_jaccard` | C_1042 split → {C_1042a, C_1042b} | Ecosystem formation/dissolution — market events, client-base restructuring, or seasonal patterns. |
| `growth_amount_pct`, `growth_nodes_pct` | month-over-month | +18% amount | Which ecosystems are heating up? Prospect-intelligence input. |

---

## 4. Graph-Level Metrics (per snapshot, per version)

| Metric | Definition | Questions answered |
|---|---|---|
| `n_nodes`, `n_edges`, `total_amount`, `total_volume` | basic mass | Overall network scale trend. |
| `density`, `avg_degree`, `avg_strength` | — | Global connectivity trend. |
| **Tail statistics** | Gini of strength & degree; Hill tail index / CCDF slope on strength | How whale-dominated is the network, and is dominance increasing? |
| `reciprocity_w_global`, `reciprocity_u_global` | Σ min(w_uv,w_vu)/Σ max(...); dyad-based | Is the payment economy becoming more bidirectional (trade-like) or one-directional (distribution-like)? |
| **Directed assortativity (4 combos)** | weighted Pearson correlation of (out-,in-)×(out-,in-) strengths across edges: out-in, out-out, in-in, in-out | Do big payers pay big receivers (core-core) or fan out to small ones (hub-spoke)? Regime shifts here are macro drift signals. |
| `modularity_Q` | Louvain modularity (amount-weighted, symmetrized) | How partitioned is the economy into ecosystems? |
| `n_communities`, community size entropy / largest-community share | — | Fragmented vs monolithic structure. |
| `trophic_incoherence_F0` | global MJR incoherence | Overall loopiness of money flow; rising F₀ = more circular movement economy-wide. |
| `rich_club_w(k)` | weighted rich-club coefficient at top-strength percentiles (normalized vs strength-preserving null when feasible; report raw φ across versions otherwise) | Do the biggest players increasingly transact among themselves? |
| `global_clustering_sym` | transitivity on symmetrized graph | Triadic closure trend in the payment economy. |
| **Cross-version panel** | every metric above × V0–V4 | Hub/whale attribution for every global trend. |

**Cross-snapshot comparisons (graph level):**

| Metric | Definition | Questions answered |
|---|---|---|
| `node_churn`, `edge_turnover` | Jaccard of node/edge sets month-over-month | Base stability of the network; seasonality fingerprint. |
| `partition_nmi`, `partition_ari` | NMI (arithmetic normalization, sklearn-consistent) / ARI between consecutive matched partitions | Is community structure stable enough to interpret longitudinally? Gate for all community-level time series. |
| `rank_stability_spearman` | Spearman of PageRank/HITS/strength rankings month-over-month, per version | Structural persistence; sudden drops localize regime changes. |
| `amount_migration_matrix` | matched-community × matched-community amount flows, Δ over months | Where is money share moving between ecosystems? |

---

## 5. Partition Matching Across Months

1. Run Louvain independently per snapshot (fixed seed, same weight transform).
2. Build the community-overlap contingency table between consecutive months.
3. Greedy maximum-Jaccard matching with thresholds: match if Jaccard ≥ 0.3; `split`/`merge` when one community maps to several above 0.15; else `born`/`died`.
4. Propagate stable community IDs forward; log NMI/ARI as partition-quality gates — if NMI < ~0.6 month-over-month, community time series are flagged low-confidence for that transition.

---

## 6. Custom Implementations (not in cuGraph) — `pkg_custom_metrics.py`

All implemented in CuPy/cuDF with SciPy/NumPy/pandas fallback, wrapped in the `safe()` isolation pattern with feature flags; each returns NaN-filled frames on failure rather than aborting the run.

| # | Function | Why custom |
|---|---|---|
| 6.1 | `weighted_hits` | cuGraph HITS is unweighted; power iteration on the log1p amount matrix (CSR, CuPy sparse) with L2 normalization and tolerance/iteration caps. |
| 6.2 | `trophic_levels`, `trophic_incoherence` | MJR 2020: solve (Λ)h = v, Λ = diag(u) − W − Wᵀ correction form, via CG on sparse matrix; incoherence F₀ from edge-level (h_j − h_i − 1)². |
| 6.3 | `weighted_reciprocity` (global + per-node) | min/max dyad formulation not in cuGraph. |
| 6.4 | `community_mixing` | weighted/unweighted internal-external ratios, conductance, internal reciprocity, per community. |
| 6.5 | `participation_z_roles` | participation coefficient, within-module z, Guimerà–Amaral role classes. |
| 6.6 | `directed_assortativity4` | weighted Pearson across edges for all four strength combinations. |
| 6.7 | `tail_stats` | Gini, Hill estimator, top-k shares. |
| 6.8 | `weighted_rich_club` | φ_w(k) over strength-ranked node fractions. |
| 6.9 | `partition_compare` | NMI (arithmetic), ARI, Jaccard matcher for lifecycle events (validated against sklearn on samples). |
| 6.10 | `edge_turnover` | node- and graph-level Jaccard across snapshots. |
| 6.11 | `threshold_ladder` | percentile + knee-inspection utilities producing the V0–V4 node removal masks. |
| 6.12 | `build_versions` | applies the ladder, returns five edge frames + retention report. |
| 6.13 | `node_concentration` | HHI, top-1 share, asymmetry, ticket sizes (groupby kernels). |

Memory posture at 3–5M nodes/edges: everything above is O(E) or O(E·iters) sparse; CSR of 5M edges ≈ 60 MB fp32 — far under 60 GB. The only guarded item is betweenness (sampled pivots) and any null-model rich-club normalization (edge-swap nulls run on V4 only, optional).

---

## 7. Community & NAICS Supergraphs — `pkg_supergraph.py`

**Community supergraph.** Collapse each community to a single node. Edges = aggregated amount/volume/edge_count between communities; **intra-community payments become self-loops** on the community node.

CSV outputs per snapshot × version:
- `community_supergraph_edges.csv`: `src_comm, dst_comm, amount, volume, edge_count, distinct_pairs, is_self`
- `community_supergraph_nodes.csv`: `comm_id, n_nodes, internal_amount, internal_volume, ext_out_amount, ext_in_amount, mixing_ratio_w, mixing_ratio_u, density_directed, conductance, naics_top_code, naics_top_share, naics_entropy, strength_hhi_internal, top_node_share`

**NAICS supergraph.** Same construction keyed on `source_naics` / `dest_naics` (cleaned: `-1|UNKNOWN`, `******`, null → `UNKNOWN`). Intra-industry payments = self-loops.
- `naics_supergraph_edges.csv`: `src_naics, dst_naics, amount, volume, edge_count, distinct_pairs, is_self`
- `naics_supergraph_nodes.csv`: `naics, n_nodes, internal_amount, ext_out_amount, ext_in_amount, mixing_ratio_w, self_loop_share, net_flow`

**What the supergraphs answer:**
- Which ecosystems/industries are net senders vs net receivers of money (macro flow map)?
- Inter-community trade routes: the 20 largest community-to-community pipes are the backbone of the regional economy — prospect-intelligence and corridor-analysis input.
- Industry input-output structure from actual payments (a payments-derived Leontief-style table): which industries fund which?
- Month-over-month supergraph diffing is cheap (thousands of nodes) and gives the macro drift story that node-level dashboards can't show.
- Run the same node-level metric suite (PageRank, HITS, trophic levels) **on the supergraph itself** — trophic levels of industries is a directly presentable slide: "money enters through NAICS 52, cascades through 42/44, terminates in 61/62."

---

## 8. Interpretation Playbook (signal combinations → payment-domain reads)

| Signature | Read |
|---|---|
| high `d_in`, `d_out`≈1–3, `hhi_out`→1, `pass_through_index`→1 | **Funnel** — N-to-1 collection then forward; legitimate for payroll/PSP, review otherwise. |
| mid trophic level, high `throughflow`, low `net_flow`, low `hhi` both sides, high `Δrank` | **Layering way-station** — money transits without accumulating; chained way-stations across months = escalate. |
| community with high `internal_incoherence`, high `internal_reciprocity_w`, low conductance | **Ring territory** — closed circulatory ecosystem; intersect with cycle enumeration on that subgraph only. |
| community `strength_hhi_internal` > 0.4, `top_node_share` > 0.5 | **Anchor-dependent ecosystem** — client-relationship risk & sales opportunity: the anchor is the account to defend. |
| node `participation_coef` > 0.6 with rising `within_module_z` | **Emerging connector** — cross-ecosystem broker; TM sales prospect for treasury products. |
| V0 metric strong, V1 metric collapses | **Hub artifact** — the signal was a connectivity hub's shadow, not real structure. |
| V0 strong, V2 collapses | **Whale artifact** — amount-driven; check whether one relationship explains it. |
| rising graph-level F₀ + falling `reciprocity_w_global` | Economy-wide shift toward loopier yet less dyadically-reciprocal flow — investigate at supergraph level first. |

---

## 9. Monthly Run Order

1. Load snapshot → clean NAICS → strip self-loops to node attrs → pair-aggregate.
2. `threshold_ladder` → `build_versions` (V0–V4) + retention report.
3. Per version: node flow/concentration block → spectral block (PageRank×2, HITS, eigenvector) → trophic block → reciprocity/clustering/core.
4. Louvain (V0 and V4 minimum) → matching vs prior month → membership block → community block.
5. Graph-level block per version; cross-version Spearman panel.
6. Supergraph builders → 4 CSVs per (snapshot × partition source).
7. Cross-snapshot block (churn, turnover, NMI/ARI, rank stability, migration matrix).
8. Persist: node metrics parquet, community metrics parquet, graph metrics row, supergraph CSVs, thresholds metadata.

*Prepared for internal use — passive reference; no external dependencies required to read.*
