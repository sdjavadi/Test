# PKG Snapshot Metrics — Output Manifest

Pipeline: `pkg_snapshot_metrics.py` · Input: `../data/customers_YYYY-M-1.csv` (source, target, amount, volume) · **All metrics amount-weighted; volume ignored.** Parallel edges are pre-aggregated, so `degree` = distinct neighbor count.

---

## 1. `node_metrics_YYYY-MM.parquet` — one row per customer per month

| Column | Definition | Question it answers |
|---|---|---|
| `in_degree` / `out_degree` / `degree` | Distinct payers / payees / both | Counterparty breadth |
| `in_strength` / `out_strength` / `strength` | Dollar inflow / outflow / gross | Size of flows |
| `net_flow` | in − out | Source, sink, or balanced? |
| `net_flow_norm` | net / gross ∈ [−1, 1] | −1 pure disburser, +1 pure collector |
| `throughput` | min(in, out) | Pass-through dollar capacity |
| `passthrough_ratio` | min/max of in vs out strength | ≈1 with high throughput = conduit signature |
| `disparity_in/out` | Y = Σp² over counterparty shares | Single-counterparty dependency (→1 = one dominates) |
| `entropy_in/out` | Normalized Shannon entropy of shares | Diversification of revenue/spend base |
| `reciprocity_w` | Σ min(w→, w←) / out_strength | Mutual trading vs one-directional flows |
| `pagerank_w` | Amount-weighted PageRank (α=0.85) | Overall flow importance |
| `hits_hub` | HITS hub score (weighted) | Disbursement-engine importance — pays important receivers |
| `hits_authority` | HITS authority score (weighted) | Collection-magnet importance — receives from important payers |
| `katz_in` | Katz over incoming binary paths | Multi-hop upstream influence reach |
| `eigenvector_u` | Eigenvector centrality, undirected weighted | Embedded among heavyweights regardless of direction |
| `betweenness_approx` | Sampled unweighted betweenness (k=256 sources) | Broker/intermediary position (hop-path proxy) |
| `core_number` / `coreness_norm` | k-core (undirected); normalized | Depth of embedding in dense core; attrition warning when falling |
| `triangles_u` / `clustering_u` | Undirected triangles; local clustering coeff. | Real ecosystem embedding vs star/broadcast pattern |
| `bridging_centrality` | betweenness / (clustering + 0.01) | Carries paths *and* sits in sparse neighborhood = articulation risk |
| `cycle_triangles`* | diag(A³): directed 3-cycles through node | Ring-detection prior (candidate before exact time-windowed queries) |
| `burt_constraint`* | Burt's constraint on weighted ego net | Low = broker spanning structural holes; high = redundant ties |
| `trophic_level`* | MacKay weighted trophic level | Tier in the value chain (money flows lower→upper) |
| `community` | Leiden (fallback Louvain) partition label | Business ecosystem membership |
| `bowtie_class` | 0=CORE (giant SCC), 1=IN, 2=OUT, 3=OTHER | Participates in circular flows vs pure pass-through position |

\* Gated by size guards (`MAX_NNZ_FOR_MATMUL`, `TROPHIC_MAX_NODES`); absent if skipped.

## 2. `edge_metrics_YYYY-MM.parquet` — one row per (source, target)

| Column | Definition | Question |
|---|---|---|
| `amount` | Aggregated monthly dollar flow | — |
| `embeddedness` | Jaccard neighborhood overlap of endpoints | 0 = long bridge; high = embedded in shared fabric |
| `is_new` | Edge absent in previous month | First-time relationship |
| `suspicious_new_edge` | new ∧ embeddedness≈0 ∧ amount ≥ p99 | Strongest single-transaction structural anomaly flag |

## 3. `graph_metrics_monthly.csv` — one row per month

| Column | Definition | Question |
|---|---|---|
| `n_nodes`, `n_edges`, `total_amount`, `self_loop_amount` | Sizes | Base rates |
| `density`, `avg_degree` | Connectivity | Consolidating or fragmenting? |
| `giant_wcc_share` | % nodes in largest weak component | Network splintering |
| `scc_core_share`, `bowtie_in/out/other_share` | Bow-tie decomposition | Circular-economy participation vs cascade structure |
| `flow_hierarchy_cnt` / `_amt` | 1 − fraction of edges (dollars) inside nontrivial SCCs | Tree-like cascade vs circulatory economy |
| `reciprocity_w` | Σ min(w→,w←)/Σw | Mutual-trading prevalence |
| `gini_strength`, `hhi_in_strength` | Concentration of flows | Systemic concentration risk |
| `assortativity_w` | Amount-weighted corr. of endpoint strengths | Do heavyweights transact with heavyweights? |
| `s_k_beta` | Exponent of strength ~ degree^β | β>1: big players transact disproportionately more per relationship |
| `rich_club_amount_share` / `_edge_density` | Flow among top 1% strength nodes | Emerging elite circuit of interlinked mega-customers |
| `modularity`, `n_communities` | Partition quality/count | Community structure strength |
| `trophic_incoherence_F0` | Σw(h_j−h_i−1)²/Σw | How strictly tiered the economy is (0 = perfect layering) |
| `vn_entropy_q` | Quadratic (Tsallis-2) approx. of von Neumann entropy | Cheap global structural-change detector (watch the MoM delta) |
| `degree_dist_entropy` | Entropy of degree distribution | Heterogeneity of the network |

## 4. `temporal_graph_metrics.csv` — month vs previous month

| Column | Definition | Question |
|---|---|---|
| `edge_persistence` | % of last month's edges surviving | Baseline relationship stability |
| `n_new_edges`, `new_edge_mean_amount`, `persisting_edge_mean_amount`, `new_edge_amount_share` | First-time edge profile | Are new relationships abnormally large? (AML precursor) |
| `new_edge_large_count_p99` | New edges above prior month's p99 amount | Count of extreme first-time transfers |
| `mean_neighbor_jaccard` | Avg per-node counterparty-set overlap | Network-wide churn level |
| `strength_autocorr_log` | Corr. of log-strength t vs t−1 | Network memory; how far back drift baselines should look |
| `community_ARI` / `community_NMI` | Partition agreement on shared nodes | Are communities real ecosystems or noise? |

## 5. `temporal_node_metrics.parquet` — per node per month

`neighbor_jaccard_prev` (counterparty-set Jaccard vs last month — stable firms ≈0.7–0.9; sudden drop = drift/fraud signal), `deg_prev`, `deg_cur`.

## 6. `node_burstiness.parquet` — per node, whole period

`strength_mean`, `strength_std`, `strength_cv` (coefficient of variation of monthly gross flow — metronomic vs bursty), `months_active`.

## 7. `ordered_rings.parquet` (optional)

Temporally ordered 3-cycles: A→B in month t, B→C in t+1, C→A in t+2, restricted to edges above the 90th amount percentile per month. Columns: `a,b,c, amt1..3, min_leg_amount, month_start`. Far stronger ring signal than static cycles.

---

## Implemented via what

- **cuGraph native:** betweenness (sampled), core_number, triangle_count, Leiden/Louvain, WCC/SCC, BFS (bow-tie), Jaccard (embeddedness).
- **Custom cupy sparse (guaranteed amount-weighted):** PageRank, HITS, Katz, eigenvector, cycle triangles, Burt constraint, trophic levels (CG solve), VN-entropy approximation.
- **cudf joins:** all flow/entropy/disparity/reciprocity metrics, all temporal metrics.

## Deliberately skipped (and why)

- **Current-flow betweenness** — not in cuGraph; O(n·m) impractical at scale. Sampled betweenness + `bridging_centrality` serve as proxies.
- **Full 16-type triadic census** — no GPU implementation; `cycle_triangles` + `clustering_u` + `reciprocity_w` capture the highest-value triad signals. Revisit as GNN feature engineering in Phase 3.
- **Borgatti–Everett core–periphery fit** — `coreness_norm` (k-core) used as the vertical-stratification proxy.
- **Temporal PageRank** — needs time-respecting path machinery; monthly `ordered_rings` + `strength_autocorr` cover the near-term need.
- **Personalized PageRank** — query-time metric requiring seed sets; belongs in the FastAPI layer, not the batch pipeline.
- **Rich-club normalization** against degree-preserving null models — omitted for compute cost; raw share is trend-comparable month over month.

## Operational notes

- Months are processed chronologically; each month's temporal metrics reference the prior month held in GPU memory. Memory pool is freed between months.
- Node ids are re-mapped per month; all cross-month joins use original customer labels.
- Betweenness is unweighted (hop paths) — treat as structural position, not dollar-flow betweenness.
- Hub handling: run once as-is, then re-run on a hub-excluded input extract for the hub-sensitive metrics (HITS, eigenvector, clustering). Keep the hub-included run for rich-club and bow-tie, where hubs are the point.
- To align with the Hadoop monthly snapshot table: node parquet files are partition-ready by month key; column names are stable across months.
