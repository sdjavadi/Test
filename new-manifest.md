# PKG Monthly Snapshot Metrics Manifest
**Payment Knowledge Graph (PKG) — Customer-to-Customer Monthly Network Analysis**
*PNC Bank · Treasury Management · Data Science*

---

## 0. Scope & Conventions

- **Input**: monthly snapshots `../data/cust_YYYY-MM.csv` with schema
  `source, source_name, source_naics, amount, volume, dest, dest_name, dest_naics`
- **Edge weight = `amount` only.** `volume` is carried as a descriptive attribute and is *never* used as a weight in any algorithm.
- **Directed graph**: `source → dest` means source *paid* dest.
- **SCC and WCC metrics are excluded** (memory budget decision).
- **Weight transform policy**:
  - **Spectral / iterative metrics** (HITS, PageRank, Katz, trophic levels, eigenvector-family): use `w = log1p(amount)` to prevent whale edges from dominating the spectrum.
  - **Flow / accounting metrics** (strength, net flow, HHI, mixing ratios, rich-club, tail stats): use **raw amount** — these are meant to reflect real dollar magnitudes.
- **Scale envelope**: 3–5M nodes/edges per snapshot; single GPU ≤ 60 GB. All algorithms selected/implemented fit this envelope (no all-pairs paths, no exact betweenness, no SCC condensation).

---

## 1. Snapshot Ablation Ladder (Graph Versions)

Hub and whale nodes dominate nearly every metric. To measure structure *with and without* their influence, every metric suite runs on five versions of each snapshot. Exclusion lists are computed **once**, from the union graph aggregated over all 23 snapshots (Jan 2024 – Nov 2025), so membership is stable across time and version-to-version comparisons are meaningful.

| Version | Definition | Threshold source (23-snapshot aggregate) |
|---|---|---|
| **V0 — Raw** | Full snapshot, nothing removed | — |
| **V1 — Degree-broad** | Remove nodes with total degree > P99.9 of aggregate degree | data-driven |
| **V2 — Degree-strict** | Remove nodes with total degree > P99.99 | data-driven |
| **V3 — Strength-broad** | Remove nodes with total strength (in+out amount) > P99.9 | data-driven |
| **V4 — Combined-broad** | Remove union of V1 ∪ V3 exclusion sets | data-driven |

**Why a ladder, not a single cleaned graph?** Comparing a metric across V0→V4 tells you *how much of the signal is hub-driven*. Example: if a community's conductance collapses from V0 to V1, that community was only held together by a payroll hub. If graph-level reciprocity barely moves across versions, reciprocity is a genuinely distributed property.

**Questions answered**: Which structural features are artifacts of a handful of mega-nodes? How robust are client rankings to hub removal? Which communities survive de-hubbing?

---

## 2. Node-Level Metrics

Output: one row per `(time_key, version, node)`.

### 2.1 Flow magnitude & balance (raw amount)

| Metric | Definition | Example | Questions it answers |
|---|---|---|---|
| `in_degree`, `out_degree`, `degree` | # of distinct payers / payees / both | Customer paid by 40 counterpart customers, pays 12 | Breadth of network relationships; sudden payer-count drops signal churn |
| `in_strength`, `out_strength`, `strength` | Σ amount received / sent / both | Receives $8.2M/mo, sends $7.9M/mo | Dollar throughput; TM sizing; deposit/liquidity relevance |
| `net_flow` | in_strength − out_strength | +$300K/mo net accumulator | Is this customer a net **source**, **sink**, or **conduit** of funds? |
| `flow_ratio` | net_flow / strength ∈ [−1, 1] | 0.02 → near-perfect pass-through | Pass-through / funnel candidate detection (near 0 with high strength = conduit) |
| `throughflow` | min(in_strength, out_strength) | $7.9M | Magnitude of money *flowing through* the node — key funnel signal |
| `avg_in_ticket`, `avg_out_ticket` | strength / volume per direction | Avg incoming payment $45K | Ticket-size drift; rail-mix and behavior change precursor |

### 2.2 Concentration (raw amount)

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `hhi_out` | Herfindahl–Hirschman of outgoing amounts across payees: Σ (aᵢⱼ/out_strength)² | 0.93 → nearly all money goes to one payee | Supplier/beneficiary concentration; single-point-of-failure; layering leg |
| `hhi_in` | HHI of incoming amounts across payers | 0.11 → diversified revenue | Customer-base concentration risk; revenue diversification for client scoring |
| `top1_out_share`, `top3_out_share` (and in-) | Share of largest 1 / 3 counterparts | 0.97 | Interpretable companion to HHI for business audiences |

### 2.3 Spectral / centrality (log1p amount)

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `pagerank_w` | Weighted PageRank (cuGraph), damping 0.85 | Top-100 PR list ≈ economic anchors | Who is structurally important as a *receiver* of weighted flow? |
| `hits_hub_w` | **Weighted HITS hub score** (custom, power iteration): h ← W·a | Large distributor paying many strong authorities | Who *funds* the important players? Distribution-side importance |
| `hits_auth_w` | **Weighted HITS authority score**: a ← Wᵀ·h | Utility receiving from many strong hubs | Who is *funded by* the important players? Collection-side importance |
| `katz_w` (optional flag) | Katz centrality, α < 1/λ_max | — | Multi-hop influence including low-degree nodes PageRank underweights |
| `betweenness_approx` (optional, k-sample) | Approx betweenness, k=256 sampled sources (cuGraph) | Broker sitting between two industry clusters | Brokerage/intermediation; layering chain middlemen |
| `core_number` | k-core index (unweighted) | core=25 → deep in dense payment core | Peripheral vs. core economy membership; robustness of engagement |
| `trophic_level` | **MacKay–Johnson–Rodgers (2020) trophic level** (custom, CG solve): position in the upstream→downstream flow hierarchy | Raw-materials firm at h≈0.4, retailer at h≈3.1 | Where does this customer sit in the supply-chain hierarchy? Detect customers whose hierarchy position shifts |

### 2.4 Local structure & reciprocity

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `reciprocity_node_w` | **Dyad-based weighted reciprocity** (custom): Σⱼ min(wᵢⱼ, wⱼᵢ) / Σⱼ (wᵢⱼ + wⱼᵢ) | 0.45 → strong two-way trading relationships | Mutual-trade vs. one-way relationships; refund/round-trip patterns |
| `clustering_local` (optional flag) | Local clustering coefficient (undirected projection) | 0.3 → counterparts also trade with each other | Embeddedness in tight business ecosystems |

### 2.5 Community position (requires Louvain partition; custom)

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `community_id` | Louvain (cuGraph, weighted log1p) community label | — | Segment membership |
| `within_module_z` | z-score of node's within-community degree vs. community peers | z=4.2 → local hub of its community | Is this node a hub *inside its own segment*? |
| `participation_coef` | P = 1 − Σ_c (k_i,c / k_i)² over communities its edges touch | P=0.72 → edges spread across many communities | Connector vs. provincial node; cross-segment brokerage |
| `ga_role` | **Guimerà–Amaral role (R1–R7)** from (z, P) plane | R6 "connector hub" | Canonical role taxonomy: ultra-peripheral → kinless hub. Role *transitions* over months are strong behavioral-drift signals |
| `frac_intra_edges_w`, `frac_intra_edges_uw` | Share of node's amount / edge count staying inside its community | 0.85 weighted intra | How dependent is the customer on its local ecosystem? |
| `naics_participation` | Same participation coefficient computed over NAICS groups instead of communities | 0.9 → pays across many industries | Industry diversification of a customer's payment book |

---

## 3. Community-Level Metrics (Louvain partition, per community)

Output: one row per `(time_key, version, community_id)`. All metrics also computed **per NAICS group** (2-digit sector and full code) — identical formulas, partition swapped.

### 3.1 Size & internal structure

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `n_nodes`, `n_internal_edges` | Membership counts | 12,400 nodes | Segment size distribution; consolidation vs. fragmentation |
| `internal_amount`, `internal_volume` | Σ amount / volume on intra-community edges | $410M/mo internal | Economic mass of the segment |
| `density_uw` | n_internal_edges / (n·(n−1)) | 3e-4 | How tightly knit is the segment (link-wise)? |
| `density_w` | internal_amount / (n·(n−1)) — avg dollar intensity per possible dyad | $27/dyad | Dollar-intensity of cohesion, comparable across community sizes |
| `internal_reciprocity_w` | Dyad-min weighted reciprocity restricted to internal edges | 0.38 | Do members trade **with** each other or just pay one another one-way? |
| `internal_avg_ticket` | internal_amount / internal_volume | $18K | Segment payment-size fingerprint |

### 3.2 Boundary & mixing

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `out_amount_ext`, `in_amount_ext` | Amount leaving to / arriving from other communities | Exports $95M, imports $60M | Trade-balance of the segment |
| `mixing_ratio_w` | internal_amount / (internal + external amount) — weighted "E-I"-style index | 0.81 | Self-contained ecosystem vs. open segment (dollar terms) |
| `mixing_ratio_uw` | Same with edge counts | 0.64 | Link-wise closure; divergence from weighted version reveals whether big money or many links cross the boundary |
| `conductance_w` | cut_amount / min(vol(S), vol(V∖S)) | 0.12 | Classic quality-of-cut: low = well-separated community |
| `net_external_flow` | in_amount_ext − out_amount_ext | −$35M | Is the segment a net exporter or importer of funds? Liquidity direction |
| `boundary_node_frac` | Fraction of members with ≥1 external edge | 0.22 | Thin vs. broad interface with the rest of the economy |

### 3.3 Composition & identity

| Metric | Definition | Example | Questions |
|---|---|---|---|
| `naics_top1`, `naics_top1_share` | Dominant NAICS sector and its member share | "23-Construction", 0.61 | Business interpretation / labeling of communities |
| `naics_entropy` | Shannon entropy of NAICS composition | 1.9 bits | Mono-industry supply chain vs. mixed regional ecosystem |
| `hub_dependence` | Δ internal_amount when top-1 internal node removed / internal_amount | 0.7 → one node carries 70% | Fragility of the segment; anchor-client identification |

### 3.4 Community lifecycle (cross-snapshot; custom)

Communities matched month-over-month by **Jaccard similarity of membership** (threshold τ=0.3, best-match with tie-breaking by overlap size).

| Event | Rule | Questions |
|---|---|---|
| `continuation` | 1↔1 match above τ | Stable segments — safe for trend analytics |
| `growth` / `shrinkage` | Matched with size change beyond ±20% | Which ecosystems are expanding? Early sales territory signals |
| `birth` / `death` | No match forward / backward | New business ecosystems forming; segment collapse |
| `merge` / `split` | Many→1 / 1→many matches above τ | Consolidation events; supply-chain restructuring |
| **Partition stability**: `nmi_vs_prev`, `ari_vs_prev` | NMI (arithmetic normalization) / Adjusted Rand between consecutive monthly partitions | Global answer to "how stable is our segmentation?" — gates whether community-level trends are meaningful |

---

## 4. NAICS-Level Metrics

Everything in §3.1–3.3 is recomputed with the NAICS partition (both full code and 2-digit sector rollup) in place of Louvain communities. Additional NAICS-specific items:

| Metric | Definition | Questions |
|---|---|---|
| `sector_pair_amount[i,j]` | Full sector-to-sector flow matrix (from the NAICS supergraph, §6) | Input–output table of the client economy; which sectors fund which? |
| `sector_self_share` | Diagonal share of the sector flow matrix | Intra-industry trading intensity by sector |
| `sector_net_position` | Row-sum − column-sum of the flow matrix | Which sectors are net payers vs. net receivers within the PNC book? |
| `unknown_naics_share` | Share of nodes/amount with `-1|UNKNOWN` NAICS | Data-quality tracker; weighting caveat for all NAICS metrics |

---

## 5. Graph-Level Metrics

Output: one row per `(time_key, version)`.

### 5.1 Scale & flow

| Metric | Definition | Questions |
|---|---|---|
| `n_nodes`, `n_edges`, `total_amount`, `total_volume`, `avg_ticket` | Basic scale | Book growth; macro payment trend |
| `density` | edges / n(n−1) | Sparsification/densification over time |
| `reciprocity_uw`, `reciprocity_w` | Edge-wise and **dyad-min weighted** reciprocity (custom) | Overall bilaterality of the client economy |

### 5.2 Heterogeneity & tails (custom)

| Metric | Definition | Questions |
|---|---|---|
| `gini_strength`, `gini_degree` | Gini coefficient of strength / degree distributions | Concentration of the economy; V0 vs V1 gap = hub contribution |
| `hill_alpha_strength` (top 5% tail) | Hill estimator of power-law tail index | Fat-tail severity; α↓ over time = growing whale dominance |
| `top_0.1pct_amount_share` | Amount share held by top 0.1% strength nodes | Board-friendly concentration statement |
| `rich_club_w(r)` | **Weighted rich-club coefficient** (custom) at rank fractions r ∈ {1%, 0.1%} | Do the biggest players preferentially transact *with each other*? Elite-core detection |

### 5.3 Mixing & hierarchy (custom)

| Metric | Definition | Questions |
|---|---|---|
| `assortativity_{in-in, in-out, out-in, out-out}` | **Four-way directed weighted assortativity** — Pearson corr. of endpoint strengths over edges, amount-weighted | Do big senders pay big receivers? Full directed mixing fingerprint; changes signal structural regime shifts |
| `trophic_incoherence_F0` | **Global trophic incoherence** (MJR 2020): Σw(hⱼ−hᵢ−1)²/Σw | Is the economy hierarchical (feed-forward supply chains, F0→0) or loopy (circular flows, F0→1)? Rising F0 can indicate growth of circular/round-trip flow |
| `modularity_Q` | Louvain modularity of the partition | Strength of community structure; comparability gate across months |
| `n_communities`, `community_size_gini` | Partition shape | Fragmentation vs. consolidation of the segment landscape |
| `hits_hub_hhi`, `hits_auth_hhi` | HHI of HITS score mass | Is "importance" spreading or concentrating? |

### 5.4 Cross-snapshot temporal panel

All node/community/graph metrics are stacked into long-format panels keyed by `(time_key, version, entity_id, metric, value)` enabling:
- MoM and YoY deltas, rolling z-scores per entity → **drift flags** feed Rail Shift Monitor and Behavioral Drift use cases (roadmap #2, #15)
- Rank-stability (Spearman of consecutive top-k rankings) for PageRank/HITS/strength → how volatile is "who matters"?
- Version-gap series, e.g. `gini_strength(V0) − gini_strength(V1)` → hub-contribution trend

---

## 6. Supergraphs (Community-as-Node / NAICS-as-Node)

For each `(time_key, version, partition ∈ {community, naics, naics2})` the snapshot is collapsed:

- Every group becomes one node; **intra-group payments → self-loop edge**, inter-group payments → directed edge, `amount` and `volume` summed.
- Output CSVs in `../snapshot/` use the **exact base schema** `source,source_name,source_naics,amount,volume,dest,dest_name,dest_naics`, so **the entire metric suite of §2–§5 re-runs unchanged on supergraphs** (community PageRank, sector HITS, sector trophic levels, etc.). Amount conservation is asserted: Σ supergraph amount = Σ snapshot amount.
- File naming: `super_{partition}_{version}_{YYYY-MM}.csv` e.g. `super_naics2_V0_2024-01.csv`.

**Questions**: Which *sector* is the authority of the economy? What is the trophic ordering of industries? Which community pairs form the strongest corridors? Sector-level rail-shift precursors.

---

## 7. Implementation Inventory

| Component | Source | Notes |
|---|---|---|
| PageRank, k-core, Louvain, betweenness (k-sample), clustering | **cuGraph** | GPU, fits 60 GB envelope |
| Weighted HITS (power iteration) | `pkg_custom_metrics.py` | cuGraph HITS is unweighted → custom |
| Trophic levels + incoherence (MJR 2020, CG solver) | `pkg_custom_metrics.py` | sparse SPD solve |
| Dyad-min weighted reciprocity (graph & node & community) | `pkg_custom_metrics.py` | hash-join on reversed edges |
| Participation coefficient, within-module z, GA roles R1–R7 | `pkg_custom_metrics.py` | groupby algebra |
| Four-way directed weighted assortativity | `pkg_custom_metrics.py` | weighted Pearson over edge list |
| Gini, Hill tail index, top-share | `pkg_custom_metrics.py` | sort-based |
| Weighted rich-club | `pkg_custom_metrics.py` | rank-threshold subgraph sums |
| NMI (arithmetic) / ARI partition comparison | `pkg_custom_metrics.py` | contingency-based |
| Jaccard lifecycle matcher | `pkg_custom_metrics.py` | sparse overlap matrix |
| Ablation ladder builder (23-snapshot aggregate, P99.9/P99.99) | `pkg_custom_metrics.py` | produces exclusion parquet |
| Supergraph builder + conservation smoke test | `pkg_supergraph.py` | writes `../snapshot/` |
| Orchestration | `pkg_pipeline.py` | per-snapshot × per-version loop, safe() isolation, NaN frames on failure |

All custom functions are wrapped in a `safe()` decorator with per-metric feature flags: any single metric failure logs and returns a NaN-filled frame instead of killing the monthly run.
