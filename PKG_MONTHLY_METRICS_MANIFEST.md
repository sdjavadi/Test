# PKG Monthly Snapshot Metrics Manifest
**Payment Knowledge Graph — GPU-Accelerated Temporal Graph Analytics**
*Node-level · Community-level · Graph-level · Cross-snapshot | cuGraph + custom CuPy/cuDF implementations*
*Version 1.0*

---

## 0. Scope, Input Contract, and Global Conventions

### 0.1 Input schema

One CSV per month:

| Column | Type | Meaning |
|---|---|---|
| `source` | id | Paying customer |
| `source_name` | str | Descriptive only (not used in metrics) |
| `source_naics` | str | Industry code of payer (used in community profiling) |
| `amount` | float | **Total dollars** source→dest in the month — the only edge weight |
| `volume` | int | Transaction count source→dest in the month — descriptive attribute |
| `dest` / `dest_name` / `dest_naics` | | Symmetric to source |

**Implication:** the edge list is already **pair-aggregated** — one row per (source, dest) per month. Therefore:
- **Degree = distinct counterparties** (the meaningful counterparty-ecosystem measure). The multigraph degree-inflation problem does not apply here.
- `volume` never enters weighted algorithms (per requirement: *amount is the only weight*). It is retained for one derived descriptive metric, average ticket size = amount / volume.

### 0.2 Global conventions

| Convention | Rule | Rationale |
|---|---|---|
| **Self-loops** | Rows with `source == dest` are removed from the graph and stored as node attributes `self_amount`, `self_volume` | Self-loops distort strength, reciprocity, HITS, trophic levels. Kept as a standalone signal (ties into self-payment detection: a customer's on-us self-transfer amount is itself a fraud/treasury-behavior feature). |
| **Weight transform** | Structural/flow metrics use **raw amount**. Spectral metrics (HITS, eigenvector, Katz, PageRank optionally) support `log1p(amount)` via config flag `WEIGHT_TRANSFORM` | Payment amounts are heavy-tailed over ~8 orders of magnitude; one $500M edge can make a spectral score degenerate (all mass on one node pair). Run both; if raw and log1p rankings diverge sharply, the raw ranking is amount-dominated, not structure-dominated. |
| **Vertex encoding** | Factorize ids to contiguous `int64` 0..n−1 once per month; keep the id↔code map | Required for CuPy CSR construction; makes cross-month joins cheap. |
| **Failure isolation** | Every algorithm call wrapped in `safe()` — log and continue | Multi-month, multi-version batch runs must not die on one metric. |
| **Skipped by design** | SCC, WCC, bow-tie decomposition (needs SCC), exact diameter/closeness (all-pairs cost) | Per requirement; flow hierarchy is covered by **trophic incoherence** instead, which needs no component analysis. |

### 0.3 Scale envelope

3–5M nodes, 3–5M edges/month. A CSR of 5M float64 edges ≈ **~100 MB**; the entire metric suite peaks well under **5 GB** of the 60 GB budget. The only genuinely expensive item is betweenness — handled by k-sampling and a feature flag. Everything else runs in seconds-to-a-few-minutes per version, so **5 versions × N months** is tractable as a single batch.

---

## 1. Graph Versions: Exclusion Framework & Data-Driven Thresholds

### 1.1 Why five versions

Payment graphs are dominated by a tiny set of super-nodes (payroll processors, card settlers, mega-treasuries). They (a) absorb nearly all centrality mass, (b) create false triangles/rings, and (c) make "the graph" mostly a picture of a few institutions. Recomputing the full suite on exclusion versions answers a question no single run can: **which structure is intrinsic to the market vs. an artifact of the hubs?**

### 1.2 Threshold selection (data-driven, computed monthly)

Two independent criteria, two thresholds each, all derived from the month's own distributions (thresholds are re-estimated each month and logged, so exclusion adapts to graph growth):

| Criterion | Soft threshold | Hard threshold | Method |
|---|---|---|---|
| **Hub** (total degree = distinct counterparties) | `D_soft` = p99.9 of degree | `D_hard` = p99.99 of degree | Percentile default; optional **knee detection** (Kneedle on log-log CCDF — the rank where the empirical curve bends away from the power-law chord marks the structural break between "large customer" and "infrastructure node") |
| **Strength** (total in+out amount) | `S_soft` = p99.9 of strength | `S_hard` = p99.99 of strength | Same |

Percentiles and knee both provided in `pkg_custom_metrics.build_versions()`; the chosen thresholds are written to `version_thresholds` output so every downstream number is auditable.

### 1.3 The five versions (ablation ladder)

| Version | Exclusion rule | What it isolates |
|---|---|---|
| **V0_raw** | none | Ground truth incl. infrastructure |
| **V1_hub_extreme** | degree > `D_hard` | Remove only mega-hubs (∼0.01% of nodes). Minimal surgery; fixes viz/centrality pollution. |
| **V2_hub_broad** | degree > `D_soft` | Remove the whole hub layer (∼0.1%). Reveals the mid-market structure. |
| **V3_combined_extreme** | degree > `D_hard` **OR** strength > `S_hard` | Also removes low-degree/huge-amount nodes (e.g., a customer wiring $2B to 3 counterparties — invisible to degree filters, dominant in weighted metrics). |
| **V4_combined_broad** | degree > `D_soft` **OR** strength > `S_soft` | The "analytical core": structure of the ordinary economy. Communities, assortativity, and clustering are most interpretable here. |

Exclusion removes the node **and all incident edges**. Degree-only vs strength-only removal are deliberately separated at the extreme tier (V1 vs the OR in V3) so you can attribute metric shifts to connectivity hubs vs. amount whales.

### 1.4 Cross-version diagnostics (computed per month)

| Metric | Definition | Question answered |
|---|---|---|
| `survival_bitmask` (node) | 5-bit mask of versions the node survives | Which customers *are* the infrastructure layer; stable exclusion registry feeds the Hub Node Registry deliverable |
| `nodes_removed`, `edges_removed`, `amount_removed_%` (per version) | Share of graph mass excluded | "Hubs are 0.08% of nodes but 61% of dollars" — a headline stakeholder number |
| **Spearman rank stability** ρ(V0, Vk) per centrality | Rank correlation of PageRank/HITS/etc. over surviving nodes | How hub-distorted is each centrality? ρ(V0,V4)=0.55 for PageRank but 0.90 for trophic level ⇒ PageRank rankings are hub artifacts; trophic position is robust |
| **Metric elasticity** | % change of each graph-level metric across the ladder | E.g., reciprocity that doubles from V0→V4 means hubs are one-directional sinks masking a reciprocal mid-market |

---

## 2. Node-Level Metrics

Format per metric: **Definition → Formula → Compute → Example → Questions it answers → Caveats.** All are emitted per (month × version).

### 2.1 Local flow & connectivity

#### 2.1.1 `in_degree`, `out_degree`, `total_degree`
- **Definition:** Number of distinct paying / paid / all counterparties this month.
- **Formula:** `out_degree(i) = |{j : (i→j) ∈ E}|`; total = in + out (a partner in both directions counts twice — intentional, it measures relationship *slots*; a `distinct_partners` variant via union is optional).
- **Compute:** cuDF `groupby.count()` on the pair-aggregated edge list.
- **Example:** A regional distributor pays 240 suppliers and receives from 1,900 retailers → out 240, in 1,900.
- **Questions:** Counterparty ecosystem breadth; TM sales sizing ("how connected is this client?"); hub identification input.
- **Caveats:** On this schema degree already equals distinct neighbors — no multigraph correction needed.

#### 2.1.2 `in_strength`, `out_strength`, `total_strength`
- **Definition:** Total dollars received / sent / both.
- **Formula:** `s_out(i) = Σ_j amount(i→j)`.
- **Compute:** cuDF `groupby.sum()`.
- **Example:** in $4.1M, out $3.9M, total $8.0M.
- **Questions:** Economic size in the network (vs. balance-sheet size); denominator for every normalized flow metric; strength-threshold exclusions.
- **Caveats:** Heavy-tailed; report log10 in dashboards.

#### 2.1.3 `net_flow`, `gross_flow`, `flow_ratio`
- **Definition:** Directional balance of the node.
- **Formula:** `net = s_in − s_out`; `gross = s_in + s_out`; `flow_ratio = net / gross ∈ [−1, +1]`.
- **Compute:** vectorized cuDF.
- **Example:** payroll processor: in $500M, out $499M → ratio ≈ 0.001 (pure conduit). A payer-only manufacturer: ratio → −1 (net source). A collections-heavy utility: ratio → +1 (net sink).
- **Questions:** Is the customer a **source, sink, or conduit**? Month-over-month drift of flow_ratio is a behavioral-drift primitive (a customer flipping from −0.6 to +0.4 changed its business or is being used as a pass-through).
- **Caveats:** Only PNC-visible flows; a "source" may fund itself outside the graph.

#### 2.1.4 `passthrough_ratio` (conduit index) — *flow metric, custom*
- **Definition:** How balanced and simultaneous in/out flows are — the core **funnel / mule / layering** primitive.
- **Formula:** `min(s_in, s_out) / max(s_in, s_out)` (0 if either side is 0).
- **Compute:** vectorized cuDF.
- **Example:** in $912K from 47 counterparties, out $897K to 2 counterparties → passthrough 0.98 **and** funnel shape (in_degree ≫ out_degree) = classic consolidation motif for the Motif Finder.
- **Questions:** Rank candidates for funnel/pass-through detection; combined with low `hhi_in` + high `hhi_out` it *is* the N-to-1 funnel score.
- **Caveats:** Legitimate treasury concentrators score high — pair with NAICS and degree asymmetry before alerting (known false-positive class from UC #3).

#### 2.1.5 `avg_ticket_in`, `avg_ticket_out`
- **Definition:** Mean transaction size per direction.
- **Formula:** `Σ amount / Σ volume` over in- (out-) edges.
- **Compute:** cuDF ratios of strength to volume sums.
- **Example:** avg ticket out drops $48K → $3.1K while out_degree jumps ⇒ shift from wires to mass disbursement (ACH-like behavior) — a rail-shift proxy inside this schema.
- **Questions:** Structuring signal (many small tickets replacing few large ones); product-fit signal for TM sales.
- **Caveats:** The single place `volume` is used; never a weight.

#### 2.1.6 `self_amount`, `self_volume`
- **Definition:** Dollars/count the customer pays itself (removed self-loops).
- **Questions:** Direct feed into the self-payment analysis; on-us liquidity shuffling; wash-flow indicator.

### 2.2 Concentration & diversification — *custom*

#### 2.2.1 `hhi_out`, `hhi_in` (Herfindahl–Hirschman of counterparty flows)
- **Definition:** Concentration of a node's dollars across its counterparties.
- **Formula:** `HHI_out(i) = Σ_j (amount(i→j)/s_out(i))²` ∈ (0,1]; 1 = single counterparty.
- **Compute:** custom cuDF (merge strength, square shares, groupby-sum).
- **Example:** supplier with 60 buyers but one buyer = 82% of receipts → hhi_in ≈ 0.68: high **counterparty dependency risk** despite broad degree.
- **Questions:** Client fragility (loss of one counterparty), funnel scoring (high hhi_out + high passthrough), prospect intel (the dominant counterparty of a valuable client is a prospect).
- **Caveats:** For degree-1 nodes HHI ≡ 1 — condition on degree ≥ 3 when screening.

#### 2.2.2 `entropy_out`, `entropy_in` and `eff_cpty_out`, `eff_cpty_in`
- **Definition:** Shannon entropy of the flow share distribution; effective number of counterparties.
- **Formula:** `H = −Σ p_j ln p_j`; `eff_cpty = exp(H)` (alternatively `1/HHI`). Both reported.
- **Example:** degree 10,000 but eff_cpty 38 → the "10K edges, 50 real relationships" distinction: the node *behaves* like it has ~38 partners.
- **Questions:** The corrected counterparty-ecosystem size for relationship scoring (UC #9); a better hub criterion than raw degree for some purposes.
- **Caveats:** Entropy uses natural log; normalize by ln(degree) if a [0,1] diversity score is needed (`entropy_norm` also emitted).

### 2.3 Global importance (centrality family)

#### 2.3.1 `pagerank_w` (weighted, directed) — cuGraph
- **Definition:** Stationary probability of a random dollar-following walk with teleport.
- **Formula:** standard PR with transition ∝ amount, α = 0.85.
- **Example:** the node where SAR-adjacent flows keep arriving ranks high even with modest strength.
- **Questions:** Who ultimately *receives* importance-weighted flow; the substrate of Label Diffusion (personalized PR from SAR seeds is the same computation with a seed vector).
- **Caveats:** Severely hub-dominated in V0 — this is the metric the version ladder exists for. Report both raw and log1p-weight variants.

#### 2.3.2 `pagerank_rev` (reverse PageRank / "source-ness") — cuGraph on reversed edges
- **Definition:** PageRank on the transposed graph.
- **Questions:** Who *originates* flows that feed important nodes — upstream analog; useful in layering to find the funding side of a chain. The pair (pagerank_w, pagerank_rev) classifies nodes into origin / relay / destination roles.

#### 2.3.3 `hits_hub_w`, `hits_auth_w` — **weighted HITS, custom** (cuGraph HITS is unweighted)
- **Definition:** Mutually reinforcing scores: a good **hub** sends money to good authorities; a good **authority** receives from good hubs. With amount weights these become the principal left/right singular vectors of the flow matrix.
- **Formula:** iterate `a ← Wᵀh`, `h ← Wa` with L2 normalization to convergence.
- **Compute:** `pkg_custom_metrics.weighted_hits` — CuPy CSR power iteration; also emit unweighted cuGraph HITS for comparison.
- **Example:** In a payroll ecosystem, the processor is an extreme hub (pays thousands of authorities = employees’ banks / vendors); a clearing-like customer that everyone pays is an extreme authority. In AML, **high hub AND high authority simultaneously** is a two-sided distribution/collection node — rarer and more interesting than either alone.
- **Questions:** Distinguishes payer-side vs payee-side systemic importance, which PageRank conflates; hub score is a principled, continuous replacement for degree-threshold "hub" labels; `hub−auth` gap is a directional-role score.
- **Caveats:** Converges to the dominant singular pair — one giant bilateral flow can capture it; the log1p variant is the default reported one, raw kept for diagnostics. Scores are meaningful *within* a snapshot; compare ranks, not values, across months.

#### 2.3.4 `eigenvector_w` / `katz_w` — cuGraph (Katz custom-weighted fallback provided)
- **Definition:** Recursive importance; Katz adds a baseline β so nodes off the dominant component still score.
- **Questions:** "Connected to the well-connected" — complementary to PR (no teleport): more sensitive to dense cores. Katz is the robust choice on directed flow graphs where eigenvector mass collapses onto the dominant hub cluster.
- **Caveats:** Report only for V2+/V4 by default; in V0 these are essentially hub indicators.

#### 2.3.5 `betweenness_approx` — cuGraph, k-sampled, **flag-gated**
- **Definition:** Share of shortest paths through the node, k source samples (k≈256–1024), distance = `1/log1p(amount)` so heavier corridors are "shorter".
- **Questions:** Broker/gatekeeper detection — the structural signature of **layering intermediaries** that pure strength misses (small amounts, critical position).
- **Caveats:** The one metric where cost is real (minutes, not seconds); run on V2/V4 only; approximation noise ⇒ use rank bands, not raw values, over time.

#### 2.3.6 `core_number` — cuGraph (undirected view)
- **Definition:** Max k such that the node survives iterative pruning of degree < k.
- **Example:** core 12 = embedded in a dense mutual-payment mesh; core 1 = pendant.
- **Questions:** Separates *embedded* economy members from peripheral one-off payers; the max-core subgraph is a natural "market backbone" export; sudden core jump = new dense cluster around the node (collusion signal).
- **Caveats:** Unweighted by definition; strength-aware embeddedness is covered by community metrics.

#### 2.3.7 `clustering_local` — cuGraph triangle counting (undirected), optional
- **Definition:** Fraction of a node's neighbor pairs that transact with each other.
- **Questions:** Supply-chain triangles vs star-shaped disbursement; low clustering + high degree = pure distributor. Mule accounts typically have ~0 clustering (their counterparties never know each other).
- **Caveats:** Compute on V2+ (hubs create astronomically many spurious open triads); weighted (Barrat) variant deliberately omitted at this scale — the unweighted value on the amount-thresholded graph is the pragmatic substitute.

### 2.4 Position in the flow hierarchy — *custom, flow-based*

#### 2.4.1 `trophic_level` (MacKay–Johnson–Rogers) and graph `trophic_incoherence`
- **Definition:** Every node gets a height h such that money, on average, flows "uphill" by one level per edge; the residual measures how hierarchical vs. circular the flow system is. This is the SCC-free replacement for flow-hierarchy analysis.
- **Formula:** solve `Λh = v` with `Λ = diag(s_in+s_out) − W − Wᵀ`, `v = s_in − s_out`; incoherence `F₀ = Σ w_ij (h_j − h_i − 1)² / Σ w_ij ∈ [0,1]` (0 = perfectly layered DAG-like flow, →1 = fully circular).
- **Compute:** `pkg_custom_metrics.trophic_levels` — CuPy sparse conjugate-gradient (diagonal-regularized); seconds at 5M nodes.
- **Example:** consumers/payers at level ≈ 0, distributors ≈ 1, manufacturers ≈ 2, a terminal collector ≈ 3. A node whose level sits mid-chain with passthrough ≈ 1 is a layering way-station.
- **Questions:** Where does each customer sit in the payment food chain (upstream funding vs downstream collection)? Which communities are internally circular (high local incoherence — ring-friendly territory for UC #6) vs. hierarchical supply chains? Month-over-month level drift is a strong behavioral-drift feature.
- **Caveats:** Levels are relative (defined up to a constant; anchored at min = 0 per snapshot) — track *rank* and *within-community* position over time, not absolute values.

### 2.5 Reciprocity — *custom, weighted*

#### 2.5.1 `reciprocity_w` (node) and graph-level `reciprocity_w`, `reciprocity_uw`
- **Definition:** Share of a node's flow that is matched by counter-flow with the same partners (Squartini et al. weighted reciprocity).
- **Formula:** node: `Σ_j min(w_ij, w_ji) + Σ_j min(w_ji, w_ij) over its edges / (s_out + s_in)`; graph: `W↔/W = Σ_ij min(w_ij,w_ji) / Σ_ij w_ij`. Unweighted analog: share of edges with a reverse edge.
- **Compute:** cuDF self-merge of the edge list with (src,dst) swapped.
- **Example:** two customers exchanging $1.0M / $0.9M monthly → each has high reciprocity; a payroll flow has ≈ 0.
- **Questions:** Bilateral trading relationships vs one-way distribution; **round-tripping / wash-flow** candidates (high reciprocity + high passthrough + few partners); market character shift when measured graph-wide across the version ladder.
- **Caveats:** Netting arrangements legitimately produce high reciprocity — a screen, not a verdict.

### 2.6 Community membership metrics (node × partition)

Community detection: **Louvain (cuGraph, amount-weighted; graph symmetrized by summing both directions)**, with Leiden as the preferred option where the installed cuGraph version provides it (better-connected communities, less resolution noise). Partition is computed **per version** — community structure with and without hubs differs materially, and that difference is itself a finding.

| Metric | Definition / formula | Compute | Payment question it answers |
|---|---|---|---|
| `community_id` | Louvain/Leiden label | cuGraph | Segment membership; join key for §3 |
| `intra_strength_share` | s_i,c(i) / s_i — share of the node's dollars staying inside its community | custom cuDF | How captive is the customer to its ecosystem? Low share = boundary operator |
| `participation_coeff` | P_i = 1 − Σ_c (s_ic/s_i)² (Guimerà–Amaral, strength-weighted) | custom | Does the node bridge many communities (P→1) or live in one (P→0)? Bridges are cross-selling conduits and layering paths |
| `within_module_z` | z_i = (κ_i − μ_κ,c) / σ_κ,c, κ = internal strength | custom | Is the node a local champion relative to its own community, independent of global size? |
| `role` (R1–R7) | Guimerà cartography on (z, P): ultra-peripheral → kinless hub | custom | One categorical label for dashboards: "provincial hub", "connector hub", etc.; role *transitions* month-over-month are high-signal drift events |
| `is_boundary` | has ≥1 inter-community edge | custom | Cut membership; boundary nodes carry all inter-segment flow |
| `community_switched` | community (matched via §5) differs from last month | custom | Ecosystem re-affiliation — merger, supplier switch, or account takeover |

### 2.7 Temporal node metrics (per node, across months)

| Metric | Definition | Question |
|---|---|---|
| `Δ` and `%Δ` of every §2.1–2.5 metric | month-over-month difference | Raw drift components |
| `z_baseline` | (x_t − mean of trailing 6 months)/std | "Is this month abnormal *for this customer*?" — the alerting normalization that kills alert fatigue (each customer is its own baseline) |
| `rank_volatility` | std of the node's PageRank/HITS percentile over trailing window | Stable pillar vs. erratic actor |
| `tenure` | # consecutive months present | New-entrant flag; mule accounts are typically low-tenure + high passthrough |
| `is_new`, `is_churned` | appears / disappears vs t−1 | Network entry/exit rates roll up to graph level |
| `flow_ratio_flip` | sign change of flow_ratio | Source↔sink inversion — top-tier drift event |

---

## 3. Community-Level Metrics (per community × month × version)

All computed by `pkg_custom_metrics.community_mixing` from three cuDF groupbys (internal edges, out-cut, in-cut) — effectively free at this scale. Notation: for community c, `e_int/w_int` = count/amount of edges with both endpoints in c; `e_out/w_out` = edges leaving c; `e_in/w_in` = edges entering c; `n_c` = node count; `W` = total graph amount.

### 3.1 Size & mass

| Metric | Formula | Question |
|---|---|---|
| `n_nodes`, `e_internal` | counts | Segment size |
| `w_internal` | Σ internal amount | Economic mass of the ecosystem |
| `w_out`, `w_in`, `e_out`, `e_in` | cut mass, both directions separately | Directional openness |
| `share_of_graph_amount` | (w_int + w_out + w_in)/ (attribution: w_int + ½·cuts) / W | Which 10 communities are 80% of the economy? |

### 3.2 Cohesion & mixing (the in/out-of-community family — unweighted AND weighted)

| Metric | Formula | Interpretation / question |
|---|---|---|
| `density_uw` | e_int / (n_c·(n_c−1)) (directed) | Structural tightness; compare across community sizes with care (density falls mechanically with n) |
| `density_w` | w_int / (n_c·(n_c−1)) | Average dollar intensity per possible internal relationship |
| `embeddedness_uw` | e_int / (e_int + e_out + e_in) | Share of relationships that are internal |
| `embeddedness_w` | w_int / (w_int + w_out + w_in) | Share of **dollars** that stay inside — the headline "how closed is this ecosystem" number |
| `ei_index_uw` | ((e_out+e_in) − e_int) / ((e_out+e_in) + e_int) ∈ [−1,1] | Krackhardt E-I: −1 fully closed, +1 fully open (unweighted) |
| `ei_index_w` | same with w | Weighted E-I; **divergence between the two is a finding**: E-I_uw ≈ +0.2 but E-I_w ≈ −0.6 ⇒ many small external relationships, but the money stays home |
| `conductance_w` | (w_out + w_in) / (2·w_int + w_out + w_in) | Cut quality; low conductance communities are natural units for targeted analysis and Cypher scoping |
| `internal_reciprocity_w` | Squartini reciprocity on internal subgraph (optional flag) | Circular/bilateral character of the ecosystem — ring-detection prior (UC #6) |

### 3.3 Flow character

| Metric | Formula | Question |
|---|---|---|
| `net_flow` | w_in − w_out | Is the community a **net sink** (collection ecosystem) or **net source** (disbursement ecosystem) vs the rest of the graph? |
| `flow_ratio` | net / (w_in + w_out) | Normalized version, comparable across communities |
| `avg_ticket_internal` | w_int / Σ internal volume | Retail-like vs wholesale-like internal economy |
| `mean_trophic_level`, `trophic_span` | mean and (p95−p5) of member levels | Flat community (peers) vs deep supply chain; span↑ = layered vertical |
| `local_incoherence` | F₀ restricted to internal edges | Circularity *inside* the community — the direct ring-hunting prior |

### 3.4 Internal concentration & composition

| Metric | Formula | Question |
|---|---|---|
| `modularity_contribution` | Q_c = w_int/W − (S_out,c · S_in,c)/W² (directed); Σ_c Q_c = graph modularity | Which communities actually carry the partition quality |
| `hhi_internal_strength` | HHI of members' internal strength shares | Is the ecosystem a peer mesh or one anchor + satellites? |
| `top1_strength_share` | max member internal strength / w_int | Anchor-dependence: if the anchor leaves PNC, the community's flows follow |
| `boundary_node_frac` | share of members with ≥1 external edge | Interface thickness |
| `naics_top_share`, `naics_entropy` | dominant industry share; Shannon entropy over member NAICS | Is this a *sector* community or a *geographic/relational* one? Low purity + high cohesion = supply-chain crossing industries (interesting); requires NAICS cleaning (−1/UNKNOWN bucketed) |
| `mean_pagerank_pct`, `n_role_hubs` | avg member PageRank percentile; count of R5–R7 roles | Community importance profile |

### 3.5 Community temporal metrics (partition matched across months, §5)

| Metric | Definition | Question |
|---|---|---|
| `persistence_jaccard` | \|C_t ∩ C_t−1\| / \|C_t ∪ C_t−1\| vs best-matched predecessor | Ecosystem stability |
| `event` | born / continued / grown / shrunk / split-suspect / absorbed | Lifecycle tracking; a *born* community of low-tenure nodes with high internal reciprocity is a fraud-ring shaped object |
| `growth_rate` | n_t / n_matched,t−1 | Market growth analytics feed (geospatial phase) |
| `mass_growth` | w_int,t / w_int,t−1 | Dollars, not just membership |

---

## 4. Graph-Level Metrics (per month × version)

| Group | Metrics | Formula / source | Question |
|---|---|---|---|
| **Size** | n_nodes, n_edges, total_amount, total_volume, avg/median degree & strength | cuDF | Growth of the observable payment economy |
| **Density & topology** | density = m/(n(n−1)); global transitivity (cuGraph triangles); max_core, top_core_size | cuGraph | Structural maturation over months |
| **Tail shape** | p99/p99.9/max of degree & strength; power-law slope α (OLS on log-log CCDF tail) | custom | Is hub dominance increasing? α drifting down = winner-take-all concentration |
| **Inequality** | Gini(degree), Gini(strength), Gini(pagerank); CR10/CR100 (top-10/100 share of total amount) | custom `gini`, `graph_distribution_stats` | "Top 100 customers move X% of every dollar" — and how that changes on the version ladder |
| **Reciprocity** | reciprocity_uw, reciprocity_w | custom §2.5 | One-way distribution economy vs bilateral trade economy |
| **Assortativity (4 directed combos × uw/w)** | Pearson over edges of (deg_src, deg_dst) for (out,in),(in,out),(out,out),(in,in); amount-weighted variants; strength versions on log1p | custom `assortativity_suite` | Do big payers pay big receivers (assortative core) or spray to the periphery (disassortative, typical of payment nets)? Trend breaks flag regime change |
| **Hierarchy** | trophic_incoherence F₀; mean/max trophic level | custom §2.4 | How circular is the whole payment system this month? F₀ creeping up graph-wide = more ring-like flow overall |
| **Partition quality** | modularity Q, n_communities, community-size Gini & entropy, coverage | cuGraph + custom | Is the economy modularizing or blending? |
| **Concentration of importance** | HHI of PageRank mass; HITS spectral gap proxy (‖iter k − iter k−1‖ decay) | custom | Single-hub capture diagnosis |
| **Rich-club (weighted, Opsahl)** | φ_w(r) = W_int(rich r) / Σ of the E_int strongest edge weights, r ∈ top {0.01%, 0.1%, 1%, 5%} by strength | custom `rich_club_weighted`, flag-gated | Do the biggest players transact preferentially *with each other* (elite clique) or only with the crowd? Track the curve monthly |
| **Excluded per §1** | SCC, WCC, bow-tie, exact diameter/avg path | — | (trophic incoherence + core decomposition cover the analytical intent) |

---

## 5. Cross-Snapshot (Temporal) Metrics

| Level | Metric | Definition / compute | Question |
|---|---|---|---|
| Edge | `edge_jaccard_uw` | \|E_t ∩ E_t−1\| / \|E_t ∪ E_t−1\| (pair identity) | Relationship churn rate of the whole network |
| Edge | `edge_overlap_w` | Σ min(w_t, w_t−1) / Σ max(w_t, w_t−1) over the pair union | Dollar-weighted stability — relationships persist, but do the *amounts*? |
| Node | entry/exit rates | from `is_new`/`is_churned` | Market entry dynamics |
| Partition | **NMI** (arithmetic normalization, sklearn-compatible) and **ARI** between consecutive monthly partitions on common nodes | custom from contingency table (cuDF groupby → CuPy) | Global community stability; a sudden NMI drop = structural regime change (seasonality, data issue, or genuine market shift) |
| Partition | community matching table | best-overlap match + Jaccard + event labels | Feeds §3.5 |
| Any score | rank-migration matrices | decile-to-decile transition counts for PageRank/strength | "Who moved into the top decile this month" — a stakeholder-friendly artifact |

---

## 6. Compute Plan (60 GB GPU · 3–5M nodes/edges · 5 versions/month)

| Stage | Items | Cost @5M edges | Notes |
|---|---|---|---|
| 1. Load & encode | factorize ids, drop self-loops, build node frame | seconds | one id map per month reused by all versions |
| 2. V0 node basics | §2.1–2.2 (pure cuDF) | seconds | thresholds for §1.2 come from these |
| 3. Version masks | build_versions → 4 exclusion masks + bitmask | trivial | log thresholds to `version_thresholds` |
| 4. Per version loop | PageRank×2, HITS_w, Katz, core, triangles, reciprocity, trophic, Louvain/Leiden, §2.6, §3 | tens of seconds each | ~≤2 GB peak; embarrassingly parallel across months if ever needed |
| 5. Gated extras | betweenness_approx (V2, V4 only), rich-club, internal reciprocity per community | minutes | `ENABLE_*` flags, default off for backfills |
| 6. Temporal joins | §2.7, §3.5, §5 vs t−1 | seconds | pure cuDF merges on original ids |

Practical notes: keep `amount` float64 (dollar sums overflow float32 semantics in HHI/entropy shares less than you'd think, but precision in min/max reciprocity merges matters); persist per-(month, version) Parquet under `../result/` mirroring the existing pipeline layout; all custom kernels are single-GPU and fit ~50× within budget — no partitioning needed at this scale.

---

## 7. Output Tables

| Table | Grain | Contents |
|---|---|---|
| `node_metrics` | node × month × version | all §2 metrics + survival bitmask |
| `node_temporal` | node × month | §2.7 deltas/z-scores/flags (V0 and V4 recommended) |
| `community_metrics` | community × month × version | §3 |
| `community_events` | community × month | §3.5 matches & lifecycle |
| `graph_metrics` | month × version | §4 (long format: metric, value) |
| `graph_temporal` | month | §5 |
| `version_thresholds` | month | D_soft/D_hard/S_soft/S_hard + counts/amount removed |
| `rank_stability` | month × metric × version-pair | Spearman ρ (§1.4) |
| `rich_club_curve` | month × version × quantile | φ_w(r) |

---

## 8. Interpretation Playbook (signal combinations)

| Pattern | Signature | Reading |
|---|---|---|
| **Funnel / consolidation mule** | passthrough ≥ 0.9 · in_degree ≫ out_degree · hhi_out high · clustering ≈ 0 · tenure low | UC #3 candidate; rank by amount |
| **Layering way-station** | passthrough high · mid trophic level · betweenness percentile high · degree modest | UC #7 candidate — position matters more than size |
| **Ring territory** | community with local_incoherence high + internal_reciprocity_w high + naics_entropy high | Scope Cypher ring queries (UC #6) to these communities instead of the whole graph |
| **Anchor-dependent ecosystem** | top1_strength_share > 0.5 · embeddedness_w high | Retention risk: protect the anchor client |
| **Connector client (cross-sell)** | participation_coeff high · role R6/R7 · pagerank stable | Bridges ecosystems — natural product-referral path (UC #14 features) |
| **Behavioral drift event** | z_baseline > 3 on flow_ratio or avg_ticket · role transition · community_switched | UC #15 primitive alert, customer-relative so fatigue-resistant |
| **Hub artifact check** | any finding that vanishes from V0→V1 | It was infrastructure, not behavior — report on V2/V4 instead |
| **Elite market formation** | rich-club φ_w rising over months at r = top 0.1% | The largest clients increasingly transact among themselves — correspondent/liquidity intelligence |

---

*Companion implementation: `pkg_custom_metrics.py` (all custom formulas; cuGraph natives assumed provided by the existing `pkg_graph_metrics.py` pipeline). Custom formulas follow NetworkX / scikit-learn conventions where an equivalent exists (reciprocity, HITS normalization, NMI arithmetic normalization, ARI, Gini) for validation on small samples.*
