# PKG Counterparty Expansion Manifest
**New Metrics, Roles, and Applications Unlocked by Counterparty Data**
*PNC Bank · Treasury Management · Data Science — Planning Document, July 2026*

Companion to: `PKG_MONTHLY_METRICS_MANIFEST.md`, `PKG_ROLES_MANIFEST.md`, `PKN_Roadmap.md`
Roadmap linkage: UC #11 (Counterparty Ecosystem), UC #12 (Interbank Flow), UC #18 (Full Ecosystem Intelligence), Phase 2

---

## 1. Purpose

This document specifies what changes in the Payment Knowledge Graph analytics stack when counterparty (outside-PNC) data is integrated: the structural shift in the graph, the new node-level metrics on both the customer side and the counterparty side, extensions to the role taxonomy, the impact on the similarity feature store, and the use cases these unlock. It is written to serve as a seed document for future implementation threads and assumes familiarity with the existing monthly pipeline (`pkg_custom_metrics.py`, `pkg_pipeline.py`, `pkg_roles.py`), the ablation ladder, and the versioned feature-store plan.

---

## 2. What Changes Structurally

### 2.1 From closed graph to asymmetrically observed bipartite-extended graph

The current PKG is a **closed graph**: every node on both ends of every edge is a fully known entity — KYC identity, NAICS, geography, product context, and a complete view of its PNC-side flow history. Counterparty integration breaks this symmetry. The expanded graph has two node classes:

| Property | PNC Customer node ("thick") | Counterparty node ("thin") |
|---|---|---|
| Identity | Verified (KYC) | Inferred from payment strings; entity resolution required |
| NAICS | Known (with known quality issues) | Absent — must be inferred behaviorally |
| Flow visibility | Complete (all rails, both directions) | Partial — only the slice touching PNC |
| Geography | Known | Absent or inferred from connected customers |
| Product context | Available (holdings data incoming) | None |
| Actionability | Servicing, cross-sell, risk | Prospecting, risk exposure, ecosystem context |

The defining analytical consequence: for counterparty nodes, **every attribute is an inference and every flow figure is a lower bound**. Metric definitions below are written with this in mind — counterparty metrics measure *observed PNC-side interaction*, never total activity.

### 2.2 Data assets and what each unlocks

| Asset | Status (per roadmap) | Unlocks |
|---|---|---|
| CPTY_PAYS (counterparty → customer, inbound) | Ingested | Inbound external metrics; partial embeddedness; collection-side inference |
| PAYS_CPTY (customer → counterparty, outbound) | Upcoming | Outbound external metrics; full on-us ratio; disbursement-side inference; symmetric ecosystem view |
| CptyFinEntity (counterparty ↔ financial institution linkage) | Future | Interbank flow intelligence, deposit-leakage attribution, cross-border share, bank-level entity grouping |

Until PAYS_CPTY lands, all "external" metrics exist in inbound-only form. Schema for outbound equivalents should be defined now (symmetrical naming) so ingestion is a column-fill, not a redesign — consistent with the roadmap's mitigation note on PAYS_CPTY delay.

### 2.3 Scale and hub consequences

Counterparty integration moves the graph from 3–5M nodes/edges per monthly snapshot toward the hundreds-of-millions regime described in the roadmap. Two direct consequences for the existing architecture:

1. **The ablation ladder must be re-derived on the expanded graph.** The current P99/P99.9/P99.99 thresholds were fit to the customer-only degree/strength distribution. External mega-hubs (payment processors, payroll providers, government/tax entities, card networks) will dominate the new tail and are categorically non-actionable. Expect the hub registry to grow substantially and to need a *typed* exclusion (see counterparty roles, §5.1) rather than purely percentile-based exclusion.
2. **Per-month streaming discipline becomes mandatory, not preferential.** The no-in-memory-accumulation architecture adopted after the OOM event is the correct posture for the expanded graph; any metric requiring the aggregate multi-month external graph must be computed incrementally or on the ablated graph only.

### 2.4 Precondition: entity resolution

Counterparty nodes arrive fragmented — the same real-world entity appears under name variants, abbreviations, and multiple account identifiers. Fragmentation silently destroys the most valuable counterparty metrics (embeddedness and capturable flow are undercounted; shared-counterparty similarity loses recall). Entity resolution is therefore a **pipeline precondition, not an enhancement**:

- Reuse the self-payment detection machinery: Jaro-Winkler scoring (jellyfish/rapidfuzz), first-token blocking, tuned thresholds.
- CptyFinEntity, when available, provides bank-level grouping as a resolution aid and a validation signal.
- Output: a `cpty_resolved_id` mapping table (raw identifier → resolved entity), versioned per snapshot vintage, applied before any counterparty metric computation.
- Every counterparty metric below is defined **on resolved entities**.

---

## 3. New Metrics — Customer Side

These extend the existing monthly node parquet (same stacking, same month-1 NaN conventions where applicable). Naming follows existing `in_`/`out_` prefix conventions; `int_` = internal (PNC-to-PNC), `ext_` = external (counterparty-facing).

### 3.1 On-us ratio / wallet visibility — the flagship addition

For the first time the denominator "total observed flow" includes off-PNC activity, making these definable:

| Column | Definition |
|---|---|
| `on_us_out_amount_share` | Internal out-amount / total out-amount |
| `on_us_in_amount_share` | Internal in-amount / total in-amount |
| `on_us_out_txn_share`, `on_us_in_txn_share` | Count-based equivalents |
| `on_us_degree_share_out/in` | Internal degree / total degree per direction |

Interpretation and applications: a **declining on-us share across snapshots is a relationship-migration early warning** (feeds Behavioral Drift, UC #15); a **low on-us share on a high-value customer is a payments/deposit expansion target**; amount-share vs. count-share divergence indicates whether large-ticket or routine flow is leaving. The trend of this metric (downstream, from stacked months) is likely the single most commercially interesting derived series in the expanded system.

### 3.2 Internal/external split of existing metrics

The concentration metrics agreed for the similarity work (top-1/top-3 share, HHI per direction) and the degree/strength/txn-count family should each gain internal/external variants:

| Family | New columns (pattern) |
|---|---|
| Degree | `int_in_degree, int_out_degree, ext_in_degree, ext_out_degree` |
| Strength | `int/ext × in/out_amount` |
| Txn counts | `int/ext × in/out_txn_count` |
| Concentration | `ext_in_top1_share, ext_in_hhi, ext_out_top1_share, ext_out_hhi` (+ top-3) |

Note: overall (undifferentiated) variants remain the canonical similarity inputs for continuity; the split variants add the new behavioral axis "how externally oriented is this customer" without breaking feature-store version comparability (they enter as a new block, §6).

### 3.3 External exposure and dependency

| Column | Definition | Primary consumer |
|---|---|---|
| `ext_in_dependency` | Largest single external counterparty's share of total in-amount | Credit/Risk — revenue concentration on an entity PNC cannot observe fully |
| `ext_out_dependency` | Largest single external counterparty's share of total out-amount | Risk, TM advisory — critical-supplier exposure |
| `ext_cpty_count_in/out` | Distinct resolved external counterparties per direction | Sales sizing, ecosystem breadth |

### 3.4 Post-CptyFinEntity additions

| Column | Definition |
|---|---|
| `bank_out_hhi`, `bank_in_hhi` | Concentration of external flow across destination/origin financial institutions |
| `top_ext_bank_out_share` | Share of external out-amount going to the single largest receiving institution |
| `cross_border_out_share`, `cross_border_in_share` | Share of external flow to/from non-US institutions (FX/international product signal) |

---

## 4. New Metrics — Counterparty Side

A **new monthly node table** (`cpty_nodes` parquet per month, stacked like customer outputs) on resolved counterparty entities. All values are PNC-observed lower bounds by construction.

### 4.1 Embeddedness (the raw prospect signal)

| Column | Definition |
|---|---|
| `pnc_customer_count_in` | Distinct PNC customers paying this counterparty this month |
| `pnc_customer_count_out` | Distinct PNC customers this counterparty pays |
| `captured_in_amount`, `captured_out_amount` | PNC-observed flow to/from this counterparty |
| `captured_edge_count` | Total PNC-side edges |
| `customer_naics_breadth` | Distinct NAICS2 groups among connected customers (count or entropy) |
| `customer_geo_breadth` | Distinct customer geographies among connections |
| `persistence_months` | Months present in the observation window (downstream from stack) |
| `first_seen`, `last_seen` | Snapshot bounds |
| `captured_flow_trend` | Slope of log monthly captured amount (downstream) |

### 4.2 Capturable flow (the prospect value quantifier)

For a prospect counterparty, **capturable flow = the monthly volume and edge count that become on-us if PNC wins the relationship** — i.e., its captured totals reread as a conversion value. This converts prospecting from firmographic guesswork into a quantified network-closure figure (on-us settlement, pricing, deposit capture), and it is inherently proprietary: no external prospecting dataset can produce it. Downstream composite:

```
prospect_score = f(captured amount level, customer count, persistence,
                   growth trend, NAICS breadth, geo proximity to PNC footprint)
```

Weighting to be tuned with TM Sales; ship the components, not only the composite, so the score is explainable per prospect.

### 4.3 Inferred attributes (shadow firmographics)

| Column | Inference basis |
|---|---|
| `inferred_naics`, `inferred_naics_conf` | Distribution of connected customers' NAICS, weighted by amount; pattern signatures (e.g., semi-monthly inflows from many employers → payroll/benefits) |
| `inferred_size_class` | Captured flow magnitude percentile (explicit lower-bound caveat) |
| `inferred_geo` | Modal/weighted geography of connected customers |
| `flow_asymmetry` | (in − out)/(in + out) on captured flow → supplier-like vs. buyer-like vs. two-way |

These inferences also feed back into the customer graph: behaviorally inferred NAICS for counterparties is the same machinery the roadmap proposes for **customer NAICS imputation** (-1|UNKNOWN cleanup) — one inference module, two applications.

### 4.4 Structural metrics on the bipartite projection

| Metric | Definition | Note |
|---|---|---|
| `shared_customer_overlap` | Pairwise Jaccard/overlap of counterparties' PNC customer sets | Compute top-K per counterparty only, hub-excluded; full pairwise is infeasible and unnecessary |
| `cpty_hub_flag`, `cpty_hub_type` | Membership + type in the extended hub registry | Typed via counterparty roles (§5.1) |

---

## 5. Role Taxonomy Extensions (`pkg_roles.py`)

### 5.1 New: counterparty role taxonomy

Assigned per resolved counterparty from captured-flow signatures. Primary purposes: typed hub exclusion, prospect filtering, and inference support.

| Role | Signature (heuristics, v1) | Treatment |
|---|---|---|
| `PAYROLL_PROCESSOR` | Very high out-fan to consumer-like accounts; semi-monthly/biweekly cadence; many employer inflows | Hub-exclude; never a prospect |
| `BENEFITS_PROVIDER` | Semi-monthly inflows from many employers, moderate out-fan | Hub-exclude |
| `GOV_TAX` | Massive in-fan, calendar-quarter concentration, near-zero outflow | Hub-exclude; never a prospect |
| `MERCHANT_ACQUIRER / PROCESSOR` | Extreme fan both directions, small tickets, daily settlement pattern | Hub-exclude |
| `FINANCIAL_INFRA` | Flow signature of bank/settlement accounts; confirmed via CptyFinEntity when available | Hub-exclude; route to interbank analytics instead |
| `MARKETPLACE_AGGREGATOR` | High in-fan of small consumerish amounts + periodic large payouts to businesses | Case-by-case; ecosystem anchor |
| `SUPPLIER_HUB` | High in-degree from many PNC business customers, business-sized tickets | **Prime prospect class** |
| `BUYER_HUB` | High out-degree to many PNC business customers | **Prime prospect class** |
| `TWO_WAY_TRADER` | Balanced reciprocated flow with a moderate set of customers | Prospect; supply-chain analytics |
| `PERIPHERAL` | Low degree, low persistence | Default; bulk of nodes |

Cadence-based signatures sharpen materially once the planned daily-grain periodicity companion table exists; v1 roles are definable from monthly data alone with reduced precision.

### 5.2 Extended: customer role taxonomy

New roles/modifiers enabled by the internal/external split:

| Role / modifier | Definition | Why it matters |
|---|---|---|
| `GATEWAY` | High share of the customer's flow bridges internal ↔ external ecosystems | These customers anchor warm-intro paths to prospects; sales priority |
| `OFF_US_LEANING` / `ON_US_ANCHORED` | Modifier from on-us amount share (thresholds TBD vs. distribution) | Retention vs. expansion framing per customer |
| `EXT_CONCENTRATED` | `ext_in_dependency` or `ext_out_dependency` above threshold | Risk flag; TM advisory conversation |
| `MIGRATING` | Sustained multi-month decline in on-us share | Early-warning role; feeds drift detection (UC #15) |

Existing roles (funnel, distributor, pass-through, etc.) should be **recomputed on the expanded graph**: patterns invisible in the closed graph (e.g., a pass-through whose inflow is internal and outflow external) become detectable, and some current role assignments will change once external edges complete the picture. Expect and document role churn at the integration boundary.

---

## 6. Similarity Feature Store Impact

Per the versioning scheme (v1 = current + agreed additions, v2 = rail, v3 = periodicity, v4 = deposit), counterparty integration contributes two new blocks — a version increment, not a rearchitecture:

| New block | Contents | Notes |
|---|---|---|
| **External-orientation block** | On-us shares (amount/count/degree), ext dependency, ext counterparty counts, int/ext concentration deltas | Standard treatment: log1p where scaled, rank/quantile transform, cosine within block |
| **Shared-neighbor block** | Jaccard/overlap similarity on **resolved, hub-excluded** counterparty sets (per direction) | First *structural* similarity block; two firms paying the same supplier network are almost certainly in the same real supply chain. Hub exclusion is a hard precondition — with hubs in, every customer is "similar" via ADP/IRS |

Neighbor parquets gain the corresponding decomposition columns; weights move in config as designed. Cross-version neighbor comparison remains prohibited.

---

## 7. Use Cases and Applications Unlocked

Numbered items reference the PKN Roadmap inventory where applicable.

| Use case | Team(s) | What cpty data adds | Depends on | New? |
|---|---|---|---|---|
| **Prospect Radar** — ranked counterparty prospects with capturable-flow sizing and warm-intro ego view ("already transacts with N PNC clients") | TM Sales, Relationship Mgmt | The entire capability: embeddedness, capturable flow, inferred firmographics | PAYS_CPTY + entity resolution; compliance clearance | New (extends the roadmap's "shadow prospect intelligence" intent) |
| **Wallet-share & relationship-migration monitor** — on-us share levels and trends per customer | TM Sales, Liquidity, Retention | On-us denominator; trend series | PAYS_CPTY | New; feeds UC #15 |
| **Deposit-leakage / interbank flow intelligence** (UC #12) — largest external sinks of PNC outflow by bank, region, NAICS | Liquidity, Correspondent Banking | Attribution of external flow to institutions | CptyFinEntity | Roadmap item, now specified |
| **Counterparty ecosystem viewer** (UC #11) — bipartite view, shared-counterparty detection | Correspondent Banking, Risk, Corporate Analytics | Shared-customer overlap, resolved entities | PAYS_CPTY (partial now via CPTY_PAYS) | Roadmap item |
| **External concentration risk** — customers dependent on one unobservable external entity | Credit, Risk, TM advisory | `ext_in/out_dependency` | CPTY_PAYS now (inbound); PAYS_CPTY for outbound | New |
| **Supply-chain community detection** — communities over the expanded graph reflect *real* ecosystems, not just intra-PNC fragments (UC #13 upgrade) | Corporate Analytics, Product | External edges complete the actual supply-chain topology | PAYS_CPTY; GPU (cuGraph) | Upgrade |
| **AML: external funnels, layering, structuring** (UC #3/#6/#7 upgrades) | Fraud/AML | Chains that exit and re-enter PNC become traceable one hop further; external funnel endpoints visible | PAYS_CPTY; typed hub exclusion to control false positives | Upgrade |
| **Cross-border / FX opportunity detection** — customers with material non-US external flow | TM Sales (International), FX | `cross_border_*_share` | CptyFinEntity | New |
| **Shared-counterparty similarity & cross-sell enrichment** — structural twin detection; supply-chain-aware look-alikes | TM Sales, Product | Shared-neighbor block (§6); combined with product holdings when available | PAYS_CPTY + holdings data | Extension of similarity program |
| **NAICS imputation for customers** — behavioral industry inference reused from cpty inference module | Corporate Analytics (UC #5 enabler) | Denser behavioral evidence per node | Inference module (§4.3) | Roadmap risk-mitigation item, now with a concrete mechanism |

### 7.1 Presentation concepts

**Prospect Radar (TM Sales):** ranked table (prospect score + decomposed components), drill-through to an ego view with the prospect centered among its existing PNC relationships — the "warm introduction" framing is the demo moment. Filters: inferred NAICS, geography, role class (`SUPPLIER_HUB`/`BUYER_HUB`/`TWO_WAY_TRADER` only), minimum persistence.

**Customer page additions:** an on-us gauge with trend sparkline; external dependency callout when flagged; "external ecosystem" mini-panel (top resolved counterparties, typed); similarity panel gains the shared-counterparty mode alongside behavioral twins.

**Liquidity/leadership view:** Sankey or flow-map of aggregate external flow by destination institution (post-CptyFinEntity), sliceable by NAICS and region — pairs naturally with the geospatial corridor work (UC #8).

All views follow the standing app conventions (three-layer architecture, `logic.py` → FastAPI liftable, dark theme, Plotly, `PKG_BACKEND` switching).

---

## 8. Governance, Risk, and Caveats

| Item | Nature | Position |
|---|---|---|
| **Permissible use for prospecting** | Using customer transaction data to build intelligence on non-customers has data-governance and permissible-use implications | Obtain a compliance read **before** any Prospect Radar demo circulates; frame v1 outputs as "network intelligence" pending clearance |
| **Lower-bound semantics** | All counterparty figures are PNC-observed minima | Bake into UI copy and score documentation; never present captured flow as "counterparty revenue" |
| **Entity-resolution error** | False merges inflate embeddedness; false splits deflate it | Version the resolution mapping; publish precision spot-check results with each version; scores cite resolution version |
| **Hub false positives in AML upgrades** | External hubs create spurious funnel/ring/layering candidates | Typed exclusion (§5.1) is a precondition for running upgraded AML motifs |
| **Role churn at integration boundary** | Customer roles recomputed on the expanded graph will shift | Communicate as a one-time restatement; keep pre/post mapping for analyst trust |
| **Compute scaling** | Hundreds of millions of nodes/edges | Streaming per-month discipline; ablated graph for iterative algorithms; top-K-only pairwise computations |

---

## 9. Phasing

| Phase | Trigger | Deliverables |
|---|---|---|
| **A — Now (CPTY_PAYS only)** | Data already ingested | Entity-resolution pipeline v1; inbound-only external metrics (`ext_in_*`, inbound embeddedness); counterparty node table v0; schema defined symmetrically for outbound |
| **B — PAYS_CPTY lands** | Ingestion milestone | Full on-us ratios; outbound metrics; capturable flow; prospect score v1; counterparty roles v1; re-derived ablation ladder + typed hub registry; customer role recomputation; shared-neighbor similarity block; Prospect Radar prototype (post compliance read) |
| **C — CptyFinEntity lands** | Linkage milestone | Bank-level metrics; deposit-leakage dashboard; cross-border shares; resolution validation against bank grouping; interbank analytics (UC #12) |
| **D — Convergence** | Holdings/deposit data + periodicity companion available | Prospect score with cadence-sharpened roles; cross-sell enrichment with structural similarity; full ecosystem intelligence (UC #18) |

Phase A is deliberately non-trivial: entity resolution and symmetric schema design are the long-lead items, and both can start against CPTY_PAYS today.

---

## 10. Open Questions

1. Counterparty identifier structure in CPTY_PAYS/PAYS_CPTY — account-level, name-level, or both? Determines resolution pipeline design.
2. Name quality in `cptyNamePrmry` (flagged sparse in roadmap) — what coverage %, and is there a secondary name field?
3. Will PAYS_CPTY arrive with historical backfill (matching the 2024–2025 window) or forward-only? Determines whether counterparty trend metrics have depth at launch.
4. Compliance owner and process for the prospecting permissible-use question — who signs off, and what evidence do they need?
5. Consumer counterparties: in scope for the counterparty node table, or filtered to business-like entities only? Materially affects scale and role taxonomy.
6. Does the ablation ladder remain percentile-based on the expanded graph, or move to typed-role exclusion as primary with percentile as backstop?

---

*Prepared as a seed/planning document. Update alongside `PKG_MONTHLY_METRICS_MANIFEST.md` and `PKG_ROLES_MANIFEST.md` when implementation begins.*
