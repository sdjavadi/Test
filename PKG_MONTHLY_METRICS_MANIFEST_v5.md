# PKG Monthly Snapshot Metrics Manifest — v3

**Payment Knowledge Graph (PKG) — Customer-to-Customer Monthly Network Analysis**
*PNC Bank · Treasury Management · Data Science*

> **v2 → v3.** Three structural changes:
> 1. **Customer attributes move out of the snapshot into `customers.csv`.** The
>    snapshot is now `source, dest, amount, volume` only. Name, NAICS, typing and
>    geography come from a single customer dimension.
> 2. **Geography is a first-class metric block** — 24 columns per node per
>    version (§2.10). Full spec: `PKG_GEO_MANIFEST.md` v1.1.
> 3. **Ladder reduced to three versions** (V0 / P99 / P99_9) and **reused from
>    disk** rather than rebuilt each run.
>
> Plus: runtime instrumentation with a session-budget projector, tracker
> checkpointing for clean resume, and the divide / cuGraph warnings **fixed
> rather than documented as benign**.

---

## 0. Scope & Conventions

### Inputs

| File | Schema | Notes |
|---|---|---|
| `../data/cust_YYYY-MM.csv` | `source, dest, amount, volume` | Only these four columns are read. Identity / NAICS columns, if still present, are ignored |
| `../data/customers.csv` | `mdm_id, customer_name, naics_cd, naics_desc, addr_loc_rec_type, longitude_degree, latitude_degree, zip_cd, state_or_province, city` | One row per customer, **current state** |
| `../metrics/ladder_thresholds.csv`, `ladder_exclusions.csv` | pre-computed | Reused when `REUSE_LADDER = True` |

**Why the split.** Attributes get corrected and enriched; edges do not.
Conflating them meant every attribute improvement required reprocessing history,
and it forced two full passes over all 23 snapshots (ladder build + node typing)
before any metric was computed. Both passes are gone.

**Current-state, not point-in-time.** One customer row applies to every
historical snapshot. Location, name and NAICS rarely change, and a correct
current value is a better estimate of last March than a stale snapshot one.
**Consequence: re-running after a customer refresh changes historical metrics.**
That is deliberate — bump `ATTR_VERSION` in `pkg_customers.py` when it happens.

**NAICS arrives as separate code and description columns.** No `|` splitting.
Absence is an **empty string, not a placeholder token** — the `-1|UNKNOWN` /
`******|UNKNOWN` conflation that drove the v2 typing precedence rule is resolved
upstream. `PLACEHOLDER_NAICS` is retained as a residual guard so a regression
shows up in the logged counts rather than silently classifying as valid.

### Conventions

- **Edge weight = `amount` only.** `volume` is descriptive (a transaction count),
  never a weight.
- **Directed**: `source → dest` = source *paid* dest.
- **IDs are strings end to end.** `mdm_id` on both sides, never coerced to int.
- **Weight-scale policy**: `log1p(amount)` for spectral / iterative metrics
  (HITS, `pagerank_logw`, trophic, Louvain); **raw amount** for flow / accounting
  metrics; **both** where the comparison is informative — `pagerank_raw` vs
  `pagerank_logw`: high on raw but not log = whale beneficiary; high on log but
  not raw = broadly connected, dollar-modest.
- **SCC excluded**; WCC on de-hubbed versions only.
- **Scale envelope**: 3–5M nodes / edges per snapshot; single GPU ≤ 60 GB.

### Outputs

```
../metrics/node/node_{YYYY-MM}.parquet        # all versions stacked, one row per (version, node)
../metrics/graph/graph_{YYYY-MM}.csv          # one row per version
../metrics/dist/dist_{YYYY-MM}.csv            # distribution summary per version
../metrics/community/community_{YYYY-MM}.parquet
../metrics/hub/hub_{YYYY-MM}.csv              # per-registry-hub monthly summary
../metrics/customers_dim.parquet              # customer dimension, written once
../metrics/node_typing.csv                    # entity_type / naics_status / node_type
../metrics/ladder_thresholds.csv              # audit
../metrics/ladder_exclusions.csv              # hub registry
../metrics/run_timings.csv                    # per-component profile
../metrics/_tracker_state.pkl                 # cross-month checkpoint
```

**Two run modes.** The default computes everything. `GEO_ONLY = True` computes
only the geographic block and writes `../metrics/geo/geo_{YYYY-MM}.parquet`,
skipping graph metrics entirely. Both key on `(time_key, version, node)`, so the
outputs combine with a plain merge — verified numerically identical to a
combined run:

```python
node = pd.read_parquet(".../node/node_2024-01.parquet")
geo  = pd.read_parquet(".../geo/geo_2024-01.parquet")
full = node.merge(geo, on=["time_key", "version", "node"], how="left")
```

**Resumable.** A month whose node file exists is skipped; the geo-only pass has
its own independent skip-check on `geo_{month}.parquet`. Cross-month tracker
state (`TemporalTracker`, `prev_partition`) is checkpointed after each month and
restored on restart, so **the v2 caveat about turnover / tenure / NMI restarting
NaN after a resume no longer applies** (`PERSIST_TRACKERS = True`). The
checkpoint is written to a temp file and `os.replace`d — never a half-written
state.

**Schema-stability rule.** Columns are never pruned dynamically, even when
all-zero or mostly-NaN in a given month. Per-month schema drift breaks the Hive
tables and cross-month concat. Legitimately sparse columns carry information in
their NaNs.

---

## 1. Ablation Ladder — three versions

Exclusion sets are computed **once from the union graph aggregated over all
snapshots**, so membership is stable across time.

| Version | Definition | Intent |
|---|---|---|
| **V0** | Full snapshot | Ground truth incl. dominance; **an infrastructure diagnostic, not an analytical base** |
| **P99_9** | Exclude aggregate degree > P99.9 **OR** strength > P99.9 | **Primary analytical graph**; also the hub registry for hub-exposure metrics |
| **P99** | Same at P99 (~1% of nodes) | Aggressive de-hubbing; sensitivity bound |

P99_99 is **removed** in v3 — measured at 10.3 min/month (4 h across the run)
for a tier that never carried an analytical decision. `build_ladder`'s default
is now `percentiles=(99.0, 99.9)`; pass the third tier explicitly if a
sensitivity check needs it.

**Why the union of degree and strength criteria?** In the 2024–2025 aggregate the
P99.9 degree-hub set and the strength-whale set overlap only ~60%. Degree-only
exclusion leaves the whales; strength-only leaves the hubs. One union list
removes both failure modes.

**`REUSE_LADDER = True`** loads `ladder_thresholds.csv` + `ladder_exclusions.csv`
from disk, falling back to a rebuild with a warning if either is absent. This
removes 23 CSV reads from every run.

**How to read the ladder.** V0 → P99_9 isolates payment infrastructure;
P99_9 → P99 peels off ordinary hubs. A metric stable across all three is a
genuinely distributed property. **A metric that changes at V0 → P99_9 was carried
by infrastructure** — at production scale the top 0.1% of nodes hold **97.3% of
the dollars**, and that one transition has already reversed six geographic
findings.

**Standing rule:** no result is reportable until computed at two adjacent rungs
and shown to agree.

---

## 2. Node-Level Metrics

Output: `../metrics/node/node_{YYYY-MM}.parquet`, one row per (version, node).

Every row carries the **full customer passthrough**, so downstream apps need no
dimension join: `cust_name`, `naics_cd`, `naics_desc`, `addr_loc_rec_type`,
`lat`, `lon`, `zip5`, `zip3`, `state`, `city`, `geo_status`, `shared_structure`,
`naics2`–`naics6`, `naics_known`, `entity_type`, `naics_status`, `node_type`.

Nodes absent from `customers.csv` are emitted with NA attributes and
`geo_status = 'not_in_customers'`, so coverage stays visible rather than silent.
Attribute coverage over the graph node universe is logged on the first month.

*(§2.1–2.7 unchanged from v2: flow magnitude & balance; concentration;
spectral / centrality; local structure; node-level community metrics;
multi-window counterparty cohorts; hub exposure.)*

### 2.8 NAICS hierarchy & industry mix

`naics_cd` is expanded into `naics2 … naics6` (first K digits, NA when shorter)
plus `naics_known`, **once in `pkg_customers.py`** rather than per snapshot.
These are identifiers, not metrics — they exist so any node metric can be
percentiled **within its industry peer group at any granularity**.

Counterparty-mix metrics use the **2-digit sector only** — finer levels are too
sparse per node to be stable. `same_naics2_in_share` / `same_naics2_out_share`
(industry homophily) are retained; entropy / top-share / n come from §2.9 under
the ≥5-counterparty guard.

### 2.9 Counterparty composition — entity typing & NAICS-status decomposition

**Motivation.** NAICS missingness confounds two different things: the
counterparty being an *individual*, vs a *business that wasn't enriched*. Node
types are decomposed first; behavioural composition is kept strictly separate
from data-quality coverage.

**Typing** now derives from `customers.csv` — one pass, not 23:

| Field | Values | Rule |
|---|---|---|
| `naics_status` | valid / placeholder / missing | Empty → missing; residual sentinel list → placeholder (expected ≈0, logged loudly if not); ≥2 leading digits → valid |
| `entity_type` | business / individual / unknown | Vectorised token typer on `customer_name`; individual = 2–3 all-alpha tokens with no business token. No fuzzy per-pair scoring at 5M-node scale |
| `node_type` | business_naics_valid / _placeholder / _missing / individual / unknown | **Valid NAICS wins; otherwise the name decides.** The v2 note about the extraction `coalesce` injecting `******\|UNKNOWN` no longer applies — absence is empty in `customers.csv`, so `business_naics_missing` is now a reachable category rather than empty by construction |

`naics_clean` (valid only) is the observed source of truth. Any future
`naics_imputed` stays a separate column forever — imputed NAICS is never fed back
into composition metrics (circularity).

**Per-node monthly metrics** — computed on the **V0 raw graph** (node-local, no
ablation needed) and attached to every version; self-edges excluded; a direction
with zero edges → NaN for all its columns:

| Family | Definition | Questions |
|---|---|---|
| `share_{d}_{w}_{cat}`, d ∈ {in,out}, w ∈ {cp,amt}, cat ∈ {individual, biz_valid, biz_placeholder, biz_missing, unknown, **hub**} | Composition shares over `node_type`; sums to 1.0 within each (d,w) | `share_in_*_individual` **is the B2C share**. The `hub` category uses the ladder registry so processor-intermediated flow is visible rather than miscounted as business |
| `naics_coverage_biz_{d}` | biz_valid / all business counterparties, cp-counted | **Data quality, NOT behaviour** — route low values to enrichment, not to analysts |
| `naics2_entropy_{d}`, `naics2_top_share_{d}`, `n_naics2_{d}` | Amount-weighted sector mix over valid-NAICS counterparties; **NaN below 5** | Guarded diversity — with 1–2 counterparties, entropy measures counterparty count, not diversity |

**Fail-fast guard retained.** The typing join samples 100K edge nodes and raises
if it misses >50%. The 2026-07 incident — int64 edge ids against a str typing
table silently typing *every* counterparty `unknown`, with plausible-looking
shares — is now structurally prevented by the single-string-dtype policy, and the
guard stays as a second line.

**Interpretation cheatsheet.** High `share_out_cp_individual` with regular
cadence → payroll-like. High `share_in_cp_individual` → consumer collector
(property mgmt, utilities, subscriptions). Low individual share both directions →
pure B2B intermediary. **cp-weighted vs amt-weighted divergence is itself
informative** — a landlord shows many small consumer inflows and one large B2B
outflow.

> **Population warning.** At P99_9 the graph is **94.3% individuals**, carrying
> 48.4% of dollars. Any aggregate that does not partition on `node_type` is
> substantially a statement about households, not about Treasury Management
> customers. See `PKG_GEO_MANIFEST.md` §2.3.

### 2.10 Geographic metrics — **new in v3**

24 columns from `pkg_geo_metrics.geo_node_metrics(edges, coords)`, computed
**per version**. Full specification in `PKG_GEO_MANIFEST.md` §4 Block F.

> **Per version, not once on RAW.** Hub removal changes who a node's
> counterparties are; the V0 → P99_9 comparison reversed six geographic findings.
> A single RAW-graph geo block would carry that distortion into every version.

Two families, deliberately separated because they have different requirements and
different failure modes:

**SPREAD** — properties of the counterparty cloud. Needs only the
*counterparties'* coordinates, so it is defined for a node with **no location of
its own**. This is the family that transfers to counterparties later.

| Column | Definition |
|---|---|
| `geo_spread_km` | Amount-weighted radius of gyration about the counterparty centroid |
| `geo_spread_in_km`, `geo_spread_out_km` | Per direction — **revenue footprint vs supply footprint** |
| `geo_centroid_lat`, `geo_centroid_lon`, `geo_R` | Amount-weighted centroid (3D unit vectors) and its resultant length |
| `geo_n_zip3_80`, `geo_zip3_entropy` | Market breadth |
| `geo_home_zip3_share_{in,out}`, `geo_home_state_share_{in,out}` | Home-market concentration, over **located** dollars |
| `geo_locality_class` | LOCAL <50 km · REGIONAL <250 · MULTI_MARKET <1000 · NATIONAL ≥1000 |

**REACH** — distance from the node's **own** point. NaN without own coordinates,
**never zero**.

| Column | Definition |
|---|---|
| `geo_reach_mean_km`, `geo_reach_p50_km`, `geo_reach_p90_km` | Amount-weighted distance, mean and percentiles |
| `geo_registered_vs_flow_km` | Registered point vs flow-weighted centroid — **the representativeness test** |

**Coverage**, always co-reported: `geo_cov_amt_{in,out}`,
`geo_n_cp_located_{in,out}`.

**Schema stability.** `GEO_COLUMNS` in `pkg_geo_metrics.py` is the canonical
list; every column is emitted on every call, NaN-filled when a month/version
cannot support it. A month where no node has its own coordinates must not
silently drop `geo_reach_*`, or the parquet schema drifts and the cross-month
concat breaks.

**Guards.** Spread is NaN below 2 located counterparties — with one, spread is
trivially 0, which is not the same as "concentrated". Entropy is NaN below 5 —
Shannon entropy is biased low at small *n*. Home-share is **0.0**, not NaN, when
a node has located counterparties and a known own key but no match; NaN is
reserved for "nothing to compare". Home-share denominators are **located**
dollars, so the metric is not confounded with coverage.

**Reading them.** Geography does **not** predict customer size — dollars per node
are flat to mildly inverse across distance rings (0.87× within 50 km of
Pittsburgh, 1.16× beyond 1,000 km), stable in all 23 months. These columns answer
*how wide* a footprint is, *how consistent* with industry peers, and *how it
changes* — not how large the customer is.

**Not yet implemented**: peer normalisation (`{metric}_pctile_naics_size`). Raw
dispersion is degree- and industry-confounded and should be percentiled within
`naics2` × size-decile × `node_type` before reporting.

---

## 3. Graph-Level Metrics

*(Unchanged from v2: scale & flow; heterogeneity & tails; mixing, hierarchy,
segmentation; temporal panel; rewiring & residual connectivity.)*

**Hub summary change**: identity columns (`cust_name`, `naics_cd`, `naics_desc`,
`state`, `city`, `zip3`, `geo_status`) now come from the customer dimension,
not from snapshot name / NAICS columns.

---

## 4. Distribution Summary · 5. Community-Level Metrics

*(Unchanged from v2.)*

---

## 6. Implementation Inventory

| Component | Source | Weight | Notes |
|---|---|---|---|
| **Customer dimension** — identity, NAICS hierarchy, typing, geo status | **`pkg_customers.py`** | — | **new in v3**; one pass over `customers.csv`, replaces two full passes over 23 snapshots |
| **Geographic block** — spread / reach / coverage / locality | **`pkg_geo_metrics.py`** | raw amount | **new in v3**; per version; vectorised haversine, 3D-vector centroids |
| **Runtime instrumentation** — per-component timing, budget projector | **`pkg_runtime.py`** | — | **new in v3** |
| PageRank (raw + log1p), Louvain, core number, k-sample betweenness | cuGraph | per §0 | PageRank graph built with `store_transposed=True` |
| Weighted HITS (power iteration) | `pkg_custom_metrics.py` | log1p | cuGraph HITS is unweighted |
| Trophic levels + incoherence (MJR 2020, CG) | `pkg_custom_metrics.py` | log1p | **Jacobi preconditioner** added; relative residual logged on every call |
| Dyad-min weighted reciprocity (graph & node) | `pkg_custom_metrics.py` | raw | hash-join on reversed edges |
| Participation, within-module z, GA roles | `pkg_custom_metrics.py` | unweighted degrees | z-score divide now guarded |
| Intra-community fractions; turnover / tenure / contagion (`TemporalTracker`); hub exposure | `pkg_pipeline.py` | raw + unweighted | tracker state now checkpointed |
| 4-way assortativity, Gini / Hill / top-share, rich-club, NMI / ARI | `pkg_custom_metrics.py` | raw | |
| Ladder builder + **`load_ladder`** | `pkg_custom_metrics.py` | — | union of degree / strength per tier; reuse from disk |
| Counterparty composition, coverage, guarded sector mix | `pkg_custom_metrics.py` | raw | typing from the customer dimension; >50%-miss fail-fast guard retained |
| Clustering coefficient (undirected, NaN k<2) | `pkg_pipeline.py` | unweighted | cuGraph `triangle_count`; CPU A³-diagonal fallback (dev only) |
| Orchestration, checkpointing, ETA | `pkg_pipeline.py` | — | `safe()` isolation; NaN frames on per-metric failure |

### 6.1 Warnings — fixed, not documented as benign

v2 recorded three warning classes as "confirmed benign". Two were masking real
correctness issues. All three are now fixed.

| Warning | Cause | Fix |
|---|---|---|
| `invalid value encountered in divide` (×14) | `np.where(cond, a/b, x)` **still evaluates `a/b` for every element** and only then discards the masked ones | `_sdiv(num, den, default)` — divides only where the denominator is positive and finite. 12 call sites in `pkg_pipeline.py`, 2 in `pkg_custom_metrics.py` |
| cuGraph `store_transposed` | PageRank wants the transposed CSR and re-transposes internally | `store_transposed=True` at graph construction — removes the warning *and* the redundant transpose |
| `trophic CG hit max_iter` | The Laplacian diagonal spans orders of magnitude at 5M nodes; unpreconditioned CG stalls | Jacobi preconditioner; `TROPHIC_TOL` / `TROPHIC_MAXITER` tunable; **relative residual logged either way**, so non-convergence is quantified rather than merely announced |

**Nothing is globally suppressed.** A blanket `filterwarnings("ignore")` would
have hidden the divide bug, which was worth seeing. Verified: the pipeline runs
clean end-to-end under `python -W error::RuntimeWarning`.

### 6.2 Runtime and the session budget

`pkg_runtime.py` records one line per component per (month, version) and writes
`run_timings.csv`. After each month the projector logs elapsed time, minutes per
month, **projected total, remaining, and the three slowest components** — so a
run that will overrun is visible by month three rather than at hour seven.

```python
TIME_BUDGET_H    = 8.0     # drives the projector; warns as it approaches
GEO_METRICS      = True    # False for a clean pre-geo baseline
GEO_ONLY         = False   # geo block only -> ../metrics/geo/
REUSE_LADDER     = True
PERSIST_TRACKERS = True
SNAPSHOT_COLS    = ["source", "dest", "amount", "volume"]
```

**Measured, first production attempt (2026-08-11, 2024-01, four versions):**
61.2 min/month → **23.5 h projected against an 8 h session.** The projector
raised `PROJECTION EXCEEDS BUDGET` after month 1 and the run was killed with
month 1 checkpointed. Three fixes followed, all measured:

| | Before | After |
|---|---|---|
| Geo, per (month, version) | 405 s V0 / 314 s P99 | ~4× faster; fixed cost removed |
| Fourth version (P99_99) | 10.3 min/month | removed from the ladder default |
| Trophic | 154 s, `max_iter` hit every time | **10.8 s, converged** (`rel_residual ≈ 9.5e-07`) |

**The geo diagnosis is worth recording.** Cost was `≈287 s + 43 s per million
edges` — P99 had 4.4× fewer edges than V0 but still took 77% as long, so almost
all of it was fixed cost. Two causes, both in how the coordinate table was
addressed rather than in the geometry:

1. `coords.set_index("node")` + `.reindex()` ran 12 times per month against the
   **full ~20 M-row customer table**, while a month touches ~2 M nodes.
2. Every `groupby("node")` re-factorized an **arrow-backed string** column —
   ~0.19 s per groupby, dozens of groupbys per call.

Fixed by `GeoIndex`: restrict to the month's nodes, factorize ids to `int32`
codes **once per month**, hold plain numpy arrays, and group on codes
throughout — ids are restored once at the boundary. Profiled before and after;
the 7.3 s `arrow.factorize` and 9.1 s `isin` entries are gone.

**Revised budget** (measured at 1.2 M edges, extrapolated):

| Run | Config | Total |
|---|---|---|
| 1 — metrics | `GEO_METRICS=False`, 3 versions | **≈6.0 h** |
| 2 — geo only | `GEO_ONLY=True`, 3 versions | **≈1.1 h** |
| or combined | both, 3 versions | ≈7.1 h |

Splitting is preferred: it fits each run comfortably inside one session, and
**geo can be re-run after a fix without repeating hours of centrality work** —
which matters while geo is the newest and least production-tested code.

### 6.3 Performance rules

Per-group `nlargest` is banned — use global sort + `groupby.head` / `cumsum`.
No `groupby.quantile` for weighted percentiles — sort by value and take the
weighted-cumulative crossing. No `groupby.apply` for entropy — two groupbys.
No per-edge `apply` for distance — vectorised haversine. Never average lat/lon
directly — 3D unit vectors. Betweenness uses k=128 sampled sources
(`BETWEENNESS_K = 0` disables).

---

## 7. Deferred

Community-level and NAICS-group metric tables; community-as-node / NAICS-as-node
supergraph construction (`pkg_supergraph.py` retained). Edge periodicity /
cadence (`pkg_edge_rhythm.py`) — independent of geography *and* of the
counterparty boundary, EDA pending. Geo peer normalisation (§2.10). B2B flow
decomposition (`PKG_GEO_MANIFEST.md` §4 Block E).

---

*Internal — PNC Treasury Management, Data Science*
