# PKG_GEO_MANIFEST.md

**Payment Knowledge Graph — Geographic Analytics Manifest**
PNC Bank · Treasury Management · Data Science
Version **1.2** — August 2026

> v1.0 → v1.1: the edge-dependent dispersion block moved from **HOLD** to
> **BUILT**. It is no longer a separate GPU job — geography is computed inside
> the monthly metric run, per version, from a customer attribute file passed
> alongside the snapshot. §4 Block F, §6 and §8 are rewritten; §2, §3, §5 and
> §10 are unchanged in substance.
>
> Companion to `PKG_MONTHLY_METRICS_MANIFEST.md` **v3**.
> Supersedes `PKG_GEO_METRICS_SPEC.md` (retired at v0.9) and UC #8 in
> `PKN_Roadmap.md`.

---

## 1. Status

| | |
|---|---|
| Address join to PKG | **99.593%** |
| Geocoded, graph-visible | **98.4%** at rooftop precision |
| Markets | 909 ZIP3 (CBSA pending — §9.1) |
| History | 23 months × 3 ablation versions |
| Geo metric block | **live in `pkg_pipeline.py`**, 24 columns per node per version |

The **analytical programme remains reduced** by the Phase A2 validation: two of
three hypotheses failed pre-registered decision rules. What is built is
deliberately narrower than what was originally scoped, and correctly targeted.

### 1.1 The reframe that governs this document

**Dollars per customer are flat to mildly inverse across geography.** Within
50 km of Pittsburgh = **0.87×** the national average; beyond 1,000 km =
**1.16×**; stable in all 23 months. At V0 the same table shows a 2.35×
Pittsburgh premium, which is a payment-infrastructure artifact.

Geography does **not** predict customer size. It predicts *shape* — footprint
width, consistency with industry peers, change over time, proximity structure.
This block is a **segmentation and anomaly instrument, not a sizing one.**

---

## 2. Evidence ledger

### 2.1 Confirmed

| Finding | Value |
|---|---|
| Identifier join `node` ↔ `mdm_id` | 99.593%, 12-digit numeric, identical under all normalisations |
| Graph closure | residual 0.0% |
| Node geo coverage (P99_9) | 98.39% |
| Strength-weighted coverage | 96.48% (P99_9), 94.07% (V0) |
| Coordinate precision | modal 6 decimals |
| Coverage vs distance | 96–99.5% every US state; **PA below median** — no gradient |
| Hub concentration | top 0.1% of nodes = **97.3%** of dollars |
| Population composition | **94.26%** of nodes individual; **51.59%** of dollars business — **⚠ measured on the name typer; pending recomputation under `party_type` (§3)** |
| Distance gradient | 0.87× within 50 km, 1.16× beyond 1,000 km, **23 of 23 months** |
| Net position persistence | 72 persistent sources, 134 sinks, 350 unstable (of 556) |

### 2.2 Reversed or withheld

| Claim | Status |
|---|---|
| Pittsburgh dollar premium (2.35×) | **Reversed** → 0.87×; hub artifact |
| 50–150 km hinterland "hole" (0.41×) | **Reversed** → 0.90×; hub artifact |
| Cleveland largest net sink (+$100.8B) | **Reversed** → persistent source, 87% of months |
| Triangle closure = where money is (8.2×) | **Reversed** → 1.5× |
| Houston trophic outlier (0.51) | **Reversed** → 5.34 |
| Local supply chains absent (0.58%) | **Unresolved** — 18.5% at P99_9 but on a 94%-household population |
| Florida inflow as a TM opportunity | **Abandoned** — business −$212M (source); household +$580M |
| Metro net flow as economic geography | **Reframed** — 89.3% industry mix; mechanism is labour intensity |
| Metro net flow magnitudes | **Withheld** — P99 vs P99_9 sign agreement 62.98% |
| Florida seasonality | **Not found** — positive all 23 months, no annual cycle |

### 2.3 The governing fact

Businesses net **−$7.5B** to households over 23 months. The dominant flow is
payroll; the persistent source/sink structure tracks **commuting geography** —
employment centres pay outward to their suburbs. Real, durable, and a **Retail
& Wealth** finding, not a Treasury Management one.

---

## 3. Population model

Three orthogonal partitions, **all declared on every emitted table.**

**Ablation** — `V0` (5,289,362 nodes) · `P99_9` (3,554,141) · `P99` (3,219,179).
`P99_9` is the analytical default; `V0` is an infrastructure diagnostic;
`P99` is **degenerate for net flow** (metro nets collapse to noise, sign becomes
random) and is used only for dispersion robustness. Every reportable result must
agree at two adjacent rungs.

**Node type** (`node_type`, from `customers.csv`) — `business_naics_valid` /
`business_naics_placeholder` / `business_naics_missing` / `individual` /
`unknown`. This is the project's single taxonomy; the Phase A2 `party_class`
label is retired in its favour.

> **⚠ v1.2 — the business/individual split is now OBSERVED, not inferred.**
> The MDM customer table carries `party_type` (`P` person / `O` organisation).
> Every geographic finding in §2 was computed when that split came from a
> token heuristic on `customer_name`, which systematically misclassifies
> **person-named organisations** — trusts, estates, single-member LLCs, sole
> proprietorships — as households. The correction runs **one way only**:
> individual → business. Nothing moves the other way.
>
> Consequences to work through before the next geographic result is briefed:
>
> | Finding | Exposure |
> |---|---|
> | 94.26% individual | Moves **down** by the `node_type` churn in `qa/customers_typing_disagreement.csv`. Recompute; do not re-cite. |
> | Businesses net −$7.5B to households (§2.3) | **Directly exposed.** Both sides of this flow are typed. Trusts and estates reclassified out of "household" reduce the household side, and person-named LLCs are precisely the entities most likely to sit on the receiving end of what was read as payroll. Recompute before the Retail & Wealth handoff. |
> | Commuting-geography source/sink structure | Exposed to the same mechanism. The persistence pattern is likely robust; the magnitudes are not. |
> | Distance gradient (0.87× / 1.16×, 23 of 23 months) | Low exposure — computed over all nodes, not partitioned by type. |
> | Hub concentration (top 0.1% = 97.3% of dollars) | No exposure — degree-based, independent of typing. |
>
> The qualitative conclusion — that the PKG is household-dominated and that
> unpartitioned geographic aggregates are Retail statements rather than TM
> ones — is very unlikely to reverse. The **magnitudes** are what is at stake,
> and the Retail & Wealth transfer package should carry recomputed figures.

**Denominator** — `MDM_ALL` (30.85M parties, profiling only) · `PKG_NODE` ·
`PKG_DOLLAR` (the only figure worth briefing). Only 17.1% of MDM parties are
graph-visible.

---

## 4. Column register

### Block A — Location (from `customers.csv`, static)

Carried on every node row; also written once to `../metrics/customers_dim.parquet`.

| Column | Definition |
|---|---|
| `geo_status` | `valid` / `placeholder` / `missing` |
| `lat`, `lon` | float32; cast only after the numeric string screen (§5.1) |
| `zip5`, `zip3` | digit-strip then substring — **never split on `-`** |
| `state`, `city` | `city` is display-only, never a key |
| `addr_loc_rec_type` | USPS AIS record type — the placeholder discriminator |
| `shared_structure` | 1 where `rec_type = HIGHRISE` |
| `cust_name`, `naics_cd`, `naics_desc` | identity passthrough |
| `naics2` … `naics6`, `naics_known` | hierarchy, derived once |
| `party_type` | `P` / `O` / null — declared MDM party type, the entity-type source of truth |
| `household_id`, `customer_start_dt` | passthrough; `household_id` is null on `O` rows |
| `attr_profile` | `identity+geo` / `identity_only` / `geo_only` / `neither` / `not_in_customers` — which side of the address↔customer outer join this node came from |
| `entity_type`, `entity_type_observed`, `entity_type_inferred`, `entity_type_source` | typing; observed and inferred kept separate permanently |
| `entity_class`, `naics_status`, `naics_applicable`, `naics_coverage_class`, `node_type` | typing + NAICS quality/applicability axes |

`geo_status` derivation:

```
missing      no numeric coordinate, or (0,0), or out of range
placeholder  addr_loc_rec_type ∈ {POSTOFFICEBOX, GENERALDELIVERY}
valid        otherwise
```

`customers.csv` carries no address lines, so `addr_loc_rec_type` is the only
placeholder signal available. Name-based PO-box matching is **not** used — it
would flag businesses literally named "PO Box Trading Co".

**`customers.csv` is an outer join of the address and customer tables**, so a
node can carry geography with no identity (`attr_profile = geo_only`) or
identity with no geography (`identity_only`), and one `mdm_id` may appear on
several address rows. Duplicates are resolved by taking the **geo block whole
from the best-ranked row** — coordinate + own-premises record type, then
coordinate + PO box, then ZIP only, then nothing — never by mixing fields
across rows. Full rules: `PKG_MONTHLY_METRICS_MANIFEST` §0.1. Geo coverage
figures in §2.1 should be re-read against `attr_profile`: a node that is
`geo_only` has a location but no NAICS and no `party_type`, so it can enter a
coverage statistic while being invisible to every population partition.

`low_precision` was trialled and dropped (3,335 rows of 30.8M — noise, not a
population). `non_us` is not derivable here (no country column); non-US rows
surface as `missing` because they are ungeocoded (Canada 8–19%, Mexico 0–1.4%).

### Block F — Geographic dispersion (edge-derived) — **BUILT**

`pkg_geo_metrics.geo_node_metrics(edges, coords)`, called **per version** inside
`node_metrics()`. 24 columns.

> **Computed per version, not once on RAW.** Hub removal changes who a node's
> counterparties are, and the V0 → P99_9 comparison reversed six geographic
> findings. A single RAW-graph geo block would carry that distortion into every
> version.

Two families, deliberately separated because they have different requirements
and different failure modes:

**SPREAD** — properties of the counterparty cloud. Needs only the
*counterparties'* coordinates, so it is **defined for a node with no location of
its own.** This is the family that transfers to counterparties (§10).

| Column | Definition |
|---|---|
| `geo_spread_km` | Amount-weighted radius of gyration about the counterparty centroid, both directions pooled: `sqrt(Σ w·d² / Σ w)` |
| `geo_spread_in_km`, `geo_spread_out_km` | Same, per direction — **revenue footprint vs supply footprint** |
| `geo_centroid_lat`, `geo_centroid_lon` | Amount-weighted centroid via 3D unit vectors |
| `geo_R` | Mean resultant length of the centroid vectors — 1 = co-located, 0 = dispersed |
| `geo_n_zip3_80` | Distinct ZIP3s covering 80% of amount |
| `geo_zip3_entropy` | Shannon entropy over ZIP3 amount shares |
| `geo_home_zip3_share_{in,out}` | Share of **located** dollars with counterparties in the node's own ZIP3 |
| `geo_home_state_share_{in,out}` | Same at state level |
| `geo_locality_class` | `LOCAL` <50 km · `REGIONAL` <250 · `MULTI_MARKET` <1000 · `NATIONAL` ≥1000, from `geo_spread_km` |

**REACH** — distance from the node's **own** point to its counterparties.
Requires the node's own coordinates; **NaN otherwise, never zero.**

| Column | Definition |
|---|---|
| `geo_reach_mean_km` | Amount-weighted mean distance |
| `geo_reach_p50_km`, `geo_reach_p90_km` | Amount-weighted distance percentiles — typical vs tail |
| `geo_registered_vs_flow_km` | Distance between the registered point and the flow-weighted centroid |

**Coverage** — always co-reported, because every metric above is computed on
located counterparties only:

| Column | Definition |
|---|---|
| `geo_cov_amt_{in,out}` | Share of that side's dollars with a located counterparty |
| `geo_n_cp_located_{in,out}` | Distinct located counterparties |

**`geo_registered_vs_flow_km` is the representativeness test.** With `addr_type`
constant in MDM, it is now the *only* available check on whether a registered
address means anything. Small gap + small spread → the address is meaningful.
Registered in Pittsburgh, centroid in Chicago, spread 900 km → HQ artifact; use
it as a label, not a location. Gate point-level analytics on this.

**Guards.** `geo_spread_*` is NaN below `MIN_CP_FOR_SPREAD = 2` — with one
counterparty spread is trivially 0, which is not the same as "concentrated".
Entropy is NaN below `MIN_CP_FOR_ENTROPY = 5` — Shannon entropy is biased low at
small *n* and would otherwise report counterparty count, not diversity.
Home-share is **0.0**, not NaN, when a node has located counterparties and a
known own key but no match — NaN is reserved for "nothing to compare".
Home-share denominators are **located** dollars, not total, so the metric is not
confounded with coverage (which is reported separately).

### Block B — Placeholder registry

Modelled on the Hub Node Registry: **labelled taxonomy with per-class policy,
not a flat exclusion list.** 909,820 parties across 16,824 clusters (≥25 parties).

Detection is **conditioned on `rec_type`.** Duplicates within `HIGHRISE` are the
building and expected (Horsham PA, 24,047 parties, benign). Duplicates within
`NORMAL` are the signal (Pittsburgh 40.44052/−80.00027, **29,780 parties**, PNC
HQ ZIP 15222). A flat screen catches both or neither. `n_distinct_line1` is the
discriminator: one distinct line = hard placeholder; many = geocoder centroid,
valid at CBSA but not at street. PO-box concentration in resort towns (Avon,
Breckenridge, Vail) is legitimate and already downgraded by `rec_type` — leave it.

*Not yet in the pipeline* — requires address lines, which `customers.csv` does
not carry. Runs against the MDM address table (`pkg_geo_address_profile_v3`).

### Block C — Relocation history

660 daily snapshots (2024-09-20 → 2026-08-05) make the change log
**retroactive**. Daily address history joined to monthly payment behaviour exists
nowhere else in the bank.

**Classification must be distance-graded, not hash-based.** The two-hash design
failed: the cleansing vendor rewrites text and coordinates together. Proof — the
5-day interval 2026-07-31 → 2026-08-05 showed 149,844 "relocations" against
137,458 in the preceding 31 days, while `regeocode` scaled cleanly (1,487/day vs
1,488/day). All rates normalised per elapsed day.

*Not yet in the pipeline.* Blocked only on scheduling.

### Block E — B2B flow decomposition

Node-level `net_flow` includes payroll outflow to individuals, which is why even
the business-only source list is Pittsburgh, Houston, Philadelphia. The TM
question is **business-to-business** flow, derivable edge-free from the
composition shares already emitted:

```
b2b_in   = in_strength  × (share_in_amt_biz_valid  + biz_placeholder + biz_missing)
b2b_out  = out_strength × (share_out_amt_biz_valid + biz_placeholder + biz_missing)
b2b_net  = b2b_in − b2b_out
payroll_out       = out_strength × share_out_amt_individual
payroll_intensity = payroll_out / out_strength
```

The NAICS shift-share must be **recomputed on `b2b_net`** — the 89.3% mix result
is a labour-intensity artifact of payroll and may not survive.

---

## 5. Derivation rules and standing traps

**Coordinate screen — mandatory before any cast.** Latitude/longitude arrive as
text. Empty strings, `'0'`, whitespace and the token `'null'` all survive
`IS NOT NULL`. Positive longitude is an error **only inside the US** — 2,675 rows
are Singapore, Dubai, Tokyo, Guam, Paris and are correct.

**Peer normalisation.** Raw dispersion is confounded by degree and industry; a
node with four counterparties *cannot* have high ZIP3 entropy. Residualise on
`log1p(degree)` and `log1p(strength)`, then percentile-rank within
`naics2` × size-decile × `node_type`. **The reportable quantity is
`{metric}_pctile_naics_size`, never the raw value.** *(Not yet implemented —
Block F emits raw values; normalisation is a downstream step.)*

| Trap | Rule |
|---|---|
| `strength` double-counts | `strength = in + out`; cross-node sums count each dollar twice. Fine as a weight, **wrong as a volume figure** |
| ZIP parsing | digit-strip then substring; **never** split on `-` (83,547 rows are `zip9_flat`) |
| ID precision | ids stay **string** end to end; 18-digit address ids exceed float64 exact range |
| `city` as a key | never — `ROYAL PLM BCH` |
| Mixing node types | never aggregate across `node_type` without declaring it |
| Single-month results | nothing reportable without temporal persistence |
| Guarded divides | `np.where(cond, a/b, x)` still evaluates `a/b` everywhere — use `_sdiv()` |

---

## 6. Scope caveat — mandatory on every emitted table

The closure residual is 0.0% **because the graph is an internally closed
subsystem**: PNC on-us C2C only. Metro net flow means *"net position versus other
PNC customers,"* not versus the economy. The household surplus (+$7.5B) is partly
a **boundary artifact** — wages arrive from on-us employers while consumer
spending leaves to off-us merchants and card networks.

Every table carries `scope = on_us_c2c`. Revisit when PAYS_CPTY lands.

---

## 7. Validation protocol

Written in advance of analysis, not after. Any geographic result clears all four
before it leaves the notebook.

| # | Test | Threshold |
|---|---|---|
| 1 | **Node type** — computed separately for business and household | mandatory |
| 2 | **Industry mix** — shift-share decomposition | reframe if mix >70% of magnitude |
| 3 | **Ablation** — two adjacent rungs | withhold if sign agreement <80% |
| 4 | **Temporal persistence** — all available months | discard unless stable; k-of-m ≥80% for sign claims |

A2 outcome: rules 1, 2 and 3 fired; rule 4 confirmed the distance gradient.

---

## 8. Compute

**The separate GPU edge pass proposed in v1.0 is cancelled.** Geography is now a
block inside the existing monthly run: the customer attribute file is passed
alongside the snapshot, the snapshot itself is reduced to
`source, dest, amount, volume`, and `geo_node_metrics()` runs per version.

Consequences:

- One pipeline, one output schema. No separate node geo block to reconcile.
- The corridor matrix (CBSA × CBSA × month) is **not** produced. It was the input
  to recirculation / local-multiplier work, which is on hold per §7 rule 3.
- Cost is visible: `GEO_METRICS = False` gives a clean baseline, and
  `run_timings.csv` reports the geo component per (month, version).

Implementation notes carried from v1.0 and honoured in the code: vectorised
haversine, never per-edge `apply`; entropy as two groupbys, never
`groupby.apply`; the 80%-coverage count as global sort + `groupby.cumsum`, never
`groupby.nlargest`; weighted percentiles by sorted cumulative weight, never
`groupby.quantile`; centroids via 3D unit vectors, never averaged lat/lon.

### 8.1 `GeoIndex` — the lookup, and why it is not a raw DataFrame

The first production run made geo the slowest component: **405 s on V0, but
still 314 s on P99 with 4.4× fewer edges.** Fitting the four versions gives
`≈287 s + 43 s per million edges` — almost all fixed cost, none of it geometry.

Two causes, both in how the coordinate table was addressed:

1. `coords.set_index("node")` then `.reindex()`, run **12 times per month**
   (3 calls × 4 versions) against the **full ~20 M-row customer table**, while a
   month touches ~2 M nodes — a hash table 10× larger than needed, rebuilt every
   time.
2. Every `groupby("node")` re-factorized an **arrow-backed string** column.
   Profiling showed 7.3 s in `arrow.factorize` and 9.1 s in `isin` per call.

`GeoIndex` fixes both: restrict to the month's nodes, factorize ids to `int32`
codes **once per month**, hold plain numpy arrays (`lat`, `lon`, `zip3`,
`state`, each with a sentinel row so a miss indexes safely), and **group on
integer codes throughout** — string ids are restored once, at the boundary.

Two correctness constraints the implementation must preserve:

- **Every node in the month gets a code, located or not.** An unlocated node
  still has SPREAD over its located counterparties; dropping it would silently
  lose exactly the rows the SPREAD/REACH split exists to keep.
- **Unknown ZIP3 / state is `-1`, and `-1 == -1` must not count as a match.**
  Home-share requires both sides known. (The earlier string form achieved this
  with distinct NA sentinels on each side.)

Measured: ~4× faster per call, fixed cost gone. Both are covered by parity tests
against the pre-rewrite path.

### 8.2 Two run modes

`GEO_METRICS=True` computes geo inline. `GEO_ONLY=True` computes **only** the
geo block, writing `../metrics/geo/geo_{YYYY-MM}.parquet` with its own
independent skip-check. Both key on `(time_key, version, node)`, so they combine
with a plain merge — verified numerically identical to a combined run.

Splitting is preferred: ≈6.0 h for metrics and ≈1.1 h for geo, each comfortably
inside an 8 h session, and **geo can be re-run after a fix without repeating
hours of centrality work.**

---

## 9. Open items

| Item | Blocks |
|---|---|
| **CBSA crosswalk** (HUD USPS ZIP→ZCTA + Census county→CBSA) | All external output. ZIP3 splits metros (Miami / Fort Lauderdale / West Palm) and merges others — fine internally, not for a deck |
| Peer normalisation (§5) as a pipeline step | Reportable dispersion |
| Block B placeholder registry into the pipeline | Needs address lines, absent from `customers.csv` |
| Block C relocation log | Scheduling only |
| Block E B2B decomposition | The actual TM question |
| `throughflow` definition — `min(in,out)`? | Turnover work |
| Counterparty geographic inference — compliance read | §10 |

---

## 10. Counterparty extension

**Same column definitions, different confidence.** Block F's SPREAD family is
already defined for a node with no coordinates of its own, which is exactly the
counterparty situation. Two columns are added:

| Column | Values |
|---|---|
| `geo_confidence` | `observed` / `payer_centroid` / `fi_prior` / `none` |
| `geo_precision_km` | estimated radius of uncertainty |

**Payer-centroid inference.** Amount-weighted centroid of a counterparty's
PNC-side payers, with payer-cloud dispersion as the confidence measure —
`geo_centroid_lat/lon` and `geo_spread_km` computed from the counterparty's
perspective. Strengthened by the customer side being 98.4% geocoded.

**FI-implied geography** (requires `CptyFinEntity`): Tier A single-market
community banks and geographically chartered CUs → soft location; Tier B
regional → state only; Tier C national → **share, never place**.

> **Charter size is not footprint.** BaaS sponsor banks (Cross River, Evolve,
> Sutton, Coastal, Lead) are small charters with nationwide digital reach — Tier
> A on branch count, badly wrong in practice. The taxonomy needs an explicit
> `footprint_reliability` flag set from **business model, not branch geography**.
> Same failure mode as a naive degree-based hub rule.

**Validation posture — corrected.** Calibrating an imputation ladder on customers
and carrying it to counterparties **does not hold**: at 98.4% coverage where the
missing 1.6% is systematically non-US and uncleansed, that population is not a
representative holdout. **Dual-mask validation** is the only honest option —
random masking for training signal plus a realistic mask reproducing observed
missingness, **both** error rates reported. Random-mask performance overstates;
say so in every output.

**Hard gate:** `geo_confidence ≠ observed` is acceptable for corridor aggregates,
market-share work and metro exposure. **Never** for hazard overlay, site-level
analysis, or any decision attached to a specific point.

---

## 11. Artifacts

**Code** — `pkg_customers.py` (dimension loader), `pkg_geo_metrics.py` (Block F),
`pkg_pipeline.py` / `pkg_custom_metrics.py` / `pkg_runtime.py` (host pipeline),
`pkg_geo_address_profile_v3` (MDM profiling, Blocks B and C),
`pkg_geo_phase_a` / `pkg_geo_phase_a2` (validation).

**Data** — `../metrics/node/node_{YYYY-MM}.parquet` (geo block inline),
`customers_dim.parquet`, `run_timings.csv`, plus the Phase A/A2 parquets.

**Documents** — this manifest; `PKG_MONTHLY_METRICS_MANIFEST.md` v3;
`PKG_Geographic_Intelligence.pdf` (executive findings, August 2026).

---

## 12. Change log

| Version | Date | Change |
|---|---|---|
| 0.9 | 2026-08 | Initial spec (retired) — assumed a business-dominated population |
| 1.0 | 2026-08 | Rebuilt on Phase A2. `party_class` promoted to partition key; Block E added; validation protocol formalised; build register added |
| 1.2 | 2026-08 | **`party_type` adopted as the observed entity type**; §2 population findings flagged for recomputation; Block A extended with `party_type`, `attr_profile` and the split observed/inferred typing columns; outer-join duplicate resolution documented |
| 1.1 | 2026-08 | **Block F BUILT** — 24 geo columns, per version, inside the monthly run. Separate GPU edge pass cancelled; corridor matrix not produced. `party_class` retired in favour of `node_type`. Geo source is `customers.csv`, not the MDM address table. SPREAD/REACH separation, guards and home-share semantics documented |
| **1.2** | **2026-08** | **`GeoIndex` rewrite** after the first production run (§8.1): geo was 405 s/version on fixed cost, now ~4× faster. `GEO_ONLY` split-run mode (§8.2). `GEO_COLUMNS` canonical schema. Ladder default cut to two tiers |

---

*Internal — PNC Treasury Management, Data Science*
