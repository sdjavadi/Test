# PKG Monthly Snapshot Metrics Manifest — v6

**Payment Knowledge Graph (PKG) — Customer-to-Customer Monthly Network Analysis**
*PNC Bank · Treasury Management · Data Science*
*Tracks `pkg_customers_v8.py`, `pkg_pipeline_v8.py`, `pkg_custom_metrics_v8.py`*

> **v5 → v6. Memory. The v5 loader OOM-killed the batch step** (job 1246888,
> 2026-08-11) about nine minutes after the parquet read logged successfully.
> Host RAM, not GPU. Three causes, all introduced in v4/v5 — see §6.4 for the
> post-mortem and the measurements. Summary:
>
> 1. **`np.select` with string choices returns a `<U26` array — 104 bytes per
>    element, 3.1 GB per column at 30M rows.** v5 built nine of them
>    (~28 GB of transient string arrays) before assembling a DataFrame.
>    Fixed: select **integer codes** and build a Categorical. 1 byte/row.
> 2. **The name typer allocated one Python list per row** (~3.6 GB of list
>    headers at 30M, held across three further `.map()` passes, 2.85 s/M).
>    Fixed: fully vectorised on `.str` accessors, **verified identical** on
>    both outputs.
> 3. **The dimension was materialised three times** — `attrs`, `typing`, and
>    the pipeline's merge of the two. Fixed: one frame; `typing` is a
>    projection and the pipeline no longer merges.
>
> Measured on a 300K-row extract with the same composition: peak RSS
> **0.69 → 0.38 GB**, load wall **6.9 → 1.7 s**, resident dimension
> **68 → 27 MB**. The 0.28 GB peak delta scales linearly to **~28 GB saved at
> 30M rows**, which is the whole of the overrun. **Every typed output column
> is byte-identical to v5** (verified column-by-column at 300K rows).
>
> Also: **all derived columns are Categorical with declared category sets**,
> which pins the node-table schema as a side effect — a category that was not
> declared cannot appear. `node_typing` is written as **parquet**, not a
> 30M-row CSV. `RESTRICT_DIM_TO_GRAPH` (off by default) is the remaining lever
> if the batch step is still tight.
>
> **v4 → v5. The customer dimension is parquet.**
>
> The extract is now `customers.parquet`, not CSV. The schema travels with
> the file, so the loader stops defending against text ambiguity: a string
> is a string, a double is a double, and **NULL is distinct from empty
> string** (normalised to one representation in `_text()`, deliberately and
> in one place). Only the mapped columns are read off disk — column
> selection happens *before* the read, which on a 30M-row extract is the
> difference between loading eleven columns and loading the whole table.
>
> **Parquet moves the failure modes, it does not remove them** (§0.1). Two
> are real, silent, and cannot occur on a CSV read with `dtype=str`:
> integer-typed `zip_cd` destroys leading zeros, and a float-typed `mdm_id`
> is not exactly representable past 2^53. Both are detected, handled, and
> written to `qa/customers_source_schema.csv`.
>
> **Columns removed:** `household_id`, `customer_start_dt` — not part of the
> extract. `date_of_birth_inc` remains deliberately un-ingested (§0.2).
>
> **v3 → v4.** The customer source table was inspected directly. Three things
> it showed that the v3 spec had wrong:
>
> 1. **`party_type` exists and is authoritative.** The MDM party model carries
>    `P` (person) / `O` (organisation) / null. Entity type is now a *declared*
>    attribute, not a name inference. The v3 token typer is retained as a
>    separate, permanently-preserved column — never overwritten in place
>    (§2.9). **This moves `node_type`, and `node_type` drives every
>    `share_*` composition column.**
> 2. **NAICS on a person is not missing data.** Empty `naics_cd` on a `P` row
>    means the field does not apply, not that enrichment failed. Applicability
>    is now an axis of its own (`naics_applicable`) and enrichment coverage is
>    reported against an organisation-only denominator. The true NAICS-gap
>    denominator is far smaller than v3 implied.
> 3. **`customers.csv` is an OUTER JOIN of the address table and the customer
>    table.** Rows exist with geography and no identity, and with identity and
>    no geography, and one `mdm_id` can appear on several address rows. v3's
>    `drop_duplicates(keep='first')` was arbitrary under that shape and could
>    discard the row holding the only valid coordinate (§0.2).
>
> Also corrected: **placeholder NAICS is expected to be non-zero.** `******` /
> `UNKNOWN` is present in the source table itself — it was not only a
> Cypher-injected artefact, as v3 asserted. The loud "expected ≈0" warning is
> replaced by a counted value with a share threshold.
>
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
| `../data/customers.parquet` | `mdm_id, customer_name, party_type, naics_cd, naics_desc, addr_loc_rec_type, longitude_degree, latitude_degree, zip_cd, state_or_province, city` | **Current state.** Not one row per customer — see §0.2. Dtype requirements in §0.1 |
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
**The `******` / `UNKNOWN` sentinel is present in the source table**, on person
rows in particular. v3 asserted this token was a Cypher-extraction artefact and
that a residual match count of ≈0 was expected; direct inspection of the source
disproves that. `PLACEHOLDER_NAICS` is a live classifier, not a regression
tripwire — counted at every load, warning only above `PLACEHOLDER_WARN_SHARE`
(5% of rows).

> **Parquet caveat.** If `naics_cd` is written as an INTEGER column, `******`
> cannot survive the type and became NULL in the extract. The placeholder count
> then drops toward zero — not because enrichment improved, but because the
> sentinel was destroyed by the cast, and those rows now read as `missing`
> instead. `naics_cd` **must be extracted as a string** (§0.1).

### 0.1 Source dtypes — **new in v5**

Parquet carries its schema, which removes a class of bug and introduces
another. Four columns **must be extracted as strings**; a numeric type
destroys information before the loader ever sees the file.

| Column | Required type | What a numeric type does |
|---|---|---|
| `mdm_id` | STRING | int64 stringifies cleanly, but a **DOUBLE id above 2^53 is not exactly the value written**, and renders as `1.0003e+11` or `...0`. The loader routes both through Int64 and warns; the fix belongs in the extract |
| `zip_cd` | STRING | **Leading zeros are gone.** `02108` arrives as `2108`, which silently relocates every New England customer to a ZIP3 that keys nothing. Zero-padded on read, but a genuinely short value is then indistinguishable from a 0-prefixed one |
| `naics_cd` | STRING | The `******` sentinel cannot survive; those rows arrive NULL and reclassify from `placeholder` to `missing`. **Placeholder counts across a dtype change are not comparable** |
| `party_type` | STRING | Single-character code; no numeric representation is meaningful |

`_dtype_report()` inspects these on every load and writes
`qa/customers_source_schema.csv` (resolved column names, arrow types, and any
issue found). **Check it on the first run after any extract change** — every
one of these failures is silent and produces plausible-looking output.

Coordinates (`latitude_degree`, `longitude_degree`) may be DOUBLE or STRING;
`_num()` branches on dtype and applies the numeric-string screen only where the
source is text. **NULL and empty string are normalised to a single absent
representation** in `_text()` — parquet preserves the distinction that CSV
collapses, and every downstream test asks only "is this absent", so the choice
is made once rather than by whichever `.fillna("")` happens to run first.

CSV remains readable (extension dispatch) for reprocessing archived extracts,
with a warning. It is not the supported source.

### 0.2 The customer file is an outer join

The customer extract is produced by an **outer join between the MDM address
table and the customer table**. Three consequences, none of them cosmetic:

| Shape | Cause | Handling |
|---|---|---|
| Row with `mdm_id` but no name / NAICS / `party_type` | Address row matched no customer record | Kept. `attr_profile = geo_only`; types as `unknown` |
| Row with identity but no coordinates | Customer with no geocoded address | Kept. `attr_profile = identity_only`; `geo_status = missing` |
| Row with **no `mdm_id` at all** | Neither side matched | **Dropped and counted** — cannot join the graph |
| **Several rows sharing one `mdm_id`** | One customer, several addresses | Collapsed — see below |

**Duplicate resolution is deterministic, not positional.** v3 used
`drop_duplicates(keep='first')`, which under a join fanout could keep a row
whose geography was a PO box or empty while a sibling row held the only valid
coordinate. v4:

- **Identity fields** (`customer_name`, `party_type`, `naics_cd`, `naics_desc`)
  are **coalesced** across the group —
  first non-null wins. Identity comes from the customer side of the join and
  should be constant within a group; where it is not, the conflict is counted
  and logged per field. **A non-zero `conflict_party_type` means the join key
  is not what we think it is** and should stop the run for investigation.
- **Geography is taken whole from one row**, ranked
  `3` usable coordinate + own-premises record type ›
  `2` usable coordinate + PO box / general delivery ›
  `1` no coordinate but a 5-digit ZIP ›
  `0` nothing locational.
  Fields are **never mixed across rows** — a latitude from one address and a
  ZIP from another describes a place that does not exist.

`attr_profile` (`identity+geo` / `identity_only` / `geo_only` / `neither` /
`not_in_customers`) is carried on every node row so join-side coverage is a
queryable fact rather than an assumption.

**Extraction SQL** — `party_type` must be in the pull, and the four
string-critical columns must be CAST explicitly (§0.1). Casting in the SELECT
is what guarantees the parquet schema; relying on the source column types is
what produces a 4-digit ZIP six months later.

```sql
CREATE TABLE <db>.customers_dim_src STORED AS PARQUET AS
SELECT CAST(COALESCE(c.mdm_id, a.mdm_id) AS STRING) AS mdm_id,
       CAST(c.customer_name          AS STRING) AS customer_name,
       CAST(c.party_type             AS STRING) AS party_type,   -- P / O
       CAST(c.naics_cd               AS STRING) AS naics_cd,     -- keeps '******'
       CAST(c.naics_desc             AS STRING) AS naics_desc,
       CAST(a.addr_loc_rec_type      AS STRING) AS addr_loc_rec_type,
       CAST(a.longitude_degree       AS DOUBLE) AS longitude_degree,
       CAST(a.latitude_degree        AS DOUBLE) AS latitude_degree,
       CAST(a.zip_cd                 AS STRING) AS zip_cd,       -- keeps '02108'
       CAST(a.state_or_province      AS STRING) AS state_or_province,
       CAST(a.city                   AS STRING) AS city
FROM   <customer_table> c
FULL OUTER JOIN <address_table> a ON a.mdm_id = c.mdm_id
WHERE  c.mdm_id IS NOT NULL OR a.mdm_id IS NOT NULL;
```

`household_id` and `customer_start_dt` are **not** part of the extract.

`date_of_birth_inc` exists in the upstream table and is **deliberately not
extracted.** Age is a prohibited basis under ECOA / Regulation B, and placing it
in the dimension that feeds every node metric creates a fair-lending surface
for no current analytical need. If a use case requires it, ingest it in a
separate access-controlled table with a documented purpose. Note the column
name is *birth **or** incorporation* — on `O` rows it is the incorporation
date, which is why organisations carry values like `1906-01-01`.

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
../metrics/node_typing.parquet                # entity_type / naics_status / node_type (parquet: 30M rows)
../metrics/qa/customers_typing_disagreement.csv   # observed vs inferred; node_type churn v3->v4
../metrics/qa/customers_join_profile.csv          # outer-join shape, fanout, identity conflicts
../metrics/qa/customers_naics_coverage.csv        # entity_type x naics_coverage_class
../metrics/qa/customers_source_schema.csv          # resolved columns + arrow types + dtype issues
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
dimension join: `cust_name`, `party_type`, `naics_cd`, `naics_desc`,
`addr_loc_rec_type`, `lat`, `lon`, `zip5`,
`zip3`, `state`, `city`, `geo_status`, `attr_profile`, `shared_structure`,
`naics2`–`naics6`, `naics_known`, `entity_type`, `entity_type_observed`,
`entity_type_inferred`, `entity_type_source`, `entity_class`, `naics_status`,
`naics_applicable`, `naics_coverage_class`, `node_type`.

Nodes absent from the customer extract are emitted with NA attributes,
`geo_status = 'not_in_customers'` and `attr_profile = 'not_in_customers'`, so
coverage stays visible rather than silent.
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

**Motivation.** NAICS missingness confounds three different things: the
counterparty being an *individual* (field does not apply), a *business that
wasn't enriched* (field applies, value absent), and a *business onboarded with
a sentinel* (field applies, value junk). Node types are decomposed first;
behavioural composition is kept strictly separate from data-quality coverage,
and **both are kept separate from applicability**.

**Entity type is observed, not inferred — new in v4.** `party_type` is a
declared MDM attribute. The v3 name typer is an inference. Both are retained
as separate columns permanently, on the same principle as
`naics_observed` / `naics_imputed`: **an inference must never overwrite an
observation in place, because the disagreement is then unauditable.**

| Field | Values | Rule |
|---|---|---|
| `entity_type_observed` | business / individual / unknown | `party_type`: `O` → business, `P` → individual, null → unknown. Domain confirmed `{P, O, null}` (2026-08); anything else types unknown and logs loudly |
| `entity_type_inferred` | business / individual / unknown | v3 token typer on `customer_name`, unchanged. Individual = 2–3 all-alpha tokens, no business token. Retained for QA and as the fallback |
| `entity_type` | business / individual / unknown | **Observed wins.** Inference is used only where `party_type` is null |
| `entity_type_source` | party_type / name / none | Which evidence decided. Filter to `party_type` for observation-only populations |
| `naics_status` | valid / placeholder / missing | **Field quality only.** Empty → missing; sentinel list → placeholder (**expected non-zero**); ≥2 leading digits → valid |
| `naics_applicable` | 1 / 0 | 1 where `entity_type = business`. NAICS is expected of an organisation, not of a person |
| `naics_coverage_class` | valid / placeholder / missing / not_applicable | Reporting roll-up. An **observed** valid code reports `valid` even on a person — never mask an observation as inapplicable. `not_applicable` = no code *and* none expected |
| `node_type` | business_naics_valid / _placeholder / _missing / individual / unknown | The **composition key**. Five categories, schema-stable. `entity_type` decides business vs individual; NAICS status then subdivides the business branch |
| `entity_class` | the above **plus** individual_naics_valid, unknown_naics_valid | Fine-grained label, **not** a composition key — see the schema guard below |
| `node_type_v5` | *(as v3 rule)* | What the previous precedence would have produced on the same row. Carried so composition churn is reproducible from the parquet |

**Why applicability is a separate axis and not a fourth `naics_status` value.**
Folding "not applicable" into the missingness enum re-creates exactly the
conflation the three-way taxonomy was built to remove — `missing` would again
mean two unrelated things. Quality and applicability are orthogonal; they stay
in orthogonal columns. **Enrichment coverage is `valid / applicable`**, and the
organisation-only denominator is far smaller than v3's, so the reported NAICS
gap will look considerably better without anything having been fixed. Report
the denominator alongside the ratio.

**Precedence switch — `PARTY_TYPE_WINS` (default `True`).** A `P` row carrying
a valid NAICS is a sole proprietor: legally a person, economically a business.
v3 promoted it to `business_naics_valid`; v4 leaves it `individual` and flags
it as `individual_naics_valid` in `entity_class`. Set
`PARTY_TYPE_WINS = False` to reproduce the v3 precedence for A/B comparison.
**Decide this empirically from the disagreement table on the first run** — it
is a live analytical choice, not a default to accept silently.

**Schema guard — `composition_metrics` now raises** if `node_type` carries a
category absent from `_CAT_SHORT`. The `share_*` denominator sums *all*
categories present while columns are emitted only for known ones, so a new
category does not error — it silently makes the shares stop summing to 1.0
while still looking plausible. This is why `individual_naics_valid` lives in
`entity_class` and not in `node_type`: adding it to the composition key is a
schema change to the node table and to `cust_c2c_metrics`, and must be a
deliberate decision rather than a side effect.

`naics_clean` (valid only) is the observed source of truth. Any future
`naics_imputed` stays a separate column forever — imputed NAICS is never fed back
into composition metrics (circularity).

**QA artefacts, written every run.** `qa/customers_typing_disagreement.csv`
carries two confusion matrices: `entity_type_observed × entity_type_inferred`,
and `node_type_v5 × node_type`. Read before briefing any composition share.
The cell that matters most is **observed = business, inferred = individual** —
person-named organisations (trusts, estates, single-member LLCs, sole
proprietorships). The name typer counted these as households. The headline number to
extract is **`node_type` churn**: the share of nodes whose composition category
changed. Above 5% the loader warns, because at that point every `share_*`
column and the population split have moved.

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
>
> **⚠ v4 — this figure is measured on the name typer and is pending
> recomputation.** It was derived when `entity_type` was inferred from
> `customer_name`. Under `party_type` the individual share will move by
> approximately the `node_type` churn reported in
> `qa/customers_typing_disagreement.csv`, and it can only move **down**:
> every reclassification runs individual → business (person-named
> organisations), never the reverse. **Recompute before citing the 94.3% or
> the 48.4% again.** The qualitative conclusion — that the graph is
> household-dominated and that unpartitioned aggregates are retail
> statements — is very unlikely to reverse; the magnitude is what is at
> stake.

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
| **Customer dimension** — identity, NAICS hierarchy, typing, geo status, outer-join resolution | **`pkg_customers.py`** | — | v3: one pass over the extract, replacing two full passes over 23 snapshots. v4: `party_type` typing, applicability axis, deterministic duplicate resolution, QA artefacts. v5: parquet source, dtype-aware ingestion + `_dtype_report`, NULL/'' normalisation, column pushdown. **v6: categorical codes throughout, vectorised name typer, single frame, per-stage RSS logging (§6.3a)** |
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
| Counterparty composition, coverage, guarded sector mix | `pkg_custom_metrics.py` | raw | typing from the customer dimension; >50%-miss fail-fast guard retained; v4 adds the `_CAT_SHORT` schema guard. **v6: `typing_from_customers` no longer copies + `astype(str)` the 30M id column.** `build_node_typing()` (snapshot-based) is **deprecated** — no access to `party_type` |
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

### 6.3a Memory — the v5 OOM post-mortem — **new in v6**

Job 1246888 was OOM-killed in the batch step ~9 minutes after
`customers: parquet read — 11 of 11 mapped columns present` logged
successfully. The read was fine; the process died in the per-row work that
followed. Three causes, in order of size:

| Cause | Cost at 30M rows | Fix |
|---|---|---|
| `np.select` with string choices returns `<U26` — **104 bytes/element** | **3.1 GB per column × 9 columns ≈ 28 GB** transient | `_select_cat()` selects int8 **codes**, builds `Categorical.from_codes` — 1 byte/row |
| Name typer built one Python list per row via `.map()` | ~3.6 GB of list headers, held across 3 further `.map()` passes; 2.85 s/M | `_name_typer()` vectorised on `.str`; agreement with the old logic verified at **1.000** on both outputs |
| Dimension materialised 3× (`attrs`, `typing`, the pipeline's merge) | 3 × the frame | One frame; `.typing` is a projection; pipeline merge removed |

**The `<U26` trap is worth internalising** — it is invisible in code review.
`np.select(conds, ["business_naics_valid", ...])` looks like it produces
strings the way pandas does, but numpy uses a **fixed-width** dtype sized to
the longest label and pads every element to it. The longest `node_type` label
is 26 characters, so every row costs 104 bytes whether it is
`business_naics_placeholder` or `individual`. Adding a typing column with a
long label silently adds ~3 GB. **Any new derived column goes through
`_select_cat`.**

Secondary reductions, all from Categorical storage on low-cardinality
columns (`party_type`, `state`, `city`, `zip3`/`zip5`, `naics_desc`,
`naics2`–`naics6`, and every typing column). `cust_name` stays a plain string
— it is ~unique, so dictionary encoding would only add overhead.

**Instrumentation.** Peak RSS is logged after every stage
(`MEM <stage> peak RSS x.xx GB rows=N`). If this OOMs again, the log names
the stage. `_dtype_report` and the row count are logged at read time, so the
extract's actual shape is on the record.

**Remaining lever — `RESTRICT_DIM_TO_GRAPH` (default `False`).** The
dimension covers ~30.8M MDM parties; the graph touches ~5.3M. Setting this
runs `node_universe()` — one streaming pass over the snapshots reading only
`source`/`dest` — and filters the dimension before the typing stage, cutting
those rows by ~80%. It is off by default because it makes coverage against
`MDM_ALL` underivable from the dimension. Turn it on if the batch step is
still tight; the filter is applied **after** the read and **before** typing,
not pushed into pyarrow (a `filters=` predicate with 5M values is evaluated
per row group and is slower than reading and masking).

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
