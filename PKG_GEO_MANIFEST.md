# PKG_GEO_MANIFEST.md

**Payment Knowledge Graph — Geographic Analytics Manifest**
PNC Bank · Treasury Management · Data Science
Version 1.0 — August 2026

> **Supersedes** `PKG_GEO_METRICS_SPEC.md` v0.9 (draft, August 2026), which was written
> before the Phase A2 validation and assumed a business-dominated node population. That
> assumption was wrong and the spec is retired, not amended.
>
> Companion to `PKG_MONTHLY_METRICS_MANIFEST.md`, `PKG_ROLES_MANIFEST.md`,
> `PKG_CPTY_EXPANSION_MANIFEST.md`, `PKG_RAIL_METRICS_SPEC.md`.
> **Supersedes** UC #8 (Geospatial Payment Corridor Analysis) in `PKN_Roadmap.md`.

---

## 1. Status and scope

The geographic layer is **built and validated**. Address geography joins to the PKG at
99.593%, locates 98.4% of graph-visible customers at rooftop precision, and covers 909 US
markets across 23 months. The infrastructure is sound and cheap to refresh.

The **analytical programme has been substantially reduced** by validation. Two of the
three hypotheses this work was built to test failed against pre-registered decision
rules. What remains is narrower, better evidenced, and correctly targeted.

**This manifest is the source of truth** for geographic column definitions, population
gating, derivation rules, and the build register. Inline conversation is not.

---

## 2. Evidence ledger

Recorded in full because the reversal rate on this workstream has been high — six
findings reversed between ablation rungs, and three of four A2 decision rules fired
against our own hypotheses. Anything not in this table is not established.

### 2.1 Confirmed

| Finding | Value | Basis |
|---|---|---|
| Identifier join `node` ↔ `mdm_id` | **99.593%**, 12-digit numeric, identical under all six normalisations | Phase A §1 |
| Graph closure | residual **0.0%** (Σ net_flow −$1.04 on $4.77B, 2025-11) | Phase A §2 |
| Node geo coverage | **98.39%** (P99_9) | Phase A §6 |
| Strength-weighted coverage | **96.48%** (P99_9), 94.07% (V0) | Phase A §6 |
| Coordinate precision | modal **6 decimals**; ≤2 dp = 3,335 rows of 30.8M | v3 §C2 |
| Coverage vs distance | 96–99.5% every US state; **PA below median** | v3 §E3 |
| Grain | one address per party, all 660 snapshots | v3 §A |
| `addr_type` | constant `ADDRPRM`, 100% | v3 §B1 |
| Hub concentration | top 0.1% of nodes = **97.3%** of dollars | V0 vs P99_9 |
| **Population composition** | **94.26% of nodes are individuals; 51.59% of dollars are business** | A2 §1.1 |
| **Distance gradient** | within 50 km = **0.87×**; beyond 1,000 km = **1.16×**; stable **23 of 23 months** | A2 §4.1 |
| Net position persistence | 72 persistent sources, 134 persistent sinks, 350 unstable (of 556) | A2 §4.2 |
| Column semantics | `strength = in+out`; `net_flow = in−out`; `degree = in+out` (max dev ≤0.5 on ~$10⁹) | A2 §2 |

### 2.2 Reversed or withheld

| Claim | Status | Evidence |
|---|---|---|
| Pittsburgh dollar premium (2.35×) | **Reversed** → 0.87× | Hub artifact; V0→P99_9 |
| 50–150 km hinterland "hole" (0.41×) | **Reversed** → 0.90× | Hub artifact |
| Cleveland largest net sink (+$100.8B) | **Reversed** → persistent net source, 87% of months | Hub artifact + A2 §4.2 |
| Local supply chains absent (0.58% of dollars) | **Reversed** → 18.5% at P99_9, but on a 94%-household population; **unresolved** | A2 §1.1 |
| Triangle closure = where money is (8.2× lift) | **Reversed** → 1.5× | Hub artifact |
| Houston trophic outlier (0.51) | **Reversed** → 5.34 | Hub artifact |
| Florida inflow as a TM opportunity | **Abandoned** — business net **−$212M** (source); household **+$580M** | A2 §1.3 |
| Metro net flow as economic geography | **Reframed** — **89.3%** industry mix, mechanism is labour intensity | A2 §2.1 |
| Metro net flow magnitudes | **Withheld** — sign agreement P99 vs P99_9 = **62.98%**, Spearman 0.116 | A2 §3 |
| Florida seasonality (snowbird) | **Not found** — positive all 23 months, no annual cycle | A2 §4.3 |

### 2.3 The governing fact

Businesses are net **−$7.5B** to households over 23 months; households net **+$7.5B**.
The dominant flow in the graph is payroll. Every geographic pattern observed before the
population split was substantially a map of where people are paid, and the persistent
source/sink structure tracks commuting geography — employment centres pay outward to
their suburbs.

**This is real, durable, and belongs to Retail & Wealth, not Treasury Management.**

---

## 3. Population model

Three orthogonal partitions. **All three must be declared on every emitted table.**
Mixing any of them produces uninterpretable aggregates — this is the central lesson of
A2 and the main structural change from v0.9.

### 3.1 Ablation (`version`)

| | nodes | rows | months |
|---|---|---|---|
| `V0` | 5,289,362 | 49,481,822 | 23 |
| `P99_9` | **3,554,141** | 25,539,971 | 23 |
| `P99` | 3,219,179 | 21,537,310 | 23 |

**`P99_9` is the analytical default.** `V0` is a diagnostic for identifying payment
infrastructure and is never an analytical base. `P99` is **degenerate for net flow** —
metro nets collapse to $0.1–0.8M and the sign becomes noise; use it only for dispersion
robustness, never for flow.

Every reportable geographic result must agree at two adjacent rungs.

### 3.2 Party class (`party_class`) — **new in v1.0, a partition key not a filter**

Dual classifier: NAICS presence (`naics_known`) and legal-entity name tokens. The two
disagreement classes are retained rather than forced binary — a NAICS code without an
entity token is usually a sole proprietor; an entity token without NAICS is an
un-enriched business.

| class | nodes | % nodes | % strength | med strength | `share_in_amt_individual` | months active |
|---|---|---|---|---|---|---|
| `1_biz_both` | 143,943 | 4.12 | 43.51 | $2,500 | 0.318 | 16.4 |
| `2_biz_naics_only` | 38,962 | 1.11 | 5.80 | $1,000 | 0.369 | 15.5 |
| `3_biz_name_only` | 17,818 | 0.51 | 2.28 | $1,649 | 0.392 | 15.0 |
| `4_consumer_like` | **3,296,046** | **94.26** | **48.41** | $340 | 0.631 | 12.5 |
| **business (1+2+3)** | **200,723** | **5.74** | **51.59** | — | — | — |

**Validation:** `share_in_amt_individual` separates 2.0× (0.318 vs 0.631) and median
strength 7.4×. Both are graph-side measures independent of the classifier, so the
separation is not a labelling artifact.

### 3.3 Denominator (`population`)

| | meaning | use |
|---|---|---|
| `MDM_ALL` | 30,854,554 parties | data-quality profiling only |
| `PKG_NODE` | 5,289,362 (V0) / 3,554,141 (P99_9) | coverage denominators |
| `PKG_DOLLAR` | weighted by `strength` | the only figure worth briefing |

Only **17.1%** of MDM parties are graph-visible at V0.

---

## 4. Column register — edge-free blocks

Grain: one row per (`node`, `time_key`, `version`), carrying `party_class`.
All blocks below derive from `cust_c2c_metrics` + the MDM address table. **No GPU pass.**

### Block A — Location

| Column | Definition |
|---|---|
| `geo_status` | `valid` / `placeholder` / `non_us` / `missing` — §5.2 |
| `lat`, `lon` | double, cast only after the string screen (§5.1) |
| `zip5`, `zip4`, `zip3` | `regexp_replace(zip_cd,'[^0-9]','')` then substring. **Never split on `-`** — 83,547 rows are `zip9_flat` |
| `state`, `country` | upper, trimmed |
| `city` | **display only.** Never a join or aggregation key (`ROYAL PLM BCH`) |
| `cbsa`, `cbsa_name` | via HUD USPS crosswalk — **blocked, §8.1** |
| `geo_unit` | interim `zip3 / modal_city / state`; 909 units. Becomes `cbsa_name` on crosswalk arrival |
| `km_from_pit` | haversine from ZIP3 median coordinate to 40.4405 / −79.9959 |
| `coord_decimals` | quality flag; modal 6 |
| `rec_type` | USPS AIS record type — primary placeholder discriminator |

Observed `geo_status` (P99_9): `valid` 3,496,769 · `missing` 25,453 · `not_in_mdm`
14,249 · `non_us` 9,728 · `placeholder` 7,942.

### Block B — Placeholder registry

Modelled on the Hub Node Registry: **labelled taxonomy with per-class policy, not a flat
exclusion list.** 909,820 parties across 16,824 clusters (≥25 parties).

Detection **conditioned on `rec_type`.** Duplicates within `HIGHRISE` are the building
and expected (Horsham PA, 24,047 parties, benign). Duplicates within `NORMAL` are the
signal (Pittsburgh 40.44052/−80.00027, **29,780 parties**, PNC HQ ZIP 15222). A flat
screen catches both or neither.

`n_distinct_line1` is the discriminator:

| Pattern | Class | Policy |
|---|---|---|
| 1 distinct line, many parties | Registered agent / service bureau / bank's own address | Hard placeholder — no point use |
| Many distinct lines, one coordinate | Geocoder snapping ZIP or town to centroid | Soft — valid at CBSA, not at street |
| PO-box concentration, resort/mountain towns | Legitimate (Avon, Breckenridge, Vail) | Already downgraded by `rec_type` — **leave alone** |

Columns: `coord_key`, `rec_type`, `n_parties`, `n_distinct_line1`, `placeholder_class`,
`policy`, `first_seen`, `last_seen`. Written every run.

Secondary uses: entity-resolution candidates, AML shared-address surface, data-quality
feedback to MDM.

### Block C — Relocation history

660 daily snapshots (2024-09-20 → 2026-08-05) make the change log **retroactive**. Daily
address history joined to monthly payment behaviour exists nowhere else in the bank.

| Column | Definition |
|---|---|
| `n_changes_24m` | address or coordinate changes in window |
| `n_moves_24m` | `move_local` + `move_distant` only |
| `km_last_move`, `months_since_move`, `move_direction_deg` | most recent qualifying move |

**Classification must be distance-graded, not hash-based.** The two-hash design failed:
the cleansing vendor rewrites text and coordinates together. Proof — the 5-day interval
2026-07-31 → 2026-08-05 showed 149,844 "relocations" against 137,458 in the preceding 31
days, while `regeocode` scaled cleanly (1,487/day vs 1,488/day).

```
none                 addr_h same AND move_km < 0.1
regeocode            addr_h same AND move_km ≥ 0.1
text_normalization   addr_h changed AND move_km < 0.1
micro_correction     addr_h changed AND move_km < 1.0
move_local           move_km < 50 AND zip5 changed
same_zip_shift       move_km < 50 AND zip5 same
move_distant         move_km ≥ 50
```

All rates normalised per elapsed day.

### Block D — Community locality (proxy)

`community_id` is the one edge-derived grouping persisted at node level; a community's
spatial spread proxies its members' counterparty spread.

| Column | Definition |
|---|---|
| `comm_radius_km` | `sqrt( Σ wᵢdᵢ² / Σ wᵢ )`, `w` = `strength`, `d` = haversine to strength-weighted centroid |
| `comm_median_d_km` | `percentile_approx(d, 0.5)` |
| `dist_to_comm_centroid_km` | node-specific |
| `locality_proxy` | `LOCAL` <50 km · `REGIONAL` <250 · `MULTI_MARKET` <1000 · `NATIONAL` ≥1000 |

Observed (P99_9, **all party classes**): LOCAL 17,795 communities / 130,909 nodes /
11.0 km · REGIONAL 3,700 / 57,005 / 122.3 · MULTI_MARKET 2,918 / 203,754 / 516.5 ·
NATIONAL 350 / 24,753 / 1,322.6.

**Caveat, unresolved:** communities were detected on the full graph, which is 94%
households. LOCAL communities at 11 km radius are more plausibly household and
local-service clusters than supply chains. **Do not describe these as supply chains until
recomputed on business nodes.**

### Block E — B2B flow decomposition — **new in v1.0, the core TM module**

Node-level `net_flow` includes payroll outflow to individuals, which is why even the
business-only source list is Pittsburgh, Houston, Philadelphia — employment centres. The
Treasury Management question is **business-to-business flow**, and it is derivable
edge-free from share columns that already exist.

```
biz_share_in  = share_in_amt_biz_valid  + share_in_amt_biz_placeholder  + share_in_amt_biz_missing
biz_share_out = share_out_amt_biz_valid + share_out_amt_biz_placeholder + share_out_amt_biz_missing

b2b_in       = in_strength  × biz_share_in
b2b_out      = out_strength × biz_share_out
b2b_net      = b2b_in − b2b_out
b2b_strength = b2b_in + b2b_out

payroll_out       = out_strength × share_out_amt_individual
payroll_intensity = payroll_out / out_strength
hub_mediated_out  = out_strength × share_out_amt_hub
```

Emit `b2b_*` alongside the existing totals; never replace them. The NAICS shift-share
(§7.2) must be **recomputed on `b2b_net`** — the 89.3% mix result is a labour-intensity
artifact of payroll and may not survive.

---

## 5. Derivation rules and known traps

### 5.1 Coordinate screen — mandatory before any cast

`latitude_degrees` / `longitude_degrees` are **STRING**. Empty strings, `'0'`,
whitespace, the token `'null'`, and out-of-range values all survive `IS NOT NULL`.

```
1_null → 2_empty → 3_null_token → 4_non_numeric → 5_null_island →
6_out_of_range → 7_US_POSITIVE_LON_BUG → 8_international → 9_ok
```

Positive longitude is an error **only inside the US**. The 2,675 rows flagged in v2 were
Singapore, Dubai, Tokyo, Guam, Paris — correct. They cluster in single-letter `rec_type`
values (`S`, `U`, `B`, `H`) at 8–12 decimal places, suggesting a separate geocoding
source; treat as distinct provenance.

### 5.2 `geo_status`

```
non_us       country ∉ {US, USA, ''}                        — scope exclusion, not a defect
missing      no numeric coordinate, or (0,0)
placeholder  rec_type ∈ {POSTOFFICEBOX, GENERALDELIVERY}
             OR addr_line_1 matches PO-box regex
             OR hard-placeholder cluster (Block B)
valid        otherwise
```

`low_precision` was trialled and **dropped** — 3,335 rows of 30.8M is noise. Collapsed
into `valid`.

### 5.3 Peer normalisation — applies to every dispersion metric

Raw dispersion is confounded by degree and by industry; a node with four counterparties
*cannot* have high CBSA entropy. Two steps:

1. Residualise on `log1p(degree)` and `log1p(strength)`.
2. Percentile-rank within `naics2` × size-decile × `party_class`.

**The reportable quantity is `{metric}_pctile_naics_size`,** never the raw value. Entropy
needs a small-sample guard (bias-corrected estimator, or refuse below 5 counterparties).

### 5.4 Standing traps

| Trap | Rule |
|---|---|
| `strength` double-counts | `strength = in + out`, so cross-node sums count each dollar twice. Consistent as a weight, **wrong as a volume figure** |
| ZIP parsing | digit-strip then substring; **never** split on `-` |
| ID precision | `node` and `mdm_id` are numeric but must stay **string** end to end; 18-digit address IDs exceed float64 exact range |
| Geography is US-only | Canada 8–19%, Mexico 0–1.4% geocoded. Exclude by `geo_status`, do not impute |
| `city` as a key | never |
| Mixing party classes | never aggregate across `party_class` without declaring it |
| Single-month results | nothing reportable without temporal persistence (§7.4) |

---

## 6. Scope caveat — mandatory on every emitted table

The closure residual is 0.0% **because the graph is an internally closed subsystem**:
PNC on-us C2C only. Metro net flow means *"net position versus other PNC customers,"* not
versus the economy.

The household surplus (+$7.5B) is partly a **boundary artifact** — wages arrive from
on-us employers while consumer spending leaves to off-us merchants and card networks.

Every table carries `scope = on_us_c2c`. Revisit when PAYS_CPTY lands, not before.

---

## 7. Validation protocol

Any geographic result must clear all four before it leaves the notebook. Written in
advance of the analysis, not after.

| # | Test | Threshold |
|---|---|---|
| 1 | **Party class** — computed separately for business and household | mandatory |
| 2 | **Industry mix** — shift-share decomposition | reframe if mix >70% of magnitude |
| 3 | **Ablation** — computed at two adjacent rungs | withhold if sign agreement <80% |
| 4 | **Temporal persistence** — across all available months | discard unless stable; k-of-m ≥80% for sign claims |

A2 outcome: rules 1, 2 and 3 fired; rule 4 confirmed the distance-gradient result.

---

## 8. Build register

| # | Item | Status | Rationale |
|---|---|---|---|
| 1 | Block A location, Block B registry | **BUILD** — near complete | Cheap, sound, gates everything |
| 2 | Block E B2B flow | **BUILD — priority** | The actual TM question, never asked, needs no new data |
| 3 | Conduit detection (§8.1) | **BUILD** | Edge-free, AML audience waiting |
| 4 | Block C relocation event study | **BUILD** | Highest novelty per unit effort; asset unique to us |
| 5 | Re-run Blocks A/D on business nodes | **BUILD** | All Phase A geography was measured on a 94%-household population |
| 6 | CBSA crosswalk | **BUILD** | Blocking for any external output |
| 7 | Edge pass (dispersion block) | **HOLD** | Gated on item 2 showing a signal that survives §7 |
| 8 | Corridor matrix + recirculation | **HOLD** | Depends on 7; value reduced by flat dollars |
| 9 | Turnover × deposit attrition rerun | **BUILD — low cost** | Only route to overturning a banked negative result |
| 10 | Metro net flow as a briefable output | **KILL** | Rule 3 failed at 62.98% |
| 11 | Corridor gravity residuals | **KILL** | Gravity on customer *counts* is a much weaker object |
| 12 | Payment ton-miles | **KILL** | Not actionable |
| 13 | Hazard overlay (FEMA) | **KILL** | Flat dollars mean this reports node counts, obtainable more cheaply |
| 14 | Geo imputation ladder (customers) | **KILL** | 98.4% coverage; missing 1.6% is non-US + uncleansed |
| 15 | Household / commuting findings | **TRANSFER** | Retail & Wealth |

### 8.1 Conduit detection — definition

All inputs are node columns; no edges required.

```
conduit_score:  high throughflow
              AND |net_flow / strength| ≈ 0
              AND hhi_in concentrated
```

Crossed with `locality_class`:

| | Reading |
|---|---|
| Local conduit | Payroll agent, escrow, title company — benign; `HubClass` candidate |
| Dispersed, industry rationale | National processor, marketplace |
| **Dispersed, no industry rationale** | AML candidate — funnel / pass-through signature |

At monthly granularity this is **candidate generation only**, never confirmation.

*Open:* confirm whether `throughflow` is `min(in, out)` — §9.

---

## 9. Open items

| Item | Blocks |
|---|---|
| **CBSA crosswalk** (HUD USPS ZIP→ZCTA + Census county→CBSA) | All external output |
| `throughflow` definition — `min(in,out)` or other? | §8.1, turnover work |
| `hub_in_share` / `hub_out_share` — what defines "hub", and does it match the ladder? | Hub-aware aggregation |
| `ga_role` — is `R1` reserved or absent? Observed: R2 peripheral, R3 non-hub connector, R4 non-hub kinless, R5 provincial hub, R6 connector hub, R7 kinless hub | Role × geography |
| `community_id` stability across months | Block D drift work |
| `not_in_mdm` nodes: 14,249 at P99_9 carrying $953.9M | Coverage completeness |
| Counterparty geographic inference — compliance read | §10 |

---

## 10. Counterparty extension

**Same column definitions, different confidence.** No new metrics; the block is reused
with two additions:

| Column | Values |
|---|---|
| `geo_confidence` | `observed` / `payer_centroid` / `fi_prior` / `none` |
| `geo_precision_km` | estimated radius of uncertainty |

**Payer-centroid inference.** Amount-weighted centroid of a counterparty's PNC-side
payers, with payer-cloud dispersion as the confidence measure. Strengthened by the
customer side being 98.4% geocoded at rooftop precision.

**FI-implied geography** (requires `CptyFinEntity`): Tier A single-market community banks
and geographically chartered CUs → soft location; Tier B regional banks → state only;
Tier C national → **share, never place**.

> **The trap, stated explicitly: charter size is not footprint.** BaaS sponsor banks
> (Cross River, Evolve, Sutton, Coastal, Lead) are small charters with nationwide digital
> reach — Tier A on branch count, badly wrong in practice. The taxonomy needs an explicit
> `footprint_reliability` flag set from **business model, not branch geography**. Same
> failure mode as a naive degree-based hub rule.

**Disagreement flag** where FI prior and payer centroid conflict: relocated entity,
national vendor at a local institution, or entity-resolution error. All three worth
seeing.

**Validation posture — corrected.** An earlier recommendation to calibrate an imputation
ladder on customers and carry it to counterparties **does not hold**: at 98.4% coverage
where the missing 1.6% is systematically non-US and uncleansed, that population is not a
representative holdout. **Dual-mask validation is the only honest option** — random
masking for training signal plus a realistic mask reproducing observed missingness, with
both error rates reported. Random-mask performance overstates; say so in every output.

**Hard gate:** `geo_confidence ≠ observed` is acceptable for corridor aggregates,
market-share work and metro exposure. **Never** for hazard overlay, site-level analysis,
or any decision attached to a specific point.

---

## 11. Artifacts

### Code
| | |
|---|---|
| `pkg_geo_address_profile_v3.py/.ipynb` | MDM profiling, coordinate screen, placeholder detection, change log |
| `pkg_geo_phase_a.py/.ipynb` | Join gate, coverage, distance rings, locality proxy |
| `pkg_geo_phase_a2.py/.ipynb` | Party class, shift-share, ablation, temporal persistence |

### Data
| Path | Grain |
|---|---|
| `{WORK_DIR}/pkg_geo_nodes_{tk}` | node × geo block, P99_9 |
| `{WORK_DIR}/pkg_geo_address_all_{snap}` | all MDM addresses, derived flags |
| `{WORK_DIR}/pkg_geo_unit_flows_{tk}` | geo_unit × flow |
| `{WORK_DIR}/pkg_geo_locality_proxy_{tk}` | node × community locality |
| `{WORK_DIR}/pkg_geo_change_log` | party × snapshot × change class |
| `{WORK_DIR}/pkg_geo_biz_vs_consumer_{tk}` | geo_unit × party class × flow |
| `{WORK_DIR}/pkg_geo_shiftshare_{tk}` | geo_unit × mix/local decomposition |
| `{WORK_DIR}/pkg_geo_net_persistence` | geo_unit × persistence class |
| `../metrics/geo/phase_a*_report_*.json` | run manifests |

### Documents
| | |
|---|---|
| `PKG_GEO_MANIFEST.md` | this document |
| `PKG_Geographic_Intelligence.pdf` | executive findings paper, August 2026 |
| ~~`PKG_GEO_METRICS_SPEC.md`~~ | retired at v0.9 |

---

## 12. Change log

| Version | Date | Change |
|---|---|---|
| 0.9 | 2026-08 | Initial spec. Assumed a business-dominated population |
| **1.0** | 2026-08 | **Rebuilt on Phase A2.** `party_class` promoted to partition key; Block E (B2B flow) added as the core TM module; §7 validation protocol formalised; build register added; metro net flow, gravity residuals, ton-miles, hazard overlay and the customer imputation ladder killed; household/commuting findings transferred to Retail & Wealth |

---

*Internal — PNC Treasury Management, Data Science*
