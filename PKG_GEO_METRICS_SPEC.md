# PKG_GEO_METRICS_SPEC.md

**Payment Knowledge Graph — Geographic Metrics Specification**
PNC Bank · Treasury Management · Data Science
Version 0.9 (draft) — August 2026

> Status: **provisional pending Phase A2.** The business-vs-consumer split and the NAICS
> shift-share decomposition can materially change §7 (money flow) and the priority of
> the Tier 1 edge pass. §3–§5 are stable regardless.

---

## 1. Purpose and scope

Defines the geographic attribute block for PKG nodes: column definitions, derivation
rules, confidence tiering, peer-normalisation scheme, and the single GPU edge pass that
unlocks the dispersion metrics.

Companion to `PKG_MONTHLY_METRICS_MANIFEST.md`, `PKG_ROLES_MANIFEST.md`,
`PKG_CPTY_EXPANSION_MANIFEST.md`, `PKG_RAIL_METRICS_SPEC.md`. **Supersedes** the
geospatial corridor use case (UC #8) in `PKN_Roadmap.md`, which assumed `geoLocation`
sparsity and city-level resolution — both wrong, in the favourable direction.

### 1.1 The reframe that governs this document

Phase A established that **dollars per customer are flat across geography**:

| Ring from Pittsburgh | Strength / node (P99_9) | Multiplier |
|---|---|---|
| < 50 km | $31.0K | 0.95 |
| 50–150 km | $29.5K | 0.91 |
| 150–400 km | $31.6K | 0.97 |
| 400–1000 km | $31.6K | 0.97 |
| > 1000 km | $36.7K | 1.13 |

Spread of 1.24×. At V0 the same table spans 5.7× and shows a 2.35× Pittsburgh
premium — **entirely a hub artifact**, which disappears when the top 0.1% of nodes are
excluded.

Consequence: **geography does not predict customer size.** Any metric pitched as "where
the money is" is dead on arrival. What geography does predict is *shape* — footprint
width, consistency with industry peers, change over time, and proximity structure.
This block is a **segmentation and anomaly instrument, not a sizing one.**

---

## 2. Empirical foundation

Everything below rests on findings from `pkg_geo_address_profile_v3` and
`pkg_geo_phase_a`. Numbers are as-run, not assumed.

### 2.1 Source data

| | |
|---|---|
| Address source | `dsihd01p_dsi.neo4j_address`, 18 columns, one row per party |
| Snapshots | **660 daily**, 2024-09-20 → 2026-08-05 |
| Parties (latest) | 30,854,554 |
| Metrics source | `bdahd01p_dlcdi1_cdi_tm.cust_c2c_metrics`, node-level, not an edge list |
| Ablation ladder | `version ∈ {V0, P99, P99_9}` — three rungs, not four |
| Time window | `time_key` 2024-01 → 2025-11, 23 months |

### 2.2 Established facts

| Finding | Value | Implication for this spec |
|---|---|---|
| Identifier join `node` ↔ `mdm_id` | **99.593%**, 12-digit numeric both sides, identical under every normalisation | No normalisation needed; use `raw` |
| Graph closure | residual **0.0%** (Σ net_flow = −$310 on $198.8B) | Net flow is exact **within the on-us subsystem** — see §7.1 |
| Node geo coverage | **98.41%** (V0), 98.39% (P99_9) | Imputation ladder not required for customers |
| Strength-weighted coverage | 94.07% (V0) → **96.48%** (P99_9) | The gap was hubs, not customers |
| Coordinate precision | modal 6 decimals; ≤2 decimals = 3,335 rows of 30.8M | Rooftop-grade. No precision tier needed |
| Coverage vs distance | 96–99.5% in every US state; **PA below median** | **No coverage gradient.** No coverage-adjusted twin statistics needed |
| Non-US coverage | Canada 8–19%, Mexico 0–1.4% | Geography is a **US-only capability** |
| Grain | `n_rows == n_parties` in all 660 snapshots | One address per party; no selection rule |
| `addr_type` | constant `ADDRPRM`, 100% | **No mailing-vs-physical signal exists in MDM** |
| Duplicate coordinates | 909,820 parties on 16,824 clusters (≥25 parties) | Placeholder registry required — §4.4 |
| Graph visibility | 5,289,362 of 30,854,554 = **17.1%** | Three-population discipline is not academic |
| Hub concentration | top 0.1% of nodes = **97.3% of dollars** | All geo work runs at P99_9 by default |

### 2.3 Findings that reversed between V0 and P99_9

Recorded because they set the epistemic posture for everything here. Six major
reversals: the Pittsburgh premium (2.35× → 0.95×), the 50–150 km hinterland hole
(0.41 → 0.91), Cleveland's net position (+$100.8B → −$125.2M, **sign flip**), the
apparent absence of local supply chains (0.58% → 18.5% of community dollars),
triangle-closure lift (8.2× → 1.5×), and the Houston trophic outlier (0.51 → 5.34).

**Rule adopted:** no geographic result is reportable until it has been computed at two
adjacent ablation rungs and shown to agree.

---

## 3. Design principles

1. **P99_9 is the default population.** V0 is a diagnostic for identifying
   infrastructure, never an analytical base. Every emitted table carries `version`.
2. **`geo_observed` and `geo_inferred` never merge.** Permanently separate columns, as
   with `naics_observed`. Applies with force to counterparties (§6), where everything
   is inferred.
3. **Peer-relative, not absolute.** Raw dispersion is confounded by degree and by
   industry. The reportable quantity is always a percentile within
   NAICS2 × size-decile — see §5.
4. **Hub exclusion is structurally prior to aggregation.** Enforced by the ladder rather
   than by a `hub_in_share` threshold, which is cruder and arbitrary.
5. **One definition, two populations.** Customer and counterparty columns share
   definitions exactly; only `geo_confidence` differs. Design once, populate twice.
6. **Fail-fast gates.** Section-level assertions on join rate, closure residual, version
   filter, and `pct_strength_valid`. No silent fallbacks.
7. **US-only scope, declared not discovered.** Non-US parties are excluded by
   `geo_status`, not treated as missing data.
8. **Auditability.** Every run writes a manifest: version, time window, thresholds,
   funnel drop counts, placeholder registry hash.

---

## 4. Tier 0 — available now, no edges required

Derivable from `cust_c2c_metrics` + the MDM address table alone. Grain: one row per
(`node`, `time_key`, `version`).

### 4.1 Location and quality

| Column | Type | Definition |
|---|---|---|
| `geo_status` | string | `valid` / `placeholder` / `non_us` / `missing` — §4.2 |
| `lat`, `lon` | double | Cast from string with the numeric-regex screen (§4.3) |
| `zip5`, `zip4`, `zip3` | string | `regexp_replace(zip_cd,'[^0-9]','')` then substring. **Never split on `-`** — 83,547 rows are `zip9_flat` |
| `state`, `city`, `country` | string | Upper, trimmed. `city` is **display-only**, never a join or aggregation key (`ROYAL PLM BCH`) |
| `cbsa`, `cbsa_name` | string | Via HUD USPS ZIP→ZCTA→county→CBSA crosswalk. **Blocked — see §9.1** |
| `geo_unit` | string | Interim: `zip3 / modal_city / state`. Becomes `cbsa_name` on crosswalk arrival |
| `coord_decimals` | int | `least(dp(lat), dp(lon))`. Retained as a quality flag though 6 is modal |
| `rec_type` | string | USPS AIS record type — the primary placeholder discriminator |
| `shared_addr_n` | int | Parties sharing `addr_norm_hash`. Also an entity-resolution signal |
| `shared_coord_n` | int | Parties at identical `coord_key`, **conditioned on `rec_type`** (§4.4) |

### 4.2 `geo_status` derivation

```
non_us       country ∉ {US, USA, ''}                       — scope exclusion, not a defect
missing      no numeric coordinate, or (0,0)
placeholder  rec_type ∈ {POSTOFFICEBOX, GENERALDELIVERY}
             OR addr_line_1 matches PO-box regex
             OR shared_coord_n ≥ threshold AND n_distinct_line1 = 1   (§4.4)
valid        otherwise
```

`low_precision` was trialled as a fifth value and **dropped** — 3,335 rows of 30.8M is
noise, not a population. Collapsed into `valid`.

Observed distribution (P99_9, graph-visible): `valid` 3,496,769 · `missing` 25,453 ·
`not_in_mdm` 14,249 · `non_us` 9,728 · `placeholder` 7,942.

### 4.3 Coordinate screen (mandatory before any cast)

`latitude_degrees` / `longitude_degrees` are **STRING**. Empty strings, `'0'`,
whitespace, the literal token `'null'`, and out-of-range values all survive
`IS NOT NULL`. Screen order:

```
1_null → 2_empty → 3_null_token → 4_non_numeric → 5_null_island →
6_out_of_range → 7_US_POSITIVE_LON_BUG → 8_international → 9_ok
```

Two notes. Positive longitude is only an error **inside the US** — 2,675 rows are
Singapore, Dubai, Tokyo, Guam, Paris, and are correct. And the international rows
cluster in the single-letter `rec_type` values (`S`, `U`, `B`, `H`) at 8–12 decimal
places, which looks like a separate geocoding source; treat as a distinct provenance.

### 4.4 Placeholder registry

Modelled on the Hub Node Registry: **a labelled taxonomy with per-class policy, not a
flat exclusion list.**

Detection must be **conditioned on `rec_type`**. Duplicates within `HIGHRISE` are the
building and are expected (Horsham PA: 24,047 parties, benign). Duplicates within
`NORMAL` are the signal (Pittsburgh 40.44052/−80.00027: **29,780 parties**, PNC's own
HQ ZIP 15222). A flat screen catches both or neither.

`n_distinct_line1` is the discriminator:

| Pattern | Reading | Policy |
|---|---|---|
| 1 distinct line, many parties | Registered agent, service bureau, bank's own address | Hard placeholder — exclude from all point use |
| Many distinct lines, one coordinate | Geocoder snapping a ZIP or town to a centroid | Soft placeholder — valid at CBSA, not at street |
| PO-box concentration in resort/mountain towns | Legitimate (Avon, Breckenridge, Vail — limited street delivery) | Already downgraded by `rec_type`. **Leave alone** |

Registry columns: `coord_key`, `rec_type`, `n_parties`, `n_distinct_line1`,
`placeholder_class`, `policy`, `first_seen`, `last_seen`. Written to disk every run.

**Secondary uses**, worth stating so the artifact isn't seen as pure overhead: shared-address
clusters are entity-resolution candidates, an AML surface, and a data-quality signal back
to MDM.

### 4.5 Community-derived locality (proxy)

`community_id` is the one edge-derived grouping already persisted at node level.
A community's spatial spread proxies its members' counterparty spread.

| Column | Definition |
|---|---|
| `comm_radius_km` | `sqrt( Σ wᵢ dᵢ² / Σ wᵢ )` where `d` = haversine to the strength-weighted community centroid, `w` = `strength` |
| `comm_median_d_km` | `percentile_approx(d, 0.5)` |
| `dist_to_comm_centroid_km` | Node-specific distance — a node at the centre of a tight community differs from one at the edge of a national one |
| `locality_proxy` | `LOCAL` < 50 km · `REGIONAL` < 250 · `MULTI_MARKET` < 1000 · `NATIONAL` ≥ 1000 |

Observed (P99_9): LOCAL 17,795 communities / 130,909 nodes / avg radius 11.0 km ·
REGIONAL 3,700 / 57,005 / 122.3 · MULTI_MARKET 2,918 / 203,754 / 516.5 ·
NATIONAL 350 / 24,753 / 1,322.6.

**This is a proxy and it is community-level** — every member inherits the same radius.
Tier 1 replaces it with true counterparty dispersion. **Retain both columns after the
swap**: the proxy-vs-actual comparison quantifies how good the proxy was, which is what
makes the eventual edge pass a validation exercise rather than a leap.

### 4.6 Relocation history

660 daily snapshots make the change log **retroactive**. This is a differentiated asset:
daily address history joined to monthly payment behaviour exists nowhere else in the
bank.

| Column | Definition |
|---|---|
| `n_changes_24m` | Address or coordinate changes in the window |
| `n_moves_24m` | Changes classified `move_local` or `move_distant` only |
| `km_last_move` | Haversine of the most recent qualifying move |
| `months_since_move` | Months since |
| `move_direction_deg` | Bearing of the most recent move |

**Classification must be distance-graded, not hash-based.** The v2 two-hash design
failed: the cleansing vendor rewrites address text and coordinates together, so
"both changed" captured normalisation, not movement. Proof — the 5-day interval
2026-07-31 → 2026-08-05 showed 149,844 "relocations" against 137,458 in the preceding
31 days, while `regeocode` scaled cleanly (1,487/day vs 1,488/day).

```
none                addr_h same AND move_km < 0.1
regeocode           addr_h same AND move_km ≥ 0.1
text_normalization  addr_h changed AND move_km < 0.1
micro_correction    addr_h changed AND move_km < 1.0
move_local          move_km < 50 AND zip5 changed
same_zip_shift      move_km < 50 AND zip5 same
move_distant        move_km ≥ 50
```

All rates normalised per elapsed day — unequal snapshot intervals are otherwise
uncomparable, which is exactly what hid the problem.

**Downstream:** the relocation event study — does payment structure change around a
move, and does it lead or lag — is self-contained, needs no edges, and is the highest
novelty-per-unit-effort item in the whole geo programme.

---

## 5. Peer normalisation (applies to Tiers 1–2)

**Raw dispersion is not reportable.** Radius of gyration, CBSA count, and entropy all
rise with counterparty count for pure sampling reasons — a node with four counterparties
*cannot* have high CBSA entropy. Correlating dispersion with strength without
conditioning rediscovers "big nodes are spread out."

Two-step, applied to every Tier 1–2 dispersion column:

1. **Residualise** on `log1p(degree)` and `log1p(strength)`; retain the residual.
2. **Percentile-rank within `naics2` × size-decile.** Emit
   `{metric}_pctile_naics_size`.

Entropy needs a small-sample guard: Shannon entropy is biased low at small *n*. Either
use a bias-corrected estimator or refuse to emit below a counterparty-count floor
(suggest 5) rather than emitting a number that silently means "small."

**The reportable quantity is `dispersion_pctile_naics_size`**, not `radius_gyration_km`.
The question it answers — *given your industry and size, is your footprint unusually
wide or narrow?* — is the one with content. A restaurant with national counterparties is
a franchise, a fraud, or a mislabelled NAICS. A wholesaler with a 20 km radius is a
distributor that hasn't scaled. Neither is visible without peer conditioning.

---

## 6. Tier 1 and Tier 2 — the edge-dependent block

### 6.1 Why edges are unavoidable

Geography is a property of the **counterparty set**. `cust_c2c_metrics` carries
counterparty *statistics* (`hhi_in`, `naics2_entropy_in`, `n_naics2_in`) but not
counterparty *identities*. Any metric of the form "distance to counterparties" requires
the edge list. No workaround exists.

### 6.2 Tier 1 columns

| Column | Definition |
|---|---|
| `radius_gyration_km` | `sqrt( Σ wᵢ dᵢ² / Σ wᵢ )`, `w` = edge amount, hubs excluded |
| `d_p50_km`, `d_p90_km` | Amount-weighted distance percentiles — typical vs tail reach |
| `n_cbsa_80` | Distinct CBSAs covering 80% of amount |
| `cbsa_entropy` | Shannon over amount shares by CBSA, bias-corrected |
| `home_cbsa_share` | Share of amount with counterparties in the node's own CBSA |
| `in_radius_km` / `out_radius_km` | **Computed separately** — revenue footprint vs supply footprint |
| `bearing_R` | Mean resultant length of amount-weighted bearings; `1 − R` is circular variance |
| `registered_vs_flow_km` | Distance between registered point and flow-weighted centroid |
| `locality_class` | Same four bands as §4.5, from `radius_gyration_km` |

**`in_radius` / `out_radius` is the highest-value pair in the block.** In-geography is
where revenue comes from (market footprint); out-geography is where money goes
(procurement footprint). The *gap* between them is a trade-role classifier on its own —
local-in/national-out is a distributor or importer; national-in/local-out is a producer
serving distant markets from a local supply base. Reads directly onto `trophic_level`:
trophic position says where in the chain, in/out spread says how wide the chain is at
that position.

**`registered_vs_flow_km` is the representativeness test.** With `addr_type` constant,
this is now the *only* available check on whether a registered address means anything.
Small gap + small radius → the address is meaningful. Registered in Pittsburgh, centroid
in Chicago, radius 900 km → it's an HQ artifact and should be used as a label, not a
location. Gate: point-level analytics (hazard, site selection, branch proximity) restricted
to nodes passing this test.

### 6.3 Tier 2 columns — needs edges plus time

| Column | Definition |
|---|---|
| `centroid_drift_km` | MoM displacement of the flow-weighted centroid |
| `drift_intensive_km` | Retained-counterparty margin only |
| `drift_entry_km` | New counterparties |
| `drift_exit_km` | Churned counterparties |
| `drift_cross_km` | Interaction term — **reported, not buried** |
| `new_cpty_distance_delta` | Mean distance of new counterparties minus mean of existing |
| `expansion_class` | `EXPANSION` / `REORIENTATION` / `CONSOLIDATION` / `FRAGMENTATION` |

The shift-share partition is nearly free: `n_payer_retained` / `_new` / `_lost`,
`n_payee_*`, and the matching `*_amount_share` columns **already exist**. Only the
distances are missing.

Why it matters: a customer whose centroid moved 400 km entirely on the intensive margin
is reallocating spend among existing partners. One that moved the same distance on the
entry margin just onboarded a vendor. Same displacement, opposite business stories, and
only the decomposition separates them.

`expansion_class` requires bearing statistics — radius alone conflates growth with
substitution:

- radius ↑, bearings dispersed → **EXPANSION**
- radius flat, mean bearing shifts → **REORIENTATION** ← competitive displacement with
  no volume change. Invisible in any dollar dashboard. The most valuable of the four.
- `n_cbsa_80` ↓, `home_cbsa_share` ↑ → **CONSOLIDATION**
- `n_cbsa_80` ↑, `home_cbsa_share` ↓ → **FRAGMENTATION**

Two guards. **Persistence:** a drift is real only if it survives k-of-m months — reuse
the temporal-persistence convention from the backbone work rather than inventing a
second one. **Seasonality:** geographic footprint has strong NAICS-conditional
seasonality (construction, agriculture, retail, tourism); measure against a NAICS-peer
seasonal baseline, not the node's own prior month.

---

## 7. Money flow modules

### 7.1 The scope caveat — governs everything in this section

The closure residual is 0.0% **because the graph is an internally closed subsystem**:
PNC on-us C2C only. Metro net flow therefore means *"net position versus other PNC
customers,"* not *"net position versus the economy."*

Philadelphia as a net source means it pays other PNC customers more than they pay it.
That is a real and interesting fact about the book. It is **not** a statement about
Philadelphia's economy, and briefed as one it will be taken apart in the room.

Every emitted table carries a `scope = on_us_c2c` column and the caveat in its header.
Once PAYS_CPTY lands the boundary becomes measurable and net flow becomes economically
interpretable — at which point this caveat is revisited, not before.

### 7.2 Money recirculation — the local multiplier

**The strongest candidate for a genuine finding rather than a dashboard.**

How many times does a dollar change hands within a metro before leaving it? On a closed
flow network this is a well-posed absorbing-Markov-chain problem on the CBSA × CBSA
corridor matrix — ~400×400, trivially small once the matrix exists.

```
P = row-normalised CBSA × CBSA amount matrix
N = (I − Q)⁻¹    where Q is the within-market block
local_multiplier[m] = expected number of within-market transfers before absorption
```

Converts net flow from a curiosity into an economic quantity with an obvious audience:
which markets are **sticky**, where money circulates versus passes through. Pairs
naturally with community cohesion (§4.5). Blocked on the corridor matrix (§8).

### 7.3 Throughflow × deposit balance = turnover

`throughflow` is a node column; deposit balances come from the deposit tables. Their
ratio is a **velocity** metric neither dataset produces alone.

This matters because of a result already banked: raw graph metrics were **coincident**
with deposit attrition, not leading. Turnover is a different object — a customer whose
payment volume holds while balances fall behaves differently from one where both fall
together, and only the ratio separates them.

**Recommended:** rerun the deposit-attrition modules unchanged, with turnover added as a
feature. Low cost, and the most likely route to overturning a negative result already on
the books. Mind the deliberate window mismatch (deposits to mid-2026, graph to late 2025)
that makes the freeze test zero-leakage.

*Open:* confirm whether `throughflow` is `min(in_strength, out_strength)` or another
definition — §9.3.

### 7.4 Conduit detection — available today, no edges

`throughflow`, `flow_ratio`, `net_flow`, `hhi_in`, `hhi_out` are all node columns.

```
conduit_score:  high throughflow
              AND |net_ratio| ≈ 0
              AND hhi_in concentrated
```

Crossed with `locality_class`, this becomes geographically legible:

| | Reading |
|---|---|
| Local conduit | Payroll agent, escrow, title company — benign, and a `HubClass` candidate |
| Dispersed conduit, industry rationale | National processor, marketplace |
| **Dispersed conduit, no industry rationale** | AML candidate. Funnel/pass-through signature |

This is the Funnel & Pass-Through Detector (UC #3) in geographic form, computable **this
week**, without the snapshots. At monthly granularity it is candidate generation only,
never confirmation — same posture as kiting.

### 7.5 Trophic × net position by metro

Both are node columns; NAICS-adjust and cross them. Two dimensions — where a market sits
in the money chain, and whether it is a net receiver — give a 2×2 more interpretable than
either alone. Upstream+source = producing region; downstream+sink = consuming region.

### 7.6 Second-order geographic exposure

One- and two-hop revenue-source concentration by metro, amount-weighted. Finds the
customer in Cleveland who is *not* Pittsburgh-located but derives 60% of inflows from
Pittsburgh payers — exposed to a regional shock, and invisible to any address-based
analysis. Genuinely beyond SQL. Needs the edge pass; two-hop needs traversal beyond the
corridor matrix.

### 7.7 Deprioritised, with reasons

| Module | Why cut |
|---|---|
| Corridor gravity residuals | Dollars are flat; a gravity model on customer *counts* is a much weaker object |
| Payment ton-miles (Σ amount × distance) | Interesting, not actionable |
| Hazard overlay (FEMA) | Flat dollars mean this reports node counts, obtainable more cheaply |
| Geo imputation ladder (customers) | 98.4% coverage; the missing 1.6% is non-US + uncleansed. Not worth building |

---

## 8. The edge pass

### 8.1 Principle

The edge dependency is a **materialisation step, not a per-analysis requirement**. One
monthly GPU job produces two artifacts; everything downstream returns to Spark.

| Artifact | Grain | Approx size |
|---|---|---|
| `pkg_geo_node_block` | node × month × version | ~20 cols × 3.5M rows |
| `pkg_geo_corridor_matrix` | cbsa_o × cbsa_d × month × version | ~100K rows/month |

The corridor matrix is the input to §7.2 and to all market-share work. It is small
enough to hold in memory indefinitely.

### 8.2 Sequence

1. Restrict the geo lookup to **PKG-visible nodes only** before it goes near the GPU.
   30.8M rows will not fit; ~3.5M will.
2. Join lat/lon onto both edge endpoints in cuDF.
3. Vectorised haversine in cuPy. **Never `.apply()` per edge.**
4. Per-node aggregations → node block.
5. Per-(cbsa_o, cbsa_d) aggregation → corridor matrix.
6. Write both to Hive. Spark takes over.

### 8.3 cuDF implementation notes

Patterns that have caused problems before, recorded so they aren't rediscovered:

- **Haversine** — vectorised cuPy on whole columns. A per-edge `.apply()` will not
  complete at this scale.
- **CBSA entropy** — two groupbys (node×cbsa → shares, then node → `−Σ p log p`), never
  a `groupby.apply`.
- **`n_cbsa_80`** — global sort by (node, amount desc), then in-group `cumsum` and a
  threshold. **Not** `groupby.nlargest`.
- **Percentiles** (`d_p50`, `d_p90`) — sort-based within-group rank, not per-group
  quantile calls.
- **Bearings** — accumulate `Σ w·sin(θ)` and `Σ w·cos(θ)` separately, then
  `R = hypot(...) / Σw`. Never average angles directly.
- **Memory** — partition by `time_key`; do not attempt the full 23-month edge set in one
  pass.
- **IDs** — `node` is 12-digit numeric but must stay **string** end to end. 18-digit MDM
  address IDs exceed float64 exact-integer range and will silently lose precision.

### 8.4 Gate

Do not schedule the edge pass until Phase A2 returns. If the shift-share shows metro net
flow is largely NAICS composition, the corridor matrix is worth considerably less than
it currently appears, and Tier 1 should be scoped down to the node block alone.

---

## 9. Open questions and blockers

### 9.1 CBSA crosswalk — **blocking for any stakeholder-facing output**

ZIP3 is the interim unit: 909 units, labelled `zip3 / modal_city / state`. It splits
metros (Miami / Fort Lauderdale / West Palm are separate rows for one market; New York
appears as `100`, `101`, `104`) and merges others. Acceptable internally, **not**
acceptable in a deck — and it demonstrably weakens the Florida result.

Required: HUD USPS ZIP→ZCTA crosswalk (small, free, quarterly) + Census county→CBSA.
Sourcing is a process question, not an analytics one. Start now.

### 9.2 Phase A2 outcomes

| Result | Consequence for this spec |
|---|---|
| Florida inflow majority **consumer** | §7 metro net flow moves to Retail/Wealth. TM keeps the business residual only |
| Shift-share `pct_mix` **> 70%** | Reframe as industry finding; scope §8 down to the node block |
| P99 vs P99_9 sign agreement **< 80%** | Net flow is a threshold artifact; §7.1–7.2 do not ship |
| Ring multipliers **unstable across months** | §1.1 reframe is a single-month accident; whole spec reverts to provisional |

### 9.3 Definitional confirmations needed

- `throughflow` — `min(in, out)` or another definition? Gates §7.3.
- `hub_in_share` / `hub_out_share` — what defines "hub" in their construction, and does
  it match the ablation ladder's criterion?
- `ga_role` taxonomy — observed values `R2_peripheral`, `R3_non_hub_connector`,
  `R4_non_hub_kinless`, `R5_provincial_hub`, `R6_connector_hub`, `R7_kinless_hub`. Is
  `R1` reserved or absent?
- `community_id` stability across months — required before Tier 2 drift work, and
  independently flagged as a risk in `PKN_Roadmap.md`.

### 9.4 Governance

- **Counterparty geographic inference** is a heavier ask than industry inference and
  folds into the compliance read already flagged as a precondition for prospecting on
  non-customers.
- Scope inference to **business entities, not individuals**.
- Any UI-facing geographic aggregate needs a **k-anonymity floor** (interim:
  `MIN_UNIT_NODES = 25`).
- The placeholder registry contains PNC's own addresses. Treat as internal.

---

## 10. Counterparty extension

**Same column definitions, different confidence.** No new metrics are defined for
counterparties; the block is reused with two additions:

| Column | Values |
|---|---|
| `geo_confidence` | `observed` / `payer_centroid` / `fi_prior` / `none` |
| `geo_precision_km` | Estimated radius of uncertainty |

### 10.1 Payer-centroid inference

The amount-weighted centroid of a counterparty's PNC-side payers, with dispersion as the
confidence input. Phase A strengthens this considerably: the customer side is **98.4%
geocoded at rooftop precision**, so the payer cloud is well-observed.

Paid only by Allegheny County customers in one NAICS → high-confidence local business.
Paid by customers in 30 states → national vendor, centroid meaningless. **The dispersion
of the payer cloud is itself the confidence measure.**

### 10.2 FI-implied geography

Requires `CptyFinEntity`. Tiered:

| Tier | Institution type | Use |
|---|---|---|
| A | Single-market community banks, geographically chartered CUs | Soft location assignment |
| B | Regional banks | State-level aggregates only |
| C | National banks, national digital footprint | **Share, never place** |

**The trap, stated explicitly: charter size is not footprint.** BaaS sponsor banks
(Cross River, Evolve, Sutton, Coastal, Lead) are small charters with nationwide digital
reach — they look Tier A on branch count and are badly wrong. The FI taxonomy needs an
explicit `footprint_reliability` flag set from **business model, not branch geography**.
Same failure mode as a naive degree-based hub rule: it misclassifies exactly the
institutions that matter most.

### 10.3 Disagreement flag

Where the FI prior and the payer centroid disagree, flag it. Three causes, all worth
seeing: a relocated entity, a national vendor banking at a local institution, or an
entity-resolution error.

### 10.4 Validation posture — a correction

An earlier recommendation to build the imputation ladder on customers now and carry it to
counterparties **does not hold.** At 98.4% customer coverage where the missing 1.6% is
systematically non-US and uncleansed, that population is not a representative holdout for
a model meant to transfer.

Consequence: **dual-mask validation is the only honest option.** Random masking of known
geographies for training signal, plus a realistic mask reproducing the observed
missingness pattern, with **both** error rates reported. Random-mask performance will
overstate — say so in every output.

### 10.5 Hard gate

`geo_confidence ≠ observed` is acceptable for corridor aggregates, market-share work, and
metro-level exposure. It is **never** acceptable for hazard overlay, site-level analysis,
or any decision attached to a specific point.

---

## 11. Implementation sequence

| # | Item | Depends on | Effort |
|---|---|---|---|
| 1 | Phase A2 four checks | — | in flight |
| 2 | CBSA crosswalk sourcing | external file | process, not analytics |
| 3 | Tier 0 node block (§4) | 1, 2 | low — mostly written |
| 4 | Placeholder registry (§4.4) | 3 | low |
| 5 | Conduit detection (§7.4) | 3 | low — no edges |
| 6 | Relocation event study (§4.6) | 3 | medium — self-contained, high novelty |
| 7 | Turnover × attrition rerun (§7.3) | 3, §9.3 | medium — reuses existing modules |
| 8 | Edge pass (§8) | 1 gate, 2 | high — one GPU job |
| 9 | Tier 1 block (§6.2) | 8 | medium |
| 10 | Corridor matrix + recirculation (§7.2) | 8 | medium |
| 11 | Tier 2 drift (§6.3) | 9 + ≥6 months | medium |
| 12 | Counterparty extension (§10) | PAYS_CPTY, CptyFinEntity, compliance | blocked |

Items 5 and 6 are the near-term wins: neither needs the edge pass, and both produce
something no other team can produce.

---

## 12. Change log

| Version | Date | Change |
|---|---|---|
| 0.9 | 2026-08 | Initial draft. Supersedes UC #8 in `PKN_Roadmap.md`. Provisional pending Phase A2 |

---

*Internal — PNC Treasury Management, Data Science*
