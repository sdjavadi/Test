# PKG Attrition Manifest — v1

**Payment Knowledge Graph (PKG) — early attrition signals from a deposit-anchored payment graph**
*PNC Bank · Treasury Management · Data Science*
*Supersedes `PKN_Attrition_Analysis_Brief.md` (working team draft)*
*Tracks: modules not yet written — see §9*

> **Why this replaces the brief rather than annotating it.**
>
> The brief was written against the assumption that the study runs on the
> published on-us C2C graph plus a deposit panel. The staging table changes
> the substrate, and three of the brief's structural constraints do not
> survive the change:
>
> 1. **Coverage inverts.** The prior attrition verdict — *the graph is a
>    coincident indicator, not a leading one* — was stated conditional on
>    ~12% of the deposit book being graph-visible. That 12% is an artefact
>    of intersecting the deposit book with a graph built customer-first.
>    Anchoring extraction on the deposit book instead means every labelled
>    customer has a payment neighbourhood **by construction**, including its
>    off-us side. The condition under which the earlier work said a rerun
>    would be warranted is now met. **This is a rerun against a different
>    graph, not against the PKG snapshot** (§1.3).
> 2. **The counterparty key is an identifier, not a string.** Account number
>    + routing number arrive from source. Entity resolution moves off the
>    critical path: the node key is the external account, and the name is an
>    attribute of it. `fi_destination_flag` becomes an **RTN lookup**, exact
>    and auditable, not a fuzzy match against a maintained name list. The
>    brief's §7 "On name matching" paragraph and the token-set-ratio-at-85%
>    rule are **removed**; see §2.
> 3. **The label can be real.** Account status is obtainable in the deposit
>    data (§3). The brief treated this as an open item; it was in fact the
>    binding constraint, because a balance-derived T0 makes a deposit model
>    predict a deposit-defined event and win by construction.
>
> **Four corrections to the brief carried forward as standing facts:**
>
> - **§6.4's account of the first attempt is wrong.** It attributes deposit
>   dominance to leakage. The prior pipeline ran an out-of-time freeze with a
>   deliberate window mismatch precisely so leakage was impossible; deposit-only
>   reached freeze AUC ≈ 0.905 and the graph added no incremental lift. The
>   cause was coverage and label construction, not a dirty harness. **This
>   correction must survive into any circulated version**, or the team will
>   re-run a clean experiment that was already clean.
> - **§7 Group F is not independent.** `cpty_spread_mom` is computed over the
>   same counterparty set whose thinning Group B measures. Correlated by
>   construction; it cannot be treated as free information.
> - **§8's weekly/monthly split is superseded.** Grain is now a scoping
>   decision per tier (§1.2), not a statistical-stability argument.
> - **Naming.** The artefact is the Payment Knowledge Graph (PKG). "PKN"
>   should not appear in circulated versions.

---

## 0. Scope & conventions

| | |
|---|---|
| **Substrate** | Neo4j ingestion staging table (transaction grain, settled) + deposit panel + account status |
| **Scope tag** | `deposit_anchored_ego_d1` — **never** written into the same columns as `on_us_c2c` (§7) |
| **Population** | Corporate DDA / MMDA book; `party_type = O` gate applied and reported separately |
| **Node key** | On-us: `mdm_id`. Off-us: `ext_acct_key` = hash(RTN, account_number) — see §2.1 |
| **Edge weight** | `amount`. `volume` is descriptive, never a weight |
| **Direction** | Confirmed trustworthy at source. Do not infer from sign |
| **Grain** | Tier 1 monthly, Tier 2 daily — §1.2 |
| **Date field** | **Settlement date, with lag.** This is a leakage surface, not a nuisance — §6.2 |

### 0.1 What this study is not

It is not an extension of `cust_c2c_metrics`. The graph built here has a
different population, a different boundary, and a different observation
window. A `geo_spread_km` or a `share_in_cp_individual` computed on this
substrate is **not the same quantity** as the published column of that name.
Different scope, both observed — which makes the conflation harder to catch
than the observed/inferred case, because nothing looks wrong.

---

## 1. Population and tiering

### 1.1 The account-to-customer roll-up

Deposits are account-level; a customer holds several accounts of different
types. Every downstream quantity depends on a roll-up rule that has not been
specified, and the case that decides it is the one that matters most:
**a customer closes one account while others continue.**

Three candidate rules, to be chosen empirically, not by preference:

| Rule | Behaviour on partial closure | Risk |
|---|---|---|
| Sum all DDA/MMDA | Balance steps down; looks like attrition | False positives on internal consolidation |
| Sum, excluding accounts closed in-window | No step; the closure is invisible | Misses the strongest true signal |
| Sum, **plus** a separate `n_accounts_closed` feature | Step is visible *and* attributable | Preferred — keeps level and event separate |

**Decision: rule 3.** Level and event are different signals and must not be
folded into one series. This is the same principle as the applicability axis
in the metrics manifest — do not encode two facts in one column.

Open: does the roll-up include time deposits, sweep vehicles, and analysis
accounts, or DDA/MMDA only? Sweep vehicles in particular will manufacture
apparent drains.

### 1.2 Two tiers

| Tier | Population | Grain | Aggregation | Serves |
|---|---|---|---|---|
| **T1** | Full corporate DDA book | Monthly | `(cust, ext_acct_key, month)` | Groups A, C, D, F; episode table; scorecard |
| **T2** | Cases + matched controls (low thousands) | Daily / event | Transaction rows retained | Group B cadence, T0 dating, sequence tests |

Tier 1 is the reusable asset and should be built to survive this study. Tier 2
is cohort-scoped and disposable.

**What daily grain actually buys.** Two things, and only two:

1. The `inflow_outflow_decline_gap` fingerprint becomes a testable **sequence**
   — do receivables redirect before payables, or is that a plausible story
   nobody has measured? Monthly buckets cannot answer it.
2. Missed-recurring detection becomes an event with a date rather than a
   bucket approximation.

It buys nothing for level-based metrics. Do not pay for daily where monthly
suffices; the reason is compute, and the reason to say so here is that "more
granular is better" will otherwise drive the design.

### 1.3 Funnel — must be counted before any modelling

The brief never applies a visibility filter to the population. Under
deposit-anchored extraction the filter mostly disappears, but the count still
gates the design:

```
corporate DDA/MMDA book
  → ≥12 months deposit history            (burn-in)
  → pre-decline median balance ≥ $25K     (floor, tune to book)
  → not seasonally recurrent              (prior-year comparison)
  → decline persists ≥3 months            (persistence)
  → present in staging table              (expected ~all; verify)
  → ≥N resolvable counterparties/month    (N tbd, gates Groups B and C)
```

**If this leaves fewer than ~500 episodes, §5 must be cut before Step 3 runs.**
Twenty-plus metrics against matched controls at two horizons on a few hundred
episodes is a multiple-comparisons exercise wearing an analysis costume. The
count is one query (§8, Q13) and it precedes everything.

---

## 2. The counterparty key — the central design change

### 2.1 Key construction

```
ext_acct_key = sha256(normalise(routing_number) || "|" || normalise(account_number))
```

- **The key is the account, not the name.** One external account = one node,
  regardless of how its name is spelled across rails and months.
- Name, when present, is an **attribute** of the account with a coverage flag,
  not a join condition.
- Account numbers are sensitive: hash at extraction, keep the salt in the
  restricted table, never let the raw number into `../metrics/`.
- Normalisation: strip whitespace and leading zeros consistently, uppercase,
  drop non-alphanumerics. **Decide leading-zero handling once and record it** —
  the zip-code lesson from the metrics manifest applies exactly.

### 2.2 What the RTN unlocks

| Derived | Source | Replaces |
|---|---|---|
| `dest_fi_rtn`, `dest_fi_name` | FRB E-Payments Routing Directory (FedACH + Fedwire participant files) | The brief's maintained bank-name list |
| `fi_destination_flag` | Deterministic: is the RTN a depository institution the customer is not us | Fuzzy name match |
| `dest_fi_type` | FDIC / NCUA join on the FI | — |
| `dest_fi_geography` | FDIC Summary of Deposits | — |

This is the same registry already scoped as the **FI Pinning Registry** for
locatability. Build it once, in one place, and let both consume it. It is the
only external ground-truth set of counterparties with known locations, and it
now has a second consumer.

### 2.3 RTN traps that will produce wrong conclusions

| Trap | Effect | Handling |
|---|---|---|
| **Correspondent masking** | Payment cleared through a correspondent carries the correspondent's RTN, not the true destination bank | Flag known correspondent RTNs; treat `dest_fi` as unresolved on those, do not report the correspondent as the competitor |
| **Sponsor-bank routing** | Fintech / neobank accounts route to a small number of sponsor banks. A sudden cluster of outflow to one sponsor RTN is not one competitor winning | Maintain a sponsor-bank list; classify separately |
| **Internal / book transfer** | On-us movement may carry a PNC RTN or none | Classify explicitly as `on_us_internal`; never counts toward `off_us_outflow_share` |
| **Same RTN, many entities** | Large banks concentrate; `dest_fi` is low-cardinality and high-frequency | Never use `dest_fi` alone as a feature; use it in combination with new-appearance and share-delta |
| **Returns and reversals** | An ACH return manufactures an outflow followed by an inflow, both real rows | Net returns before aggregation; keep a returned-volume feature — rising returns are their own distress signal |

### 2.4 Self-payment detection — staged

**v1: exact match on normalised name**, per your call. Normalisation only:
uppercase, strip legal suffixes (LLC / INC / CORP / LP / LTD / CO), collapse
whitespace and punctuation. No fuzzy scoring in v1.

**Calibration set, available now and free:** on-us self-payments between two
PNC accounts held by the same `mdm_id`. Same entity, same name-string problem,
**known answer**. Fit precision and recall there before extending the matcher
off-us. Same design as the locatability harness — hide the answer, predict it
back, transfer with a stated shift caveat. The shift to bound: off-us names
come from rail fields, not MDM, so they are dirtier.

**The behavioural point matters more than the matching.** A same-name outflow
is *not* attrition. Most corporates permanently maintain multiple bank
relationships; the base rate is high. **The signal is the delta, never the
level** — a *newly appearing* same-name destination, or a step change in share
to an existing one. Measure the base rate first (§8, Q10) or the flag fires on
a large fraction of the book on day one.

---

## 3. The label — three definitions, ranked

The brief has one label, derived from the balance series. That is circular:
a deposit model predicting a deposit-defined event wins by construction, and
the graph can only add value in the narrow band where it moves before a
balance-defined onset. With account status available, that changes.

| # | Label | Definition | Role |
|---|---|---|---|
| **L1** | **Status-based** | Account closed / dormant per the deposit system | **Primary model target.** The real event |
| **L2** | **Behavioural** | Sustained transaction silence — no originated activity for K consecutive periods after a decline | Secondary target; catches functional departure without formal closure |
| **L3** | **Balance-derived T0** | The brief's §6.1 definition | **Retained as the ops-facing event and as the episode anchor**, not as the model target |

L3 stays because Treasury Management Operations acts on it and because
`lead_gap` is measured against it. It is demoted from target to anchor.

**Report all three and their overlap.** The confusion matrix between L1, L2 and
L3 is itself a deliverable — it tells Operations how often a balance decline
actually ends in departure, which nobody currently knows.

### 3.1 What is still missing on the label

Status flags exist in the deposit data but the mechanism for attaching them is
unresolved (your answer 9). Three things must be pinned before L1 is usable:

- **Is status point-in-time or current-state?** A current-state flag applied to
  history is retroactive leakage of the worst kind — the model learns the answer.
  If only current-state exists, the closure *date* must come from somewhere else
  (last activity, last non-zero balance) and that derivation becomes part of the
  label definition and must be documented as such.
- **Voluntary vs involuntary closure.** Bank-initiated closures (risk exits,
  dormancy sweeps) are a different event and must be excluded or modelled apart.
- **Is dormancy a status or an inference?** If inferred from inactivity it is
  L2 wearing an L1 label, and it must not be counted as independent confirmation.

---

## 4. Episode construction

Carried from the brief §6, with two changes.

**T0** — most recent month at or before the flag where the rolling 3-month
average balance was still ≥ 95% of the client's trailing 12-month **median**.
Median, not mean; the brief is right and the reason is in its own §6.2.

**`lead_gap = flag_month − T0`.** Distribution is the first deliverable and
stands alone: it quantifies, in Operations' own data, how much warning the
current 30% rule discards.

**Change 1 — exclusions run in a fixed order and each is counted.** The funnel
in §1.3 is an output artefact, not a preamble. Reviewers will ask how a book of
N became n episodes.

**Change 2 — `n_accounts_closed` is carried alongside, per §1.1**, so a
consolidation-driven step-down is separable from a genuine drain.

### 4.1 Controls

3–5 per case, matched on NAICS2, payment-volume size decile, tenure band and
calendar month. Unchanged from the brief and correct — without it the study
learns that large stable clients do not leave.

---

## 5. Feature blocks

Reorganised from the brief §7 by **what the new substrate makes computable**,
which is a different ordering than the brief's expected-value ordering.

### Block A — Money leaving the franchise *(now fully computable; upgraded)*

| Feature | Definition | Note vs brief |
|---|---|---|
| `off_us_outflow_share` | outflow to non-PNC RTN / total outflow | Denominator now **real**, not on-us-only |
| `off_us_outflow_share_d3` | 3-month change | — |
| `n_new_ext_rtn` | Distinct destination RTNs first seen this period | **New.** Not possible without the RTN |
| `new_rtn_amount_share` | Share of outflow to first-seen RTNs | **New.** The strongest form of the switch signal |
| `dest_fi_hhi` | Concentration of off-us outflow across destination FIs | **New** |
| `self_pay_new_flag` | New same-name destination at a non-PNC RTN | Replaces `same_name_outflow_flag` — **delta, not level** (§2.4) |
| `self_pay_amount_share` | Share of outflow to self-matched external accounts | Level; report base rate alongside |
| `fi_destination_flag` | Deterministic RTN classification | Was fuzzy; now exact |
| `large_single_outflow_flag` | Largest single outflow > 3× median monthly, non-recurring | Unchanged |
| `net_flow_ratio` | (inflow − outflow) / inflow | Unchanged |
| `returned_volume_share` | Returns / total outflow | **New** (§2.3) |

### Block B — Relationships breaking *(depends on edge rhythm)*

`n_recurring_missed` is a `FIXED_RECURRING` or `VARIABLE_RECURRING` edge that
did not fire in the as-of window. The taxonomy is already specified in
`pkg_edge_rhythm_eda.ipynb` — six classes, 6-month minimum lookback — and is
**awaiting the EDA run results to set the modal amount band width**. Attrition
should be the first production consumer of that module.

| Feature | Definition |
|---|---|
| `n_recurring_missed` | Established-cadence pairs with no payment in the expected window |
| `missed_recurring_amount_share` | Prior-period value of missed / prior total outflow |
| `payroll_active_flag`, `payroll_amount_mom` | Payroll processor relationship live; MoM change |
| `lost_relationship_avg_tenure` | Mean months-active of relationships lost |
| `n_lost_newest_tercile` | Lost relationships from the shortest-tenure third |

**Payroll as signal hub.** The brief asks for payroll processors to be exempted
from hub exclusion. Agreed, and it needs no new machinery — this is a class in
the graded hub registry, the same pattern as the locatability hub classes.
Blanket exclusion is already recorded as creating blind spots.

**Tenure direction is a real hypothesis and should be tested as one.** Losing
newest-first suggests a planned move (lowest switching cost moves first);
losing oldest-first suggests distress. If it does not separate, say so.

### Block C — Moving or shrinking *(on-us only — read the caveat)*

`lost_cpty_health_index`: of the counterparties this client stopped paying, the
share whose own total inbound **from other PNC customers** held steady (±10%).

**The off-us asymmetry does not dissolve here.** Deposit-anchored extraction
fixes coverage of the *ego's* behaviour; it does nothing for the counterparty's.
A counterparty that reads as shrinking may simply have moved its own banking
off-us. Therefore:

- Report conditional on the counterparty's own on-us degree, **never pooled**.
- Counterparties below a degree floor return `UNKNOWN`, not a health score.
- `UNKNOWN` is a real answer and must not be backfilled with the population mean.

Same principle as `NO_EVIDENCE` in `locate()`.

### Block D — Balance and flow together

`balance_to_flow_ratio`, `balance_flow_divergence`. Both fine **only if** §8 Q8
resolves that deposits and transactions are separate ledgers. If balance change
is simply the sum of these transactions, these two are algebraic restatements,
not independent evidence, and the block collapses (§6.3).

### Block E — Behaviour and cadence

`payment_cadence_cv`, `payment_cadence_cv_d3`, `origination_count`,
`origination_count_mom`, `inflow_outflow_decline_gap`.

The receivables-before-payables asymmetry is the interesting one and is now
**testable as a sequence** at Tier 2 daily grain rather than inferred from
monthly deltas.

### Block F — Geography

`cpty_spread_mom`, `cpty_spread_contraction_flag`. Retained, **demoted**: not
independent of Block B (§0 correction), and cheap only because the geo block
exists. Low expected marginal value; run last.

---

## 6. Observation frame and leakage discipline

```
Unit        : customer-month (T1) / customer-week (T2)
Target      : L1 primary, L2 secondary, L3 anchor only
Features    : computed only from data available as of t
Blackout    : nothing at or after t+1 enters features, ever
Horizons    : H = 3 and H = 6, run separately
Excluded    : months T0 → trough (neither clean positive nor clean negative)
Re-entry    : recovery = balance ≥90% of prior level for 3 months
Freeze      : out-of-time holdout, no refit, no threshold tuning on it
```

### 6.1 The deposit-only benchmark is the ceiling, not a baseline

Do not re-derive detection rules from scratch. The existing deposit-only
classifier reaches ≈0.905 freeze AUC and defines the ceiling. The brief's Step 2
should be reframed:

> Not *"is there a better rule than the 30% rule"* — there is —
> but *"how much of 0.905 does a rule an operations team can read recover, and
> at what detection lag."*

Candidates: MoM persistence over 2–3 months, CUSUM on level change, 15%-over-3-
months, deviation from trailing-12 median. Report detection lag **and** false
positive rate for each, against the current rule and against the model.

### 6.2 Settlement lag is a leakage surface

The transaction date may be the settlement date, delayed relative to the real
transaction. At monthly grain this mostly rounds away. At daily grain it does
not, and it breaks as-of correctness in a specific way: **a feature computed
"as of day t" using rows dated ≤ t is not the same as a feature computable on
day t**, because rows for activity before t had not settled yet.

Required:

- Emit `settlement_lag_days` distribution by rail (§8, Q4).
- Define as-of features with an explicit **lag buffer** — features at t use
  rows settled ≤ t, and the buffer is the rail-specific p95 lag.
- Any Tier 2 sequence claim (receivables-before-payables) must be robust to the
  lag distribution, or it is measuring settlement mechanics rather than
  treasury behaviour. **This is the single most likely way to produce a
  convincing false finding in this study.**

### 6.3 If it is one ledger

If balance movement is the sum of these transactions, payment and deposit
features are two views of one table. Amount and timing are shared;
**counterparty identity is the genuinely new axis.** That is still a real
contribution, and it sharpens the hypothesis usefully: features keyed on *who*
(Blocks A, B, C) can add lift; features keyed on *how much* (Block D, parts of
E) cannot, by construction. Resolve Q8 before writing Block D.

---

## 7. Standing traps

| Trap | Rule |
|---|---|
| **Scope conflation** | `deposit_anchored_ego_d1` metrics never land in `on_us_c2c` columns. Both are observations; only the population differs, which makes this harder to catch than observed/inferred |
| **Settlement lag** | §6.2. Lag buffer on every as-of feature; rail-specific |
| **Same-name base rate** | Level is not signal. Only the delta. Measure the base rate first |
| **Name coverage is non-random by rail** | Wire beneficiary good, ACH company name workable, check and card mostly nothing. Rail mix correlates with industry and size — so name-based features are confounded with segment. Report coverage by rail **and by dollar** |
| **Correspondent / sponsor RTN** | §2.3. Do not name a competitor from a masked RTN |
| **Account roll-up** | §1.1. Level and closure event stay in separate columns |
| **Current-state status flag** | §3.1. A retroactively applied closure flag is direct leakage of the target |
| **Join dtype** | Account/routing keys are strings end to end. The 2026-07 int64/str incident typed every counterparty `unknown` and produced plausible output |
| **Returns** | Net before aggregating, keep as a feature |
| **Two labels, one claim** | L2 (silence) and an inferred dormancy flag are the same evidence. Never count as independent confirmation |
| **Multiple comparisons** | ~25 features × 2 horizons × cases-vs-controls. Pre-register the primary hypotheses; treat the rest as exploratory and label them so |

---

## 8. What is missing — profiling queries before any module is written

Each of these is a query, not an analysis. Together they take a day and they
determine which of §5 is real.

| # | Question | Blocks | Output |
|---|---|---|---|
| **Q1** | Staging table row count, date range, distinct customers, distinct external accounts | Everything | `qa/staging_profile.csv` |
| **Q2** | Rail mix — share of rows and of dollars by rail | A, name features | `qa/rail_mix.csv` |
| **Q3** | **Counterparty name population rate, by rail and by dollar** | A, self-pay | `qa/name_coverage_by_rail.csv` |
| **Q4** | **Settlement lag distribution by rail** (p50/p90/p95, and share where transaction date ≠ initiation date) | §6.2, all Tier 2 | `qa/settlement_lag.csv` |
| **Q5** | RTN population rate; distinct RTNs; top-50 RTNs by volume and by dollar | A, §2.2 | `qa/rtn_profile.csv` |
| **Q6** | Share of RTNs matching the FRB directory; unmatched top-50 | §2.2 | `qa/rtn_registry_match.csv` |
| **Q7** | Account-number format variation by rail (length, leading zeros, masking) | §2.1 key stability | `qa/acct_format.csv` |
| **Q8** | **Do transaction sums reconcile to deposit balance movement?** Per customer-month, on a sample | Block D, §6.3 | `qa/ledger_reconciliation.csv` |
| **Q9** | Deposit account inventory — types present, accounts per customer, closure events per year | §1.1, §3 | `qa/deposit_account_profile.csv` |
| **Q10** | **Same-name outflow base rate** — share of corporate customers with ≥1 exact-name external destination in a normal month | §2.4 | `qa/self_pay_base_rate.csv` |
| **Q11** | On-us same-`mdm_id` cross-account transfer count — the matcher calibration set size | §2.4 | `qa/self_pay_calibration_set.csv` |
| **Q12** | Account status field: values present, point-in-time vs current-state, closure date availability, voluntary/involuntary distinction | §3.1 | `qa/account_status_profile.csv` |
| **Q13** | **The §1.3 funnel, counted** | Go/no-go on Steps 3–7 | `qa/episode_funnel.csv` |

**Q3, Q4, Q8, Q12 and Q13 are the blocking five.** Q3 decides whether name-based
features exist. Q4 decides whether Tier 2 is trustworthy. Q8 decides whether
Block D is real. Q12 decides whether L1 exists. Q13 decides whether the study
is powered.

---

## 9. Modules to build

None of these exist yet. Ordered by dependency.

| Module | Purpose | Blocked on |
|---|---|---|
| `pkg_attr_profile.py` | The §8 queries; writes every `qa/` artefact above | Table access |
| `pkg_rtn_registry.py` | FRB directory + FDIC/NCUA join; RTN → FI identity, type, geography; correspondent and sponsor flags. **Shared with the FI Pinning Registry** | Q5, Q6 |
| `pkg_attr_extract.py` | Tier 1 aggregation to `(cust, ext_acct_key, month)`; Tier 2 event retention for the cohort. Hashing, normalisation, return netting | Q1–Q7 |
| `pkg_attr_selfpay.py` | Exact-match self-payment v1 + on-us calibration harness | Q10, Q11 |
| `pkg_attr_episodes.py` | T0 back-tracing, exclusion funnel, `lead_gap`, control matching, L1/L2/L3 labels and their overlap | Q9, Q12, Q13 |
| `pkg_attr_features.py` | Blocks A–F against the observation frame; lag buffers | All above; Block B needs `pkg_edge_rhythm.py` |
| `pkg_attr_eval.py` | Separation curves, lead-time table, discriminator split, scorecard, precision@K, freeze evaluation | Features |

**Two dependencies outside this manifest:**

- `pkg_edge_rhythm.py` — not yet written; blocked on your EDA run results
  (modal amount band width). Block B cannot be built without it, and building
  a parallel cadence implementation here would be a duplicate that drifts.
- The FI Pinning Registry — currently scoped only for locatability. Building it
  inside `pkg_rtn_registry.py` and having locatability consume it is the right
  order, since this study needs it sooner.

---

## 10. Sequencing

1. **Run §8.** All thirteen queries. Nothing else starts.
2. **Episode table + `lead_gap`.** Deliverable to Operations regardless of what
   follows; no payment data needed.
3. **Step 2 reframed** (§6.1) — deposit rules benchmarked against the existing
   model, not derived from scratch.
4. **Tier 1 extract + RTN registry.** The reusable asset.
5. **Block A separation curves.** Cases vs controls, T0−6 … T0. If Block A does
   not separate with a real off-us denominator and an exact FI flag, the
   payment-signal hypothesis is dead and that is a publishable answer in two
   weeks rather than a quarter.
6. **Block B**, once edge rhythm lands.
7. **Block C discriminator.** Arguably more valuable than prediction — moving
   versus shrinking is the question only the graph can answer.
8. Scorecard, then a model only if the scorecard shows lift. **Not a neural
   network** — the brief's reasoning on sample size, explainability to a
   relationship manager, and the model governance path is correct.

---

## 11. Governance — flag now, in parallel

- **Permissible use.** Inferring that an external account belongs to a
  customer's banking relationship at another institution is a sharper version
  of the read already flagged as a precondition for Prospect Radar. It should
  go in as one request, not two.
- **Account-number handling.** Hashing at extraction, salt in a restricted
  location, raw numbers never in `../metrics/`. Confirm this satisfies the
  data-handling standard for the staging table's classification.
- **Retention outreach on inferred competitor movement** needs a fair-lending
  view if the queue is routed on anything geography- or segment-derived — the
  same view flagged for locatability.

---

## 12. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08 | Initial. Reworks `PKN_Attrition_Analysis_Brief.md` against the Neo4j staging table: deposit-anchored extraction, RTN/account counterparty key, status-based label, two-tier grain, settlement-lag discipline. Corrects the brief's leakage attribution, Group F independence claim, and name-matching design |

---

*Internal — PNC Treasury Management, Data Science*
