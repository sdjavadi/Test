# PKG Attrition EDA — Output Interpretation Manifest

**Companion to `pkg_attrition_source_eda_v4.ipynb`**
*PNC Bank · Treasury Management · Data Science*
*v1 — September 2026*

---

## 0. How to use this document

This manifest is **self-contained**. It is written so that an analyst or AI
assistant with **no prior context** can be handed a folder of result CSVs plus
this file and interpret them correctly.

If you are an AI assistant reading this:

- §1–§3 give you the context you need. Read them before looking at any numbers.
- §4 is the output catalogue. For each file: the question it answers, what the
  columns mean, and the thresholds that make a number good or bad.
- **§5 is mandatory.** It lists artefacts that produce plausible-looking but
  wrong numbers. Several results in this pipeline have already been
  misinterpreted; §5 records how.
- Where a threshold is given, apply it. Where a number is flagged as
  "pessimistic" or "artefact", say so rather than reporting it at face value.
- Do not infer causality from any correlation in this pipeline. Every finding
  here is descriptive.

Outputs live in `../eda/attrition/` as CSVs named for their stage
(`A4_monthly_volume.csv`, `E5_rule_scores.csv`, …).

---

## 1. What this project is

PNC Treasury Management wants to detect **corporate deposit attrition** — clients
withdrawing balances and moving their banking relationship elsewhere — earlier
and more precisely than the incumbent rule allows.

**The incumbent rule:** flag a customer when their trailing 3-month average
balance falls ≥30% below the prior 6-month average. It is believed to be
imprecise. Quantifying that is one of this pipeline's two main jobs.

**The hypothesis under test:** payment-network structure (who a customer pays,
who pays them, through which bank) contains a departure signal that balance
movements alone do not. A prior study found no incremental value from payment
data, but that study saw only ~12% of the deposit book. This pipeline rebuilds
the extraction to raise coverage and re-tests.

**Two data sources:**

| Source | Grain | Notes |
|---|---|---|
| Staging transaction table | one row per transaction | Feeds a Neo4j payment graph. Very large (~100M rows/month). |
| Deposit panel | one row per customer, months as columns | Monthly **average** balance, 2024-01 → 2026-06. See §5.1 — this table has a serious defect. |

---

## 2. Data model

### 2.1 Ego legs — the central abstraction

The pipeline's one expensive pass converts transactions into **ego legs**. One
leg = one PNC customer's view of one transaction.

```
transaction:  ACME CORP  --$50,000-->  SUPPLIER LLC (external)
  => 1 leg:   ego=ACME, ego_dir=OUT, cpty=SUPPLIER, cpty_key_src=off_us

transaction:  ACME CORP  --$50,000-->  BETA INC (also a PNC customer)
  => 2 legs:  ego=ACME, ego_dir=OUT, cpty=PNC:BETA, cpty_key_src=on_us
              ego=BETA, ego_dir=IN,  cpty=PNC:ACME, cpty_key_src=on_us
```

### 2.2 Direction is inferred from a null pattern

There is **no direction column**. Direction comes from which side carries an MDM
customer id:

| `mdm_id_pays` | `mdm_id_receives` | `topology` | Meaning |
|---|---|---|---|
| present | present | `INTERNAL_C2C` | both sides are PNC customers |
| absent | present | `INBOUND` | external party pays a PNC customer |
| present | absent | `OUTBOUND` | PNC customer pays an external party |
| absent | absent | `ORPHAN` | data defect; should be 0 |

**`OUTBOUND` is the analytically important half.** It is money leaving PNC, and
it was entirely invisible to the prior study.

### 2.3 Counterparty keys

`unq_cpty_acct_id` is built upstream as `RTN-account` (e.g.
`021000021-00000000000454247219`), falling back to the counterparty **name** when
the account is absent. `key_provenance` separates these:

| Value | Meaning | Reliability |
|---|---|---|
| `account_derived` | real account key | **stable node.** Disappearance is a real event |
| `name_derived` | key is a name string | unstable. A spelling change looks like a lost relationship |
| `on_us` | counterparty is a PNC customer (`PNC:` prefixed) | stable |
| `none` | no key at all | excluded from counterparty metrics |

### 2.4 Key fields

| Field | Meaning |
|---|---|
| `cust_pwr_id` | customer identifier; joins transactions to deposits. 1:1 with `mdm_id` |
| `cpty_key_src` | `on_us` (PNC↔PNC) vs `off_us` (external). **Never merge these** |
| `cpty_fin_entity_name` | the counterparty's **bank**, by name. Enables competitor analysis directly |
| `cpty_type` | `CPTY` / `NON_PNC_ACH_ORIGINATOR` / `MERCHANT` / `P2P_CPTY` |
| `rail` | `WIRE` / `ACH` / `CHECK` / `RTP_PRT` / `PCARD` / `DEBIT_CARD_SIGNATURE` / `RTP_P2P` |

---

## 3. Established facts — do not re-derive

Measured on three months (2025-04 → 2025-06), 66,947 corporate customers, 305M
legs, $4.34T. Stable and not expected to change with a wider window.

| Fact | Value |
|---|---|
| Counterparty identifiability | **97.5% of off-us dollars** carry key + name + bank |
| Key provenance | **96.3% of dollars** account-derived → stable nodes |
| Topology mix (legs) | OUT 52.2% · IN 41.8% · INTERNAL 6.0% · ORPHAN 0 |
| Topology mix (dollars) | OUT 40.3% · IN 40.3% · INTERNAL 19.4% |
| Rail mix | WIRE 0.5% of legs but **58.2% of dollars**; ACH 86.3% of legs, 39.3% of dollars |
| Coverage of deposit book | 80.8% any transaction · 65.0% off-us outbound *(prior study: ~12%)* |
| Counterparty fan-in | **91%** of counterparties seen by exactly **one** customer; only **0.2%** by five or more |
| Non-USD | ~0.06% of dollars. Ignore |
| `legs_per_txn` | ~1.0 — no double-booking |

**Consequence of fan-in:** any metric requiring a counterparty's behaviour to be
observed across multiple PNC customers is undefined for 99.8% of counterparties.

**Consequence of rail mix:** every dollar threshold must be rail-relative. Median
transaction is $27,487 on WIRE and $65 on PCARD.

---

## 4. Output catalogue

### PHASE A — extraction

#### `A2_mdm_per_pwr_id`
**Q:** Is `cust_pwr_id` ↔ `mdm_id` one-to-one?
**Columns:** `n_mdm` (MDM ids per customer), `count`.
**Read:** All mass should sit at `n_mdm = 1`. **Any row with `n_mdm > 1` is a
blocking problem** — one deposit "customer" is several graph nodes, and every
per-customer aggregate is wrong by an unknown factor. Report it prominently.

#### `A4_monthly_volume`
**Q:** Did the extraction cover the intended window, and is volume stable?
**Columns:** `month`, `n_legs`, `n_egos`, `dollars`.
**Read:** **Check this first, before anything else.** Confirm the month list
matches the configured range. A ramp at either end is an ingestion artefact, not
a business trend. `n_egos` should be roughly flat (~49,000/month).
**Trap:** see §5.2 — a stale-cache bug has produced a 3-month table when 30 were
requested. If the month count is wrong, **every Phase C/D/E output is void.**

### PHASE B — taxonomy *(usually skipped; see §3)*

`B_topology_mix`, `B_rail`, `B_category`, `B_cpty_type`, `B_key_provenance`,
`B_top_financial_entities`, `B_top_counterparty_names`. Descriptive profiling,
already settled. `B_top_financial_entities` is useful on its own as a
**competitive map**: destination banks ranked by customer count and dollars.

### PHASE C — derived tables

#### `C3_cpty_fanin`
**Q:** How many counterparties are shared across PNC customers?
**Columns:** `n_counterparties`, `share_2plus`, `share_5plus`, `max_customers`.
**Read:** `share_5plus` bounds any counterparty-health metric. Measured at
**0.0020**. At that level, metrics requiring cross-customer observation of a
counterparty should be **dropped or re-scoped to the 0.2% subset**.

#### `C3_outbound_fanout`
**Q:** How many counterparties does a customer pay per month?
**Read:** Extremely skewed — median 5, mean 359, max ~2.5M. Use medians and
percentiles; **never report the mean.**

### PHASE D — deposit panel

#### `D1_duplicate_customers`
**Q:** Does the deposit table have one row per customer?
**Read:** `n_dup_customers = 0` is the correct and observed answer. See §5.4 —
an earlier apparent discrepancy was a measurement artefact, not a data problem.

#### `D1_deposit_shape`
**Q:** What is the panel's shape and as-of date?
**Columns:** `n_customers`, `n_relationships`, `n_distinct_latest_month`,
`max_latest_month`, `mean_accounts`, `share_any_closed`.
**Read:** **`n_distinct_latest_month = 1` is a positive survivorship finding** —
it means the table is a single snapshot as of `max_latest_month`, built from the
book as it existed on that date. See §5.1.

#### `D2_series_shapes`
**Q:** How many customers' balance series stop before the panel ends?
**Columns:** `shape` ∈ {`live_at_end`, `STOPPED_BEFORE_END`, `never_live`}, `count`.
**Read:** `STOPPED_BEFORE_END` is the **best available proxy for departure** — but
only inside the observable window (§5.1). Measured: 4,831 of 66,947 (7.2%).

#### `D2b_starts_and_stops` — **read this before any historical claim**
**Q:** Are departures observable across the whole panel, or only recently?
**Columns:** `month`, `n_start` (series beginning), `n_stop` (series ending).
**Read:** In an unbiased panel, stops occur throughout. Observed:

- 2024-01 → 2025-09: **zero stops in twenty-one consecutive months**
- 2025-10 onward: 533–699 stops per month
- `n_start` runs 650–940/month through 2025-09, then collapses to 285 → 105 → 89 → 40 → 29 → 0

Customers stop appearing and start disappearing in the **same month**. No real
book behaves that way. This is a data-maintenance cutover, not customer behaviour.

#### `D2b_stop_concentration`
**Columns:** `n_stops`, `share_in_last_8mo`.
**Read:** **`share_in_last_8mo = 1.0000` is a confirmed positive.** Every
departure in the panel falls in the final eight months. Departures before the
cutover were removed from the table.

#### `D2b_month_coverage`
**Columns:** `month`, `share_non_null`, `share_positive`, `p50_balance`, `n`.
**Read:** `share_non_null` climbs monotonically 76.0% → 100% at 2025-10 and never
falls. A live book loses customers, so coverage should wobble rather than
saturate — this is the signature of a backfilled current-book extract.
**Do not interpret the declining `p50_balance` (≈$123K → ≈$82K) as a business
trend.** It is confounded with the composition change.

#### `D3_coverage`
**Q:** What share of the deposit book is visible in payment data?
**Columns:** `n_deposit_customers`, `n_any_txn`, `n_off_us_outbound`,
`coverage_any`, `coverage_outbound`.
**Read:** This replaces the ~12% figure that made the prior study's verdict
conditional. `coverage_outbound` is the population on which switching metrics can
be tested. **Only valid if `A4_monthly_volume` covers the full window** — with a
narrow transaction window, customers who were simply quiet read as invisible.

#### `D4_flow_vs_balance`
**Q:** Do transactions explain balance movement — i.e. are deposits and payments
one ledger or two?
**Columns:** `c1_delta_vs_netflow`, `c2_level_vs_grossout`,
`c3_logbal_vs_loggrossout`, `c4_logbal_vs_loggrossin`.
**Read:** **`c3`/`c4` are the honest test. Do not report `c1`.** For a business
gross-in ≈ gross-out, so net flow is a small difference of two very large numbers
and noise dominates `c1`. High `c3`/`c4` → one ledger, meaning the only genuinely
new information payments contribute is **counterparty identity**, not amount or
timing. Low → two systems.
**Caveat:** balance is a monthly *average*, so its delta is a smoothed function of
within-month flow. This caps all four correlations regardless. A moderate value
is not evidence of missing data.

#### `D4_unexplained_movement`
**Columns:** `bucket`, `n`, `p50_abs_delta`.
**Read:** `NO_TXN_BUT_BALANCE_MOVED` is money movement the transaction table does
not carry — a ceiling on anything transaction-derived. **This is only meaningful
when the transaction window matches the deposit window.** If the transaction
window is shorter, this bucket is inflated by months that simply were not
extracted, and the number is meaningless (see §5.2).

### PHASE E — metrics, labels, power

#### `E1_ach_orig_share_by_month` — the candidate replacement metric
**Q:** What share of a customer's outbound ACH is originated through a bank other
than PNC?
**Columns:** `month`, `n_customers`, `mean_share`, `p50`, `p90`,
`share_any_nonpnc`, `dollar_weighted`.
**Why it matters:** ACH origination is a treasury service PNC either provides or
loses. A customer shifting origination to another bank is losing PNC services
directly — closer to "leaving" than a balance decline. It needs no name matching,
and it is a true wallet-share ratio. Roughly 20% of outbound ACH dollars are
non-PNC originated.
**NULL, not zero,** where a customer originated no outbound ACH — zero would
falsely read as "all with PNC".

#### `E1_ach_orig_delta` — **the number that decides the programme**
**Columns:** `p50_d3`, `p90_d3`, `p99_d3`, `share_shift_gt_10pt`,
`share_shift_gt_25pt`.
**Read:** `share_shift_gt_25pt` is the candidate alert rate — the share of
customers whose non-PNC origination share rose more than 25 points over three
months. **Compare directly against the incumbent rule's 39% flag rate**
(`E5_rule_scores.flag_rate`). If this lands at 2–5%, it is a far tighter signal
and the argument for replacing the rule is quantitative.

#### `E2_same_name_base_rate`
**Q:** How often do customers send money to an entity sharing their own name
(i.e. an account they hold at another bank)?
**Columns:** `month`, `n_same_name`, `n_total`, `dollar_share`, `base_rate`.
**Read:** Measured at **6.6% of customers per month, flat**. Because most
corporates permanently maintain multiple bank relationships, **the level is not a
signal** — a flag on it fires on ~6.6% of the book every month, forever. Use the
delta instead.
*Note: an earlier measurement of 7.8% included tax payments where the
counterparty name field carries the payer's own name. Those are now excluded;
they were ~16% of matches.*

#### `E2_new_same_name_destination` — the event version
**Columns:** `month`, `n_customers_new_dest`.
**Read:** Customers for whom a same-name destination appears for the **first
time**. **Ignore the first month in the series** — everything is new by
construction (left censoring). Subsequent months measured at ~1.5–1.7% of
customers, roughly a 4× reduction in alert volume versus the standing base rate.

#### `E3_counterparty_churn`
**Q:** How often do customers lose payment relationships?
**Columns:** `month`, `n_customers`, `mean_lost`, `mean_lost_dollar_share`,
`p90_lost_share`.
**Read:** Establishes the base rate for relationship-dissolution metrics. A
stable series is the null model that any dissolution signal must beat. Restricted
to `account_derived` keys, so a disappearance is a real event.

#### `E4_account_count_agreement`
**Q:** Can transaction activity substitute for the missing monthly
account-closure field?
**Columns:** `corr`, `share_exact`, `mean_txn`, `mean_dep`.
**Read:** Measured corr 0.813, exact agreement 68%. Transaction-derived counts
undercount (1.71 vs 2.36) because not every account transacts. Usable as a dated
closure proxy, with that bias stated.

#### `E5_episode_funnel`
**Columns:** `step`, `n_customers`.
**Read:** The incumbent rule plus its exclusions, cumulative. Observed:
47,942 → 42,394 → 33,603 → **26,139**. The last row is the real flagged
population: **39.0% of the corporate book.**

#### `E5_rule_confusion` / `E5_rule_scores` — the headline
**Q:** How well does the incumbent 30% rule identify departure?
**Columns:** `n_book`, `n_flagged`, `flag_rate`, `n_stops`, `stop_rate`,
`precision`, `recall`, `lift_vs_random`.
**Observed:** flag_rate 0.390 · precision 0.113 · recall 0.613 · **lift 1.57×**.
**Read:** `lift_vs_random` is the fairest single number — it says the rule is
barely better than flagging at random. This is the benchmark every proposed
replacement must beat, not a result in itself.
**IMPORTANT — the precision figure is pessimistic.** See §5.3. Report it with
that caveat attached, and prefer the aligned re-scoring if it is available.

#### `E5_episodes_graph_visible`
**Columns:** `n_episodes`, `n_outbound_visible`, `coverage`.
**Read:** Sample size for any matched-control study. Rough guide: <300
underpowered · 300–1,000 descriptive only · >1,000 fully powered. Observed
~20,291 — power is not a constraint.

#### `E6_label_agreement` / `E6_label_prevalence`
**Q:** How much do the three candidate departure targets disagree?
**Targets:** `balance_30pct` (the incumbent rule — **circular**, since it is
deposit-derived and would be predicted by deposit features) · `series_stopped`
(trailing nulls — dated, not threshold-based) · `txn_silence` (N months with no
transactions — **independent of deposits**, the only non-circular option).
**Read:** `rate_txn_silence` should sit near the complement of
`D3_coverage.coverage_any`. A value near 1.0 means the silence window is broken
(§5.5). `share_silence_unobservable` reports how much of the panel the label
cannot speak to.
**Observed:** 0.390 / 0.072 / 0.192 — genuine disagreement, so the choice of
target is consequential and must be made explicitly.

#### `E7_SUMMARY` / `E7_TIMINGS`
Headline metrics and per-stage wall time. Use `E7_TIMINGS` to project a wider run.

---

## 5. Artefacts and traps — **read before reporting any number**

### 5.1 The deposit panel is survivor-only *(confirmed, not suspected)*

`n_distinct_latest_month = 1` and `share_in_last_8mo = 1.0000` both fired.
Twenty-one consecutive months with zero stops, then ~600/month.

**Consequence:** usable departure history is **nine months (2025-10 → 2026-06)**,
not thirty. Approximately 4,831 observable departures.

**Rules that follow:**
- No episode analysis before the cutover. Everything earlier is survivors only.
- Any recall or prevalence figure computed across the full panel is wrong.
- Transaction history only needs to reach far enough back to give lead time on
  the observable window — roughly 2025-01.

### 5.2 Stale derived tables can silently narrow the window

The notebook skips a build stage if its output parquet **exists**, without
checking whether it matches the current configuration. A widened month range with
`REBUILD = False` therefore reuses the old narrow tables and produces valid-looking
output on the wrong data.

**Detection:** `A4_monthly_volume` shows fewer months than configured, and/or
`legs_per_txn` is `nan`.

**When this happens, these are void:** `D3_coverage` (quiet customers read as
invisible), `D4_unexplained_movement` (months never extracted read as unexplained
movement — inflating the figure severalfold), `E6` prevalence, and all of Phase C.

**Fix:** `REBUILD = True`, or key the working paths on the month range.

### 5.3 The rule's precision is measured on mismatched windows

`E5_rule_scores` compares a flag that can fire in **any of thirty months** against
a label that can only be positive in **eight** (§5.1). A customer flagged in
2024-05 and still present is counted a false positive although no window existed
in which their departure could have been recorded.

**The 0.113 precision is therefore pessimistic.** The flag rate (0.390) and recall
are not affected in the same way.

**Correct approach:** flag in month *m*, label positive if a stop occurs in
*m+1 … m+6*, both restricted to the observable window. Prefer that figure when
available. Where only the unaligned number exists, report it **with this caveat
stated explicitly.**

The qualitative conclusion is robust either way: a rule firing on ~39% of the
corporate book cannot be a departure detector under any label, because departure
base rates are nowhere near 39%.

### 5.4 `approx_count_distinct` has a ±1% error band

An earlier run appeared to show 66,947 rows against 66,733 distinct customers —
"214 duplicates". The exact check returns **zero duplicates**; the gap was inside
the HLL error bar. Treat any small discrepancy between an approximate distinct
count and an exact row count as noise until confirmed exactly.

### 5.5 Silence labels are sensitive to window width

If the transaction window is narrower than the deposit panel, every uncovered
month is trivially "silent" and the label saturates at 1.0. The current build
restricts evaluation to months with full forward transaction coverage and returns
NULL — not 0 — outside that range. An unobserved month is not a quiet one.

### 5.6 Left censoring in first-occurrence metrics

Any "first time X appeared" metric marks everything as new in the first month of
the window. **Always discard the first month** of `E2_new_same_name_destination`
and similar series.

### 5.7 Averaged balances cap all flow/balance correlations

The panel carries monthly *average* balance, not month-end. Its delta is a
smoothed function of within-month flow, so exact reconciliation against
transactions is impossible in principle. Moderate correlations are expected and
are not evidence of missing data.

---

## 6. Open questions

| # | Question | Status |
|---|---|---|
| 1 | Why did the deposit panel's maintenance change at 2025-10, and can pre-cutover departures be recovered? | Escalated to the deposit team. Blocking for historical work |
| 2 | Is there a monthly account closed-count, or per-account open/close dates? | Requested. Would replace the balance-derived label and remove circularity |
| 3 | Does `off_pnc_ach_orig_share` beat the incumbent rule? | Awaiting `E1_ach_orig_delta` on the full window |
| 4 | Are deposits and payments one ledger? | Awaiting `D4` `c3`/`c4` on the full window |
| 5 | Is relationship dissolution measurable? | Awaiting `E3_counterparty_churn` |

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **ego leg** | one PNC customer's view of one transaction |
| **on-us / off-us** | both parties are PNC customers / the counterparty is external |
| **topology** | INBOUND / OUTBOUND / INTERNAL_C2C / ORPHAN, from the null pattern |
| **key provenance** | whether a counterparty key came from an account number (stable) or a name (unstable) |
| **the 30% rule** | the incumbent detector: trailing-3 average ≥30% below prior-6 average |
| **survivor-only** | a panel from which departed customers were removed, making historical departures unobservable |
| **circular label** | a target derived from the same data as the features predicting it |
| **lift vs random** | precision ÷ base rate. 1.0 = no better than chance |
| **left censoring** | a first-occurrence metric marking everything new in the window's first period |
| **ACH origination** | which bank initiates an ACH payment. A treasury service PNC provides or loses |
| **`cust_pwr_id`** | customer identifier joining transactions to deposits |
| **`cpty_fin_entity_name`** | the counterparty's bank, by name |

---

*Internal — PNC Treasury Management, Data Science*
