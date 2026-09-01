# Response to the Attrition Analysis Brief

**Payment Knowledge Graph (PKG) — findings from source profiling**
*PNC Bank · Treasury Management · Data Science*
*v1 — August 2026*

> Measurement basis: three months (2025-04 → 2025-06) of the Neo4j staging
> transaction table, filtered to the 66,947 corporate customers in the deposit
> book. 305M transaction legs, $4.34T, 44.1M distinct counterparties. The
> deposit panel covers 2024-01 → 2026-06 at monthly average balance, one row per
> `cust_pwr_id`. Full outputs in `../eda/attrition/`.

---

## Summary

1. **The current 30% rule flags 39% of the corporate book.** Measured precision
   against departure is ~11%. This is the finding that should drive the next
   quarter, and it required none of §7.
2. **Counterparty data is substantially better than §5.2 assumed.** 97.5% of
   off-us dollars carry a counterparty key, a name, and the counterparty's bank.
   Group A is fully computable today.
3. **Coverage is 65–81%, not 12%.** The precondition the prior study attached to
   its verdict is now met, and a rerun is warranted.
4. **Group C does not survive contact with the data.** 91% of counterparties are
   seen by exactly one PNC customer.
5. **A metric nobody proposed is better than the brief's priority signal:** the
   share of a customer's ACH originated through a non-PNC bank. $499B, 14.3% of
   off-us dollars, no name matching required.

---

## 1. Corrections to the brief's premises

### 1.1 §6.4's account of the first attempt is not accurate

The brief attributes the prior study's result — deposit momentum dominating — to
leakage. That is not what happened. The prior pipeline ran an out-of-time freeze
with a deliberate window mismatch specifically so that leakage was impossible.
Deposit-only reached 0.905 freeze AUC and the graph added no incremental lift.

The cause was **coverage** (~12% of the deposit book was graph-visible) and the
**label** (a deposit-derived target predicted by deposit-derived features).

This matters operationally. If the leakage account stands, the team concludes the
signal was always there and only the harness was sloppy, and re-runs a clean
version of an experiment that was already clean.

### 1.2 §5.2 understates what is in hand

The brief treats counterparty name coverage as an open item to confirm. Measured:

| Identifiability class | Legs | Share of legs | Share of dollars |
|---|---:|---:|---:|
| key + name + bank | 256,143,390 | 89.4% | **97.5%** |
| key only | 15,671,218 | 5.5% | 1.4% |
| name only | 14,566,224 | 5.1% | 1.1% |
| key + name | 21,060 | 0.0% | 0.0% |

`unq_cpty_acct_id` resolves as `RTN-account` (confirmed by format inspection),
and **96.3% of dollars are account-derived** rather than name-derived. These are
stable nodes: the same counterparty account keys identically every month, so
appearance and disappearance are real events rather than spelling changes.
Relationship-dissolution metrics do not need a stability caveat.

`cpty_fin_entity_name` is populated and eliminates §12's FI-name open item
entirely. `fi_destination_flag` is a lookup.

**Where coverage is thin, it is thin by rail, not at random.** Outbound checks
(7.6M legs, $35.9B) carry a name but no key. Card rails carry neither. Both are
small in dollars; both should be excluded explicitly rather than silently.

### 1.3 Coverage of the deposit book

| | Customers | Share of book |
|---|---:|---:|
| Deposit book (corporate) | 66,947 | 100% |
| Any transaction visible | 54,075 | **80.8%** |
| Off-us outbound visible | 43,479 | **65.0%** |
| *(prior on-us C2C graph)* | — | *~12%* |

Dollar visibility moves further than customer coverage does. The on-us slice the
prior study was confined to is 6% of legs and **19.4% of dollars**. The
deposit-anchored extraction sees all of it.

---

## 2. The headline: the current rule has ~11% precision

Applying the 30% rule across the 30-month panel, then §6.2's own exclusions:

| Funnel step | Customers | Share of book |
|---|---:|---:|
| Raw 30% rule | 47,942 | 71.6% |
| + 12 months of history | 42,394 | 63.3% |
| + $25K balance floor | 33,603 | 50.2% |
| + 3-month persistence | **26,139** | **39.0%** |

Cross-tabbed against customers whose balance series stops and does not resume —
the closest available proxy for an actual departure:

| | Real stop | No stop | Total |
|---|---:|---:|---:|
| **Rule fires** | 2,962 | 23,177 | 26,139 |
| **Rule silent** | 1,869 | 38,939 | 40,808 |
| **Total** | 4,831 | 62,116 | 66,947 |

**Precision 11.3% · Recall 61.3%.**

`series_stopped` is a proxy and is itself affected by the panel issue in §5, so
the precision figure will move. **The order of magnitude will not.** A rule that
fires on 39% of the corporate book over 30 months cannot be a departure detector
regardless of which label it is scored against — the base rate of departure is
nowhere near 39%. Nine of ten flags reaching TM Ops today are wrong, and the
alert-fatigue cost of that is a present operational problem rather than a
modelling one.

**This is shippable now.** It needs no counterparty data, no graph, and no model.

---

## 3. Consequences for §7's metric groups

| Group | Status | Basis |
|---|---|---|
| **A — Switching** | **Live.** Fully computable today | 97.5% dollar coverage; `cpty_fin_entity_name` gives destination bank directly |
| **B — Relationship breaking** | **Live and trustworthy** | 96.3% account-derived keys; stable nodes across months |
| **C — Counterparty health** | **Re-scope or drop** | 91% of counterparties seen by one customer; only **0.2%** by five or more |
| **D — Flow/balance** | Proceed, with §5 caveat | |
| **E — Payroll** | Proceed | ADP, Paychex clearly visible in the hub seed |
| **F — Geography** | Proceed, **not independent** | Computed over the same counterparty set as Group B; correlated by construction |

**Group A's priority signal needs restating as a delta.** Same-name outflow has a
base rate of **7.8% of customers per month** (11% of dollars), flat across all
three months. As a level, the flag fires on ~3,100 customers every month
permanently — most corporates simply maintain multiple bank relationships. The
signal is a *new* same-name destination or a step change in share.

One measurement note for whoever builds it: `TREASURY IRS EFTPS RECV` and
`US TREASURY SINGLE TAXPAYORS` appear among same-name destinations, which means
the counterparty name field sometimes carries the payer's own name on tax
payments. That false-positive mechanism has to be excluded before the base rate
is meaningful.

**Power is not a constraint.** 20,291 episodes survive the full funnel with
off-us outbound visibility — twenty times the threshold at which the brief's
matched-control design becomes runnable.

---

## 4. New: ACH origination wallet share

Not in the brief, and stronger than what is.

The `category` field carries an ACH origination taxonomy:
`ACH_OrigViaPNC_wTPO` / `_woTPO` versus **`ACH_OrigViaNONPNC`**. The last of these
is **18.4M legs and $499B — 14.3% of off-us dollars** — and maps exactly onto
`cpty_type = NON_PNC_ACH_ORIGINATOR`.

A customer whose ACH origination shifts from PNC to another institution is losing
PNC treasury services directly. That is a more precise statement of "leaving" than
a balance decline or a same-name transfer, and it is:

- computable today, no name matching, no fuzzy resolution
- a direct wallet-share ratio rather than a proxy
- interpretable to a TM relationship manager without explanation

Proposed metric: `off_pnc_ach_orig_share` per customer per month, plus its
three-month delta. This should be built before anything in Group A.

---

## 5. The open risk: is the deposit panel survivorship-biased?

Two patterns in the panel point the same way:

- Non-null share climbs monotonically from **76.0%** (2024-01) to **100%**
  (2025-10 onward) and stays there.
- **All 4,831 balance-series stops fall in 2025-10 → 2026-05**, at a flat
  ~600/month, and none before.

The straightforward reading is that the table is the *current* book with balances
backfilled — customers who left before roughly late 2025 are not in it. If that
is right, the label is not merely circular; it is **censored**, and historical
episode analysis before 2025-10 is measuring survivors only.

Median balance also falls from $122,846 to $81,783 across the window. Part of
that is real and part is the composition change from backfilled onboarding; the
two cannot be separated until the question above is answered.

**This now outranks the monthly-closure request.** It should be resolved before
Steps 3–6 begin.

*(Related, minor: the table has 66,947 rows against 66,733 distinct
`cust_pwr_id` — 214 duplicates to resolve.)*

---

## 6. Revised sequencing

1. **Ship the precision finding to TM Ops.** Done; needs writing up, not
   analysis.
2. **Resolve the panel survivorship question** with the deposit team. Blocking.
3. **Build `off_pnc_ach_orig_share`** and its delta. Highest ratio of signal to
   effort in the whole programme.
4. **Widen the extraction to the full 30 months.** Measured at 176 seconds for
   three months; roughly 30 minutes for thirty.
5. **Steps 1–2 of the brief** — episode table, `lead_gap`, and a rule comparison
   benchmarked against the existing deposit-only model as the ceiling rather than
   re-derived from scratch.
6. **Group B via the edge rhythm module**, whose taxonomy is already specified.
7. **Group A as deltas**, after the same-name false-positive mechanism is
   excluded.
8. **Group C** only if the 0.2% shared-counterparty subset is worth a separate
   study. Otherwise drop it.

---

## 7. Data requests

| Request | Owner | Why |
|---|---|---|
| Confirm whether the deposit panel is current-book-only | Deposit team | §5 — determines whether historical episodes are censored |
| Monthly closed-account count, or per-account open/close dates | Deposit team | Converts the target from a balance-derived event to an actual departure |
| Resolve 214 duplicate `cust_pwr_id` rows | Deposit team | Grain |

---

## Appendix — one number that needs interpreting, not reporting

Correlation between monthly balance delta and net transaction flow is **0.04**.
Read naively this says deposits and payments are two independent systems.

It is more likely a measurement artefact. Balance is a monthly *average*, so its
delta is a smoothed function of within-month flow; and for a business,
gross-in ≈ gross-out, making net flow a small difference of two very large
numbers where noise dominates. Supporting this: only **6.6%** of customer-months
show balance movement above $1,000 with zero transaction legs, which suggests the
data mostly *is* there.

Re-test on gross flows and log-scaled balance before drawing any conclusion. Do
not brief the 0.04.

---

*Internal — PNC Treasury Management, Data Science*
