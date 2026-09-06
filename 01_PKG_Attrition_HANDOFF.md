# PKG Deposit Attrition — HANDOFF (master brief)

**Payment Knowledge Graph (PKG) · PNC Treasury Management · Data Science**
*State after the v6 run. Read this first; the other two files are reference.*

| File | Contains |
|---|---|
| **01_PKG_Attrition_HANDOFF.md** *(this)* | Objective, what is settled, what is open, next steps |
| **02_PKG_Attrition_DATA_AND_CODE.md** | Tables, columns, dtypes, grain, paths, config, engineering traps |
| **03_PKG_Attrition_RESULTS_LOG.md** | Every number from every run, v1 → v6 |

---

## 1. Objective

Determine whether payment behaviour flags a departing treasury client earlier than the
deposit balance does, and whether it can separate *moving to a competitor* from
*business contraction*.

**Answer to the first half: yes, by about 2 months on dense features — and one feature
carries the whole result.** The second half remains unanswered.

---

## 2. The one-paragraph summary

`fin_out_n` — the count of distinct financial institutions a customer pays — is the best
single early-warning signal at **every horizon from 12 months out to 1**, with lift over
the base rate running 8.0× at −12 and 30.1× at −1. Combining the twelve candidate signals
as an unweighted count **never beats it**. The deposit balance separates at rel_m −5;
dense payment features separate at −7, so the defensible lead is **2 months**. The
incumbent 30%-balance rule gives a median of **2 months** warning, never fires for 17% of
attriters and fires *after* the exit for another 19% — so a `fin_out_n` rule is
competitive on timing and far better on precision.

---

## 3. What is settled

### 3.1 Data mechanics (v2)

- **`acct_status` carries two code systems.** Numeric = HOGAN (`src_system_cd = DDA`),
  single letters = sweep (`AGILETICS_SWEEP`, `SWP`). Closed codes are **07 CLOSED** and
  **08 PURGEABLE**, not `'C'`. 08 is post-07 retention (mean 264.7 days since `closed_dt`
  vs 98.9 for 07).
- **Grain is `acct_full_acct_id + sub_product_cd + edw_tda_load_dt`.** Zero duplicate keys
  at that grain versus 3,675,351 at account-day. Sweep legs are not duplicates; account
  balance is the **sum** over legs.
- **Closure label** = first month after the last month with a live leg. Agrees with
  `closed_dt` on 64,102 accounts at a **1-month median gap**. Absorbing at 99.5%.
- **No CDs.** No time-deposit family, `maturity_dt` null throughout. Q4 not applicable.
- **`avg_monthly_bal_1`** is the prior complete month, refreshed at month end. Populated
  on HOGAN legs (1.000), absent on sweep legs (0.000). Do not use it as a within-month
  signal.
- **Payments is the whole bank.** 12.8B rows, 12.4M mdm_ids. The deposit book is 229,363
  accounts / 120,457 customers. Real coverage: **72% of deposit accounts, 80% of deposit
  customers are payment-visible**; study population **94,179**.

### 3.2 Measurement (v5–v6)

- **Difference-in-differences is the right normalisation.**
  `chg = value_t / mean(value over t−6…t−4)`, then `dd = chg / median(chg across the
  customer's frozen peer group)`. `median_dd = 1.000` for every covered feature.
  Needs 6 months of history, not a 12-month event-aligned pre-window, so it survives the
  permanent 2024 floor **and** is computable in production every month.
- **Peer deciles must be frozen** (first 3 observed months). Recomputed monthly, a
  shrinking customer slides down the deciles alongside its own decline.
- **Coverage floor is mandatory.** A `dd` exists only for customers who used the feature
  in both windows. Without a floor, `amt_out_rtp` (2.6% coverage) tops the ordering.

### 3.3 The twelve signals, ranked by best lift (A_full_exit)

| # | Signal | Feature | Separates | Best lift | at | Recall there |
|---|---|---|---|---|---|---|
| 12 | Banks they pay drop away | `fin_out_n` | −5 | **30.1×** | −1 | 0.086 |
| 11 | Relationship list shrinks | `cpty_out_n` | −6 | 23.8× | −1 | 0.123 |
| 9 | What monitoring sees | `bal_live` | −5 | 17.7× | −1 | 0.572 |
| 8 | Inbound activity thins | `n_in` | −7 | 17.0× | −1 | 0.570 |
| 7 | Fewer payments | `n_out` | −7 | 15.7× | −1 | 0.385 |
| 6 | Spending through us falls | `amt_out` | −7 | 5.9× | −1 | 0.529 |
| 10 | Payments to PNC customers fall | `amt_out_internal` | −11 | 4.3× | −1 | 0.770 |
| 5 | Their customers stop paying | `amt_in_internal` | −12 | 3.6× | −1 | 0.854 |
| 3 | Cheque | `amt_out_check` | −12 | 3.6× | −1 | 0.868 |
| 4 | Net flow turns | `net_flow` | −7 | 1.6× | −1 | 0.702 |
| 1 | No new trading partners | `cpty_new_out` | −12 | 1.3× | −4 | 0.794 |
| 2 | No new unfamiliar banks | `fin_new_out` | −12 | 1.1× | −4 | 0.885 |

**Read the pattern:** *counts of relationships* (institutions, counterparties) carry
precision; *dollar volumes* carry recall. They are complementary, which is the case for a
two-tier queue rather than a single rule.

### 3.4 Three findings that survived every rebuild

1. **Fewer payments, not smaller ones**, at every month. `count_dd` 0.946 → 0.191 → 0.000
   while `ticket_dd` holds 0.968 → 0.616. Stayers flat at ~1.00 throughout.
2. **Net-flow-negative share** climbs 0.474 → 0.702 against a stayer line pinned at
   0.440–0.456. The only signal that is a *rate* with a monotone trend.
3. **The standing marker.** A full year out, attriters are already materially lighter than
   size peers: `n_out` median 0.600 vs 1.143, `amt_out` 0.591 vs 1.080. Best marker lift is
   `bal_live` at **1.61×** on "below half your peers". No onset date, so it is a
   **watch-list criterion, not a trigger**.
4. **ACH is the sticky rail.** At rel_m −6: cheque 0.220, RTP 0.015, wire 0.530, card
   0.698, **ACH last at 0.736**. They keep using ACH while everything else stops.

---

## 4. What is open

### 4.1 The onset of four signals is outside the window

`amt_out_check`, `amt_in_internal`, `cpty_new_out` and `fin_new_out` all separate at
**exactly rel_m −12**, the edge of the observation window. Because dd has no baseline
window, this is *not* an artefact — it means those signals were already diverging 12
months out and **we cannot see when they started**.

**Fix:** widen `EVENT_PRE` to 15–18 for the subset of attriters with enough history and
re-read. Panel is 31 months and labels need 12, so ~18 is the practical ceiling.

### 4.2 "Combining doesn't help" is only proven for an unweighted count

`SCORE` gives every signal equal weight, so a 30.1× feature and a 1.1× feature count the
same. Single beats combined in 11 of 12 months on raw lift and 11 of 12 at matched recall.

**That is a real result for a plain count rule, and not a result about models.** A
weighted score or a fitted hazard model is untested.

### 4.3 Nobody has priced the precision/recall trade

`fin_out_n` at −1: 30.1× lift but **8.6% recall**. At 20% recall the best available lift
drops to 25.5× at −1 and **2.7× at −12**. `bal_live` is the mirror: 17.7× at 57.2% recall.
No one has designed the actual queue.

### 4.4 The competitor-vs-contraction question is unanswered

Same-name outflow (v3 §6) is a **standing trait, not a trigger**: equal share of both
cohorts ever self-pay (21.8% vs 21.7%), but attriters route 1.7× more of their outflow to
themselves (8.2% vs 4.9%), and that gap is flat from −12. It segments; it does not warn.

### 4.5 Known cosmetic bug

In v6 §5h the `kv()` dict has **duplicate `"  ...of"` keys**, so only the last survives —
it prints `3` (= `len(_near)`) where the first two should read `12` and `7`. The counts
themselves (0, 1, 0, 0) are correct. Give each key a distinct label.

---

## 5. Next steps, in priority order

1. **Design the queue, not another study.** Two tiers:
   `fin_out_n` dd below threshold → small high-precision list;
   `bal_live` dd below threshold → larger triage list.
   Produce precision, recall, flagged-count and alerts-per-true-positive at each month for
   a grid of thresholds. This is the deliverable TM Sales can act on.
2. **Widen the pre-window** to find the onset of the four −12 signals (§4.1).
3. **Fit a weighted model** — discrete-time hazard, rolling-origin split on `m_idx`, dd
   features at t, label at t+1…t+k, nothing at or after t+1 in the feature set. 158,066
   censored accounts make hazard the right frame. This is the only fair test of §4.2.
4. **Validate against `B_bal_exit`.** Ordering is broadly the same but everything separates
   *later* (`bal_live` at −3 vs −5), which is expected: their balance holds up until the
   end. Confirm the queue works for the shell-account population too.
5. **Counterparty data.** `PAYS_CPTY` and `CptyFinEntity` are what would let "banks they
   pay drop away" — the best signal in the whole study — be measured properly rather than
   from the PNC-visible slice alone.
6. **Compliance read** before any use of counterparty data for prospecting.

---

## 6. What to tell a stakeholder today

> The number of distinct banks a client pays is the best early-warning signal we have. A
> client whose institution count drops is up to 30× more likely to leave than a random
> client — but the alert list is small, so pair it with a balance-based triage list for
> coverage. The signal shows up about two months before the balance moves, and today's 30%
> rule gives no usable warning at all for roughly a third of the clients who leave.

**Do not say:** that payments lead by 7 months (that rests on a 27.6%-coverage feature), or
that combining signals fails (only proven for an unweighted count).

---

## 7. Working preferences

- PySpark; temp views fine. **Never `.show()`** — convert to pandas via `disp()`.
- Output batched into a handful of dense cells; results reviewed by screenshot.
- Fail-fast guards and QA diagnostics baked into the code.
- Discuss architecture before coding; write code directly for implementation.
- Opinions and recommendations wanted, not passive implementation.
- Always **Payment Knowledge Graph (PKG)** — never "Payment Knowledge Network".

*Internal — PNC Treasury Management, Data Science*
