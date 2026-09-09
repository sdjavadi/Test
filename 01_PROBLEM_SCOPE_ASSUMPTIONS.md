# 01 · Problem, scope and assumptions

---

## 1. The objective

Two questions, stated at the start of the programme.

**Q1 — Does payment behaviour flag a departing treasury client earlier, or better, than the
deposit balance does?**
Answered: **yes, and the more important half is "better", not "earlier".** Payment features find
clients the balance never sees. On the population that keeps accounts open and drains them,
balance-based scoring is close to useless (see `07`).

**Q2 — Can it separate *moving to a competitor* from *business contraction*?**
**Unanswered.** One attempt returned a clean null. The construct and the two fixes that would give
it a chance are in `08`.

---

## 2. Why this matters commercially

The bank runs a 30% balance rule today. Measured against 16,384 qualified departures it:

- **never fires** for 2,829 of them (17%)
- **fires after the exit** for another 3,059 (19%)
- fires on **43,143 customers who never left**

So roughly **36% of departing clients get no usable warning at all**, and the alert list is mostly
noise. The opportunity is not marginal accuracy — it is the third of departures that are currently
invisible.

---

## 3. Scope

### In scope
- Deposit customers of PNC Treasury Management, 2024-01-01 → 2026-07-31.
- Payment behaviour from the transaction-level source table.
- Deposit balances, account status, account structure.
- Prediction of **client-level departure**, not account-level closure.

### Out of scope (and why)
| Excluded | Reason |
|---|---|
| Anything before 2024-01 | **Nothing before 2024 is trusted and none can be added.** This is the single constraint that shapes the whole design |
| The Neo4j graph | No graph is built for this use case. The analysis reads the payment source table directly |
| Customer attributes — NAICS, segment, size band, product holdings | **Deliberately held back** so their lift is measurable against a fixed feature baseline rather than confounded with the payment rebuild |
| Account-level churn | 33,428 account closures belong to customers who stayed. Account closure ≠ customer attrition (see `03`) |
| CDs / time deposits | None exist in this universe. `maturity_dt` is null throughout |
| Returns and reversals | Not present in the payment table |

---

## 4. Population

| | Count |
|---|---|
| Deposit customers | 120,600 |
| Deposit accounts / account-legs | 229,363 / 255,313 |
| Relationships (`rltn_pwr_id`) | 49,294 |
| Payment-visible deposit customers | ~96,000 (80% of the book) |
| **Study population — both sources** | **94,179** |
| At-risk client-months | 2,005,315 |

Payments is the **whole bank**: 12.8bn transactions and 12.4M mdm ids against a deposit book of
229,363 accounts. Real coverage is **72% of deposit accounts and 80% of deposit customers**.

---

## 5. Standing assumptions

Each of these is a modelling choice that could be revisited. They are listed so that a future
reader knows what was assumed rather than discovered.

### 5.1 About the target
1. **A client that closes every account and stays closed has left.** Verified absorbing at 99.5% —
   of 65,975 accounts that ever went fully non-live, 314 came back.
2. **A closure label is the first month after the last month with a live leg.** Agrees with the
   `closed_dt` field on 64,102 accounts at a **1-month median gap**.
3. **Status `08 PURGEABLE` is post-`07` retention, not a distinct reason.** Mean 264.7 days since
   `closed_dt` against 98.9 for `07`, both at `share_zero_bal = 1.000`.
4. **Sweep `C` is a leg state, not an account state.** Only 76 accounts are ever judged non-live
   purely because sweep legs are idle.
5. **Qualification is necessary.** A client must have 12 months of history and 6 live months before
   the event, or a client already winding down at panel start counts as an exit. This removes 3,864
   of A's 20,248 raw events.

### 5.2 About the features
6. **Relative beats absolute.** No feature is used at its raw level. Everything is the client
   against its own trailing window, then against its frozen peer group in the same calendar month.
7. **Peer groups must be frozen.** Recomputed monthly, a shrinking client slides down the balance
   deciles alongside its own decline and the signal cancels.
8. **A 3-month reference window at `t−6 … t−4`** is used rather than a 12-month event-aligned
   pre-window. The latter would permanently discard ~4,200 of 16,384 attriters against the 2024
   floor, and is not computable at scoring time.
9. **Missingness is informative.** Every `dd` is paired with an explicit missingness indicator. A
   client with no measurable institution activity is a fact, not a gap.
10. **`segment_desc` is not point-in-time.** It is restated retrospectively — restatement lift 21.7
    at rel_m −1 — and is excluded from the peer grouping and every feature set.

### 5.3 About the evaluation
11. **A queue is a rank within a calendar month over the whole live book.** Not a threshold, not an
    event-time statistic. This is the single most consequential assumption in the programme and it
    changed several conclusions.
12. **Precision at the operating capacity is the ship metric**, not AUC. AUC averages over the whole
    ranking; a 250-name list lives in its top 0.3%. They disagreed materially (see `05`).
13. **A client re-flagged within 3 months is the same conversation.** Alerts-per-true-positive is
    reported both raw and conversation-collapsed.
14. **Right-censored rows are dropped, not zeroed.** Zeroing relabels every censored client a
    stayer.

### 5.4 Known-unverified assumptions
15. **Seven folds is enough.** AUC standard deviation across folds is ~0.017 and per-fold precision
    spread has **not** been examined. If one origin carries a result, the conclusion changes.
16. **The 94,179 study population is representative of the rest.** Clients invisible to payments
    (20% of the book) are excluded from everything and have never been profiled.
17. **A 6-month horizon is the right one.** H=1 and H=3 are computed but the queue design has only
    ever been priced at H=6.
18. **Precision translates to retained balances.** Untested. Every alert is weighted equally; a
    value-weighted queue has never been built.

---

## 6. What "done" would look like

1. A monthly scoring job producing a ranked list at an agreed capacity.
2. A named owner for the queue and an agreed alert volume.
3. Compliance and fair-lending sign-off.
4. A pilot that measures **retained balances**, not precision — that is the number the business
   would judge this on, and only a live run can produce it.

*Internal — PNC Treasury Management, Data Science*
