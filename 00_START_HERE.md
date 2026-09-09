# PKG Deposit Attrition — START HERE

**Payment Knowledge Graph (PKG) · PNC Treasury Management · Data Science**
*State after the v8b run. This file set supersedes `01_PKG_Attrition_HANDOFF.md`,
`02_PKG_Attrition_DATA_AND_CODE.md` and `03_PKG_Attrition_RESULTS_LOG.md`.*

---

## If you are picking this up cold

Read `01`, `03` and `05` first — problem, target, and how anything is judged. Everything
else is reference you can query as needed. If you only have room for one file, read
**`07_FINDINGS_AND_SUPERSEDED.md`**: it carries both what is true and what was briefed and
later withdrawn, and the withdrawn list is the more useful half.

---

## Current state, one paragraph

Payment behaviour predicts a departing treasury client, and it predicts the ones the deposit
balance cannot see at all. At the same alert volume the bank works today (~12,667 a month) a
fitted model finds **5,860 departing clients against the incumbent rule's 4,103**, with **2,041
fewer client conversations** and the **same 4-month median warning**. On clients who drain an
account rather than close it, balance-based scoring retains **12%** of its precision and
payment-based scoring retains **81%**. The current best specification adds inbound-institution
and outbound-counterparty churn features and reaches **precision 0.7269 at a 250-name monthly
list**, against 0.6514 for the v7 baseline.

---

## The recommended specification

```
+ fin_in + cpty_acct_out
84 features · AUC 0.8006 (sd 0.0171 across 7 folds) · precision@K=250 0.7269 · recall 0.0465
baseline: v7 36 features · AUC 0.7908 · precision@K=250 0.6514
```

`fin_in` alone reaches 0.7246, so the second block is worth +0.0023 for 24 more features. If
simplicity matters, ship `+ fin_in`.

---

## File map

| File | Contains |
|---|---|
| `00_START_HERE.md` | This file — state, conventions, vocabulary, do-not-say list |
| `01_PROBLEM_SCOPE_ASSUMPTIONS.md` | Objective, business framing, population, scope, every standing assumption |
| `02_DATA_AND_FIELDS.md` | Source tables, every column with population rate, artefact inventory, HDFS paths |
| `03_TARGET_DEFINITION.md` | Three attrition definitions, closure mechanics, dependent variable, risk set |
| `04_FEATURES.md` | The dd transform, every feature family, coverage, exclusions |
| `05_MODEL_AND_METRICS.md` | Model spec, rolling origin, why precision@K is the ship metric, the queue |
| `06_RESULTS_LOG.md` | Every number from every run, v1 → v8b |
| `07_FINDINGS_AND_SUPERSEDED.md` | What is settled; every claim withdrawn and why |
| `08_NEXT_STEPS.md` | Prioritised backlog with effort and dependency |
| `09_ENGINEERING_TRAPS.md` | Environment, code conventions, every trap that cost a run |

---

## Vocabulary — use these terms exactly

| Term | Means |
|---|---|
| **PKG** | Payment Knowledge Graph. **Never** "Payment Knowledge Network" |
| **dd** | Difference-in-differences: client vs own recent history, then vs frozen peer group |
| **rel_m** | Event-time index — months relative to the exit. Used only in §2/§4 of v7 |
| **m_idx** | **Absolute** month index (~24289–24319). NOT 0-based. See `09` |
| **Risk set** | One row per (client, calendar month) while live and pre-event |
| **Queue coverage** | Share of the at-risk book a rule can score at all |
| **precision@K** | Precision of the top-K ranked clients within a calendar month |
| **A_full_exit** | Primary label — every account non-live and stays that way |
| **B_bal_exit** | Validation label — balance drained but accounts stay open |
| **The incumbent** | The 30% balance rule in production (`C_p30`) |

---

## Do not say

These were briefed and are now known to be wrong. If you find them in a slide, correct them.

1. **"`fin_out_n` is the number of banks a client pays, 30× lift."** It is *the number of
   distinct institutions a client sends **ACH** to*. It scores 42% of the book and ranks 5th of
   8 across the whole population.
2. **"Combining signals doesn't help."** Withdrawn, not amended. Even an unweighted count of
   twelve equal votes beats the best single feature in calendar time.
3. **"Payments lead the balance by 2 months."** Event-time claim. The calendar-time replacement
   is a median lead of 2 months at a 250-name list and 4 months at incumbent volume.
4. **"Counterparty coverage is blocked on `PAYS_CPTY` / `CptyFinEntity`."** Those milestones
   belong to the **Neo4j graph**. This analysis does not use the graph — it reads the source
   table, which already carries counterparty name, account and bank on both directions.
5. **"Re-key counterparties on name for 2.5× coverage."** The account key wins on both coverage
   (0.504 vs 0.464) and lift (+0.0064 vs +0.0038).
6. **"Trend features are free information."** 94 trend features cost 11 precision points.
7. **"The two-tier queue is the deliverable."** One ranked list taken to the same depth beats it
   on precision *and* recall.

---

## Quick facts

| | |
|---|---|
| Panel | 2024-01-01 → 2026-07-31, 31 months, **hard floor, nothing before 2024 is trusted** |
| Payments | 12,811,995,156 transactions |
| Deposits | 111,371,750 leg-days, 229,363 accounts, 120,600 customers, 49,294 relationships |
| Study population | 94,179 clients visible in both deposits and payments |
| At-risk client-months | 2,005,315 |
| Qualified attriters (A) | 16,384 · monthly hazard 0.9115% |
| Evaluation | 7 rolling origins, m_idx 24307–24313, horizon 6 months |
| Base rate at H=6 | ~5.05% |

---

## Working preferences (carried from the original handoff)

- PySpark; temp views fine. **Never `.show()`** — convert to pandas via a `disp()` helper.
- Output batched into a handful of dense cells; results reviewed by screenshot.
- Fail-fast guards and QA diagnostics baked into the code.
- Discuss architecture before coding; write code directly for implementation tasks.
- Opinions and recommendations wanted, not passive implementation.
- Always **Payment Knowledge Graph (PKG)**.

*Internal — PNC Treasury Management, Data Science*
