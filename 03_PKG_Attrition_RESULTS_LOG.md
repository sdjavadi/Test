# PKG Deposit Attrition — RESULTS LOG

**Every number from every run.** Companion to `01_PKG_Attrition_HANDOFF.md`.
Where a later run supersedes an earlier one, the earlier is kept and marked, because the
*reason* a number changed is usually the finding.

---

## 0. Version history — what each run got wrong

| Run | Verdict | Defect |
|---|---|---|
| v1 | **Void** | `CLOSED_CODE='C'` (sweep code, 3,182 accts) instead of 07/08 (~112k). Deduped to account-day, discarding sweep legs. Alarmed at a 0.9% join rate that was a denominator mistake |
| v2 | **Panels are good** | Built the corrected panels and payment features. Account→customer map fanned out (241,499 rows / 229,363 accounts) — fixed and rebuilt |
| v3 | **Labels are good** | Lead detector searched *inside* its own baseline window (−12…−10), so the 5-month headline was the window edge |
| v4 | **Void** | Swapped self-relative *change* for peer-relative *level* and kept the same thresholds. Deciles built on `bal_live` made `bal_live` uninformative by construction. Lift collapsed to 1.1–5.9× |
| v5 | **Good** | dd construction correct (`median_dd = 1.000`). Headline named a 2.6%-coverage rail; stacking verdict compared different months |
| v6 | **Current** | Coverage floor + per-month single-vs-combined |

---

## 1. Data shape

| Metric | Value |
|---|---|
| Payments rows in scope | 12,811,995,156 (`trans_id` unique — confirmed) |
| Payment date range | 2024-01-01 … 2026-07-31 |
| Deposit leg-day rows | 111,371,750 |
| Deposit date range | 2024-01-02 … 2026-07-31 (647 load dates) |
| Accounts | 229,363 |
| Account-legs | 255,313 |
| Customers (`cust_pwr_id`) | 120,600 |
| Relationships (`rltn_pwr_id`) | 49,294 |

**Grain test:** account-day → **3,675,351** duplicate keys, max 4 rows/key.
account-**leg**-day → **0**.

**Business calendar:** Fri 134, Wed 132, Tue 132, Thu 129, Mon 120 dates. 19–22 load dates
per month. No weekends.

---

## 2. Status codes (v2 §1a)

| source | src_system_cd | code | desc | class | accounts | share_zero_bal | share_closed_dt |
|---|---|---|---|---|---|---|---|
| HOGAN | DDA | 09 | ACTIVE-DO NOT CLOSE | open | 175,905 | 0.311 | 0.000 |
| HOGAN | DDA | 99 | ACTIVE | open | 36,489 | 0.078 | 0.002 |
| HOGAN | DDA | 08 | PURGEABLE | closed | 57,026 | 1.000 | 0.637 |
| HOGAN | DDA | 07 | CLOSED | closed | 55,081 | 1.000 | 0.580 |
| HOGAN | DDA | 03 | INACTIVE | inactive | 7,798 | 0.031 | 0.002 |
| HOGAN | DDA | 05 | DORMANT | dormant | 4,667 | 0.063 | 0.003 |
| HOGAN | DDA | 06 | ESCHEATABLE | escheat | 158 | 0.035 | 0.003 |
| HOGAN | DDA | 01 | NEW | open | 113 | 0.225 | 0.000 |
| HOGAN | DDA | 12 | IN PROCESS OF CLOSING | closing | 7 | 0.000 | 1.000 |
| AGILETICS_SWEEP | SWP | O | UNKNOWN | open | 7,336 | 0.115 | 0.000 |
| AGILETICS_SWEEP | SWP | C | UNKNOWN | closed | 3,182 | 1.000 | 0.071 |

**Mean days since `closed_dt`:** 08 → 264.7 · 07 → 98.9 · C → 243.2.
Confirms 08 is post-07 retention.

---

## 3. Closure mechanics (v2)

| Metric | Value |
|---|---|
| Accounts | 229,363 |
| Closed in window (status-derived) | 64,554 |
| …evaluable (12+ months history) | 25,050 |
| With `closed_dt` | 65,580 |
| `closed_dt` but never non-live | 1,478 |
| Non-live but no `closed_dt` | 452 |
| Both fire, **median month gap** | 64,102 → **1.0** |
| Still live at panel end (censored) | 158,066 |
| Ever fully non-live | 65,975 |
| …later became live again | 314 (**0.5%** — absorbing) |
| Non-live purely on idle sweep legs | **76 accounts** |
| Pre-2000 `closed_dt` on live accounts | ~239 (recycled numbers) |

---

## 4. `avg_monthly_bal_1` (v2 §3)

| data_source | family | share_populated |
|---|---|---|
| HOGAN | NIBDDA / IBDDA / MMDA / Retail | **1.000** |
| AGILETICS_SWEEP | Sweep On/Off Balance Sheet | **0.000** |

Fit (HOGAN, 82,293,332 rows): `mean_bal_prior_month` **0.013** median abs err /
0.462 within 1%; `mtd_mean` 0.151 / 0.216; `mean_bal_month` 0.152 / 0.201.

Within month: constant 0.334 · changes daily 0.009 · mean distinct per day 0.092.

**Restatement day:** 31 → 1,353,001 · 30 → 1,134,680 · 29 → 564,089 · 28 → 340,972 ·
27 → 114,090 · 1 → 69,736 · 2 → 36,081 · 3 → 21,834 · rest noise.
→ **prior complete month, refreshed at month end.**

---

## 5. Panels (v2 §4, §6)

| Metric | Value |
|---|---|
| Account-months | 5,253,585 |
| Customer-months (deposits) | 2,746,664 |
| Accounts / Customers | 229,363 / 120,457 |
| Account-months with >1 leg | 182,757 |
| Accounts ever multi-leg | 7,538 |
| Median balance, **all rows** | ~$11.7k–15.7k |
| Median balance, **live only** | ~$22.6k–24.8k |

Post-closure zero rows were roughly halving the median.

**Study population:** deposit customers 120,457 · payment-visible **94,186** ·
**both 94,179** · customer-months payments 1,952,288 · joined 1,829,202.

**Payments build:** map 229,363 rows / 229,363 accounts (assert passes);
~98–115M sides per month; 60–130 s each; ~40 min for 31 months.

---

## 6. Labels and the incumbent (v6 §2)

| Definition | Raw | Qualified | Monthly hazard |
|---|---|---|---|
| `A_full_exit` | 20,248 | **16,384** | **0.9115%** |
| `B_bal_exit` | 14,016 | 14,016 | 0.7798% |
| `C_p30` | 57,551 | 57,551 | — |

**Evaluable months: 19.**

### The incumbent 30% rule, measured

| Metric | Value |
|---|---|
| Attriters | 16,384 |
| …30% rule ever fires for | 13,555 |
| …**never fires for** | **2,829** (17%) |
| …**fires AFTER the exit** | **3,059** (19%) |
| **Median months of warning** | **2.0** |
| p25 / p75 | 0.0 / 7.0 |

Plus (v2 §5a): it fired on **43,143 customers who never left**.
→ **36% of attriters get no usable warning.**

### Definition comparison (v2 §5a)

| Definition | n | share | also A | median lead vs A | fires without A |
|---|---|---|---|---|---|
| A_full_exit | 20,248 | 0.214 | 20,248 | 0 | 0 |
| B_bal_exit | 14,016 | 0.148 | 10,953 | 0 | 3,063 |
| C_p30 | 57,551 | 0.608 | 14,408 | 1 | **43,143** |

`B ⊂ C` exactly (14,016 = 14,016).

### Account closure ≠ customer attrition (v2 §5c)

| Class | Closures | Customers | Mean peak balance |
|---|---|---|---|
| account churn, customer stayed | 33,428 | 19,108 | $4,180,879 |
| customer left entirely | 25,219 | 19,638 | $1,686,499 |
| part of a later full exit | 5,907 | 2,976 | $784,923 |

---

## 7. dd coverage (v6 §1a) — `median_dd = 1.000` for every covered feature

| Feature | share_with_dd | Feature | share_with_dd |
|---|---|---|---|
| `bal_live` | 0.651 | `amt_out_internal` | 0.338 |
| `amt_in` / `amt_out` | 0.466 | `fin_out_n` | 0.311 |
| `n_out` | 0.425 | `amt_in_internal` | 0.290 |
| `n_in` | 0.409 | `amt_out_check` | 0.245 |
| `amt_out_ach` | 0.403 | `amt_out_wire` | 0.191 |
| `avg_ticket_out` | 0.435 | `cpty_new_out` | 0.118 |
| `avg_ticket_in` | 0.426 | `amt_out_card` | 0.067 |
| `cpty_out_n` | 0.365 | `fin_new_out` | 0.049 |
| | | `amt_out_rtp` | 0.024 |
| | | `amt_out_other` / `net_flow` | 0.000 (NaN by design) |

---

## 8. v6 — lead table, `A_full_exit` (16,384 attriters / 71,998 stayers)

| Feature | read_on | coverage | eligible | sep_rel_m | lead | gap@−6 | gap@event | max_gap |
|---|---|---|---|---|---|---|---|---|
| `amt_out_check` | med_dd | 0.276 | ✔ | **−12** | 12 | 0.809 | 1.023 | 1.042 |
| `amt_in_internal` | med_dd | 0.324 | ✔ | **−12** | 12 | 0.792 | 1.028 | 1.029 |
| `cpty_new_out` | rate_any | 0.649 | ✔ | **−12** | 12 | 0.137 | 0.306 | 0.325 |
| `fin_new_out` | rate_any | 0.649 | ✔ | **−12** | 12 | 0.077 | 0.182 | 0.193 |
| `amt_out_internal` | med_dd | 0.378 | ✔ | −11 | 11 | 0.435 | 1.016 | 1.021 |
| `amt_out_ach` | med_dd | 0.451 | ✔ | −8 | 8 | 0.272 | 1.006 | 1.009 |
| `amt_in` | med_dd | 0.520 | ✔ | −8 | 8 | 0.272 | 0.975 | 1.002 |
| `amt_out` | med_dd | 0.520 | ✔ | −7 | 7 | 0.238 | 1.007 | 1.008 |
| `n_out` | med_dd | 0.474 | ✔ | −7 | 7 | 0.189 | 1.003 | 1.005 |
| `n_in` | med_dd | 0.457 | ✔ | −7 | 7 | 0.212 | 0.900 | 0.948 |
| `net_flow` | rate_neg | 0.684 | ✔ | −7 | 7 | 0.066 | 0.212 | 0.290 |
| `cpty_out_n` | med_dd | 0.406 | ✔ | −6 | 6 | 0.159 | 0.684 | 0.684 |
| `bal_live` | med_dd | 0.736 | ✔ | **−5** | 5 | 0.115 | 1.016 | 1.017 |
| `fin_out_n` | med_dd | 0.347 | ✔ | −5 | 5 | 0.143 | 0.667 | 0.667 |
| `avg_ticket_out` | med_dd | 0.484 | ✔ | −3 | 3 | 0.057 | 0.815 | 0.815 |
| `avg_ticket_in` | med_dd | 0.475 | ✔ | −2 | 2 | 0.043 | 0.583 | 0.608 |
| `amt_out_rtp` | med_dd | **0.026** | ✘ | (−12) | | 0.985 | 1.039 | 1.102 |
| `amt_out_wire` | med_dd | **0.212** | ✘ | (−12) | | 0.489 | 1.018 | 1.021 |
| `amt_out_card` | med_dd | **0.075** | ✘ | (−9) | | 0.312 | 1.013 | 1.013 |

**16 eligible, 3 excluded.** Headline as printed: `amt_out_check` at −12, lead 7 months.
**Conservative claim on dense features: `amt_out`/`n_out`/`n_in` at −7 vs `bal_live` at −5
→ 2 months.**

### `B_bal_exit` (14,016 / 72,009) — same ordering, everything later

`amt_in_internal` −12 · `cpty_new_out` −12 · `amt_out_check` −11 · `amt_out_internal` −8 ·
`net_flow` −7 · `fin_new_out` −7 · `amt_out_ach` −6 · `amt_in` −5 · `n_in` −5 ·
`amt_out` −4 · `n_out` −4 · `bal_live` **−3** · `cpty_out_n` −3 · `fin_out_n` −2 ·
`avg_ticket_in` −1 · `avg_ticket_out` 0.

Expected: their balance holds up until the end, so it separates *later*, which widens the
payment lead for this population.

---

## 9. v6 — operating points, `A_full_exit`

| Feature | best lift | at rel_m | precision | recall | usable at 25%/20%? |
|---|---|---|---|---|---|
| `fin_out_n` | **30.1×** | −1 | 0.274 | 0.086 | no |
| `cpty_out_n` | 23.8× | −1 | 0.217 | 0.123 | no |
| `bal_live` | 17.7× | −1 | 0.161 | 0.572 | no |
| `n_in` | 17.0× | −1 | 0.155 | 0.570 | no |
| `n_out` | 15.7× | −1 | 0.143 | 0.385 | no |
| `amt_out` | 5.9× | −1 | 0.054 | 0.529 | no |
| `amt_out_internal` | 4.3× | −1 | 0.039 | 0.770 | no |
| `amt_in_internal` | 3.6× | −1 | 0.033 | 0.854 | no |
| `amt_out_check` | 3.6× | −1 | 0.033 | 0.868 | no |
| `net_flow` | 1.6× | −1 | 0.014 | 0.702 | no |
| `cpty_new_out` | 1.3× | −4 | 0.012 | 0.794 | no |
| `fin_new_out` | 1.1× | −4 | 0.010 | 0.885 | no |

`B_bal_exit`: `fin_out_n` 25.9× · `cpty_out_n` 18.2× · `n_in` 12.1× · `bal_live` 10.8× ·
`n_out` 9.9× · `amt_out` 3.6× · `amt_out_internal` 3.2× · `amt_in_internal` 2.9× ·
`amt_out_check` 2.7× · `net_flow` 1.7× · `cpty_new_out` 1.2× · `fin_new_out` 1.1×.

**Nothing reaches 25% precision at 20% recall.** Lift is the readable metric at a 0.91%
base rate.

---

## 10. v6 §5g — single vs combined, WITHIN each month ★

| rel_m | best single | feature | best combined | k | winner | single @recall≥20% | combined @recall≥20% | winner @recall |
|---|---|---|---|---|---|---|---|---|
| −12 | 8.0 | `fin_out_n` | 4.4 | 9 | single | 2.7 | 2.0 | single |
| −11 | 7.1 | `fin_out_n` | 5.0 | 9 | single | 2.6 | 2.0 | single |
| −10 | 6.0 | `fin_out_n` | 5.0 | 9 | single | 3.1 | 2.6 | single |
| −9 | 8.6 | `fin_out_n` | 5.3 | 9 | single | 3.4 | 2.8 | single |
| −8 | 8.5 | `fin_out_n` | 6.0 | 9 | single | 3.9 | 3.1 | single |
| −7 | 12.0 | `fin_out_n` | 6.3 | 9 | single | 4.4 | 3.3 | single |
| −6 | 13.2 | `fin_out_n` | 7.9 | 9 | single | 5.6 | 5.3 | single |
| −5 | 15.2 | `fin_out_n` | 9.3 | 9 | single | 7.1 | 6.1 | single |
| −4 | 20.2 | `fin_out_n` | 10.3 | 9 | single | 10.9 | 10.3 | single |
| −3 | 23.0 | `fin_out_n` | 12.0 | 9 | single | 11.8 | **12.0** | **combined** |
| −2 | 25.9 | `fin_out_n` | 13.8 | 9 | single | 18.1 | 13.8 | single |
| −1 | 30.1 | `fin_out_n` | 16.3 | 9 | single | 25.5 | 16.3 | single |

**`fin_out_n` is the best single feature at all 12 months.**
Combining wins **0 of 12** on raw lift, **1 of 12** at matched recall.
Far window (≤ −6): combined wins **0 of 7**. Near window (≥ −3): **0 of 3**.

*(Display bug: the `kv` summary prints "…of 3" for all three counts — duplicate dict key.
Counts 0 / 1 / 0 / 0 are correct.)*

Combined-score lift by `min_signals` (rel_m −12 → −1):
k=5 → 1.6…3.4 · k=7 → 2.5…7.6 · k=9 → 4.4…16.3.
Recall at k=9 falls from 0.082 to 0.396.

---

## 11. Deep dives

### 11a. Rails — which goes first (v6 §5a)

| Rail | coverage | sep_rel_m | dd@−6 | dd@−3 |
|---|---|---|---|---|
| WIRE | 0.212 | −12 | 0.530 | 0.199 |
| CHECK | 0.276 | −12 | **0.220** | 0.000 |
| RTP | 0.026 | −12 | **0.015** | 0.000 |
| CARD | 0.075 | −9 | 0.698 | 0.411 |
| **ACH** | 0.451 | −8 | **0.736** | 0.272 |
| OTHER | 0.000 | — | — | too sparse |

**Cheque and RTP go first and hardest; ACH is the sticky rail.**

### 11b. Fewer vs smaller (v6 §5c) — `driver = FEWER payments` at every month

| rel_m | count_dd | ticket_dd | amount_dd |
|---|---|---|---|
| −12 | 0.946 | 0.968 | 0.930 |
| −9 | 0.907 | 0.977 | 0.887 |
| −6 | 0.816 | 0.943 | 0.769 |
| −3 | 0.547 | 0.850 | 0.430 |
| −1 | 0.191 | 0.616 | 0.073 |
| 0 | 0.000 | 0.186 | 0.000 |

Stayers flat at 1.00–1.01 on all three throughout.

### 11c. Net flow sign (v6 §5d) — share running negative

| rel_m | attriter | stayer |
|---|---|---|
| −12 | 0.474 | 0.440 |
| −9 | 0.481 | 0.442 |
| −6 | 0.510 | 0.443 |
| −3 | 0.570 | 0.448 |
| −1 | **0.702** | 0.450 |
| 0 | 0.240 | 0.452 |
| +3 | 0.166 | 0.456 |

Gap 0.034 → 0.252, monotone. Collapse after 0 is mechanical — no flow at all.

### 11d. Standing marker (v6 §4c/4d) — a full year out

| Feature | cohort | n | median vs peer | share below half of peers | lift |
|---|---|---|---|---|---|
| `amt_out` | attriter | 13,467 | 0.591 | 0.482 | **1.25×** |
| | stayer | 55,901 | 1.080 | 0.385 | |
| `bal_live` | attriter | 11,637 | 0.868 | 0.279 | **1.61×** |
| | stayer | 49,345 | 1.032 | 0.174 | |
| `cpty_out_n` | attriter | 13,467 | 1.000 | 0.252 | **1.24×** |
| | stayer | 55,901 | 1.111 | 0.204 | |
| `n_out` | attriter | 13,467 | 0.600 | 0.458 | **1.30×** |
| | stayer | 55,901 | 1.143 | 0.351 | |

No onset date → **watch-list criterion, not a trigger.**

### 11e. Same-name outflow (v3 §6) — a trait, not a trigger

| | attriter | stayer |
|---|---|---|
| n customers | 12,994 | 69,937 |
| share ever self-pay | 0.218 | 0.217 |
| mean self-pay share | **0.082** | **0.049** |
| median share, conditional on baseline self-payers | **0.392** | 0.050 |

Gap present at −12 and flat. Max rate gap 0.089 at rel_m +1.

---

## 12. Superseded numbers, kept for the record

| Run | Claim | Why it was wrong |
|---|---|---|
| v3 | payments lead by **5 months** (`cpty_new_out` at −10) | Search started inside the baseline window |
| v4 | best lifts **1.1–5.9×**; "combining doesn't help" | Peer *levels* against change-tuned thresholds; a 0.70 cut flags ~half the population |
| v4 | `bal_live` separates at rel_m **0** | Deciles built on `bal_live` — circular |
| v5 | payments lead by **7 months** (`amt_out_rtp` at −12) | 2.6% coverage |
| v5 | "single 30.1× beats combined 16.3×" | −1 vs −12 — different months |
| v1 | 3,675,351 duplicate account-days | Deduped on all ~70 columns incl. rates |
| v1 | 0.9% payments-to-deposits join rate | Payments is the whole bank — denominator mistake |

*Internal — PNC Treasury Management, Data Science*
