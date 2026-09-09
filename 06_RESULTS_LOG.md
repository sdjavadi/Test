# 06 · Results log — every number from every run

*Where a later run supersedes an earlier one, the earlier is kept and marked, because the reason a
number changed is usually the finding.*

---

## 0. Run history

| Run | Verdict | What |
|---|---|---|
| v1 | **Void** | `CLOSED_CODE='C'` (sweep, 3,182) instead of 07/08 (~112k). Deduped to account-day |
| v2 | Kept | Panels and payment features. Account→customer map fanned out (241,499 rows / 229,363 accounts) — fixed with an assert |
| v3 | Kept | Labels. Lead detector searched inside its own baseline window |
| v4 | **Void** | Swapped self-relative *change* for peer-relative *level*, same thresholds. Deciles on `bal_live` made `bal_live` uninformative by construction |
| v5 | Kept | dd construction correct (`median_dd = 1.000`). Headline named a 2.6%-coverage rail |
| v6 | Superseded | Coverage floor, per-month single-vs-combined. **Event time throughout** |
| v7 | Kept | **Calendar time.** Rolling-origin hazard, queue, capacity, B validation |
| v8 | Partly void | 74 features, rail audit. `MIN_REF` defect nulled five families |
| v8b | **Current** | Ratio repair, re-ablation, shipping decision |

---

## 1. Data shape

| Metric | Value |
|---|---|
| Payments rows in scope | 12,811,995,156 (`trans_id` unique — confirmed) |
| Payment date range | 2024-01-01 … 2026-07-31 |
| Deposit leg-day rows | 111,371,750 |
| Deposit date range | 2024-01-02 … 2026-07-31 (647 load dates) |
| Accounts / legs | 229,363 / 255,313 |
| Customers / relationships | 120,600 / 49,294 |
| Grain test | account-day → **3,675,351** duplicate keys, max 4 rows/key; account-**leg**-day → **0** |
| Business calendar | Fri 134, Wed 132, Tue 132, Thu 129, Mon 120. 19–22 load dates/month. No weekends |
| Account-months / customer-months | 5,253,585 / 2,746,664 |
| Study population | deposit 120,457 · payment-visible **94,186** · **both 94,179** |
| At-risk client-months (v7) | **2,005,315** |

---

## 2. Labels

| Definition | Raw | Qualified | Monthly hazard |
|---|---|---|---|
| `A_full_exit` | 20,248 | **16,384** | **0.9115%** |
| `B_bal_exit` | 14,016 | 14,016 | 0.7798% |
| `C_p30` | 57,551 | 57,551 | — |

Evaluable months: 19. Base rate at H=6 ≈ **5.05%**. Attriting clients in the 7-month test window
≈ **7,845**.

---

## 3. v6 — event-time signal ordering (SUPERSEDED as a ranking)

See `04` §2 for the full twelve-signal table. Headline lifts: `fin_out_n` 30.1× at rel_m −1 with
8.6% recall; `bal_live` 17.7× at 57.2% recall. **Nothing reached 25% precision at 20% recall.**

Single vs combined, per month (v6 §5g): `fin_out_n` best single at all 12 months; combining won
0 of 12 on raw lift, 1 of 12 at matched recall. **This conclusion is withdrawn — see `07`.**

---

## 4. v7 — widened pre-window (event time, with a within-cohort control)

| | |
|---|---|
| Attriters with ≥18 months of dd history | **4,734** (53.0% of the 12-month cohort) |
| Matched stayers | 62,816 |
| Of the four −12 signals, onset now visible | **0** |
| Of the four, still pinned at the edge | **4** |
| Median cohort effect across eligible features | **0.000** |

`amt_in_internal`, `amt_out_check`, `cpty_new_out`, `fin_new_out`: −12 → −12 → **−18**, all still
at the edge. `amt_out_internal` moved −11 → −10 → −14 (a real onset). Everything else unchanged.

---

## 5. v7 — model comparison (calendar time, 7 rolling origins)

### AUC by horizon, with queue coverage
| Model | H=1 | H=3 | **H=6** | Queue coverage | AUC sd |
|---|---|---|---|---|---|
| `M4_pay_plus_bal` | 0.909 | 0.845 | **0.795** | 1.000 | 0.050 |
| `M3_pay_only` | 0.827 | 0.792 | 0.760 | 1.000 | 0.029 |
| `R_count12` | 0.892 | 0.812 | 0.748 | 0.623 | 0.061 |
| `R_fin_out_n` | 0.877 | 0.777 | 0.714 | **0.423** | 0.073 |
| `R_bal_live` | 0.869 | 0.766 | 0.704 | 0.838 | 0.071 |
| `M1_bal_only` | 0.838 | 0.748 | 0.691 | 1.000 | 0.064 |
| `M2_fin_only` | 0.682 | 0.669 | 0.655 | 1.000 | 0.014 |
| `R_p30` | 0.689 | 0.644 | 0.613 | 1.000 | 0.035 |

### Pooled precision at K (H=6)
| Model | 50 | 100 | **250** | 500 | 1000 | 2500 |
|---|---|---|---|---|---|---|
| `M4_pay_plus_bal` | 0.834 | 0.771 | **0.651** | 0.587 | 0.554 | 0.384 |
| `M3_pay_only` | 0.557 | 0.549 | 0.515 | 0.446 | 0.329 | 0.211 |
| `M1_bal_only` / `R_bal_live` | 0.503 | 0.514 | 0.508 | 0.513 | 0.512 | 0.347 |
| `R_count12` | 0.500 | 0.437 | 0.373 | 0.325 | 0.261 | 0.178 |
| `R_fin_out_n` | 0.366 | 0.320 | 0.291 | 0.262 | 0.190 | 0.119 |
| `M2_fin_only` | 0.366 | 0.320 | 0.291 | 0.262 | 0.179 | 0.114 |
| `R_p30` | 0.066 | 0.080 | 0.103 | 0.107 | 0.112 | 0.112 |

### Recall at K (client-months)
| Model | 50 | 100 | 250 | 500 | 1000 | 2500 |
|---|---|---|---|---|---|---|
| `M4_pay_plus_bal` | 0.011 | 0.020 | 0.042 | 0.075 | 0.142 | 0.246 |
| `M1_bal_only` | 0.006 | 0.013 | 0.033 | 0.066 | 0.131 | 0.222 |
| `M3_pay_only` | 0.007 | 0.014 | 0.033 | 0.057 | 0.084 | 0.135 |
| `R_count12` | 0.006 | 0.011 | 0.024 | 0.042 | 0.067 | 0.114 |
| `R_fin_out_n` | 0.005 | 0.008 | 0.019 | 0.034 | 0.049 | 0.076 |
| `R_p30` | 0.001 | 0.002 | 0.007 | 0.014 | 0.029 | 0.071 |

### Coverage or discrimination? (re-scored on `fin_out_n`'s own ground)
| Model | AUC whole book (n=541,128) | AUC where fin defined (n=228,977) |
|---|---|---|
| `M4_pay_plus_bal` | 0.795 | **0.819** |
| `M3_pay_only` | 0.761 | 0.812 |
| `R_count12` | 0.747 | 0.775 |
| `R_bal_live` | 0.704 | 0.724 |
| `M1_bal_only` | 0.692 | 0.714 |
| `R_fin_out_n` | 0.713 | 0.713 |
| `M2_fin_only` | 0.659 | 0.712 |

**The subset is easier for everyone.** What matters is the ordering: on the champion's own ground
`M4` reaches 0.819 against its 0.713. The fitted model buys reach **and** sharper ranking.

---

## 6. v7 — the queue, capacity-matched

| Queue | Alerts/mo | Alerts | Conversations | TP client-months | **TP clients** | Precision | Conv/TP | Recall | Median lead | p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| Model @ K=250 | 250 | 1,750 | 1,122 | 1,140 | 754 | **0.651** | 1.488 | 0.096 | 2 | 6 |
| Model @ K=1,000 | 1,000 | 7,000 | 3,818 | 3,879 | 2,308 | 0.554 | 1.654 | 0.294 | 2 | 6 |
| Model @ K=2,500 | 2,500 | 17,500 | 8,708 | 6,715 | 3,447 | 0.384 | 2.526 | 0.440 | 2 | 6 |
| **Model @ K=12,667** | 12,667 | 88,669 | **28,386** | 14,456 | **5,860** | **0.163** | **4.844** | **0.747** | **4** | 6 |
| **Incumbent 30% rule** | 12,667 | 88,668 | 30,427 | 10,356 | 4,103 | 0.117 | 7.416 | 0.523 | 4 | 6 |

**At identical volume: +1,757 clients (+42.8%), −2,041 conversations, −35% cost per save, same
lead.** No axis on which the incumbent wins.

### Two tiers vs one list
| Design | Alerts | TP | Precision | Recall |
|---|---|---|---|---|
| Tier 1 only (model) | 1,750 | 1,140 | 0.651 | 0.042 |
| Tier 2 only (balance, net of tier 1) | 7,000 | 3,161 | 0.452 | 0.115 |
| Two tiers combined | 8,750 | 4,301 | 0.492 | 0.157 |
| **ONE list, 1,250 deep** | 8,750 | **4,579** | **0.523** | **0.167** |

Top-250 overlap `fin_out_n` vs `bal_live`: Jaccard **0.012**, 1,710 names unique to each.

---

## 7. v7 — validation on `B_bal_exit`

| Model | AUC A | AUC B | Δ | Precision A | Precision B | **Retention** |
|---|---|---|---|---|---|---|
| `M1_bal_only` | 0.691 | 0.674 | −0.017 | 0.508 | 0.063 | **0.12** |
| `R_bal_live` | 0.704 | 0.613 | −0.091 | 0.508 | 0.063 | **0.12** |
| `M4_pay_plus_bal` | 0.795 | 0.759 | −0.036 | 0.651 | 0.335 | 0.51 |
| `M3_pay_only` | 0.760 | 0.690 | −0.070 | 0.515 | 0.333 | 0.65 |
| `R_fin_out_n` | 0.714 | 0.640 | −0.073 | 0.291 | 0.237 | **0.81** |
| `M2_fin_only` | 0.655 | 0.615 | −0.040 | 0.291 | 0.237 | **0.81** |
| `R_count12` | 0.748 | 0.678 | −0.070 | 0.373 | 0.314 | **0.84** |
| `R_p30` | 0.613 | 0.575 | −0.038 | 0.103 | 0.088 | 0.86 |

Base rates differ by only 0.86×, so **the 0.12 is a collapse, not a denominator effect.**

---

## 8. v8 — direction and rail audit

Coverage by direction, three sample months — see `02` §3 for the full tables.
`cpty_name` out 0.999 / in 0.874 · `unq_cpty_acct_id` out 0.390 / in 0.925 ·
`cpty_fin_entity_name` out 0.368 / in 0.857.

**Institution coverage decomposition (outbound):**
`ACH 0.3584 × 1.000 + RTP_PRT 0.0102 × 0.996 + WIRE 0.0012 × 1.000 = 0.3698` vs measured 0.3700.
Of populated rows: **ACH 96.93%**, RTP_PRT 2.75%, WIRE 0.32%.
**Dollar-weighted coverage: 96.1%.**

**Wire crosstab (client-months):**
| | fin=0 | fin>0 | share with fin |
|---|---|---|---|
| No wire this month | 445,648 | 870,056 | 0.661 |
| Sent a wire | 56,025 | 580,559 | 0.912 |

Overall 74.3% of client-months carry a `fin_out_n` value.

**Rail used alone:** ach 508,380 cm → 0.847 · wire 188,621 → 0.749 · rtp 5,095 → 0.743 ·
check 84,707 → **0.000** · card 13,774 → **0.000**.

**Conditional AUC (the decisive test):**
| Population | n | base | AUC fin_out_n | AUC bal_live | scoreable |
|---|---|---|---|---|---|
| All clients | 501,046 | 0.0438 | 0.7204 | 0.7024 | 0.4193 |
| Wire users only | 117,943 | 0.0250 | **0.6763** | 0.6960 | 0.7344 |
| Non-wire clients | 383,103 | 0.0496 | **0.7430** | 0.7030 | 0.3223 |

---

## 9. v8 — ablation WITH the defect present (partly void)

Blocks `railmix`, `concentration` and `accounts` reported `n_feat = 36` — identical to the
baseline — because every column they contributed was null and dropped.

| Spec | AUC | Precision @250 | Features |
|---|---|---|---|
| `+ everything` | **0.7973** | **0.5977** | 206 |
| `+ fin_in` | 0.7969 | 0.7143 | 54 |
| `+ recurring` | 0.7964 | 0.6874 | 76 |
| `+ cpty_acct_out` | 0.7960 | 0.6949 | 54 |
| `+ cpty_in` | 0.7954 | 0.6726 | 54 |
| `+ cpty_name_out` | 0.7939 | 0.6931 | 54 |
| `+ timing` | 0.7924 | 0.6451 | 46 |
| `+ accounts` | 0.7910 | 0.6509 | 38 |
| `+ selfpay` | 0.7909 | 0.6503 | 42 |
| `+ concentration` | 0.7908 | 0.6514 | **36 ← null block** |
| `+ railmix` | 0.7908 | 0.6514 | **36 ← null block** |
| `v7_baseline` | 0.7908 | 0.6514 | 36 |
| `+ trend` | 0.7899 | 0.6023 | 108 |

---

## 10. v8b — the repair and the current ablation ★

**Repair:** 23 bounded-ratio features rebuilt additively. Mean dd coverage **0.000 → 0.3281**.
`median_dd = 0.000` for all 23. Risk-set columns 345. Wall 189s.

| Spec | AUC | sd | Features | **Precision @250** | Recall | Δ AUC | **Δ precision** | Ships |
|---|---|---|---|---|---|---|---|---|
| **`+ fin_in + cpty_acct_out`** | 0.8006 | 0.0171 | 84 | **0.7269** | 0.0465 | 0.0098 | **0.0754** | ✅ |
| `+ fin_in` | 0.7975 | 0.0172 | 60 | 0.7246 | 0.0464 | 0.0067 | 0.0731 | ✅ |
| `+ cpty_acct_out` | 0.7972 | 0.0161 | 60 | 0.7234 | 0.0463 | 0.0064 | 0.0720 | ✅ |
| `+ fin_out2` | 0.7973 | 0.0158 | 60 | 0.7149 | 0.0457 | 0.0065 | 0.0634 | ✅ |
| `+ everything (L2=20)` | **0.8031** | 0.0146 | 184 | 0.7120 | 0.0456 | 0.0124 | 0.0606 | ✅ |
| `+ fin_in + recurring + cpty_acct_out` | 0.8007 | 0.0175 | 144 | 0.7109 | 0.0455 | 0.0099 | 0.0594 | ✅ |
| `+ everything (L2=2)` | 0.8031 | 0.0146 | 184 | 0.7103 | 0.0454 | 0.0123 | 0.0589 | ✅ |
| `+ everything (L2=100)` | 0.8032 | 0.0146 | 184 | 0.7097 | 0.0454 | 0.0124 | 0.0583 | ✅ |
| `+ fin_in + recurring` | 0.7999 | 0.0176 | 120 | 0.7091 | 0.0454 | 0.0091 | 0.0577 | ✅ |
| `+ everything (L2=400)` | **0.8035** | 0.0145 | 184 | 0.7080 | 0.0453 | 0.0127 | 0.0566 | ✅ |
| `+ cpty_name_out` | 0.7946 | 0.0175 | 60 | 0.7046 | 0.0451 | 0.0038 | 0.0531 | ✅ |
| `+ recurring` | 0.7987 | 0.0172 | 96 | 0.7046 | 0.0451 | 0.0079 | 0.0531 | ✅ |
| `+ accounts` | 0.7932 | 0.0142 | 40 | 0.6783 | 0.0434 | 0.0024 | 0.0269 | ✅ |
| `+ cpty_in` | 0.7957 | 0.0178 | 60 | 0.6754 | 0.0432 | 0.0049 | 0.0240 | ✅ |
| `+ railmix` | 0.7922 | 0.0168 | 40 | 0.6583 | 0.0421 | 0.0015 | 0.0069 | ❌ |
| `+ timing` | 0.7925 | 0.0171 | 46 | 0.6531 | 0.0418 | 0.0018 | 0.0017 | ❌ |
| `+ concentration` | 0.7914 | 0.0173 | 40 | 0.6514 | 0.0417 | 0.0006 | 0.0000 | ❌ |
| `v7_baseline` | 0.7908 | 0.0174 | 36 | 0.6514 | 0.0417 | — | — | — |
| `+ selfpay` | 0.7909 | 0.0174 | 42 | 0.6503 | 0.0416 | 0.0001 | −0.0011 | ❌ |

### What the repair was worth, block by block
| Block | v8 precision | v8b precision | Δ |
|---|---|---|---|
| `+ everything` | 0.5977 | **0.7103** | **+0.1126** ← mostly trend removal |
| `+ cpty_acct_out` | 0.6949 | 0.7234 | +0.0285 |
| `+ accounts` | 0.6509 | 0.6783 | +0.0274 |
| `+ recurring` | 0.6874 | 0.7046 | +0.0172 |
| `+ cpty_name_out` | 0.6931 | 0.7046 | +0.0115 |
| `+ fin_in` | 0.7143 | 0.7246 | +0.0103 |
| `+ railmix` | 0.6514 | 0.6583 | +0.0069 |
| `+ timing` | 0.6451 | 0.6531 | +0.0080 |
| `+ cpty_in` | 0.6726 | 0.6754 | +0.0028 |
| `+ concentration` | 0.6514 | 0.6514 | 0.0000 |

### L2 sweep — was the wide spec overfitting? No
| L2 | Features | AUC | Precision |
|---|---|---|---|
| 2 | 184 | 0.8031 | 0.7103 |
| 20 | 184 | 0.8031 | 0.7120 |
| 100 | 184 | 0.8032 | 0.7097 |
| 400 | 184 | 0.8035 | 0.7080 |

**Flat across a 200-fold change.** Regularisation was not the problem. The v8 collapse was the 94
`trend` features; once they are gone the wide spec behaves normally.

### Re-keying verdict
Account key **+0.0064** Δauc / 0.7234 precision · Name key **+0.0038** / 0.7046. Coverage 0.504 vs
0.464. **Account key wins on both.**

---

## 11. v8 — competitor vs contraction (NULL RESULT)

| | |
|---|---|
| Attriters classified | 1,150 of ~7,600 |
| Stayers as baseline | 26,671 |
| Survival cut (stayer median) | **1.000 ← degenerate** |
| Ticket cut (stayer median) | 1.053 |
| CONTRACTION quadrant | 616 attriters / 13,335 stayers · lift **1.07** |
| DISPLACEMENT quadrant | 534 / 13,336 · lift **0.93** |
| Both "partners gone" quadrants | **empty** |

The axis is degenerate: essentially every baseline counterparty of a typical stayer is still being
paid by someone. See `08` for the two fixes.

---

## 12. Superseded numbers, kept for the record

| Run | Claim | Why it was wrong |
|---|---|---|
| v1 | 3,675,351 duplicate account-days | Deduped on all ~70 columns incl. rates |
| v1 | 0.9% payments-to-deposits join rate | Payments is the whole bank — denominator mistake |
| v3 | payments lead by **5 months** (`cpty_new_out` at −10) | Search started inside the baseline window |
| v4 | best lifts **1.1–5.9×**; "combining doesn't help" | Peer *levels* against change-tuned thresholds |
| v4 | `bal_live` separates at rel_m **0** | Deciles built on `bal_live` — circular |
| v5 | payments lead by **7 months** (`amt_out_rtp` at −12) | 2.6% coverage |
| v5 | "single 30.1× beats combined 16.3×" | −1 vs −12 — different months |
| v6 | `fin_out_n` is the best signal at 30.1× | Event time; 42% coverage; ranks 5th of 8 across the book |
| v6 | combining never beats the best single | True only for an unweighted count, and false even then in calendar time |
| v7 | two-tier queue is the deliverable | One list taken deeper wins on precision AND recall |
| v8 | `+ concentration` / `+ railmix` add nothing | Their columns were null and dropped — never tested |
| v8 | `+ everything` collapses on precision | It was carrying 94 trend features |
| roadmap | counterparty coverage blocked on `PAYS_CPTY` | Graph milestone; irrelevant here |
| roadmap | re-key counterparties on name | Account key wins on coverage and lift |
| roadmap | trend features are free information | They cost 11 precision points |

*Internal — PNC Treasury Management, Data Science*
