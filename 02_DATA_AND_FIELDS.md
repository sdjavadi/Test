# 02 · Data, fields and artefacts

---

## 1. Source tables

| | |
|---|---|
| Payments | `dsihd01p_dsi.neo4j_payments` |
| Deposits | `dsihd01p_dsi.lap_dsi_universe_optimized` |
| Scope | `2024-01-01` … `2026-07-31` — **31 months, hard floor** |

> The payment table's name contains "neo4j" because it is the **source** the graph is built from.
> **This analysis does not use the graph.** It reads the table directly, at transaction-day grain.
> The `PAYS_CPTY` / `CptyFinEntity` ingestion milestones are graph milestones and block nothing
> here.

---

## 2. `neo4j_payments` — 12,811,995,156 rows in scope

| Column | Type | Meaning |
|---|---|---|
| `trans_id` | string | **Unique.** One row per transaction. A book transfer appears **once**, not twice |
| `trans_dt` | date | Transaction date — **day grain, this is what enables timing features** |
| `trans_amt` | double | Amount |
| `mdm_id_pays` / `_receives` | string | MDM id of payer / payee, if a PNC customer |
| `customer_name_pays` / `_receives` | string | Names on the PNC side |
| `pnc_dep_acct_pays` / `_receives` | string | Deposit account id, PNC side. 20-char zero-padded |
| `unq_cpty_acct_id` | string | Counterparty account id (either side) |
| `cpty_name` | string | Counterparty name |
| `cpty_fin_entity_name` | string | **Counterparty's bank** — any institution except PNC |
| `cpty_type`, `unq_cpty_id` | string | 0.979 populated |
| `payment_rail` | string | ACH, WIRE, CHECK, RTP_P2P, RTP_PRT, DEBIT_CARD_SIGNATURE, PCARD |
| `category` | string | Finer path, e.g. `2B.ACH_OrigViaPNC_woTPO_PAYS_CPTY` |
| `merchant_id`, `merchant_cat_cd`, `merchant_city`, `merchant_state`, `merchant_zip_cd` | string | 0.41–0.45 populated — **never used** |
| `trans_purpose` | string | 0.477 populated — **never used** |
| `originating_company` | string | 0.125 populated — **never used** |
| `zelle_recipient_token` | string | 0.037 — **never used** |
| `card_entry_mode` | string | 0.000 — empty |

**Direction rule.** Both mdm ids present → `internal_c2c`; only `_receives` → `inbound_cpty`; only
`_pays` → `outbound_cpty`.
Whole-bank mix: outbound 9.32bn, inbound 3.22bn, internal 272M.

**`category` encodes origination, not on-us.** `OrigViaPNC` vs `OrigViaNONPNC` is *who
originated*. True on-us is `internal_c2c`.

**No returns or reversals.** Negative values do occur in deposit balances.

---

## 3. Counterparty field coverage — measured, three sample months

Averaged over 2024-06, 2025-06 and 2026-06. Stable across all three, so this is **structure, not a
data incident**.

| Field | Outbound | Inbound | Inbound advantage |
|---|---|---|---|
| `cpty_name` | **0.999** | 0.874 | 0.87 |
| `unq_cpty_acct_id` | 0.390 | **0.925** | 2.37 |
| `cpty_fin_entity_name` | 0.368 | **0.857** | 2.33 |

`internal_c2c` carries none of them (0.000) by construction.

### The rail × institution result — the single most important data finding

`has_fin` is **1.0000 or 0.0000 for every rail**. Deterministic, not sparse.

**Outbound:**
| rail | transactions | share | amount ($bn) | has_fin | has_acct | has_name |
|---|---|---|---|---|---|---|
| DEBIT_CARD_SIGNATURE | 541,367,638 | 0.5905 | 26.4 | **0.0000** | 0.0000 | 1.0000 |
| ACH | 328,564,832 | 0.3584 | 898.8 | **1.0000** | 1.0000 | 0.9966 |
| RTP_P2P | 19,686,206 | 0.0215 | 5.4 | 0.0000 | 1.0000 | 1.0000 |
| RTP_PRT | 9,368,679 | 0.0102 | 10.4 | 0.9962 | 1.0000 | 1.0000 |
| CHECK | 9,114,769 | 0.0099 | 55.5 | 0.0000 | 0.0000 | 1.0000 |
| PCARD | 7,668,418 | 0.0084 | 3.0 | 0.0000 | 0.0000 | 1.0000 |
| WIRE | 1,080,527 | 0.0012 | **1,314.3** | 1.0000 | 1.0000 | 0.9999 |

**Inbound:**
| rail | transactions | share | amount ($bn) | has_fin | has_acct | has_name |
|---|---|---|---|---|---|---|
| ACH | 212,360,383 | 0.6996 | 937.9 | 1.0000 | 1.0000 | 0.9989 |
| CHECK | 37,870,209 | 0.1248 | 88.1 | 1.0000 | 1.0000 | **0.0000** |
| DEBIT_CARD_SIGNATURE | 22,848,725 | 0.0753 | 3.0 | 0.0000 | 0.0000 | 1.0000 |
| RTP_P2P | 20,495,243 | 0.0675 | 5.6 | 0.0000 | 1.0000 | 1.0000 |
| RTP_PRT | 8,632,692 | 0.0284 | 5.1 | 0.9869 | 1.0000 | 1.0000 |
| WIRE | 1,326,819 | 0.0044 | 1,463.9 | 0.9999 | 1.0000 | 1.0000 |

### Three consequences

**1. The 37% is arithmetic.**
```
ACH 0.3584 × 1.000  +  RTP_PRT 0.0102 × 0.996  +  WIRE 0.0012 × 1.000  =  0.3698
measured overall outbound has_fin                                      =  0.3700
```
The 63 missing points are 59 points of debit card plus 4 of cheque, PCARD and RTP-P2P.
**No field choice moves this.**

**2. `fin_out_n` is an ACH signal.** Of every outbound transaction carrying an institution,
**96.9% is ACH**, RTP-PRT 2.75%, wire 0.32%.

**3. Weighted by dollars, coverage is 96%, not 37%.** Debit card is 59.1% of outbound transactions
and 1.1% of outbound dollars; wire is 0.12% of transactions and 56.8% of dollars. Summing amounts
on rails where `has_fin = 1` gives **96.1% of outbound money**. For a Treasury book measured in
dollars the field is nearly complete.

**Refuted:** the hypothesis that `fin_out_n` is a wire proxy. It performs *better* among non-wire
clients (AUC 0.7430) than among wire users (0.6763), where `bal_live` beats it.

### Structurally unscoreable clients
84,707 client-months are cheque-only in a month and 13,774 card-only. **98,481 client-months
(~5% of the panel) can never carry an institution feature.**

---

## 4. Deposit table — 111,371,750 leg-day rows

| | |
|---|---|
| Accounts / legs | 229,363 / 255,313 |
| Customers (`cust_pwr_id`) | 120,600 |
| Relationships (`rltn_pwr_id`) | 49,294 |
| Load dates | 647 (2024-01-02 … 2026-07-31), 19–22 per month, no weekends |

**Grain:** `acct_full_acct_id` + `sub_product_cd` + `edw_tda_load_dt`. Null `sub_product_cd`
coalesces to `__NA__`.

### Columns currently used (`DEP_KEEP`)
`acct_full_acct_id`, `sub_product_cd`, `edw_tda_load_dt`, `balance`, `avg_monthly_bal_1`,
`acct_status`, `acct_status_desc`, `deposit_family`, `sub_product_desc`, `account_type`,
`opened_dt`, `closed_dt`, `cust_pwr_id`, `cust_name`, `rltn_pwr_id`, `data_source`,
`src_system_cd`, `segment_desc`, `market_desc`, `state`, `lob_indicator`, `cust_naics_cd_val`,
`bdh_hdfs_load_ts`, `cod_hdfs_load_ts`

`deposit_family`: NIBDDA 171,549 · IBDDA 32,344 · MMDA 20,105 · Sweep-On 3,687 · Sweep-Off 1,653 ·
Retail 20 · null 5.

**`avg_monthly_bal_1`** is the **prior complete month**, refreshed at month end. Populated on HOGAN
legs (1.000), absent on sweep (0.000). Fit against candidates: `mean_bal_prior_month` 0.013 median
abs err; `mtd_mean` 0.151; `mean_bal_month` 0.152. **Do not use as a within-month signal.**

### Columns we have NEVER read but are populated — the biggest untapped list

| Column | Population | Why it matters |
|---|---|---|
| `curr_int_rate` / `dep_curr_int_rate` | **1.000** | The rate paid. Pricing position is a first-order driver of balance movement |
| `interest_paid_mtd` / `_ytd` | 1.000 / 0.965 | Realised interest — the client's actual economics with us |
| `earnings_credit_rate` | 0.676 | ECR is the TM pricing lever |
| `ofsa_monthly_ftp_rate` / `ofsa_as_of_dt` | 0.832 | **Funds-transfer-pricing rate** — enables an *expected-value-at-risk* queue instead of a probability queue |
| `rltn_pwr_id` / `rltn_name` | 1.000 | 49,294 relationships behind 120,600 customers. **Sibling-entity decline is unexploited and one join away** |
| `ritn_naics_cd_val` | 0.990 | Relationship-level NAICS |
| `acquisition_cd` | 0.914 | How the relationship was won — classic retention covariate |
| `public_funds_flag`, `lob_indicator` | 1.000 | Segmentation axes better than balance deciles |
| `market_desc` | 0.868 | Geography |
| `gl_cc_roll_up_level_2..5_desc` | 0.868 | Cost-centre hierarchy |
| `promo_cd`, `promo_start_dt`, `promo_exp_dt` | 0.224 / 0.965 / 0.965 | Promotional pricing — a churn trigger when it expires |
| `dep_daily_int_expense`, `daily_int_expense` | 1.000 | Cost of the deposit |
| `sweep_acct_ind` | 0.068 | Sweep participation |
| `ishybrid`, `ui_pulse_product` | 1.000 | Product flags |

### Confirmed empty (0.0000)
`certificate_number`, `maturity_dt`, `request_maturity_dt`, `rate_index`, `rate_option`,
`rate_change_dt`, `bus_company_id`, `ecr_time_key`. **There are no CDs in this universe.**

---

## 5. Artefacts already on HDFS — do not rebuild

Base: `hdfs://nameservice1/user/pk36814/`

| Path | Grain | Rows | Cols | Written by |
|---|---|---|---|---|
| `attrition_v2/panel_account_month` | account × month | 5,253,585 | 20 | v2 |
| `attrition_v2/panel_customer_month` | customer × month | 2,746,664 | 13 | v2 |
| `attrition_v2/panel_pay_features` | customer × month, 21 features | 1,952,288 | 59 | v2 |
| `attrition_v2/pay_pairs/ym=…` | (customer, counterparty\|institution) | 1,384,217,026 | 5 | v2 |
| `attrition_v3/labels_customer` | one row per customer | — | — | v3 |
| `attrition_v5/labels_customer` | rebuilt **with `C_p30`** | — | — | v5/v6 |
| `attrition_v6/peer_anchor` | frozen decile per customer | 120,457 | 4 | v6 |
| `attrition_v6/labels_customer` | labels | 94,602 | 12 | v6 |
| `attrition_v7/risk_set` | **calendar-time risk set** | 2,005,315 | 52 | v7 |
| `attrition_v7/p30_month` | monthly incumbent flag | 2,746,664 | 3 | v7 |
| `attrition_v7/k_payers` | counterparty payer counts | 529,321,118 | 3 | v7 |
| `attrition_v8/pairs/ym=…` | customer × dir × keytype × key | — | — | v8 |
| `attrition_v8/rail/ym=…` | customer × dir × rail | — | — | v8 |
| `attrition_v8/selfpay/ym=…` | customer × destination institution | — | — | v8 |
| `attrition_v8/features_v8` | 74 new features | 2,883,278 | 76 | v8 |
| `attrition_v8/risk_set_v8` | merged panel | — | 368 | v8 |
| `attrition_v8/risk_set_v8b` | **after the ratio repair** | — | 345 | v8b |
| `attrition_v8/generic_names` | generic-name registry | 21,773 | 2 | v8 |

**Local CSV outputs:** `/projects/DSI/sa15474/repos/pkg/eda/attrition_v{2..8}/`
— `FINDINGS_v*.csv` plus per-block QA tables.

**Notebooks:** `repos/pkg/code/`
`pkg_attrition_eda_v2.ipynb` (panels) · `_v3` (labels) · `_v5` (dd) · `_v6` (coverage floor) ·
`_v7.ipynb` (calendar time, queue) · `_v8.ipynb` (features, rail audit) · `_v8b.ipynb` (repair,
ablation).

---

## 6. Counterparty key QA

### Under-merge (one entity split into several name keys)
| direction | mean names/account | p50 | p99 | share split |
|---|---|---|---|---|
| inbound | 1.2940 | 1 | 5 | **10.8%** |
| outbound | 1.0931 | 1 | 2 | **3.6%** |

### Over-merge (several entities collapsed into one name)
| direction | mean accounts/name | p50 | p999 | max |
|---|---|---|---|---|
| inbound | 1.2848 | 1 | 23 | **190,464** |
| outbound | 1.3314 | 1 | 24 | **3,264** |

Worst outbound offenders include `EXTRA SPACE MANAGEMENT` (3,264), `BANK OF AMERICA` (1,944),
`CHASE` (1,609), and then **retail P2P strings**: `MOM` (1,146), `JOSE RODRIGUEZ` (898),
`CHECKING` (667), `DAD` (599), `LINK` (574).

### Generic-name registry
Median-multiple rule (15× median payers, floored at 500) — **never a percentile**, which flags a
fixed fraction by construction.

| | |
|---|---|
| Distinct normalised names | 38,808,093 |
| Flagged generic | 21,773 (**0.06% of names**) |
| Share of customer-counterparty links they carry | **40.66%** |
| Top entries | CAPITAL ONE (1.44M payers), APPLE COM BILL, CHASE CREDIT CRD, NETFLIX COM, VENMO, DISCOVER, PAYPAL |

**Two problems with it as built:**
1. **Fitted on the whole bank, not the study population.** The threshold is calibrated to consumer
   volumes; the top over-merge offenders are retail P2P strings a TM client would never pay.
2. **It strips 40.7% of links.** For a corporate client `AMEX EPAYMENT` or `CAPITAL ONE` may be a
   material counterparty. This is why name-keyed features ended with *lower* coverage (0.464) than
   account-keyed (0.504).

*Internal — PNC Treasury Management, Data Science*
