# PKG Deposit Attrition — DATA, PATHS, CODE

**Reference companion to `01_PKG_Attrition_HANDOFF.md`.**

---

## 1. Source tables

| | |
|---|---|
| Payments | `dsihd01p_dsi.neo4j_payments` |
| Deposits | `dsihd01p_dsi.lap_dsi_universe_optimized` |
| Scope | `2024-01-01` … `2026-07-31` — **31 months, hard floor** |

**Nothing before 2024 is trusted and none can be added.** This is the single constraint
that shapes the whole design: a 12-month event-aligned baseline would permanently discard
~4,200 of 16,384 attriters, which is why the analysis moved to difference-in-differences.

### 1.1 `neo4j_payments` (12,811,995,156 rows in scope)

| Column | Type | Meaning |
|---|---|---|
| `trans_id` | string | **Unique** — one row per transaction. A book transfer appears **once**, not twice |
| `trans_dt` | date `YYYY-MM-DD` | Transaction date |
| `trans_amt` | double | Amount |
| `mdm_id_pays` | string | MDM id of payer, if a PNC customer |
| `mdm_id_receives` | string | MDM id of payee, if a PNC customer |
| `customer_name_pays` / `_receives` | string | Names on the PNC side |
| `pnc_dep_acct_pays` / `_receives` | string | Deposit account id, PNC side. 20-char zero-padded |
| `unq_cpty_acct_id` | string | Counterparty account id (either side) |
| `cpty_name` | string | Counterparty name |
| `cpty_fin_entity_name` | string | **Counterparty's bank** — any institution except PNC |
| `payment_rail` | string | ACH, WIRE, CHECK, RTP_P2P, RTP_PRT, DEBIT_CARD_SIGNATURE, PCARD |
| `category` | string | Finer path, e.g. `2B.ACH_OrigViaPNC_woTPO_PAYS_CPTY` |

**Direction rule.** Both mdm ids present → `internal_c2c`; only `_receives` →
`inbound_cpty`; only `_pays` → `outbound_cpty`.

Whole-bank mix: outbound 9.32B txns, inbound 3.22B, internal 272M.
Field coverage: `cpty_name` 0.999 out / 0.876 in; `unq_cpty_acct_id` 0.395 / 0.927;
`cpty_fin_entity_name` 0.374 / 0.863.

**`category` encodes origination, not on-us.** `OrigViaPNC` vs `OrigViaNONPNC` is *who
originated*. True on-us is `internal_c2c`. The feature is named `share_out_origpnc`
deliberately.

**No returns or reversals.** Negative values do occur in deposit balances.

### 1.2 Deposit table (111,371,750 leg-day rows in scope)

| | |
|---|---|
| Accounts | 229,363 |
| Account-legs | 255,313 |
| Customers (`cust_pwr_id`) | 120,600 |
| Relationships (`rltn_pwr_id`) | 49,294 |
| Load dates | 647 (2024-01-02 … 2026-07-31) |

**Grain: `acct_full_acct_id` + `sub_product_cd` + `edw_tda_load_dt`.** Zero duplicate keys.
At account-day: 3,675,351 duplicates. Sweep legs are **not** duplicates.

Analytical columns (`DEP_KEEP`), ~70 in source, these 11 + 5 optional are used:

| Column | Type | Notes |
|---|---|---|
| `acct_full_acct_id` | string | Joins to `pnc_dep_acct_*`. 20-char zero-padded, both sides |
| `sub_product_cd` | string | **Part of the key.** Null → coalesce to `__NA__` |
| `edw_tda_load_dt` | date | Business date |
| `balance` | double | End-of-day. Negatives legitimate |
| `avg_monthly_bal_1` | double | **Prior complete month**, refreshed at month end. HOGAN only |
| `acct_status` | string | **Two code systems — see §2** |
| `acct_status_desc` | string | Description |
| `deposit_family` | string | NIBDDA 171,549 · IBDDA 32,344 · MMDA 20,105 · Sweep-On 3,687 · Sweep-Off 1,653 · Retail 20 · null 5 |
| `sub_product_desc`, `account_type` | string | Leg descriptions |
| `opened_dt`, `closed_dt` | date | `closed_dt` populated on ~58% of status-07 rows |
| `cust_pwr_id`, `cust_name` | string | Customer |
| `rltn_pwr_id` | string | **Relationship** — 49,294 vs 120,600 customers. Candidate alternative unit |
| `data_source` | string | `HOGAN` \| `AGILETICS_SWEEP` |
| `src_system_cd` | string | `DDA` \| `SWP` |
| `segment_desc`, `market_desc`, `state`, `lob_indicator`, `cust_naics_cd_val` | string | Segmentation |
| `bdh_hdfs_load_ts`, `cod_hdfs_load_ts` | ts | Dedup tiebreak |

Present but **deliberately unused**: `maturity_dt`, `request_maturity_dt`,
`certificate_number` (no CDs in this universe — all null); rates, promo codes,
`interest_paid_*`, `mnth_end_flg`, `holiday_flg`.

---

## 2. The status taxonomy — the correction that drives everything

Numeric codes come from HOGAN; single letters from the sweep source. **Different
vocabularies in one column.**

```python
STATUS_CLASS = {
    "01": "open",     # NEW                      113 accts
    "99": "open",     # ACTIVE                36,489
    "09": "open",     # ACTIVE-DO NOT CLOSE  175,905
    "O":  "open",     # sweep open             7,336
    "03": "inactive", # INACTIVE               7,798
    "05": "dormant",  # DORMANT                4,667
    "06": "escheat",  # ESCHEATABLE              158
    "12": "closing",  # IN PROCESS OF CLOSING      7
    "07": "closed",   # CLOSED                55,081
    "08": "closed",   # PURGEABLE             57,026
    "C":  "closed",   # sweep closed           3,182
}
CLOSED_CLASSES = {"closed"}
LIVE_CLASSES   = {"open","inactive","dormant","escheat","closing"}
```

- **08 is post-07 retention**, not a distinct reason: mean days since `closed_dt` 264.7 vs
  98.9 for 07, both at `share_zero_bal = 1.000`.
- **Sweep `C` is a LEG state, not an account state.** Only **76 accounts** are ever judged
  non-live purely because sweep legs are idle.
- Pre-2000 `closed_dt` on live accounts: only ~239 accounts. Recycled numbers. Ignore.

---

## 3. Labels

```python
MIN_HIST_M, MIN_LIVE_BEFORE   = 12, 6
BAL_EXIT_FRAC, BAL_EXIT_HOLD  = 0.05, 3
P30_DROP                      = 0.70
```

| Definition | Rule | Raw | Qualified | Monthly hazard |
|---|---|---|---|---|
| `A_full_exit` | every account non-live, and it stays that way | 20,248 | **16,384** | **0.9115%** |
| `B_bal_exit` | balance < 5% of trailing-12 median, held 3 months | 14,016 | 14,016 | 0.7798% |
| `C_p30` | avg3 < 0.70 × prior-6 avg — **the incumbent** | 57,551 | 57,551 | — |

`q_*` = qualified: requires `MIN_LIVE_BEFORE` live months before the event, so a customer
already wound down at panel start is not counted as an exit.

**Evaluable months: 19** (31 − 12 burn-in).

---

## 4. The measurement construction (v5/v6)

```python
chg_t = value_t / mean(value over t-6 .. t-4)      # needs 2+ obs, ref > 1.0
dd_t  = chg_t / median(chg_t across the customer's FROZEN peer group, same calendar month)
```

- **Frozen peer group**: 10 balance deciles assigned from the customer's **first 3
  observed months**, never recomputed. (Monthly recomputation makes a shrinking customer
  slide down the deciles alongside its own decline.)
- `median_dd = 1.000` for every covered feature — the sanity check.
- `net_flow` is in `NO_RATIO` (can be negative, ratio meaningless) — read on **sign**.
- `cpty_new_out` / `fin_new_out` read on **rate-of-any**, not dd.
- New-entity features blanked for 2024-01…03 (in the first months everything is "new").
- **Coverage floor `MIN_COVERAGE = 0.25`** for the ordering claim.

### Feature list (21)

`bal_live` · `amt_out` · `amt_in` · `net_flow` · `n_out` · `n_in` ·
`avg_ticket_out` · `avg_ticket_in` · `amt_out_{ach,wire,check,card,rtp,other}` ·
`amt_out_internal` · `amt_in_internal` · `cpty_out_n` · `fin_out_n` ·
`cpty_new_out` · `fin_new_out`

---

## 5. Paths

### Notebooks — `repos/pkg/code/`

| File | Role |
|---|---|
| `pkg_attrition_eda.ipynb` (v1) | superseded — wrong closure code, wrong grain |
| `pkg_attrition_eda_v2.ipynb` | **builds the panels.** Status taxonomy, leg grain, payments features |
| `pkg_attrition_eda_v3.ipynb` | **builds the labels.** Sparsity audit, first operating points |
| `pkg_attrition_eda_v4.ipynb` | superseded — peer *levels*, wrong thresholds |
| `pkg_attrition_eda_v5.ipynb` | difference-in-differences |
| `pkg_attrition_eda_v6.ipynb` | **current.** Coverage floor, per-month single vs combined |

### HDFS parquet

| Path | Grain | Written by |
|---|---|---|
| `hdfs://nameservice1/user/pk36814/attrition_v2/panel_account_month` | account × month | v2 |
| `…/attrition_v2/panel_customer_month` | `cust_pwr_id` × month | v2 |
| `…/attrition_v2/panel_pay_features` | `cust_pwr_id` × month, 21 features | v2 |
| `…/attrition_v2/pay_features/ym=…` | per-month partitions | v2 |
| `…/attrition_v2/pay_pairs/ym=…` | distinct (customer, counterparty\|institution) | v2 |
| `…/attrition_v3/labels_customer` | one row per customer | v3 |
| `…/attrition_v5/labels_customer` | rebuilt **with `C_p30`** | v5/v6 |
| `…/attrition_v6/peer_anchor` | frozen decile per customer | v6 |

**v6 reads v2's panels and v3's labels and rebuilds nothing.** Runs in minutes.

### Local CSV — `/projects/DSI/sa15474/repos/pkg/eda/attrition_v{2..6}/`

`FINDINGS_v*.csv` plus per-block QA tables. v6 also writes
**`PKG_Attrition_Signals.html`** — self-contained, inline SVG, no CDN, opens offline.

---

## 6. Engineering traps — every one of these cost a run

| Trap | Symptom | Fix |
|---|---|---|
| **HDFS URI in `pathlib.Path`** | `Permission denied … inode="/"` | `Path` collapses `hdfs://host/p` → `hdfs:/host/p`. Keep local `Path` and HDFS **string** as separate config vars |
| **Arrow + `decimal(15,0)`** | `module 'numpy' has no attribute 'object0'` | `spark.sql.execution.arrow.pyspark.enabled=false`. Collected frames are ≤60 rows |
| **Fanned-out account→customer map** | 241,499 rows for 229,363 accounts; ~5% of customers' payment volume silently inflated | One row per account, latest link wins, **assert** |
| **Single unfiltered payments job** | SparkContext killed | One Spark job **per month**, partitioned, skipped on re-run. ~100–115M sides/month, 60–130 s each |
| **`np.concatenate(...) or default`** | `truth value of an array is ambiguous` | Filter empty arrays explicitly |
| **Spark pivot naming** | `Column 'fin_out_new_out' does not exist` | Output is `{pivotValue}_{aggAlias}` → `fin_new_out` |
| **Duplicate keys in a `kv()` dict** | Silently prints only the last value | Distinct labels. *Still present in v6 §5h* |
| **`rowsBetween` on months** | Windows shift where a month is missing | `rangeBetween` on an integer `m_idx` |
| **`F.avg(boolean)`** | `avg requires numeric, not boolean` | `.cast("double")` |
| **`countDistinct` ignores NULL** | A column null on one row and populated on another reads as identical | `coalesce(col, "<NULL>")` first |
| **`element_at` with a Column index** | Version-fragile | `F.expr("element_at(arr, cast(size(arr)/2 as int)+1)")` |
| **int64/string id mismatch** | Silent, plausible-looking output | Every id is a string end to end |
| **Referencing a column a prior notebook never persisted** | `AnalysisException: m_C_p30` | v3 dropped `C_p30`; v5 rebuilds it |

---

## 7. Standard config block (v6)

```python
DATE_START, DATE_END          = "2024-01-01", "2026-07-31"
CHG_LAG_FAR, CHG_LAG_NEAR     = -6, -4        # reference window
CHG_MIN_OBS, MIN_REF          = 2, 1.0
PEER_DECILES, PEER_ANCHOR_M   = 10, 3         # FROZEN
PEER_MIN_N, PEER_USE_SEG      = 50, False
EVENT_PRE, EVENT_POST         = 12, 3
SEARCH_FROM                   = -12           # dd has no baseline to search inside
MIN_CELL_N, MIN_COVERAGE      = 200, 0.25
SEP_LEVEL, SEP_RATE, HOLD     = 0.15, 0.05, 2
BURN_IN_YM                    = ["2024-01","2024-02","2024-03"]
OP_THRESH   = [.95,.90,.85,.80,.70,.60,.50,.40,.30,.20,.10]
TARGET_PREC, MIN_RECALL, CMP_RECALL = 0.25, 0.20, 0.20
SCORE_THRESH = 0.70
```

---

## 8. Spark session

```python
spark = (SparkSession.builder.appName("pkg_attrition_eda_vN")
         .config("spark.sql.shuffle.partitions", "400")
         .config("spark.sql.execution.arrow.pyspark.enabled", "false")
         .enableHiveSupport().getOrCreate())
```

Parcel `DeprecationWarning` / `ResourceWarning` are suppressed so tables are
screenshot-able. Nothing else is.

*Internal — PNC Treasury Management, Data Science*
