# Deposit Attrition EDA — Kickoff Spec

**Payment Knowledge Graph (PKG) · PNC Treasury Management · Data Science**
*Self-contained brief for starting this work in a fresh session.*

---

## 0. How to use this document

This is the data-and-decisions layer for the study described in
`PKG_Attrition_Analysis_Brief.md` (attach that document alongside this one).

The brief says **what** we are trying to learn. This document says **what data we
actually have**, **which of the brief's open assumptions have since been answered**,
and **what still needs to be resolved in EDA**.

The immediate deliverable is a **PySpark EDA notebook**. Read §7 before writing code,
and ask questions before building — anything unanswered goes on the EDA to-do list
rather than being assumed.

---

## 1. Objective, in one paragraph

Determine whether payment-network behaviour flags a departing client earlier than the
deposit balance does, and whether it can distinguish *moving to a competitor* from
*business contraction*. Before any of that, establish ground truth: what a closed or
attriting account actually looks like in the deposit data, and how well the current
30%-decline proxy corresponds to it.

---

## 2. Data assets

Two tables. Both are queried with PySpark; temporary views are fine.

### 2.1 `neo4j_payments` — payment staging table

The staging table holding all transactions destined for ingestion into the Neo4j
payment graph.

| Column | Meaning |
|---|---|
| `trans_id` | Unique. **One row per transaction** — a book-to-book transfer appears once, not twice |
| `trans_dt` | Transaction date, `YYYY-MM-DD` |
| `trans_amt` | Transaction amount |
| `mdm_id_pays` | MDM id of the paying party, if that party is a PNC customer |
| `customer_name_pays` | Name of the paying PNC customer |
| `mdm_id_receives` | MDM id of the receiving party, if that party is a PNC customer |
| `customer_name_receives` | Name of the receiving PNC customer |
| `pnc_dep_acct_pays` | Deposit account id of the paying PNC customer |
| `pnc_dep_acct_receives` | Deposit account id of the receiving PNC customer |
| `unq_cpty_acct_id` | Unique account id of the **counterparty**, used whenever a counterparty sits on either side |
| `cpty_name` | Counterparty name |
| `cpty_fin_entity_name` | Name of the counterparty's bank (any institution except PNC) |
| `payment_rail` | ACH, wire, RTP, etc. |
| `category` | Finer rail detail — e.g. `ach on_us`, `ach off_us` |

**Edge-direction rule (the only rule needed to classify a transaction):**

| `mdm_id_pays` | `mdm_id_receives` | Interpretation |
|---|---|---|
| not null | not null | **Internal (C2C)** — transaction between two PNC customers |
| null | not null | **Inbound** — external counterparty pays a PNC customer |
| not null | null | **Outbound** — PNC customer pays an external counterparty |

**Facts already established:**
- There are **no returns or reversals** in this table.
- There should be a deposit account for **every** `mdm_id` present in the payment network — verify coverage and report the gap.
- For counterparty name matching, use **exact match** against the customer name for now. No entity resolution / fuzzy matching in this pass.

### 2.2 Deposit table

Contains **all deposit accounts**, all account types. One record per account per
business day — with "unless closed" as an open question (§6).

| Column | Meaning |
|---|---|
| `acct_full_acct_id` | Account id. **Joins to `pnc_dep_acct_pays` / `pnc_dep_acct_receives`** — same format, same padding, both stored as strings |
| `edw_tda_load_dt` | Business date of the record |
| `balance` | **End-of-day balance** for that date. Negative values are possible and legitimate |
| `avg_monthly_bal_1` | Average monthly balance — **unclear whether month-to-date or prior complete month** (§6) |
| `acct_status` | Status code. **Anything other than `'C'` means the account is open** |
| `acct_status_desc` | Description of each `acct_status` code |
| `deposit_family` | Account type family |
| `opened_dt` | Account open date |
| `closed_dt` | Account close date |
| `cust_pwr_id` | Customer id; one customer may hold many accounts |
| `cust_name` | Customer name |

**Facts already established:**
- `cust_pwr_id` is expected to be **one-to-one with `mdm_id`** — verify.
- The table contains **all** account types, not only DDA/MMDA. Use all of them to understand the `acct_status` code space, but keep account-type work to a **quick checkup** — do not sink the EDA into an account-type taxonomy.
- **CD accounts may be present**, which means an account can close on maturity rather than by customer decision. That is not attrition and must be separable — verify whether the data supports the distinction.

### 2.3 Join keys

| From | To | Key |
|---|---|---|
| Payments (PNC side) | Deposits | `pnc_dep_acct_pays` / `pnc_dep_acct_receives` → `acct_full_acct_id` |
| Deposits | Customer level | `cust_pwr_id` |
| Deposits ↔ Payment graph | Customer level | `cust_pwr_id` ↔ `mdm_id` (expected 1:1) |

**ID discipline:** treat every id as a string throughout. An int64/string mismatch on
these joins fails silently and produces plausible-looking output.

---

## 3. Corrections to the analysis brief

The brief was written under assumptions that the data has since disproved. Where the
two conflict, **this document wins**.

| Brief said | Reality |
|---|---|
| No closure or dormancy flag exists; balance decline is the only proxy for departure | **A real flag exists** — `acct_status` (`'C'` = closed) plus `closed_dt`. The proxy is now testable against ground truth rather than standing in for it |
| Use the 30% rule (trailing 3-month average vs prior 6-month average) | Use it as **one candidate among several**. A better rule — or the flag itself — may come out of the data. Do not treat 30% as settled |
| Analysis scoped to a limited number of months | **Use everything available** in the agreed range (§4). Do not restrict to the months the brief happened to mention |
| Deposit panel scoped to corporate DDA / MMDA | Pull **all account types**, at least for the status-code checkup |

**The central ground-truth question:** what is the relationship between the closed-account
flag, `closed_dt`, and the behavioural proxies we could infer from balance alone? Test
several closure definitions — including the brief's 30% rule — and decide empirically
which is the right operational definition of attrition.

---

## 4. Scope

| Dimension | Decision |
|---|---|
| **Date range** | **2024-01-01 through 2026-07-31**, for both payments and deposits |
| **Grain** | **Monthly first**, for both sources. Drop to daily later only if the monthly work shows it is necessary |
| **Population** | Only counterparties connected to customers for whom we hold deposit data |
| **Analysis order** | **Account level first** (`acct_full_acct_id`), then roll up to customer level (`cust_pwr_id`) |

---

## 5. Analysis sequence

1. **Profile the deposit table at account level** — record cardinality per account-day, status code space, `deposit_family` × `acct_status` cross-tab, `opened_dt` / `closed_dt` population.
2. **Establish the closure definition** — compare the true flag against candidate behavioural proxies; quantify agreement, lead/lag, and disagreement cases.
3. **Profile the payments table** — direction mix, rail and `category` mix, counterparty coverage, name-match rate against customer names.
4. **Join coverage** — what share of payment-network `mdm_id`s have deposit accounts, and vice versa.
5. **Roll up to `cust_pwr_id`** — verify the 1:1 with `mdm_id`, and re-run the closure work at customer level (a customer with one closed account of five has not attrited).
6. Only then move on to the signal work in §7–9 of the analysis brief.

---

## 6. Open questions — the EDA to-do list

These are unresolved. Answer them in the notebook rather than assuming.

1. **Does a closed account keep appearing?** If `closed_dt` is populated for an account, do we still see daily records for it afterwards, or does it drop out of the panel?
2. **What is `avg_monthly_bal_1`?** Month-to-date, or the prior complete month? Determine this from the data (compare it against the daily `balance` series for the same account).
3. **Closure taxonomy.** How do `acct_status = 'C'`, `closed_dt`, and behavioural proxies relate? Which definition is the right one operationally?
4. **CD maturity vs. voluntary closure.** Can maturity-driven closures be distinguished from customer-initiated ones? Which status codes and account families carry them?
5. **`acct_status` code meanings.** Use `acct_status_desc` in combination with account type to build the code map.
6. **Duplicate account-days.** Are there multiple records for the same `acct_full_acct_id` on the same `edw_tda_load_dt`? If so, inspect **all** columns to find what differs, and decide a deterministic resolution rule.
7. **`cust_pwr_id` ↔ `mdm_id`.** Confirm the 1:1 relationship; report any violations in both directions.
8. **Payment-side account coverage.** Every `mdm_id` in the payment network should have a deposit account — measure and report the shortfall.
9. **Business-date calendar.** All dates are business dates. Confirm the calendar (weekends, holidays) so gaps are not misread as missing data.

---

## 7. Working preferences

- **PySpark.** Temporary views are fine and encouraged.
- **Do not use `.show()`.** Convert Spark DataFrames to pandas for display so output formatting is controllable.
- **Consolidate output.** Results are reviewed by screenshotting cell output, so keep diagnostics to **a handful of dense output cells** rather than many small ones. Anomalous cases (e.g. duplicate account-days with all columns shown) are worth their own cell; routine profiling should be batched.
- **Ask before building.** Raise questions up front; anything unanswered goes on the §6 to-do list rather than into a silent assumption.
- **Naming.** Always **Payment Knowledge Graph (PKG)** — never "Payment Knowledge Network (PKN)".

---

*Internal — PNC Treasury Management, Data Science*
