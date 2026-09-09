# 08 · Next steps

*Ordered. Items 1–2 are cheap and unblock the reporting; 3–4 are the ones that change what the
programme can claim.*

---

## Priority list

| # | Step | Why now | Owner | Effort |
|---|---|---|---|---|
| 1 | **Re-run the capacity-matched table on `+ fin_in + cpty_acct_out`** | The 0.7269 is a client-month precision. The distinct-client / conversation / lead-time translation is what a manager reads and it has not been computed | DS | Hours |
| 2 | **Check per-fold precision spread** | AUC sd ≈ 0.017 on 7 folds and every headline is pooled. If one origin carries `fin_in`, the story changes | DS | Hours |
| 3 | **Delete every block marked "no"** | `railmix`, `timing`, `concentration`, `selfpay`, `trend`. A feature that moved nothing on 7 folds against a fixed baseline will not start working later | DS | Trivial |
| 4 | **Pilot at Option B (1,000/month) on one segment for a quarter**, alongside the incumbent | The only way to learn whether an alert changes an outcome. Produces retained-balance evidence, which precision cannot | TM Sales + DS | Weeks |
| 5 | **Collect 200–300 win/loss outcomes** from relationship teams | Unblocks the competitor question and gives the programme its first outcome labels. Cheapest high-value item on the list | TM Sales / CRM | Small |
| 6 | **Add customer attributes** — segment, NAICS, size band, product holdings | Measured against `+ fin_in + cpty_acct_out` as the fixed baseline, so their lift is separable from the payment rebuild's | DS | 2–3 weeks |
| 7 | **Read the deposit columns never opened** — ECR, FTP rate, `acquisition_cd`, `rltn_pwr_id` | Pricing position and sibling-entity decline are whole categories the model is blind to, all populated today | DS | 1–2 weeks |
| 8 | **Revisit the dd construction** | 74.3% of client-months have a `fin_out_n` value, 31% have a dd. The transform costs 43 coverage points — a bigger prize than any new field | DS | 1–2 weeks |
| 9 | **Compliance and fair-lending review** | Required before any client-facing use, and better started while the model is still moving | Compliance + DS | Lead time |
| 10 | **Productionisation path** — monthly scoring job, delivery, threshold ownership | The analysis is a notebook. Nothing runs on a schedule and nothing writes an alert anywhere | DS + dev team | Medium |

---

## Detail on the ones that need a design decision

### 4 · The pilot
Recommend **Option B, 1,000 alerts a month**, not the concentrated 250 list.

| Option | Alerts/mo | Precision | Departures caught | Median lead |
|---|---|---|---|---|
| A · Concentrated | 250 | 65.1% | 9.6% | **2 months** |
| **B · Working list** | 1,000 | 55.4% | 29.4% | 2 months |
| C · Replace incumbent | 12,667 | 16.3% | 74.7% | **4 months** |

A has the highest precision and the **shortest** warning — its top ranks are clients already
visibly winding down. If the objective is *saving* relationships rather than recording them
accurately, A is a supplement, not the pilot.

Measure **retained balances**, not precision. Precision is how models are compared; retained
balance is how the business judges it, and only a pilot produces it.

### 5 · Win/loss labels
Ask for departures in the last 12 months coded as: moved to a competitor / business contraction /
M&A or dissolution / other. 200–300 is enough to test the segmentation in §3.1 of `07`. This is the
single cheapest item with the highest ceiling.

### 6 · Customer attributes — the protocol
1. Fix the baseline at `+ fin_in + cpty_acct_out` (84 features, AUC 0.8006, precision 0.7269).
2. Add attribute blocks **one at a time**, same 7 folds, same clients, same horizon.
3. Report Δ AUC **and** Δ precision@250.
4. Ship on Δ precision > 0.010.
5. Watch `n_feat` — if a block's feature count matches the baseline, its columns were dropped and
   it was never tested.

**Candidate blocks, in expected-value order:**
| Block | Fields | Note |
|---|---|---|
| Product holdings | Count and mix of TM products; lockbox, sweep, card, FX presence | Depth of relationship is the classic retention covariate and is entirely absent |
| Relationship family | `rltn_pwr_id` — is a sibling entity also declining? | 49,294 relationships behind 120,600 customers. One join |
| Pricing position | `curr_int_rate`, `earnings_credit_rate`, `ofsa_monthly_ftp_rate` vs peers | Also enables a value-weighted queue |
| Industry / size | `cust_naics_cd_val`, `ritn_naics_cd_val`, balance band | Replace balance-decile peers with NAICS × size peers — **check point-in-time first**, `segment_desc` is restated and excluded for exactly this reason |
| Acquisition | `acquisition_cd`, tenure, months since last account opened | Young relationships churn on a different clock |
| Service friction | Fee waivers, overdraft/NSF counts, complaints | Attrition usually has a cause and the cause often leaves an operational trace first |

### 8 · The dd construction question
Three things to test, cheaply, against the fixed baseline:
- A **shorter reference window** (`t−3 … t−2`) — more rows clear `CHG_MIN_OBS`.
- A **level-and-change pair** rather than a change alone — the level is always available.
- Dropping `MIN_REF` for count features and relying on `CHG_MIN_OBS` alone.

### 10 · Productionisation shape
- Monthly Spark job: refresh panels → dd → score → write the ranked list.
- Persist the score, not just the rank, so calibration can be monitored.
- The model is trained on a rolling window and **will drift**; schedule a quarterly refit and a
  monthly AUC/precision monitor against realised outcomes.
- Delivery target unresolved — Streamlit, a table for CRM ingestion, or an API. Decide before
  building.

---

## Explicitly not recommended

| Idea | Why not |
|---|---|
| Two-tier queue | One ranked list taken to the same depth beats it on precision *and* recall |
| Re-keying counterparties on normalised name | Account key wins on coverage and lift. Revisit only with a registry rebuilt on the study population |
| Trend features as specified | −0.0009 AUC, −0.0491 precision across 94 features |
| Gradient boosting, right now | Better inputs have beaten a better model class at every step. The current model is explainable line by line, which matters for compliance. Revisit after attributes land |
| Waiting for `PAYS_CPTY` | Graph milestone. Nothing here is blocked on it |

*Internal — PNC Treasury Management, Data Science*
