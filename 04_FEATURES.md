# 04 · Features (independent variables)

---

## 1. The transform — everything is relative

No feature is used at its raw level.

```python
chg(t) = value(t) / mean( value over t-6 … t-4 )      # needs 2+ obs, reference > 1.0
dd(t)  = chg(t) / median( chg(t) across the client's FROZEN peer group, same calendar month )
```

**Sanity check: `median_dd = 1.000` for every covered feature.** It holds.

### Configuration
```python
CHG_LAG_FAR, CHG_LAG_NEAR = -6, -4
CHG_MIN_OBS, MIN_REF      = 2, 1.0
PEER_DECILES, PEER_ANCHOR_M, PEER_MIN_N = 10, 3, 50
PEER_USE_SEG = False          # segment_desc is restated → leakage
DD_CLIP = (0.01, 100.0)       # then log
```

### Two design choices that matter more than the formula
1. **The reference window is `t−6 … t−4`, not a 12-month event-aligned pre-window.** The latter
   would permanently discard ~4,200 of 16,384 attriters against the 2024 floor and is not
   computable at scoring time.
2. **Peer deciles are frozen** from the client's first three observed months. Recomputed monthly, a
   shrinking client slides down the deciles alongside its own decline.

### The additive variant — REQUIRED for bounded ratios
`MIN_REF > 1.0` is correct for counts and amounts and **impossible for a share**. Every ratio
feature in v8 came back null on every row. The repair:

```python
chg(t) = value(t) - mean( value over t-6 … t-4 )      # NO MIN_REF gate
dd(t)  = chg(t) - median( chg(t) over peers )
# sanity check flips to median_dd = 0.000
```

Applies to anything ending `_share`, `_retention`, `_ratio`, plus `conc_top1/3`,
`railmix_shift`, `railmix_rails_collapsed`, `tim_day_shift`, `tim_day_sd`. **23 features.**
Mean dd coverage moved 0.000 → 0.328.

### Model encoding
Every `dd` enters the model as **two columns**:
```python
ld_{f} = log(clip(dd, 0.01, 100))   # NaN → 0  (moving exactly with peers)
md_{f} = 1.0 if dd is null else 0.0 # the missingness indicator
```
Missingness is informative. A rule leaves the row unscoreable; a model scores it and knows it did.
This is why fitted models reach **queue coverage 1.000** and single-feature rules do not.

---

## 2. Base feature set (v2, 21 features + balance)

`bal_live` · `amt_out` · `amt_in` · `net_flow` · `n_out` · `n_in` · `avg_ticket_out` ·
`avg_ticket_in` · `amt_out_{ach,wire,check,card,rtp,other}` · `amt_out_internal` ·
`amt_in_internal` · `cpty_out_n` · `fin_out_n` · `cpty_new_out` · `fin_new_out`

### Read rules
| Feature | Read on |
|---|---|
| Most | `dd` (median) |
| `net_flow` | **sign** — in `NO_RATIO`, can be negative so a ratio is meaningless |
| `cpty_new_out`, `fin_new_out` | **rate-of-any**, not dd |
| New-entity features | blanked for 2024-01…03 (everything is "new" in the first months) |

### The twelve signals, as v6 ranked them (event time)
| # | Signal | Feature | Separates | Best lift | at | Recall | dd coverage |
|---|---|---|---|---|---|---|---|
| 12 | Banks they pay drop away | `fin_out_n` | −5 | **30.1×** | −1 | 0.086 | **0.347** |
| 11 | Relationship list shrinks | `cpty_out_n` | −6 | 23.8× | −1 | 0.123 | 0.406 |
| 9 | What monitoring sees | `bal_live` | −5 | 17.7× | −1 | 0.572 | 0.736 |
| 8 | Inbound activity thins | `n_in` | −7 | 17.0× | −1 | 0.570 | 0.457 |
| 7 | Fewer payments | `n_out` | −7 | 15.7× | −1 | 0.385 | 0.474 |
| 6 | Spending through us falls | `amt_out` | −7 | 5.9× | −1 | 0.529 | 0.520 |
| 10 | Payments to PNC customers fall | `amt_out_internal` | −11 | 4.3× | −1 | 0.770 | 0.378 |
| 5 | Their customers stop paying | `amt_in_internal` | −12 | 3.6× | −1 | 0.854 | 0.324 |
| 3 | Cheque | `amt_out_check` | −12 | 3.6× | −1 | 0.868 | 0.276 |
| 4 | Net flow turns | `net_flow` | −7 | 1.6× | −1 | 0.702 | 0.684 |
| 1 | No new trading partners | `cpty_new_out` | −12 | 1.3× | −4 | 0.794 | 0.649 |
| 2 | No new unfamiliar banks | `fin_new_out` | −12 | 1.1× | −4 | 0.885 | 0.649 |

**Read the pattern:** counts of relationships carry **precision**; dollar volumes carry **recall**.

> ⚠ **These lifts are event-time and are not the ranking that matters.** In calendar time
> `fin_out_n` ranks 5th of 8. See `05` and `07`.

---

## 3. v8 feature families (74 features, built from the transaction table)

All monthly, keyed on `(cust_pwr_id, m_idx)`, merged into the risk set with a plain join.

| Family | Features | Claim | Ships? |
|---|---|---|---|
| **`fin_in`** | `fin_in_n_k`, `_lost_k`, `_new_k`, `_retention`, `_rec_active`, `_rec_broken`, `_rec_broken_share`, `_rec_broken_amt_share` (+ md) | The institution signal on the direction where the field is 86% populated instead of 37% | **YES — best block** |
| **`cpty_acct_out`** | Same shape, outbound counterparties on the v2 account key | "The relationship list shrinks", measured directly | **YES** |
| `fin_out2` | Same shape, outbound institutions rebuilt | Rebuild of `fin_out_n` with churn terms | YES |
| `cpty_name_out` | Same shape on a normalised name key | Tests the re-keying proposal | Yes but loses to account key |
| `cpty_in` | Inbound counterparty count and churn | "Their customers stop paying them" without the 29%-covered proxy | Yes, weak |
| `recurring` | `rec_active`, `rec_broken`, `rec_broken_amt`, `rec_active_amt` + shares, per direction × key type | A standing monthly payment that stopped | Yes |
| `accounts` | `acc_live_ratio`, `acc_closed_share` | The model sees account closure only through the balance | Yes, weak |
| `railmix` | `railmix_shift` (total-variation distance from own baseline), `railmix_rails_collapsed` | Turns 6 sparse rail columns into 1 dense one | **NO** |
| `concentration` | `conc_top1`, `conc_top3` | Flow narrowing to fewer partners | **NO** |
| `timing` | `tim_day_mean`, `_sd`, `_shift`, `_days_active`, `_day_last` | Paying later before paying less | **NO** |
| `selfpay` | `selfpay_amt`, `_fin_n`, `_txn` | Payments to own name at a non-PNC bank | **NO** — 8.4% coverage |
| `trend` | Each signal minus its own trailing 3-month mean, 94 features | v7 read every signal as a level | **NO — actively harmful** |

### Definitions worth carrying

**Recurring series.** A (client, direction, keytype, key) is *standing* at *t* if present in ≥4 of
the last 6 months. *Broken* if standing and absent at *t*.
```python
REC_WINDOW, REC_MIN_HITS = 6, 4
PAIR_TOP_N = 200   # counterparties kept per client-month by amount
```
Detected from **monthly presence**, not intra-month dates. A twice-monthly payment dropping to
monthly is invisible — a known limitation, not a defect.

**Self-payment.** Outbound to a counterparty whose normalised name matches the client's own, where
`cpty_fin_entity_name` is populated (i.e. non-PNC). Matching is **exact-after-normalise or shared
8-character prefix** — Spark has no `jaro_winkler` builtin. Deliberately conservative: it
under-detects rather than inventing matches.

**Name normalisation.** Upper → strip non-alphanumeric → collapse whitespace → drop legal suffixes
(`INC LLC LTD CORP CO LP LLP PLC PC PA TRUST THE`) → generic-registry exclusion.

---

## 4. Feature coverage (v8, before the dd transform)

| Feature | Value coverage |
|---|---|
| `acc_closed_share` | 0.9551 |
| `acc_live_ratio` | 0.8767 |
| `fin_in_n_k` | **0.5482** |
| `railmix_shift` | 0.5225 |
| `cptya_out_n_k` | 0.5036 |
| `fin_out2_n_k` | 0.5035 |
| `cptyn_out_n_k` | 0.4639 |
| `conc_top1` / `tim_day_mean` | 0.4639 |
| `cptyn_in_n_k` | 0.4543 |
| `tim_day_sd` | 0.3481 |
| `selfpay_amt` | **0.0843** |

**After the dd transform these fall to ~0.27–0.35.** That is the real coverage constraint now:
**74.3% of client-months carry a `fin_out_n` value; only 31% carry a `dd`.** The transform costs
43 points — far more than any field choice.

---

## 5. Deliberately excluded

| Variable | Why |
|---|---|
| `segment_desc` | **Leakage.** Restated retrospectively — restatement lift 21.7 at rel_m −1. Out of the peer grouping and every feature set |
| `avg_monthly_bal_1` as a within-month signal | Prior complete month, refreshed at month end, absent on sweep legs |
| `maturity_dt`, `certificate_number`, rates, promo codes | Empty or out of scope at the time; **several are populated and worth revisiting — see `02` §4** |
| Customer attributes — NAICS, size, product holdings | **Held back on purpose** so their lift is measurable against a fixed baseline |
| Graph structural features | Needs counterparty edges in Neo4j. Previously tested and added nothing at 12% book coverage |

---

## 6. Event-time findings that survived every rebuild

These are descriptive, not model inputs, but they are the best available narrative.

**Which rail goes first** — dd at rel_m −6:
cheque **0.220** · RTP **0.015** · wire 0.530 · card 0.698 · **ACH 0.736 (sticky)**

**Fewer payments, not smaller ones** — stayers flat at 1.00–1.01 throughout:
| rel_m | count_dd | ticket_dd | amount_dd |
|---|---|---|---|
| −12 | 0.946 | 0.968 | 0.930 |
| −6 | 0.816 | 0.943 | 0.769 |
| −3 | 0.547 | 0.850 | 0.430 |
| −1 | 0.191 | 0.616 | 0.073 |
| 0 | 0.000 | 0.186 | 0.000 |

**Net flow turns** — share running negative, the only monotone rate signal:
attriters 0.474 → 0.702 against a stayer line pinned at 0.440–0.456.

**Four signals have no detectable onset.** `amt_in_internal`, `amt_out_check`, `cpty_new_out`,
`fin_new_out` sit at exactly −12 in a 12-month window and exactly **−18 in an 18-month window on
the same 4,734 clients**, median cohort effect **0.000**. They are watch-list criteria, not
triggers, and should stop being counted among the early-warning signals.

**The standing marker.** A full year out, attriters are already lighter than size peers:
`n_out` median 0.600 vs 1.143, `amt_out` 0.591 vs 1.080. Best marker lift `bal_live` **1.61×** on
"below half your peers". No onset date → watch-list criterion.

*Internal — PNC Treasury Management, Data Science*
