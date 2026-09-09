# 03 · Target definition

---

## 1. Attrition is not one event

Three definitions were built and are carried throughout. They are different populations with
different signatures, and conflating them made several early runs unreadable.

| Definition | Rule | Raw | Qualified | Monthly hazard | Role |
|---|---|---|---|---|---|
| `A_full_exit` | Every account non-live, and it stays that way | 20,248 | **16,384** | **0.9115%** | **Primary** |
| `B_bal_exit` | Balance < 5% of trailing-12 median, held 3 months | 14,016 | 14,016 | 0.7798% | Validation |
| `C_p30` | 3-month avg < 0.70 × prior-6 avg | 57,551 | 57,551 | — | Incumbent benchmark |

`B ⊂ C` exactly (14,016 = 14,016). Evaluable months: **19** (31 − 12 burn-in).

### Qualification
```python
MIN_HIST_M, MIN_LIVE_BEFORE = 12, 6
```
`q_*` requires `MIN_LIVE_BEFORE` live months before the event, so a client already wound down at
panel start is not counted as an exit.

### Why `A_full_exit` is primary
- The only definition that is unambiguously a departure rather than a decline.
- **Absorbing at 99.5%** — 314 of 65,975 ever-non-live accounts came back.
- Agrees with `closed_dt` on 64,102 accounts at a 1-month median gap.

### Why `B_bal_exit` is carried
It is the harder and more commercially painful population: the client keeps the relationship open
and takes the money elsewhere. **It is also where the whole thesis is proven** — see `07`.

---

## 2. Closure mechanics

### The correction that voided run v1
`acct_status` carries **two code systems in one column**. Numeric codes come from HOGAN
(`src_system_cd = DDA`); single letters from the sweep source (`AGILETICS_SWEEP`, `SWP`). v1 used
`'C'` (3,182 accounts) as the closure marker instead of `07` and `08` (112,107 accounts).

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

| source | code | desc | class | accounts | share_zero_bal | share_closed_dt |
|---|---|---|---|---|---|---|
| HOGAN | 09 | ACTIVE-DO NOT CLOSE | open | 175,905 | 0.311 | 0.000 |
| HOGAN | 99 | ACTIVE | open | 36,489 | 0.078 | 0.002 |
| HOGAN | 08 | PURGEABLE | closed | 57,026 | 1.000 | 0.637 |
| HOGAN | 07 | CLOSED | closed | 55,081 | 1.000 | 0.580 |
| HOGAN | 03 | INACTIVE | inactive | 7,798 | 0.031 | 0.002 |
| HOGAN | 05 | DORMANT | dormant | 4,667 | 0.063 | 0.003 |
| HOGAN | 06 | ESCHEATABLE | escheat | 158 | 0.035 | 0.003 |
| HOGAN | 01 | NEW | open | 113 | 0.225 | 0.000 |
| HOGAN | 12 | IN PROCESS OF CLOSING | closing | 7 | 0.000 | 1.000 |
| SWEEP | O | (sweep open) | open | 7,336 | 0.115 | 0.000 |
| SWEEP | C | (sweep closed) | closed | 3,182 | 1.000 | 0.071 |

### Grain
```
acct_full_acct_id + sub_product_cd + edw_tda_load_dt
```
**Zero duplicate keys** at that grain, against **3,675,351** at account-day. Sweep legs are not
duplicates; account balance is the **sum** over legs.

### Other closure facts
| | |
|---|---|
| Closed in window (status-derived) | 64,554 |
| …evaluable (12+ months history) | 25,050 |
| With `closed_dt` | 65,580 |
| `closed_dt` but never non-live | 1,478 |
| Non-live but no `closed_dt` | 452 |
| Both fire, **median month gap** | 64,102 → **1.0** |
| Still live at panel end (censored) | 158,066 |
| Ever fully non-live | 65,975 |
| …later became live again | 314 (**0.5%**) |
| Non-live purely on idle sweep legs | **76 accounts** |
| Pre-2000 `closed_dt` on live accounts | ~239 (recycled numbers — ignore) |

### Account closure ≠ customer attrition
| Class | Closures | Customers | Mean peak balance |
|---|---|---|---|
| account churn, customer stayed | 33,428 | 19,108 | $4,180,879 |
| customer left entirely | 25,219 | 19,638 | $1,686,499 |
| part of a later full exit | 5,907 | 2,976 | $784,923 |

**The largest balances belong to accounts that closed while the customer stayed.** Any
account-level target would be dominated by them.

---

## 3. The dependent variable

For client *c* at calendar month *t* and horizon *H*:

```
y_H(c, t) = 1          if the event month for c falls in (t, t + H]
          = 0          otherwise
          = undefined  if t + H is outside the panel and no event occurred
```

- **Primary horizon `H = 6`.** `H = 1` and `H = 3` are computed alongside.
- **Base rate at H=6: ~5.05%.**

### Three rules that keep it honest

**Risk set.** A row exists only where the client has ≥1 live account at *t* and has not yet had the
event. 2,005,315 at-risk client-months.

**Observability.** A row whose panel ends before *t + H* with no event is **dropped, not zeroed**.
```python
observable = (t + H <= M_MAX) | (y == 1)
```

**Nothing at or after *t* enters a feature.** `dd_t` uses `value_t` against `t−6 … t−4`; the peer
median is cross-sectional in the same calendar month (available in production at *t*); the frozen
decile comes from the client's first three observed months.

---

## 4. The incumbent, measured two ways

### As a lead time (event time — how v6 measured it)
| Metric | Value |
|---|---|
| Attriters | 16,384 |
| …30% rule ever fires for | 13,555 |
| …**never fires for** | **2,829** (17%) |
| …**fires AFTER the exit** | **3,059** (19%) |
| **Median months of warning** | **2.0** |
| p25 / p75 | 0.0 / 7.0 |
| Fires on customers who never left | 43,143 |

### As a queue (calendar time — how v7 measured it)
| | Alerts/month | Alerts | Conversations | TP clients | Precision | Conv/TP | Recall | Median lead |
|---|---|---|---|---|---|---|---|---|
| Incumbent | 12,667 | 88,668 | 30,427 | 4,103 | 11.7% | 7.42 | 52.3% | 4 |
| Model (v7) at same volume | 12,667 | 88,669 | 28,386 | 5,860 | 16.3% | 4.84 | 74.7% | 4 |

**These are different measurements of the same rule and both are correct.** The first answers
"how much warning does it give when it works"; the second answers "what does it cost". Only the
second sizes a queue.

---

## 5. Definition comparison (v2 §5a)

| Definition | n | share | also A | median lead vs A | fires without A |
|---|---|---|---|---|---|
| A_full_exit | 20,248 | 0.214 | 20,248 | 0 | 0 |
| B_bal_exit | 14,016 | 0.148 | 10,953 | 0 | 3,063 |
| C_p30 | 57,551 | 0.608 | 14,408 | 1 | **43,143** |

*Internal — PNC Treasury Management, Data Science*
