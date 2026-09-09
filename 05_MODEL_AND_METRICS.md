# 05 · Model, evaluation and metrics

---

## 1. The framing decision — calendar time, not event time

Everything before v7 measured statistics inside an **event-time cell**: how different an attriter
looks six months before it leaves, against stayers handed a pseudo-event. That answers a scientific
question and cannot be converted into an operational one. Three things go wrong in the translation
and all three are silent:

1. The stayer side is a matched pseudo-event sample, not the book.
2. Coverage gates the *ordering* claim but never the *queue*.
3. Nothing prices repeat-flagging.

**v7 onward: score every at-risk client every calendar month, rank within the month, work the
top K.** That is the object a sales team receives, and moving to it changed several conclusions.

---

## 2. Model specification

| Element | Choice |
|---|---|
| Unit | (client, calendar month), live and pre-event |
| Class | Discrete-time hazard — logistic, ridge-penalised, intercept unpenalised |
| Solver | **Pure-numpy IRLS.** No sklearn dependency on the cluster; a 50×50 normal equation is free |
| Regularisation | `L2 = 2.0` on standardised features; betas returned on the raw scale |
| Imbalance | Negatives sampled at `NEG_SAMPLE = 0.15` in **training only** |
| Prior correction | King & Zeng: `b0_corrected = b0 + ln(NEG_SAMPLE)`. Ranking is unaffected; **precision is not**, which is why it is applied |
| Evaluation set | **Full book** — no sampling, so precision is exact |
| Missing values | `ld_*` filled at 0 paired with `md_*` indicator (see `04` §1) |

```python
def logit_irls(X, y, l2=2.0, max_iter=60, tol=1e-9):
    # Newton-Raphson. X carries its own intercept at column 0 and that column
    # is NOT penalised.
```

---

## 3. Validation — rolling origin

For test month `T` and horizon `H`, training uses only rows with **`t ≤ T − H`**, so every
training label has already resolved by `T`.

```python
ORIGIN_START_OFF, MAX_ORIGINS = 18, 24      # OFFSETS from M_MIN — m_idx is ABSOLUTE
ORIGINS = range(M_MIN + ORIGIN_START_OFF, M_MAX - PRIMARY_H + 1)
# → m_idx 24307–24313, 7 folds
HORIZONS, PRIMARY_H = [1, 3, 6], 6
MIN_TRAIN_POS = 300
```

313,811 training rows (negatives sampled) · 541,128 test rows (full) · 7 folds.

---

## 4. Metrics — and which one ships

| Metric | Definition | Use |
|---|---|---|
| **AUC** | Mann-Whitney from ranks. Non-finite scores **dropped**, not imputed | Global discrimination. Report *beside* queue coverage or it is meaningless |
| **Queue coverage** | Share of at-risk rows a rule can score at all | The number no v6 table carried |
| **precision@K** | Precision of the top-K within a calendar month, pooled across folds as `Σtp / Σk` | **The ship metric** |
| **recall@K** | Share of departures caught | The trade against precision |
| **lift** | precision / base rate (~5.05% at H=6) | Readable at a low base rate |
| **Conversations** | Alerts collapsing re-flags inside `COOLDOWN_M = 3` | The honest cost figure |
| **Conversations per TP** | Cost per client saved | What a manager reads |
| **Median lead** | Event month − first flag month, of caught attriters | Competes with the incumbent's 2-month median |

### Why precision@K and not AUC
They disagreed materially. In v8 the widest specification had the **best AUC (0.7973) and the worst
precision of any shipping block (0.5977)**. AUC averages over the entire ranking; a 250-name list
lives in its top 0.3%. **Ship on the metric that matches the decision.**

```python
CAPACITY = [50, 100, 250, 500, 1000, 2500]
QUEUE_K  = 250        # the headline capacity
COOLDOWN_M = 3        # re-alert inside this window is the SAME conversation
SHIP_RULE = "d_precision at K > 0.010"
```

### Pooled, not averaged
Precision is `Σ tp / Σ alerts` across folds — **not** the mean of per-fold precisions. A mean of
ratios over months with different at-risk counts is not the precision anyone experiences.

---

## 5. Specifications compared

| Spec | What |
|---|---|
| `M1_bal_only` | Fitted on the balance dd alone — the deposit view, given a fair fit |
| `M2_fin_only` | Fitted on `fin_out_n` alone |
| `M3_pay_only` | All payment features, no balance |
| `M4_pay_plus_bal` | Everything (the v7 baseline) |
| `R_bal_live`, `R_fin_out_n` | The single features as **rules** — rank the dd directly, unscoreable rows stay unscoreable |
| `R_count12` | v6's unweighted count: how many of the twelve are firing, equal weights, ≥6 evaluable required |
| `R_p30` | The incumbent 30% rule as a monthly binary flag |

Then in v8/v8b, `M4` becomes the baseline and blocks are added one at a time.

---

## 6. The queue

A queue is a **capacity decision**, not a threshold decision. Three settings priced (on the v7
specification):

| Option | Alerts/month | Precision | Departures caught | Conv per save | Median lead |
|---|---|---|---|---|---|
| **A · Concentrated** | 250 | 65.1% | 754 (9.6%) | 1.49 | **2 months** |
| **B · Working list** | 1,000 | 55.4% | 2,308 (29.4%) | 1.65 | 2 months |
| **C · Replace incumbent** | 12,667 | 16.3% | 5,860 (74.7%) | 4.84 | **4 months** |

> **The concentrated list has the highest precision and the SHORTEST warning.** The top of the
> ranking is dominated by clients already visibly winding down. If the objective is *saving*
> relationships rather than recording them accurately, B or C is the better instrument and A is a
> supplement.

### Two tiers vs one list — settled
| Design | Alerts | TP | Precision | Recall |
|---|---|---|---|---|
| Tier 1 only (model) | 1,750 | 1,140 | 0.651 | 0.042 |
| Tier 2 only (balance, net of tier 1) | 7,000 | 3,161 | 0.452 | 0.115 |
| Two tiers combined | 8,750 | 4,301 | 0.492 | 0.157 |
| **ONE list, 1,250 deep, same model** | 8,750 | **4,579** | **0.523** | **0.167** |

One list wins on **both** axes. The two-tier proposal is withdrawn. Top-250 overlap between
`fin_out_n` and `bal_live` is Jaccard **0.012** — they find almost entirely different clients and
it *still* does not pay, because the fitted model already absorbs it.

---

## 7. Ablation protocol

```
baseline = v7 spec (36 features)
for each block: fit baseline + block, same 7 folds, same clients, same horizon
report Δ AUC and Δ precision@250
ship if Δ precision > 0.010
```

Rules:
- **One block at a time**, then small combinations, then everything.
- **Never rank on AUC.**
- **A block that moves neither metric gets deleted, not parked.**
- **Watch `n_feat`.** If a block reports the same feature count as the baseline, its columns were
  dropped by the zero-variance guard and it was never tested — this happened to three blocks in v8.

*Internal — PNC Treasury Management, Data Science*
