# 07 · Findings, and every claim that was withdrawn

---

## 1. What is settled

### 1.1 Payment behaviour beats balance monitoring at every operating point
Rolling-origin AUC **0.795 vs 0.691** at six months; precision **0.651 vs 0.508** at K=250. `M4`
dominates every rule across the entire precision–recall frontier — there is no capacity at which a
balance-only approach wins.

### 1.2 …and it is the only thing that works on clients who drain rather than close ★
On `B_bal_exit`, balance-based scoring retains **12%** of its precision (0.508 → 0.063). Payment
signals retain **81%** (`fin_out_n` 0.291 → 0.237, `R_count12` 0.373 → 0.314). Base rates differ by
only 0.86×.

**The mechanism, and it matters:** `bal_live` sums over **live legs**. A client closing nine of ten
accounts at month *t* shows a 90% balance drop at *t* and fires `A_full_exit` at *t+2*. That is
observable at *t* and is not leakage — but it means the balance signal on label A is substantially a
**closure-in-progress detector**, not a money-leaving detector. Which is exactly why it evaporates
on B, where nothing closes.

> **Say it this way:** *on clients who close, the balance largely records the exit as it happens;
> on clients who drain, only payments see anything at all.* That is the strongest form of this
> programme's thesis, and it came from the definition that was being carried as a robustness check.

### 1.3 At matched volume the model dominates the incumbent on every axis
Same 12,667 alerts a month: **5,860 clients vs 4,103** (+42.8%), **28,386 vs 30,427 conversations**,
precision **16.3% vs 11.7%**, cost per save **4.84 vs 7.42**, **same 4-month median lead**.

### 1.4 Combining signals helps — the v6 conclusion was an artefact of event time
Even an unweighted count of twelve equally-weighted votes beats the champion single feature at
every horizon (0.748 vs 0.714 at H=6) and every list length (0.373 vs 0.291 at K=250). A weighted
model beats it again by a wide margin. **Withdraw the v6 headline, do not amend it.**

### 1.5 The institution field is deterministic by rail, and the rail is ACH
`has_fin` is **1.0000 or 0.0000 for every rail**. `ACH 0.3584 + RTP_PRT 0.0102 + WIRE 0.0012 =
0.3698` against a measured 0.3700. 96.9% of populated rows are ACH. Refuted: the wire-proxy
hypothesis — `fin_out_n` performs *better* among non-wire clients (0.7430) than among wire users
(0.6763).

### 1.6 Weighted by dollars, institution coverage is 96%, not 37%
Debit card is 59.1% of outbound transactions and 1.1% of outbound dollars; wire is 0.12% of
transactions and 56.8% of dollars. **The coverage problem is a retail-transaction-count problem.
For a Treasury book measured in money, the field is nearly complete.**

### 1.7 The dd construction, not the field, is now the coverage constraint
74.3% of client-months carry a `fin_out_n` **value**; 31% carry a **dd**. The transform costs
43 points — more than any field choice available.

### 1.8 Fewer payments, not smaller ones
`count_dd` 0.946 → 0.000 while `ticket_dd` holds 0.968 → 0.616. Stayers flat at 1.00–1.01
throughout. A departing client does not negotiate its invoices down; it stops sending them through
us.

### 1.9 ACH is the sticky rail; cheque and RTP go first
At rel_m −6: cheque 0.220, RTP 0.015, wire 0.530, card 0.698, **ACH 0.736**.

### 1.10 Four signals have no detectable onset in this panel
`amt_in_internal`, `amt_out_check`, `cpty_new_out`, `fin_new_out` pin at −12 in a 12-month window
and at −18 in an 18-month window **on the same 4,734 clients**, median cohort effect **0.000**.
They are watch-list criteria, not triggers, and should be removed from the early-warning list —
which takes it from twelve signals to eight.

### 1.11 One ranked list beats a two-tier design
Same 8,750 alerts: 0.523 / 0.167 against 0.492 / 0.157. Wins on both axes.

### 1.12 The inbound institution signal is the best new feature family
`+ fin_in` lifts precision at K=250 from 0.6514 to **0.7246** for 18 features. `+ fin_in +
cpty_acct_out` reaches **0.7269** for 84.

### 1.13 Ship on precision at capacity, not AUC
They disagreed materially. The widest specification has the best AUC (0.8031–0.8035) and is beaten
on precision by four much smaller ones. AUC averages the whole ranking; a 250-name list lives in
its top 0.3%.

### 1.14 Account closure ≠ customer attrition
33,428 closures belong to customers who stayed, with a **mean peak balance of $4.18M** — higher
than either exiting group. Any account-level target would be dominated by them.

---

## 2. Every claim that was withdrawn

| Run | Claim as briefed | Why it changed |
|---|---|---|
| v1 | 3,675,351 duplicate account-days | Deduped on all ~70 columns including rates |
| v1 | 0.9% payments-to-deposits join rate | Payments is the whole bank — denominator mistake |
| v1 | `'C'` is the closure code | Sweep code, 3,182 accounts. Real codes are 07/08, ~112k |
| v3 | Payments lead by **5 months** | Lead detector searched inside its own baseline window |
| v4 | Best lifts 1.1–5.9×; combining doesn't help | Peer *levels* against change-tuned thresholds; a 0.70 cut flags half the population |
| v4 | `bal_live` separates at rel_m 0 | Deciles built on `bal_live` — circular by construction |
| v5 | Payments lead by **7 months** | Rested on a 2.6%-coverage feature |
| v5 | Single 30.1× beats combined 16.3× | Compared rel_m −1 against −12 — different months |
| v6 | **`fin_out_n` is the best signal, 30.1× lift** | Event time only. It scores 42% of the book and ranks **5th of 8** across the whole population |
| v6 | **Combining signals never helps** | True only for an unweighted count, and false even then once measured in calendar time |
| v6 | Payments lead the balance by 2 months | Event-time claim. Calendar-time replacement: 2 months at K=250, 4 months at incumbent volume |
| v7 | The two-tier queue is the deliverable | One ranked list taken to the same depth wins on precision *and* recall |
| v8 | `+ concentration` and `+ railmix` add nothing | **Their columns were null and silently dropped.** Never tested |
| v8 | `+ everything` collapses on precision | It was carrying 94 `trend` features. Remove them and it behaves normally |
| roadmap | Counterparty coverage blocked on `PAYS_CPTY` / `CptyFinEntity` | **Graph milestones.** This analysis reads the source table directly |
| roadmap | Re-key counterparties on name for 2.5× coverage | Account key wins on coverage (0.504 vs 0.464) *and* lift (+0.0064 vs +0.0038) |
| roadmap | Trend features are free information | 94 of them cost 11 precision points |
| roadmap | `fin_out_n` is a wire signal | Refuted. It is ACH, and it performs better where wires are absent |

### Three that need active correction, not quiet retirement
1. **`fin_out_n` = "banks a client pays", 30× lift.** It is *institutions a client sends ACH to*,
   at 42% coverage, ranked 5th of 8.
2. **"Combining signals doesn't help."** Withdraw outright.
3. **"Payments lead by 2 months."** Different measurement from the calendar-time lead.

---

## 3. Open questions

### 3.1 Competitor loss vs business contraction — the null result
The two-axis split (counterparty survival × ticket-hold) returned a **stayer-median survival cut
of 1.000**. Essentially every baseline counterparty of a typical stayer is still being paid by
someone, so the axis is degenerate. Both "partners gone" quadrants are empty; the surviving two sit
at lift 1.07 and 0.93. Only 1,150 of ~7,600 attriters were classifiable.

**Two fixes, both cheap:**
- Raise the liveness bar from "any other payer" to **≥2 other payers**, excluding the target.
- Restrict the baseline set to counterparties with a **low PNC payer count**. A processor tells you
  nothing; a regional supplier tells you everything. One payer-count histogram decides whether a
  cut leaves enough counterparties per client.

**And a construct not yet tried that needs no new data:** a payment to a counterparty carrying the
client's **own name** at a **named non-PNC institution**. Departing clients already route 1.7× more
of their outflow to themselves (8.2% vs 4.9%); attaching `cpty_fin_entity_name` turns that trait
into a named destination. The v8 version was too sparse to survive the dd transform (8.4%
coverage) — read as a **level** rather than a dd it may work.

### 3.2 Per-fold stability — unexamined
Seven folds, AUC sd ≈ 0.017. **Every headline number is pooled.** The per-fold precision spread has
not been looked at. If one origin carries the `fin_in` result, the conclusion changes.

### 3.3 The customer-level translation of the current best spec — not run
v8b reports precision on **client-months**. The capacity-matched table — distinct clients,
conversations, lead time — was computed on the **v7** specification. Until it is re-run, the honest
statement is a precision improvement, not a client count.

### 3.4 No live evidence that an alert changes an outcome
Everything is retrospective. Nothing has been scored forward, no alert has reached a relationship
manager, and **retained balances** — the measure the business would judge this on — cannot be
produced without a pilot.

### 3.5 The 20% of the deposit book invisible to payments
Excluded from everything and never profiled. If they differ systematically, the model has a blind
segment nobody has characterised.

### 3.6 Value weighting
Every alert counts the same. A queue ranked by **expected balance at risk** rather than probability
of departure is a different and probably more useful list, and `ofsa_monthly_ftp_rate` (0.832
populated) would let it be built.

*Internal — PNC Treasury Management, Data Science*
