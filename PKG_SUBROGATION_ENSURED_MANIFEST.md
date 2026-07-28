# PKG — Ensured Subrogation Network
## Module Manifest v1.0

**PNC Bank · Treasury Management | Data Science**
**Module:** `pkg_subrogation_ensured.ipynb`

---

## 1. Purpose & scope

Produce **one bounded, defensible, visualisable subrogation network** from the
monthly PKG snapshots — suitable for putting in front of a stakeholder.

This module is deliberately **not** the exploratory notebook
(`pkg_subrogation_network.ipynb`). That one casts a wide net and tiers
everything it finds. This one is a **funnel**: every stage discards cases, and
the output is capped at a few hundred nodes.

| | Exploratory module | **This module** |
|---|---|---|
| Objective | Discover the candidate space | Show only defensible cases |
| Bias | Recall | **Precision** |
| Ambiguous cases | Tiered down (C_low) | **Discarded** |
| Output size | Unbounded | **Hard node cap** |
| Audience | Analyst | Stakeholder / demo |

### The honesty constraint

There is **no transaction-level ground truth for subrogation today** — no
confirmed labels from Arbitration Forums, carrier claims systems, or ACH
addenda. Therefore:

> **"Ensured" means highest structural confidence with ambiguous cases
> deliberately discarded. It does NOT mean verified.**

Precision is **unmeasured** until a label source exists. Every artifact this
module writes carries that caveat, including the machine-readable run
manifest. Do not let the word "ensured" travel into a stakeholder deck without
it.

---

## 2. Data scope

| Property | Value |
|---|---|
| Input | `../data/cust_YYYY-MM.csv` monthly snapshots |
| Schema | `source, source_name, source_naics, amount, volume, dest, dest_name, dest_naics` |
| Default window | `2025-01` … `2025-11` (11 months) |
| Graph coverage | **PNC↔PNC only** (PAYS). Counterparty tiers not yet included. |

**Scope caveat (carried from the strategy document).** Because the graph is
PNC-customer-to-PNC-customer today, this module can only surface subrogation
where **both** parties bank at PNC. When `CPTY_PAYS` / `PAYS_CPTY` are folded
in, the same gates rerun unchanged for a clean before/after comparison — the
module is designed for that rerun.

---

## 3. Design decisions

### 3.1 Direction is the signal for law firms — and it inverts the carrier logic

This is the central analytical decision in the module.

- **Carrier ↔ carrier** subrogation appears as **balanced reciprocity**: two
  carriers each subrogating against the other across a claims portfolio.
- **Carrier ↔ law firm** does **not**. Here direction carries the meaning:

| Direction | Reading | Disposition |
|---|---|---|
| **firm → insurer** | Plaintiff/recovery firm trust account remitting settlement proceeds against a WC lien or subrogation interest | **KEEP** — the recovery signal |
| **insurer → firm** | Panel defense counsel / litigation funding | **DROP** |
| **bidirectional** | Recovery counsel: carrier funds pursuit, firm remits net of contingency fee | **KEEP** |

**Why this matters:** carriers retain standing panels of defense firms and pay
them **every month**. Those relationships are *maximally* persistent. Ranking
law-firm edges on persistence alone therefore surfaces defense panels at the
top and buries the actual recoveries. The direction gate exists specifically
to prevent that inversion, and it is validated in testing (see §6).

### 3.2 Entity admission is NAICS-verified, with one exception

Strict mode (`STRICT_REQUIRE_VALID_NAICS = True`) admits an entity only on a
**valid** NAICS code. Name patterns alone are rejected — they were a
demonstrated source of false positives, because `LLP` / `PC` / `PLLC` are
generic entity-form suffixes shared with accounting and medical practices.

**The one exception:** trust-account wording (`IOLTA`, `client trust account`,
`attorney trust`) is admitted on its own. It is unambiguous, and it is
frequently the **only** available signal when NAICS is missing — which is
precisely the WC/GL settlement channel we most want to retain.

The cost of precision is made visible: `rejected_name_only_insurer` and
`rejected_name_only_law` count what strict mode threw away.

### 3.3 The streak gate carries a reasoned exemption for trust accounts

Persistence is gated on **both** active months and longest consecutive run.
But a hard streak requirement is **biased against the exact pattern this
module targets**: subrogation recoveries are lumpy — settlements close
irregularly and arrive net of contingency fees — whereas an unbroken monthly
run is the signature of a **retainer**, i.e. the defense panel just excluded.

Trust-account pairs are therefore admitted on **months-active alone**. Gapped
activity is *expected behaviour* for them, not weak evidence.

This was not a theoretical concern: in testing, a trust account active 7 of 11
months was being dropped by a streak gate of 4, and the exemption is what
recovers it.

### 3.4 Reinsurance is cut on a percentile, not a dollar figure

Reinsurance treaty settlement is *also* reciprocal carrier↔carrier, so it
passes every structural test for subrogation. The discriminator is shape:
subrogation is **many moderate** settlements; reinsurance is **few and
large**. The cut is a percentile of the observed avg-amount-per-txn
distribution (default p90), so it travels across windows and books rather than
being pinned to a dollar amount that goes stale.

### 3.5 Clearinghouses are labelled, not dropped

A node reciprocally linked to a large share of admitted insurers is an
arbitration/settlement clearinghouse (the Arbitration Forums pattern). It is
the single strongest subrogation marker in the graph, so it is **kept and
labelled** — never silently excluded. Blanket hub exclusion would also create
the AML blind spot noted in the roadmap. Its *drawn* edges are capped at render
time so one node cannot turn the picture into a hairball.

### 3.6 The network contains ensured pairs ONLY

The final graph is built from the surviving **pairs**, not as the induced
subgraph on the surviving **nodes**. This distinction is load-bearing: the
induced subgraph would pull back every edge between those nodes — including
the ones the gates just rejected for failing persistence, balance, reinsurance
shape, or direction. That would quietly undo the entire funnel and put unvetted
relationships on a chart labelled "ensured".

In testing this was a real bug: the induced-subgraph version drew 58 edges
where only 25 had passed the gates.

### 3.7 Node budget is greedy, not optimal

Pairs are scored, sorted best-first, and added until the next pair would
breach the cap. Greedy rather than globally optimal because it (a) guarantees
the cap, (b) keeps the strongest evidence, and (c) is trivial to explain when
someone asks why a given pair is or isn't on the chart. Explainability beats
optimality for a stakeholder artifact.

---

## 4. The funnel

| # | Gate | Condition |
|---|---|---|
| 1 | Load | Monthly snapshots over the window; missing files reported, not fatal |
| 2 | NAICS parse | `CODE\|DESCRIPTION` split; sentinels (`-1`, `UNKNOWN`, `******`) marked invalid, never truncated into a fake code |
| 3 | Node table | Last non-null attribute per node |
| 4 | Aggregate | `amount_total`, `volume_total`, `n_months_active` (distinct months), `max_streak` |
| 5 | **Entity gate** | NAICS-verified insurer (524x) / law firm (5411x), or trust-account wording |
| 6 | **Carrier↔carrier** | reciprocal **AND** persistent **AND** balanced **AND** not reinsurance-shaped |
| 7 | **Carrier↔law-firm** | recovery direction **AND** persistent (trust exemption) |
| 8 | Clearinghouse | Label nodes linked to ≥ 50% of admitted insurers |
| 9 | **Node budget** | Greedy best-first until `MAX_NODES` |
| 10 | Visualise | Static (matplotlib) + interactive (pyvis, self-contained HTML) |
| 11 | Persist | CSVs + machine-readable run manifest |

Every gate appends to a **drop log** reporting kept/dropped counts. That log is
printed, written to `gate_drop_log_*.csv`, and embedded in the run manifest.
**The drop log is the audit trail** — it is how you answer "why isn't X on
this chart?"

---

## 5. Configuration

All thresholds live in §0 of the notebook.

| Parameter | Default | Meaning |
|---|---|---|
| `STRICT_REQUIRE_VALID_NAICS` | `True` | Reject name-only entity matches |
| `INSURER_NAICS3` | `["524"]` | Insurance carriers & related activities |
| `LAW_FIRM_NAICS4` | `["5411"]` | Legal services |
| `MIN_MONTH_FRACTION` | `0.55` | Share of window a relationship must recur in |
| `MIN_STREAK_FRACTION` | `0.35` | Longest unbroken run required |
| `MIN_BALANCE_RATIO` | `0.30` | Real two-way movement, not a token reverse payment |
| `REINSURANCE_AVG_AMT_PCTL` | `0.90` | Percentile cut for few-and-large treaty flows |
| `RECOVERY_RATIO_HIGH` / `LOW` | `0.80` / `0.20` | Direction classification band |
| `CLEARINGHOUSE_INSURER_FRACTION` | `0.50` | Share of insurers implying a clearinghouse |
| `MAX_NODES` | `200` | Hard cap on the final network |
| `MAX_HUB_EDGES_DRAWN` | `40` | Per-clearinghouse edge cap at render time |

### Scoring weights (node budget)

Components are rank-normalised to `[0,1]` so no term dominates on raw scale:

| Component | Weight |
|---|---|
| Persistence (months / window) | 0.30 |
| Streak (consecutive run / window) | 0.20 |
| Evidence weight (rank-normalised `log1p(amount)`) | 0.20 |
| Directional clarity (balance, or recovery share) | 0.20 |
| Trust-account bonus | 0.10 |

---

## 6. Validation status

Executed end-to-end on synthetic data matching the production schema
(11 snapshots, 416 nodes), with adversarial cases planted specifically to
attack the gates.

| Test case | Expected | Result |
|---|---|---|
| Panel defense counsel — 11/11 months, insurer→firm only | Dropped despite maximum persistence | ✅ Dropped by direction gate |
| Accounting firm sharing `LLP` suffix | Rejected as law firm | ✅ Rejected |
| Medical practice sharing `PC` suffix | Rejected as law firm | ✅ Rejected |
| Trust account with **missing** NAICS | Admitted on name | ✅ Admitted |
| Trust account, 7/11 months, gapped (streak 3) | Admitted via streak exemption | ✅ Admitted |
| Arbitration clearinghouse | Labelled, kept | ✅ Labelled |
| Reinsurance-shaped pair (few, large) | Dropped | ✅ Dropped |
| Node budget forced to 6 | Cap enforced | ✅ 6 nodes, 8 best pairs kept |
| Ensured-pairs-only network | No gate-failing edges drawn | ✅ 25 edges, not 58 |

Notebook executes clean via `nbconvert` (exit 0, zero error outputs, figure
renders inline).

**What is NOT validated:** precision and recall against real subrogation.
That requires ground truth and cannot be tested synthetically. Synthetic data
confirms the gates behave as specified — **not** that the specification
correctly identifies subrogation in the real book.

---

## 7. Outputs

Written to `./ensured_output/`:

| File | Contents |
|---|---|
| `ensured_pairs_{window}.csv` | Surviving pairs with scores and components |
| `ensured_entities_{window}.csv` | Entities in the final network, with class/NAICS/flags |
| `excluded_defense_panels_{window}.csv` | Dropped defense-direction pairs — retained deliberately, since they are a legitimate TM relationship signal even though they are not subrogation |
| `gate_drop_log_{window}.csv` | Kept/dropped per gate — the audit trail |
| `run_manifest_{window}.json` | Thresholds + funnel + result shape + caveat |
| `ensured_subrogation_network.html` | Self-contained interactive network |

The run manifest makes any chart traceable to the exact configuration that
produced it.

---

## 8. Known limitations

1. **No ground truth.** Precision is unmeasured. Resolve via AF integration,
   carrier claims integration, or payment addenda/memo text if reachable
   upstream of the aggregated snapshot.
2. **Thresholds are heuristics, not calibrated cutoffs.** The
   `RECOVERY_RATIO_HIGH/LOW` band in particular determines how much lands in
   the kept set versus labelled defense. Check against real distributions.
3. **PNC↔PNC only.** Subrogation with a non-PNC counterparty is invisible
   until the counterparty tiers land.
4. **Monthly granularity.** Intra-month sequencing (demand → settlement →
   remittance) is not observable.
5. **Strict mode under-detects.** Legitimate insurers and law firms with
   missing or placeholder NAICS are excluded by design. The
   `rejected_name_only_*` counters quantify this; loosening
   `STRICT_REQUIRE_VALID_NAICS` trades precision for recall.
6. **WC and GL remain hardest.** Even with law-firm detection, these lines
   leave the least distinct structural trace and lean most on the trust-account
   signal.

---

## 9. Dependencies

`pandas`, `numpy`, `networkx`, `matplotlib`; `pyvis` optional (interactive
view degrades cleanly with a message if absent).

No GPU required — this module operates on the aggregated window, not the full
graph. Deliberately CPU/`networkx`, since the ensured set is bounded by
construction.

---

## 10. Next steps

1. **Run against real snapshots**, then tune thresholds on the printed
   distributions — do not accept the defaults.
2. **Review the drop log** before trusting the output. If a gate drops >95% of
   candidates, check whether it is doing analytical work or just misconfigured.
3. **Establish ground truth** — the single highest-value unblock for this
   module and for the Match layer in the strategy document.
4. **Rerun unchanged when counterparty data lands** for a clean before/after.
5. **Port to Streamlit** — `node_class` → node styling, `edge_kind` → edge
   styling maps directly onto st-link-analysis.

---

*Internal use — PNC Treasury Management, Data Science.*
