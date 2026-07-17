# PKG Explorer — Application Guide
**Payment Knowledge Graph (PKG) · Treasury Management · Data Science**
*Companion to `PKG_MONTHLY_METRICS_MANIFEST.md` and `PKG_ROLES_MANIFEST.md`*

---

## 1. What this app is

PKG Explorer is an interactive window into the customer-to-customer payment
network. It is built around **cases, not filters**: instead of asking users
to explore raw metrics, the app surfaces pre-ranked customers with
auto-generated one-line stories (Spotlight), and lets any case be opened
into a full behavioral profile (Customer Deep Dive).

The app is read-only. All numbers come from two Hive tables produced by the
PKG pipeline: the **customer metrics** table (graph metrics per customer per
month per graph version) and the **customer roles** table (named behavioral
roles, stability scores, and industry peer percentiles). Nothing is computed
about a customer inside the app that isn't already in those tables — apart
from queue ranking and narrative text.

---

## 2. Global controls (top of page)

| Control | What it does |
|---|---|
| **Month** | Selects the reporting month. All Spotlight queues describe that month (and its comparison to the previous one). |
| **Analyst mode → Graph version** | Chooses which graph the metrics describe. `V0` = full network including infrastructure flow (banks, processors, payroll systems). `P99`, `P99_9`, `P99_99` progressively exclude the highest-degree/strength nodes. Default is V0; managers normally never need to change this. Rule of thumb from the roles manifest: flow and dependence stories are most honest on V0; embeddedness and dynamics on de-hubbed versions. |

---

## 3. Spotlight tab 🔦

The demo-case finder. Pick a queue; the app shows up to nine cards, each
with the customer name and ID, its 2-digit NAICS sector, a headline number,
and a one-sentence narrative generated from its metrics. **Deep dive →**
opens the customer's full profile; **🎲 Surprise me** samples a random case
from the current queue.

### The five queues

| Queue | Who appears | How it's ranked | Why it matters |
|---|---|---|---|
| **Revenue at risk** | Customers whose stable dynamics role is `bleeding` (sustained loss of payer revenue) | strength × lost-payer-amount share — dollars at risk, not percentages | The TM call list. A $40K/mo bleeder is noise; a $4M/mo bleeder is a phone call. |
| **Sustained drift** | Customers who flipped steady → bleeding this month, or changed ≥ 3 stable roles at once | Book size | Multiple simultaneous, sustained behavioral changes — the strongest drift signal in the stack. |
| **Losing to peers** | Bottom-quintile momentum within an industry whose median is flat or growing | Gap to sector median | Separates "customer is shrinking" from "their whole industry is shrinking" — a competitive-loss story, and an early churn flag. |
| **New conduits** | Customers whose stable flow role just became `conduit` (high pass-through, near-zero net) | Throughflow | Commerce turned into pass-through — an AML review teaser for the Fraud audience. |
| **Rising stars** | `expanding` customers and top-decile peer momentum | Momentum | The positive queue — growth accounts deepening their book; keeps the demo from being all doom. |

Narratives are template-generated from the underlying metrics (e.g. *"$193.8M/mo book; lost 59% of payer revenue; anchor payer changed; 7 payers gone this month"*). Before a live demo, hand-verify a few golden cases end-to-end — the templates are right most of the time, and the demo lives on the exceptions.

---

## 4. Customer Deep Dive tab 🔬

Opened from a Spotlight card, or by picking any customer from the selector
(shown as *Name (ID)*). Sections top to bottom:

**Identity strip** — customer name and ID, its five current stable role
badges (red = risk roles such as bleeding/conduit, green = healthy roles
such as steady/diversified), NAICS code, months active, and the industry
peer group used for percentiles (e.g. "peer group 412 @ naics4"). The
**role stability gauge** (0–1) summarizes how behaviorally locked-in the
customer is: near 1 = long-held roles; low despite long tenure = chronic
flux, itself a risk marker.

**Monthly flow chart** — inflow, outflow, and net flow over the full
history. Dotted vertical markers annotate every month in which a stable
role changed, labeled with the new role — this is where "*when* did the
behavior break" becomes visible.

**Role ribbon** — one horizontal band per role taxonomy (flow, dynamics,
dependence, embeddedness, hierarchy) across all months; color changes are
stabilized role transitions. The fastest way to read a customer's
behavioral history at a glance.

**Vs. industry peers** — the customer's percentile within its NAICS peer
group on six dimensions: size, connectivity, net-flow, revenue
concentration, retention, momentum. Bars past the dotted 0.5 line are
above the industry median; the momentum bar turns red in the bottom
quintile. Percentiles are precomputed over the **full** customer
population, so they remain correct even when the app runs with a row limit
(see §6).

**Inflow by payer cohort** — each month's inflow split into *retained*
vs. *new-payer* dollars (above zero), with *lost-payer* dollars (previous
month's revenue that left) below zero. The visual counterpart of the
bleeding role.

---

## 5. Data & backends

```
PKG_BACKEND=impala (default) | parquet (local dev)
PKG_METRICS_TABLE / PKG_ROLES_TABLE   Hive table names
PKG_ROW_LIMIT                          prototyping row cap (see §6)
```

The Impala backend queries the unified tables through the bank-standard
`dbi.db_get_query` helper. Every query is filtered on
`version + time_key` (or `node` for the deep dive) and column-pruned — the
app never pulls a full panel. Results are cached per (version, month), so
the first view of a month pays the query cost and subsequent interactions
are instant.

Architecture is three files: `data.py` (access), `logic.py` (queues,
narratives, series builders — no UI imports), `app.py` (UI only). At
dev-team handoff, `logic.py` + `data.py` become the FastAPI layer
essentially verbatim.

---

## 6. Prototyping row limit — read before quoting numbers

With `PKG_ROW_LIMIT = N` (default 50,000; `0` disables), each month loads
only the **top N customers by total payment strength**, with the roles
query semi-joined to the identical customer set.

What this means in practice:

- **Still correct**: everything on the Deep Dive (single-customer queries
  are never limited), and all peer percentiles (precomputed on the full
  population before loading).
- **Top-book only**: queue membership (a mid-size case below the strength
  cutoff won't surface) and the sector-median momentum in "Losing to
  peers" (median of large customers, not the whole sector).
- **Do not** present queue counts as population statistics while a limit
  is active.

---

## 7. Known behaviors & caveats

- **Terminal-heavy book**: ~76% of customers are terminal payers/payees on
  V0 — structural, not a bug: their other side lives outside PNC. Expect a
  large re-labeling when counterparty (PAYS_CPTY) data lands.
- **Burn-in**: dynamics roles need ~3 months of history; early panel months
  show mostly `newcomer` and empty drift queues.
- **First-hit latency**: the first load of each (version, month) runs the
  Impala query (seconds to ~30s at full scale); it's cached afterwards.
- **Names**: `cust_name` is displayed wherever available; the customer ID
  remains the join key everywhere.

## 8. Planned next sections (not yet built)

Pulse (network-level KPIs + auto-insights), Roles & Flows (role share
trends, transition Sankey, role heatmap with drill-down), Industry Lens
(sector league table and distributions).
