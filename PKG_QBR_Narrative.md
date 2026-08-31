# Payment Knowledge Graph (PKG) — QBR Narrative
**PNC Bank · Treasury Management · Data Science**
*Prepared for quarterly business review*

---

## Where we are

The Payment Knowledge Graph is now a running monthly analytics platform rather than a
research prototype. Each month we extract the customer-to-customer payment network from
the graph database, land it on the GPU cluster, and compute a full node-, graph-, and
community-level metric set that publishes to production tables the team can query
directly. A recurring structural problem shapes all of it: a small number of very
high-degree nodes — payroll and payment processors, clearinghouses, government and tax
accounts — carry a disproportionate share of the dollars and distort nearly every metric
computed over them. We therefore compute everything at multiple levels of hub removal and
report no finding unless it holds consistently across them. That discipline has already
overturned several conclusions that turned out to be statements about payment
infrastructure rather than about customer behavior, and it is why we maintain a labeled
hub registry rather than a flat exclusion list: blanket removal both breaks the analytics
and creates blind spots exactly where illicit funnel accounts imitate legitimate
aggregators. On top of that metric layer we have built three application tracks, each
prototyped in Streamlit for stakeholder review and designed so the computation sits behind
a Python service the development team can consume when they rebuild the interface in the
bank's web framework. **Attrition:** a pipeline combining the deposit panel with graph
structure. The honest finding is that at today's graph coverage the network signal moves
alongside customer departure rather than ahead of it — but the deposit-side model that
came out of the work is a shippable early-warning capability now, and the graph modules
are built to re-run the moment counterparty data lands. **Prospecting:** counterparty
locatability, where we infer a geographic position for non-customer counterparties from
the PNC customers they transact with, and — critically — return a radius and a confidence
with every estimate rather than a bare coordinate, with an explicit *no estimate* class
instead of a fabricated one. **Fraud and AML:** funnel and pass-through detection, risk
propagation from analyst-supplied seed lists, and a subrogation-network detector that
identifies insurer-to-law-firm ecosystems from payment direction alone.

## What is next, and what stands in the way

The next build cycle is gated less by modeling capacity than by data. The single largest
unlock is the counterparty feed: it converts on-us flow into true wallet share, gives us
counterparty embeddedness and inferred firmographics for non-customers, and is the
prerequisite for prospect intelligence at any useful scale. Ahead of that we are building
a customer-similarity capability and a payment-cadence module, standing up rail-mix
metrics, and running a diagnostic that will tell us whether graph machine learning is even
worth attempting on the business-to-business subgraph. The constraints are worth stating
plainly. First, **coverage** — only a minority of the deposit book is currently visible in
the graph, and that ceiling, not model quality, is what limits graph-derived predictive
value today. Second, **population composition** — the large majority of graph nodes are
individuals, so any aggregate that does not partition on entity type is substantially a
retail statement rather than a Treasury Management one. The household findings from that
work have been handed to Retail and Wealth; our analysis is now scoped to the
business-to-business subgraph. Third, **data quality** — industry-code missingness
confounds three distinct conditions (not applicable, not enriched, and placeholder
values), which we now carry separately, with upstream fixes requested from the database
team. Fourth, **granularity** — monthly aggregation structurally rules out several
techniques, and we have retired those lines of work rather than build them badly. Finally,
**governance** — a compliance read on permissible use of inferred counterparty attributes
for non-customer prospecting is a hard precondition before any prospecting demonstration,
and we would like to open that conversation this quarter rather than at build completion.
We are also still waiting on per-month account closure data from the deposit team, which
is the missing input for a clean attrition label.
