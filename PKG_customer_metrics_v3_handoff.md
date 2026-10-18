# PKG customer metrics v3 — handoff (2026-10-18)

Starting point for the customer-metrics v3 chat. Everything below was decided and built in the counterparty-metrics chat (Oct 2026). Database `bdahd01p_dlcdi1_cdi_tm`, all tables external parquet; monthly ones are partitioned by `month` 'YYYY-MM', with history from 2024-01 to 2026-08 (32 months).

## What exists now (new since customer metrics v2)

| Table | Notebook | Notes |
|---|---|---|
| `pkg_cpty_entity_map_v2` | `pkg_cpty_entity_map_v2.ipynb` | cpty_id, cpty_id_src, ent_l0, ent_l1, ent_l2, ent_l2_fi, rule, name_class, type_reason, name_clean, name_key, key_blocked. Typer v2 name classes are business / person / bank / unknown / numeric_name / placeholder / no_name. **The map has some duplicate cpty_ids: always reduce it to one row per id (lowest ent_l2).** |
| `pkg_fi_routing` | `pkg_fi_canonical.ipynb` | One row per 9-digit routing number. The canonical name is the routing's name in its latest month, with spellings unified within the routing only. fi_canonical is NULL for routings listed in FI_KEEP_RAW_ROUTINGS (none so far). |
| `pkg_fi_name_map` | same | fi_raw → fi_canonical, source = routing / ambiguous / no routing. Ambiguous names (FIRSTBANK, PREMIER BANK, CHOICE …) are kept raw. |
| `pkg_cpty_entity_dim` | `pkg_cpty_metrics.ipynb` | One row per ent_l2: dominant name/class/type (deterministic), n_ids, fi_dom, n_fi, lifetime $, pnc_match_n / pnc_tm_match_n (name matches PNC org customers). |
| `pkg_cpty_metrics_monthly` | same | 98 columns. Eligible (full metric set) = ≥ $10k in both directions over the trailing 12 months; slim row for every entity active this month or last. Direction is the counterparty's view (`_in` = receives from PNC customers). Includes hub_tier 0–3 (P99 / P99.9 / P99.99 of degree or strength, exact cut-offs). |
| `pkg_cpty_metrics_dictionary` | same | Definition and empty-value rule for every column. |

**Canonical FI for any counterparty id** = `pkg_fi_routing.fi_canonical` via `regexp_extract(cpty_id, '^(\d{9})-', 1)`, else `pkg_fi_name_map.fi_canonical` via the raw FI name, else the raw name. Use the monthly FI from `pkg_cpty_month` (max_by amount per id) for month-level bank metrics.

## Results worth knowing
- Counterparty run: 32 months, every month reconciled exactly to the edges, 5–17 min per month. Eligible entities settle at 16–18M a month from 2025 (≈ 99% of $); median eligible degree is 1, so `hub_med15` adds nothing (use `hub_tier`).
- Bank names: across all months, canonical names remove only 0.8–1.4% of lost-bank events per month; the big renames (Comerica→Fifth Third, Cadence/Veritex→Huntington, Webster→Santander, HTLF→UMB, Umpqua/Pacific Premier→Columbia, Ent→Wings, Dogwood→TowneBank …) are handled. The remaining bank swaps are single counterparties moving accounts (ClickPay, Cinergy Metronet, AFBA, Credibly) — real, so leave them. FI_FIXES = {ALLLY BANK → ALLY BANK}; "X ADO / A DIVISION OF Y" maps to the parent bank.
- The underlying lost-bank rate is high (~0.8 lost bank per TM customer link per month) because payments are intermittent. Attrition signals should rely on recurring-bank breaks, not month-to-month losses.

## v3 scope (agreed direction)
1. Canonical FI in all bank metrics (n_fi, n_fi_ach = fin_out_n, bank churn, fi_rec_broken).
2. Flag bank changes driven by the counterparty: the counterparty itself moved banks (ClickPay-type), not a customer decision.
3. Map v2, one row per id. Decide whether counterparties are counted at ent_l1 (v2 decision) or ent_l2 v2 (consistent with the cpty table); at minimum carry ent_l2 so the two tables join.
4. Hub exposure: share of flow and number of counterparties by hub_tier (from `pkg_cpty_metrics_monthly`).
5. Links to the cpty table: eligible counterparties, share with non-PNC banks by competitor, self-transfer bank metrics (which banks hold clients' own other accounts — attrition + TM).
6. Keep v2 identity columns and dictionary conventions; the attrition join uses customer_pwr_id (watch pwr_shared).

Design principle: new metric families (payment frequency and timing, location, network) go in **separate tables keyed by mdm_id × month**, each with its own notebook that resumes after a stop, so that adding one never forces a rebuild of the others. Suggested order after v3: frequency (daily edges) → location (customer side, per PKG_LOCATABILITY_MANIFEST_v3) → network metrics (quarterly snapshots, hubs excluded).

## Engineering rules learned (apply in v3)
- Every lookup joined to edge rows must have **one row per key** (assert it). Duplicate map ids inflated the 2024-01 totals.
- Picks must be **deterministic** (`max(struct(amount, id, …))`, not `max_by` with ties), and anything used twice must be **written to disk once**. A tie in max_by evaluated twice put entities into both halves of a split.
- Percentile cut-offs in tails: compute **exactly** (k-th largest); `approxQuantile` with 1e-4 is as wide as the P99.99 tail.
- **Exact reconciliation per month** (output $ and transactions = edges), with the offending ids named on failure.
- Resume design: a month counts as done only with `_SUCCESS` plus a configuration-signature marker; steps are written to disk; state is pruned to KEEP_STATE; staging folder → reconcile → rename. Any change to version or input fingerprint rebuilds from 2024-01, so settle inputs before long runs.
- Markdown cells: no `$` (Jupyter renders `$…$` as math). One consolidated HTML output per cell; numpy/pandas compatibility shims, including `np.object`.

## Open checks before the v3 run
- Monthly $ jumps in 2026-07 / 2026-08 ($3.39T / $2.90T vs ~$1.6–1.9T), driven by a few huge flows (e.g. "Instl Red Same Day Wire" at BNY Mellon, $187B to one customer). Decide whether this is a new feed or real.
- Source of the duplicate cpty_ids (see `pkg_cpty_metrics` §1b: rows in map v1 / cpty_dim).
- Fannie Mae split into a business entity and a bank entity (typer: name_is_fi); PNC's own entities need excluding from prospect lists.

## To start the new chat
Attach the current `pkg_customer_metrics_v2.ipynb` and say: "Customer metrics v3 — see the handoff doc."
