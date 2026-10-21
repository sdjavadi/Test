# Prospect Finder (first draft)

Outside companies (counterparty entities) that PNC clients pay or are paid by. You can filter them by bank, by the
industry and state of their PNC clients, by what the name says and by size, then rank them by a lens. TPPP prospects
is one of the lenses.

```
streamlit run app.py
```

Runs on Impala through `dbi` (no Spark). Needs `streamlit` ≥ 1.23 (≥ 1.35 to open a profile by ticking a row),
`pandas`, `numpy` and `plotly`.

## Files

| File | What it holds |
|---|---|
| `prospect_core.py` | configuration, SQL builders, filters, lenses and scores. It has no Streamlit code, so the dev team can wrap it in FastAPI |
| `app.py` | the Streamlit page: controls, filters, the "Who they deal with" segment, the ranked list with a profile under it, the chart, and About |
| `.streamlit/config.toml` | light theme |

## Check on the first run (top of `prospect_core.py`)

- **`ADDR`:** column names for client addresses (`mdm_id`, `state_or_province`, `load_dt`), taken from the 2026-08
  address profile. If the state filter fails, fix them here or set `ADDR["snapshot"]` to a known `load_dt` value.
- **`DEFAULT_END = "2026-06"`:** the last month before the July–August dollar jump.
- **`TPPP_TABLES`:** the lens uses the first table that exists (`pkg_tppp_prospects_v8`, then v7, then v6).
- Optional counterparty columns are used only if present: `n_cust_tm_in/out`, `n_fi`, `hub_tier`, `cpty_type`. The
  About tab lists every query with its time and row count.

## Queries

| When | Query | Expected cost |
|---|---|---|
| Window or floor changes | universe: averages per company from `pkg_cpty_metrics_monthly`, latest name and bank, name match from `pkg_cpty_entity_dim` | a few window partitions; around 150K rows at the 10-clients floor; cached 6 h |
| Once | TPPP scores | small |
| **Apply** under "Who they deal with" | `pkg_cust_pair_month_v3` (window) × `pkg_cust_dim` (TM, NAICS), plus addresses when states are chosen | the heaviest query; cached per choice |
| A row is ticked | that company's months and its TM clients | one scan of `other` in the pair table; cached |

Filtering, scoring and the chart run in pandas on the cached universe, so changing them doesn't query Impala.

## Not in this draft (next candidates)

The company's own inferred industry (naive Bayes over its clients' industries), its own location, RM names,
verdicts saved in the app, and "more like these" from seed examples. Also pending: the permissible-use review before
anyone outside the working group sees it.
