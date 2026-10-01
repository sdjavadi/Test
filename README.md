# Deposit attrition early warning — RM prototype (Streamlit)

Two tabs:

* **🚨 Signal cards** — one card per client on this month's warning list: rank, chance the money leaves within six
  months, money at stake, balance against normal, a 12-month balance sparkline, **why** (the three measures that raised
  the score most, in plain words) and **what is likely next** (urgency, typical warning time of the signals that fired,
  suggested conversation). Filters: list size, ranking (likelihood / likelihood × size / dollars), money leaving now,
  new this month, relationship risk, competitive threat, segment, minimum balance, search.
* **🔎 Customer detail** — opened from a card's *Prepare the call* button: balance history with months on the list,
  score history, a reason chart (what raises and lowers the score), every early-warning signal against the threshold of
  similar-sized clients who stayed, payment behavior (money in/out, payments, receiving days, payers, payees, banks,
  transfers to own accounts elsewhere), other entities in the same relationship, payers who stopped and where outgoing
  money goes by bank (on demand), a downloadable call brief and an outcome form (saved to a CSV, to measure the list).

## Real-time scoring

The model runs **inside the app**. At start-up (and on *Refresh data*) the app reads the latest months of client
features from Impala, scores every client with the exported LightGBM model, calibrates the probabilities and computes
exact per-client reason codes (TreeSHAP). Opening a client re-scores its months live. Changing the list size or the
ranking re-ranks instantly.

## Setup

1. **Export from the model notebook.** Paste `export_for_app_cell.py` as a cell in `pkg_attrition_v13_m4_model.ipynb`
   (before §12) and run it. It fits the final model on all labelled months and writes:
   * `attr_model_bundle.joblib` (model, calibration, feature list, thresholds, signal timings)
   * `bdahd01p_dlcdi1_cdi_tm.pkg_attr_app_features` — money-still-here clients × last 6 months, model features
   * `bdahd01p_dlcdi1_cdi_tm.pkg_attr_app_history` — 18 months of balances and payment metrics for listed clients and their relationships

   Then in Impala: `INVALIDATE METADATA` on both tables. Re-run monthly after the model notebook.
2. **App server:** `pip install -r requirements.txt`, and make sure the team's `dbi` module is importable (same folder
   or on `PYTHONPATH`). All Impala queries go through `dbi.db_get_query(sql, dsn="DSN=bdpimp04-impala;",
   pool="root.CIB-AMG_Impala", conn_options={"SocketTimeout": 0})`; connection and authentication are handled there.
   Point the app at the model: `ATTR_BUNDLE=/path/to/attr_model_bundle.joblib`, or `bundle_path` in `.streamlit/secrets.toml`.
3. **Run:** `streamlit run streamlit_app.py`

With `ATTR_APP_MODE=demo` the app runs on **synthetic clients** and trains a small demo model at start-up, so it can be
shown without data access.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | the two tabs |
| `attr_app/model.py` | loads the bundle, scores, reason codes, fired signals |
| `attr_app/data.py` | Impala queries (through `dbi`) and the synthetic demo source |
| `attr_app/labels.py` | plain-language names, display units, suggested conversations |
| `attr_app/config.py` | settings and defaults |
| `export_for_app_cell.py` | notebook cell that writes the model bundle and the two app tables |

## Notes for the prototype

* Counterparties are read on demand from the payment edge tables; the first read for a large client can take a minute.
* Queries use only validated ids and months; no free text reaches SQL.
* The outcome log is a local CSV; a shared table is the natural next step for a pilot.
