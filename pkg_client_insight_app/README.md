# PKG client insight — prototype (Streamlit)

One app, several modules, one customer page. Each module reads the Payment Knowledge Graph customer metrics for one
use case; whatever brings a user to a client, the **customer page** shows every lens that user is allowed to see.

| Page | What it shows |
|---|---|
| **Overview** | One tile per module with this month's numbers, and the clients that appear in more than one module |
| **Deposit attrition** | Signal cards: clients whose deposits are likely to leave within 6 months — chance, money at stake, why, the conversation to have |
| **Suspicious activity** | Signal cards: payment entities matching a money-laundering pattern (pass-through, funnel, split payments, sudden spike, round-trip) — tier, score, $ involved, the numbers behind it, and the ordinary explanation it could also be |
| **Payment network** | Any client's payers and payees, one hop: a network map (PNC clients are clickable — follow the money) and a flow diagram |
| **Customer** | Header with a chip per module, then one tab per lens: attrition (balance, score history, reasons, signals vs stayers, call brief, outcome form) · suspicious activity (each pattern with its chart, evidence and rule table, peer percentiles, score heat map, explanations already considered, case summary, reviewer's note) · payment network |

Cards open the customer page on the lens of the module they came from. Deep link: `/customer?c=<client ref>&lens=fraud`.

## Access (who sees what)

Set per module in `.streamlit/secrets.toml` (`[app.access]`, see `secrets.toml.example`): a list of Posit Connect user
or group names; an empty list means everyone. The app reads the signed-in user and groups from the
`RStudio-Connect-Credentials` header that Posit Connect sends. A module a user cannot open disappears everywhere — page,
customer tab, header chip, network flags.

**Suspicious activity must not reach client-facing staff** (tipping-off risk): give `fraud` to Compliance / AML and
leadership only, and confirm the groups with Compliance before any wider demo. The signals are patterns worth a review,
not evidence of wrongdoing, and every fraud page says so.

The **View as…** menu (top right of each page) previews what each audience sees — *Leadership*, *Relationship manager*,
*Compliance / AML*. It can only hide modules, never reveal one the user is not allowed.

## Data

| Module | Tables (`bdahd01p_dlcdi1_cdi_tm`) | Built by |
|---|---|---|
| Deposit attrition | `pkg_attr_app_features`, `pkg_attr_app_history` + model file `attr_model_bundle.joblib` | export cell of `pkg_attrition_v13_m4_model.ipynb` (`export_for_app_cell.py`) |
| Suspicious activity | `pkg_fraud_signals`, `pkg_fraud_history`, `pkg_fraud_evidence` | `pkg_fraud_signals_v1.ipynb` |
| Network and shared | `pkg_edge_monthly`, `pkg_cust_dim`, `pkg_cpty_dim`, `pkg_cust_metrics_monthly` | the PKG build |

After writing new tables, once in Impala: `INVALIDATE METADATA <table>;`. Re-run both notebooks monthly. Each module
shows a short message instead of failing when its tables are missing, so modules can go live one at a time.

## Setup (Posit Connect)

1. `pip install -r requirements.txt`; the team's `dbi` module must be importable (same folder or `PYTHONPATH`). Every
   query goes through `dbi.db_get_query(sql, dsn="DSN=bdpimp04-impala;", pool="root.CIB-AMG_Impala",
   conn_options={"SocketTimeout": 0})`.
2. Environment variables (no secrets file needed):
   * `ATTR_BUNDLE=/path/to/attr_model_bundle.joblib` — the attrition model
   * `PKG_APP_MODE=demo` — synthetic data, no data access (default `impala`)
   * `PKG_APP_START=fraud` — landing page (`home`, `attrition`, `fraud`, `network`, `customer`)
3. Entry point: `streamlit_app.py` (`streamlit run streamlit_app.py`).
4. Feedback (attrition outcomes, fraud reviewer notes) is appended to `feedback_log.csv` and `fraud_review_log.csv`; on
   Posit Connect point `feedback_path` / `fraud_feedback_path` to a persistent folder (the app folder is replaced on redeploy).

## Demo mode

`PKG_APP_MODE=demo` builds one synthetic world (900 clients, 20 months) that feeds every module, so the same client can be
on the attrition list, flagged for a pattern and connected to other flagged clients (a pass-through chain and funnels
that feed pass-through accounts are planted for the network view). It trains a small LightGBM at start-up and computes the
fraud signals with the notebook's rules in pandas. Start-up takes about 15 seconds.

## Code layout

```
streamlit_app.py            navigation (top bar), pages filtered by access
pkg_app/config.py           defaults; secrets [app] and environment variables override
pkg_app/core/               shared: state (source, user, access) · data (Impala queries) · demo (synthetic world)
                            ui (look, banners, header) · nav (page registry, open customer) · network (one-hop view) · fmt
pkg_app/home.py             overview page
pkg_app/customer.py         customer page: picker, header chips, one tab per lens
pkg_app/attrition/          service (scoring, list) · pages (cards, lens) · model · labels
pkg_app/fraud/              service (signals, history, evidence, review log) · pages (cards, lens) · typologies (plain words, rules)
pkg_app/network/            pages (explorer, lens)
```

**Adding a module:** a `service.py` (data, cached) and a `pages.py` with `cards_page()` and `lens(info)`; register it in
`streamlit_app.py`, `core/state.MODULES`, `core/nav.LENS_LABEL` and `customer.LENS_ORDER`. The full ego-network explorer
can replace `core/network.ego()` without touching the other modules.
