"""Settings: optional .streamlit/secrets.toml [app] overrides the defaults; ATTR_APP_MODE=demo|impala and ATTR_BUNDLE override mode and model path."""
import os

DEFAULTS = dict(
    mode="impala",                                 # "impala" reads the tables written by the model notebook; "demo" makes synthetic clients
    dbi_module="dbi",                              # in-house helper: dbi.db_get_query(sql, dsn=..., pool=..., conn_options=...) → pandas
    dbi_dsn="DSN=bdpimp04-impala;",
    dbi_pool="root.CIB-AMG_Impala",
    dbi_conn_options={"SocketTimeout": 0},
    bundle_path="attr_model_bundle.joblib",        # model exported by the notebook (export cell)
    database="bdahd01p_dlcdi1_cdi_tm",
    features_table="pkg_attr_app_features",        # money-still-here clients × month, model features (partitioned by month)
    history_table="pkg_attr_app_history",          # 18 months of balances and raw payment metrics per listed client
    edge_table="pkg_edge_monthly",                 # counterparties on demand
    cpty_table="pkg_cpty_dim", cpty_name_col="cpty_name", cpty_fi_col="fi",
    cust_dim_table="pkg_cust_dim",
    months_scored=3,                               # months scored at start-up (for "months on the list")
    list_size=1000, alpha=0.5,
    feedback_path="feedback_log.csv",
)


def settings():
    s = dict(DEFAULTS)
    try:
        import streamlit as st
        s.update({k: v for k, v in dict(st.secrets.get("app", {})).items()})
    except Exception:
        pass                                           # no secrets file: defaults apply
    if os.getenv("ATTR_APP_MODE"): s["mode"] = os.getenv("ATTR_APP_MODE")
    if os.getenv("ATTR_BUNDLE"): s["bundle_path"] = os.getenv("ATTR_BUNDLE")
    return s
