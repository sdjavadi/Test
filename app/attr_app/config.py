"""Settings: .streamlit/secrets.toml ([app], [impala]) override the defaults; ATTR_APP_MODE=demo|impala overrides mode."""
import os

DEFAULTS = dict(
    mode="demo",                                   # "impala" reads the tables written by the model notebook; "demo" makes synthetic clients
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
    s = dict(DEFAULTS); imp = {}
    try:
        import streamlit as st
        s.update({k: v for k, v in dict(st.secrets.get("app", {})).items()})
        imp = dict(st.secrets.get("impala", {}))
    except Exception:
        pass
    if os.getenv("ATTR_APP_MODE"): s["mode"] = os.getenv("ATTR_APP_MODE")
    if os.getenv("ATTR_BUNDLE"): s["bundle_path"] = os.getenv("ATTR_BUNDLE")
    s["impala"] = imp
    return s
