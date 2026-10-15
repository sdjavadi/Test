"""Settings. Defaults below; optional .streamlit/secrets.toml [app] overrides them; environment variables override both:
PKG_APP_MODE (or ATTR_APP_MODE) = impala | demo, ATTR_BUNDLE = path to the attrition model file."""
import os

DEFAULTS = dict(
    mode="impala",                                 # "impala" reads the tables built by the notebooks; "demo" uses synthetic clients
    # ── Impala through the in-house helper: dbi.db_get_query(sql, dsn=..., pool=..., conn_options=...) → pandas ──
    dbi_module="dbi", dbi_dsn="DSN=bdpimp04-impala;", dbi_pool="root.CIB-AMG_Impala", dbi_conn_options={"SocketTimeout": 0},
    database="bdahd01p_dlcdi1_cdi_tm",
    # ── shared PKG tables ──
    edge_table="pkg_edge_monthly", cpty_table="pkg_cpty_dim", cpty_name_col="cpty_name", cpty_fi_col="fi",
    cust_dim_table="pkg_cust_dim", metrics_table="pkg_cust_metrics_monthly",
    # ── attrition (written by the export cell of pkg_attrition_v13_m4_model.ipynb) ──
    bundle_path="attr_model_bundle.joblib",
    features_table="pkg_attr_app_features", history_table="pkg_attr_app_history",
    months_scored=3, list_size=1000, alpha=0.5, feedback_path="feedback_log.csv",
    # ── suspicious activity (written by pkg_fraud_signals_v1.ipynb) ──
    fraud_signals_table="pkg_fraud_signals", fraud_history_table="pkg_fraud_history", fraud_evidence_table="pkg_fraud_evidence",
    fraud_feedback_path="fraud_review_log.csv",
    # ── who sees what: module → list of Posit Connect groups or user names; empty = everyone (prototype) ──
    access={"attrition": [], "fraud": [], "network": []},
)


def settings():
    s = dict(DEFAULTS)
    try:
        import streamlit as st
        s.update({k: v for k, v in dict(st.secrets.get("app", {})).items()})
    except Exception:
        pass                                           # no secrets file: defaults apply
    mode = os.getenv("PKG_APP_MODE") or os.getenv("ATTR_APP_MODE")
    if mode: s["mode"] = mode
    if os.getenv("ATTR_BUNDLE"): s["bundle_path"] = os.getenv("ATTR_BUNDLE")
    return s
