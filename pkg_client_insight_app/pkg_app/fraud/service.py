"""Suspicious activity: reads the signals written by pkg_fraud_signals_v1.ipynb (logic only — pages draw it)."""
import datetime
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

from ..core import state
from ..core.fmt import cust_key, norm_pwr
from . import typologies as T

BOOL = ["processor_like", "industry_pt", "treasury_sweep", "existed_before"]
TEXT = ["mdm_id", "customer_pwr_id", "pwr", "master_pwr_id", "customer_name", "naics_cd", "naics_desc", "party_type", "month",
        "typologies", "primary", "tier", "peer_group", "rt_top_other", "rt_top_name"]
DISPOSITIONS = ["Not reviewed yet", "Explained — ordinary business", "Already known to Compliance", "Refer to Compliance / AML",
                "Data issue (pattern is wrong)", "Need more information"]


def _bool(v):
    if isinstance(v, (bool, np.bool_)): return bool(v)
    return str(v).strip().lower() in ("true", "1", "t", "yes")


@st.cache_data(ttl=6 * 3600, show_spinner="Reading suspicious-activity signals…")
def _signals(mode):
    S = state.source().fraud_signals().copy()
    if not len(S): raise ValueError("the signals table is empty")
    for c in S.columns:
        if c in BOOL: S[c] = S[c].map(_bool)
        elif c not in TEXT: S[c] = pd.to_numeric(S[c], errors="coerce")
    for c in TEXT:
        if c in S: S[c] = S[c].where(S[c].notna(), None)
    S["month"] = S.month.astype(str); S["typologies"] = S.typologies.fillna("").astype(str)
    S["score"] = S.score.fillna(0.0)
    pw = S["customer_pwr_id"] if "customer_pwr_id" in S else S.get("pwr")
    S["key"] = [cust_key(p, m) for p, m in zip(pw if pw is not None else [None] * len(S), S.mdm_id)]
    months = sorted(S.month.unique())
    on = S[S.score > 0]
    S["months_flagged"] = S.mdm_id.map(on.groupby("mdm_id").month.nunique()).fillna(0).astype(int)
    S["first_flagged"] = S.mdm_id.map(on.groupby("mdm_id").month.min())
    return S, months


def signals():
    return _signals(state.MODE)


def status():
    try:
        signals(); return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def latest(any_month=False):
    """One row per flagged entity: its latest-month row (or, with any_month, its highest-scoring month)."""
    S, months = signals()
    if not any_month:
        L = S[(S.month == months[-1]) & (S.score > 0)].copy()
    else:
        L = S[S.score > 0].sort_values(["score", "month"], ascending=False).drop_duplicates("mdm_id").copy()
    L["new"] = (L.months_flagged == 1) & (L.first_flagged == months[-1])
    return L.sort_values("score", ascending=False)


def flag_map():
    """Customer key → (tier, score, primary pattern) in the latest month."""
    try:
        L = latest()
    except Exception:
        return {}
    L = L.sort_values("score", ascending=False).drop_duplicates("key")
    return {k: (t, s, p) for k, t, s, p in zip(L.key, L.tier, L.score, L.primary) if k}


def rows_for_key(key):
    try:
        S, _ = signals()
    except Exception:
        return pd.DataFrame(columns=["mdm_id", "customer_name", "customer_pwr_id", "naics_desc"])
    return S[S.key == key]


@st.cache_data(ttl=3600, show_spinner=False)
def history(mode, mdm_ids):
    h = state.source().fraud_history(list(mdm_ids))
    for c in h.columns:
        if c not in ("mdm_id", "month"): h[c] = pd.to_numeric(h[c], errors="coerce")
    return h.sort_values(["mdm_id", "month"])


@st.cache_data(ttl=3600, show_spinner=False)
def evidence(mode, mdm_ids):
    e = state.source().fraud_evidence(list(mdm_ids))
    for c in ("amount", "amount_back", "n_txn", "threshold"):
        if c in e: e[c] = pd.to_numeric(e[c], errors="coerce")
    if "split_day" in e: e["split_day"] = e.split_day.map(_bool)
    return e


@st.cache_data(ttl=3600, show_spinner=False)
def metrics_history(mode, mdm_ids, start_month):
    h = state.source().metrics_history(list(mdm_ids), start_month)
    for c in h.columns:
        if c not in ("mdm_id", "month", "customer_pwr_id", "master_pwr_id", "customer_name", "naics_cd", "naics_desc", "cust_pwr_id",
                     "rltn_pwr_id", "segment_desc"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h


def log_review(mdm_id, month, score, disposition, note):
    row = pd.DataFrame([dict(saved=datetime.datetime.now().isoformat(timespec="seconds"), user=state.current_user()["user"],
                             mdm_id=mdm_id, month=month, score=round(float(score or 0), 1), disposition=disposition, note=note)])
    fp = pathlib.Path(state.CFG["fraud_feedback_path"])
    row.to_csv(fp, mode="a", header=not fp.exists(), index=False)
