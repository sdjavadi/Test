"""Deposit attrition: model scoring, the warning list and per-client detail (logic only — pages draw it)."""
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

from ..core import state
from ..core.fmt import norm_pwr
from . import labels as LB
from . import model as MD

NUM = ["bal", "bal_norm", "capped"]
ALPHAS = {0.0: "likelihood", 0.5: "likelihood × size", 1.0: "dollars at risk"}


@st.cache_resource(show_spinner="Loading the attrition model…")
def _bundle(mode, path):
    p = pathlib.Path(path)
    if p.exists():
        return MD.load_bundle(p)
    if mode == "demo":
        from ..core.demo import demo_bundle
        b = demo_bundle(state.source()); b["_booster"] = MD._booster(b); return b
    raise FileNotFoundError(f"attrition model not found at {p.resolve()} — run the export cell of the model notebook "
                            f"and set ATTR_BUNDLE (or bundle_path)")


def bundle():
    return _bundle(state.MODE, state.CFG["bundle_path"])


@st.cache_data(ttl=6 * 3600, show_spinner="Scoring every client whose money is still here…")
def _score_all(mode, months_scored):
    src, b = state.source(), bundle()
    months = src.months()[:months_scored]
    parts = []
    for i, m in enumerate(months):
        df = src.features(m).reset_index(drop=True)
        for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        X = MD.matrix(b, df)
        p, C = MD.predict(b, X, contrib=(i == 0))
        keep = [c for c in ["cust_pwr_id", "rltn_pwr_id", "customer_name", "naics_desc", "segment_desc", "mdm_ids", "month"] + NUM if c in df.columns]
        out = df[keep].copy(); out["p"] = p
        if i == 0:                       # reasons and fired signals for the current month
            feats = b["feats"]; top = np.argsort(-C, axis=1)[:, :5]
            out["reasons"] = [[(feats[j], float(C[r, j]), float(X[r, j])) for j in top[r] if C[r, j] > 0] for r in range(len(df))]
            fm = MD.fired_matrix(b, df)
            out["fired"] = [list(fm.columns[v]) for v in fm.to_numpy(bool)] if len(fm.columns) else [[] for _ in range(len(df))]
        parts.append(out)
    S = pd.concat(parts, ignore_index=True)
    S["cust_pwr_id"] = S.cust_pwr_id.astype(str)
    S["key"] = S.cust_pwr_id.map(norm_pwr)
    return S, months


def scored():
    return _score_all(state.MODE, int(state.CFG["months_scored"]))


def status():
    """(True, None) when the module can run; (False, reason) otherwise — pages show the reason instead of failing."""
    try:
        scored(); return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def default_k():
    return 120 if state.MODE == "demo" else int(state.CFG["list_size"])


def current_settings():
    return int(st.session_state.get("att_K") or default_k()), float(st.session_state.get("att_alpha", float(state.CFG["alpha"])))


def label(f):
    return LB.plain(f, bundle().get("labels", {}).get(f))


def card_story(b, r):
    fired = list(r.fired or []); late = [f for f in fired if f in b.get("late_feats", [])]
    early = sorted([f for f in fired if f not in late], key=lambda f: b.get("onset", {}).get(f, 0))
    reasons = [f for f, _, _ in (r.reasons or [])]
    themes = [LB.theme(f) for f in (early + reasons)]
    theme = next((t for t in themes if t != "activity"), themes[0] if themes else "activity")
    pct_norm = r.bal / r.bal_norm if r.bal_norm else np.nan
    urgent = bool(late) or (pd.notna(pct_norm) and pct_norm < 0.7)
    nxt = [f"**{r.p:.0%}** chance the money leaves within 6 months — **{LB.money(r.capped)}** at stake."]
    if urgent:
        nxt.append("Money appears to be leaving **this month** — call now.")
    elif early:
        on_ = b.get("onset", {}).get(early[0])
        if on_: nxt.append(f"Clients showing *{LB.plain(early[0], b.get('labels', {}).get(early[0]))}* typically left within **{abs(int(on_))} months**.")
    return dict(late=late, early=early, reasons=r.reasons or [], theme=theme, urgent=urgent, pct_norm=pct_norm, next=nxt)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _watchlist(mode, K, alpha):
    S, months = scored(); b = bundle()
    S = S.copy(); S["rank"] = MD.rank_month(S, alpha)
    on = set(zip(S.loc[S["rank"] <= K, "cust_pwr_id"], S.loc[S["rank"] <= K, "month"]))
    def run(c):
        n = 0
        for m in months:
            if (c, m) in on: n += 1
            else: break
        return n
    L = S[(S.month == months[0]) & (S["rank"] <= K)].copy()
    L["months_on_list"] = L.cust_pwr_id.map(run)
    L["story"] = [card_story(b, r) for r in L.itertuples()]
    L["urgent"] = L.story.map(lambda s: s["urgent"]); L["theme"] = L.story.map(lambda s: s["theme"])
    return S, L.sort_values("rank"), months


def watchlist(K=None, alpha=None):
    k0, a0 = current_settings()
    return _watchlist(state.MODE, int(K or k0), float(a0 if alpha is None else alpha))


def list_map():
    """Customer key → rank, for clients on the current list."""
    try:
        _, L, _ = watchlist()
        return dict(zip(L.key, L["rank"]))
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _key_index(mode):
    S, months = scored()
    cur = S[S.month == months[0]]
    return {k: r for k, r in zip(cur.key, cur.drop(columns=["reasons", "fired"], errors="ignore").to_dict("records")) if k}


def row_for_key(key):
    try:
        return _key_index(state.MODE).get(key)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def history(mode, custs):
    h = state.source().history(list(custs))
    for c in h.columns:
        if c not in ("cust_pwr_id", "rltn_pwr_id", "customer_name", "month", "naics_desc", "segment_desc", "mdm_id", "naics_cd"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h


@st.cache_data(ttl=3600, show_spinner="Scoring this client month by month…")
def customer_detail(mode, cust):
    src, b = state.source(), bundle()
    df = src.customer_features(cust).sort_values("month").reset_index(drop=True)
    for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    X = MD.matrix(b, df); p, C = MD.predict(b, X, contrib=True)
    return df, X, p, C, MD.fired_matrix(b, df)


@st.cache_data(ttl=3600, show_spinner=False)
def relationship(mode, rltn):
    r = state.source().relationship(rltn)
    for c in ("bal", "bal_norm"): r[c] = pd.to_numeric(r[c], errors="coerce")
    return r


@st.cache_data(ttl=3600, show_spinner="Reading counterparties…")
def counterparties(mode, mdm_ids, end_month):
    return state.source().counterparties(list(mdm_ids), end_month)
