"""
logic.py — PKG Explorer business logic: spotlight queues, narratives,
deep-dive series builders. No streamlit imports — this file becomes the
FastAPI layer at handoff.

Every queue returns a DataFrame with at least:
    node, headline ($ or score), narrative (one sentence), plus queue-
    specific supporting columns. Ranked, top-N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data

TOP_N = 30


def _latest(version: str) -> str:
    return list(data.months(version))[-1]


def _fmt_money(x: float) -> str:
    if pd.isna(x):
        return "—"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"${x / div:,.1f}{suf}"
    return f"${x:,.0f}"


def _mr(df: pd.DataFrame, month: str) -> pd.DataFrame:
    return df[df["time_key"] == month]


def _joined(version: str, month: str) -> pd.DataFrame:
    n = data.node_month(version, month)
    r = data.roles_month(version, month)
    return n.merge(r.drop(columns=["naics2", "naics4"], errors="ignore"),
                   on=["time_key", "version", "node"], how="inner",
                   suffixes=("", "_r"))


def _prev_roles(version: str, month: str) -> pd.DataFrame | None:
    ms = data.months(version)
    i = ms.index(month)
    if i == 0:
        return None
    return data.roles_month(version, ms[i - 1])



def _empty_guard(df: pd.DataFrame) -> pd.DataFrame:
    out = df.head(0).copy()
    out["headline"] = pd.Series(dtype="string")
    out["narrative"] = pd.Series(dtype="string")
    return out


# ---------------------------------------------------------------------------
# Spotlight queues
# ---------------------------------------------------------------------------

def q_revenue_at_risk(version: str, month: str | None = None,
                      top_n: int = TOP_N) -> pd.DataFrame:
    """Bleeding customers ranked by dollars at risk =
    strength x lost_payer_amount_share."""
    month = month or _latest(version)
    j = _joined(version, month)
    b = j[j["dynamics_role_stable"] == "bleeding"].copy()
    if b.empty:
        return _empty_guard(b)
    b["at_risk"] = b["strength"] * b["lost_payer_amount_share"].fillna(0)
    b = b.sort_values("at_risk", ascending=False).head(top_n)
    b["headline"] = [_fmt_money(v) + " at risk" for v in b["at_risk"]]
    b["narrative"] = b.apply(
        lambda r: (f"{_fmt_money(r['strength'])}/mo book; lost "
                   f"{r['lost_payer_amount_share']:.0%} of payer revenue"
                   + ("" if r.get("top_payer_same", 1) == 1
                      else "; anchor payer changed")
                   + f"; {int(r['n_payer_lost'])} payers gone this month."),
        axis=1)
    return b


def q_sustained_drift(version: str, month: str | None = None,
                      top_n: int = TOP_N) -> pd.DataFrame:
    """steady->bleeding stable transitions this month, plus customers with
    n_role_switches >= 3 — multiple sustained behavioral changes at once."""
    month = month or _latest(version)
    j = _joined(version, month)
    prev = _prev_roles(version, month)
    hits = []
    if prev is not None:
        pmap = prev.set_index("node")["dynamics_role_stable"]
        was_steady = pmap.reindex(j["node"]).to_numpy() == "steady"
        s2b = j[was_steady & (j["dynamics_role_stable"] == "bleeding")]
        s2b = s2b.assign(reason="steady \u2192 bleeding")
        hits.append(s2b)
    multi = j[j["n_role_switches"] >= 3].assign(
        reason=lambda d: d["n_role_switches"].astype(int).astype(str)
        + " roles switched")
    hits.append(multi)
    out = (pd.concat(hits, ignore_index=True)
           .drop_duplicates("node")
           .sort_values("strength", ascending=False).head(top_n))
    if out.empty:
        return _empty_guard(out)
    out["headline"] = out["reason"]
    out["narrative"] = out.apply(
        lambda r: (f"{_fmt_money(r['strength'])}/mo customer; "
                   f"{r['reason']}; stability "
                   f"{r['role_stability_norm']:.2f}."), axis=1)
    return out


def q_losing_to_peers(version: str, month: str | None = None,
                      top_n: int = TOP_N) -> pd.DataFrame:
    """Bottom-quintile momentum within a sector whose median momentum is
    flat or positive: shrinking in a healthy industry."""
    month = month or _latest(version)
    j = _joined(version, month)
    sector_mom = (j.dropna(subset=["naics2"])
                  .groupby("naics2")["strength_mom"].median()
                  .rename("sector_mom"))
    j = j.merge(sector_mom, left_on="naics2", right_index=True, how="left")
    hit = j[(j["strength_mom_pctl_naics"] <= 0.2)
            & (j["sector_mom"] >= -0.02)
            & (j["strength_mom"] < 0)].copy()
    if hit.empty:
        return _empty_guard(hit)
    hit["gap"] = hit["sector_mom"] - hit["strength_mom"]
    hit = hit.sort_values(["gap", "strength"],
                          ascending=False).head(top_n)
    hit["headline"] = ("-" + (np.expm1(-hit["strength_mom"]) * 100)
                       .round(0).astype(int).astype(str) + "% vs sector")
    hit["narrative"] = hit.apply(
        lambda r: (f"{_fmt_money(r['strength'])}/mo; shrank "
                   f"{abs(np.expm1(r['strength_mom'])):.0%} while sector "
                   f"{r['naics2']} held steady — bottom "
                   f"{r['strength_mom_pctl_naics']:.0%} of "
                   f"{int(r['peer_size']) if pd.notna(r['peer_size']) else '—'} peers."),
        axis=1)
    return hit


def q_new_conduits(version: str, month: str | None = None,
                   top_n: int = TOP_N) -> pd.DataFrame:
    """Stable flow_role transitions into 'conduit' — commerce became
    pass-through. AML review teaser."""
    month = month or _latest(version)
    j = _joined(version, month)
    prev = _prev_roles(version, month)
    if prev is None:
        return j.head(0).assign(headline=None, narrative=None)
    pmap = prev.set_index("node")["flow_role_stable"]
    was = pmap.reindex(j["node"])
    hit = j[(j["flow_role_stable"] == "conduit")
            & was.notna().to_numpy()
            & (was != "conduit").to_numpy()].copy()
    if hit.empty:
        return _empty_guard(hit)
    hit["was"] = was[was != "conduit"].reindex(hit["node"]).to_numpy()
    hit = hit.sort_values("throughflow", ascending=False).head(top_n)
    hit["headline"] = [_fmt_money(v) + " pass-through" for v in hit["throughflow"]]
    hit["narrative"] = hit.apply(
        lambda r: (f"was {r['was'].replace('_', ' ')}, now conduit: "
                   f"{_fmt_money(r['throughflow'])}/mo flows through with "
                   f"net {r['flow_ratio']:+.0%}."), axis=1)
    return hit


def q_rising_stars(version: str, month: str | None = None,
                   top_n: int = TOP_N) -> pd.DataFrame:
    """Expanding customers outgrowing their peers — the positive queue."""
    month = month or _latest(version)
    j = _joined(version, month)
    hit = j[(j["dynamics_role_stable"] == "expanding")
            | ((j["strength_mom_pctl_naics"] >= 0.9)
               & (j["strength_mom"] > 0.1))].copy()
    hit = hit.sort_values("strength_mom", ascending=False).head(top_n)
    if hit.empty:
        return _empty_guard(hit)
    hit["headline"] = ("+" + (np.expm1(hit["strength_mom"]) * 100)
                       .round(0).astype(int).astype(str) + "% MoM")
    hit["narrative"] = hit.apply(
        lambda r: (f"{_fmt_money(r['strength'])}/mo and growing "
                   f"{np.expm1(r['strength_mom']):.0%}; "
                   f"{r['new_payer_amount_share']:.0%} of inflow from "
                   f"brand-new payers."), axis=1)
    return hit


def _fmt_series(s: pd.Series) -> pd.Series:
    return s.map(_fmt_money)


QUEUES = {
    "Revenue at risk": (q_revenue_at_risk,
                        "Bleeding customers, ranked by dollars at risk"),
    "Sustained drift": (q_sustained_drift,
                        "Multiple stable-role changes or steady→bleeding"),
    "Losing to peers": (q_losing_to_peers,
                        "Shrinking while their industry holds steady"),
    "New conduits": (q_new_conduits,
                     "Flow pattern shifted from commerce to pass-through"),
    "Rising stars": (q_rising_stars,
                     "Expanding books outgrowing their peers"),
}


# ---------------------------------------------------------------------------
# Deep-dive builders
# ---------------------------------------------------------------------------

TAXONOMIES = ["flow_role", "dynamics_role", "dependence_role",
              "embeddedness_role", "hierarchy_role"]


def history(node: str, version: str) -> pd.DataFrame:
    return data.customer_history(node, version)


def role_ribbon(h: pd.DataFrame) -> pd.DataFrame:
    """Long frame (taxonomy, role, start, end) of stable-role segments,
    for the Gantt-style ribbon."""
    segs = []
    for tax in TAXONOMIES:
        col = f"{tax}_stable"
        if col not in h.columns:
            continue
        cur, start = None, None
        for _, row in h.iterrows():
            r = row[col]
            if r != cur:
                if cur is not None:
                    segs.append((tax, cur, start, row["time_key"]))
                cur, start = r, row["time_key"]
        if cur is not None:
            segs.append((tax, cur, start, h["time_key"].iloc[-1]))
    return pd.DataFrame(segs, columns=["taxonomy", "role", "start", "end"])


def role_changes(h: pd.DataFrame) -> pd.DataFrame:
    """(time_key, label) rows where any stable role changed — timeline
    markers for the strength chart."""
    out = []
    for tax in TAXONOMIES:
        col = f"{tax}_stable"
        if col not in h.columns:
            continue
        s = h[["time_key", col]].dropna()
        ch = s[s[col].ne(s[col].shift()) & s[col].shift().notna()]
        for _, r in ch.iterrows():
            out.append((r["time_key"],
                        f"{tax.replace('_role', '')}: {r[col]}"))
    return pd.DataFrame(out, columns=["time_key", "label"])


PEER_METRICS_SHOW = [
    ("strength_pctl_naics", "Size"),
    ("degree_pctl_naics", "Connectivity"),
    ("net_flow_pctl_naics", "Net-flow"),
    ("hhi_in_pctl_naics", "Revenue concentration"),
    ("payer_jaccard_pctl_naics", "Retention"),
    ("strength_mom_pctl_naics", "Momentum"),
]


def peer_snapshot(h: pd.DataFrame) -> pd.DataFrame:
    last = h.dropna(subset=["strength_pctl_naics"]).tail(1)
    if last.empty:
        last = h.tail(1)
    rows = [(lbl, float(last[c].iloc[0]) if c in last and
             pd.notna(last[c].iloc[0]) else np.nan)
            for c, lbl in PEER_METRICS_SHOW]
    return pd.DataFrame(rows, columns=["metric", "pctl"])


def turnover_series(h: pd.DataFrame) -> pd.DataFrame:
    """Per month: inflow split into retained vs new payers, plus the
    lost-payer amount (previous month's dollars that left) as negative."""
    t = h[["time_key", "in_strength", "new_payer_amount_share",
           "lost_payer_amount_share"]].copy()
    t["new"] = t["in_strength"] * t["new_payer_amount_share"].fillna(0)
    t["retained"] = t["in_strength"] - t["new"]
    prev_in = t["in_strength"].shift()
    t["lost"] = -(prev_in * t["lost_payer_amount_share"].fillna(0))
    return t[["time_key", "retained", "new", "lost"]]
