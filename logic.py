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
    # 6-month-memory 'new' + a retained relationship base kills the
    # one-month-wonder false positives (payers appearing then vanishing)
    hit = j[((j["dynamics_role_stable"] == "expanding")
             | ((j["strength_mom_pctl_naics"] >= 0.9)
                & (j["strength_mom"] > 0.1)))
            & (j["n_payer_retained"].fillna(0) >= 1)].copy()
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
# Definitions surfaced in the UI (kept here so FastAPI can serve them too)
# ---------------------------------------------------------------------------

ROLE_DEFS = {
    # flow_role
    "collector": "Net accumulator: receives substantially more than it pays (flow ratio > +0.5).",
    "distributor": "Net payer: pays out substantially more than it receives (flow ratio < \u22120.5). Payroll-like.",
    "conduit": "High-volume pass-through: in \u2248 out with top-decile throughflow. AML-relevant.",
    "trader": "Genuine two-way commerce: balanced flows with above-median dollar-matched reciprocity.",
    "one_sided": "Only pays OR only receives within the visible network \u2014 the other side lives outside PNC.",
    "mixed": "No dominant flow pattern.",
    # hierarchy_role
    "upstream_supplier": "Bottom third of the flow hierarchy: money reaches it early in supply chains.",
    "midstream": "Middle third of the flow hierarchy: an intermediary position.",
    "downstream_buyer": "Top third of the flow hierarchy: an end-buyer position.",
    "unknown": "Hierarchy position could not be computed.",
    # dependence_role
    "single_relationship": "Fewer than 3 counterparts \u2014 too thin a file for concentration to be meaningful.",
    "infra_dependent": ">70% of inflow arrives via infrastructure hubs (processors, payroll, settlement).",
    "concentrated": ">70% of inflow from one payer OR >70% of outflow to one payee \u2014 fragile on one side.",
    "balanced": "No dominant single relationship on either side.",
    # embeddedness_role
    "peripheral": "Edge of the visible network: very few counterparts or shallow core position.",
    "connector": "Payment relationships spread across many communities \u2014 a cross-segment broker.",
    "embedded": "\u226570% of dollars stay inside its own payment community.",
    "intermediate": "Between peripheral and embedded.",
    # dynamics_role
    "newcomer": "First appeared within the last ~3 months \u2014 too young to judge.",
    "intermittent": "Reappeared after a gap of inactivity.",
    "bleeding": "Sustained loss of payer revenue (\u226540% of payer dollars lost, or sharp decline with low retention). Requires \u22653 payers last month.",
    "expanding": "\u226540% of inflow from first-time payers (6-month memory) with positive momentum. Requires \u22653 payers.",
    "steady": "Stable counterpart set and stable dollars month-over-month.",
    "variable": "Moving, but not classifiable as any of the above.",
}

TAXONOMY_DEFS = {
    "flow_role": "What the customer DOES with money: accumulate, distribute, pass through, or trade.",
    "dynamics_role": "What is HAPPENING to the customer month-over-month.",
    "dependence_role": "How fragile its revenue/spend base is (concentration).",
    "embeddedness_role": "Its position in the network fabric.",
    "hierarchy_role": "Where it sits in the supply-chain flow hierarchy.",
}

METRIC_DEFS = {
    "Role stability": "Average months the customer has held its current roles, divided by months observed (0\u20131). Near 1 = behaviorally locked in; low despite long tenure = chronic flux.",
    "Inflow / Outflow / Net": "Total dollars received / paid / their difference, per month. Dotted markers show the month a stable role changed.",
    "Role ribbon": "One band per role taxonomy across all months; a color change is a stabilized role transition (confirmed two consecutive months).",
    "Size": "Percentile of total payment dollars within the customer's NAICS peer group.",
    "Connectivity": "Percentile of number of distinct counterparts within its peer group.",
    "Net-flow": "Percentile of net accumulation vs. peers \u2014 high = collector-like for its industry.",
    "Revenue concentration": "Percentile of inflow concentration (HHI). High = more dependent on few payers than peers.",
    "Retention": "Percentile of payer-set retention (Jaccard vs. last month) among peers.",
    "Momentum": "Percentile of month-over-month growth among peers. Red = bottom quintile: losing ground to its industry.",
    "Counterparty cohorts": "Each month's relationships split by memory: retained (also present last month), returning (absent last month but seen within 6 months), new (first appearance in the 6-month window) \u2014 above zero; lost (present last month, gone now) below zero. Toggle between dollars and counterparty counts, payer or payee side.",
    "Peer group": "Ranked within the finest NAICS level (4\u21923\u21922 digits) having \u226530 members that month. Percentiles are computed on the full customer population.",
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


# Metric trend selector for the deep dive: label -> (column, taxonomy
# whose stable-role changes get marked on the chart, y-axis format)
METRIC_TREND_OPTIONS = {
    "Total flow ($)": ("strength", None, "$"),
    "Flow ratio (net / total)": ("flow_ratio", "flow_role", "%"),
    "Throughflow ($)": ("throughflow", "flow_role", "$"),
    "Inflow concentration (HHI)": ("hhi_in", "dependence_role", ""),
    "Top payer share": ("top1_in_share", "dependence_role", "%"),
    "Hub inflow share": ("hub_in_share", "dependence_role", "%"),
    "Payer retention (Jaccard)": ("payer_jaccard", "dynamics_role", "%"),
    "Lost payer amount share": ("lost_payer_amount_share",
                                "dynamics_role", "%"),
    "New payer amount share (6-mo memory)": ("new_payer_amount_share",
                                             "dynamics_role", "%"),
    "Momentum (MoM log-ratio)": ("strength_mom", "dynamics_role", ""),
    "Community participation": ("participation_coef",
                                "embeddedness_role", ""),
    "Intra-community $ share": ("frac_intra_edges_w",
                                "embeddedness_role", "%"),
    "Supply-chain position (trophic)": ("trophic_level",
                                        "hierarchy_role", ""),
}


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


def cohort_series(h: pd.DataFrame, side: str = "payer",
                  measure: str = "amount") -> pd.DataFrame:
    """Counterparty cohorts per month, multi-window memory:
    retained (also present last month), returning (absent last month, seen
    within 6 months), new (first appearance in 6-month memory) above zero;
    lost (present last month, gone now) below zero.
    measure: 'amount' ($) or 'count' (# counterparties)."""
    if measure == "amount":
        base = h["in_strength"] if side == "payer" else h["out_strength"]
        prev_base = base.shift()
        return pd.DataFrame({
            "time_key": h["time_key"],
            "retained": base * h[f"retained_{side}_amount_share"],
            "returning": base * h[f"returning_{side}_amount_share"],
            "new": base * h[f"new_{side}_amount_share"],
            "lost": -(prev_base
                      * h[f"lost_{side}_amount_share"].fillna(0)),
        })
    return pd.DataFrame({
        "time_key": h["time_key"],
        "retained": h[f"n_{side}_retained"],
        "returning": h[f"n_{side}_returning"],
        "new": h[f"n_{side}_new"],
        "lost": -h[f"n_{side}_lost"].fillna(0),
    })
