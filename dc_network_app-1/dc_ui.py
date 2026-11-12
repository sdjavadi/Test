"""
Shared pieces of the Data Center Network Explorer pages: style, number formats, cached Impala reads, the network
drawing and small chart helpers. All data logic lives in dc_core.py.
"""
from __future__ import annotations

import html
import logging
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dc_core as dc

log = logging.getLogger("dc_network")

TTL = 6 * 3600
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
SELECTABLE = _V >= (1, 35)          # row / point selection in st.dataframe and st.plotly_chart
HAS_QP = hasattr(st, "query_params")

# ---- colours. Nodes also differ by shape, edge states by line pattern and a marker, so colour never works alone.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAVY, BLUE, SLATE, VIOLET = "#0d366b", "#2a78d6", "#5b6573", "#6250d6"
ORANGE = "#eb6834"
C_IN, C_OUT = BLUE, ORANGE                     # paired series on line charts
FLOW_IN, FLOW_OUT = "#0ca30c", "#d03b3b"       # money in (bars above zero) / money out (bars below zero)
STATE_COLOR = {"on schedule": "#0ca30c", "late or overdue": ORANGE, "stopped": "#d03b3b", "casual": "#8f8e86",
               "others": "#8f8e86", "between": "#c2c1b9", "workbook": VIOLET}
STATE_DASH = {"on schedule": "solid", "late or overdue": "solid", "stopped": "solid", "casual": "dash", "others": "solid",
              "between": "solid", "workbook": "dot"}
STATE_LABEL = {"on schedule": "Regular, on schedule", "late or overdue": "Regular, late or overdue", "stopped": "Regular, stopped",
               "casual": "Not regular", "others": "To / from the neighbours not drawn", "between": "Between neighbours",
               "workbook": "Relationship in the workbook"}
STATE_GLYPH = {"late or overdue": "triangle-up", "stopped": "x"}
KIND_LABEL = {"cust": "PNC customer", "ent": "Outside counterparty"}

TABLES = dict(cdim=dc.T_CDIM, pair=dc.T_PAIR, cust_met=dc.T_CUST_MET, cpty_met=dc.T_CPTY_MET, tm=dc.T_TM, reg=dc.T_REG,
              dep=dc.T_DEP, map=dc.T_MAP, cpty_month=dc.T_CPTY_MONTH, fi_routing=dc.T_FI_ROUTING, fi_names=dc.T_FI_NAMES)
REQUIRED = {"cdim": {"mdm_id", "customer_pwr_id", "master_pwr_id", "customer_name"},
            "pair": {"cust", "dir", "scope", "other", "amount", "n_txn", "month"}}


def inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1560px; }
.dc-title { font-size: 1.9rem !important; font-weight: 600; letter-spacing: -0.015em; color: #1B2430; line-height: 1.2; }
.dc-sub { color: #5B6573; margin: 0.2rem 0 0.6rem 0; }
.dc-chip { display: inline-block; font-size: 0.8rem; color: #1B2430; background: #F1F3F6; border: 1px solid #E2E6EC;
           border-radius: 999px; padding: 2px 10px; margin: 0 6px 6px 0; }
.dc-chip.warn { color: #8A4B14; background: #FFF4EA; border-color: #F5D3B5; }
.dc-chip.bad { color: #8f1d1d; background: #fdeaea; border-color: #f3c2c2; }
.dc-chip.ok { color: #14653a; background: #e5f5eb; border-color: #bfe3cc; }
.dc-note { color: #5B6573; font-size: 0.86rem; }
.dc-h { font-size: 1.05rem; font-weight: 600; color: #1B2430; margin: 0.4rem 0 0.2rem 0; }
[data-testid="stMetricValue"] { font-size: 1.45rem; }
/* multiselects grow with what is chosen and show each name in full (two generations of Streamlit markup) */
[data-testid="stMultiSelect"] div[role="group"] { max-height: none !important; height: auto !important; min-height: 40px; }
[data-testid="stMultiSelectTagsContainer"] { flex-wrap: wrap !important; overflow: visible !important; height: auto !important;
  row-gap: 4px; padding-top: 5px; padding-bottom: 5px; }
[data-testid="stMultiSelectTagsContainer"] > span[role="group"] > span { max-width: 100% !important; }
[data-testid="stMultiSelect"] [data-baseweb="select"] > div { max-height: none !important; height: auto !important; overflow: visible !important; }
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div { max-height: none !important; overflow: visible !important; flex-wrap: wrap !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] { max-width: 100% !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] span[title] { max-width: none !important; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ formats
def usd(x) -> str:
    """Dollars with thousands separators and no cents."""
    try:
        if x is None or pd.isna(x):
            return "–"
    except (TypeError, ValueError):
        return "–"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def num(x, dec: int = 0) -> str:
    try:
        if x is None or pd.isna(x):
            return "–"
    except (TypeError, ValueError):
        return "–"
    return f"{x:,.{dec}f}"


def pct(x, dec: int = 2) -> str:
    """A fraction shown as a percentage."""
    try:
        if x is None or pd.isna(x):
            return "–"
    except (TypeError, ValueError):
        return "–"
    return f"{100 * x:,.{dec}f}%"


def yes(v) -> bool:
    """True for a real True (numpy or Python), False for anything else including missing values."""
    return isinstance(v, (bool, np.bool_)) and bool(v)


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def chips(items) -> None:
    """items: [(text, class)] where class is '', 'ok', 'warn' or 'bad'."""
    st.markdown("".join(f'<span class="dc-chip {c}">{esc(t)}</span>' for t, c in items if t), unsafe_allow_html=True)


def table(df: pd.DataFrame, where=None, money=(), one_dec=(), pct_cols=(), skip=(), **kw):
    """st.dataframe with thousands separators ($ and no cents on money columns). Formatting is display only:
    column sorting still uses the numbers."""
    df = df.reset_index(drop=True)
    fmt = {}
    for c in df.columns:
        if c in skip or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        fmt[c] = "${:,.0f}" if c in money else "{:.2%}" if c in pct_cols else "{:,.1f}" if c in one_dec else "{:,.0f}"
    for c in df.columns:                      # empty text cells would show as "None"
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(object).map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else v)
    if "height" in kw:
        kw["height"] = int(min(kw["height"], 38 + 35 * max(len(df), 1) + 4))
    return (where or st).dataframe(df.style.format(fmt, na_rep="–"), hide_index=True, **WIDE, **kw)


def fail(what: str, e: Exception) -> None:
    log.exception("loading %s failed", what)
    st.error(f"Could not load {what}. Please try again; if it keeps failing, send the time to the app owner.")


# ------------------------------------------------------------------ cached reads
@st.cache_data(ttl=TTL, show_spinner=False)
def cols_of(table_name: str):
    return dc.table_columns(table_name)


def all_cols() -> dict:
    return {k: cols_of(t) for k, t in TABLES.items()}


@st.cache_data(ttl=TTL, show_spinner=False)
def months_of(table_name: str) -> list:
    return dc.table_months(table_name)


def _key(focus) -> tuple:
    return tuple((f.key, f.label, f.mdm, f.ent, f.wb_ids, f.cut) for f in focus)


@st.cache_resource(ttl=TTL, max_entries=40, show_spinner=False)
def _ego(fkey: tuple, first: str, last: str, tm_month):
    focus = [dc.Focus(*t) for t in fkey]
    return dc.load_ego(focus, first, last, all_cols(), tm_month)


@st.cache_resource(ttl=TTL, max_entries=40, show_spinner=False)
def _streams(fkey: tuple, last: str, n: int):
    focus = [dc.Focus(*t) for t in fkey]
    c = all_cols()
    return dc.load_streams(focus, last, n, c.get("reg"), months_of(dc.T_REG) if c.get("reg") else [])


def load_ego(focus, first, last, tm_month):
    return _ego(_key(focus), first, last, tm_month)


def load_streams(focus, last, n):
    return _streams(_key(focus), last, n)


@st.cache_data(ttl=TTL, max_entries=60, show_spinner=False)
def load_between(mem_key: tuple, first: str, last: str) -> pd.DataFrame:
    members = {k: {"mdm": list(m), "ent": list(e)} for k, m, e in mem_key}
    return dc.between_edges(members, first, last)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_family(master_pwr_id: str, tm_month) -> pd.DataFrame:
    c = all_cols()
    return dc.family_members(master_pwr_id, c.get("cdim"), c.get("tm"), tm_month)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_customers(mdm_ids: tuple, tm_month) -> pd.DataFrame:
    c = all_cols()
    return dc.cust_attrs(list(mdm_ids), c.get("cdim"), c.get("tm"), tm_month)


@st.cache_data(ttl=TTL, show_spinner=False)
def search_customers(q: str) -> pd.DataFrame:
    return dc.search_customers(q, all_cols().get("cdim"))


def calendar() -> dict:
    """Months available: payments (the period ends at the last one), credit, stored streams."""
    c = all_cols()
    miss = [k for k, need in REQUIRED.items() if not c.get(k) or not need <= c[k]]
    if miss:
        raise RuntimeError(f"required tables unavailable: {miss}")
    pay = months_of(dc.T_PAIR)
    tm = months_of(dc.T_TM) if c.get("tm") and {"customer_pwr_id", "month"} <= c["tm"] else []
    return {"pay": pay, "last": pay[-1], "tm": tm, "tm_last": tm[-1] if tm else None}


# ------------------------------------------------------------------ network drawing
def _curve(x0, y0, x1, y1, bend: float, n: int = 20):
    """Points of a gently bent line from (x0, y0) to (x1, y1). Opposite directions bend to opposite sides."""
    cx, cy = (x0 + x1) / 2 - (y1 - y0) * bend, (y0 + y1) / 2 + (x1 - x0) * bend
    t = np.linspace(0, 1, n)
    return (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1, (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1


def _short(s, n: int = 28) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def edge_hover(e, name: dict, period_txt: str) -> str:
    a, b = esc(_short(name.get(e.src, e.src), 40)), esc(_short(name.get(e.dst, e.dst), 40))
    st_ = e.style
    if st_ == "workbook":
        amt = getattr(e, "amt", np.nan)
        return (f"<b>In the workbook</b>: {a} and {esc(getattr(e, 'wb_name', ''))}<br>{esc(getattr(e, 'wb_types', ''))} · tab {esc(getattr(e, 'wb_tabs', ''))}"
                + (f"<br>Workbook figure {amt:,.0f}" if amt == amt else "")
                + ("<br>Payments seen between them" if yes(getattr(e, "wb_paid", False)) else "<br>No payments seen between them"))
    if st_ == "others":
        return f"{a} → {b}<br><b>{usd(e.amt)}</b> · {num(getattr(e, 'n_streams', np.nan))} neighbours"
    if st_ in ("casual", "between"):
        return f"{a} → {b}<br><b>{usd(e.amt)}</b><br>Not regular" if st_ == "casual" else f"{a} → {b}<br><b>{usd(e.amt)}</b>"
    lines = [f"{a} → {b}", f"Total <b>{usd(e.amt)}</b> in {num(e.n_txn)} payments · {period_txt}"]
    med = getattr(e, "amt_med", np.nan)
    cad = getattr(e, "cadence", None)
    lines.append(f"Median payment <b>{usd(med)}</b>" + (f" · {cad}" if isinstance(cad, str) else ""))
    tail = STATE_LABEL[st_]
    if isinstance(getattr(e, "last_dt", None), str):
        tail += f" · last paid {e.last_dt}"
    if isinstance(getattr(e, "next_dt", None), str):
        tail += f" · next expected {e.next_dt}"
    lines.append(tail)
    ns, nr = getattr(e, "n_streams", 1), getattr(e, "n_reg", 1)
    if ns == ns and ns > 1:
        lines.append(f"{int(nr)} of {int(ns)} payment streams on this line are regular (figures above: the largest)")
    return "<br>".join(lines)


def node_hover(r, family: bool) -> str:
    L = [f"<b>{esc(r.label)}</b>"]
    g = lambda c, d=None: getattr(r, c, d)   # noqa: E731
    if r.kind == "focus":
        parts = []
        if g("n_mdm", 0):
            parts.append(f"{int(r.n_mdm):,} PNC customer{'s' if r.n_mdm > 1 else ''}")
        if g("n_ent", 0):
            parts.append(f"{int(r.n_ent):,} outside account group{'s' if r.n_ent > 1 else ''} (name match)")
        L.append("Selected" + (": " + " + ".join(parts) if parts else " · not in the payment data"))
        L.append(f"Paid to the neighbours shown in the table: {usd(r.amt_out)}<br>Received from them: {usd(r.amt_in)}")
    elif r.kind == "others":
        L.append(f"Neighbours not drawn. Raise 'Neighbours shown' to see more.<br>They paid the selection {usd(r.amt_in)} and received {usd(r.amt_out)}")
        return "<br>".join(L)
    elif r.kind == "wb_only":
        L.append(f"In the workbook, no payments with the selection in the period<br>{esc(g('wb_status', ''))} · workbook id {esc(r.id)}")
        return "<br>".join(L)
    else:
        if r.kind == "cust":
            m = int(g("members", 1) or 1)
            L.append(("PNC customer family" + (f" · {m:,} customers linked" if m > 1 else "")) if family else "PNC customer")
            if isinstance(g("pwr"), str):
                L.append(f"Power ID {esc(r.pwr)}" + (f" · {esc(r.naics)}" if isinstance(g("naics"), str) else ""))
        else:
            cls = g("name_class")
            L.append("Outside counterparty" + (f" · {esc(cls)}" if isinstance(cls, str) else "")
                     + (f" · hub (deals with {num(g('n_cust_all'))} PNC customers)" if (g("hub_tier") or 0) >= dc.HUB_TIER else ""))
            if isinstance(g("bank"), str):
                L.append(f"Main bank on these payments: {esc(r.bank)}")
        L.append(f"Paid the selection {usd(r.amt_in)} · received {usd(r.amt_out)}")
    if r.kind in ("focus", "cust"):
        dhe, pdp = g("dhe", np.nan), g("pd_pct", np.nan)
        if dhe == dhe or pdp == pdp:
            L.append(f"DHE {usd(dhe)} · PD {pct(pdp)}" + (f" (grade {esc(r.pd_grade)})" if isinstance(g("pd_grade"), str) else "")
                     + (" · <b>in default</b>" if yes(g("is_default", False)) else ""))
    wt = g("wb_types")
    if yes(g("in_wb", False)) and r.kind != "focus":
        L.append("In the workbook" + (f": {esc(wt)}" if isinstance(wt, str) and wt else ""))
    return "<br>".join(L)


def draw_graph(G: dc.Graph, period_txt: str, family: bool = True, height: int = 720, label_n: int = 45) -> go.Figure:
    """Directed network. Arrows point the way the money moved; line width and node size grow with the square root
    of the dollars. Solid coloured = regular stream (colour and marker give its state), dashed grey = not regular,
    dotted violet = relationship listed in the workbook."""
    nodes, edges = G.nodes, G.edges
    fig = go.Figure()
    P = nodes.set_index("key")
    name = P.label.to_dict()
    wbt = G.table.set_index("node").wb_types.to_dict() if len(G.table) else {}
    nodes = nodes.assign(wb_types=nodes.key.map(wbt))

    # ---- edges: one trace per (style, width step); arrow heads as annotations; a hover point in the middle of each
    hov = {}
    if len(edges):
        order = {"between": 0, "others": 1, "casual": 2, "workbook": 3, "stopped": 4, "late or overdue": 5, "on schedule": 6}
        E = edges.assign(_o=edges["style"].map(order).fillna(2), wstep=(edges.width * 2).round() / 2).sort_values(["_o", "amt"])
        for (style, w), grp in E.groupby(["style", "wstep"], sort=False):
            xs, ys = [], []
            col = STATE_COLOR[style]
            for e in grp.itertuples():
                if e.src not in P.index or e.dst not in P.index:
                    continue
                x0, y0, x1, y1 = P.at[e.src, "x"], P.at[e.src, "y"], P.at[e.dst, "x"], P.at[e.dst, "y"]
                bend = -0.20 if style == "workbook" else 0.07 if style == "between" else 0.10
                cx, cy = _curve(x0, y0, x1, y1, bend)
                cut = len(cx) if style == "workbook" else len(cx) - 4          # the arrow annotation draws the last stretch
                xs += list(cx[:cut]) + [None]
                ys += list(cy[:cut]) + [None]
                if style != "workbook":
                    fig.add_annotation(x=cx[-1], y=cy[-1], ax=cx[-5], ay=cy[-5], xref="x", yref="y", axref="x", ayref="y", text="",
                                       showarrow=True, arrowhead=2, arrowsize=max(0.5, min(1.6, 1.9 / math.sqrt(max(w, 0.5)))),
                                       arrowwidth=max(w, 0.8), arrowcolor=col, standoff=P.at[e.dst, "size"] / 2 + 1)
                mid = len(cx) // 2
                hov.setdefault(style, []).append((cx[mid], cy[mid], edge_hover(e, name, period_txt)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", showlegend=False,
                                     line={"width": max(w, 0.8), "color": col, "dash": STATE_DASH[style]},
                                     opacity=0.55 if style in ("between", "others") else 0.9))
        for style, pts in hov.items():
            glyph = STATE_GLYPH.get(style)
            fig.add_trace(go.Scatter(
                x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", hovertext=[p[2] for p in pts], hoverinfo="text",
                showlegend=False, hoverlabel={"align": "left"}, unselected={"marker": {"opacity": 1}},
                marker={"size": 9 if glyph else 13, "symbol": glyph or "circle", "color": STATE_COLOR[style] if glyph else "rgba(0,0,0,0)",
                        "line": {"width": 1.2 if glyph else 0, "color": "#ffffff"}}))
        for style in [s for s in ("on schedule", "late or overdue", "stopped", "casual", "others", "between", "workbook") if s in hov]:
            glyph = STATE_GLYPH.get(style)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers" if glyph else "lines", name=STATE_LABEL[style],
                                     line={"width": 3, "color": STATE_COLOR[style], "dash": STATE_DASH[style]},
                                     marker={"symbol": glyph or "circle", "size": 8, "color": STATE_COLOR[style]}, legendrank=20 + len(fig.data)))

    # ---- nodes
    nb = nodes[nodes.kind.isin(["cust", "ent"])]
    labelled = set(nb.sort_values("amt", ascending=False).head(label_n).key) | set(nodes[~nodes.kind.isin(["cust", "ent"])].key)
    style = {"wb_only": ("#ffffff", "diamond", VIOLET, 1.6, "In the workbook, no payments seen", 5),
             "others": ("#ffffff", "circle", "#8f8e86", 1.6, "Neighbours not drawn", 4),
             "cust": (BLUE, "circle", "#ffffff", 1.2, "PNC customer family" if family else "PNC customer", 2),
             "ent": (SLATE, "square", "#ffffff", 1.2, "Outside counterparty", 3),
             "focus": (NAVY, "diamond", "#ffffff", 2.0, "Selected", 1)}
    for kind, (fill, sym, line, lw, lab, rank) in style.items():
        d = nodes[nodes.kind == kind]
        if d.empty:
            continue
        pos = ["middle left" if x < 0 else "middle right" for x in d.x]
        if kind == "focus":
            pos = ["top center" if y >= 0 else "bottom center" for y in d.y]
        txt = [("<b>" + esc(_short(t, 34)) + "</b>") if kind == "focus" else (esc(_short(t)) if k in labelled else "") for t, k in zip(d.label, d.key)]
        fig.add_trace(go.Scatter(
            x=d.x, y=d.y, mode="markers+text", name=lab, text=txt, textposition=pos, customdata=d.key,
            hovertext=[node_hover(r, family) for r in d.itertuples()], hoverinfo="text", hoverlabel={"align": "left"},
            textfont={"size": 13 if kind == "focus" else 11, "color": INK if kind == "focus" else MUTED},
            marker={"size": d["size"], "color": fill, "symbol": sym, "opacity": 1, "line": {"width": lw, "color": line}},
            unselected={"marker": {"opacity": 1}, "textfont": {"color": INK if kind == "focus" else MUTED}},
            selected={"marker": {"opacity": 1}}, legendrank=rank))
    fig.update_layout(
        height=height, template="plotly_white", font={"family": FONT, "size": 12, "color": INK}, hovermode="closest", dragmode="pan",
        margin={"l": 8, "r": 8, "t": 8, "b": 8}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "v", "y": 1.0, "x": 0, "xanchor": "left", "yanchor": "top", "font": {"size": 12}, "itemclick": False,
                "itemdoubleclick": False, "itemsizing": "constant", "bgcolor": "rgba(255,255,255,0.85)"},
        xaxis={"visible": False, "range": [-1.95, 1.95], "domain": [0.15, 1.0]}, yaxis={"visible": False, "range": [-1.22, 1.22]},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": GRID, "font": {"family": FONT, "size": 12, "color": INK}})
    return fig


def clicked_key(ev):
    """The node key behind a click on the network, or None."""
    pts = list(getattr(getattr(ev, "selection", None), "points", None) or [])
    for p in pts:
        cd = p.get("customdata") if isinstance(p, dict) else None
        if cd:
            return cd[0] if isinstance(cd, (list, tuple)) else cd
    return None


# ------------------------------------------------------------------ small charts
def line_chart(df: pd.DataFrame, x: str, series, unit: str = "usd", height: int = 280, title: str = "") -> go.Figure:
    """series: [(column, label, colour)]. One y axis; 2 px lines; a hover line across the series."""
    fig = go.Figure()
    fmt = {"usd": "$%{y:,.0f}", "count": "%{y:,.0f}", "pct": "%{y:.2%}", "days": "%{y:,.0f} days"}[unit]
    top = 0.0
    for col, lab, colr in series:
        if col not in df:
            continue
        top = max(top, float(pd.to_numeric(df[col], errors="coerce").max() or 0))
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=lab, line={"width": 2, "color": colr},
                                 marker={"size": 5, "color": colr}, hovertemplate=fmt + "<extra>" + esc(lab) + "</extra>"))
    fig.update_layout(height=height, template="plotly_white", font={"family": FONT, "size": 12, "color": INK}, hovermode="x unified",
                      margin={"l": 8, "r": 8, "t": 34 if title else 12, "b": 8},
                      title={"text": title, "font": {"size": 13}, "x": 0, "xanchor": "left"} if title else None,
                      showlegend=len(fig.data) > 1, legend={"orientation": "h", "y": 1.16, "x": 1, "xanchor": "right"},
                      xaxis={"type": "category", "showgrid": False, "tickangle": -45, "nticks": 12},
                      yaxis={"gridcolor": GRID, "rangemode": "tozero", "zeroline": False,
                             "tickformat": {"usd": "$,.0f", "count": ",.0f", "pct": ".1%", "days": ",.0f"}[unit]})
    if unit in ("count", "days") and top <= 6:
        fig.update_yaxes(dtick=1, range=[0, max(top, 1) + 0.4])
    return fig


def flow_bars(df: pd.DataFrame, x: str, col_in: str, col_out: str, title: str = "", height: int = 300, lab_in: str = "money in",
              lab_out: str = "money out") -> go.Figure:
    """Money in and out over time: green bars above zero for money in, red bars below zero for money out."""
    fig = go.Figure()
    for col, lab, colr, sign in ((col_in, lab_in, FLOW_IN, 1), (col_out, lab_out, FLOW_OUT, -1)):
        if col not in df:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(go.Bar(x=df[x], y=sign * v, name=lab, marker={"color": colr, "line": {"width": 0}}, customdata=v,
                             hovertemplate="$%{customdata:,.0f}<extra>" + esc(lab) + "</extra>"))
    fig.update_layout(height=height, template="plotly_white", font={"family": FONT, "size": 12, "color": INK}, hovermode="x unified",
                      barmode="relative", bargap=0.25, margin={"l": 8, "r": 8, "t": 34 if title else 12, "b": 8},
                      title={"text": title, "font": {"size": 13}, "x": 0, "xanchor": "left"} if title else None,
                      showlegend=True, legend={"orientation": "h", "y": 1.16, "x": 1, "xanchor": "right"},
                      xaxis={"type": "category", "showgrid": False, "tickangle": -45, "nticks": 12},
                      yaxis={"gridcolor": GRID, "zeroline": True, "zerolinecolor": MUTED, "zerolinewidth": 1, "tickformat": "$,.0f"})
    return fig


def deposit_chart(df: pd.DataFrame, x: str, bar, lines, title: str = "", height: int = 320) -> go.Figure:
    """Deposits: bars for the monthly (month-end) balance, lines for the other balance measures.
    bar: (column, label); lines: [(column, label, colour)]."""
    fig = go.Figure()
    if bar[0] in df:
        fig.add_trace(go.Bar(x=df[x], y=df[bar[0]], name=bar[1], marker={"color": "#86b6ef", "line": {"width": 0}},
                             hovertemplate="$%{y:,.0f}<extra>" + esc(bar[1]) + "</extra>"))
    for col, lab, colr in lines:
        if col in df:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=lab, line={"width": 2, "color": colr},
                                     marker={"size": 5, "color": colr}, hovertemplate="$%{y:,.0f}<extra>" + esc(lab) + "</extra>"))
    fig.update_layout(height=height, template="plotly_white", font={"family": FONT, "size": 12, "color": INK}, hovermode="x unified",
                      bargap=0.25, margin={"l": 8, "r": 8, "t": 34 if title else 12, "b": 8},
                      title={"text": title, "font": {"size": 13}, "x": 0, "xanchor": "left"} if title else None,
                      showlegend=len(fig.data) > 1, legend={"orientation": "h", "y": 1.16, "x": 1, "xanchor": "right"},
                      xaxis={"type": "category", "showgrid": False, "tickangle": -45, "nticks": 12},
                      yaxis={"gridcolor": GRID, "rangemode": "tozero", "zeroline": False, "tickformat": "$,.0f"})
    return fig


def export_html(fig: go.Figure, title: str) -> bytes:
    """The network as one self-contained interactive HTML file (hover and zoom keep working; no internet needed)."""
    f = go.Figure(fig)
    f.update_layout(title={"text": esc(title), "font": {"size": 15}, "x": 0.01, "xanchor": "left"}, margin={"l": 8, "r": 8, "t": 48, "b": 8},
                    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=None, autosize=True)
    return f.to_html(include_plotlyjs=True, full_html=True, default_height="92vh",
                     config={"displaylogo": False, "scrollZoom": True, "toImageButtonOptions": {"format": "png", "scale": 2}}).encode("utf-8")


def net_config(filename: str) -> dict:
    """Chart toolbar of the network: a camera button that saves a PNG (twice the screen resolution) and a reset-view button."""
    return {"displayModeBar": True, "displaylogo": False, "scrollZoom": True, "modeBarButtons": [["toImage", "resetScale2d"]],
            "toImageButtonOptions": {"format": "png", "filename": filename, "scale": 2}}


def show(fig, **kw):
    return st.plotly_chart(fig, **WIDE, config={"displayModeBar": False}, **kw)
