"""
Counterparty page shared by the PKG Streamlit apps (Prospect Finder, TPPP Prospect Finder): page style, number
formatting, cached Impala reads for one counterparty, the network drawing and the detail tabs.
"""
from __future__ import annotations

import html
import logging
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import prospect_core as pc

log = logging.getLogger("pkg_apps")

TTL = 6 * 3600
ORANGE, BLUE, SLATE, GREY = "#E35205", "#2F6DB5", "#1B2430", "#9AA5B1"
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
SELECTABLE = _V >= (1, 35)          # row / point selection in st.dataframe and st.plotly_chart
HAS_QP = hasattr(st, "query_params")


def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');
    html, body, .stApp, .stMarkdown, button, input, textarea, select,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
    .block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1480px; }
    .pf-title { font-size: 1.9rem !important; font-weight: 600; letter-spacing: -0.015em; color: #1B2430; line-height: 1.2; }
    .pf-sub { color: #5B6573; margin: 0.2rem 0 0.5rem 0; }
    .pf-note { display: inline-block; font-size: 0.78rem; color: #8A4B14; background: #FFF4EA;
               border: 1px solid #F5D3B5; border-radius: 6px; padding: 1px 10px; }
    .pf-chip { display: inline-block; font-size: 0.8rem; background: #EEF2F6; color: #1B2430; border-radius: 12px;
               padding: 2px 10px; margin: 0 6px 4px 0; }
    .pf-desc { color: #5B6573; font-size: 0.86rem; margin: -0.3rem 0 0.4rem 0; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
    [data-testid="stMetricLabel"] p { color: #5B6573; font-size: 0.8rem; }
    </style>""", unsafe_allow_html=True)


def viewer() -> str:
    for get in (lambda: st.user.get("email"), lambda: st.experimental_user.email):
        try:
            v = get()
            if v:
                return str(v)
        except Exception:
            pass
    return "unknown"


def chips(items):
    return "".join(f'<span class="pf-chip">{html.escape(str(x))}</span>' for x in items if x)


def fail(what: str, e: Exception):
    log.exception("%s failed", what)
    st.error(f"Couldn't load {what}. Please try again; if it keeps failing, send the time to the app owner.")


def k_usd(x) -> str:
    return "–" if pd.isna(x) else f"{x / 1e3:,.0f}"


def pct(x) -> str:
    if pd.isna(x):
        return "–"
    return "new" if np.isinf(x) else f"{100 * x:+.0f}%"


def table(df: pd.DataFrame, where=None, one_dec=(), skip=(), **kw):
    """st.dataframe with thousands separators. Formatting is display only: column sorting still uses the numbers."""
    df = df.reset_index(drop=True)
    fmt = {c: ("{:,.1f}" if c in one_dec else "{:,.0f}") for c in df.columns
           if c not in skip and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])}
    return (where or st).dataframe(df.style.format(fmt, na_rep=""), hide_index=True, **WIDE, **kw)


# ------------------------------------------------------------------ cached reads
@st.cache_data(ttl=TTL, show_spinner=False)
def cols_of(table: str):
    return pc.table_columns(table)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_trend(eid: str, months: tuple) -> pd.DataFrame:
    cols = cols_of(pc.T_CPTY_MET) or set()
    return pc._num(pc.run_sql(pc.cpty_trend_sql(eid, months, cols), "trend"), pc.trend_columns(cols))


@st.cache_data(ttl=TTL, show_spinner=False)
def load_clients(eid: str, per: tuple) -> pd.DataFrame:
    return pc.prepare_clients(pc.run_sql(pc.cpty_clients_sql(eid, per), "clients"))


@st.cache_data(ttl=TTL, show_spinner=False)
def load_accounts(eid: str, per: tuple) -> pd.DataFrame:
    rc = cols_of(pc.T_FI_ROUTING) or set()
    rcol = next((c for c in sorted(rc) if "rout" in c and "fi" not in c), None) if "fi_canonical" in rc else None
    names = {"fi_raw", "fi_canonical"} <= (cols_of(pc.T_FI_NAMES) or set())
    return pc.prepare_accounts(pc.run_sql(pc.cpty_accounts_sql(eid, per, rcol, names), "accounts"))


@st.cache_data(ttl=TTL, show_spinner=False)
def load_account_edges(eid: str, per: tuple) -> pd.DataFrame:
    return pc.run_sql(pc.cpty_account_edges_sql(eid, per), "account edges")


# ------------------------------------------------------------------ network drawing
def draw_graph(nodes: pd.DataFrame, edges: pd.DataFrame, height: int = 720) -> go.Figure:
    """Directed graph: arrows point the way money moved; width grows with the money (square root)."""
    nodes = nodes.copy()
    top = nodes[nodes.kind.isin(["client", "account"])].amt.max() or 1
    size = {"counterparty": 36, "bank": 24, "other": 20}
    nodes["size"] = [size.get(k, 0) or 10 + 16 * math.sqrt(max(a, 0) / top)
                     for k, a in zip(nodes.kind, nodes.amt.fillna(0))]
    P = nodes.set_index("id")
    fig = go.Figure()
    flows = edges[edges.flow != "holds"]
    hold = edges[edges.flow == "holds"]
    if len(hold):
        xs, ys = [], []
        for s, d in zip(hold.src, hold.dst):
            xs += [P.at[s, "x"], P.at[d, "x"], None]
            ys += [P.at[s, "y"], P.at[d, "y"], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#7B8794", "width": 1.6, "dash": "dot"},
                                 hoverinfo="skip", showlegend=False))
    plain = flows[(flows.src != "OTHER") & (flows.dst != "OTHER")]
    wmax = (plain.amt.max() if len(plain) else flows.amt.max()) or 1
    pairs = set(zip(flows.src, flows.dst))
    hx, hy, ht = [], [], []
    for e in flows.itertuples():
        x0, y0, x1, y1 = P.at[e.src, "x"], P.at[e.src, "y"], P.at[e.dst, "x"], P.at[e.dst, "y"]
        if (e.dst, e.src) in pairs:                       # money both ways: two parallel arrows
            L = math.hypot(x1 - x0, y1 - y0) or 1
            nx, ny = -(y1 - y0) / L * 0.022, (x1 - x0) / L * 0.022
            x0, y0, x1, y1 = x0 + nx, y0 + ny, x1 + nx, y1 + ny
        w = 1.0 + 9.0 * math.sqrt(min(e.amt / wmax, 1.0))
        col = "rgba(227,82,5,0.62)" if e.flow == "paid_to_it" else "rgba(47,109,181,0.62)"
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="",
                           showarrow=True, arrowhead=2, arrowsize=max(0.45, 1.5 / math.sqrt(w)), arrowwidth=w,
                           arrowcolor=col, standoff=P.at[e.dst, "size"] / 2 + 2,
                           startstandoff=P.at[e.src, "size"] / 2 + 1)
        hx.append((x0 + x1) / 2)
        hy.append((y0 + y1) / 2)
        ht.append(f"{P.at[e.src, 'label']} → {P.at[e.dst, 'label']}<br>${e.amt / 1e3:,.0f}K · "
                  f"{e.n_txn:,.0f} payments")
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker={"size": 14, "color": "rgba(0,0,0,0)"},
                             hovertext=ht, hoverinfo="text", showlegend=False))
    style = {"counterparty": (ORANGE, "Counterparty"), "bank": ("#5B6573", "Bank"), "account": ("#F2A36B", "Account"),
             "client": (BLUE, "PNC client"), "other": (GREY, "Other PNC clients")}
    for kind, (color, label) in style.items():
        d = nodes[nodes.kind == kind]
        if not len(d):
            continue
        pos = ["middle left" if x < -0.05 else "middle right" if x > 0.05 else ("bottom center" if y < 0
               else "top center") for x, y in zip(d.x, d.y)]
        inner = kind in ("bank", "counterparty", "account")
        fig.add_trace(go.Scatter(
            x=d.x, y=d.y, mode="markers" if inner else "markers+text", name=label, hovertext=d.hover,
            hoverinfo="text", text=None if inner else [(t[:30] + "…") if len(t) > 31 else t for t in d.label.astype(str)],
            textposition=pos, textfont={"size": 11, "color": SLATE},
            marker={"size": d["size"], "color": color, "line": {"width": 1.5, "color": "white"},
                    "symbol": "square" if kind == "bank" else "circle"}))
        if inner:                                         # labels on a white card so arrows don't hide them
            for t, x, y, sz in zip(d.label.astype(str), d.x, d.y, d["size"]):
                side = 0 if kind != "bank" or abs(x) < 0.05 else (1 if x > 0 else -1)   # banks: label beside
                fig.add_annotation(x=x, y=y, text=html.escape((t[:34] + "…") if len(t) > 35 else t),
                                   showarrow=False, yshift=0 if side else -(sz / 2 + 11),
                                   xshift=side * (sz / 2 + 4), xanchor={1: "left", -1: "right", 0: "center"}[side],
                                   bgcolor="rgba(255,255,255,0.88)", borderpad=1,
                                   font={"size": 12 if kind == "counterparty" else 10.5, "color": SLATE})
    for col, lab in ((ORANGE, "Client pays it"), (BLUE, "It pays the client")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line={"color": col, "width": 4}, name=lab))
    fig.update_layout(height=height, template="plotly_white", font_family=FONT, hoverlabel={"font_family": FONT},
                      margin={"l": 10, "r": 10, "t": 10, "b": 10}, plot_bgcolor="white",
                      legend={"orientation": "h", "y": 1.02, "x": 0, "title": None},
                      xaxis={"visible": False, "range": [-1.9, 1.9]},
                      yaxis={"visible": False, "range": [-1.3, 1.3], "scaleanchor": "x"})
    return fig


# ------------------------------------------------------------------ counterparty page
def render_header(r: pd.Series, per, n: int, extra_chips=()) -> None:
    """Name, chips and headline numbers of one counterparty (r = its row in the app's universe)."""
    st.markdown(f'<div class="pf-title">{html.escape(r.ent_name)}</div>', unsafe_allow_html=True)
    first = r.get("first_month")
    st.markdown(chips([f"{per[0]} to {per[-1]}", f"Bank today: {r.fi_dom}",
                       f"Banks in its last active month: {r.n_fi:.0f}" if pd.notna(r.get("n_fi")) else "",
                       f"PNC account: {r.pnc_account}", r.flow,
                       f"First seen {first}" if isinstance(first, str) else "", *extra_chips]),
                unsafe_allow_html=True)
    has_change = pd.notna(r.get("change"))
    m = st.columns(7 if has_change else 6)
    m[0].metric("PNC clients / mo", f"{r.clients:,.1f}")
    m[1].metric("TM clients paying it / mo", f"{r.tm_in:,.1f}" if pd.notna(r.get("tm_in")) else "–")
    m[2].metric("TM clients it pays / mo", f"{r.tm_out:,.1f}" if pd.notna(r.get("tm_out")) else "–")
    m[3].metric("$K in / mo", k_usd(r.amt_in))
    m[4].metric("$K out / mo", k_usd(r.amt_out))
    m[5].metric("Months active", f"{r.months_active:.0f} of {n}")
    if has_change:
        m[6].metric("PNC clients, change", pct(r.change), help="Last 3 months of the period vs the 3 before")


def render_detail_tabs(eid: str, r: pd.Series, per, n: int, first_tabs=()) -> None:
    """Network, PNC clients, accounts/banks/names and month-by-month tabs. first_tabs: [(label, render_fn)] shown
    before them (an app's own view of the counterparty)."""
    try:
        with st.spinner("Loading its clients ..."):
            CL = load_clients(eid, per)
    except Exception as e:
        fail("its clients", e)
        CL = None
    names = [t for t, _ in first_tabs] + ["Network", "PNC clients", "Accounts, banks and names", "Month by month"]
    tabs = st.tabs(names)
    for tab, (_, fn) in zip(tabs, first_tabs):
        with tab:
            fn()
    t_net, t_cli, t_acc, t_mon = tabs[len(first_tabs):]

    with t_net:
        g1, g2, g3, _ = st.columns([1.3, 1.3, 1.3, 2])
        view = g1.radio("View", ["Consolidated", "Account level"], horizontal=True, key="g_view",
                        help="Consolidated: the counterparty as one node. Account level: each of its accounts, "
                             "grouped under the bank that holds it.")
        top_n = g2.slider("Largest TM clients shown", 5, 60, 25, key="g_top")
        top_acc = g3.slider("Accounts shown", 3, 30, 12, key="g_acc") if view == "Account level" else None
        if CL is not None and len(CL):
            try:
                if view == "Consolidated":
                    nodes, edges = pc.graph_consolidated(r.ent_name, CL, top_n)
                else:
                    with st.spinner("Loading its accounts ..."):
                        AE, ACC = load_account_edges(eid, per), load_accounts(eid, per)
                    nodes, edges = pc.graph_accounts(AE, ACC, CL, top_n, top_acc)
                if len(edges):
                    st.plotly_chart(draw_graph(nodes, edges), **WIDE)
                    st.caption("Arrows point the way money moved in the period; thicker = more money. Orange: the "
                               "client pays the counterparty; blue: the counterparty pays the client. Hover an arrow "
                               "for the amount. Clients beyond the largest ones, and clients that are not TM clients, "
                               "are combined in 'Other PNC clients'.")
                else:
                    st.info("No payments with PNC clients in the period.")
            except Exception as e:
                fail("the network", e)
        else:
            st.info("No PNC clients found for this counterparty in the period.")

    with t_cli:
        if CL is not None and len(CL):
            tm, other = CL[CL.is_tm], CL[~CL.is_tm]
            c1, c2, c3 = st.columns(3)
            c1.metric("TM clients", f"{len(tm):,}")
            c2.metric("Other PNC clients", f"{len(other):,}" + (" +" if len(CL) >= 2000 else ""))
            c3.metric("Money with TM clients, $K in the period", k_usd(tm.total.sum()))
            h1, h2 = st.columns([1.55, 1])
            ct = pd.DataFrame({"client": tm.customer_name, "mdm_id": tm.mdm_id, "power id": tm.customer_pwr_id,
                               "industry": tm.industry,
                               "$K it received": (tm.paid_to_it / 1e3).round(1),
                               "$K it paid": (tm.received_from_it / 1e3).round(1),
                               "payments": (tm.n_paid + tm.n_received).round(0), "months": tm.months})
            table(ct.head(300), where=h1, one_dec=["$K it received", "$K it paid"], height=460)
            h1.caption("TM clients by money with this counterparty in the period (window totals). "
                       "'It received' = the client paid it.")
            sec = (tm.assign(side=np.select([tm.paid_to_it > 0, tm.received_from_it > 0],
                                            ["Clients paying it", "Clients it pays"], "")))
            both = tm[(tm.paid_to_it > 0) & (tm.received_from_it > 0)]
            sec = pd.concat([sec[sec.side != ""], both.assign(side="Clients it pays")])
            agg = sec.groupby(["sector", "side"]).mdm_id.nunique().reset_index(name="TM clients")
            order = agg.groupby("sector")["TM clients"].sum().sort_values().index.tolist()[-14:]
            f3 = px.bar(agg[agg.sector.isin(order)], x="TM clients", y="sector", color="side", orientation="h",
                        barmode="group", category_orders={"sector": order[::-1]}, template="plotly_white",
                        height=460, color_discrete_map={"Clients paying it": ORANGE, "Clients it pays": BLUE})
            f3.update_xaxes(tickformat=",")
            f3.update_layout(font_family=FONT, title="Industries of its TM clients", yaxis_title="",
                             legend={"orientation": "h", "y": -0.15, "title": None},
                             margin={"l": 10, "r": 10, "t": 40, "b": 10})
            h2.plotly_chart(f3, **WIDE)
            h2.caption("NAICS sector of the TM clients. Clients paying it hint at what it sells and to whom; "
                       "clients it pays hint at what it buys.")
        else:
            st.info("No PNC clients found for this counterparty in the period.")

    with t_acc:
        try:
            with st.spinner("Loading its accounts ..."):
                ACC = load_accounts(eid, per)
        except Exception as e:
            fail("its accounts", e)
            ACC = pd.DataFrame()
        if len(ACC):
            total = ACC.amt.sum() or 1
            bk = (ACC.groupby("bank").agg(accounts=("cpty_id", "nunique"), amt=("amt", "sum"),
                                          names=("cpty_name", "nunique"))
                     .sort_values("amt", ascending=False).reset_index())
            bk = pd.DataFrame({"bank": bk.bank, "accounts": bk.accounts, "names": bk.names,
                               "$K in the period": (bk.amt / 1e3).round(0), "share": (100 * bk.amt / total).round(1)})
            nm = (ACC.groupby("cpty_name").agg(accounts=("cpty_id", "nunique"), banks=("bank", "nunique"),
                                               amt=("amt", "sum"))
                     .sort_values("amt", ascending=False).reset_index())
            nm = pd.DataFrame({"name on the payments": nm.cpty_name, "accounts": nm.accounts, "banks": nm.banks,
                               "$K in the period": (nm.amt / 1e3).round(0)})
            ac = (ACC.groupby(["cpty_id", "account", "bank"])
                     .agg(names=("cpty_name", lambda s: "; ".join(sorted(set(s)))),
                          n_names=("cpty_name", "nunique"), amt=("amt", "sum"),
                          n_txn=("n_txn", "sum"), first=("first_month", "min"), last=("last_month", "max"))
                     .sort_values("amt", ascending=False).reset_index())
            ac = pd.DataFrame({"account": ac.account, "bank": ac.bank, "names": ac.n_names, "names used": ac.names,
                               "$K in the period": (ac.amt / 1e3).round(0), "payments": ac.n_txn.round(0),
                               "first month": ac["first"], "last month": ac["last"]})
            c1, c2, c3 = st.columns(3)
            c1.metric("Banks", f"{len(bk):,}")
            c2.metric("Accounts", f"{ACC.cpty_id.nunique():,}" + (" +" if len(ACC) >= 5000 else ""))
            c3.metric("Names it appeared under", f"{len(nm):,}")
            s1, s2 = st.columns(2)
            s1.markdown("**Banks holding its accounts**")
            table(bk, where=s1, skip=["share"], height=260, column_config={
                "share": st.column_config.ProgressColumn("share %", min_value=0, max_value=100, format="%.0f")})
            s2.markdown("**Names it appeared under (merged into this counterparty)**")
            table(nm, where=s2, height=260)
            st.markdown("**Accounts**")
            table(ac.head(500), height=320, column_config={"names used": st.column_config.TextColumn(width="large")})
            st.caption("Accounts active with PNC clients in the period. Account numbers are masked to the routing "
                       "number and the last 4 digits. Names are merged by cleaned business name across accounts and "
                       "banks; a wrong merge shows up here as an unrelated name.")
        else:
            st.info("No accounts found for this counterparty in the period.")

    with t_mon:
        months = tuple(pc.month_range(max(pc.FIRST_MONTH, pc.add_months(per[0], -6)), pc.LAST_MONTH))
        try:
            TR = load_trend(eid, months)
        except Exception as e:
            fail("its months", e)
            TR = pd.DataFrame()
        panels = [(t, u, [(c, lab) for c, lab in ser if c in TR], d) for t, u, ser, d in pc.TREND_PANELS]
        panels = [p for p in panels if p[2] and TR[[c for c, _ in p[2]]].notna().any().any()]
        if len(TR) and panels:
            st.caption(f"{months[0]} to {months[-1]}; shaded: the selected period. Amounts jump in Jul-Aug 2026 in "
                       "the source data. The note under each chart says what it measures.")
            colors = [ORANGE, BLUE, "#2E8B57", "#8E6CC0", "#C9A227", GREY]
            for i in range(0, len(panels), 2):
                cols = st.columns(2)
                for tgt, (title, unit, ser, desc) in zip(cols, panels[i:i + 2]):
                    scale = {"pct": 100.0, "usd_k": 1e-3}.get(unit, 1.0)
                    data = TR[["mon"] + [c for c, _ in ser]].melt(id_vars="mon", var_name="s", value_name="v")
                    data["s"] = data.s.map(dict(ser))
                    data["v"] = data.v * scale
                    fg = px.line(data, x="mon", y="v", color="s", markers=True, template="plotly_white", height=290,
                                 color_discrete_sequence=colors)
                    fg.update_traces(hovertemplate="%{x}: %{y:,.1f}" if unit == "pct" else "%{x}: %{y:,.0f}")
                    fg.update_yaxes(tickformat=",", rangemode="tozero", title=None)
                    fg.update_layout(font_family=FONT, title={"text": title, "font": {"size": 14}}, xaxis_title="",
                                     legend={"orientation": "h", "y": -0.25, "title": None},
                                     margin={"l": 10, "r": 10, "t": 40, "b": 10})
                    fg.add_vrect(x0=per[0], x1=per[-1], fillcolor="#EEF2F6", opacity=0.6, line_width=0,
                                 layer="below")
                    tgt.plotly_chart(fg, **WIDE)
                    tgt.caption(desc)
        else:
            st.info("No monthly data for this counterparty.")
