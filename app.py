"""
Prospect Finder (draft 2): outside companies our clients pay or are paid by.

Two pages in one script:
  list          filters (with presets), a sortable list and a chart; a click on a row or a point opens ...
  counterparty  one counterparty: its network with PNC clients, clients and their industries, accounts, banks and
                names, month by month
Run:  streamlit run app.py      (reads Impala through dbi; logic in prospect_core.py)
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

st.set_page_config(page_title="Prospect Finder", layout="wide", initial_sidebar_state="collapsed")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("prospect_finder")

TTL = 6 * 3600
ORANGE, BLUE, SLATE, GREY = "#E35205", "#2F6DB5", "#1B2430", "#9AA5B1"
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}
SELECTABLE = _V >= (1, 35)          # row / point selection in st.dataframe and st.plotly_chart
HAS_QP = hasattr(st, "query_params")

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

# ------------------------------------------------------------------ state: filters survive a visit to a counterparty
FKEYS = {f"f_{k}": v for k, v in pc.FILTER_DEFAULTS.items()}
FKEYS.update({"f_start": pc.DEFAULT_START, "f_end": pc.DEFAULT_END, "preset": "None"})
for k, v in FKEYS.items():
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]       # keeps widget values while their page is not shown
    else:
        st.session_state[k] = list(v) if isinstance(v, list) else v
st.session_state.setdefault("nav", 0)


def current_cpty():
    v = st.query_params.get("cpty") if HAS_QP else st.session_state.get("cpty")
    return v or None


def go_to(eid):
    st.session_state["nav"] += 1                         # fresh list / chart widgets: no stale selection
    st.session_state["cpty"] = eid
    if HAS_QP:
        if eid:
            st.query_params["cpty"] = eid
        elif "cpty" in st.query_params:
            del st.query_params["cpty"]
    st.rerun()


def apply_preset():
    p = pc.PRESETS[st.session_state["preset"]]
    for k, v in pc.FILTER_DEFAULTS.items():
        st.session_state[f"f_{k}"] = list(v) if isinstance(v, list) else v
    for k, v in p["set"].items():
        st.session_state[f"f_{k}"] = len(pc.period(st.session_state.f_start, st.session_state.f_end)) if v == "all" else v


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


def search_bar(where, U: pd.DataFrame, per) -> None:
    """Find a counterparty by name from any page; the matches open its page."""
    nav = st.session_state.nav
    q = where.text_input("Find a counterparty by name", key=f"q_{nav}", placeholder="Type part of its name",
                         help=f"Searches every counterparty with at least {pc.UNIVERSE_MIN_CLIENTS} PNC clients a "
                              f"month in {per[0]} to {per[-1]}, whatever the filters.")
    if not q.strip():
        return
    R = pc.search_names(U, q)
    if not len(R):
        st.info(f'No counterparty name contains "{q.strip()}" in {per[0]} to {per[-1]}.')
        return
    V = pd.DataFrame({"counterparty": R.ent_name, "bank today": R.fi_dom, "PNC clients / mo": R.clients.round(1),
                      "$K in / mo": (R.amt_in / 1e3).round(0), "$K out / mo": (R.amt_out / 1e3).round(0),
                      "first seen": R.first_month, "ent_id": R.ent_id})
    st.markdown(f"**Matches for \"{html.escape(q.strip())}\"** ({len(V)} shown)")
    if SELECTABLE:
        ev = table(V, one_dec=["PNC clients / mo"], height=min(38 + 35 * len(V), 320), on_select="rerun",
                   selection_mode="single-row", key=f"search_{nav}", column_config={"ent_id": None})
        rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
        st.caption("Tick a row to open the counterparty.")
        if rows and rows[0] < len(V):
            go_to(V.ent_id.iloc[rows[0]])
    else:
        names = dict(zip(V.ent_id, V.counterparty))
        pick = st.selectbox("Open", [""] + list(V.ent_id), key=f"spick_{nav}",
                            format_func=lambda e: "Choose …" if not e else names[e])
        if pick:
            go_to(pick)
    st.divider()


# ------------------------------------------------------------------ cached reads
@st.cache_data(ttl=TTL, show_spinner=False)
def cols_of(table: str):
    return pc.table_columns(table)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_universe(per: tuple) -> pd.DataFrame:
    cols = cols_of(pc.T_CPTY_MET)
    if cols is None:
        raise RuntimeError("counterparty metrics unavailable")
    sql = pc.universe_sql(per, pc.UNIVERSE_MIN_CLIENTS, cols, cols_of(pc.T_ENT_DIM))
    return pc.prepare_universe(pc.run_sql(sql, "universe"))


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


def get_universe(per):
    try:
        with st.spinner(f"Loading counterparties for {per[0]} to {per[-1]}. The first load of a period takes a "
                        "minute or two; after that it is quick."):
            return load_universe(per)
    except Exception as e:
        fail("the counterparties", e)
        st.stop()


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


# ------------------------------------------------------------------ list page
def filter_chips(f: pc.Filters, n: int) -> list[str]:
    few = lambda xs: ", ".join(xs[:3]) + (f" +{len(xs) - 3}" if len(xs) > 3 else "")
    return [f'name has "{f.name_contains.strip()}"' if f.name_contains.strip() else "",
            f"PNC account: {f.pnc_account}" if f.pnc_account != "Any" else "",
            f"money flow: {few(f.flows)}" if f.flows else "",
            f"bank today: {few(f.banks)}" if f.banks else "",
            f"not at: {few(f.exclude_banks)}" if f.exclude_banks else "",
            f"at least {f.min_banks} banks" if f.min_banks else "",
            f"at least {f.min_clients:g} PNC clients a month",
            f"at least {f.min_tm:g} TM clients a month" if f.min_tm else "",
            f"at least ${f.min_amt_k:,.0f}K a month" if f.min_amt_k else "",
            f"active at least {f.min_months} of {n} months" if f.min_months else "",
            f"first seen: {f.first_seen.lower()}" if f.first_seen != "Any time" else "",
            f"PNC clients: {f.change.lower()}" if f.change != "Any" else "",
            "hiding PNC, government, tax and card networks" if f.hide_pnc_gov_card else "",
            "hiding generic descriptors" if f.hide_generic else "",
            f"sorted by {f.sort.lower()}"]


def list_view(T: pd.DataFrame) -> pd.DataFrame:
    v = pd.DataFrame({"rank": np.arange(1, len(T) + 1), "counterparty": T.ent_name.to_numpy(),
                      "bank today": T.fi_dom.to_numpy()})
    if "n_fi" in T:
        v["banks"] = T.n_fi.to_numpy()
    v["PNC account"] = T.pnc_account.to_numpy()
    v["money flow"] = T.flow.to_numpy()
    v["PNC clients / mo"] = T.clients.round(1).to_numpy()
    if "tm_in" in T:
        v["TM clients paying it / mo"] = T.tm_in.round(1).to_numpy()
        v["TM clients it pays / mo"] = T.tm_out.round(1).to_numpy()
    v["$K in / mo"] = (T.amt_in / 1e3).round(0).to_numpy()
    v["$K out / mo"] = (T.amt_out / 1e3).round(0).to_numpy()
    v["PNC clients, change"] = [pct(x) for x in T.change]
    v["first seen"] = T.first_month.to_numpy()
    v["months active"] = T.months_active.to_numpy()
    v["ent_id"] = T.ent_id.to_numpy()
    return v


LIST_DEC = ["PNC clients / mo", "TM clients paying it / mo", "TM clients it pays / mo"]


def render_list(U: pd.DataFrame, per, n: int):
    h1, h2 = st.columns([2.3, 1])
    h1.markdown('<div class="pf-title">Prospect Finder</div><p class="pf-sub">Outside companies our clients pay or '
                'are paid by. Filter the list, then open a counterparty for its accounts, banks, clients and '
                'network.</p><span class="pf-note">Internal working draft · shows client names · prospecting use '
                'pending the permissible-use review</span>', unsafe_allow_html=True)
    search_bar(h2, U, per)
    st.write("")
    months = pc.month_range(pc.FIRST_MONTH, pc.LAST_MONTH)[::-1]
    with st.expander("Filters", expanded=True):
        r = st.columns([1.5, 0.85, 0.85, 1.2])
        r[0].selectbox("Preset", list(pc.PRESETS), key="preset", on_change=apply_preset,
                       help="Sets the filters below to a predefined combination; you can change them afterwards.")
        r[1].selectbox("Period starts", months, key="f_start")
        r[2].selectbox("Period ends", months, key="f_end",
                       help="Amounts jump in Jul-Aug 2026 in the source data (open check); June 2026 is the last "
                            "month before the jump.")
        r[3].selectbox("Sort by", list(pc.SORTS), key="f_sort")
        st.markdown(f'<p class="pf-desc">{html.escape(pc.PRESETS[st.session_state.preset]["desc"])}</p>',
                    unsafe_allow_html=True)
        bank_n = U.fi_dom.fillna("(unknown)").value_counts()
        bank_opts = list(bank_n.index[:400])
        for k in ("f_banks", "f_exclude_banks"):
            st.session_state[k] = [b for b in st.session_state[k] if b in bank_opts]
        st.session_state.f_min_months = min(st.session_state.f_min_months, n)
        a, b, c, d = st.columns(4)
        a.markdown("**Counterparty**")
        a.text_input("Name contains", key="f_name_contains")
        if (U.pnc_account != "Unknown").any():
            a.radio("Has an account at PNC", ["Any", "Yes", "No"], key="f_pnc_account", horizontal=True,
                    help="Yes = its cleaned name matches a PNC client (any client, TM or not).")
        a.multiselect("Money flow", pc.FLOWS, key="f_flows",
                      help="Receives from PNC clients = 80% or more of its money with PNC clients comes in from "
                           "them; pays PNC clients = 80% or more goes out to them; both ways = in between.")
        a.checkbox("Hide PNC, government, tax and card networks", key="f_hide_pnc_gov_card")
        a.checkbox("Hide generic descriptors (MERCH BNKCD, ACH TRANSFER …)", key="f_hide_generic")
        b.markdown("**Bank**")
        b.multiselect("Bank today", bank_opts, key="f_banks", format_func=lambda x: f"{x} ({bank_n[x]:,})",
                      help="The bank holding its main account in its last active month of the period.")
        b.multiselect("Not at these banks", bank_opts, key="f_exclude_banks",
                      format_func=lambda x: f"{x} ({bank_n[x]:,})")
        if "n_fi" in U:
            b.selectbox("Banks it uses, at least", [0, 1, 2, 3, 4, 5], key="f_min_banks",
                        format_func=lambda k: "any" if k == 0 else str(k),
                        help="Number of banks holding its accounts in its last active month of the period.")
        c.markdown("**Size and activity**")
        c.number_input("PNC clients a month, at least", min_value=float(pc.UNIVERSE_MIN_CLIENTS), step=1.0,
                       key="f_min_clients", help="Distinct PNC clients paying it or paid by it, averaged over "
                                                 "the months it was active in the period.")
        if "tm_all" in U:
            c.number_input("TM clients a month, at least", min_value=0.0, step=1.0, key="f_min_tm",
                           help="TM clients paying it plus TM clients it pays, averaged over its active months.")
        c.number_input("Money with PNC clients, $K a month, at least", min_value=0.0, step=10.0, key="f_min_amt_k",
                       help="Money in plus money out, averaged over the months it was active in the period.")
        c.slider("Active at least … months of the period", 0, n, key="f_min_months")
        d.markdown("**New and growing**")
        d.selectbox("First seen with PNC clients", list(pc.FIRST_SEEN), key="f_first_seen",
                    help=f"The first month it moved money with any PNC client, counted back from the end of the "
                         f"period. History starts {pc.FIRST_MONTH}, so 'any time' includes older counterparties.")
        d.selectbox("Change in PNC clients", list(pc.CHANGE), key="f_change",
                    help="Average PNC clients a month in the last 3 months of the period vs the 3 months before. "
                         "Counterparties with none in the 3 months before count as new (growing).")

    f = pc.Filters(**{k: st.session_state[f"f_{k}"] for k in pc.FILTER_DEFAULTS})
    D = pc.sort_frame(U[pc.apply_filters(U, f, per[-1])], f.sort)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Counterparties", f"{len(D):,}")
    k2.metric("Money with PNC clients, $M a month", f"{D.amt.sum() / 1e6:,.1f}")
    k3.metric("Without a PNC account", f"{int((D.pnc_account == 'No').sum()):,}")
    k4.metric("First seen in the last 6 months", f"{int((D.first_month >= pc.add_months(per[-1], -5)).sum()):,}")
    st.markdown(chips([f"{per[0]} to {per[-1]}"] + filter_chips(f, n)), unsafe_allow_html=True)

    tab_list, tab_chart = st.tabs(["List", "Chart"])
    nav = st.session_state.nav
    with tab_list:
        l1, l2, _ = st.columns([1, 1.4, 4])
        top_n = l1.selectbox("Show top", [100, 250, 500, 1000], index=1)
        V = list_view(D.head(top_n))
        num = st.column_config.NumberColumn
        cfg = {"$K in / mo": num(help="Money it received from PNC clients, per active month"),
               "$K out / mo": num(help="Money it paid to PNC clients, per active month"),
               "PNC clients, change": st.column_config.TextColumn(
                   help="Last 3 months of the period vs the 3 before"),
               "counterparty": st.column_config.TextColumn(width="medium"), "ent_id": None}
        l2.download_button("Download this list (CSV)", V.drop(columns="ent_id").to_csv(index=False).encode(),
                           file_name=f"prospects_{per[-1]}.csv", mime="text/csv", on_click=pc.log_event,
                           args=(viewer(), "export", " | ".join(x for x in filter_chips(f, n) if x), len(V)))
        if SELECTABLE:
            ev = table(V, one_dec=LIST_DEC, height=560, column_config=cfg, on_select="rerun",
                       selection_mode="single-row", key=f"list_{nav}")
            rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
            st.caption("Tick a row to open the counterparty. Per-month figures are averages over the months it was "
                       "active in the period.")
            if rows and rows[0] < len(V):
                go_to(V.ent_id.iloc[rows[0]])
        else:
            table(V, one_dec=LIST_DEC, height=560, column_config=cfg)
            names = dict(zip(V.ent_id, V["rank"].astype(str) + "  " + V.counterparty))
            pick = st.selectbox("Open a counterparty", [""] + list(V.ent_id), key=f"pick_{nav}",
                                format_func=lambda e: "Choose …" if not e else f"#{names[e]}")
            if pick:
                go_to(pick)

    with tab_chart:
        Pt = D.head(5000)
        if not len(Pt):
            st.info("No counterparties match the filters.")
        else:
            fig = px.scatter(Pt.assign(k=Pt.amt / 1e3), x="clients", y="k", color="pnc_account",
                             custom_data=["ent_id"], log_x=True, log_y=True, hover_name="ent_name",
                             hover_data={"fi_dom": True, "flow": True, "clients": ":.1f", "k": ":,.0f",
                                         "pnc_account": False},
                             color_discrete_map={"No": ORANGE, "Yes": BLUE, "Unknown": GREY},
                             labels={"clients": "PNC clients a month", "k": "Money with PNC clients, $K a month",
                                     "fi_dom": "bank today", "flow": "money flow", "pnc_account": "PNC account"},
                             template="plotly_white", height=580, opacity=0.75)
            fig.update_traces(marker={"size": 7, "line": {"width": 0}})
            fig.update_xaxes(tickformat=",")
            fig.update_yaxes(tickformat=",")
            fig.update_layout(font_family=FONT, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                              legend={"orientation": "h", "y": 1.06, "title": {"text": "PNC account "}})
            if SELECTABLE:
                ev = st.plotly_chart(fig, **WIDE, on_select="rerun", selection_mode="points", key=f"chart_{nav}")
                pts = list(getattr(getattr(ev, "selection", None), "points", None) or [])
                st.caption(f"Top {len(Pt):,} counterparties in the list order. Click a point to open it.")
                if pts:
                    cd = pts[0].get("customdata")
                    if cd:
                        go_to(cd[0] if isinstance(cd, (list, tuple)) else cd)
            else:
                st.plotly_chart(fig, **WIDE)


# ------------------------------------------------------------------ counterparty page
def render_cpty(eid: str, U: pd.DataFrame, per, n: int):
    b1, _, b3 = st.columns([1, 1.3, 1])
    if b1.button("← Back to the list"):
        go_to(None)
    search_bar(b3, U, per)
    row = U[U.ent_id == eid]
    if not len(row):
        st.warning("This counterparty has no activity with PNC clients in the selected period.")
        return
    r = row.iloc[0]
    if st.session_state.get("_logged") != eid:
        st.session_state["_logged"] = eid
        pc.log_event(viewer(), "open counterparty", r.ent_name)
    st.markdown(f'<div class="pf-title">{html.escape(r.ent_name)}</div>', unsafe_allow_html=True)
    st.markdown(chips([f"{per[0]} to {per[-1]}", f"Bank today: {r.fi_dom}",
                       f"Banks in its last active month: {r.n_fi:.0f}" if "n_fi" in r and pd.notna(r.n_fi) else "",
                       f"PNC account: {r.pnc_account}", r.flow,
                       f"First seen {r.first_month}" if isinstance(r.first_month, str) else ""]),
                unsafe_allow_html=True)
    m = st.columns(7)
    m[0].metric("PNC clients / mo", f"{r.clients:,.1f}")
    m[1].metric("TM clients paying it / mo", f"{r.tm_in:,.1f}" if "tm_in" in r else "–")
    m[2].metric("TM clients it pays / mo", f"{r.tm_out:,.1f}" if "tm_out" in r else "–")
    m[3].metric("$K in / mo", k_usd(r.amt_in))
    m[4].metric("$K out / mo", k_usd(r.amt_out))
    m[5].metric("Months active", f"{r.months_active:.0f} of {n}")
    m[6].metric("PNC clients, change", pct(r.change), help="Last 3 months of the period vs the 3 before")

    try:
        with st.spinner("Loading its clients ..."):
            CL = load_clients(eid, per)
    except Exception as e:
        fail("its clients", e)
        CL = None
    t_net, t_cli, t_acc, t_mon = st.tabs(["Network", "PNC clients", "Accounts, banks and names", "Month by month"])

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

PER = pc.period(st.session_state.f_start, st.session_state.f_end)
N_MONTHS = len(PER)
UNIV = get_universe(PER)
eid = current_cpty()
if eid:
    render_cpty(eid, UNIV, PER, N_MONTHS)
else:
    render_list(UNIV, PER, N_MONTHS)
