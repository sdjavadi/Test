"""
Prospect Finder (first draft): outside companies our clients pay or are paid by, filtered and ranked.

Run:  streamlit run app.py
Reads Impala through dbi (no Spark). Data and ranking logic live in prospect_core.py.
"""
from __future__ import annotations

import datetime as dt
import html
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import prospect_core as pc

st.set_page_config(page_title="Prospect Finder", layout="wide", initial_sidebar_state="collapsed")
TTL = 6 * 3600
REL_COLORS = {"New to bank": "#E35205", "PNC client (same name)": "#2F6DB5", "TM client (same name)": "#6B7A8C",
              "Unknown": "#B8C1CC"}
FONT = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"
# full-width elements: width="stretch" on newer Streamlit, use_container_width on older versions
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; }
code, pre { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1480px; }
.pf-title { font-size: 1.9rem !important; font-weight: 600; letter-spacing: -0.015em; margin: 0; color: #1B2430; line-height: 1.2; }
.pf-sub { color: #5B6573; margin: 0.2rem 0 0.5rem 0; }
.pf-note { display: inline-block; font-size: 0.78rem; color: #8A4B14; background: #FFF4EA;
           border: 1px solid #F5D3B5; border-radius: 6px; padding: 1px 10px; }
.pf-lens { color: #5B6573; font-size: 0.88rem; margin-top: -0.4rem; }
.pf-chip { display: inline-block; font-size: 0.8rem; background: #EEF2F6; color: #1B2430; border-radius: 12px;
           padding: 2px 10px; margin: 0 6px 4px 0; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
[data-testid="stMetricLabel"] p { color: #5B6573; font-size: 0.8rem; }
h3 { font-weight: 600; letter-spacing: -0.01em; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached Impala reads
@st.cache_data(ttl=TTL, show_spinner=False)
def cols_of(table: str):
    return pc.table_columns(table)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_universe(win: tuple, min_clients: int) -> pd.DataFrame:
    cols = cols_of(pc.T_CPTY_MET)
    if cols is None:
        raise RuntimeError(f"cannot read {pc.T_CPTY_MET}")
    df = pc.run_sql(pc.universe_sql(win, min_clients, cols, cols_of(pc.T_ENT_DIM)), "universe")
    return pc.prepare_universe(df, len(win))


@st.cache_data(ttl=TTL, show_spinner=False)
def load_tppp():
    for t in pc.TPPP_TABLES:
        c = cols_of(t)
        if c and {"ent_id", "score"} <= c:
            df = pc.run_sql(pc.tppp_sql(t, c), "TPPP scores").drop_duplicates("ent_id")
            df["tppp_score"] = 100 * pd.to_numeric(df.tppp_score, errors="coerce")
            return t, df
    return None, None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def addr_snapshot():
    if pc.ADDR["snapshot"]:
        return str(pc.ADDR["snapshot"])
    v = pc.run_sql(pc.latest_addr_snapshot_sql(), "address snapshot").iloc[0, 0]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.isoformat()[:10] if hasattr(v, "isoformat") else str(v)


@st.cache_data(ttl=TTL, show_spinner=False)
def load_segment(win: tuple, min_clients: int, seg: pc.Segment, snap):
    return pc.prepare_segment(pc.run_sql(pc.segment_sql(win, min_clients, seg, snap), "segment"))


@st.cache_data(ttl=TTL, show_spinner=False)
def load_trend(ent_id: str, months: tuple) -> pd.DataFrame:
    return pc._num(pc.run_sql(pc.profile_trend_sql(ent_id, months), "profile trend"),
                   ["amt_in", "amt_out", "n_cust_in", "n_cust_out", "n_cust_all"])


@st.cache_data(ttl=TTL, show_spinner=False)
def load_clients(ent_id: str, win: tuple) -> pd.DataFrame:
    return pc._num(pc.run_sql(pc.profile_clients_sql(ent_id, win), "profile clients"),
                   ["paid_to_it", "received_from_it", "months"])


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


def filter_chips(f: pc.Filters, floor: int) -> list[str]:
    """Active filters in words, so they stay visible when the Filters panel is closed."""
    few = lambda xs: ", ".join(xs[:3]) + (f" +{len(xs) - 3}" if len(xs) > 3 else "")
    return [f'name has "{f.name_contains.strip()}"' if f.name_contains.strip() else "",
            f"name suggests: {few(f.name_types)}" if f.name_types else "",
            f"relationship: {few(f.relationships)}" if f.relationships else "",
            f"money flow: {few(f.roles)}" if f.roles else "",
            f"bank today: {few(f.banks)}" if f.banks else "",
            f"not at: {few(f.exclude_banks)}" if f.exclude_banks else "",
            f"at least {f.min_banks} banks" if f.min_banks else "",
            f"at least {f.min_clients:g} PNC clients a month" if f.min_clients > floor else "",
            f"at least {f.min_tm:g} TM clients a month" if f.min_tm else "",
            f"at least ${f.min_amt_k:,.0f}K a month" if f.min_amt_k else "",
            f"active at least {f.min_months} months" if f.min_months else "",
            "hiding PNC, government, tax and card networks" if f.hide_pnc_gov_card else "",
            "hiding generic descriptors" if f.hide_generic else "",
            "known processors only" if f.known_proc_only else "",
            "TPPP-scored only" if f.tppp_scored_only else ""]


# ------------------------------------------------------------------ header and main controls
st.markdown('<div class="pf-title">Prospect Finder</div>'
            '<p class="pf-sub">Outside companies our clients pay or are paid by, filtered and ranked by what you care '
            'about.</p><span class="pf-note">Internal working draft · shows client names · prospecting use pending the '
            'permissible-use review</span>', unsafe_allow_html=True)
st.write("")

months_all = pc.month_range(pc.FIRST_MONTH, pc.LAST_MONTH)
c1, c2, c3, c4 = st.columns([3.4, 1.1, 0.9, 1.4])
lens = c1.radio("Lens", list(pc.LENSES), horizontal=True, key="lens")
end = c2.selectbox("Window ends", months_all[::-1], index=months_all[::-1].index(pc.DEFAULT_END),
                   help="Amounts jump in Jul-Aug 2026 in the source data; June 2026 is the last stable month.")
n_months = c3.selectbox("Months", pc.WINDOW_LENGTHS, index=pc.WINDOW_LENGTHS.index(pc.DEFAULT_LEN))
floor = c4.selectbox("Companies with at least", pc.MIN_CLIENT_CHOICES,
                     index=pc.MIN_CLIENT_CHOICES.index(pc.DEFAULT_MIN_CLIENTS),
                     format_func=lambda k: f"{k} PNC clients a month",
                     help="Average PNC clients a month, paying it or paid by it. Lower floors pull more companies "
                          "and take longer the first time.")
st.markdown(f'<p class="pf-lens">{pc.LENSES[lens]["desc"]}</p>', unsafe_allow_html=True)
win = pc.window(end, n_months)

try:
    with st.spinner(f"Reading companies from Impala for {win[0]} to {win[-1]}. The first load of a window takes a "
                    "minute or two; after that it is cached for 6 hours."):
        U = load_universe(win, floor)
except Exception as e:
    st.error(f"Could not load the companies: {e}")
    st.stop()

n_window = len(U)
tppp_table, TP = load_tppp()
if TP is not None:
    U = U.merge(TP, on="ent_id", how="left")

# ------------------------------------------------------------------ who they deal with (client segment)
seg = st.session_state.get("segment")
with st.expander("Who they deal with: industry and location of their PNC clients", expanded=seg is not None):
    st.caption("Counts the TM clients each company deals with in the industries and states you choose. Location "
               "means where those clients are (client address), not where the company is. Runs one Impala query "
               "per choice, cached afterwards.")
    with st.form("segment_form"):
        s1, s2, s3, s4 = st.columns([2.2, 1.6, 1.2, 0.9])
        sectors = s1.multiselect("Industry of the clients (NAICS sector)", list(pc.SECTORS),
                                 default=list(seg.sectors) if seg else [],
                                 format_func=lambda k: f"{k} {pc.SECTORS[k]}")
        states = s2.multiselect("State of the clients", pc.STATES, default=list(seg.states) if seg else [])
        side = s3.radio("The company", ["sells to", "buys from", "either way"],
                        index=["sells to", "buys from", "either way"].index(seg.side) if seg else 0,
                        help="sells to = those clients pay it; buys from = it pays those clients")
        min_seg = s4.number_input("Min clients in segment", 1, 1000, seg.min_clients if seg else 2)
        b1, b2, _ = st.columns([1, 1, 6])
        apply = b1.form_submit_button("Apply", type="primary")
        clear = b2.form_submit_button("Clear")
    if apply:
        st.session_state["segment"] = (pc.Segment(tuple(sectors), tuple(states), side, int(min_seg))
                                       if (sectors or states) else None)
        st.rerun()
    if clear:
        st.session_state["segment"] = None
        st.rerun()

seg = st.session_state.get("segment")
seg_note = ""
if seg is not None:
    try:
        snap = addr_snapshot() if seg.states else None
        with st.spinner(f"Counting TM clients that the companies {seg.label()} ..."):
            SG = load_segment(win, floor, seg, snap)
        U = U.merge(SG, on="ent_id", how="inner")
        seg_note = f"Segment: companies that {seg.label()} (at least {seg.min_clients}, {win[0]} to {win[-1]})"
    except Exception as e:
        st.error(f"The segment query failed, so it is not applied: {e}. Location needs the address table settings "
                 "in prospect_core.ADDR.")
        seg = None

# ------------------------------------------------------------------ filters
f = pc.Filters()
with st.expander("Filters", expanded=False):
    a, b, c, d = st.columns(4)
    a.markdown("**Company**")
    f.name_contains = a.text_input("Name contains", "")
    f.name_types = a.multiselect("Name suggests", list(pc.NAME_TYPES),
                                 help="From words in the name only, e.g. FREIGHT, DENTAL, PAYROLL. A hint, not an "
                                      "industry code.")
    f.relationships = a.multiselect("Relationship to PNC", [r for r in pc.RELATIONSHIPS if r in set(U.relationship)],
                                    help="Same name = the cleaned name equals a PNC organisation's name. 'New to bank' "
                                         "means no PNC client was found by name.")
    f.roles = a.multiselect("Money flow", pc.ROLES, help="Receives from PNC clients = 80%+ of its money with our "
                                                         "clients comes in from them (it sells to them).")
    b.markdown("**Bank**")
    bank_n = U.fi_dom.fillna("(unknown)").value_counts()
    bank_opts = list(bank_n.index[:300])
    f.banks = b.multiselect("Bank today", bank_opts, format_func=lambda x: f"{x} ({bank_n[x]:,})",
                            help="The bank holding its main account in the last active month of the window.")
    f.exclude_banks = b.multiselect("Not at these banks", bank_opts, format_func=lambda x: f"{x} ({bank_n[x]:,})")
    if "n_fi" in U:
        f.min_banks = b.selectbox("Banks it uses, at least", [0, 1, 2, 3, 4, 5], index=0,
                                  format_func=lambda k: "any" if k == 0 else str(k))
    c.markdown("**Size and activity**")
    f.min_clients = c.number_input("PNC clients a month, at least", 0.0, 1e6, float(floor), step=1.0)
    if "tm_all" in U:
        f.min_tm = c.number_input("TM clients a month, at least", 0.0, 1e6, 0.0, step=1.0)
    f.min_amt_k = c.number_input("Money with PNC clients, $K a month, at least", 0.0, 1e9, 0.0, step=10.0)
    f.min_months = c.slider("Months active in the window, at least", 0, n_months, 0)
    d.markdown("**Screens**")
    f.hide_pnc_gov_card = d.checkbox("Hide PNC, government, tax and card networks", True)
    f.hide_generic = d.checkbox("Hide generic descriptors (MERCH BNKCD, ACH TRANSFER …)", True)
    f.known_proc_only = d.checkbox("Known processors only", False, help="ADP, Paychex, Stripe, Fiserv … (seed list)")
    if lens == "TPPP-like processors" and TP is not None:
        f.tppp_scored_only = True
        d.caption(f"TPPP lens: only companies scored by {tppp_table.split('.')[-1]}.")

# ------------------------------------------------------------------ ranking weights
avail = pc.usable_measures(U)
with st.expander("Ranking weights", expanded=False):
    st.caption("Each measure becomes a percentile among the companies on screen; the score is their weighted mean "
               "(0-100). The lens sets the starting weights; change them freely.")
    wcols = st.columns(4)
    weights = {}
    for i, k in enumerate(pc.MEASURES):
        default = pc.LENSES[lens]["w"].get(k, 0)
        if k in avail:
            weights[k] = wcols[i % 4].slider(pc.MEASURES[k][0], 0, 5, int(default), key=f"w_{lens}_{k}")
        else:
            wcols[i % 4].caption(f"{pc.MEASURES[k][0]}: " + ("apply a segment first" if k.startswith("seg")
                                                              else "no TPPP table found" if k == "tppp" else "n/a"))
if lens == "Segment fit" and seg is None:
    st.info("The Segment fit lens ranks by clients in a segment: choose an industry or state under 'Who they deal "
            "with' and press Apply. Until then it ranks by money with PNC clients.")
if lens == "TPPP-like processors" and TP is None:
    st.warning("No TPPP prospect table found (looked for " + ", ".join(t.split(".")[-1] for t in pc.TPPP_TABLES)
               + "). Run the TPPP notebook with WRITE_TABLES = True. Until then this lens ranks by PNC clients.")
if not any(weights.values()):
    weights = {"amt": 1}

mask = pc.apply_filters(U, f)
D = pc.score(U[mask], weights)

# ------------------------------------------------------------------ headline numbers
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Companies in the window", f"{n_window:,}")
k2.metric("In the segment, after filters" if seg is not None else "After filters", f"{len(D):,}")
k3.metric("New to bank", f"{int((D.relationship == 'New to bank').sum()):,}")
k4.metric("Known processors", f"{int(D.known_proc.sum()):,}")
k5.metric("Median PNC clients a month", f"{D.clients.median():,.0f}" if len(D) else "–")
st.markdown(chips([f"Window {win[0]} to {win[-1]}", f"Lens: {lens}", seg_note] + filter_chips(f, floor)),
            unsafe_allow_html=True)


def view(T: pd.DataFrame) -> pd.DataFrame:
    v = pd.DataFrame({"rank": np.arange(1, len(T) + 1), "company": T.ent_name.to_numpy(),
                      "score": T.score.round(1).to_numpy(),
                      "why": [pc.reasons(r, weights) for _, r in T.iterrows()],
                      "relationship to PNC": T.relationship.to_numpy(), "money flow": T.role.to_numpy(),
                      "bank today": T.fi_dom.to_numpy()})
    if "n_fi" in T:
        v["banks"] = T.n_fi.to_numpy()
    v["PNC clients / mo"] = T.clients.round(1).to_numpy()
    if "tm_in" in T:
        v["TM clients paying it / mo"] = T.tm_in.round(1).to_numpy()
        v["TM clients it pays / mo"] = T.tm_out.round(1).to_numpy()
    v["$K from PNC clients / mo"] = (T.amt_in / 1e3).round(1).to_numpy()
    v["$K to PNC clients / mo"] = (T.amt_out / 1e3).round(1).to_numpy()
    v["client growth %"] = (100 * T.growth).round(0).to_numpy()
    v["months active"] = T.months_active.to_numpy()
    v["name suggests"] = T.name_type.to_numpy()
    v["known processor"] = T.known_proc.to_numpy()
    if "seg_clients" in T:
        v["TM clients in segment"] = T.seg_clients.to_numpy()
        v["segment share %"] = (100 * T.seg_share).round(0).to_numpy()
        v["$K with segment / mo"] = (T.seg_amt / 1e3).round(1).to_numpy()
    if "tppp_score" in T:
        v["TPPP score"] = T.tppp_score.round(1).to_numpy()
        if "tppp_reasons" in T:
            v["TPPP why"] = T.tppp_reasons.to_numpy()
    v["ent_id"] = T.ent_id.to_numpy()
    return v


ROW_SELECT = _V >= (1, 35) and os.environ.get("PROSPECT_ROW_SELECT", "auto") != "off"


def render_profile(r: pd.Series) -> None:
    """One company: headline facts, months around the window, its TM clients and their industries."""
    eid = r.ent_id
    if st.session_state.get("_last_profile") != eid:
        st.session_state["_last_profile"] = eid
        pc.log_event(viewer(), "profile", r.ent_name)

    st.markdown(f"### {r.ent_name}")
    st.markdown(chips([f"Bank today: {r.fi_dom}", r.relationship, r.role,
                       f"Name suggests: {r.name_type}" if r.name_type else "",
                       "Known processor" if r.known_proc else "",
                       f"Score {r.score:.0f} ({lens})"]), unsafe_allow_html=True)
    m = st.columns(6)
    m[0].metric("PNC clients / mo", f"{r.clients:,.1f}")
    m[1].metric("TM clients / mo", f"{r.tm_all:,.1f}" if "tm_all" in r else "–")
    m[2].metric("$K in from PNC clients / mo", f"{r.amt_in / 1e3:,.0f}")
    m[3].metric("$K out to PNC clients / mo", f"{r.amt_out / 1e3:,.0f}")
    m[4].metric("Banks it uses", f"{r.n_fi:.0f}" if "n_fi" in r and pd.notna(r.n_fi) else "–")
    m[5].metric("Months active", f"{r.months_active:.0f} of {n_months}")
    st.caption("Why it ranks here: " + (pc.reasons(r, weights) or "–"))
    if "tppp_score" in r and pd.notna(r.get("tppp_score")):
        st.caption(f"TPPP model: score {r.tppp_score:.0f}"
                   + (f"; {r.tppp_reasons}" if pd.notna(r.get("tppp_reasons", np.nan)) else "")
                   + (f"; most similar example: {r.tppp_similar}" if pd.notna(r.get("tppp_similar", np.nan)) else ""))

    trend_months = tuple(pc.month_range(pc.FIRST_MONTH, win[-1])[-max(12, n_months):])
    try:
        with st.spinner("Loading this company's months and clients ..."):
            TR = load_trend(eid, trend_months)
            CL = load_clients(eid, win)
    except Exception as e:
        st.error(f"Could not load the profile: {e}")
        TR, CL = pd.DataFrame(), pd.DataFrame()
    g1, g2 = st.columns(2)
    if len(TR):
        t1 = TR.melt(id_vars="month", value_vars=["n_cust_in", "n_cust_out"], var_name="side", value_name="clients")
        t1["side"] = t1.side.map({"n_cust_in": "PNC clients paying it", "n_cust_out": "PNC clients it pays"})
        f1 = px.line(t1, x="month", y="clients", color="side", markers=True, template="plotly_white", height=300,
                     color_discrete_sequence=["#E35205", "#2F6DB5"])
        t2 = TR.assign(**{"$K in": TR.amt_in / 1e3, "$K out": TR.amt_out / 1e3}).melt(
            id_vars="month", value_vars=["$K in", "$K out"], var_name="side", value_name="$K")
        f2 = px.line(t2, x="month", y="$K", color="side", markers=True, template="plotly_white", height=300,
                     color_discrete_sequence=["#E35205", "#2F6DB5"])
        for fg, ttl in ((f1, "PNC clients a month"), (f2, "Money with PNC clients, $K a month")):
            fg.update_layout(font_family=FONT, title=ttl, legend_title_text="", xaxis_title="", yaxis_title="",
                             margin={"l": 10, "r": 10, "t": 40, "b": 10}, legend={"orientation": "h", "y": -0.2})
            fg.add_vrect(x0=win[0], x1=win[-1], fillcolor="#EEF2F6", opacity=0.6, line_width=0, layer="below")
        g1.plotly_chart(f1, **WIDE)
        g2.plotly_chart(f2, **WIDE)
        st.caption("Shaded: the window. Months outside it are shown for context only.")

    st.markdown("#### Warm introductions: its TM clients")
    if len(CL):
        CL["total"] = CL.paid_to_it.fillna(0) + CL.received_from_it.fillna(0)
        CL["sector"] = np.where(CL.naics_status.astype(str) == "valid",
                                CL.naics2.astype(str).map(pc.NAICS2_SECTOR).fillna("unknown"), "unknown")
        ct = pd.DataFrame({"client": CL.customer_name, "power id": CL.customer_pwr_id,
                           "industry": (CL.naics_cd.fillna("").astype(str) + " "
                                        + CL.naics_desc.fillna("").astype(str)).str.strip(),
                           "$K paid to it": (CL.paid_to_it / 1e3).round(1),
                           "$K received from it": (CL.received_from_it / 1e3).round(1),
                           "months": CL.months})
        h1, h2 = st.columns([1.6, 1])
        h1.dataframe(ct.head(25), hide_index=True, **WIDE, height=420)
        h1.caption(f"Top 25 of {len(CL):,} TM clients by money with this company, {win[0]} to {win[-1]} "
                   "(window totals).")
        sec = (CL.groupby("sector").agg(clients=("mdm_id", "nunique")).reset_index()
                 .sort_values("clients", ascending=True).tail(12))
        f3 = px.bar(sec, x="clients", y="sector", orientation="h", template="plotly_white", height=420,
                    color_discrete_sequence=["#2F6DB5"])
        f3.update_layout(font_family=FONT, title="Industries of its TM clients", xaxis_title="TM clients",
                         yaxis_title="", margin={"l": 10, "r": 10, "t": 40, "b": 10})
        h2.plotly_chart(f3, **WIDE)
    else:
        st.info("No TM clients found for this company in the window.")


tab_list, tab_chart, tab_about = st.tabs(["Ranked list", "Chart", "About the data"])

with tab_list:
    l1, l2, _ = st.columns([1, 1.4, 4])
    top_n = l1.selectbox("Show top", [100, 250, 500, 1000], index=1)
    T = D.head(top_n)
    V = view(T)
    num = st.column_config.NumberColumn
    cfg = {"score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%.0f"),
           "banks": num(format="%.0f"), "PNC clients / mo": num(format="%.1f"),
           "$K from PNC clients / mo": num(format="%.0f"), "$K to PNC clients / mo": num(format="%.0f"),
           "client growth %": num(format="%.0f"), "why": st.column_config.TextColumn(width="large"),
           "company": st.column_config.TextColumn(width="medium")}
    review = V.assign(verdict="", note="", reviewer="", reviewed_at="")
    l2.download_button("Download this list (CSV)", review.to_csv(index=False).encode(),
                       file_name=f"prospects_{lens.split()[0].lower()}_{win[-1]}.csv", mime="text/csv",
                       on_click=pc.log_event, args=(viewer(), "export", f"{lens} | {win[0]}..{win[-1]} | {seg_note}",
                                                    len(review)))
    if ROW_SELECT:
        ev = st.dataframe(V, hide_index=True, **WIDE, height=520, column_config=cfg, on_select="rerun",
                          selection_mode="single-row")
        rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
        eid = V.ent_id.iloc[rows[0]] if rows and rows[0] < len(V) else None
        st.caption("Tick a row to open the company's profile below. Per-month figures are averages over the window, "
                   "counting months without activity as zero.")
    else:
        st.dataframe(V, hide_index=True, **WIDE, height=520, column_config=cfg)
        names = dict(zip(V.ent_id, V["rank"].astype(str) + "  " + V.company))
        eid = st.selectbox("Open a company's profile", [""] + list(V.ent_id),
                           format_func=lambda e: "Choose a company ..." if not e else f"#{names[e]}") or None
    if eid:
        st.divider()
        render_profile(T[T.ent_id == eid].iloc[0])

with tab_chart:
    P = D.head(5000)
    if len(P):
        fig = px.scatter(P.assign(**{"$K a month with PNC clients": P.amt / 1e3}),
                         x="clients", y="$K a month with PNC clients", color="relationship", log_x=True, log_y=True,
                         color_discrete_map=REL_COLORS, hover_name="ent_name",
                         hover_data={"fi_dom": True, "score": ":.0f", "role": True, "clients": ":.1f"},
                         labels={"clients": "PNC clients a month", "fi_dom": "bank today", "role": "money flow"},
                         template="plotly_white", height=560, opacity=0.75)
        fig.update_traces(marker={"size": 7, "line": {"width": 0}})
        fig.update_layout(font_family=FONT, legend_title_text="", margin={"l": 10, "r": 10, "t": 30, "b": 10},
                          legend={"orientation": "h", "y": 1.06})
        st.plotly_chart(fig, **WIDE)
        st.caption(f"Top {len(P):,} companies after filters, by score. Both axes on a log scale.")
    else:
        st.info("No companies match the filters.")

with tab_about:
    st.markdown(f"""
**What a prospect is.** An outside company (a counterparty entity, merged across its accounts and banks) that PNC
clients pay or are paid by, with eligible counterparty metrics in the window and a business-like name. The list starts
from `{pc.T_CPTY_MET.split('.')[-1]}`: companies averaging at least the chosen number of PNC clients a month.

**Measures.** Per-month figures are averages over the window, counting inactive months as zero. *Money flow* compares
money in from PNC clients with money out to them. *Growth* compares PNC clients in the second half of the window with
the first half. *Relationship to PNC* compares the company's cleaned name with PNC organisations; it misses clients
known under another name. *Name suggests* reads words in the name only.

**Who they deal with.** Industry and state describe the company's **TM clients** (their NAICS code and address), so
"sells to health care in PA" means: TM clients in health care with a PA address pay this company. It does not say
where the company is or what industry it is in.

**Score.** Each weighted measure becomes a percentile among the companies on screen; the score is the weighted mean,
0 to 100. Change the filters and the scores move, because the comparison set moves.

**Not in this draft.** The company's own industry (inferred), its own location, RM names, verdicts saved in the app,
"more like these" from examples.
""")
    st.markdown("##### Settings")
    st.dataframe(pd.DataFrame([
        {"item": "Counterparty metrics", "value": pc.T_CPTY_MET}, {"item": "Entity dimension", "value": pc.T_ENT_DIM},
        {"item": "Client pairs", "value": pc.T_PAIR}, {"item": "Client dimension", "value": pc.T_CDIM},
        {"item": "TPPP scores", "value": tppp_table or "none found"},
        {"item": "Client addresses", "value": f"{pc.ADDR['table']} ({pc.ADDR['state']}, snapshot "
                                              f"{pc.ADDR['snapshot'] or 'latest'})"},
        {"item": "Window", "value": f"{win[0]} to {win[-1]}"}, {"item": "Universe floor",
                                                                 "value": f"{floor} PNC clients a month"}]),
        hide_index=True, **WIDE)
    st.markdown("##### Impala queries in this session")
    if pc.QUERY_LOG:
        ql = pd.DataFrame(pc.QUERY_LOG)
        st.dataframe(ql[["at", "query", "seconds", "rows"]].iloc[::-1], hide_index=True, **WIDE)
        with st.expander("SQL of the last query"):
            st.code(ql.sql.iloc[-1], language="sql")
    else:
        st.caption("All results came from the cache.")
