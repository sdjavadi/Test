"""
app.py — Rail Shift Group Analyzer · Population Overview
PNC Treasury Management · Payment Knowledge Network

Run:
    streamlit run app.py

.streamlit/secrets.toml:
    [neo4j]
    uri      = "bolt://your-neo4j-host:7687"
    user     = "neo4j"
    password = "your-password"

    [pkn]
    include_pays_cpty = false   # flip to true once PAYS_CPTY is ingested

Layout (top → bottom):
  1.  Header (title + Neo4j status)
  2.  Cohort bar: NAICS | or | file upload | time-range slider | Analyze
  3.  Query hint strip
  4.  Metric cards (7)
  5.  Dominant rail cards (primary inbound + outbound)
  6.  Filter bar: Direction | Transacted with   ← pandas only, instant
  7.  Rail mix over time (Plotly stacked bar)
  8.  Distributions: net flow · degree · deviation score
  9.  NAICS neighbor graph (PyVis)
"""

import io
import json
import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from queries import (
    RAILS,
    get_driver,
    load_cohort_by_naics,
    load_cohort_by_ids,
    run_query,
    QUERY_COHORT_COUNT,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rail Shift Analyzer · PKN",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 100%; }
  [data-testid="collapsedControl"] { display: none; }
  div[data-testid="metric-container"] {
    background: #f8f8f6; border-radius: 8px; padding: 0.65rem 0.9rem 0.5rem;
  }
  div[data-testid="metric-container"] label { font-size: 11px !important; color: #888 !important; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 18px !important; font-family: monospace !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

RAIL_COLORS = {
    "ACH":   "#1D9E75",
    "Wire":  "#378ADD",
    "R2P":   "#EF9F27",
    "Check": "#888780",
}

ALL_MONTHS = [
    202401,202402,202403,202404,202405,202406,
    202407,202408,202409,202410,202411,202412,
    202501,202502,202503,202504,202505,202506,
    202507,202508,202509,202510,202511,202512,
]

MONTH_LABELS = {
    202401:"Jan 2024",202402:"Feb 2024",202403:"Mar 2024",202404:"Apr 2024",
    202405:"May 2024",202406:"Jun 2024",202407:"Jul 2024",202408:"Aug 2024",
    202409:"Sep 2024",202410:"Oct 2024",202411:"Nov 2024",202412:"Dec 2024",
    202501:"Jan 2025",202502:"Feb 2025",202503:"Mar 2025",202504:"Apr 2025",
    202505:"May 2025",202506:"Jun 2025",202507:"Jul 2025",202508:"Aug 2025",
    202509:"Sep 2025",202510:"Oct 2025",202511:"Nov 2025",202512:"Dec 2025",
}

MONTH_OPTS         = [MONTH_LABELS[m] for m in ALL_MONTHS]
MONTH_LABEL_TO_INT = {v: k for k, v in MONTH_LABELS.items()}

DIR_OPTIONS  = {"Both": "both", "Incoming": "in", "Outgoing": "out"}
TYPE_OPTIONS = {
    "Customers & Counterparties": "both",
    "Customers only":             "customer",
    "Counterparties only":        "counterparty",
}

# ── NAICS loader ──────────────────────────────────────────────────────────────

@st.cache_data
def load_naics(path: str = "naics.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"naics_code": str})
    df["label"] = df["naics_code"] + " · " + df["naics_label"]
    return df

# ── Neo4j ─────────────────────────────────────────────────────────────────────

try:
    _cfg     = st.secrets["neo4j"]
    driver   = get_driver(_cfg["uri"], _cfg["user"], _cfg["password"])
    neo4j_ok = True
except Exception as _e:
    neo4j_ok  = False
    neo4j_err = str(_e)

pkn_cfg      = st.secrets.get("pkn", {})
include_cpty = bool(pkn_cfg.get("include_pays_cpty", False))

try:
    naics_df = load_naics("naics.csv")
except FileNotFoundError:
    st.error("naics.csv not found.")
    st.stop()

# ── Data helpers ──────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, direction: str, node_type: str) -> pd.DataFrame:
    out = df.copy()
    if direction != "both":
        out = out[out["direction"] == direction]
    if node_type != "both":
        out = out[out["node_type"] == node_type]
    return out


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {k: 0 for k in
                ["n_customers","total_inflow","total_outflow","net_flow",
                 "avg_in_deg","avg_out_deg","avg_deg"]}
    customers = df["mdmId"].unique()
    n         = len(customers)
    inflow    = df[df["direction"] == "in"]["amount"].sum()
    outflow   = df[df["direction"] == "out"]["amount"].sum()
    in_deg    = (df[df["direction"] == "in"]
                 .groupby("mdmId")["counterpart_id"].nunique()
                 .reindex(customers, fill_value=0))
    out_deg   = (df[df["direction"] == "out"]
                 .groupby("mdmId")["counterpart_id"].nunique()
                 .reindex(customers, fill_value=0))
    return {
        "n_customers":   n,
        "total_inflow":  inflow,
        "total_outflow": outflow,
        "net_flow":      inflow - outflow,
        "avg_in_deg":    float(in_deg.mean())             if n else 0.0,
        "avg_out_deg":   float(out_deg.mean())            if n else 0.0,
        "avg_deg":       float((in_deg + out_deg).mean()) if n else 0.0,
    }


def compute_dominant_rails(df: pd.DataFrame) -> dict:
    result = {}
    for direction, key in [("in", "inbound"), ("out", "outbound")]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            result[key] = {r: 0.0 for r in RAILS}
            continue
        rail_totals = sub.groupby("rail")["amount"].sum()
        total       = rail_totals.sum()
        shares      = {r: round(rail_totals.get(r, 0.0) / total * 100, 1)
                       for r in RAILS} if total > 0 else {r: 0.0 for r in RAILS}
        result[key] = dict(sorted(shares.items(), key=lambda x: x[1], reverse=True))
    return result


def compute_deviation_scores(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    pivot = (df.groupby(["mdmId", "rail"])["amount"]
               .sum().unstack(fill_value=0.0))
    for r in RAILS:
        if r not in pivot.columns:
            pivot[r] = 0.0
    pivot  = pivot[RAILS]
    totals = pivot.sum(axis=1).replace(0, np.nan)
    mix    = pivot.div(totals, axis=0).fillna(0.0)
    z      = (mix - mix.mean()) / mix.std().replace(0, np.nan)
    return z.fillna(0.0).pow(2).mean(axis=1).apply(math.sqrt)


def fmt_amount(v: float) -> str:
    a, s = abs(v), ("-" if v < 0 else "")
    if a >= 1e9: return f"{s}${a/1e9:.2f}B"
    if a >= 1e6: return f"{s}${a/1e6:.1f}M"
    return f"{s}${a:,.0f}"

# ── NAICS graph helpers ───────────────────────────────────────────────────────

def compute_naics_graph_data(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Aggregate the loaded DataFrame into a graph-ready summary:
        counterpart_naics, rail, direction, total_amount

    Only customer-type rows are used (counterparty rows have no NAICS).
    Returns the top_n NAICS codes by total flow volume.
    """
    cust = df[(df["node_type"] == "customer") & df["counterpart_naics"].notna()].copy()
    if cust.empty:
        return pd.DataFrame(
            columns=["counterpart_naics", "rail", "direction", "total_amount"]
        )

    grp = (cust.groupby(["counterpart_naics", "rail", "direction"])["amount"]
               .sum()
               .reset_index()
               .rename(columns={"amount": "total_amount"}))

    # Keep only the top-N NAICS by combined volume
    top_naics = (
        grp.groupby("counterpart_naics")["total_amount"]
           .sum()
           .nlargest(top_n)
           .index.tolist()
    )
    return grp[grp["counterpart_naics"].isin(top_naics)].copy()


def build_pyvis_graph(
    graph_df: pd.DataFrame,
    cohort_label: str,
    super_node_id: str = "COHORT",
) -> str:
    """
    Build a PyVis directed network HTML string.

    Nodes:
      - One super-node representing the selected cohort
      - One node per NAICS code in graph_df

    Edges:
      - One edge per (NAICS, rail, direction) combination
      - Only drawn when total_amount > 0
      - Color  = rail color
      - Width  = log-scaled by amount (so small and large flows are both visible)
      - Label  = formatted amount
      - Arrows show payment direction relative to the super-node

    Multiple edges between the same pair are rendered as smooth curves (vis.js
    "dynamic" smooth mode fans them out automatically).
    """
    if graph_df.empty:
        return "<p>No customer-to-customer data available for graph.</p>"

    net = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#444441",
    )

    # ── Super-node ────────────────────────────────────────────────────────────
    short_label = cohort_label.split("·")[0].strip() if "·" in cohort_label else cohort_label
    net.add_node(
        super_node_id,
        label=short_label,
        title=cohort_label,
        size=36,
        color={"background": "#2C2C2A", "border": "#2C2C2A",
               "highlight": {"background": "#444441", "border": "#888780"}},
        font={"color": "#ffffff", "size": 13, "bold": True},
        shape="dot",
        borderWidth=0,
    )

    # ── NAICS nodes ───────────────────────────────────────────────────────────
    # Look up labels from naics.csv if available; fall back to the code itself.
    naics_label_map: dict = {}
    try:
        nl = load_naics("naics.csv")
        naics_label_map = dict(zip(nl["naics_code"], nl["naics_label"]))
    except Exception:
        pass

    for naics in graph_df["counterpart_naics"].unique():
        display = str(naics)
        sub     = naics_label_map.get(str(naics), "")
        tooltip = f"{naics}" + (f" · {sub}" if sub else "")
        net.add_node(
            str(naics),
            label=display,
            title=tooltip,
            size=20,
            color={"background": "#F1EFE8", "border": "#B4B2A9",
                   "highlight": {"background": "#E1F5EE", "border": "#1D9E75"}},
            font={"color": "#444441", "size": 11},
            shape="dot",
            borderWidth=1,
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    def edge_width(amount: float) -> float:
        """Log-scaled width. $1M → ~1.5px, $100M → ~5px, $1B → ~8px."""
        return max(1.0, math.log10(max(amount, 1) / 1e6 + 1) * 3.5 + 1.0)

    for _, row in graph_df.iterrows():
        naics_node = str(row["counterpart_naics"])
        color      = RAIL_COLORS.get(row["rail"], "#888")
        width      = edge_width(row["total_amount"])
        label      = fmt_amount(row["total_amount"])
        title      = f"{row['rail']} · {label}"

        if row["direction"] == "out":
            src, dst = super_node_id, naics_node
        else:
            src, dst = naics_node, super_node_id

        net.add_edge(
            src, dst,
            color={"color": color, "highlight": color, "opacity": 0.82},
            width=width,
            title=title,
            label=label,
            font={"size": 8, "color": "#888780", "align": "middle", "strokeWidth": 0},
            arrows={"to": {"enabled": True, "scaleFactor": 0.55}},
            smooth={"type": "dynamic"},  # fans out parallel edges automatically
        )

    # ── Physics / layout ──────────────────────────────────────────────────────
    # forceAtlas2Based keeps the super-node roughly central and NAICS nodes spread
    # around it. After 300 stabilisation iterations the layout freezes.
    net.set_options(json.dumps({
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.015,
                "springLength": 220,
                "springConstant": 0.06,
                "damping": 0.45,
                "avoidOverlap": 0.6,
            },
            "stabilization": {
                "enabled": True,
                "iterations": 300,
                "updateInterval": 25,
            },
        },
        "edges": {
            "smooth": {"type": "dynamic"},
            "font": {"size": 8, "align": "middle"},
            "scaling": {"min": 1, "max": 12},
        },
        "nodes": {
            "scaling": {"min": 14, "max": 40},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 150,
            "hideEdgesOnDrag": True,
        },
    }))

    return net.generate_html(notebook=False)

# ── Chart builders ────────────────────────────────────────────────────────────

def build_rail_mix_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    grp    = df.groupby(["time_key", "rail"])["amount"].sum().reset_index()
    totals = grp.groupby("time_key")["amount"].sum().rename("total")
    grp    = grp.join(totals, on="time_key")
    grp["pct"] = (grp["amount"] / grp["total"] * 100).round(1)
    months     = sorted(grp["time_key"].unique())
    labels     = [MONTH_LABELS.get(m, str(m)) for m in months]
    fig = go.Figure()
    for rail in RAILS:
        sub = grp[grp["rail"] == rail].set_index("time_key")
        y   = [float(sub.loc[m, "pct"]) if m in sub.index else 0.0 for m in months]
        fig.add_trace(go.Bar(
            name=rail, x=labels, y=y,
            marker_color=RAIL_COLORS[rail],
            hovertemplate=f"<b>{rail}</b><br>%{{x}}<br>%{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        yaxis=dict(title="% share", ticksuffix="%", range=[0, 100],
                   gridcolor="rgba(128,128,120,0.12)"),
        xaxis=dict(title=None, showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0, font_size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        height=260,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_dist_chart(series: pd.Series, color: str,
                     log_y: bool = False, x_label: str = "") -> go.Figure:
    fig = go.Figure()
    if series.dropna().empty:
        return fig
    fig.add_trace(go.Histogram(
        x=series.dropna(), nbinsx=20,
        marker_color=color, opacity=0.85,
        hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(type="log" if log_y else "linear",
                   gridcolor="rgba(128,128,120,0.12)", title="Count"),
        xaxis=dict(title=x_label, showgrid=False),
        margin=dict(l=0, r=0, t=6, b=0),
        height=185, bargap=0.05,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def render_dominant_rail_card(label, rail_shares, bg_color,
                               border_color, text_color, bar_color) -> None:
    sorted_rails = list(rail_shares.items())
    if not sorted_rails:
        return
    top_rail, top_pct = sorted_rails[0]
    others            = sorted_rails[1:]
    other_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:11px;'
        f'color:{text_color};opacity:.75;margin-top:3px">'
        f'<span style="min-width:36px">{r}</span>'
        f'<div style="flex:1;height:3px;background:rgba(0,0,0,0.1);border-radius:2px">'
        f'<div style="width:{int(p)}%;height:100%;background:{bar_color};'
        f'opacity:.45;border-radius:2px"></div></div>'
        f'<span style="min-width:30px;text-align:right;font-family:monospace">{p}%</span>'
        f'</div>'
        for r, p in others
    )
    st.markdown(
        f'<div style="background:{bg_color};border:.5px solid {border_color};'
        f'border-radius:10px;padding:12px 16px">'
        f'<div style="font-size:10px;font-weight:500;color:{text_color};'
        f'text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">{label}</div>'
        f'<div style="display:flex;align-items:baseline;gap:8px">'
        f'<span style="font-size:22px;font-weight:500;color:{text_color}">{top_rail}</span>'
        f'<span style="font-size:14px;font-family:monospace;color:{bar_color}">{top_pct}%</span>'
        f'</div>'
        f'<div style="width:100%;height:4px;background:rgba(0,0,0,0.08);'
        f'border-radius:2px;margin:6px 0 10px">'
        f'<div style="width:{int(top_pct)}%;height:100%;background:{bar_color};'
        f'border-radius:2px"></div></div>'
        f'{other_rows}</div>',
        unsafe_allow_html=True,
    )

# ── ── ── ── ── ── ── ── ── ── UI ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

# Header
h_left, h_right = st.columns([4, 1])
with h_left:
    st.markdown(f"**{st.session_state.get('page_title', 'Rail shift · population overview')}**")
    st.caption(st.session_state.get("page_cap", "Payment Knowledge Network · PNC Treasury Management"))
with h_right:
    status = "● Neo4j connected" if neo4j_ok else "✕ Neo4j error"
    color  = "#0F6E56"           if neo4j_ok else "#A32D2D"
    st.markdown(
        f'<div style="text-align:right;font-size:11px;color:{color};padding-top:4px">{status}</div>',
        unsafe_allow_html=True,
    )

# ── Cohort + time range bar ───────────────────────────────────────────────────

c_naics, c_sep, c_upload, c_time, c_btn = st.columns([2.2, 0.15, 1.4, 2.2, 0.6])

with c_naics:
    naics_opts     = ["— Select NAICS —"] + naics_df["label"].tolist()
    selected_label = st.selectbox("NAICS cohort", naics_opts, label_visibility="collapsed")

with c_sep:
    st.markdown('<div style="text-align:center;color:#aaa;padding-top:8px;font-size:12px">or</div>',
                unsafe_allow_html=True)

with c_upload:
    uploaded_file = st.file_uploader("Upload MDM ID list", type=["csv","txt"],
                                     label_visibility="collapsed")

with c_time:
    time_range = st.select_slider(
        "Time range", options=MONTH_OPTS,
        value=(MONTH_OPTS[0], MONTH_OPTS[-1]),
        label_visibility="collapsed",
        help="Sets time_key filter in Cypher — click Analyze to apply",
    )
    st.caption(f"Query window: {time_range[0]} – {time_range[1]}  *(apply on Analyze)*")

with c_btn:
    analyze = st.button("Analyze ↗", type="primary", use_container_width=True)

# ── Run query on Analyze ──────────────────────────────────────────────────────

if analyze:
    if not neo4j_ok:
        st.error(f"Neo4j not connected: {neo4j_err}")
        st.stop()

    time_start = MONTH_LABEL_TO_INT[time_range[0]]
    time_end   = MONTH_LABEL_TO_INT[time_range[1]]

    use_file  = uploaded_file is not None
    use_naics = selected_label != "— Select NAICS —"

    if not use_file and not use_naics:
        st.warning("Select a NAICS code or upload an MDM ID list.")
        st.stop()

    src_label = uploaded_file.name if use_file else selected_label

    with st.spinner(f"Querying Neo4j — {src_label} · {time_range[0]} – {time_range[1]}…"):
        t0 = time.perf_counter()
        try:
            if use_file:
                content = uploaded_file.read().decode("utf-8")
                mdm_ids = [ln.strip() for ln in content.splitlines() if ln.strip()]
                raw = load_cohort_by_ids(driver, mdm_ids, time_start, time_end, include_cpty)
            else:
                naics_code = naics_df.loc[naics_df["label"] == selected_label, "naics_code"].iloc[0]
                count_df   = run_query(driver, QUERY_COHORT_COUNT, {"naics": naics_code})
                cohort_n   = int(count_df["n"].iloc[0]) if not count_df.empty else 0
                if cohort_n == 0:
                    st.warning(f"No customers found for NAICS {naics_code}.")
                    st.stop()
                raw = load_cohort_by_naics(driver, naics_code, time_start, time_end, include_cpty)
        except Exception as e:
            st.error(f"Query failed: {e}")
            st.stop()
        elapsed = time.perf_counter() - t0

    st.session_state.update({
        "raw":        raw,
        "src_label":  src_label,
        "time_start": time_start,
        "time_end":   time_end,
        "n_edges":    len(raw),
        "elapsed":    elapsed,
        "page_title": src_label + " · population overview",
        "page_cap": (
            f"Snapshot: {time_range[0]} – {time_range[1]}"
            f" · {raw['mdmId'].nunique():,} customers with transactions"
        ),
    })
    st.rerun()

# ── Guard ─────────────────────────────────────────────────────────────────────

if "raw" not in st.session_state:
    st.info("Select a cohort, set the time range, then click **Analyze ↗**.")
    st.stop()

raw = st.session_state["raw"]

# ── Query hint strip ──────────────────────────────────────────────────────────

ts_lbl = MONTH_LABELS.get(st.session_state["time_start"], "")
te_lbl = MONTH_LABELS.get(st.session_state["time_end"],   "")
st.markdown(
    f'<div style="font-size:10px;color:var(--text-color);font-family:monospace;'
    f'display:flex;align-items:center;gap:8px;padding:6px 12px;'
    f'background:#f8f8f6;border-radius:6px;border:.5px solid #e0e0d8;margin-bottom:.75rem">'
    f'Last query: '
    f'<span style="background:#e6f1fb;color:#0c447c;padding:1px 8px;border-radius:3px">'
    f'{st.session_state["src_label"]} · {ts_lbl} – {te_lbl}</span>'
    f'· {st.session_state["n_edges"]:,} edges · {st.session_state["elapsed"]:.1f}s'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Filter bar (pandas — instant) ─────────────────────────────────────────────

f_dir, f_type, _ = st.columns([1, 1.7, 3])
with f_dir:
    direction_label = st.selectbox("Direction",       list(DIR_OPTIONS.keys()))
with f_type:
    type_label      = st.selectbox("Transacted with", list(TYPE_OPTIONS.keys()))

direction = DIR_OPTIONS[direction_label]
node_type = TYPE_OPTIONS[type_label]
df        = apply_filters(raw, direction, node_type)

st.divider()

# ── Metric cards ──────────────────────────────────────────────────────────────

m   = compute_metrics(df)
mc  = st.columns(7)
net_val = m["net_flow"]

mc[0].metric("Customers",     f"{m['n_customers']:,}")
mc[1].metric("Total inflow",  fmt_amount(m["total_inflow"]))
mc[2].metric("Total outflow", fmt_amount(m["total_outflow"]))
mc[3].metric("Net flow",      (("+" if net_val >= 0 else "") + fmt_amount(net_val)))
mc[4].metric("Avg in-degree",  f"{m['avg_in_deg']:.1f}")
mc[5].metric("Avg out-degree", f"{m['avg_out_deg']:.1f}")
mc[6].metric("Avg degree",     f"{m['avg_deg']:.1f}")

# ── Dominant rail cards ───────────────────────────────────────────────────────

dom = compute_dominant_rails(raw)   # always from full raw frame
dr_in, dr_out = st.columns(2)

with dr_in:
    render_dominant_rail_card(
        "Primary inbound rail",  dom["inbound"],
        "#E1F5EE","#5DCAA5","#085041","#1D9E75",
    )
with dr_out:
    render_dominant_rail_card(
        "Primary outbound rail", dom["outbound"],
        "#E6F1FB","#85B7EB","#0C447C","#378ADD",
    )

st.divider()

if df.empty:
    st.warning("No data for this filter combination.")
    st.stop()

# ── Rail mix over time ────────────────────────────────────────────────────────

st.markdown("##### Rail mix over time")
st.plotly_chart(build_rail_mix_chart(df), use_container_width=True)

st.divider()

# ── Distributions ─────────────────────────────────────────────────────────────

st.markdown("##### Distributions")

net_per_cust = (
    df[df["direction"] == "in"].groupby("mdmId")["amount"].sum()
    .sub(df[df["direction"] == "out"].groupby("mdmId")["amount"].sum(), fill_value=0)
)
deg_per_cust = df.groupby("mdmId")["counterpart_id"].nunique()
dev_scores   = compute_deviation_scores(df)

d1, d2, d3 = st.columns(3)
with d1:
    log_f = st.checkbox("Log scale", key="log_flow")
    st.caption("Net flow per customer ($)")
    st.plotly_chart(build_dist_chart(net_per_cust,"#378ADD",log_f,"Net flow ($)"),
                    use_container_width=True)
with d2:
    log_d = st.checkbox("Log scale", key="log_deg")
    st.caption("Degree distribution")
    st.plotly_chart(build_dist_chart(deg_per_cust,"#1D9E75",log_d,"Unique counterparts"),
                    use_container_width=True)
with d3:
    log_v = st.checkbox("Log scale", key="log_dev")
    st.caption("Deviation score distribution")
    st.plotly_chart(build_dist_chart(dev_scores,"#EF9F27",log_v,"Composite deviation score"),
                    use_container_width=True)

st.divider()

# ── NAICS neighbor graph ──────────────────────────────────────────────────────

st.markdown("##### NAICS payment flow network")
st.caption(
    "Super-node = selected cohort. "
    "Neighbor nodes = top 10 NAICS codes by total flow volume. "
    "One edge per rail per direction — only drawn when amount > 0. "
    "Edge thickness ∝ log(amount). Aggregated over the selected time window."
)

graph_df = compute_naics_graph_data(raw, top_n=10)

if graph_df.empty:
    st.info(
        "No customer-to-customer NAICS data available. "
        "This graph requires PAYS edges with counterpart_naics populated."
    )
else:
    cohort_label = st.session_state.get("src_label", "Cohort")
    graph_html   = build_pyvis_graph(graph_df, cohort_label)

    # Inject rail-color legend above the graph
    legend_html = (
        '<div style="display:flex;gap:18px;flex-wrap:wrap;'
        'font-size:11px;color:#888780;margin-bottom:8px;align-items:center">'
    )
    for rail, color in RAIL_COLORS.items():
        legend_html += (
            f'<span style="display:flex;align-items:center;gap:5px">'
            f'<span style="display:inline-block;width:20px;height:2.5px;'
            f'background:{color};border-radius:2px"></span>{rail}</span>'
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    components.html(graph_html, height=540, scrolling=False)
