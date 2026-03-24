"""
Rail Shift Monitor — PNC Payment Knowledge Network
Streamlit module for Treasury Management analysts.

Grouping dimension: NAICS industry (2-, 4-, or 6-digit granularity).

Architecture:
  Neo4j (PAYS edges) → Cypher queries → pandas DataFrames
  → Streamlit UI (filters, charts, anomaly highlights)

Dependencies:
  pip install streamlit neo4j pandas plotly scipy numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from neo4j import GraphDatabase
from datetime import date
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Rail Shift Monitor · PNC TM",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME TOKENS
# ─────────────────────────────────────────────────────────────────────────────

RAIL_COLORS = {
    "ACH":   "#1A6FBF",
    "Wire":  "#0A8A5A",
    "R2P":   "#7B4FBF",
    "Check": "#C47C0A",
    "Debit": "#BF3A2A",
}

ANOMALY_RED   = "#E53935"
ANOMALY_AMBER = "#FB8C00"
ANOMALY_GREEN = "#43A047"
NEUTRAL_GRAY  = "#6B7280"

RAIL_COLS   = ["pctACH", "pctWire", "pctR2P", "pctCheck", "pctDebit"]
RAIL_LABELS = ["ACH", "Wire", "R2P", "Check", "Debit"]
AMOUNT_COLS = ["totalACH", "totalWire", "totalR2P", "totalCheck", "totalDebit"]

# NAICS granularity options: label → substring length
NAICS_GRANULARITY = {"2-digit sector": 2, "4-digit subsector": 4, "6-digit industry": 6}

# ─────────────────────────────────────────────────────────────────────────────
# CYPHER QUERY LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
# All queries use Neo4j driver parameter injection (no f-strings in Cypher)
# to prevent injection and enable server-side query plan caching.
#
# $naics_digits  : int  — controls grouping granularity (2, 4, or 6)
# $naics_prefix  : str  — optional prefix filter (e.g. "52" for Finance)
# $start_month   : str  — "YYYY-MM-01"
# $end_month     : str  — "YYYY-MM-01"
# ─────────────────────────────────────────────────────────────────────────────

CYPHER = {}

# ── Q1: NAICS industry channel mix MoM ─────────────────────────────────────
# Aggregates by N-digit NAICS code. $naics_digits controls the cut.
# Neo4j substring() is 0-indexed, end-exclusive:
#   substring(src.naicsCode, 0, 2)  → 2-digit sector
#   substring(src.naicsCode, 0, 4)  → 4-digit subsector
#   substring(src.naicsCode, 0, 6)  → full 6-digit code
# The CASE block avoids APOC dependency for portability.
CYPHER["naics_mix_mom"] = """
MATCH (src:PNCCustomer)-[p:PAYS]->(dst:PNCCustomer)
WHERE p.time_key >= $start_month
  AND p.time_key <= $end_month
  AND src.naicsCode IS NOT NULL
  AND ($naics_prefix IS NULL OR src.naicsCode STARTS WITH $naics_prefix)
WITH p.time_key AS time_key,
     CASE $naics_digits
       WHEN 2 THEN substring(src.naicsCode, 0, 2)
       WHEN 4 THEN substring(src.naicsCode, 0, 4)
       ELSE        substring(src.naicsCode, 0, 6)
     END AS naicsCode,
     sum(p.totalAmount)      AS totalAmount,
     sum(p.totalAmountACH)   AS totalACH,
     sum(p.totalAmountWire)  AS totalWire,
     sum(p.totalAmountR2P)   AS totalR2P,
     sum(p.totalAmountCheck) AS totalCheck,
     sum(p.totalAmountDebit) AS totalDebit,
     sum(p.totalCount)       AS totalCount
RETURN
  time_key,
  naicsCode,
  totalAmount,
  totalACH,
  totalWire,
  totalR2P,
  totalCheck,
  totalDebit,
  totalCount,
  CASE WHEN totalAmount > 0 THEN round(totalACH   / totalAmount * 10000) / 100 ELSE 0 END AS pctACH,
  CASE WHEN totalAmount > 0 THEN round(totalWire  / totalAmount * 10000) / 100 ELSE 0 END AS pctWire,
  CASE WHEN totalAmount > 0 THEN round(totalR2P   / totalAmount * 10000) / 100 ELSE 0 END AS pctR2P,
  CASE WHEN totalAmount > 0 THEN round(totalCheck / totalAmount * 10000) / 100 ELSE 0 END AS pctCheck,
  CASE WHEN totalAmount > 0 THEN round(totalDebit / totalAmount * 10000) / 100 ELSE 0 END AS pctDebit
ORDER BY naicsCode, time_key
"""

# ── Q2: Counterparty-pair channel mix MoM ──────────────────────────────────
# Returns per-edge-pair data for a specific src/dst pair (both IDs optional).
# Also accepts $naics_prefix so analysts can drill from industry → pair.
CYPHER["pair_mix_mom"] = """
MATCH (src:PNCCustomer)-[p:PAYS]->(dst:PNCCustomer)
WHERE p.time_key >= $start_month
  AND p.time_key <= $end_month
  AND ($src_id IS NULL OR src.customerId = $src_id)
  AND ($dst_id IS NULL OR dst.customerId = $dst_id)
  AND ($naics_prefix IS NULL OR src.naicsCode STARTS WITH $naics_prefix)
WITH p.time_key              AS time_key,
     src.customerId          AS srcId,
     src.customerName        AS srcName,
     src.naicsCode           AS srcNaics,
     dst.customerId          AS dstId,
     dst.customerName        AS dstName,
     p.totalAmount           AS totalAmount,
     p.totalAmountACH        AS totalACH,
     p.totalAmountWire       AS totalWire,
     p.totalAmountR2P        AS totalR2P,
     p.totalAmountCheck      AS totalCheck,
     p.totalAmountDebit      AS totalDebit,
     p.totalCount            AS totalCount
RETURN
  time_key,
  srcId,
  srcName,
  srcNaics,
  dstId,
  dstName,
  totalAmount,
  totalACH,
  totalWire,
  totalR2P,
  totalCheck,
  totalDebit,
  totalCount,
  CASE WHEN totalAmount > 0 THEN round(totalACH   / totalAmount * 10000) / 100 ELSE 0 END AS pctACH,
  CASE WHEN totalAmount > 0 THEN round(totalWire  / totalAmount * 10000) / 100 ELSE 0 END AS pctWire,
  CASE WHEN totalAmount > 0 THEN round(totalR2P   / totalAmount * 10000) / 100 ELSE 0 END AS pctR2P,
  CASE WHEN totalAmount > 0 THEN round(totalCheck / totalAmount * 10000) / 100 ELSE 0 END AS pctCheck,
  CASE WHEN totalAmount > 0 THEN round(totalDebit / totalAmount * 10000) / 100 ELSE 0 END AS pctDebit
ORDER BY time_key
"""

# ── Q3: Top-N anomalous pairs by peak rail shift ────────────────────────────
# Computes max MoM absolute %-point shift for every src→dst pair in-graph.
# Returns pairs exceeding $threshold. Scope with $naics_prefix.
# Heavy query — always pass a tight date range and a NAICS prefix in production.
CYPHER["anomaly_pairs"] = """
MATCH (src:PNCCustomer)-[p:PAYS]->(dst:PNCCustomer)
WHERE p.time_key >= $start_month
  AND p.time_key <= $end_month
  AND src.naicsCode IS NOT NULL
  AND ($naics_prefix IS NULL OR src.naicsCode STARTS WITH $naics_prefix)
WITH src.customerId AS srcId,
     src.customerName AS srcName,
     src.naicsCode AS srcNaics,
     dst.customerId AS dstId,
     dst.customerName AS dstName,
     p.time_key AS time_key,
     p.totalAmount AS totalAmount,
     CASE WHEN p.totalAmount > 0 THEN p.totalAmountACH   / p.totalAmount ELSE 0 END AS pctACH,
     CASE WHEN p.totalAmount > 0 THEN p.totalAmountWire  / p.totalAmount ELSE 0 END AS pctWire,
     CASE WHEN p.totalAmount > 0 THEN p.totalAmountR2P   / p.totalAmount ELSE 0 END AS pctR2P,
     CASE WHEN p.totalAmount > 0 THEN p.totalAmountCheck / p.totalAmount ELSE 0 END AS pctCheck,
     CASE WHEN p.totalAmount > 0 THEN p.totalAmountDebit / p.totalAmount ELSE 0 END AS pctDebit
ORDER BY srcId, dstId, time_key
WITH srcId, srcName, srcNaics, dstId, dstName,
     collect(time_key)    AS months,
     collect(totalAmount) AS amounts,
     collect(pctACH)      AS achSeries,
     collect(pctWire)     AS wireSeries,
     collect(pctR2P)      AS r2pSeries,
     collect(pctCheck)    AS checkSeries,
     collect(pctDebit)    AS debitSeries
WHERE size(months) >= 2
WITH srcId, srcName, srcNaics, dstId, dstName, months, amounts,
     reduce(maxShift = 0.0, i IN range(1, size(achSeries)-1) |
       CASE WHEN abs(achSeries[i]   - achSeries[i-1])   > maxShift
            THEN abs(achSeries[i]   - achSeries[i-1])   ELSE maxShift END
     ) AS maxACHShift,
     reduce(maxShift = 0.0, i IN range(1, size(wireSeries)-1) |
       CASE WHEN abs(wireSeries[i]  - wireSeries[i-1])  > maxShift
            THEN abs(wireSeries[i]  - wireSeries[i-1])  ELSE maxShift END
     ) AS maxWireShift,
     reduce(maxShift = 0.0, i IN range(1, size(r2pSeries)-1) |
       CASE WHEN abs(r2pSeries[i]   - r2pSeries[i-1])   > maxShift
            THEN abs(r2pSeries[i]   - r2pSeries[i-1])   ELSE maxShift END
     ) AS maxR2PShift,
     reduce(maxShift = 0.0, i IN range(1, size(checkSeries)-1) |
       CASE WHEN abs(checkSeries[i] - checkSeries[i-1]) > maxShift
            THEN abs(checkSeries[i] - checkSeries[i-1]) ELSE maxShift END
     ) AS maxCheckShift,
     reduce(maxShift = 0.0, i IN range(1, size(debitSeries)-1) |
       CASE WHEN abs(debitSeries[i] - debitSeries[i-1]) > maxShift
            THEN abs(debitSeries[i] - debitSeries[i-1]) ELSE maxShift END
     ) AS maxDebitShift,
     last(amounts) AS latestVolume
WITH *,
     reduce(m = 0.0, x IN [maxACHShift, maxWireShift, maxR2PShift, maxCheckShift, maxDebitShift] |
       CASE WHEN x > m THEN x ELSE m END) AS peakShift
WHERE peakShift >= $threshold
RETURN
  srcId, srcName, srcNaics, dstId, dstName,
  round(maxACHShift   * 10000) / 100 AS maxACHShift_pct,
  round(maxWireShift  * 10000) / 100 AS maxWireShift_pct,
  round(maxR2PShift   * 10000) / 100 AS maxR2PShift_pct,
  round(maxCheckShift * 10000) / 100 AS maxCheckShift_pct,
  round(maxDebitShift * 10000) / 100 AS maxDebitShift_pct,
  round(peakShift     * 10000) / 100 AS peakShift_pct,
  latestVolume
ORDER BY peakShift DESC
LIMIT $top_n
"""

# ── Q4: Distinct NAICS codes present in the graph ──────────────────────────
# Used to populate the prefix filter dropdown.
CYPHER["list_naics"] = """
MATCH (src:PNCCustomer)
WHERE src.naicsCode IS NOT NULL
RETURN DISTINCT
  substring(src.naicsCode, 0, 2) AS naics2,
  substring(src.naicsCode, 0, 4) AS naics4,
  src.naicsCode                  AS naics6,
  src.naicsDescription           AS naicsDesc
ORDER BY naics2, naics4, naics6
LIMIT 200
"""

# ─────────────────────────────────────────────────────────────────────────────
# NEO4J CONNECTION (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to graph…")
def get_driver(uri: str, user: str, password: str):
    return GraphDatabase.driver(uri, auth=(user, password))


@st.cache_data(ttl=300, show_spinner="Running Cypher…")
def run_query(_driver, query: str, **params) -> pd.DataFrame:
    """Execute a Cypher query and return a DataFrame. Cached 5 minutes."""
    with _driver.session() as session:
        result = session.run(query, **params)
        return pd.DataFrame([r.data() for r in result])


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mom_shifts(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Add shift_{rail} columns — %-point MoM change per rail per group."""
    df = df.sort_values([group_col, "time_key"]).copy()
    for col in RAIL_COLS:
        df[f"shift_{col}"] = df.groupby(group_col)[col].diff()
    return df


def zscore_anomalies(df: pd.DataFrame, group_col: str, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag rows where any rail's MoM shift exceeds z_threshold σ from the
    group's own historical mean. Adds: z_shift_* columns, max_abs_z, is_anomaly.
    """
    shift_cols = [f"shift_{c}" for c in RAIL_COLS]
    records = []
    for _group, gdf in df.groupby(group_col):
        gdf = gdf.copy()
        for col in shift_cols:
            series = gdf[col].dropna()
            if len(series) < 3:
                continue
            mu, sigma = series.mean(), series.std()
            if sigma == 0:
                continue
            gdf[f"z_{col}"] = (gdf[col] - mu) / sigma
        records.append(gdf)
    result = pd.concat(records, ignore_index=True) if records else df.copy()
    z_cols = [c for c in result.columns if c.startswith("z_shift_")]
    if z_cols:
        result["max_abs_z"] = result[z_cols].abs().max(axis=1)
        result["anomaly_rail"] = result[z_cols].abs().idxmax(axis=1).str.replace("z_shift_pct", "")
        result["is_anomaly"] = result["max_abs_z"] >= z_threshold
    return result


def format_amount(val: float) -> str:
    if abs(val) >= 1e9:
        return f"${val/1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.1f}M"
    if abs(val) >= 1e3:
        return f"${val/1e3:.1f}K"
    return f"${val:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

DEMO_NAICS = {
    "11": "Agriculture",
    "22": "Utilities",
    "31": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "62": "Health Care",
    "72": "Accommodation & Food",
}


def generate_demo_data(naics_digits: int = 2, naics_prefix: str = None,
                       n_months: int = 12) -> pd.DataFrame:
    """
    Synthetic DataFrame mirroring naics_mix_mom output.
    Injects deliberate rail shifts at months 6 and 10 to exercise anomaly logic.
    """
    rng = np.random.default_rng(42)

    if naics_digits == 2:
        codes = list(DEMO_NAICS.keys())
    elif naics_digits == 4:
        codes = [f"{k}{rng.integers(10, 99)}" for k in DEMO_NAICS]
    else:
        codes = [f"{k}{rng.integers(1000, 9999)}" for k in DEMO_NAICS]

    if naics_prefix:
        codes = [c for c in codes if c.startswith(naics_prefix)] or codes[:3]

    months = pd.date_range("2024-01-01", periods=n_months, freq="MS").strftime("%Y-%m-%d").tolist()
    rows = []
    for code in codes:
        base = rng.dirichlet(np.ones(5) * 3) * 100
        for m_idx, month in enumerate(months):
            mix = base.copy()
            # Inject anomaly: Wire spike in Finance (52) at month 6
            if code.startswith("52") and m_idx == 6:
                mix[1] += 25
                mix[0] -= 25
            # Inject anomaly: R2P surge in Professional Services (54) at month 10
            if code.startswith("54") and m_idx == 10:
                mix[2] += 20
                mix[4] -= 20
            mix = np.clip(mix + rng.normal(0, 1.5, 5), 0, None)
            mix = mix / mix.sum() * 100
            total = rng.uniform(5e6, 5e8)
            rows.append({
                "time_key":    month,
                "naicsCode":   code,
                "totalAmount": total,
                "totalACH":    total * mix[0] / 100,
                "totalWire":   total * mix[1] / 100,
                "totalR2P":    total * mix[2] / 100,
                "totalCheck":  total * mix[3] / 100,
                "totalDebit":  total * mix[4] / 100,
                "totalCount":  int(rng.uniform(100, 50000)),
                "pctACH":   round(mix[0], 2),
                "pctWire":  round(mix[1], 2),
                "pctR2P":   round(mix[2], 2),
                "pctCheck": round(mix[3], 2),
                "pctDebit": round(mix[4], 2),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def stacked_area_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """100% stacked area chart of rail mix over time."""
    fig = go.Figure()
    for rail, col, color in zip(RAIL_LABELS, RAIL_COLS, RAIL_COLORS.values()):
        fig.add_trace(go.Scatter(
            x=df["time_key"],
            y=df[col],
            name=rail,
            stackgroup="one",
            groupnorm="percent" if df[col].sum() != 0 else None,
            line=dict(width=0.5, color=color),
            fillcolor=color,
            mode="lines",
            hovertemplate=f"<b>{rail}</b><br>%{{y:.1f}}%<br>%{{x}}<extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title="Share of total payments (%)",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        margin=dict(t=60, b=40, l=50, r=20),
        height=380,
    )
    return fig


def mom_shift_heatmap(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    """
    Diverging heatmap — rows = NAICS codes, columns = rail × month.
    Blue = rail gained share; red = rail lost share.
    """
    pivot_frames = []
    for rail, col in zip(RAIL_LABELS, RAIL_COLS):
        p = df.pivot_table(
            index=group_col, columns="time_key",
            values=f"shift_{col}", aggfunc="sum",
        )
        p.columns = [f"{rail}\n{c}" for c in p.columns]
        pivot_frames.append(p)

    heatmap_df = pd.concat(pivot_frames, axis=1).fillna(0)
    z = heatmap_df.values
    max_abs = np.abs(z).max() or 1

    fig = go.Figure(go.Heatmap(
        z=z,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale=[[0.0, "#BF3A2A"], [0.5, "#F5F5F5"], [1.0, "#1A6FBF"]],
        zmid=0, zmin=-max_abs, zmax=max_abs,
        colorbar=dict(title="Shift (ppt)", thickness=12, len=0.8),
        hovertemplate="%{y} · %{x}<br>MoM shift: %{z:.1f} ppt<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        title=title,
        height=max(300, len(heatmap_df) * 28 + 100),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=60, l=120, r=20),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
    )
    return fig


def waterfall_chart(row_prev: pd.Series, row_curr: pd.Series, label: str) -> go.Figure:
    """Waterfall of %-point shift from prev to curr month per rail."""
    values = [round(row_curr[col] - row_prev[col], 2) for col in RAIL_COLS]
    texts  = [f"{'+' if v >= 0 else ''}{v:.1f} ppt" for v in values]
    fig = go.Figure(go.Waterfall(
        x=RAIL_LABELS,
        measure=["relative"] * len(RAIL_LABELS),
        y=values,
        text=texts,
        textposition="outside",
        connector=dict(line=dict(color=NEUTRAL_GRAY, width=0.8, dash="dot")),
        increasing=dict(marker_color=ANOMALY_GREEN),
        decreasing=dict(marker_color=ANOMALY_RED),
    ))
    fig.update_layout(
        title=f"Rail shift waterfall — {label}",
        yaxis_title="%-point change",
        yaxis_ticksuffix=" ppt",
        showlegend=False,
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig


def anomaly_scatter(df: pd.DataFrame, group_col: str) -> go.Figure:
    """Scatter: x = time_key, y = max MoM shift magnitude. Anomalies = diamond."""
    fig = go.Figure()
    palette = px.colors.qualitative.Set2
    shift_cols = [f"shift_{c}" for c in RAIL_COLS]

    for i, grp in enumerate(df[group_col].unique()):
        gdf = df[df[group_col] == grp].copy()
        available = [c for c in shift_cols if c in gdf.columns]
        if not available:
            continue
        gdf["max_shift"] = gdf[available].abs().max(axis=1)
        color = palette[i % len(palette)]
        is_anom = gdf.get("is_anomaly", pd.Series(False, index=gdf.index))
        normal, flagged = gdf[~is_anom], gdf[is_anom]

        if not normal.empty:
            fig.add_trace(go.Scatter(
                x=normal["time_key"], y=normal["max_shift"],
                mode="markers", name=str(grp),
                marker=dict(color=color, size=7, opacity=0.7),
                hovertemplate=f"<b>{grp}</b><br>Max shift: %{{y:.1f}} ppt<br>%{{x}}<extra></extra>",
            ))
        if not flagged.empty:
            fig.add_trace(go.Scatter(
                x=flagged["time_key"], y=flagged["max_shift"],
                mode="markers", name=f"{grp} ⚠",
                marker=dict(color=ANOMALY_RED, size=14, symbol="diamond",
                            line=dict(width=1.5, color="white")),
                hovertemplate=f"<b>{grp} ANOMALY</b><br>Max shift: %{{y:.1f}} ppt<br>%{{x}}<extra></extra>",
            ))

    fig.add_hline(y=10, line_dash="dash", line_color=ANOMALY_AMBER, opacity=0.6,
                  annotation_text="10 ppt reference", annotation_position="top right")
    fig.update_layout(
        title="Month-over-month rail shift magnitude",
        xaxis_title=None,
        yaxis_title="Max single-rail shift (ppt)",
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONNECTION + FILTERS
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Rail Shift Monitor")
    st.caption("PNC Treasury Management · Payment Knowledge Network")
    st.divider()

    with st.expander("🔌 Neo4j connection", expanded=False):
        neo4j_uri  = st.text_input("URI",      value="bolt://localhost:7687")
        neo4j_user = st.text_input("User",     value="neo4j")
        neo4j_pass = st.text_input("Password", type="password")
        use_demo   = st.checkbox("Use demo data (no Neo4j)", value=True)

    st.divider()

    st.markdown("### 📅 Date range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_dt = st.date_input("From", value=date(2024, 1, 1),
                                  min_value=date(2020, 1, 1), max_value=date(2026, 1, 1))
    with col_e:
        end_dt = st.date_input("To", value=date(2024, 12, 1),
                                min_value=date(2020, 1, 1), max_value=date(2026, 1, 1))
    start_month = start_dt.strftime("%Y-%m-01")
    end_month   = end_dt.strftime("%Y-%m-01")

    st.divider()

    st.markdown("### 🏭 NAICS granularity")
    naics_gran_label = st.radio(
        "Group by",
        list(NAICS_GRANULARITY.keys()),
        index=0,
        help="Controls how finely to slice the NAICS hierarchy",
    )
    naics_digits = NAICS_GRANULARITY[naics_gran_label]

    st.markdown("### 🔍 NAICS prefix filter")
    naics_prefix_input = st.text_input(
        "Filter to prefix (optional)",
        placeholder="e.g. 52 for Finance",
        help="Leave blank to include all industries.",
    )
    naics_prefix = naics_prefix_input.strip() or None

    st.divider()

    st.markdown("### ⚠ Anomaly detection")
    z_threshold = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.25,
                             help="Higher = fewer, more extreme anomalies flagged")
    ppt_threshold = st.slider("Min shift to surface (ppt)", 2, 30, 10, 1,
                               help="Pairs below this threshold are excluded from pair view")

    st.divider()

    st.markdown("### 🚂 Rail filter")
    rails_visible = st.multiselect(
        "Show rails", RAIL_LABELS, default=RAIL_LABELS,
        help="Hide rails you're not focused on",
    )

    refresh = st.button("🔄  Refresh data", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_naics_data(start_month, end_month, naics_digits, naics_prefix,
                    use_demo, neo4j_uri, neo4j_user, neo4j_pass):
    if use_demo:
        return generate_demo_data(naics_digits=naics_digits, naics_prefix=naics_prefix)
    driver = get_driver(neo4j_uri, neo4j_user, neo4j_pass)
    return run_query(
        driver, CYPHER["naics_mix_mom"],
        start_month=start_month,
        end_month=end_month,
        naics_digits=naics_digits,
        naics_prefix=naics_prefix,
    )


if refresh:
    st.cache_data.clear()

GROUP_COL = "naicsCode"

with st.spinner("Loading payment data…"):
    df_raw = load_naics_data(
        start_month, end_month, naics_digits, naics_prefix,
        use_demo, neo4j_uri, neo4j_user, neo4j_pass,
    )

if df_raw.empty:
    st.warning("No data returned. Check your date range and NAICS filter.")
    st.stop()

df = compute_mom_shifts(df_raw, GROUP_COL)
df = zscore_anomalies(df, GROUP_COL, z_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

naics_scope = f"prefix '{naics_prefix}'" if naics_prefix else "all industries"
st.markdown(f"## Rail Shift Monitor — NAICS {naics_gran_label} · {naics_scope}")
st.caption(
    f"{start_month} → {end_month}  ·  "
    f"{df_raw['time_key'].nunique()} months  ·  "
    f"{df_raw[GROUP_COL].nunique()} NAICS codes"
)

# ── KPI strip ────────────────────────────────────────────────────────────────
sorted_months = sorted(df_raw["time_key"].unique())
latest_month  = sorted_months[-1]
prev_month    = sorted_months[-2] if len(sorted_months) >= 2 else latest_month

latest = df_raw[df_raw["time_key"] == latest_month]
prev   = df_raw[df_raw["time_key"] == prev_month]

total_vol = latest["totalAmount"].sum()
prev_vol  = prev["totalAmount"].sum()
delta_vol = (total_vol - prev_vol) / prev_vol * 100 if prev_vol else 0
anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0

kpi_cols = st.columns(len(rails_visible) + 2)
kpi_cols[0].metric("Total volume (latest)", format_amount(total_vol), f"{delta_vol:+.1f}% MoM")
kpi_cols[1].metric("⚠ Anomalous periods", anomaly_count,
                   delta_color="inverse" if anomaly_count > 0 else "off")
for i, rail in enumerate(rails_visible):
    col_name  = f"pct{rail}"
    lat_share = latest[col_name].mean() if col_name in latest.columns else 0
    prv_share = prev[col_name].mean()   if col_name in prev.columns   else 0
    kpi_cols[i + 2].metric(
        f"{rail} share", f"{lat_share:.1f}%", f"{lat_share - prv_share:+.1f} ppt",
    )

st.divider()

tab_mix, tab_shift, tab_anomaly, tab_pair, tab_cypher = st.tabs([
    "📊 Channel Mix", "📈 MoM Shifts", "🚨 Anomaly Surface",
    "🔗 Pair Drilldown", "🗂 Cypher Reference",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — CHANNEL MIX
# ═══════════════════════════════════════════════════════════════════════
with tab_mix:
    st.subheader("Payment rail mix over time by NAICS industry")

    all_codes = sorted(df_raw[GROUP_COL].dropna().unique().tolist())
    selected_codes = st.multiselect(
        "NAICS codes to include", all_codes, default=all_codes[:6], key="mix_codes",
    )
    df_filtered = df_raw[df_raw[GROUP_COL].isin(selected_codes)]

    if df_filtered.empty:
        st.info("Select at least one NAICS code above.")
    else:
        agg = (
            df_filtered.groupby("time_key")[AMOUNT_COLS + ["totalAmount"]]
            .sum().reset_index()
        )
        for rail, col in zip(RAIL_LABELS, RAIL_COLS):
            agg[col] = agg[f"total{rail}"] / agg["totalAmount"].replace(0, np.nan) * 100

        label_str = ", ".join(str(c) for c in selected_codes[:3])
        if len(selected_codes) > 3:
            label_str += f" +{len(selected_codes)-3} more"

        left, right = st.columns([3, 1])
        with left:
            st.plotly_chart(stacked_area_chart(agg, f"Rail mix — {label_str}"),
                            use_container_width=True)
        with right:
            st.markdown("#### Latest mix snapshot")
            latest_row = agg[agg["time_key"] == agg["time_key"].max()].iloc[0]
            for rail, col in zip(RAIL_LABELS, RAIL_COLS):
                if rail in rails_visible:
                    share = latest_row[col]
                    color = RAIL_COLORS[rail]
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;padding:6px 0;border-bottom:1px solid #eee'>"
                        f"<span style='color:{color};font-weight:600'>{rail}</span>"
                        f"<span style='font-size:1.1em;font-weight:700'>{share:.1f}%</span></div>",
                        unsafe_allow_html=True,
                    )

        if len(selected_codes) > 1:
            st.markdown("#### Per-industry breakdown")
            cols = st.columns(min(len(selected_codes), 3))
            for i, code in enumerate(selected_codes):
                gdf = df_filtered[df_filtered[GROUP_COL] == code]
                gagg = gdf.groupby("time_key")[AMOUNT_COLS + ["totalAmount"]].sum().reset_index()
                for rail, col in zip(RAIL_LABELS, RAIL_COLS):
                    gagg[col] = gagg[f"total{rail}"] / gagg["totalAmount"].replace(0, np.nan) * 100
                with cols[i % 3]:
                    desc = DEMO_NAICS.get(code[:2], "") if use_demo else ""
                    label = f"{code}" + (f" · {desc}" if desc else "")
                    fig = stacked_area_chart(gagg, label)
                    fig.update_layout(height=250, showlegend=False, title_font_size=12,
                                      margin=dict(t=36, b=20, l=30, r=10))
                    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — MOM SHIFTS
# ═══════════════════════════════════════════════════════════════════════
with tab_shift:
    st.subheader("Month-over-month rail shift analysis")
    st.caption("Heatmap cells show %-point change vs. prior month. "
               "Blue = rail gained share, red = rail lost share.")

    all_codes = sorted(df[GROUP_COL].dropna().unique().tolist())
    selected_codes = st.multiselect(
        "NAICS codes to include", all_codes, default=all_codes[:8], key="shift_codes",
    )
    df_sel = df[df[GROUP_COL].isin(selected_codes)]

    if not df_sel.empty:
        st.plotly_chart(
            mom_shift_heatmap(df_sel, GROUP_COL, "MoM rail shift heatmap (ppt)"),
            use_container_width=True,
        )
        st.plotly_chart(anomaly_scatter(df_sel, GROUP_COL), use_container_width=True)

        st.markdown("#### Waterfall — compare two months")
        months = sorted(df_sel["time_key"].unique().tolist())
        if len(months) >= 2:
            wf_c1, wf_c2 = st.columns(2)
            with wf_c1:
                m_from = st.selectbox("From month", months[:-1],
                                      index=len(months) - 2, key="wf_from")
            with wf_c2:
                m_to_opts = [m for m in months if m > m_from]
                m_to = st.selectbox("To month", m_to_opts,
                                    index=len(m_to_opts) - 1, key="wf_to")

            wf_code = st.selectbox("NAICS code", selected_codes, key="wf_code")
            gdf = df_sel[df_sel[GROUP_COL] == wf_code]
            row_from = gdf[gdf["time_key"] == m_from]
            row_to   = gdf[gdf["time_key"] == m_to]
            if not row_from.empty and not row_to.empty:
                st.plotly_chart(
                    waterfall_chart(row_from.iloc[0], row_to.iloc[0],
                                    f"NAICS {wf_code}: {m_from} → {m_to}"),
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY SURFACE
# ═══════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.subheader("🚨 Anomaly surface")
    st.caption(f"NAICS codes with any rail's MoM shift exceeding {z_threshold}σ "
               "from its historical mean.")

    if "is_anomaly" not in df.columns or not df["is_anomaly"].any():
        st.success("No anomalies detected at the current sensitivity threshold.")
    else:
        anomalies = df[df["is_anomaly"]].copy()
        st.markdown(f"**{len(anomalies)} anomalous period(s)** across "
                    f"{anomalies[GROUP_COL].nunique()} NAICS codes.")

        shift_cols_display = {
            f"shift_{c}": f"Δ {r} (ppt)" for c, r in zip(RAIL_COLS, RAIL_LABELS)
        }
        display_cols = (
            [GROUP_COL, "time_key"]
            + [c for c in shift_cols_display if c in anomalies.columns]
            + (["max_abs_z"] if "max_abs_z" in anomalies.columns else [])
        )
        styled = (
            anomalies[display_cols]
            .rename(columns={**shift_cols_display, "max_abs_z": "Max |z|"})
            .sort_values("Max |z|", ascending=False)
            .reset_index(drop=True)
        )

        def color_shift(val):
            if not isinstance(val, (int, float)):
                return ""
            if val > 5:
                return "background-color: #dbeafe; color: #1e3a5f"
            if val < -5:
                return "background-color: #fee2e2; color: #7f1d1d"
            return ""

        st.dataframe(
            styled.style.applymap(color_shift,
                                  subset=[c for c in styled.columns if "Δ" in c]),
            use_container_width=True, height=350,
        )

        csv = styled.to_csv(index=False)
        st.download_button(
            "⬇ Download anomaly report (CSV)", csv,
            file_name=f"rail_anomalies_naics_{start_month}_{end_month}.csv",
            mime="text/csv",
        )

        st.markdown("#### Anomaly deep-dive")
        anom_opts = anomalies.apply(
            lambda r: f"NAICS {r[GROUP_COL]} · {r['time_key']}", axis=1
        ).tolist()
        selected_anom = st.selectbox("Select anomaly to inspect", anom_opts)
        if selected_anom:
            parts    = selected_anom.split(" · ")
            code_val = parts[0].replace("NAICS ", "")
            t_key    = parts[1]
            months_list = sorted(df[df[GROUP_COL] == code_val]["time_key"].unique())
            t_idx = months_list.index(t_key) if t_key in months_list else -1
            if t_idx > 0:
                prev_row = df[(df[GROUP_COL] == code_val) &
                              (df["time_key"] == months_list[t_idx - 1])].iloc[0]
                curr_row = df[(df[GROUP_COL] == code_val) &
                              (df["time_key"] == t_key)].iloc[0]
                st.plotly_chart(
                    waterfall_chart(prev_row, curr_row, selected_anom),
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — PAIR DRILLDOWN
# ═══════════════════════════════════════════════════════════════════════
with tab_pair:
    st.subheader("Counterparty pair drilldown")
    st.caption("Inspect channel mix for a specific src → dst relationship. "
               "Scope by NAICS prefix to browse all pairs within an industry.")

    col_src, col_dst = st.columns(2)
    with col_src:
        src_id = st.text_input("Source customer ID", placeholder="e.g. CUST-001")
    with col_dst:
        dst_id = st.text_input("Destination customer ID", placeholder="e.g. CUST-002")

    pair_naics_prefix = st.text_input(
        "Scope to NAICS prefix (optional)",
        value=naics_prefix or "",
        placeholder="e.g. 52",
        help="Pre-filled from the sidebar filter; override here if needed.",
        key="pair_naics_filter",
    )

    if src_id or dst_id or pair_naics_prefix:
        if use_demo:
            st.info("Pair drilldown requires a live Neo4j connection. "
                    "Uncheck 'Use demo data' in the sidebar and enter your credentials.")
        else:
            driver = get_driver(neo4j_uri, neo4j_user, neo4j_pass)
            pair_df = run_query(
                driver, CYPHER["pair_mix_mom"],
                start_month=start_month, end_month=end_month,
                src_id=src_id or None, dst_id=dst_id or None,
                naics_prefix=pair_naics_prefix.strip() or None,
            )
            if pair_df.empty:
                st.warning("No PAYS edges found for this filter in the selected date range.")
            else:
                pair_df = compute_mom_shifts(pair_df, "srcId")
                pair_df = zscore_anomalies(pair_df, "srcId", z_threshold)

                pair_ids = pair_df[["srcId", "srcName", "dstId", "dstName", "srcNaics"]].drop_duplicates()
                if len(pair_ids) > 1:
                    pair_opts = pair_ids.apply(
                        lambda r: f"{r['srcName']} ({r['srcId']}) → {r['dstName']} ({r['dstId']}) · NAICS {r['srcNaics']}",
                        axis=1,
                    ).tolist()
                    chosen     = st.selectbox("Select pair", pair_opts)
                    chosen_src = pair_ids.iloc[pair_opts.index(chosen)]["srcId"]
                    pair_df    = pair_df[pair_df["srcId"] == chosen_src]

                label = f"{pair_df['srcName'].iloc[0]} → {pair_df['dstName'].iloc[0]}"
                st.markdown(f"#### {label}")
                st.caption(f"Source NAICS: {pair_df['srcNaics'].iloc[0]}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Total payments", format_amount(pair_df["totalAmount"].sum()))
                m2.metric("Payment count",  f"{int(pair_df['totalCount'].sum()):,}")
                m3.metric("Months active",  pair_df["time_key"].nunique())

                st.plotly_chart(stacked_area_chart(pair_df, f"Rail mix — {label}"),
                                use_container_width=True)

                if "is_anomaly" in pair_df.columns and pair_df["is_anomaly"].any():
                    st.warning(f"{int(pair_df['is_anomaly'].sum())} anomalous month(s) detected for this pair.")
    else:
        st.markdown("Enter a source and/or destination customer ID, "
                    "or a NAICS prefix to explore all pairs within an industry.")
        st.markdown("**Tip:** Find interesting pairs in the Anomaly Surface tab first, "
                    "then paste their IDs here for a full channel mix history.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 5 — CYPHER REFERENCE
# ═══════════════════════════════════════════════════════════════════════
with tab_cypher:
    st.subheader("Cypher query reference")
    st.caption("All queries used by this module. Copy and run in Neo4j Browser or Bloom.")
    for key, query in CYPHER.items():
        with st.expander(f"📋 {key}"):
            st.code(query, language="cypher")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Rail Shift Monitor · PNC Payment Knowledge Network · "
    f"NAICS {naics_gran_label} · "
    f"Data range: {start_month} → {end_month} · "
    f"Source: {'Demo data' if use_demo else 'Neo4j @ ' + neo4j_uri}"
)
