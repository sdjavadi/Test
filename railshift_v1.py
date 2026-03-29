"""
Payment Rail Shift Monitor
PNC Treasury Management · Payment Knowledge Network
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4j"
DATABASE       = "payments2"

RAILS     = ["ACH", "Wire", "R2P", "Check", "Debit"]
RAIL_COLS = ["ach", "wire", "r2p", "check", "debit"]
MONTHS    = [f"{y}-{m:02d}" for y in [2024, 2025] for m in range(1, 13)]

# ── Neo4j ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_query(query: str, params: dict) -> pd.DataFrame:
    with get_driver().session(database=DATABASE) as session:
        result = session.run(query, params)
        return pd.DataFrame([r.data() for r in result])


# ── Queries ───────────────────────────────────────────────────────────────────

# Undirected PAYS match; fromId / toId let us infer direction per row
QUERY_CUSTOMERS = """
MATCH (n:PncCustomer {naics: $naics})-[r:PAYS]-(m:PncCustomer)
WHERE r.time_key >= date($start) AND r.time_key <= date($end)
RETURN
  n.mdmId                              AS nodeId,
  n.naics                              AS nodeNaics,
  m.mdmId                              AS neighborId,
  m.naics                              AS neighborNaics,
  startNode(r).mdmId                   AS fromId,
  endNode(r).mdmId                     AS toId,
  toString(r.time_key)                 AS time_key,
  coalesce(r.totalTransAmtACH,   0)    AS ach,
  coalesce(r.totalTransAmtWire,  0)    AS wire,
  coalesce(r.totalTransAmtR2P,   0)    AS r2p,
  coalesce(r.totalTransAmtCheck, 0)    AS check,
  coalesce(r.totalTransAmtDebit, 0)    AS debit,
  coalesce(r.totalTransAmt,      0)    AS totalAmt,
  coalesce(r.transFreq,          0)    AS transFreq
"""

QUERY_COUNTERPARTY = """
MATCH (n:PncCustomer {naics: $naics})-[r:PAYS_CPTY]->(c:Counterparty)
WHERE r.time_key >= date($start) AND r.time_key <= date($end)
RETURN
  n.mdmId                              AS nodeId,
  n.naics                              AS nodeNaics,
  c.unqCptyId                          AS neighborId,
  null                                 AS neighborNaics,
  n.mdmId                              AS fromId,
  c.unqCptyId                          AS toId,
  toString(r.time_key)                 AS time_key,
  coalesce(r.totalTransAmtACH,   0)    AS ach,
  coalesce(r.totalTransAmtWire,  0)    AS wire,
  coalesce(r.totalTransAmtR2P,   0)    AS r2p,
  coalesce(r.totalTransAmtCheck, 0)    AS check,
  coalesce(r.totalTransAmtDebit, 0)    AS debit,
  coalesce(r.totalTransAmt,      0)    AS totalAmt,
  coalesce(r.transFreq,          0)    AS transFreq
UNION ALL
MATCH (n:PncCustomer {naics: $naics})<-[r:CPTY_PAYS]-(c:Counterparty)
WHERE r.time_key >= date($start) AND r.time_key <= date($end)
RETURN
  n.mdmId                              AS nodeId,
  n.naics                              AS nodeNaics,
  c.unqCptyId                          AS neighborId,
  null                                 AS neighborNaics,
  c.unqCptyId                          AS fromId,
  n.mdmId                              AS toId,
  toString(r.time_key)                 AS time_key,
  coalesce(r.totalTransAmtACH,   0)    AS ach,
  coalesce(r.totalTransAmtWire,  0)    AS wire,
  coalesce(r.totalTransAmtR2P,   0)    AS r2p,
  coalesce(r.totalTransAmtCheck, 0)    AS check,
  coalesce(r.totalTransAmtDebit, 0)    AS debit,
  coalesce(r.totalTransAmt,      0)    AS totalAmt,
  coalesce(r.transFreq,          0)    AS transFreq
"""

QUERY_NAICS = """
MATCH (n:PncCustomer)
WHERE n.naics IS NOT NULL AND n.naics <> ''
RETURN DISTINCT n.naics AS naics
ORDER BY naics
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["direction"] = np.where(df["nodeId"] == df["fromId"], "outgoing", "incoming")
    return df


def fmt_amount(val: float) -> str:
    sign = "-" if val < 0 else ""
    val = abs(val)
    if val >= 1e9:
        return f"{sign}${val/1e9:.2f}B"
    if val >= 1e6:
        return f"{sign}${val/1e6:.2f}M"
    if val >= 1e3:
        return f"{sign}${val/1e3:.2f}K"
    return f"{sign}${val:,.0f}"


def apply_filters(
    cust_df: pd.DataFrame,
    cpty_df: pd.DataFrame,
    direction: str,
    node_type: str,
) -> pd.DataFrame:
    """Filter combined dataframe by direction and counterpart node type."""
    parts = []
    dir_val = {"All": None, "Incoming": "incoming", "Outgoing": "outgoing"}[direction]

    for df, include in [
        (cust_df, node_type in ("All", "Customers")),
        (cpty_df, node_type in ("All", "Counterparties")),
    ]:
        if include and not df.empty:
            tmp = df if dir_val is None else df[df["direction"] == dir_val]
            parts.append(tmp)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── Page Setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rail Shift Monitor", layout="wide")
st.title("Payment Rail Shift Monitor")
st.caption("PNC Treasury Management · Payment Knowledge Network")

# ── Load NAICS ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading industries from database…")
def load_naics():
    return run_query(QUERY_NAICS, {})["naics"].tolist()

naics_list = load_naics()

# ── Filters ───────────────────────────────────────────────────────────────────
col_naics, col_time = st.columns([2, 3])

with col_naics:
    selected_naics = st.selectbox("Industry (NAICS)", naics_list)

with col_time:
    time_range = st.select_slider(
        "Time Range",
        options=MONTHS,
        value=(MONTHS[0], MONTHS[-1]),
    )

start_date = time_range[0] + "-01"
end_date   = time_range[1] + "-01"

run_btn = st.button("Analyze", type="primary")

# ── Analysis ──────────────────────────────────────────────────────────────────
if run_btn:
    params = {"naics": selected_naics, "start": start_date, "end": end_date}

    with st.spinner("Fetching customer–customer transactions…"):
        cust_df = run_query(QUERY_CUSTOMERS, params)

    with st.spinner("Fetching counterparty transactions…"):
        cpty_df = run_query(QUERY_COUNTERPARTY, params)

    if cust_df.empty and cpty_df.empty:
        st.warning("No transactions found for the selected NAICS and time range.")
        st.stop()

    if not cust_df.empty:
        cust_df = add_direction(cust_df)
    if not cpty_df.empty:
        cpty_df = add_direction(cpty_df)

    # Full combined view — used for all fixed metrics (unaffected by distribution filters)
    all_df  = pd.concat([cust_df, cpty_df], ignore_index=True)
    in_df   = all_df[all_df["direction"] == "incoming"]
    out_df  = all_df[all_df["direction"] == "outgoing"]

    # ── Summary Metrics ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary")

    n_customers   = all_df["nodeId"].nunique()
    total_inflow  = in_df["totalAmt"].sum()
    total_outflow = out_df["totalAmt"].sum()
    net_flow      = total_inflow - total_outflow

    in_deg_per_node  = in_df.groupby("nodeId")["neighborId"].nunique()
    out_deg_per_node = out_df.groupby("nodeId")["neighborId"].nunique()
    avg_in_deg  = in_deg_per_node.mean()  if not in_deg_per_node.empty  else 0.0
    avg_out_deg = out_deg_per_node.mean() if not out_deg_per_node.empty else 0.0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Customers",         f"{n_customers:,}")
    m2.metric("Total Inflow",      fmt_amount(total_inflow))
    m3.metric("Total Outflow",     fmt_amount(total_outflow))
    m4.metric("Net Flow",          fmt_amount(net_flow))
    m5.metric("Avg In-Degree",     f"{avg_in_deg:.1f}")
    m6.metric("Avg Out-Degree",    f"{avg_out_deg:.1f}")

    # ── Contextual Tables ─────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3, c4 = st.columns(4)

    # Top receiving industries (who sends money TO our segment)
    with c1:
        st.markdown("**Top Inbound Industries**")
        recv_naics_df = (
            in_df[in_df["neighborNaics"].notna()]
            .groupby("neighborNaics", as_index=False)["totalAmt"].sum()
            .nlargest(5, "totalAmt")
            .rename(columns={"neighborNaics": "NAICS", "totalAmt": "Amount"})
        )
        recv_naics_df["Amount"] = recv_naics_df["Amount"].apply(fmt_amount)
        st.dataframe(recv_naics_df, hide_index=True, use_container_width=True)

    # Top sending industries (who our segment pays)
    with c2:
        st.markdown("**Top Outbound Industries**")
        send_naics_df = (
            out_df[out_df["neighborNaics"].notna()]
            .groupby("neighborNaics", as_index=False)["totalAmt"].sum()
            .nlargest(5, "totalAmt")
            .rename(columns={"neighborNaics": "NAICS", "totalAmt": "Amount"})
        )
        send_naics_df["Amount"] = send_naics_df["Amount"].apply(fmt_amount)
        st.dataframe(send_naics_df, hide_index=True, use_container_width=True)

    # Dominant inbound rails by amount
    with c3:
        st.markdown("**Inbound Rail Volume**")
        rail_in_totals = {r: in_df[c].sum() for r, c in zip(RAILS, RAIL_COLS)}
        rail_in_df = (
            pd.DataFrame(rail_in_totals.items(), columns=["Rail", "Amount"])
            .sort_values("Amount", ascending=False)
        )
        rail_in_df["Amount"] = rail_in_df["Amount"].apply(fmt_amount)
        st.dataframe(rail_in_df, hide_index=True, use_container_width=True)

    # Dominant outbound rails by amount
    with c4:
        st.markdown("**Outbound Rail Volume**")
        rail_out_totals = {r: out_df[c].sum() for r, c in zip(RAILS, RAIL_COLS)}
        rail_out_df = (
            pd.DataFrame(rail_out_totals.items(), columns=["Rail", "Amount"])
            .sort_values("Amount", ascending=False)
        )
        rail_out_df["Amount"] = rail_out_df["Amount"].apply(fmt_amount)
        st.dataframe(rail_out_df, hide_index=True, use_container_width=True)

    # ── Rail Mix Over Time ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Rail Mix Over Time")

    tab_in, tab_out = st.tabs(["Inbound", "Outbound"])

    def build_rail_mix_chart(df: pd.DataFrame) -> go.Figure:
        if df.empty:
            return go.Figure()
        mix = df.groupby("time_key")[RAIL_COLS].sum().reset_index()
        row_total = mix[RAIL_COLS].sum(axis=1).replace(0, np.nan)
        for col, rail in zip(RAIL_COLS, RAILS):
            mix[rail] = mix[col] / row_total * 100
        mix_long = (
            mix.melt(id_vars="time_key", value_vars=RAILS, var_name="Rail", value_name="Percent")
            .sort_values("time_key")
        )
        fig = px.bar(
            mix_long, x="time_key", y="Percent", color="Rail",
            barmode="stack",
            labels={"time_key": "Month", "Percent": "% of Amount"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(yaxis_range=[0, 100], height=340, legend_title="Rail",
                          margin=dict(t=20, b=40))
        return fig

    with tab_in:
        st.plotly_chart(build_rail_mix_chart(in_df), use_container_width=True)

    with tab_out:
        st.plotly_chart(build_rail_mix_chart(out_df), use_container_width=True)

    # ── Distribution Filters ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Distributions")

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        dir_filter  = st.selectbox("Direction",        ["All", "Incoming", "Outgoing"])
    with f2:
        type_filter = st.selectbox("Counterpart Type", ["All", "Customers", "Counterparties"])
    with f3:
        use_log = st.checkbox("Log Scale", value=False)

    filtered_df = apply_filters(cust_df, cpty_df, dir_filter, type_filter)

    if filtered_df.empty:
        st.info("No data for the selected direction / counterpart type.")
    else:
        # Per-node degree and flow aggregations
        in_filt  = filtered_df[filtered_df["direction"] == "incoming"]
        out_filt = filtered_df[filtered_df["direction"] == "outgoing"]

        def node_stats(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
            """Build per-node in/out/net degree and flow."""
            all_nodes = set(df_in["nodeId"].tolist()) | set(df_out["nodeId"].tolist())

            in_deg  = df_in.groupby("nodeId")["neighborId"].nunique().rename("in_degree")
            out_deg = df_out.groupby("nodeId")["neighborId"].nunique().rename("out_degree")
            in_flow = df_in.groupby("nodeId")["totalAmt"].sum().rename("in_flow")
            out_flow= df_out.groupby("nodeId")["totalAmt"].sum().rename("out_flow")

            stats = (
                pd.DataFrame(index=list(all_nodes))
                .join(in_deg, how="left")
                .join(out_deg, how="left")
                .join(in_flow, how="left")
                .join(out_flow, how="left")
                .fillna(0)
            )
            stats["net_degree"] = stats["in_degree"] - stats["out_degree"]
            stats["net_flow"]   = stats["in_flow"]   - stats["out_flow"]
            return stats.reset_index().rename(columns={"index": "nodeId"})

        stats_df = node_stats(in_filt, out_filt)

        # Decide which series to show based on direction filter
        show_in  = dir_filter in ("All", "Incoming")
        show_out = dir_filter in ("All", "Outgoing")
        show_net = dir_filter == "All"

        col_deg, col_flow = st.columns(2)

        # ── Degree Distribution
        with col_deg:
            st.markdown("**Degree Distribution**")
            deg_traces = []
            if show_in:
                deg_traces.append(
                    go.Histogram(x=stats_df["in_degree"],  name="In-Degree",
                                 marker_color="#3B82F6", opacity=0.75)
                )
            if show_out:
                deg_traces.append(
                    go.Histogram(x=stats_df["out_degree"], name="Out-Degree",
                                 marker_color="#F59E0B", opacity=0.75)
                )
            if show_net:
                deg_traces.append(
                    go.Histogram(x=stats_df["net_degree"], name="Net-Degree",
                                 marker_color="#10B981", opacity=0.75)
                )
            fig_deg = go.Figure(data=deg_traces)
            fig_deg.update_layout(
                barmode="overlay",
                height=320,
                xaxis_title="Degree",
                yaxis_title="Count",
                yaxis_type="log" if use_log else "linear",
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig_deg, use_container_width=True)

        # ── Flow Distribution
        with col_flow:
            st.markdown("**Flow Distribution**")
            flow_traces = []
            if show_in:
                flow_traces.append(
                    go.Histogram(x=stats_df["in_flow"],  name="Inflow",
                                 marker_color="#3B82F6", opacity=0.75)
                )
            if show_out:
                flow_traces.append(
                    go.Histogram(x=stats_df["out_flow"], name="Outflow",
                                 marker_color="#F59E0B", opacity=0.75)
                )
            if show_net:
                flow_traces.append(
                    go.Histogram(x=stats_df["net_flow"], name="Net Flow",
                                 marker_color="#10B981", opacity=0.75)
                )
            fig_flow = go.Figure(data=flow_traces)
            fig_flow.update_layout(
                barmode="overlay",
                height=320,
                xaxis_title="Transaction Amount ($)",
                yaxis_title="Count",
                yaxis_type="log" if use_log else "linear",
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=10, b=40),
            )
            st.plotly_chart(fig_flow, use_container_width=True)
