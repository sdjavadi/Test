"""
app.py — PKG Explorer (Spotlight + Deep Dive)
Run:  streamlit run app.py
Layers: data.py (access) / logic.py (computation) / app.py (UI only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import logic

# ---------------------------------------------------------------- style ----
st.set_page_config(page_title="PKG Explorer", layout="wide",
                   initial_sidebar_state="collapsed")

ACCENT = "#0b6ee6"          # single informational accent
ALERT = "#d92d20"           # reserved for risk only
GOOD = "#067647"
MUTED = "#667085"
BG = "#f7f8fa"
PANEL = "#ffffff"
TEXT = "#1a1d26"
BORDER = "#e4e7ec"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono&display=swap');
html, body, [class*="css"] {{ font-family:'IBM Plex Sans',sans-serif;
        color:{TEXT}; }}
.stApp {{ background:{BG}; }}
h1,h2,h3 {{ font-weight:600; letter-spacing:-0.01em; color:{TEXT}; }}
div[data-testid="stMetricValue"] {{ font-family:'IBM Plex Mono',monospace; }}
.card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px;
        padding:16px 18px 12px 18px; margin-bottom:4px; min-height:150px;
        box-shadow:0 1px 2px rgba(16,24,40,0.05); }}
.card h4 {{ margin:0 0 2px 0; font-size:0.95rem; color:{TEXT}; }}
.card .sub {{ color:{MUTED}; font-size:0.78rem;
             font-family:'IBM Plex Mono',monospace; }}
.card .headline {{ font-family:'IBM Plex Mono',monospace; font-size:1.15rem;
                  margin:4px 0; }}
.card .story {{ color:{MUTED}; font-size:0.82rem; line-height:1.35; }}
.badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:0.75rem; margin-right:6px; margin-bottom:6px;
         border:1px solid {BORDER}; background:#f2f4f7; color:#344054; }}
.badge.risk {{ border-color:{ALERT}; color:{ALERT}; background:#fef3f2; }}
.badge.good {{ border-color:{GOOD}; color:{GOOD}; background:#ecfdf3; }}
</style>""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    template="plotly_white", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(family="IBM Plex Sans", size=12, color=TEXT),
    margin=dict(l=40, r=20, t=40, b=30), height=320,
)

RISK_ROLES = {"bleeding", "conduit", "anchor_dependent", "infra_dependent",
              "intermittent"}
GOOD_ROLES = {"expanding", "steady", "diversified", "local_anchor"}


def badge(text: str) -> str:
    cls = ("risk" if text in RISK_ROLES
           else "good" if text in GOOD_ROLES else "")
    tip = logic.ROLE_DEFS.get(text, "").replace('"', "&quot;")
    return (f'<span class="badge {cls}" title="{tip}">'
            f'{text.replace("_", " ")}</span>')


# ------------------------------------------------------------- controls ----
st.title("PKG Explorer")
top = st.columns([2, 2, 6])
with st.expander("Analyst mode (graph version)", expanded=False):
    version = st.radio(
        "Graph version", ["V0", "P99_9", "P99", "P99_99"], horizontal=True,
        help="V0 = full graph incl. infrastructure flow. P-tiers exclude "
             "hub/whale nodes above that percentile (see metrics manifest).")
months = list(data.months(version))
month = top[0].selectbox("Month", months, index=len(months) - 1)

if "selected_node" not in st.session_state:
    st.session_state.selected_node = None

tab_spot, tab_dive = st.tabs(["🔦 Spotlight", "🔬 Customer Deep Dive"])

# ------------------------------------------------------------ SPOTLIGHT ----
with tab_spot:
    qcol, scol = st.columns([5, 1])
    queue_name = qcol.radio("Queue", list(logic.QUEUES.keys()),
                            horizontal=True, label_visibility="collapsed")
    fn, desc = logic.QUEUES[queue_name]
    st.caption(desc)
    q = fn(version, month)

    if scol.button("🎲 Surprise me", use_container_width=True) and len(q):
        pick = q.sample(1).iloc[0]
        st.session_state.selected_node = pick["node"]
        st.info(f"Picked **{pick.get('cust_name') or pick['node']}** — "
                f"open the Deep Dive tab.")

    if q.empty:
        st.warning("No cases in this queue for the selected month/version.")
    else:
        rows = [q.iloc[i:i + 3] for i in range(0, min(len(q), 9), 3)]
        for chunk in rows:
            cols = st.columns(3)
            for col, (_, r) in zip(cols, chunk.iterrows()):
                risk_q = queue_name in ("Revenue at risk", "Sustained drift",
                                        "New conduits")
                color = ALERT if risk_q else GOOD
                with col:
                    st.markdown(f"""
<div class="card">
 <h4>{r.get('cust_name') or r['node']}</h4>
 <div class="sub">{r['node']} · NAICS {r.get('naics2','—')}</div>
 <div class="headline" style="color:{color}">{r['headline']}</div>
 <div class="story">{r['narrative']}</div>
</div>""", unsafe_allow_html=True)
                    if st.button("Deep dive →", key=f"dd_{r['node']}",
                                 use_container_width=True):
                        st.session_state.selected_node = r["node"]
                        st.toast("Open the Customer Deep Dive tab",
                                 icon="🔬")

# ------------------------------------------------------------ DEEP DIVE ----
with tab_dive:
    latest_df = data.node_month(version, months[-1])
    name_map = dict(zip(latest_df["node"], latest_df["cust_name"]))
    all_nodes = sorted(latest_df["node"].unique())
    default_ix = (all_nodes.index(st.session_state.selected_node)
                  if st.session_state.selected_node in all_nodes else 0)
    node = st.selectbox(
        "Customer", all_nodes, index=default_ix,
        format_func=lambda x: f"{name_map.get(x) or x}  ({x})")
    h = logic.history(node, version)
    if h.empty:
        st.stop()
    last = h.iloc[-1]

    # --- identity strip -----------------------------------------------------
    id_l, id_r = st.columns([3, 1])
    with id_l:
        st.subheader(str(last.get("cust_name") or node))
        st.caption(node)
        badges = "".join(
            badge(str(last.get(f"{t}_stable", "—")))
            for t in logic.TAXONOMIES if pd.notna(last.get(f"{t}_stable")))
        st.markdown(badges, unsafe_allow_html=True)
        st.caption(
            f"NAICS {last.get('naics4') or last.get('naics2') or 'unknown'}"
            f" · active {int(last.get('months_active', 0))} months"
            f" · peer group {int(last['peer_size']) if pd.notna(last.get('peer_size')) else '—'}"
            f" @ naics{int(last['peer_level']) if pd.notna(last.get('peer_level')) and last.get('peer_level') else '—'}")
    with id_r:
        stab = last.get("role_stability_norm", np.nan)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=float(stab) if pd.notna(stab) else 0,
            number={"valueformat": ".2f"},
            title={"text": "Role stability", "font": {"size": 13}},
            gauge={"axis": {"range": [0, 1]},
                   "bar": {"color": ACCENT},
                   "bgcolor": "#f2f4f7", "borderwidth": 0}))
        fig.update_layout(**{**PLOTLY_LAYOUT, "height": 170,
                             "margin": dict(l=10, r=10, t=30, b=5)})
        st.plotly_chart(fig, use_container_width=True)

    # --- definitions ---------------------------------------------------------
    with st.expander("\u2139\ufe0f What do these roles and metrics mean?"):
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("**This customer's current roles** *(hover any "
                        "badge above for the same definitions)*")
            for tax in logic.TAXONOMIES:
                cur = last.get(f"{tax}_stable")
                if pd.isna(cur):
                    continue
                st.markdown(
                    f"- **{str(cur).replace('_', ' ')}** "
                    f"({tax.replace('_role', '')} \u2014 "
                    f"{logic.TAXONOMY_DEFS[tax].lower()}): "
                    f"{logic.ROLE_DEFS.get(str(cur), '')}")
        with gcol2:
            st.markdown("**Charts & metrics on this page**")
            for name, d in logic.METRIC_DEFS.items():
                st.markdown(f"- **{name}**: {d}")

    # --- strength time series with role-change markers ----------------------
    fig = go.Figure()
    for col, name, colr in (("in_strength", "Inflow", GOOD),
                            ("out_strength", "Outflow", ACCENT),
                            ("net_flow", "Net", MUTED)):
        fig.add_trace(go.Scatter(x=h["time_key"], y=h[col], name=name,
                                 mode="lines+markers",
                                 line=dict(color=colr, width=2)))
    for _, ch in logic.role_changes(h).iterrows():
        fig.add_vline(x=ch["time_key"], line_dash="dot",
                      line_color="#98a2b3", opacity=0.6)
        fig.add_annotation(x=ch["time_key"], yref="paper", y=1.02,
                           text=ch["label"], showarrow=False,
                           font=dict(size=10, color=MUTED),
                           textangle=-25)
    fig.update_layout(**PLOTLY_LAYOUT, title="Monthly flow ($)",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([3, 2])

    # --- role ribbon ---------------------------------------------------------
    with c1:
        rib = logic.role_ribbon(h)
        mlist = list(h["time_key"])
        pos = {m: i for i, m in enumerate(mlist)}
        fig = go.Figure()
        palette = ["#4cc2ff", "#57d9a3", "#f2c14e", "#ff5c5c", "#b48cff",
                   "#7ce0e0", "#e88fc0", "#9aa46b", "#8b93a7"]
        role_color = {}
        for _, r in rib.iterrows():
            role_color.setdefault(r["role"],
                                  palette[len(role_color) % len(palette)])
            x0, x1 = pos[r["start"]], pos[r["end"]]
            fig.add_trace(go.Bar(
                y=[r["taxonomy"].replace("_role", "")],
                x=[max(x1 - x0, 0.9)], base=[x0], orientation="h",
                marker_color=role_color[r["role"]],
                hovertemplate=(f"{r['role']}<br>{r['start']} → "
                               f"{r['end']}<extra></extra>"),
                showlegend=False, marker_line_width=0))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                          title="Role ribbon (stable roles)",
                          xaxis=dict(tickmode="array",
                                     tickvals=list(range(len(mlist))),
                                     ticktext=mlist))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("One band per taxonomy; color changes are stabilized "
                   "role transitions.")

    # --- peer bullet ----------------------------------------------------------
    with c2:
        ps = logic.peer_snapshot(h)
        fig = go.Figure(go.Bar(
            x=ps["pctl"], y=ps["metric"], orientation="h",
            marker_color=[ALERT if (m == "Momentum" and v is not None
                                    and v <= 0.2)
                          else ACCENT for m, v in
                          zip(ps["metric"], ps["pctl"])]))
        fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED)
        fig.update_layout(**PLOTLY_LAYOUT, title="Vs. industry peers "
                          "(percentile)", xaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

    # --- payer turnover --------------------------------------------------------
    t = logic.turnover_series(h)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["time_key"], y=t["retained"], name="Retained",
                         marker_color=ACCENT))
    fig.add_trace(go.Bar(x=t["time_key"], y=t["new"], name="New payers",
                         marker_color=GOOD))
    fig.add_trace(go.Bar(x=t["time_key"], y=t["lost"], name="Lost payers",
                         marker_color=ALERT))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="relative",
                      title="Inflow by payer cohort ($): retained / new "
                            "above, lost below")
    st.plotly_chart(fig, use_container_width=True)
