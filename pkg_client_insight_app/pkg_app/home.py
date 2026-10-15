"""Overview for leadership: what each module sees this month, and the clients that show up in more than one."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .core import nav, state, ui
from .core.fmt import display_name, money


def page():
    ui.page_header("Payment Knowledge Graph · client insight",
                   "What our TM clients' payments tell us: whose deposits may leave, which payment patterns deserve a second look, "
                   "and how money connects our clients.", "🧭")
    mods = state.allowed_modules()
    if not mods:
        ui.empty_state("Your account has no module enabled. Ask the app owner for access."); return
    tiles = {}
    cols = st.columns(len(mods))
    for col, m in zip(cols, mods):
        with col.container(border=True, height="stretch"):
            if m == "attrition": tiles[m] = _attrition_tile()
            elif m == "fraud": tiles[m] = _fraud_tile()
            else: _network_tile()

    if tiles.get("attrition") and tiles.get("fraud"):
        both = sorted(tiles["attrition"]["keys"] & tiles["fraud"]["keys"])
        st.markdown("#### Clients in more than one view")
        st.caption("The same client seen through different lenses — one page holds everything we know about it.")
        if both:
            from .attrition import service as AS
            from .fraud import service as FS
            _, L, _ = AS.watchlist(); F = FS.latest().drop_duplicates("key").set_index("key")
            A = L.set_index("key")
            t = pd.DataFrame([dict(key=k, client=display_name(A.customer_name.get(k), k), industry=ui.first_txt(A.naics_desc.get(k)),
                                   **{"attrition rank": int(A["rank"].get(k)), "chance money leaves": A.p.get(k),
                                      "pattern": FS.T.name(F.primary.get(k)), "tier": F.tier.get(k), "$ involved": F.usd_involved.get(k)})
                               for k in both]).sort_values("attrition rank")
            ev = st.dataframe(t, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="home_both",
                              column_order=["client", "industry", "attrition rank", "chance money leaves", "pattern", "tier", "$ involved"],
                              column_config={"chance money leaves": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                                             "$ involved": st.column_config.NumberColumn(format="dollar")})
            rows = ev.selection.rows if ev else []
            if rows:
                k = t.iloc[rows[0]].key
                if st.button(f"Open {t.iloc[rows[0]].client} →", type="primary", key="home_open"):
                    nav.open_customer(k, lens="fraud")
            else:
                st.caption("Select a row to open the client.")
        else:
            st.caption("No overlap this month.")

    with st.expander("How it works", expanded=False):
        st.markdown(
            "1. **Payment Knowledge Graph.** Every ACH, wire, RTP, check and card payment between our clients and their "
            "counterparties becomes an edge between two parties, by day and by month.\n"
            "2. **Customer metrics.** Each month, about 120 measures per client: money in and out, counterparties, concentration, "
            "new and lost relationships, banks used, payment rails, transfers to its own accounts, pass-through.\n"
            "3. **Modules.** Each use case reads the same metrics: an attrition model joined with deposit balances; rule-and-peer "
            "scores for suspicious-activity patterns; and the network itself, one hop around any client.\n"
            "4. **One customer page.** Whatever brought you to a client, the same page shows every lens you are allowed to see.\n\n"
            "Access is set per module, by Posit Connect group." + (" Suspicious-activity signals are restricted to Compliance / AML "
            "and leadership, so they never reach client-facing staff." if "fraud" in mods else ""))


def _attrition_tile():
    from .attrition import pages as AP
    st.markdown("<div class='tile-h'>📉 Deposit attrition</div><div class='tile-p'>Clients whose deposits are likely to leave in the "
                "next six months, with the reason and the conversation to have.</div>", unsafe_allow_html=True)
    t = AP.home_tile()
    if not t:
        st.caption("No data yet."); return None
    c1, c2 = st.columns(2)
    c1.metric("On the warning list", f"{t['n']:,}"); c2.metric("Money at stake", money(t["at_stake"]))
    c1.metric("Expected to leave", money(t["expected"])); c2.metric("Money leaving now", f"{t['urgent']:,}")
    st.caption(f"Scored {t['month']}")
    if st.button("Open the warning list →", key="home_att", type="primary", width="stretch"): nav.go("attrition")
    return t


def _fraud_tile():
    from .fraud import pages as FP
    from .fraud import typologies as T
    st.markdown("<div class='tile-h'>🛡️ Suspicious activity</div><div class='tile-p'>Payment patterns that resemble known "
                "money-laundering typologies, for Compliance / AML review.</div>", unsafe_allow_html=True)
    t = FP.home_tile()
    if not t:
        st.caption("No data yet."); return None
    c1, c2 = st.columns(2)
    c1.metric("Entities flagged", f"{t['n']:,}"); c2.metric("High tier", f"{t['high']:,}")
    c1.metric("Money involved", money(t["usd"])); c2.metric("New this month", f"{t['new']:,}")
    fig = go.Figure(go.Bar(x=[n for _, n in t["by"]], y=[f"{T.icon(k)} {T.name(k)}" for k, _ in t["by"]], orientation="h",
                           marker_color=ui.RED, text=[n for _, n in t["by"]], textposition="auto"))
    fig.update_yaxes(autorange="reversed"); fig.update_xaxes(visible=False)
    fig.update_layout(height=170, margin=dict(l=0, r=0, t=0, b=0), template="plotly_white", font=dict(family=ui.FONT))
    st.plotly_chart(fig, config={"displayModeBar": False}, key="home_fr_bar")
    st.caption(f"Scored {t['month']} · internal review only")
    if st.button("Open the review queue →", key="home_fr", type="primary", width="stretch"): nav.go("fraud")
    return t


def _network_tile():
    st.markdown("<div class='tile-h'>🕸️ Payment network</div><div class='tile-p'>Who pays whom: payers, payees and banks around any "
                "client, and the PNC clients it is connected to.</div>", unsafe_allow_html=True)
    st.markdown("<div class='kv'>• <b>Flow diagram</b> — payers → client → payees<br>• <b>Network map</b> — click a PNC client to follow the "
                "money<br>• <b>Flags</b> from the other modules, on every node you are allowed to see</div>", unsafe_allow_html=True)
    st.caption("A one-hop preview; the full ego-network explorer will plug in here.")
    if st.button("Explore the network →", key="home_net", type="primary", width="stretch"): nav.go("network")
