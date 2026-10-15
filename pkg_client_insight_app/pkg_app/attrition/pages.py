"""Deposit attrition: the signal-card page and the customer lens."""
import datetime
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..core import nav, state, ui
from ..core.fmt import display_name, md2html
from . import labels as LB
from . import service as AS

FOCUS = ["All", "Money leaving now", "New this month", "Relationship risk", "Competitive threat"]


def _name(r):
    return display_name(getattr(r, "customer_name", None), r.key if hasattr(r, "key") else getattr(r, "cust_pwr_id", ""))


def spark(h, key):
    fig = go.Figure(go.Scatter(x=h.month, y=h.bal / 1e6, mode="lines", line=dict(color=ui.BLUE, width=2), fill="tozeroy",
                               fillcolor="rgba(29,78,137,.08)", hovertemplate="%{x}: $%{y:,.2f}m<extra></extra>"))
    if len(h): fig.add_hline(y=float(h.bal_norm.iloc[-1]) / 1e6, line=dict(color=ui.GREY, dash="dot", width=1))
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False, template="plotly_white")
    st.plotly_chart(fig, config={"displayModeBar": False}, key=key)


# ═════════════════════════════════════════════════════════════════════
# SIGNAL CARDS
# ═════════════════════════════════════════════════════════════════════
def cards_page():
    ui.page_header("Deposit attrition early warning", "Clients whose deposits are likely to leave within six months — ranked, "
                   "explained, with the conversation to have.", "📉")
    ok, err = AS.status()
    if not ok:
        ui.empty_state(f"The attrition module has no data yet ({err}). Run the export cell of the attrition model notebook, "
                       "then `INVALIDATE METADATA` on its two tables.", "🛠️")
        return
    with st.container(border=True):
        f1, f2, f3 = st.columns([1.1, 2.3, 5], vertical_alignment="bottom")
        K = f1.number_input("Clients on the list", 20, 5000, AS.default_k(), step=20, key="att_K", persist_state="session",
                            help="How many clients the team can call this month.")
        ALPHA = f2.segmented_control("Rank by", list(AS.ALPHAS), default=float(state.CFG["alpha"]), format_func=AS.ALPHAS.get,
                                     key="att_alpha", required=True, persist_state="session")
        focus = f3.pills("Show", FOCUS, default="All", key="att_focus", required=True, persist_state="session")
        S, L, MONTHS = AS.watchlist(int(K), float(ALPHA))
        LATEST = MONTHS[0]
        f4, f5, f6 = st.columns([3, 1.6, 3], vertical_alignment="bottom")
        seg = f4.multiselect("Segment", sorted(L.segment_desc.dropna().unique()) if "segment_desc" in L else [], key="att_seg",
                             persist_state="session")
        min_bal = f5.selectbox("Min. normal balance", [0, 1e5, 1e6, 1e7, 1e8], format_func=lambda v: LB.money(v, 0) if v else "any",
                               key="att_minbal", persist_state="session")
        search = f6.text_input("Search name or id", key="att_q", persist_state="session")
    view = L
    if focus == "Money leaving now": view = view[view.urgent]
    elif focus == "New this month": view = view[view.months_on_list == 1]
    elif focus == "Relationship risk": view = view[view.story.map(lambda s: any(LB.theme(f) == "relationship" for f in s["early"] + [x[0] for x in s["reasons"]]))]
    elif focus == "Competitive threat": view = view[view.theme == "competitive"]
    if seg: view = view[view.segment_desc.isin(seg)]
    if min_bal: view = view[view.bal_norm >= min_bal]
    if search: view = view[view.customer_name.fillna("").str.contains(search, case=False, regex=False) | view.cust_pwr_id.str.contains(search, regex=False)]

    b = AS.bundle()
    ui.kpis([("Clients on the list", f"{len(L):,}", f"Top {int(K):,} this month, ranked by {AS.ALPHAS[float(ALPHA)]}"),
             ("Money at stake", LB.money(L.capped.sum()), "Balance now, never more than the client's normal balance"),
             ("Expected to leave", LB.money((L.p * L.capped).sum()), "Chance × money at stake, summed"),
             ("New this month", f"{int((L.months_on_list == 1).sum()):,}"),
             ("Money leaving now", f"{int(L.urgent.sum()):,}")])
    per = 12; pages = max(1, int(np.ceil(len(view) / per)))
    p1, p2 = st.columns([6, 1], vertical_alignment="center")
    p1.caption(f"Scored {LATEST} · model trained through {b.get('trained_through', '—')} · {len(view):,} cards match the filters"
               + (f" · {per} per page" if pages > 1 else ""))
    page = p2.number_input("Page", 1, pages, 1, key="att_page", label_visibility="collapsed") if pages > 1 else 1
    chunk = view.iloc[(page - 1) * per: page * per]
    H = AS.history(state.MODE, tuple(chunk.cust_pwr_id)) if len(chunk) else pd.DataFrame()
    cols = st.columns(3)
    for i, r in enumerate(chunk.itertuples()):
        s = r.story
        with cols[i % 3].container(border=True):
            st.markdown(f"<div class='card-name'>#{r.rank} · {_name(r)}</div>"
                        f"<div class='card-sub'>{ui.first_txt(getattr(r, 'segment_desc', ''))} · {ui.first_txt(getattr(r, 'naics_desc', ''))} · "
                        f"ref {r.key}</div>", unsafe_allow_html=True)
            with st.container(horizontal=True, gap="small"):
                st.badge("Money leaving now" if s["urgent"] else "Early warning", color="red" if s["urgent"] else "orange",
                         icon=":material/priority_high:" if s["urgent"] else ":material/schedule:")
                st.badge("New" if r.months_on_list == 1 else f"{r.months_on_list} mo on list", color="blue")
                st.badge(LB.PLAYS[s["theme"]][0], color="violet")
            c1, c2, c3 = st.columns(3)
            c1.metric("Chance (6 mo)", f"{r.p:.0%}")
            c2.metric("At stake", LB.money(r.capped, 0))
            c3.metric("vs normal", f"{s['pct_norm']:.0%}" if pd.notna(s["pct_norm"]) else "—")
            h = H[H.cust_pwr_id == r.cust_pwr_id].tail(12) if len(H) else H
            if len(h): spark(h, f"att_sp_{r.cust_pwr_id}")
            why = "".join(f"<li>{'▲' if (LB.to_display(f, x) or 0) > 0 else '▼'} {AS.label(f)}: <b>{LB.fmt_value(f, x)}</b></li>"
                          for f, _, x in s["reasons"][:3])
            st.markdown(f"<b>Why</b><ul class='why'>{why}</ul>", unsafe_allow_html=True)
            st.markdown("<div class='next'>" + "<br>".join(md2html(x) for x in s["next"]) + "</div>", unsafe_allow_html=True)
            if st.button("Prepare the call →", key=f"att_go_{r.cust_pwr_id}", type="primary", width="stretch"):
                nav.open_customer(r.key, lens="attrition")
    if not len(view):
        ui.empty_state("No client matches these filters.")


# ═════════════════════════════════════════════════════════════════════
# CUSTOMER LENS
# ═════════════════════════════════════════════════════════════════════
def lens(info):
    ok, err = AS.status()
    if not ok:
        ui.empty_state(f"Attrition data is unavailable ({err})."); return
    cust = info.get("attr_cust")
    if not cust:
        ui.empty_state("This client is not in the attrition population this month: it holds no deposits with us, or its money has "
                       "already left (the model scores only clients whose money is still here).")
        return
    S, L, MONTHS = AS.watchlist()
    LATEST = MONTHS[0]; K, _ = AS.current_settings(); b = AS.bundle(); feats = b["feats"]
    rr = S[(S.cust_pwr_id == cust) & (S.month == LATEST)]
    if not len(rr):
        ui.empty_state("No score for this client in the latest month."); return
    r = rr.iloc[0]
    on_list = r["rank"] <= K
    s = AS.card_story(b, r)
    months_on = int(L.loc[L.cust_pwr_id == cust, "months_on_list"].iloc[0]) if on_list else 0
    df, X, p_hist, C, FM = AS.customer_detail(state.MODE, cust)
    h = AS.history(state.MODE, (cust,))

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Chance money leaves (6 mo)", f"{r.p:.0%}", delta=f"{(p_hist[-1] - p_hist[-2]) * 100:+.0f} pts vs last month" if len(p_hist) > 1 else None,
              delta_color="inverse")
    k2.metric("Rank this month", f"#{int(r['rank']):,}", help=f"The list holds the top {K:,}" + ("" if on_list else " — this client is below the cut"))
    k3.metric("Money at stake", LB.money(r.capped))
    k4.metric("Balance now", LB.money(r.bal), delta=f"{s['pct_norm']:.0%} of normal" if pd.notna(s["pct_norm"]) else None, delta_color="off")
    k5.metric("Normal balance", LB.money(r.bal_norm))
    k6.metric("Months on the list", f"{months_on}")
    t_name, t_text = LB.PLAYS[s["theme"]]
    st.markdown(f"<div class='play'><b>{t_name}.</b> {t_text}</div>", unsafe_allow_html=True)

    g1, g2 = st.columns([1.6, 1])
    with g1:
        fig = go.Figure()
        fig.add_bar(x=h.month, y=h.bal / 1e6, name="average balance", marker_color="rgba(29,78,137,.75)")
        if "bal_eom" in h: fig.add_scatter(x=h.month, y=h.bal_eom / 1e6, name="month-end", mode="lines+markers", line=dict(color=ui.ORANGE))
        fig.add_scatter(x=h.month, y=h.bal_norm / 1e6, name="normal", mode="lines", line=dict(color=ui.GREY, dash="dash"))
        for m in S[(S.cust_pwr_id == cust) & (S["rank"] <= K)].month:
            fig.add_vrect(x0=m, x1=m, line=dict(color=ui.RED, width=10), opacity=.15)
        fig.update_yaxes(title_text="$m")
        st.plotly_chart(ui.base_fig(fig, 340, "Balance ($m) — shaded months: on the list"), key="att_bal")
    with g2:
        fig = go.Figure(go.Scatter(x=df.month, y=p_hist * 100, mode="lines+markers", line=dict(color=ui.RED, width=3),
                                   hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
        fig.update_yaxes(title_text="%", rangemode="tozero")
        st.plotly_chart(ui.base_fig(fig, 340, "Chance the money leaves within 6 months"), key="att_score")

    st.markdown("#### Why the model flagged this client")
    cc = pd.DataFrame({"f": feats, "c": C[-1], "x": X[-1]}).assign(a=lambda d: d.c.abs()).sort_values("a", ascending=False).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=cc.c, y=[f"{AS.label(f)}  ({LB.fmt_value(f, x)})" for f, x in zip(cc.f, cc.x)], orientation="h",
                           marker_color=[ui.RED if v > 0 else ui.TEAL for v in cc.c], hovertemplate="%{y}<extra></extra>"))
    fig.update_xaxes(title_text="← lowers risk      raises risk →")
    st.plotly_chart(ui.base_fig(fig, 380), key="att_why")

    st.markdown("#### Signals against clients of the same size who stayed")
    rows = []
    for f, (th, d) in b.get("thr", {}).items():
        if f not in FM.columns: continue
        fired_m = list(df.month[FM[f].to_numpy(bool)])
        cur = X[-1, feats.index(f)] if f in feats else np.nan
        rows.append({"signal": AS.label(f), "now": LB.fmt_value(f, cur), "stayers' threshold": LB.fmt_value(f, th),
                     "leavers": "fall below" if d == "falls" else "rise above", "firing now": "🔴" if bool(FM[f].iloc[-1]) else "—",
                     "months fired": len(fired_m), "first fired": fired_m[0] if fired_m else "—",
                     "typical warning": f"{abs(int(b['onset'][f]))} mo" if b.get("onset", {}).get(f) else "—"})
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values(["firing now", "months fired"], ascending=[True, False]), hide_index=True, width="stretch")

    st.markdown("#### Payment behavior")
    _pay = [c for c in ("amt_all_in", "amt_all_out", "ntxn_all_in") if c in h]
    if not _pay or h[_pay].fillna(0).abs().sum().sum() == 0:
        st.info("No payment activity recorded for this client in the payment data (it may bank its payments through another "
                "entity of the relationship, or only hold deposits here).")
    pc = st.columns(3)
    panels = [("Dollars in and out ($m)", [("amt_all_in", "in", ui.TEAL), ("amt_all_out", "out", ui.ORANGE)], 1e6),
              ("Payments and receiving days", [("ntxn_all_in", "incoming payments", ui.TEAL), ("active_days_in", "days with receipts", ui.BLUE)], 1),
              ("Payers, payees and banks", [("ncpty_all_in", "payers", ui.TEAL), ("ncpty_all_out", "payees", ui.ORANGE), ("n_fi_out", "banks paid", ui.RED)], 1)]
    for col, (title, series, scale) in zip(pc, panels):
        fig = go.Figure()
        for c, nm, colr in series:
            if c in h: fig.add_scatter(x=h.month, y=h[c] / scale, name=nm, mode="lines+markers", line=dict(color=colr, width=2))
        col.plotly_chart(ui.base_fig(fig, 280, title), key=f"att_pay_{title}")
    if "selfpay_amt_out" in h and h.selfpay_amt_out.fillna(0).sum() > 0:
        fig = go.Figure(go.Bar(x=h.month, y=h.selfpay_amt_out / 1e6, marker_color=ui.RED))
        st.plotly_chart(ui.base_fig(fig, 240, "Money sent to its own accounts at other banks ($m)"), key="att_self")

    st.markdown("#### Other entities in the same relationship")
    rltn = r.get("rltn_pwr_id")
    if ui.is_txt(rltn):
        rel = AS.relationship(state.MODE, rltn)
        oth = rel[rel.cust_pwr_id != cust]
        if len(oth):
            fig = go.Figure()
            for c_, g_ in oth.groupby("cust_pwr_id"):
                fig.add_scatter(x=g_.month, y=g_.bal / 1e6, mode="lines", name=display_name(g_.customer_name.iloc[0], str(c_).lstrip("0"))[:30])
            st.plotly_chart(ui.base_fig(fig, 280, "Balances of other entities ($m)"), key="att_rel")
        else:
            st.caption("No other entities in this relationship.")
    else:
        st.caption("No relationship recorded.")

    st.markdown("#### Payers who cut back, and where outgoing money goes")
    if st.toggle("Load payers, payees and banks (reads payment detail; can take a minute)", key=f"att_cp_{cust}"):
        mdm = info.get("mdm_ids") or state.source().mdm_ids(cust)
        if not mdm:
            st.caption("No payment entities linked to this client.")
        else:
            inn, out = AS.counterparties(state.MODE, tuple(mdm), LATEST)
            ms = sorted(set(inn.get("month", pd.Series(dtype=str))) | set(out.get("month", pd.Series(dtype=str)))); rec = ms[-3:]
            def change(d, keys=("cpty_id", "name", "bank")):
                cols = list(keys) + ["prior 3 mo", "last 3 mo", "change"]
                if d is None or not len(d): return pd.DataFrame(columns=cols)
                d = d.assign(name=d.get("name", pd.Series(index=d.index, dtype=object)).fillna(d.cpty_id.astype(str)),
                             bank=d.get("bank", pd.Series(index=d.index, dtype=object)).fillna("unknown bank"),
                             amount=pd.to_numeric(d.amount, errors="coerce").fillna(0.0),
                             period=np.where(d.month.isin(rec), "last 3 mo", "prior 3 mo"))
                g = d.pivot_table(index=list(keys), columns="period", values="amount", aggfunc="sum", fill_value=0.0).reset_index()
                for c in ("prior 3 mo", "last 3 mo"):
                    if c not in g: g[c] = 0.0
                g["change"] = g["last 3 mo"] - g["prior 3 mo"]
                return g[cols]
            ci = change(inn)
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("**Payers that stopped or cut back** (money in)")
                if not len(ci): st.caption("No incoming payments in the last 6 months.")
                lost = ci[ci["prior 3 mo"] > 0].sort_values("change").head(8)
                st.dataframe(ui.money_col(lost, ["prior 3 mo", "last 3 mo", "change"], LB.money)[["name", "bank", "prior 3 mo", "last 3 mo", "change"]],
                             hide_index=True, width="stretch")
            with a2:
                st.markdown("**Where outgoing money goes, by bank**")
                out2 = out.assign(own=np.where(out["is_self"].fillna(False).astype(bool), "own account", "")) if len(out) and "is_self" in out else out.assign(own="")
                ob = change(out2, keys=("bank", "own")).sort_values("last 3 mo", ascending=False).head(8)
                st.dataframe(ui.money_col(ob, ["prior 3 mo", "last 3 mo", "change"], LB.money), hide_index=True, width="stretch")

    st.markdown("#### Call brief")
    nm = display_name(info.get("name"), info.get("key"))
    reasons_txt = "\n".join(f"- {AS.label(f)}: {LB.fmt_value(f, x)}" for f, _, x in s["reasons"][:5])
    fired_txt = "\n".join(f"- {AS.label(f)}" for f in s["early"] + s["late"]) or "- none beyond the stayers' threshold"
    brief = (f"# Call brief — {nm}\n\nClient ref {info.get('key')} · scored {LATEST} · rank #{int(r['rank'])}\n\n"
             f"- Chance the money leaves within 6 months: {r.p:.0%}\n- Money at stake: {LB.money(r.capped)} "
             f"(balance {LB.money(r.bal)}, normal {LB.money(r.bal_norm)})\n- Months on the list: {months_on}\n\n"
             f"## Why the model flagged it\n{reasons_txt}\n\n## Signals firing\n{fired_txt}\n\n## Suggested conversation\n**{t_name}.** {t_text}\n")
    b1, b2 = st.columns([2, 1])
    with b1:
        with st.expander("Preview the brief", expanded=False):
            st.markdown(brief)
    with b2:
        st.download_button("Download call brief", brief, file_name=f"call_brief_{info.get('key')}_{LATEST}.md", icon="📄", width="stretch")

    with st.form(f"att_fb_{cust}", border=True):
        st.markdown("**After the call** — helps measure what the list is worth")
        useful = st.feedback("thumbs", key=f"att_thumb_{cust}")
        outcome = st.selectbox("Outcome", ["Not called yet", "Client staying", "Client moving part of the money", "Client leaving anyway",
                                           "Already known to the RM", "Could not reach", "Not relevant"])
        note = st.text_area("Notes", height=80)
        if st.form_submit_button("Save", icon="💾"):
            row = pd.DataFrame([dict(saved=datetime.datetime.now().isoformat(timespec="seconds"), user=state.current_user()["user"],
                                     cust_pwr_id=cust, month=LATEST, p=round(float(r.p), 4), rank=int(r["rank"]),
                                     useful={0: "no", 1: "yes"}.get(useful, ""), outcome=outcome, note=note)])
            fp = pathlib.Path(state.CFG["feedback_path"])
            row.to_csv(fp, mode="a", header=not fp.exists(), index=False)
            st.toast("Saved", icon="✅")


def home_tile():
    """Numbers for the overview page."""
    ok, err = AS.status()
    if not ok: return None
    S, L, MONTHS = AS.watchlist()
    return dict(month=MONTHS[0], n=len(L), at_stake=L.capped.sum(), expected=(L.p * L.capped).sum(), urgent=int(L.urgent.sum()),
                new=int((L.months_on_list == 1).sum()), keys=set(L.key))
