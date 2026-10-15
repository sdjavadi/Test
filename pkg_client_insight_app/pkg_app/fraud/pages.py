"""Suspicious activity: the signal-card page and the customer lens."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ..core import nav, network, state, ui
from ..core.data import ym_add
from ..core.fmt import display_name, md2html, money
from . import service as FS
from . import typologies as T

TIERS = ["High", "Elevated", "Watch"]
SORTS = {"Score": "score", "$ involved": "usd_involved", "Months flagged": "months_flagged"}


def _typ_badges(r):
    ks = sorted(T.listed(r), key=lambda k: -float(getattr(r, f"score_{k}", 0) or 0))
    with st.container(horizontal=True, gap="small"):
        st.badge(f"{r.tier}", color=ui.TIER_COLOR.get(r.tier, "gray"), icon=":material/flag:")
        for k in ks[:2]:
            st.badge(f"{T.icon(k)} {T.name(k)}", color=T.TYPES[k]["color"])
        if len(ks) > 2: st.badge(f"+{len(ks) - 2} more", color="gray")
        st.badge("New" if r.new else f"{r.months_flagged} of 3 months", color="blue")


# ═════════════════════════════════════════════════════════════════════
# SIGNAL CARDS
# ═════════════════════════════════════════════════════════════════════
def cards_page():
    ui.page_header("Suspicious activity review", "Payment patterns that resemble known money-laundering typologies, found in the "
                   "monthly payment metrics — for Compliance / AML review.", "🛡️")
    ui.compliance_note()
    ok, err = FS.status()
    if not ok:
        ui.empty_state(f"The suspicious-activity module has no data yet ({err}). Run pkg_fraud_signals_v1.ipynb, then "
                       "`INVALIDATE METADATA` on its three tables.", "🛠️")
        return
    S, months = FS.signals(); LATEST = months[-1]
    with st.container(border=True):
        f1, f3 = st.columns([5.2, 1.6], vertical_alignment="bottom")
        when = f3.segmented_control("Flagged", ["This month", "Any of last 3"], default="This month", key="fr_when", required=True,
                                    persist_state="session")
        L = FS.latest(any_month=(when == "Any of last 3"))
        cnt = {k: int(L.typologies.str.contains(k).sum()) for k in T.ORDER}
        typ = f1.pills("Pattern", T.ORDER, selection_mode="multi", format_func=lambda k: f"{T.icon(k)} {T.name(k)} · {cnt[k]}",
                       key="fr_typ", persist_state="session")
        f2, g1, g2, g3, g4 = st.columns([2.2, 1.1, 1.6, 2.4, 1.4], vertical_alignment="bottom")
        tier = f2.segmented_control("Tier", TIERS, selection_mode="multi", default=["High", "Elevated"], key="fr_tier", persist_state="session")
        only_new = g1.toggle("New only", key="fr_new", persist_state="session")
        min_usd = g2.selectbox("Minimum $ involved", [0, 1e5, 1e6, 1e7], format_func=lambda v: money(v, 0) if v else "any", key="fr_min",
                               persist_state="session")
        q = g3.text_input("Search name or id", key="fr_q", persist_state="session")
        sort = g4.selectbox("Sort by", list(SORTS), key="fr_sort", persist_state="session")
    view = L
    if typ: view = view[view.typologies.map(lambda s: any(k in s.split(",") for k in typ))]
    if tier: view = view[view.tier.isin(tier)]
    if only_new: view = view[view.new]
    if min_usd: view = view[view.usd_involved >= min_usd]
    if q: view = view[view.customer_name.fillna("").str.contains(q, case=False, regex=False) | view.key.fillna("").str.contains(q, regex=False)
                      | view.mdm_id.str.contains(q, regex=False)]
    view = view.sort_values([SORTS[sort], "score"], ascending=False)

    ui.kpis([("Entities flagged", f"{len(L):,}", f"Payment entities matching at least one pattern ({'in ' + LATEST if when == 'This month' else 'in any of the last 3 months'})"),
             ("High tier", f"{int((L.tier == 'High').sum()):,}", "Top 50 per pattern each month, scoring ≥ 70"),
             ("Money involved", money(L.usd_involved.sum()), "Dollars in the matched pattern over 3 months (largest pattern per entity)"),
             ("New this month", f"{int(L.new.sum()):,}", "Flagged for the first time in the latest month"),
             ("More than one pattern", f"{int((L.n_typ > 1).sum()):,}")])
    with st.expander("How the patterns are found and scored", expanded=False):
        st.markdown(
            "Each pattern has a **rule** (all conditions must hold over a 3-month window) and a **score** of 0–100 that ranks how "
            "extreme the client is against **similar clients** (same industry where large enough, and the same size band). "
            "Ordinary explanations are built in: processor-like clients and industries where pass-through is normal (payroll, "
            "escrow, title, brokers) are discounted; transfers to the client's own accounts elsewhere are left out. "
            "Tiers follow review capacity: within each pattern and month, **High** = the top 50 scoring ≥ 70, **Elevated** = the next 150 "
            "scoring ≥ 50, **Watch** = the rest. A client matching several patterns takes its best tier and gets +5 per extra pattern. "
            "Left out of screening: individuals, PNC's own entities, banks, payment processors, payroll companies, GSEs and government.")
        st.dataframe(pd.DataFrame([dict(pattern=f"{T.icon(k)} {T.name(k)}", what=v["what"], why=v["why"],
                                         **{"could also be": "; ".join(v["innocent"])}) for k, v in T.TYPES.items()]),
                     hide_index=True, width="stretch")
        fig = go.Figure()
        for t_, col in (("High", ui.RED), ("Elevated", ui.ORANGE), ("Watch", ui.GREY)):
            fig.add_bar(y=[T.name(k) for k in T.ORDER], x=[int((L.typologies.str.contains(k) & (L.tier == t_)).sum()) for k in T.ORDER],
                        name=t_, orientation="h", marker_color=col)
        fig.update_layout(barmode="stack"); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(ui.base_fig(fig, 240, "Flagged entities by pattern and tier"), key="fr_overview")
    per = 12; pages = max(1, int(np.ceil(len(view) / per)))
    p1, p2 = st.columns([6, 1], vertical_alignment="center")
    p1.caption(f"Signals for {months[0]} – {LATEST} · {len(view):,} cards match the filters" + (f" · {per} per page" if pages > 1 else ""))
    page = p2.number_input("Page", 1, pages, 1, key="fr_page", label_visibility="collapsed") if pages > 1 else 1
    chunk = view.iloc[(page - 1) * per: page * per]
    H = FS.history(state.MODE, tuple(chunk.mdm_id)) if len(chunk) else pd.DataFrame()
    cols = st.columns(3)
    for i, r in enumerate(chunk.itertuples()):
        prim = r.primary if r.primary in T.TYPES else (T.listed(r) or [None])[0]
        with cols[i % 3].container(border=True):
            st.markdown(f"<div class='card-name'>{display_name(r.customer_name, r.key)}</div>"
                        f"<div class='card-sub'>{ui.first_txt(r.naics_desc, default='industry unknown')} · ref {r.key}"
                        f"{' · entity ' + r.mdm_id if str(r.key).removeprefix('M:') != r.mdm_id else ''}</div>", unsafe_allow_html=True)
            _typ_badges(r)
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{r.score:.0f}")
            c2.metric("$ involved", money(r.usd_involved, 1), help="Over the 3-month window")
            lbl, val = T.key_stat(prim, r); c3.metric(lbl, val)
            h = H[H.mdm_id == r.mdm_id].tail(12) if len(H) else H
            if len(h): ui.spark_in_out(h, f"fr_sp_{r.mdm_id}", out_col="amt_all_out")
            ev = T.evidence(prim, r)[:2]
            for k2 in [k for k in T.listed(r) if k != prim][:1]: ev += T.evidence(k2, r)[:1]
            st.markdown("<ul class='why'>" + "".join(f"<li>{md2html(x)}</li>" for x in ev) + "</ul>", unsafe_allow_html=True)
            g = T.guards(r)
            alt = ("<b>Already considered:</b> " + g[0]) if g else ("<b>Could also be:</b> " + "; ".join(T.TYPES[prim]["innocent"][:2]) if prim else "")
            st.markdown(f"<div class='alt'>{alt}</div>", unsafe_allow_html=True)
            if st.button("Review the case →", key=f"fr_go_{r.mdm_id}", type="primary", width="stretch"):
                nav.open_customer(r.key, lens="fraud", entity=r.mdm_id)
    if not len(view):
        ui.empty_state("No entity matches these filters.")


# ═════════════════════════════════════════════════════════════════════
# CUSTOMER LENS
# ═════════════════════════════════════════════════════════════════════
def _flows_chart(h, r, key):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=h.month, y=h.amt_all_in / 1e6, name="money in", marker_color="rgba(18,163,154,.8)")
    fig.add_bar(x=h.month, y=h.out_adj / 1e6, name="money out (excl. own accounts)", marker_color="rgba(242,140,40,.8)")
    if "bal_avg" in h and h.bal_avg.notna().any():
        fig.add_scatter(x=h.month, y=h.bal_avg / 1e6, name="average balance", mode="lines+markers", line=dict(color=ui.NAVY, width=2), secondary_y=False)
    fig.update_yaxes(title_text="$m")
    st.plotly_chart(ui.base_fig(fig, 300, "Money in, money out and the balance left behind ($m)"), key=key)


def _typ_chart(k, r, h, e, key):
    if k == "passthrough":
        _flows_chart(h, r, key); return
    if k == "funnel":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=h.month, y=h.ncpty_all_in, name="payers", marker_color="rgba(18,163,154,.8)")
        fig.add_bar(x=h.month, y=h.ncpty_all_out, name="payees", marker_color="rgba(242,140,40,.8)")
        fig.add_scatter(x=h.month, y=h.top1_share_out * 100, name="% to the largest recipient", mode="lines+markers",
                        line=dict(color=ui.VIOLET, width=2), secondary_y=True)
        fig.update_yaxes(title_text="count", secondary_y=False); fig.update_yaxes(title_text="%", range=[0, 100], secondary_y=True)
        st.plotly_chart(ui.base_fig(fig, 300, "Many payers in, few payees out"), key=key); return
    if k == "split":
        d = e[e.typology == "split"].copy() if len(e) else e
        if not len(d):
            st.caption("Payment-level evidence is not stored for this window."); return
        d["per"] = d.amount / d.n_txn.clip(lower=1); thr = float(d.threshold.dropna().iloc[0]) if d.threshold.notna().any() else 1e4
        fig = go.Figure()
        for flag, col, nm in ((True, ui.RED, "split day (several payments, one counterparty, total over the threshold)"), (False, ui.ORANGE, "single payment just under")):
            g = d[d.split_day == flag]
            if len(g):
                fig.add_scatter(x=g.txn_date, y=g.per, mode="markers", name=nm, marker=dict(color=col, size=6 + 3 * g.n_txn.clip(upper=5)),
                                customdata=np.c_[g.n_txn, g.amount, g.cpty_name.fillna("")],
                                hovertemplate="%{x}: %{customdata[0]:.0f} payment(s) averaging $%{y:,.0f}, total $%{customdata[1]:,.0f} to %{customdata[2]}<extra></extra>")
        fig.add_hline(y=thr, line=dict(color=ui.NAVY, dash="dash"), annotation_text=f"threshold {money(thr)}")
        fig.update_yaxes(title_text="payment size ($)", range=[thr * .85, thr * 1.08])
        st.plotly_chart(ui.base_fig(fig, 300, "Payments just under the threshold, by day"), key=key); return
    if k == "spike":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=h.month, y=h.flow_m / 1e6, name="money in + out", marker_color="rgba(29,78,137,.8)")
        if pd.notna(getattr(r, "base", None)): fig.add_hline(y=float(r.base) / 1e6, line=dict(color=ui.GREY, dash="dot"), annotation_text="its normal month")
        if "new_amt_share_in" in h:
            fig.add_scatter(x=h.month, y=h.new_amt_share_in * 100, name="% from new payers", mode="lines+markers", line=dict(color=ui.RED), secondary_y=True)
        fig.update_yaxes(title_text="$m", secondary_y=False); fig.update_yaxes(title_text="%", range=[0, 100], secondary_y=True)
        st.plotly_chart(ui.base_fig(fig, 300, "Activity against its own normal"), key=key); return
    if k == "roundtrip":
        d = e[e.typology == "roundtrip"] if len(e) else e
        if not len(d):
            st.caption("Pair-level evidence is not stored for this window."); return
        d = d.assign(cp=d.cpty_name.where(d.cpty_name.notna() & (d.cpty_name.astype(str) != ""), d.cpty_id).astype(str))
        g = d.groupby("cp", as_index=False).agg(sent=("amount", "sum"), back=("amount_back", "sum")).sort_values("sent", ascending=False).head(8)
        fig = go.Figure()
        fig.add_bar(y=g.cp, x=g.sent / 1e6, name="sent to it", orientation="h", marker_color=ui.ORANGE)
        fig.add_bar(y=g.cp, x=g.back / 1e6, name="came back from it", orientation="h", marker_color=ui.TEAL)
        fig.update_layout(barmode="group"); fig.update_xaxes(title_text="$m"); fig.update_yaxes(autorange="reversed")
        st.plotly_chart(ui.base_fig(fig, 300, "Money that went out and came back, by counterparty (3 months)"), key=key)


def _rules_table(k, r):
    rows = T.rules(k, r)
    st.dataframe(pd.DataFrame([dict(check=a, **{"this client": b}, rule=c, met="✅" if d else "—") for a, b, c, d in rows]),
                 hide_index=True, width="stretch")


def _unflagged(info):
    st.success("Not flagged by any of the five patterns in the last three months.", icon="✅")
    mdm = info.get("mdm_ids") or []
    if not mdm: return
    try:
        end = network.latest_month(state.MODE)
        h = FS.metrics_history(state.MODE, tuple(mdm), ym_add(end, -11))
    except Exception as e:
        st.caption(f"Payment history unavailable ({type(e).__name__})."); return
    if not len(h): return
    g = h.groupby("month", as_index=False)[["amt_all_in", "amt_all_out"]].sum()
    fig = go.Figure()
    fig.add_bar(x=g.month, y=g.amt_all_in / 1e6, name="money in", marker_color="rgba(18,163,154,.8)")
    fig.add_bar(x=g.month, y=g.amt_all_out / 1e6, name="money out", marker_color="rgba(242,140,40,.8)")
    st.plotly_chart(ui.base_fig(fig, 280, "Money in and out, last 12 months ($m)"), key="fr_unflagged")


def lens(info):
    ui.compliance_note(short=True)
    ok, err = FS.status()
    if not ok:
        ui.empty_state(f"Suspicious-activity data is unavailable ({err})."); return
    S, months = FS.signals()
    rows = S[(S.key == info["key"]) | S.mdm_id.isin(info.get("mdm_ids") or [])]
    flagged = rows[rows.score > 0]
    if not len(flagged):
        _unflagged(info); return
    ents = list(flagged.sort_values("score", ascending=False).mdm_id.unique())
    ent = ents[0]
    if len(ents) > 1:
        want = st.session_state.get("nav_entity")
        ent = st.segmented_control("Payment entity", ents, default=want if want in ents else ents[0], key=f"fr_ent_{info['key']}", required=True)
    R = rows[rows.mdm_id == ent].sort_values("month")
    F = R[R.score > 0]
    r = (F[F.month == months[-1]] if (F.month == months[-1]).any() else F.sort_values("score", ascending=False)).iloc[0]
    rr = next(R[R.index == r.name].itertuples())
    h = FS.history(state.MODE, (ent,)); e = FS.evidence(state.MODE, (ent,))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tier", r.tier, help="Within each pattern and month: High = top 50 scoring ≥ 70 · Elevated = next 150 scoring ≥ 50 · Watch = the rest")
    k2.metric("Score", f"{r.score:.0f}", delta=(f"{r.score - R.score.iloc[-2]:+.0f} vs last month" if len(R) > 1 and r.month == months[-1] else None),
              delta_color="inverse")
    k3.metric("$ involved (3 mo)", money(r.usd_involved))
    k4.metric("Months flagged", f"{int(r.months_flagged)} of 3", help=f"First flagged {r.first_flagged}")
    k5.metric("Patterns", f"{int(r.n_typ)}", help=", ".join(T.name(k) for k in T.listed(rr)))
    st.caption(f"Scored {r.month} · compared with {r.peer_group}, size band {int(r.size_band) if pd.notna(r.size_band) else '—'} of 5"
               f" ({int(r.peer_n) if pd.notna(r.peer_n) else '—'} clients) · payment entity {ent}")

    st.markdown("#### What we see")
    for k in sorted(T.listed(rr), key=lambda k: -float(getattr(rr, f"score_{k}", 0) or 0)):
        t = T.TYPES[k]
        with st.container(border=True):
            st.markdown(f"**{t['icon']} {t['name']}** · score **{float(getattr(rr, f'score_{k}', 0) or 0):.0f}** — {t['what']}")
            a, b = st.columns([1.35, 1])
            with a:
                _typ_chart(k, rr, h, e, key=f"fr_ch_{k}_{ent}")
            with b:
                st.markdown("".join(f"<div class='ev'>• {md2html(x)}</div>" for x in T.evidence(k, rr)), unsafe_allow_html=True)
                _rules_table(k, rr)
                st.markdown(f"<div class='muted'><b>Why it matters:</b> {t['why']}<br><b>Could also be:</b> {'; '.join(t['innocent'])}</div>",
                            unsafe_allow_html=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Compared with similar clients")
        prs = list(dict.fromkeys(p for k in T.listed(rr) for p in T.PEER_FOR[k] + ["pr_flow_3m"]))
        prs = [p for p in prs if p in R.columns and pd.notna(getattr(rr, p, None))]
        if prs:
            v = [float(getattr(rr, p)) * 100 for p in prs]
            fig = go.Figure(go.Bar(x=v, y=[T.PEER_LABEL[p] for p in prs], orientation="h", marker_color=[ui.RED if x >= 90 else ui.ORANGE if x >= 75 else ui.GREY for x in v],
                                   text=[("highest among them" if x >= 99 else f"higher than {x:.0f}%") for x in v], textposition="auto"))
            fig.add_vline(x=50, line=dict(color=ui.NAVY, dash="dot"), annotation_text="typical")
            fig.update_xaxes(range=[0, 100], title_text="percentile among similar clients"); fig.update_yaxes(autorange="reversed")
            st.plotly_chart(ui.base_fig(fig, 60 + 46 * len(prs)), key=f"fr_peer_{ent}")
        else:
            st.caption("No peer comparison for this pattern (it is measured against a fixed threshold).")
    with c2:
        st.markdown("#### Score over the last three months")
        Z = np.array([[float(x or 0) for x in R[f"score_{k}"]] for k in T.ORDER])
        fig = go.Figure(go.Heatmap(z=Z, x=list(R.month), y=[T.name(k) for k in T.ORDER], colorscale=[[0, "#f3f6fa"], [.5, "#f6c58f"], [1, ui.RED]],
                                   zmin=0, zmax=100, text=np.where(Z > 0, np.round(Z).astype(int).astype(str), ""), texttemplate="%{text}",
                                   showscale=False, hovertemplate="%{y} · %{x}: %{z:.0f}<extra></extra>"))
        fig.update_yaxes(autorange="reversed"); fig.update_xaxes(type="category")
        st.plotly_chart(ui.base_fig(fig, 260), key=f"fr_heat_{ent}")

    st.markdown("#### Ordinary explanations already considered")
    g = T.guards(rr)
    if g:
        for x in g: st.markdown(f"- {x}")
    else:
        st.markdown("None of the built-in explanations applies: not processor-like, not an industry where pass-through is normal, "
                    "and transfers to its own accounts are already left out.")

    st.markdown("#### Payment profile, last 12 months")
    p1, p2 = st.columns(2)
    with p1:
        fig = go.Figure()
        for c, nm, col in (("share_WIRE_out", "% of outgoing money by wire", ui.BLUE), ("top1_share_out", "% to the largest recipient", ui.VIOLET),
                           ("new_amt_share_in", "% of incoming money from new payers", ui.RED)):
            if c in h: fig.add_scatter(x=h.month, y=h[c] * 100, name=nm, mode="lines+markers", line=dict(width=2, color=col))
        fig.update_yaxes(range=[0, 100], title_text="%")
        st.plotly_chart(ui.base_fig(fig, 300, "How the money moves"), key=f"fr_prof_{ent}")
    with p2:
        fig = go.Figure()
        for c, nm, col in (("ncpty_all_in", "payers", ui.TEAL), ("ncpty_all_out", "payees", ui.ORANGE), ("n_fi_out", "banks paid", ui.RED)):
            if c in h: fig.add_scatter(x=h.month, y=h[c], name=nm, mode="lines+markers", line=dict(color=col, width=2))
        fig.update_yaxes(type="log" if h.get("ncpty_all_out", pd.Series([0])).max() > 500 else "linear")
        st.plotly_chart(ui.base_fig(fig, 300, "Payers, payees and banks"), key=f"fr_cnt_{ent}")

    st.markdown("#### Largest counterparties (3 months)")
    if st.toggle("Load counterparties", key=f"fr_cp_{ent}", help="Reads payment detail. The Payment network tab shows the same as a map."):
        a = network.counterparties([ent], months[-1], 3)
        if len(a):
            fl = network.flags()
            t = a.assign(flags=[", ".join(fl.get(k, [])) if k else "" for k in a.key]).head(25)[["label", "type", "bank", "amt_in", "amt_out", "flags"]]
            st.dataframe(t, hide_index=True, width="stretch",
                         column_config={"label": "counterparty", "amt_in": st.column_config.NumberColumn("money in", format="dollar"),
                                        "amt_out": st.column_config.NumberColumn("money out", format="dollar")})
        else:
            st.caption("No payments recorded in this window.")

    st.markdown("#### Case summary and review")
    nm = display_name(info.get("name"), info.get("key"))
    lines = [f"# Pattern review — {nm}", "", f"Client ref {info.get('key')} · payment entity {ent} · scored {r.month}",
             f"Tier **{r.tier}** · score {r.score:.0f} · {money(r.usd_involved)} involved over 3 months · flagged {int(r.months_flagged)} of 3 months", ""]
    for k in T.listed(rr):
        lines += [f"## {T.name(k)} (score {float(getattr(rr, f'score_{k}', 0) or 0):.0f})", T.TYPES[k]["what"], ""]
        lines += [f"- {x.replace('**', '')}" for x in T.evidence(k, rr)] + [""]
    lines += ["## Ordinary explanations considered"] + ([f"- {x}" for x in g] or ["- none applies"])
    lines += ["", "_Internal. A payment pattern is not evidence of wrongdoing. Do not discuss with the client._"]
    case = "\n".join(lines)
    b1, b2 = st.columns([2, 1])
    with b1:
        with st.expander("Preview the case summary"):
            st.markdown(case)
    with b2:
        st.download_button("Download case summary", case, file_name=f"pattern_review_{ent}_{r.month}.md", icon="📄", width="stretch")
    with st.form(f"fr_fb_{ent}", border=True):
        st.markdown("**Reviewer's note** — kept for tuning the patterns; it is not a referral and nothing is sent to the client")
        disp = st.selectbox("Disposition", FS.DISPOSITIONS)
        note = st.text_area("Notes", height=80)
        if st.form_submit_button("Save", icon="💾"):
            FS.log_review(ent, r.month, r.score, disp, note); st.toast("Saved", icon="✅")


def home_tile():
    ok, err = FS.status()
    if not ok: return None
    S, months = FS.signals(); L = FS.latest()
    return dict(month=months[-1], n=len(L), high=int((L.tier == "High").sum()), usd=L.usd_involved.sum(), new=int(L.new.sum()),
                by=[(k, int(L.typologies.str.contains(k).sum())) for k in T.ORDER], keys=set(L.key))
