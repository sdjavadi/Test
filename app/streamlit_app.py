"""Deposit attrition early warning — RM prototype.  Run:  streamlit run streamlit_app.py"""
import datetime, pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from attr_app import data as D, labels as LB, model as MD
from attr_app.config import settings

st.set_page_config(page_title="Deposit attrition early warning", page_icon="📉", layout="wide")
CFG = settings()
MODE = CFG["mode"]
TAB_CARDS, TAB_DETAIL = "🚨 Signal cards", "🔎 Customer detail"
NAVY, BLUE, TEAL, ORANGE, RED, GREY = "#0b2545", "#1d4e89", "#12a39a", "#f28c28", "#d1495b", "#8d99ae"
st.markdown("""<style>
.block-container{padding-top:1.2rem}
.card-name{font-size:1.08rem;font-weight:700;color:#0b2545;line-height:1.2}
.card-sub{color:#6b7785;font-size:.82rem;margin-bottom:.3rem}
.why li{margin:.1rem 0;font-size:.9rem}
.next{background:#f3f6fa;border-radius:8px;padding:.5rem .7rem;font-size:.88rem;margin-top:.3rem}
.play{background:#e8f6f4;border-left:4px solid #12a39a;border-radius:6px;padding:.55rem .8rem;margin:.4rem 0}
</style>""", unsafe_allow_html=True)


# ── data and model (cached) ──────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to the data…")
def get_source(mode):
    return D.DemoSource() if mode == "demo" else D.ImpalaSource(CFG)


@st.cache_resource(show_spinner="Loading the model…")
def get_bundle(mode, path):
    p = pathlib.Path(path)
    if p.exists():
        return MD.load_bundle(p)
    if mode == "demo":
        b = D.demo_bundle(get_source(mode)); b["_booster"] = MD._booster(b); return b
    raise FileNotFoundError(f"model bundle not found: {p.resolve()} — run the export cell of the model notebook")


NUM = ["bal", "bal_norm", "capped"]


@st.cache_data(ttl=6 * 3600, show_spinner="Scoring every client whose money is still here…")
def score_all(mode, months_scored):
    src, b = get_source(mode), get_bundle(mode, CFG["bundle_path"])
    months = src.months()[:months_scored]
    parts = []
    for i, m in enumerate(months):
        df = src.features(m).reset_index(drop=True)
        for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        X = MD.matrix(b, df)
        p, C = MD.predict(b, X, contrib=(i == 0))
        keep = [c for c in ["cust_pwr_id", "rltn_pwr_id", "customer_name", "naics_desc", "segment_desc", "mdm_ids", "month"] + NUM if c in df.columns]
        out = df[keep].copy(); out["p"] = p
        if i == 0:                       # reasons and fired signals for the current month
            feats = b["feats"]; top = np.argsort(-C, axis=1)[:, :5]
            out["reasons"] = [[(feats[j], float(C[r, j]), float(X[r, j])) for j in top[r] if C[r, j] > 0] for r in range(len(df))]
            fm = MD.fired_matrix(b, df)
            out["fired"] = [list(fm.columns[v]) for v in fm.to_numpy(bool)] if len(fm.columns) else [[] for _ in range(len(df))]
        parts.append(out)
    return pd.concat(parts, ignore_index=True), months


@st.cache_data(ttl=3600, show_spinner=False)
def history(mode, custs):
    h = get_source(mode).history(list(custs))
    for c in h.columns:
        if c not in ("cust_pwr_id", "rltn_pwr_id", "customer_name", "month", "naics_desc", "segment_desc"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
    return h


@st.cache_data(ttl=3600, show_spinner="Scoring this client month by month…")
def customer_detail(mode, cust):
    src, b = get_source(mode), get_bundle(mode, CFG["bundle_path"])
    df = src.customer_features(cust).sort_values("month").reset_index(drop=True)
    for c in NUM: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    X = MD.matrix(b, df); p, C = MD.predict(b, X, contrib=True)
    return df, X, p, C, MD.fired_matrix(b, df)


@st.cache_data(ttl=3600, show_spinner=False)
def relationship(mode, rltn):
    r = get_source(mode).relationship(rltn)
    for c in ("bal", "bal_norm"): r[c] = pd.to_numeric(r[c], errors="coerce")
    return r


@st.cache_data(ttl=3600, show_spinner="Reading counterparties…")
def counterparties(mode, mdm_ids, end_month):
    return get_source(mode).counterparties(list(mdm_ids), end_month)


# ── helpers ──────────────────────────────────────────────────────────
def label(b, f):
    return LB.plain(f, b.get("labels", {}).get(f))


def select_customer(cust):
    st.session_state.sel = cust
    st.session_state.main_tabs = TAB_DETAIL


def streaks(S, months, k):
    on = set(zip(S.loc[S["rank"] <= k, "cust_pwr_id"], S.loc[S["rank"] <= k, "month"]))
    def run(c):
        n = 0
        for m in months:
            if (c, m) in on: n += 1
            else: break
        return n
    return run


def card_story(b, r):
    fired = list(r.fired or []); late = [f for f in fired if f in b.get("late_feats", [])]
    early = sorted([f for f in fired if f not in late], key=lambda f: b.get("onset", {}).get(f, 0))
    reasons = [f for f, _, _ in (r.reasons or [])]
    themes = [LB.theme(f) for f in (early + reasons)]
    theme = next((t for t in themes if t != "activity"), themes[0] if themes else "activity")
    pct_norm = r.bal / r.bal_norm if r.bal_norm else np.nan
    urgent = bool(late) or (pd.notna(pct_norm) and pct_norm < 0.7)
    nxt = [f"**{r.p:.0%}** chance the money leaves within 6 months — **{LB.money(r.capped)}** at stake."]
    if urgent:
        nxt.append("Money appears to be leaving **this month** — call now.")
    elif early:
        on_ = b.get("onset", {}).get(early[0])
        if on_: nxt.append(f"Clients showing *{label(b, early[0])}* typically left within **{abs(int(on_))} months**.")
    return dict(late=late, early=early, reasons=r.reasons or [], theme=theme, urgent=urgent, pct_norm=pct_norm, next=nxt)


def spark(h):
    fig = go.Figure(go.Scatter(x=h.month, y=h.bal / 1e6, mode="lines", line=dict(color=BLUE, width=2), fill="tozeroy",
                               fillcolor="rgba(29,78,137,.08)", hovertemplate="%{x}: $%{y:,.2f}m<extra></extra>"))
    if len(h): fig.add_hline(y=float(h.bal_norm.iloc[-1]) / 1e6, line=dict(color=GREY, dash="dot", width=1))
    fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False, template="plotly_white")
    return fig


def base_fig(fig, h=320, title=None):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=40, r=20, t=40 if title else 10, b=30), title=title,
                      legend=dict(orientation="h", y=-0.2), font=dict(size=12))
    return fig


# ── load ─────────────────────────────────────────────────────────────
try:
    B = get_bundle(MODE, CFG["bundle_path"])
    S, MONTHS = score_all(MODE, int(CFG["months_scored"]))
except Exception as e:
    st.error(f"Could not load data or model ({type(e).__name__}): {e}")
    st.stop()
LATEST = MONTHS[0]

with st.sidebar:
    st.header("Warning list")
    st.caption(f"{'Demo data' if MODE == 'demo' else 'Impala · ' + CFG['dbi_pool']} · scored {LATEST} · model trained through {B.get('trained_through', '—')}")
    K = st.number_input("Clients on the list", 20, 5000, int(CFG["list_size"]) if MODE != "demo" else 120, step=20,
                        help="How many clients the team can call this month.")
    ALPHA = st.select_slider("Rank by", options=[0.0, 0.5, 1.0], value=float(CFG["alpha"]),
                             format_func=lambda a: {0.0: "likelihood", 0.5: "likelihood × size", 1.0: "dollars at risk"}[a])
    focus = st.pills("Show", ["All", "Money leaving now", "New this month", "Relationship risk", "Competitive threat"], default="All")
    seg = st.multiselect("Segment", sorted(S.segment_desc.dropna().unique()) if "segment_desc" in S else [])
    min_bal = st.select_slider("Minimum normal balance", options=[0, 1e5, 1e6, 1e7, 1e8], value=0, format_func=lambda v: LB.money(v, 0) if v else "any")
    search = st.text_input("Search name or client id")
    if st.button("Refresh data", icon="🔄"):
        st.cache_data.clear(); st.rerun()

S["rank"] = MD.rank_month(S, ALPHA)
run = streaks(S, MONTHS, K)
L = S[(S.month == LATEST) & (S["rank"] <= K)].copy()
L["months_on_list"] = L.cust_pwr_id.map(run)
L["story"] = [card_story(B, r) for r in L.itertuples()]
L["urgent"] = L.story.map(lambda s: s["urgent"]); L["theme"] = L.story.map(lambda s: s["theme"])
view = L
if focus == "Money leaving now": view = view[view.urgent]
elif focus == "New this month": view = view[view.months_on_list == 1]
elif focus == "Relationship risk": view = view[view.story.map(lambda s: any(LB.theme(f) == "relationship" for f in s["early"] + [x[0] for x in s["reasons"]]))]
elif focus == "Competitive threat": view = view[view.theme == "competitive"]
if seg: view = view[view.segment_desc.isin(seg)]
if min_bal: view = view[view.bal_norm >= min_bal]
if search: view = view[view.customer_name.fillna("").str.contains(search, case=False) | view.cust_pwr_id.str.contains(search)]
view = view.sort_values("rank")

st.title("Deposit attrition early warning")
tabs = st.tabs([TAB_CARDS, TAB_DETAIL], key="main_tabs", on_change="rerun")

# ════════════════════════════════════════════════════════════════════
# TAB 1 · SIGNAL CARDS
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Clients on the list", f"{len(L):,}", help=f"Top {K:,} this month, ranked by {'likelihood × size' if ALPHA == .5 else 'likelihood' if ALPHA == 0 else 'dollars at risk'}")
    m2.metric("Money at stake", LB.money(L.capped.sum()), help="Balance now, never more than the client's normal balance")
    m3.metric("Expected to leave", LB.money((L.p * L.capped).sum()), help="Chance × money at stake, summed")
    m4.metric("New this month", f"{int((L.months_on_list == 1).sum()):,}")
    m5.metric("Money leaving now", f"{int(L.urgent.sum()):,}")
    st.caption(f"{len(view):,} cards match the filters · most urgent and largest first within the ranking")
    per = 12; pages = max(1, int(np.ceil(len(view) / per)))
    page = st.number_input("Page", 1, pages, 1, key="page") if pages > 1 else 1
    chunk = view.iloc[(page - 1) * per: page * per]
    H = history(MODE, tuple(chunk.cust_pwr_id)) if len(chunk) else pd.DataFrame()
    cols = st.columns(3)
    for i, r in enumerate(chunk.itertuples()):
        s = r.story
        with cols[i % 3].container(border=True):
            st.markdown(f"<div class='card-name'>#{r.rank} · {r.customer_name or r.cust_pwr_id}</div>"
                        f"<div class='card-sub'>{r.segment_desc if isinstance(getattr(r, 'segment_desc', None), str) else ''} · "
                        f"{r.naics_desc if isinstance(getattr(r, 'naics_desc', None), str) else ''} · ref {r.cust_pwr_id[-8:]}</div>",
                        unsafe_allow_html=True)
            b1, b2, b3 = st.columns([1.3, 1, 1.2])
            with b1:
                st.badge("Money leaving now" if s["urgent"] else "Early warning", color="red" if s["urgent"] else "orange",
                         icon=":material/priority_high:" if s["urgent"] else ":material/schedule:")
            with b2:
                st.badge("New" if r.months_on_list == 1 else f"{r.months_on_list} mo on list", color="blue")
            with b3:
                st.badge(LB.PLAYS[s["theme"]][0], color="violet")
            c1, c2, c3 = st.columns(3)
            c1.metric("Chance (6 mo)", f"{r.p:.0%}")
            c2.metric("At stake", LB.money(r.capped))
            c3.metric("vs normal", f"{s['pct_norm']:.0%}" if pd.notna(s["pct_norm"]) else "—")
            h = H[H.cust_pwr_id == r.cust_pwr_id].tail(12) if len(H) else H
            if len(h): st.plotly_chart(spark(h), config={"displayModeBar": False}, key=f"sp_{r.cust_pwr_id}")
            why = "".join(f"<li>{'▲' if (LB.to_display(f, x) or 0) > 0 else '▼'} {label(B, f)}: <b>{LB.fmt_value(f, x)}</b></li>"
                          for f, _, x in s["reasons"][:3])
            st.markdown(f"<b>Why</b><ul class='why'>{why}</ul>", unsafe_allow_html=True)
            st.markdown("<div class='next'>" + "<br>".join(s["next"]) + "</div>", unsafe_allow_html=True)
            st.button("Prepare the call →", key=f"go_{r.cust_pwr_id}", on_click=select_customer, args=(r.cust_pwr_id,),
                      type="primary", width="stretch")
    if not len(view):
        st.info("No client matches these filters.")

# ════════════════════════════════════════════════════════════════════
# TAB 2 · CUSTOMER DETAIL
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    opts = list(L.sort_values("rank").cust_pwr_id)
    if not opts:
        st.info("The list is empty."); st.stop()
    names = dict(zip(L.cust_pwr_id, L.customer_name.fillna("")))
    ranks = dict(zip(L.cust_pwr_id, L["rank"]))
    if st.session_state.get("sel") not in opts: st.session_state.sel = opts[0]
    cust = st.selectbox("Client", opts, key="sel", format_func=lambda c: f"#{ranks[c]} · {names.get(c) or c} · ref {c[-8:]}")
    r = L[L.cust_pwr_id == cust].iloc[0]; s = r.story
    df, X, p_hist, C, FM = customer_detail(MODE, cust)
    h = history(MODE, (cust,))
    feats = B["feats"]

    st.subheader(r.customer_name or cust)
    st.caption(" · ".join(x for x in [str(getattr(r, "segment_desc", "") or ""), str(getattr(r, "naics_desc", "") or ""),
                                       f"client ref {cust}", f"relationship {r.rltn_pwr_id}" if isinstance(r.rltn_pwr_id, str) else ""] if x))
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Chance money leaves (6 mo)", f"{r.p:.0%}", delta=f"{(p_hist[-1] - p_hist[-2]) * 100:+.0f} pts vs last month" if len(p_hist) > 1 else None,
              delta_color="inverse")
    k2.metric("Rank this month", f"#{r['rank']:,}")
    k3.metric("Money at stake", LB.money(r.capped))
    k4.metric("Balance now", LB.money(r.bal), delta=f"{s['pct_norm']:.0%} of normal" if pd.notna(s["pct_norm"]) else None, delta_color="off")
    k5.metric("Normal balance", LB.money(r.bal_norm))
    k6.metric("Months on the list", f"{r.months_on_list}")

    t_name, t_text = LB.PLAYS[s["theme"]]
    st.markdown(f"<div class='play'><b>{t_name}.</b> {t_text}</div>", unsafe_allow_html=True)

    g1, g2 = st.columns([1.6, 1])
    with g1:
        fig = go.Figure()
        fig.add_bar(x=h.month, y=h.bal / 1e6, name="average balance", marker_color="rgba(29,78,137,.75)")
        if "bal_eom" in h: fig.add_scatter(x=h.month, y=h.bal_eom / 1e6, name="month-end", mode="lines+markers", line=dict(color=ORANGE))
        fig.add_scatter(x=h.month, y=h.bal_norm / 1e6, name="normal", mode="lines", line=dict(color=GREY, dash="dash"))
        onl = S[(S.cust_pwr_id == cust) & (S["rank"] <= K)].month
        for m in onl: fig.add_vrect(x0=m, x1=m, line=dict(color=RED, width=10), opacity=.15)
        fig.update_yaxes(title_text="$m")
        st.plotly_chart(base_fig(fig, 340, "Balance ($m) — shaded months: on the list"), key="bal")
    with g2:
        fig = go.Figure(go.Scatter(x=df.month, y=p_hist * 100, mode="lines+markers", line=dict(color=RED, width=3),
                                   hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
        fig.update_yaxes(title_text="%", rangemode="tozero")
        st.plotly_chart(base_fig(fig, 340, "Chance the money leaves within 6 months"), key="score")

    st.markdown("#### Why the model flagged this client")
    cc = pd.DataFrame({"f": feats, "c": C[-1], "x": X[-1]}).assign(a=lambda d: d.c.abs()).sort_values("a", ascending=False).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=cc.c, y=[f"{label(B, f)}  ({LB.fmt_value(f, x)})" for f, x in zip(cc.f, cc.x)], orientation="h",
                           marker_color=[RED if v > 0 else TEAL for v in cc.c], hovertemplate="%{y}<extra></extra>"))
    fig.update_xaxes(title_text="← lowers risk      raises risk →")
    st.plotly_chart(base_fig(fig, 380), key="why")

    st.markdown("#### Signals against clients of the same size who stayed")
    rows = []
    for f, (th, d) in B.get("thr", {}).items():
        if f not in FM.columns: continue
        fired_m = list(df.month[FM[f].to_numpy(bool)])
        cur = X[-1, feats.index(f)] if f in feats else np.nan
        rows.append({"signal": label(B, f), "now": LB.fmt_value(f, cur), "stayers' threshold": LB.fmt_value(f, th),
                     "leavers": "fall below" if d == "falls" else "rise above", "firing now": "🔴" if bool(FM[f].iloc[-1]) else "—",
                     "months fired": len(fired_m), "first fired": fired_m[0] if fired_m else "—",
                     "typical warning": f"{abs(int(B['onset'][f]))} mo" if B.get("onset", {}).get(f) else "—"})
    sig = pd.DataFrame(rows)
    if len(sig):
        st.dataframe(sig.sort_values(["firing now", "months fired"], ascending=[True, False]), hide_index=True, width="stretch")

    st.markdown("#### Payment behavior")
    pc = st.columns(3)
    panels = [("Dollars in and out ($m)", [("amt_all_in", "in", TEAL), ("amt_all_out", "out", ORANGE)], 1e6),
              ("Payments and receiving days", [("ntxn_all_in", "incoming payments", TEAL), ("active_days_in", "days with receipts", BLUE)], 1),
              ("Payers, payees and banks", [("ncpty_all_in", "payers", TEAL), ("ncpty_all_out", "payees", ORANGE), ("n_fi_out", "banks paid", RED)], 1)]
    for col, (title, series, scale) in zip(pc, panels):
        fig = go.Figure()
        for c, nm, colr in series:
            if c in h: fig.add_scatter(x=h.month, y=h[c] / scale, name=nm, mode="lines+markers", line=dict(color=colr, width=2))
        col.plotly_chart(base_fig(fig, 280, title), key=f"pay_{title}")
    if "selfpay_amt_out" in h and h.selfpay_amt_out.fillna(0).sum() > 0:
        fig = go.Figure(go.Bar(x=h.month, y=h.selfpay_amt_out / 1e6, marker_color=RED))
        st.plotly_chart(base_fig(fig, 240, "Money sent to its own accounts at other banks ($m)"), key="self")

    st.markdown("#### Other entities in the same relationship")
    if isinstance(r.rltn_pwr_id, str) and r.rltn_pwr_id:
        rel = relationship(MODE, r.rltn_pwr_id)
        oth = rel[rel.cust_pwr_id != cust]
        if len(oth):
            fig = go.Figure()
            for c_, g_ in oth.groupby("cust_pwr_id"):
                fig.add_scatter(x=g_.month, y=g_.bal / 1e6, mode="lines", name=(g_.customer_name.iloc[0] or c_)[:30])
            st.plotly_chart(base_fig(fig, 280, "Balances of other entities ($m)"), key="rel")
        else:
            st.caption("No other entities in this relationship.")

    st.markdown("#### Counterparties")
    if st.toggle("Load payers, payees and banks (reads payment detail; can take a minute)", key=f"cp_{cust}"):
        mdm = [x for x in str(getattr(r, "mdm_ids", "") or "").split(",") if x] or get_source(MODE).mdm_ids(cust)
        if not mdm:
            st.caption("No payment entities linked to this client.")
        else:
            inn, out = counterparties(MODE, tuple(mdm), LATEST)
            ms = sorted(set(inn.month) | set(out.month)); rec, pri = ms[-3:], ms[:-3]
            def change(d):
                g = d.groupby(["cpty_id", "name", "bank"], dropna=False).apply(
                    lambda x: pd.Series({"prior 3 mo": x[x.month.isin(pri)].amount.sum(), "last 3 mo": x[x.month.isin(rec)].amount.sum()}),
                    include_groups=False).reset_index()
                g["change"] = g["last 3 mo"] - g["prior 3 mo"]
                return g
            ci = change(inn)
            a1, a2 = st.columns(2)
            with a1:
                st.markdown("**Payers that stopped or cut back** (money in)")
                lost = ci[ci["prior 3 mo"] > 0].sort_values("change").head(8)
                st.dataframe(lost.assign(**{c: lost[c].map(LB.money) for c in ["prior 3 mo", "last 3 mo", "change"]})[
                             ["name", "bank", "prior 3 mo", "last 3 mo", "change"]], hide_index=True, width="stretch")
            with a2:
                st.markdown("**Where outgoing money goes, by bank**")
                ob = out.assign(own=out.is_self.astype(bool)).groupby(["bank", "own"], dropna=False).apply(
                    lambda x: pd.Series({"prior 3 mo": x[x.month.isin(pri)].amount.sum(), "last 3 mo": x[x.month.isin(rec)].amount.sum()}),
                    include_groups=False).reset_index().sort_values("last 3 mo", ascending=False).head(8)
                ob["own"] = ob.own.map({True: "own account", False: ""})
                st.dataframe(ob.assign(**{c: ob[c].map(LB.money) for c in ["prior 3 mo", "last 3 mo"]}), hide_index=True, width="stretch")

    st.markdown("#### Call brief")
    reasons_txt = "\n".join(f"- {label(B, f)}: {LB.fmt_value(f, x)}" for f, _, x in s["reasons"][:5])
    fired_txt = "\n".join(f"- {label(B, f)}" for f in s["early"] + s["late"]) or "- none beyond the stayers' threshold"
    brief = (f"# Call brief — {r.customer_name or cust}\n\nClient ref {cust} · scored {LATEST} · rank #{r['rank']}\n\n"
             f"- Chance the money leaves within 6 months: {r.p:.0%}\n- Money at stake: {LB.money(r.capped)} "
             f"(balance {LB.money(r.bal)}, normal {LB.money(r.bal_norm)})\n- Months on the list: {r.months_on_list}\n\n"
             f"## Why the model flagged it\n{reasons_txt}\n\n## Signals firing\n{fired_txt}\n\n## Suggested conversation\n**{t_name}.** {t_text}\n")
    b1, b2 = st.columns([2, 1])
    with b1:
        with st.expander("Preview the brief", expanded=False):
            st.markdown(brief)
    with b2:
        st.download_button("Download call brief", brief, file_name=f"call_brief_{cust[-8:]}_{LATEST}.md", icon="📄", width="stretch")

    with st.form(f"fb_{cust}", border=True):
        st.markdown("**After the call** — helps measure what the list is worth")
        useful = st.feedback("thumbs", key=f"thumb_{cust}")
        outcome = st.selectbox("Outcome", ["Not called yet", "Client staying", "Client moving part of the money", "Client leaving anyway",
                                           "Already known to the RM", "Could not reach", "Not relevant"])
        note = st.text_area("Notes", height=80)
        if st.form_submit_button("Save", icon="💾"):
            row = pd.DataFrame([dict(saved=datetime.datetime.now().isoformat(timespec="seconds"), cust_pwr_id=cust, month=LATEST,
                                     p=round(float(r.p), 4), rank=int(r["rank"]), useful={0: "no", 1: "yes"}.get(useful, ""),
                                     outcome=outcome, note=note)])
            fp = pathlib.Path(CFG["feedback_path"])
            row.to_csv(fp, mode="a", header=not fp.exists(), index=False)
            st.toast("Saved", icon="✅")
