"""
TPPP Prospect Finder: outside companies (counterparties) that move money like third-party payment and payroll
processors, ranked by a model that learns from examples, with a review loop.

  main page     settings, headline numbers, filters, then List / Map / Explore / Examples / How the score works /
                Reviews / Export and research
  counterparty  TPPP view (score, value, type, why, similar examples, evidence, outside research, review, next to
                review) + the shared network, clients, accounts and month-by-month tabs (cpty_view.py)
Run:  streamlit run app.py      (Impala through dbi; scikit-learn; logic in tppp_core.py)
"""
from __future__ import annotations

import html
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import cpty_view as cv
import prospect_core as pc
import tppp_core as tc

st.set_page_config(page_title="TPPP Prospect Finder", layout="wide", initial_sidebar_state="collapsed")
logging.basicConfig(level=logging.INFO)
cv.inject_css()
table, chips, fail, WIDE, SELECTABLE = cv.table, cv.chips, cv.fail, cv.WIDE, cv.SELECTABLE
TYPE_COLORS = {tc.PAYROLL: "#E35205", tc.CARD: "#2F6DB5", tc.BILLPAY: "#2E8B57", tc.PROPERTY: "#8E6CC0",
               tc.OTHER: "#7B8794", "–": "#B8C1CC"}
TYPE_MIN_SCORE = 80          # "looks like" is shown only for counterparties at least this processor-like
NEW_NOW, NEW_BEFORE = 95, 80  # newly processor-like: score >= 95 now, below 80 (or not scored) before

# ------------------------------------------------------------------ state (survives a visit to a counterparty)
DEFAULTS = {"t_start": pc.DEFAULT_START, "t_end": pc.DEFAULT_END, "t_src": list(tc.SOURCES), "t_size": True,
            "t_prev": False, "t_min_score": 95.0, "t_min_value": 0.0, "t_pnc": "Any", "t_known": "Show",
            "t_hide_no": True, "t_banks": [], "t_types": [], "t_research": "Any", "t_change": "Any",
            "t_min_clients": float(tc.MIN_CLIENTS), "t_name": "", "t_top": 250, "t_switch": False,
            "x_axis": "clients_in", "y_axis": "clients_out", "x_log": True, "y_log": True}
for k, v in DEFAULTS.items():
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]
    else:
        st.session_state[k] = list(v) if isinstance(v, list) else v
st.session_state.setdefault("nav", 0)


def current_cpty():
    v = st.query_params.get("cpty") if cv.HAS_QP else st.session_state.get("cpty")
    return v or None


def go_to(eid):
    st.session_state["nav"] += 1
    st.session_state["cpty"] = eid
    if cv.HAS_QP:
        if eid:
            st.query_params["cpty"] = eid
        elif "cpty" in st.query_params:
            del st.query_params["cpty"]
    st.rerun()


def pick_row(ev, V):
    rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
    if rows and rows[0] < len(V):
        go_to(V.ent_id.iloc[rows[0]])


def pick_point(ev):
    pts = list(getattr(getattr(ev, "selection", None), "points", None) or [])
    if pts:
        cd = pts[0].get("customdata")
        if cd is not None and len(cd):
            go_to(cd[0] if isinstance(cd, (list, tuple)) else cd)


# ------------------------------------------------------------------ cached work (read-only results)
@st.cache_resource(ttl=cv.TTL, show_spinner=False)
def load_registered() -> pd.DataFrame:
    pw = pc.run_sql(tc.tppp_pwr_sql(), "registered TPPP power ids")
    pwrs = sorted({p for p in map(tc.norm_pwr, pw.pwr) if p})
    if not pwrs:
        raise RuntimeError("no registered TPPPs in the latest vector_customer snapshot")
    return pc.run_sql(tc.tppp_customers_sql(pwrs), "registered TPPP customers")


@st.cache_resource(ttl=cv.TTL, show_spinner=False)
def load_universe(per: tuple) -> pd.DataFrame:
    cols = cv.cols_of(pc.T_CPTY_MET)
    if cols is None:
        raise RuntimeError("counterparty metrics unavailable")
    raw = pc.run_sql(tc.tppp_universe_sql(per, cols, cv.cols_of(pc.T_ENT_DIM)), "TPPP universe")
    ev = None
    ecols = cv.cols_of(tc.T_EVIDENCE)                       # optional table: None when it does not exist
    sql = tc.evidence_sql(per, ecols) if ecols else None
    if sql:
        try:
            ev = pc.run_sql(sql, "TPPP evidence")
        except Exception:
            logging.getLogger("tppp").warning("evidence table unreadable; scoring without it", exc_info=True)
    return tc.prepare_universe(raw, load_registered(), ev)


@st.cache_resource(ttl=cv.TTL, show_spinner=False, max_entries=8)
def load_model(per: tuple, sources: tuple, use_size: bool, verdict_sig: tuple) -> dict:
    V = pd.DataFrame(list(verdict_sig), columns=["ent_id", "verdict", "at"]) if verdict_sig else \
        pd.DataFrame(columns=["ent_id", "verdict", "at"])
    return tc.run_model(load_universe(per), V, sources, use_size)


def verdict_sig(V: pd.DataFrame) -> tuple:
    L = tc.latest_verdicts(V)
    L = L[L.verdict.isin(["processor", "not a processor", "clear"])] if len(L) else L
    return tuple(sorted(zip(L.ent_id, L.verdict, L["at"]))) if len(L) else ()


# ------------------------------------------------------------------ shared pieces
def search_bar(where, U: pd.DataFrame, S: pd.DataFrame, per) -> None:
    nav = st.session_state.nav
    q = where.text_input("Find a counterparty by name", key=f"q_{nav}", placeholder="Type part of its name",
                         help=f"Searches every counterparty with at least {tc.MIN_CLIENTS} PNC clients a month in "
                              f"{per[0]} to {per[-1]}, scored or not, whatever the filters.")
    if not q.strip():
        return
    R = pc.search_names(U, q)
    if not len(R):
        st.info(f'No counterparty name contains "{q.strip()}" in {per[0]} to {per[-1]}.')
        return
    sc = dict(zip(S.ent_id, S.score))
    V = pd.DataFrame({"counterparty": R.ent_name, "TPPP score": R.ent_id.map(sc).round(1),
                      "bank today": R.fi_dom, "PNC clients / mo": R.clients.round(1),
                      "$K in / mo": (R.amt_in / 1e3).round(0), "$K out / mo": (R.amt_out / 1e3).round(0),
                      "ent_id": R.ent_id})
    st.markdown(f'**Matches for "{html.escape(q.strip())}"** ({len(V)} shown)')
    if SELECTABLE:
        ev = table(V, one_dec=["PNC clients / mo", "TPPP score"], height=min(38 + 35 * len(V), 320),
                   on_select="rerun", selection_mode="single-row", key=f"search_{nav}",
                   column_config={"ent_id": None})
        st.caption("Tick a row to open the counterparty. An empty score means it was not scored (see its page).")
        pick_row(ev, V)
    else:
        names = dict(zip(V.ent_id, V.counterparty))
        pick = st.selectbox("Open", [""] + list(V.ent_id), key=f"spick_{nav}",
                            format_func=lambda e: "Choose …" if not e else names[e])
        if pick:
            go_to(pick)
    st.divider()


def why_not_scored(r: pd.Series) -> str:
    if r.excluded:
        return "its name is on the exclusion list (government, card networks, insurers, PNC)"
    if r.generic:
        return "its name is a generic payment descriptor"
    return f"less than {100 * tc.MIN_ORG_SHARE:.0f}% of its money is with organisations on either side"


def research_text(row) -> str:
    if row is None or not isinstance(row.get("answer"), str) or not row.get("answer"):
        return ""
    c = f" ({row['confidence']}%)" if str(row.get("confidence", "")).strip() else ""
    return f"{row['answer']}{c}"


def compare_maps(per, sources, use_size, vsig):
    """Score in the previous period of the same length, when the comparison is switched on."""
    if not st.session_state.t_prev:
        return None, None
    prev = tc.previous_period(per)
    if prev is None:
        return None, None
    try:
        with st.spinner(f"Scoring the previous period {prev[0]} to {prev[-1]} for the comparison ..."):
            R0 = load_model(prev, sources, use_size, vsig)
    except Exception as e:
        logging.getLogger("tppp").warning("previous period not scored", exc_info=True)
        st.warning(f"The previous period ({prev[0]} to {prev[-1]}) could not be scored, so there is no comparison"
                   + (f": {e}" if isinstance(e, ValueError) else "."))
        return None, None
    return prev, dict(zip(R0["S"].ent_id, R0["S"].score))


def change_label(now, before) -> str:
    if before is None or pd.isna(before):
        return "new (not scored before)" if now >= NEW_NOW else ""
    if now >= NEW_NOW and before < NEW_BEFORE:
        return "newly processor-like"
    if now >= NEW_NOW and before >= NEW_NOW:
        return "consistently high"
    return ""


# ------------------------------------------------------------------ filters (shared by List, Map and the queue)
def filtered(S: pd.DataFrame, research: dict, prev_map: dict | None) -> pd.DataFrame:
    s = st.session_state
    P = S[~S.registered]
    m = (P.score >= s.t_min_score) & (P.value >= s.t_min_value) & (P.clients >= s.t_min_clients)
    if s.t_pnc != "Any":
        m &= P.pnc_account == s.t_pnc
    if s.t_known == "Hide":
        m &= ~P.known_proc
    elif s.t_known == "Only":
        m &= P.known_proc
    if s.t_hide_no:
        m &= P.verdict != "not a processor"
    if s.t_banks:
        m &= P.fi_dom.isin(s.t_banks)
    if s.t_types:
        m &= P.ptype.isin(s.t_types) & (P.score >= TYPE_MIN_SCORE)
    if s.t_name.strip():
        m &= P.ent_name.str.contains(s.t_name.strip().upper(), regex=False)
    if s.t_switch and "switch_n" in P:
        m &= P.switch_n.fillna(0) > 0
    if s.t_research != "Any":
        ans = P.ent_id.map(lambda e: (research.get(e) or {}).get("answer", ""))
        m &= {"Says processor": ans == "processor", "Says not a processor": ans == "not a processor",
              "No research yet": ans == ""}[s.t_research]
    D = P[m]
    if prev_map is not None and s.t_change != "Any":
        lab = [change_label(a, prev_map.get(e)) for a, e in zip(D.score, D.ent_id)]
        want = {"Newly processor-like": ("newly processor-like", "new (not scored before)"),
                "Consistently high": ("consistently high",)}[s.t_change]
        D = D[[x in want for x in lab]]
    return D.sort_values(["score", "amt"], ascending=False)


def filters_panel(S: pd.DataFrame, prev_on: bool):
    with st.expander("Filters", expanded=True):
        a, b, c, d = st.columns(4)
        a.slider("TPPP score, at least", 0.0, 100.0, step=0.5, key="t_min_score",
                 help="How processor-like, as a percentile: 99 = more processor-like than 99% of the counterparties.")
        a.slider("Value, at least", 0.0, 100.0, step=1.0, key="t_min_value",
                 help="How much business it does with PNC clients, as a percentile: the average of its percentiles "
                      "for money with PNC clients a month and TM clients a month.")
        a.radio("Has an account at PNC", ["Any", "Yes", "No"], key="t_pnc", horizontal=True,
                help="Yes = its cleaned name matches a PNC client (any client, TM or not).")
        b.multiselect("Looks like", tc.TYPES, key="t_types",
                      help=f"The type of processor most of its {tc.K_NEAR} nearest examples are (by brand, registered "
                           f"TPPP name or industry code). Only for TPPP scores of {TYPE_MIN_SCORE}+.")
        b.radio("Known processors", ["Show", "Hide", "Only"], key="t_known", horizontal=True,
                help="Names on the known-processor seed list (ADP, Paychex, Stripe, Fiserv …).")
        b.checkbox("Hide counterparties marked 'not a processor'", key="t_hide_no")
        banks = S.fi_dom.fillna("(unknown)").value_counts()
        st.session_state.t_banks = [x for x in st.session_state.t_banks if x in banks.index[:300]]
        c.multiselect("Bank today", list(banks.index[:300]), key="t_banks",
                      format_func=lambda x: f"{x} ({banks[x]:,})")
        c.number_input("PNC clients a month, at least", min_value=float(tc.MIN_CLIENTS), step=1.0,
                       key="t_min_clients")
        c.selectbox("Outside research", ["Any", "Says processor", "Says not a processor", "No research yet"],
                    key="t_research", help="Results imported under 'Export and research'.")
        d.text_input("Name contains", key="t_name")
        if "switch_n" in S and S.switch_n.notna().any():
            d.checkbox("Only those that took clients from a registered TPPP", key="t_switch",
                       help="Organisations started paying it right after they stopped paying a registered TPPP "
                            "(evidence table). Shown to reviewers; not used by the score.")
        if prev_on:
            d.selectbox("Compared with the previous period", ["Any", "Newly processor-like", "Consistently high"],
                        key="t_change", help=f"Newly: TPPP score {NEW_NOW}+ now and below {NEW_BEFORE} (or not "
                                             f"scored) in the previous period. Consistently: {NEW_NOW}+ in both.")
        d.selectbox("Show top", [100, 250, 500, 1000], key="t_top")


def settings_panel():
    months = pc.month_range(pc.FIRST_MONTH, pc.LAST_MONTH)
    with st.expander("Settings: period, examples, size, comparison", expanded=False):
        s1, s2, s3, s4 = st.columns([0.8, 0.8, 2, 1.4])
        s1.selectbox("Period starts", months[::-1], key="t_start")
        s2.selectbox("Period ends", months[::-1], key="t_end",
                     help="Amounts jump in Jul-Aug 2026 in the source data (open check).")
        s3.multiselect("Examples come from", list(tc.SOURCES), key="t_src", format_func=lambda k: tc.SOURCES[k])
        s4.checkbox("Let size count", key="t_size",
                    help="On: big counterparties (many clients, much money) can score higher, as processors are "
                         "usually big. Off: the score uses only how it behaves, so small processors rank as well.")
        s4.checkbox("Compare with the previous period", key="t_prev",
                    help="Also scores the period of the same length just before this one, to flag counterparties "
                         "that became processor-like. The first load of that period takes as long as this one's.")
        st.caption("Changing these recomputes the ranking (a few seconds; the first load of a new period takes a "
                   "minute or two).")


# ------------------------------------------------------------------ main page
def prospects_view(T: pd.DataFrame, research: dict, prev_map: dict | None) -> pd.DataFrame:
    v = pd.DataFrame({
        "rank": T["rank"].to_numpy(), "counterparty": T.ent_name.to_numpy(), "TPPP score": T.score.round(1).to_numpy(),
        "value": T.value.round(0).to_numpy(),
        "looks like": np.where(T.score >= TYPE_MIN_SCORE, T.ptype, "–"),
        "why": T.why.to_numpy(), "most similar example": T.similar.to_numpy(), "review": T.verdict.to_numpy(),
        "outside research": [research_text(research.get(e)) for e in T.ent_id],
        "known processor": np.where(T.known_proc, "yes", ""), "PNC account": T.pnc_account.to_numpy(),
        "money flow": T.flow.to_numpy(), "bank today": T.fi_dom.to_numpy(), "banks": T.n_fi.to_numpy(),
        "PNC clients / mo": T.clients.round(1).to_numpy(),
        "TM clients paying it / mo": T.tm_in.round(1).to_numpy() if "tm_in" in T else np.nan,
        "TM clients it pays / mo": T.tm_out.round(1).to_numpy() if "tm_out" in T else np.nan,
        "$K in / mo": (T.amt_in / 1e3).round(0).to_numpy(), "$K out / mo": (T.amt_out / 1e3).round(0).to_numpy()})
    if "switch_n" in T and T.switch_n.notna().any():
        v.insert(8, "took clients from a TPPP", T.switch_n.fillna(0).astype(int).to_numpy())
    if prev_map is not None:
        before = [prev_map.get(e) for e in T.ent_id]
        v.insert(3, "score before", pd.Series(before, dtype="float").round(1).to_numpy())
        v.insert(4, "change", [change_label(a, b) for a, b in zip(T.score, before)])
    v["ent_id"] = T.ent_id.to_numpy()
    return v


def scatter(fig_df: pd.DataFrame, x, y, color, cmap, labels, hover, height=560, logx=False, logy=False, order=None):
    fig = px.scatter(fig_df, x=x, y=y, color=color, color_discrete_map=cmap, custom_data=["ent_id"],
                     hover_name="ent_name", hover_data=hover, labels=labels, log_x=logx, log_y=logy,
                     render_mode="webgl", template="plotly_white", height=height, opacity=0.8,
                     category_orders={color: order} if order else None)
    fig.update_traces(marker={"size": 8, "line": {"width": 0}})
    fig.update_xaxes(tickformat=",")
    fig.update_yaxes(tickformat=",")
    fig.update_layout(font_family=cv.FONT, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": 1.08, "title": None})
    return fig


def render_main(U, res, per, V, research, prev, prev_map):
    S = res["S"]
    h1, h2 = st.columns([2.3, 1])
    h1.markdown('<div class="pf-title">TPPP Prospect Finder</div><p class="pf-sub">Outside companies that move money '
                'like third-party payment and payroll processors, ranked by a model that learns from examples. '
                'Review them here; confirmed ones teach the next ranking.</p><span class="pf-note">Internal working '
                'draft · shows client names · prospecting use pending the permissible-use review</span>',
                unsafe_allow_html=True)
    search_bar(h2, U, S, per)
    settings_panel()
    rel = res["reliability"].set_index("top")
    top1 = rel.loc["top 1%", "share of hidden examples found"]
    k = st.columns(4)
    k[0].metric("Counterparties scored", f"{len(S):,}", help=f"Out of {len(U):,} with at least {tc.MIN_CLIENTS} PNC "
                "clients a month; the rest fail the screen (see a counterparty's page for why).")
    k[1].metric("Examples it learned from", f"{res['n_examples']:,}",
                help=" · ".join(f"{s}: {n}" for s, n in res["by_source"].items())
                + f" · plus {res['n_negatives']} firm negatives (marked 'not a processor')")
    k[2].metric("Hidden examples found in the top 1%", f"{100 * top1:.0f}%",
                help="Each example is scored by a model that never saw it. A random list would find 1%.")
    k[3].metric("Reviewed", f"{len(tc.latest_verdicts(V)):,}")

    filters_panel(S, prev is not None)
    D = filtered(S, research, prev_map)
    T = D.head(st.session_state.t_top)
    st.markdown(chips([f"{per[0]} to {per[-1]}",
                       f"compared with {prev[0]} to {prev[-1]}" if prev else "",
                       f"{len(D):,} counterparties match", f"showing the top {len(T):,}",
                       "evidence measures in use: " + ", ".join(tc.LABEL[f] for f in res["evidence_used"])
                       if res["evidence_used"] else "no evidence table yet (timing and client overlap)"]),
                unsafe_allow_html=True)

    tabs = st.tabs(["List", "Map", "Explore", "Examples", "How the score works", "Reviews", "Export and research"])
    t_list, t_map, t_exp, t_ex, t_how, t_rev, t_out = tabs
    nav = st.session_state.nav
    with t_list:
        Vw = prospects_view(T, research, prev_map)
        cfg = {"ent_id": None, "why": st.column_config.TextColumn(width="large"),
               "TPPP score": st.column_config.NumberColumn(help="How processor-like, 0-100 percentile, from a model "
                                                                "that did not see this counterparty"),
               "value": st.column_config.NumberColumn(help="Business with PNC clients, 0-100 percentile (money and "
                                                           "TM clients)"),
               "counterparty": st.column_config.TextColumn(width="medium")}
        one = ["TPPP score", "score before", "PNC clients / mo", "TM clients paying it / mo", "TM clients it pays / mo"]
        if SELECTABLE:
            ev = table(Vw, one_dec=one, height=560, column_config=cfg, on_select="rerun",
                       selection_mode="single-row", key=f"list_{nav}")
            st.caption("Tick a row to open the counterparty, see why it ranks there and record a review. Registered "
                       "TPPPs' own accounts at other banks are left out: they are existing clients.")
            pick_row(ev, Vw)
        else:
            table(Vw, one_dec=one, height=560, column_config=cfg)
            names = dict(zip(Vw.ent_id, Vw["rank"].astype(str) + "  " + Vw.counterparty))
            pick = st.selectbox("Open a counterparty", [""] + list(Vw.ent_id), key=f"pick_{nav}",
                                format_func=lambda e: "Choose …" if not e else f"#{names[e]}")
            if pick:
                go_to(pick)

    with t_map:
        if not len(D):
            st.info("No counterparties match the filters.")
        else:
            M = D.head(5000).assign(looks=lambda d: np.where(d.score >= TYPE_MIN_SCORE, d.ptype, "–"))
            fig = scatter(M, "score", "value", "looks", TYPE_COLORS,
                          {"score": "TPPP score (how processor-like)", "value": "Value (business with PNC clients)",
                           "looks": "looks like"},
                          {"score": ":.1f", "value": ":.0f", "fi_dom": True, "pnc_account": True, "verdict": True,
                           "looks": False}, order=tc.TYPES + ["–"])
            fig.add_hline(y=80, line_dash="dot", line_color="#9AA5B1")
            fig.add_annotation(x=1, y=1, xref="paper", yref="paper", text="processor-like and valuable",
                               showarrow=False, xanchor="right", yanchor="bottom", font={"color": "#5B6573"})
            if SELECTABLE:
                ev = st.plotly_chart(fig, **WIDE, on_select="rerun", selection_mode="points", key=f"map_{nav}")
                pick_point(ev)
            else:
                st.plotly_chart(fig, **WIDE)
            st.caption(f"The {len(M):,} counterparties matching the filters: how processor-like (right) against how "
                       "much business they do with PNC clients (up). The top-right corner is where to start. Click a "
                       "point to open it.")

    with t_exp:
        opts = ["score", "value"] + [f for f in res["feats"]] + [c for c in ("clients", "amt") if c in S]
        lab = {"score": "TPPP score", "value": "Value score", "clients": "PNC clients a month",
               "amt": "money with PNC clients a month", **tc.LABEL}
        for kx in ("x_axis", "y_axis"):
            if st.session_state[kx] not in opts:
                st.session_state[kx] = opts[0] if kx == "x_axis" else opts[1]
        e1, e2, e3, e4 = st.columns([2, 0.6, 2, 0.6])
        e1.selectbox("Across", opts, key="x_axis", format_func=lambda c: lab.get(c, c))
        e2.checkbox("log", key="x_log")
        e3.selectbox("Up", opts, key="y_axis", format_func=lambda c: lab.get(c, c))
        e4.checkbox("log", key="y_log")
        xa, ya = st.session_state.x_axis, st.session_state.y_axis
        G = S.assign(group="Other counterparties")
        G.loc[G.ent_id.isin(set(D.ent_id)), "group"] = "Prospect in your filters"
        G.loc[G.verdict == "not a processor", "group"] = "Reviewed: not a processor"
        for src, nm in (("registered", "Example: registered TPPP's account"), ("known", "Example: known processor"),
                        ("confirmed", "Example: confirmed by a reviewer")):
            G.loc[G.example == src, "group"] = nm
        rest = G[G.group == "Other counterparties"]
        G = pd.concat([rest.sample(min(len(rest), 6000), random_state=1), G[G.group != "Other counterparties"]])
        for c, lg in ((xa, "x_log"), (ya, "y_log")):
            if st.session_state[lg] and (G[c] <= 0).any():
                G = G[G[c] > 0]
        cmap = {"Other counterparties": "#D5DBE1", "Prospect in your filters": "#F2A36B",
                "Reviewed: not a processor": "#5B6573", "Example: registered TPPP's account": "#8B1E1E",
                "Example: known processor": "#E35205", "Example: confirmed by a reviewer": "#2E8B57"}
        fig = scatter(G, xa, ya, "group", cmap, {xa: lab.get(xa, xa), ya: lab.get(ya, ya)},
                      {"score": ":.1f", xa: ":,.2f", ya: ":,.2f", "group": False}, height=600,
                      logx=st.session_state.x_log, logy=st.session_state.y_log, order=list(cmap))
        if SELECTABLE:
            ev = st.plotly_chart(fig, **WIDE, on_select="rerun", selection_mode="points", key=f"explore_{nav}")
            pick_point(ev)
        else:
            st.plotly_chart(fig, **WIDE)
        st.caption("Any two measures against each other. Examples in red and green, your filtered prospects in light "
                   "orange, a sample of the other counterparties in grey. Measures are averages over the months "
                   "each counterparty was active in the period. Click a point to open it.")

    with t_ex:
        E = S[S.example != ""].sort_values("score", ascending=False)
        st.markdown(f"**{len(E):,} examples** the model learned from. Each one's score comes from a model that did "
                    "not see it, so a high score means the model would have found it.")
        Ev = pd.DataFrame({"example": E.ent_name, "why it is an example": E.example.map(tc.SOURCES),
                           "type": E.example_type,
                           "matched to": np.where(E.registered, E.tppp_match.fillna(""), E.known_brand.fillna("")),
                           "TPPP score": E.score.round(1), "rank": E["rank"], "bank today": E.fi_dom,
                           "PNC clients / mo": E.clients.round(1), "$K in / mo": (E.amt_in / 1e3).round(0),
                           "$K out / mo": (E.amt_out / 1e3).round(0), "ent_id": E.ent_id})
        if SELECTABLE:
            ev = table(Ev, one_dec=["TPPP score", "PNC clients / mo"], height=480, column_config={"ent_id": None},
                       on_select="rerun", selection_mode="single-row", key=f"ex_{nav}")
            pick_row(ev, Ev)
        else:
            table(Ev, one_dec=["TPPP score", "PNC clients / mo"], height=480, column_config={"ent_id": None})
        st.caption("Low-scoring examples are worth a look: either the name match is wrong (not really the processor) "
                   "or that processor behaves differently from the rest. Example sources can be switched off under "
                   "Settings. Types: " + " · ".join(f"{t} {n}" for t, n in res["by_type"].items()))
        rej = S[S.verdict == "not a processor"]
        if len(rej):
            st.markdown(f"**Marked 'not a processor' by reviewers ({len(rej)})**: firm negatives, never examples.")
            table(pd.DataFrame({"counterparty": rej.ent_name, "TPPP score": rej.score.round(1)}),
                  one_dec=["TPPP score"], height=min(38 + 35 * len(rej), 260))

    with t_how:
        ev_used = res["evidence_used"]
        st.markdown(f"""
**Examples.** Counterparties we're fairly sure are processors: registered TPPPs' accounts at other banks (name
match), known processor names, and counterparties reviewers marked *processor*. Everything else counts as "probably
not a processor"; the hidden processors among them are what the list is for. Counterparties marked *not a processor*
count {tc.NEG_WEIGHT:g} times as much as an ordinary one, so the model learns what a false alarm looks like.

**Behaviour.** For each counterparty, {len(res['feats'])} measures, averaged over the months it was active in
{per[0]} to {per[-1]}: who pays it and whom it pays, organisations vs individuals, days with payments, average
payment, rails, banks, retention and more{', plus ' + str(len(ev_used)) + ' evidence measures on timing and client overlap' if ev_used else ''}.
Each is turned into a percentile so dollars, counts and shares are comparable.

**Model.** A logistic regression weighs the behaviours so that examples score high. Every counterparty is scored by
a model trained without it, and examples of one brand are held out together, so an example's score is an honest
test. The **TPPP score** is a percentile (99 = more processor-like than 99% of the counterparties).

**Evidence table.** When `pkg_tppp_cpty_evidence_monthly` exists (Spark notebook `pkg_tppp_cpty_evidence.ipynb`),
timing and client-overlap measures join the behaviour. Organisations that switched to a counterparty from a registered
TPPP are shown in the list and on its page but kept out of the score: the examples rarely show it, so the model would
read it the wrong way round.

**Looks like.** The processor type most of its {tc.K_NEAR} nearest examples are: payroll and HR, card and merchant
payments, bill pay and payables, property payments, or other (from the brand, the registered TPPP's name or its
industry code). **Value** is separate from the TPPP score: the average of its percentiles for money with PNC clients
a month and TM clients a month.
""")
        r1, r2 = st.columns([1, 1.3])
        rl0 = res["reliability"]
        rl = pd.DataFrame({"review the top": rl0["top"].str.replace("top ", ""),
                           "counterparties": rl0["counterparties to review"],
                           "hidden examples found": (100 * rl0["share of hidden examples found"]).round(0)
                           .astype(int).astype(str) + "%",
                           "a random list would find": (100 * rl0["a random list would find"]).map(lambda x: f"{x:g}%")})
        r1.markdown("**How reliable it is**")
        table(rl, where=r1)
        r1.caption("If reviewers go through the top of the list, this is the share of the hidden examples they would "
                   "meet. The gap to a random list is what the model adds.")
        if ev_used:
            r1.markdown("**Evidence measures**")
            table(pd.DataFrame({"measure": [tc.LABEL[f] for f in ev_used],
                                "what it means": [tc.EVIDENCE_HELP[f] for f in ev_used]}), where=r1,
                  column_config={"what it means": st.column_config.TextColumn(width="large")})
        W = res["weights"]
        Wp = pd.concat([W.head(10), W.tail(5)]).drop_duplicates("feature")
        Wp = Wp.assign(direction=np.where(Wp.weight > 0, "more processor-like", "less processor-like"))
        fw = px.bar(Wp.iloc[::-1], x="weight", y="label", color="direction", orientation="h", template="plotly_white",
                    height=470, color_discrete_map={"more processor-like": cv.ORANGE, "less processor-like": cv.BLUE})
        fw.update_layout(font_family=cv.FONT, title="What the model weighs most (high value of the measure)",
                         yaxis_title="", xaxis_title="weight", legend={"orientation": "h", "y": -0.15, "title": None},
                         margin={"l": 10, "r": 10, "t": 40, "b": 10})
        r2.plotly_chart(fw, **WIDE)
        r2.caption("Orange: a high value pushes the score up; blue: pushes it down. The model learns these from the "
                   "examples; they change when examples or reviews change.")

    with t_rev:
        L = tc.latest_verdicts(V)
        if len(L):
            Lv = L.sort_values("at", ascending=False)[["at", "ent_name", "verdict", "note", "reviewer", "ent_id"]]
            if SELECTABLE:
                ev = table(Lv, height=420, column_config={"ent_id": None}, on_select="rerun",
                           selection_mode="single-row", key=f"rev_{nav}")
                pick_row(ev, Lv)
            else:
                table(Lv, height=420, column_config={"ent_id": None})
            by = L.reviewer.replace("", "(no name)").value_counts()
            st.caption("Latest review per counterparty. 'processor' adds it to the examples; 'not a processor' makes "
                       "it a firm negative and, by default, keeps it off the list. Reviews per person: "
                       + " · ".join(f"{r} {n}" for r, n in by.items()))
            st.download_button("Download every review (CSV)", V.to_csv(index=False).encode(),
                               file_name="tppp_reviews.csv", mime="text/csv")
        else:
            st.info("No reviews yet. Open a counterparty from the List to record one; its page has 'Next to review'.")

    with t_out:
        st.markdown("**The list as filtered** (every row that matches, with empty review columns).")
        out = prospects_view(D, research, prev_map).drop(columns="ent_id").assign(verdict="", note="", reviewer="",
                                                                                   reviewed_at="")
        st.download_button(f"Download {len(out):,} prospects (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"tppp_prospects_{per[-1]}.csv", mime="text/csv", on_click=pc.log_event,
                           args=(cv.viewer(), "export prospects", f"{per[0]}..{per[-1]}", len(out)))
        st.divider()
        st.markdown("**Outside research, step 1: names file.** Names only, behind a random-looking row id, for "
                    "asking an AI or a vendor whether each one is a processor. The key file joins the row ids back "
                    "and stays inside PNC. Get sign-off before any file leaves the bank.")
        names, key = tc.enrichment_files(D.head(1000))
        e1, e2, _ = st.columns([1, 1, 2])
        e1.download_button(f"Names file ({len(names):,})", names.to_csv(index=False).encode(),
                           file_name="tppp_research_names.csv", mime="text/csv", on_click=pc.log_event,
                           args=(cv.viewer(), "export research names", f"{per[0]}..{per[-1]}", len(names)))
        e2.download_button("Key file (internal)", key.to_csv(index=False).encode(),
                           file_name="tppp_research_key_INTERNAL.csv", mime="text/csv")
        st.markdown("**Step 2: import the results.** A CSV or Excel file with the `row_id` column from the names file "
                    "and an answer column (`is_tppp`, `answer` or `verdict`: yes / no / unsure). Optional columns: "
                    "`confidence` (0-1 or %), `evidence` (text or links), `type`. Results show on the list and on "
                    "each counterparty's page; they do not change the score until a reviewer confirms them.")
        up = st.file_uploader("Research results", type=["csv", "xlsx"], key=f"up_{nav}")
        if up is not None:
            try:
                raw = pd.read_excel(up) if up.name.lower().endswith(".xlsx") else pd.read_csv(up)
                rows, msg = tc.parse_research(raw, U, up.name)
            except Exception as e:
                fail("the research file", e)
                rows, msg = pd.DataFrame(), ""
            if msg:
                (st.success if len(rows) else st.warning)(msg)
            if len(rows):
                table(rows[["ent_name", "answer", "confidence", "ptype", "evidence"]].head(50), height=260)
                if st.button(f"Save {len(rows):,} results", type="primary"):
                    try:
                        tc.save_research(rows)
                        pc.log_event(cv.viewer(), "import research", up.name, len(rows))
                        st.session_state["nav"] += 1
                        st.rerun()
                    except Exception as e:
                        fail("the research results (could not write the file)", e)
        if research:
            st.caption(f"{len(research):,} counterparties have outside research on file.")


# ------------------------------------------------------------------ counterparty page
def render_cpty(eid, U, res, per, n, V, research, prev, prev_map):
    S = res["S"]
    b1, b2, _, b3 = st.columns([0.8, 1, 0.5, 1])
    if b1.button("← Back to the list"):
        go_to(None)
    queue = filtered(S, research, prev_map)
    todo = queue[(queue.verdict == "") & (queue.ent_id != eid)]
    if len(todo) and b2.button(f"Next to review → ({len(todo):,} left in your filters)",
                               help="The highest-ranked counterparty in the current filters without a review."):
        go_to(todo.ent_id.iloc[0])
    search_bar(b3, U, S, per)
    row = U[U.ent_id == eid]
    if not len(row):
        st.warning("This counterparty has fewer than 5 PNC clients a month (or no activity) in the selected period.")
        return
    r = row.iloc[0]
    srow = S[S.ent_id == eid]
    s = srow.iloc[0] if len(srow) else None
    if st.session_state.get("_logged") != eid:
        st.session_state["_logged"] = eid
        pc.log_event(cv.viewer(), "open counterparty", r.ent_name)
    rs = research.get(eid)
    extra = [f"TPPP score {s.score:.1f} · rank {s['rank']:,} of {len(S):,}" if s is not None else "Not scored",
             f"Value {s.value:.0f}" if s is not None else "",
             f"Looks like: {s.ptype}" if s is not None and s.score >= TYPE_MIN_SCORE else "",
             "Known processor" if r.known_proc else "",
             f"Registered TPPP's account: {r.tppp_match}" if r.registered else "",
             f"Example ({tc.SOURCES[s.example].lower()})" if s is not None and s.example else "",
             f"Reviewed: {s.verdict}" if s is not None and s.verdict else "",
             f"Outside research: {research_text(rs)}" if rs else ""]
    cv.render_header(r, per, n, extra_chips=extra)

    def tppp_tab():
        if s is None:
            st.info(f"Not scored: {why_not_scored(r)}.")
        else:
            c1, c2 = st.columns([1.1, 1])
            c1.markdown("**Why it ranks here**")
            parts = [p for p in str(s.why).split(" · ") if p]
            c1.markdown("\n".join(f"- {html.escape(p)}" for p in parts) if parts else "No single measure stands out.")
            c1.caption("The measures that push its score up most, with where it sits among all scored counterparties.")
            if s.score >= TYPE_MIN_SCORE:
                c1.markdown(f"**Looks like:** {s.ptype} ({s.ptype_share:.0%} of its {tc.K_NEAR} nearest examples)")
            if prev is not None:
                b = prev_map.get(eid)
                c1.markdown(f"**Previous period** ({prev[0]} to {prev[-1]}): "
                            + (f"TPPP score {b:.1f}" if b is not None else "not scored")
                            + (f" · {change_label(s.score, b)}" if change_label(s.score, b) else ""))
            c2.markdown("**Most similar examples**")
            table(tc.similar_examples(res, int(srow.index[0])), where=c2)
            c2.caption("Closest examples on the measures the model weighs most.")
            ev = [f for f in [a for a, _, _ in tc.EVIDENCE] if f in S and pd.notna(s.get(f))]
            if ev:
                c2.markdown("**Evidence**")
                pr = {f: S[f].rank(pct=True)[srow.index[0]] for f in ev}
                table(pd.DataFrame({"measure": [tc.LABEL[f] for f in ev],
                                    "value": [f"{100 * s[f]:.0f}%" for f in ev],
                                    "vs others": [f"top {max(1, round(100 * (1 - pr[f])))}%" for f in ev]}), where=c2)
                if pd.notna(s.get("switch_n")) and s.switch_n > 0:
                    c2.caption(f"{s.switch_n:.0f} organisations started paying it right after they stopped paying a "
                               "registered TPPP.")
        if rs:
            st.markdown("**Outside research**")
            st.markdown(f"{html.escape(research_text(rs))}" + (f" · type: {html.escape(rs['ptype'])}"
                                                               if rs.get("ptype") else "")
                        + f" · from {html.escape(rs.get('source', ''))} ({rs.get('at', '')[:10]})")
            if rs.get("evidence"):
                st.caption(rs["evidence"])
        st.divider()
        st.markdown("**Review**")
        hist = V[V.ent_id == eid].sort_values("at", ascending=False) if len(V) else V
        with st.form(f"review_{eid}"):
            f1, f2, f3 = st.columns([1.2, 2, 1])
            cur = s.verdict if s is not None and s.verdict in tc.VERDICTS else "unsure"
            verdict = f1.radio("Verdict", tc.VERDICTS, index=tc.VERDICTS.index(cur))
            note = f2.text_area("Note", placeholder="What you checked (website, KYC, name …)", height=90)
            reviewer = f3.text_input("Reviewer", value=cv.viewer() if cv.viewer() != "unknown" else "")
            ok = st.form_submit_button("Save review", type="primary")
            ok_next = st.form_submit_button("Save and open the next to review") if len(todo) else False
        if ok or ok_next:
            if not reviewer.strip():
                st.error("Please enter the reviewer's name.")
            else:
                try:
                    tc.save_verdict(eid, r.ent_name, verdict, note, reviewer.strip())
                    pc.log_event(cv.viewer(), "review", f"{r.ent_name}: {verdict}")
                    if ok_next:
                        go_to(todo.ent_id.iloc[0])
                    st.rerun()
                except Exception as e:
                    fail("the review (could not write the review file)", e)
        if len(hist):
            st.caption("Earlier reviews")
            table(hist[["at", "verdict", "note", "reviewer"]], height=min(38 + 35 * len(hist), 220))

    cv.render_detail_tabs(eid, r, per, n, first_tabs=[("TPPP view", tppp_tab)])


# ------------------------------------------------------------------ data and routing
PER = pc.period(st.session_state.t_start, st.session_state.t_end)
N_MONTHS = len(PER)
SRC, SIZE = tuple(sorted(st.session_state.t_src)), bool(st.session_state.t_size)
try:
    with st.spinner(f"Loading and scoring counterparties for {PER[0]} to {PER[-1]}. The first load of a period takes "
                    "a minute or two."):
        UNIV = load_universe(PER)
        VERD = tc.load_verdicts()
        VSIG = verdict_sig(VERD)
        RES = load_model(PER, SRC, SIZE, VSIG)
    PREV, PREV_MAP = compare_maps(PER, SRC, SIZE, VSIG)
except Exception as e:
    fail("the counterparties or the model", e)
    if isinstance(e, ValueError):
        st.info(str(e))
    st.stop()
RESEARCH = {r.ent_id: r._asdict() for r in tc.load_research().itertuples(index=False)}

eid = current_cpty()
if eid:
    render_cpty(eid, UNIV, RES, PER, N_MONTHS, VERD, RESEARCH, PREV, PREV_MAP)
else:
    render_main(UNIV, RES, PER, VERD, RESEARCH, PREV, PREV_MAP)
