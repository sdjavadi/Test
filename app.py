"""
TPPP Prospect Finder: outside companies (counterparties) that move money like third-party payment and payroll
processors, ranked by a model that learns from examples, with a review loop.

  main page     settings, headline numbers, Prospects / Examples / How the score works / Reviews / Export
  counterparty  TPPP view (score, why, similar examples, verdict) + the shared network, clients, accounts and
                month-by-month tabs (cpty_view.py)
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

# ------------------------------------------------------------------ state (survives a visit to a counterparty)
DEFAULTS = {"t_start": pc.DEFAULT_START, "t_end": pc.DEFAULT_END, "t_src": list(tc.SOURCES), "t_size": True,
            "t_min_score": 95.0, "t_pnc": "Any", "t_known": "Show", "t_hide_no": True, "t_banks": [],
            "t_min_clients": float(tc.MIN_CLIENTS), "t_name": "", "t_top": 250}
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
    return tc.prepare_universe(raw, load_registered())


@st.cache_resource(ttl=cv.TTL, show_spinner=False, max_entries=8)
def load_model(per: tuple, sources: tuple, use_size: bool, verdict_sig: tuple) -> dict:
    V = pd.DataFrame(list(verdict_sig), columns=["ent_id", "verdict", "at"]) if verdict_sig else \
        pd.DataFrame(columns=["ent_id", "verdict", "at"])
    return tc.run_model(load_universe(per), V, sources, use_size)


def verdicts_now() -> pd.DataFrame:
    return tc.load_verdicts()


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
    if r.registered:
        return f"its name matches the registered TPPP {r.tppp_match}: it is an existing client's account"
    if r.excluded:
        return "its name is on the exclusion list (government, card networks, insurers, PNC)"
    if r.generic:
        return "its name is a generic payment descriptor"
    return f"less than {100 * tc.MIN_ORG_SHARE:.0f}% of its money is with organisations on either side"


def settings_panel():
    months = pc.month_range(pc.FIRST_MONTH, pc.LAST_MONTH)
    with st.expander("Settings: period, examples, size", expanded=False):
        s1, s2, s3, s4 = st.columns([0.8, 0.8, 2, 1.2])
        s1.selectbox("Period starts", months[::-1], key="t_start")
        s2.selectbox("Period ends", months[::-1], key="t_end",
                     help="Amounts jump in Jul-Aug 2026 in the source data (open check).")
        s3.multiselect("Examples come from", list(tc.SOURCES), key="t_src", format_func=lambda k: tc.SOURCES[k])
        s4.checkbox("Let size count", key="t_size",
                    help="On: big counterparties (many clients, much money) can score higher, as processors are "
                         "usually big. Off: the score uses only how it behaves, so small processors rank as well.")
        st.caption("Changing these recomputes the ranking (a few seconds; the first load of a new period takes a "
                   "minute or two).")


# ------------------------------------------------------------------ main page
def prospects_view(T: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "rank": T["rank"].to_numpy(), "counterparty": T.ent_name.to_numpy(), "TPPP score": T.score.round(1).to_numpy(),
        "why": T.why.to_numpy(), "most similar example": T.similar.to_numpy(), "review": T.verdict.to_numpy(),
        "known processor": np.where(T.known_proc, "yes", ""), "PNC account": T.pnc_account.to_numpy(),
        "money flow": T.flow.to_numpy(), "bank today": T.fi_dom.to_numpy(), "banks": T.n_fi.to_numpy(),
        "PNC clients / mo": T.clients.round(1).to_numpy(),
        "TM clients paying it / mo": T.tm_in.round(1).to_numpy() if "tm_in" in T else np.nan,
        "TM clients it pays / mo": T.tm_out.round(1).to_numpy() if "tm_out" in T else np.nan,
        "$K in / mo": (T.amt_in / 1e3).round(0).to_numpy(), "$K out / mo": (T.amt_out / 1e3).round(0).to_numpy(),
        "ent_id": T.ent_id.to_numpy()})


def render_main(U, res, per, V):
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
                help=" · ".join(f"{s}: {n}" for s, n in res["by_source"].items()))
    k[2].metric("Hidden examples found in the top 1%", f"{100 * top1:.0f}%",
                help="Each example is scored by a model that never saw it. A random list would find 1%.")
    k[3].metric("Reviewed", f"{len(tc.latest_verdicts(V)):,}")

    t_pro, t_ex, t_how, t_rev, t_exp = st.tabs(["Prospects", "Examples", "How the score works", "Reviews", "Export"])
    nav = st.session_state.nav
    P = S[~S.registered]
    with t_pro:
        a, b, c, d = st.columns(4)
        a.slider("TPPP score, at least", 0.0, 100.0, step=0.5, key="t_min_score",
                 help="The score is a percentile: 99 = more processor-like than 99% of the counterparties.")
        a.radio("Has an account at PNC", ["Any", "Yes", "No"], key="t_pnc", horizontal=True,
                help="Yes = its cleaned name matches a PNC client (any client, TM or not).")
        b.radio("Known processors", ["Show", "Hide", "Only"], key="t_known", horizontal=True,
                help="Names on the known-processor seed list (ADP, Paychex, Stripe, Fiserv …).")
        b.checkbox("Hide counterparties marked 'not a processor'", key="t_hide_no")
        banks = P.fi_dom.fillna("(unknown)").value_counts()
        st.session_state.t_banks = [x for x in st.session_state.t_banks if x in banks.index[:300]]
        c.multiselect("Bank today", list(banks.index[:300]), key="t_banks",
                      format_func=lambda x: f"{x} ({banks[x]:,})")
        c.number_input("PNC clients a month, at least", min_value=float(tc.MIN_CLIENTS), step=1.0,
                       key="t_min_clients")
        d.text_input("Name contains", key="t_name")
        d.selectbox("Show top", [100, 250, 500, 1000], key="t_top")
        m = (P.score >= st.session_state.t_min_score) & (P.clients >= st.session_state.t_min_clients)
        if st.session_state.t_pnc != "Any":
            m &= P.pnc_account == st.session_state.t_pnc
        if st.session_state.t_known == "Hide":
            m &= ~P.known_proc
        elif st.session_state.t_known == "Only":
            m &= P.known_proc
        if st.session_state.t_hide_no:
            m &= P.verdict != "not a processor"
        if st.session_state.t_banks:
            m &= P.fi_dom.isin(st.session_state.t_banks)
        if st.session_state.t_name.strip():
            m &= P.ent_name.str.contains(st.session_state.t_name.strip().upper(), regex=False)
        D = P[m].sort_values(["score", "amt"], ascending=False)
        T = D.head(st.session_state.t_top)
        Vw = prospects_view(T)
        st.caption(f"{len(D):,} counterparties match; showing the top {len(T):,}. Registered TPPPs' own accounts at "
                   "other banks are left out: they are existing clients (they are examples, see the Examples tab).")
        cfg = {"ent_id": None, "why": st.column_config.TextColumn(width="large"),
               "TPPP score": st.column_config.NumberColumn(help="Percentile, 0-100; from a model that did not see "
                                                                "this counterparty"),
               "counterparty": st.column_config.TextColumn(width="medium")}
        one = ["TPPP score", "PNC clients / mo", "TM clients paying it / mo", "TM clients it pays / mo"]
        if SELECTABLE:
            ev = table(Vw, one_dec=one, height=560, column_config=cfg, on_select="rerun",
                       selection_mode="single-row", key=f"list_{nav}")
            st.caption("Tick a row to open the counterparty, see why it ranks there and record a review.")
            pick_row(ev, Vw)
        else:
            table(Vw, one_dec=one, height=560, column_config=cfg)
            names = dict(zip(Vw.ent_id, Vw["rank"].astype(str) + "  " + Vw.counterparty))
            pick = st.selectbox("Open a counterparty", [""] + list(Vw.ent_id), key=f"pick_{nav}",
                                format_func=lambda e: "Choose …" if not e else f"#{names[e]}")
            if pick:
                go_to(pick)

    with t_ex:
        E = S[S.example != ""].sort_values("score", ascending=False)
        st.markdown(f"**{len(E):,} examples** the model learned from. Each one's score comes from a model that did "
                    "not see it, so a high score means the model would have found it.")
        Ev = pd.DataFrame({"example": E.ent_name, "why it is an example": E.example.map(tc.SOURCES),
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
                   "Settings.")
        rej = S[S.verdict == "not a processor"]
        if len(rej):
            st.markdown(f"**Marked 'not a processor' by reviewers ({len(rej)})**: never used as examples.")
            table(pd.DataFrame({"counterparty": rej.ent_name, "TPPP score": rej.score.round(1)}), one_dec=["TPPP score"],
                  height=min(38 + 35 * len(rej), 260))

    with t_how:
        st.markdown(f"""
**Examples.** Counterparties we're fairly sure are processors: registered TPPPs' accounts at other banks (name
match), known processor names, and counterparties reviewers marked *processor*. Everything else counts as "probably
not a processor"; the hidden processors among them are what the list is for.

**Behaviour.** For each counterparty, {len(res['feats'])} measures from the counterparty metrics, averaged over the
months it was active in {per[0]} to {per[-1]}: who pays it and whom it pays, organisations vs individuals, days
with payments, average payment, rails, banks, retention and more. Each is turned into a percentile so dollars,
counts and shares are comparable.

**Model.** A logistic regression weighs the behaviours so that examples score high. Every counterparty is scored by
a model trained without it, and examples of one brand are held out together, so an example's score is an honest
test. The score shown is a percentile (99 = more processor-like than 99% of the counterparties).
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
            st.download_button("Download every review (CSV)", V.to_csv(index=False).encode(),
                               file_name="tppp_reviews.csv", mime="text/csv")
            st.caption("Latest review per counterparty. 'processor' adds it to the examples; 'not a processor' keeps "
                       "it out of them and, by default, off the list.")
        else:
            st.info("No reviews yet. Open a counterparty from the Prospects tab to record one.")

    with t_exp:
        st.markdown("**The prospects list as filtered** (all rows that match, with empty review columns).")
        out = prospects_view(D).drop(columns="ent_id").assign(verdict="", note="", reviewer="", reviewed_at="")
        st.download_button(f"Download {len(out):,} prospects (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"tppp_prospects_{per[-1]}.csv", mime="text/csv", on_click=pc.log_event,
                           args=(cv.viewer(), "export prospects", f"{per[0]}..{per[-1]}", len(out)))
        st.markdown("**For outside research** (for example asking an AI or a vendor whether each is a processor): "
                    "names only, behind a random-looking row id. The key file joins the row ids back and stays "
                    "inside PNC. Get sign-off before any file leaves the bank.")
        names, key = tc.enrichment_files(D.head(1000), salt=f"tppp|{per[-1]}")
        e1, e2, _ = st.columns([1, 1, 2])
        e1.download_button("Names file", names.to_csv(index=False).encode(), file_name="tppp_enrich_names.csv",
                           mime="text/csv", on_click=pc.log_event,
                           args=(cv.viewer(), "export enrichment names", f"{per[0]}..{per[-1]}", len(names)))
        e2.download_button("Key file (internal)", key.to_csv(index=False).encode(),
                           file_name="tppp_enrich_key_INTERNAL.csv", mime="text/csv")


# ------------------------------------------------------------------ counterparty page
def render_cpty(eid, U, res, per, n, V):
    S = res["S"]
    b1, _, b3 = st.columns([1, 1.3, 1])
    if b1.button("← Back to the list"):
        go_to(None)
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
    extra = [f"TPPP score {s.score:.1f} · rank {s['rank']:,} of {len(S):,}" if s is not None else "Not scored",
             "Known processor" if r.known_proc else "",
             f"Registered TPPP's account: {r.tppp_match}" if r.registered else "",
             f"Example ({tc.SOURCES[s.example].lower()})" if s is not None and s.example else "",
             f"Reviewed: {s.verdict}" if s is not None and s.verdict else ""]
    cv.render_header(r, per, n, extra_chips=extra)

    def tppp_tab():
        if s is None:
            st.info(f"Not scored: {why_not_scored(r)}.")
        else:
            c1, c2 = st.columns([1.2, 1])
            c1.markdown("**Why it ranks here**")
            parts = [p for p in str(s.why).split(" · ") if p]
            c1.markdown("\n".join(f"- {html.escape(p)}" for p in parts) if parts else "No single measure stands out.")
            c1.caption("The measures that push its score up most, with where it sits among all scored counterparties.")
            c2.markdown("**Most similar examples**")
            sim = tc.similar_examples(res, int(srow.index[0]))
            table(sim, where=c2)
            c2.caption("Closest examples on the measures the model weighs most.")
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
        if ok:
            if not reviewer.strip():
                st.error("Please enter the reviewer's name.")
            else:
                try:
                    tc.save_verdict(eid, r.ent_name, verdict, note, reviewer.strip())
                    pc.log_event(cv.viewer(), "review", f"{r.ent_name}: {verdict}")
                    st.success("Saved. 'processor' and 'not a processor' change the examples, so the ranking is "
                               "recomputed.")
                    st.rerun()
                except Exception as e:
                    fail("the review (could not write the review file)", e)
        if len(hist):
            st.caption("Earlier reviews")
            table(hist[["at", "verdict", "note", "reviewer"]], height=min(38 + 35 * len(hist), 220))

    cv.render_detail_tabs(eid, r, per, n, first_tabs=[("TPPP view", tppp_tab)])


# ------------------------------------------------------------------ settings, data, routing
PER = pc.period(st.session_state.t_start, st.session_state.t_end)
N_MONTHS = len(PER)
try:
    with st.spinner(f"Loading and scoring counterparties for {PER[0]} to {PER[-1]}. The first load of a period takes "
                    "a minute or two."):
        UNIV = load_universe(PER)
        VERD = verdicts_now()
        RES = load_model(PER, tuple(sorted(st.session_state.t_src)), bool(st.session_state.t_size), verdict_sig(VERD))
except Exception as e:
    fail("the counterparties or the model", e)
    if isinstance(e, ValueError):
        st.info(str(e))
    st.stop()

eid = current_cpty()
if eid:
    render_cpty(eid, UNIV, RES, PER, N_MONTHS, VERD)
else:
    render_main(UNIV, RES, PER, VERD)
