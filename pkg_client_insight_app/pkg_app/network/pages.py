"""Payment network: an explorer page (any client) and the customer lens."""
import streamlit as st

from ..core import nav, network, state, ui


def _picker_options():
    """Clients worth exploring first: those flagged or listed in a module this viewer can see."""
    opts = {}
    if state.can("fraud"):
        from ..fraud import service as FS
        try:
            for r in FS.latest().drop_duplicates("key").head(60).itertuples():
                opts.setdefault(r.key, f"{ui.first_txt(r.customer_name, default='Entity ' + r.mdm_id)} · {r.tier} · {FS.T.name(r.primary)}")
        except Exception:
            pass
    if state.can("attrition"):
        from ..attrition import service as AS
        try:
            _, L, _ = AS.watchlist()
            for r in L.head(60).itertuples():
                opts.setdefault(r.key, f"{ui.first_txt(r.customer_name, default='Client ' + r.key)} · attrition #{r.rank}")
        except Exception:
            pass
    return opts


def explorer_page():
    ui.page_header("Payment network", "Who pays whom: follow money between our clients and their counterparties, one hop at a time.", "🕸️")
    opts = _picker_options()
    extra = st.session_state.setdefault("net_extra", {})
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    with c2:
        q = st.text_input("Find any client (name or id)", key="net_q", placeholder="e.g. Summit, 100045")
        if q:
            for k, lbl in nav.search(q)[:20]: extra[k] = lbl
    allopts = {**extra, **opts}
    if not allopts:
        ui.empty_state("Search for a client to draw its payment network."); return
    cur = st.session_state.get("net_sel")
    if cur not in allopts: st.session_state["net_sel"] = next(iter(extra)) if extra and q else next(iter(allopts))
    with c1:
        key = st.selectbox("Client", list(allopts), format_func=allopts.get, key="net_sel")
    info = nav.resolve(key)
    h1, h2 = st.columns([4, 1.3], vertical_alignment="center")
    h1.markdown(f"### {ui.first_txt(info.get('name'), default='Client ' + key)}")
    h1.caption(" · ".join(x for x in [ui.first_txt(info.get("naics_desc")), f"ref {key}", f"{len(info.get('mdm_ids') or [])} payment entit"
                                      + ("y" if len(info.get("mdm_ids") or []) == 1 else "ies")] if x))
    if h2.button("Open the client's page →", key="net_open", width="stretch"):
        nav.open_customer(key, lens="network")
    network.ego(info, key="netx")


def lens(info):
    network.ego(info, key=f"netl_{info['key']}")
