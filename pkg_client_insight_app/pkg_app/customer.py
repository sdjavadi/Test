"""The shared customer page: one header, one tab per module lens (only the modules the viewer may see)."""
import streamlit as st

from .core import nav, state, ui
from .core.fmt import display_name, money

LENS_ORDER = ["attrition", "fraud", "network"]


def _lens_fn(m):
    if m == "attrition":
        from .attrition.pages import lens
    elif m == "fraud":
        from .fraud.pages import lens
    else:
        from .network.pages import lens
    return lens


def _options():
    from .network.pages import _picker_options
    opts = dict(st.session_state.get("cust_recent", {}))
    opts.update({k: v for k, v in _picker_options().items() if k not in opts})
    return opts


def page():
    mods = [m for m in LENS_ORDER if state.can(m)]
    lens_from = None
    opts = _options()
    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    with c2:
        q = st.text_input("Find any client (name or id)", key="cust_q", placeholder="e.g. Summit, 100045")
    found = dict(nav.search(q)[:20]) if q else {}
    if q and not found: c2.caption("No client found.")
    opts = {**found, **opts}
    if st.session_state.get("nav_cust"):                         # arrived from a card, a table or the network
        key = st.session_state.pop("nav_cust"); lens_from = st.session_state.pop("nav_lens", None)
    elif q and found and q != st.session_state.get("_cust_q_last"):
        key = next(iter(found))                                  # a new search: show its best match
    else:
        key = st.session_state.get("cust_sel") or st.query_params.get("c")
        if "cust_lens" not in st.session_state: lens_from = st.query_params.get("lens")   # deep link: ?c=<client>&lens=fraud
    st.session_state["_cust_q_last"] = q
    if key and key not in opts:
        info0 = nav.resolve(key)
        opts = {key: f"{display_name(info0.get('name'), key)} · ref {key}", **opts}
    if not opts:
        ui.empty_state("Open a client from a signal card, or search for one."); return
    if not key: key = next(iter(opts))
    if st.session_state.get("cust_sel") != key: st.session_state["cust_sel"] = key
    with c1:
        sel = st.selectbox("Client", list(opts), format_func=lambda k: opts.get(k, k), key="cust_sel", persist_state="session")
    if st.query_params.get("c") != sel: st.query_params["c"] = sel
    info = nav.resolve(sel)
    name = display_name(info.get("name"), sel)
    rec = st.session_state.setdefault("cust_recent", {})
    rec.pop(sel, None); st.session_state["cust_recent"] = {sel: f"{name} · ref {sel}", **dict(list(rec.items())[:9])}

    # ── header ──
    st.markdown(f"## {name}")
    st.caption(" · ".join(x for x in [ui.first_txt(info.get("naics_desc")), ui.first_txt(info.get("segment_desc")), f"client ref {sel}",
                                      f"relationship {info['rltn_pwr_id']}" if ui.is_txt(info.get("rltn_pwr_id")) else "",
                                      f"payment entities: {', '.join(info.get('mdm_ids') or []) or 'none'}"] if x))
    with st.container(horizontal=True, gap="small"):
        if "attrition" in mods:
            from .attrition import service as AS
            rank = AS.list_map().get(sel)
            row = AS.row_for_key(sel)
            if rank: st.badge(f"On the attrition list · #{rank} · {row['p']:.0%} chance money leaves" if row else f"On the attrition list · #{rank}",
                              color="orange", icon=":material/trending_down:")
            elif row: st.badge(f"Deposits {money(row.get('bal'))} · not on the attrition list", color="gray")
            else: st.badge("No deposits scored", color="gray")
        if "fraud" in mods:
            from .fraud import service as FS
            f = FS.flag_map().get(sel)
            if f: st.badge(f"Suspicious activity · {f[0]} · {FS.T.name(f[2])}", color=ui.TIER_COLOR.get(f[0], "gray"), icon=":material/flag:")
            else: st.badge("No suspicious-activity pattern", color="green", icon=":material/check:")
    if not mods:
        ui.empty_state("You have no module enabled."); return

    labels = [nav.LENS_LABEL[m] for m in mods]
    if lens_from in mods:
        st.session_state["cust_lens"] = nav.LENS_LABEL[lens_from]
    elif st.session_state.get("cust_lens") not in labels:
        st.session_state["cust_lens"] = labels[0]
    tabs = st.tabs(labels, key="cust_lens", on_change="rerun")
    for t, m in zip(tabs, mods):
        if t.open:
            with t:
                try:
                    _lens_fn(m)(info)
                except Exception as e:                   # one lens failing must not take the page down
                    st.error(f"This view could not be drawn ({type(e).__name__}: {e}).")
