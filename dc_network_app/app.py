"""
Data Center Network Explorer (draft 1): who pays and is paid by the data centers in the workbook.

Pages in one script:
  network       upload the workbook, pick data centers / parties (or find any PNC customer), see the payment
                network around them: regular and casual flows, workbook relationships, credit snapshot
  customer      one PNC customer or family: payments, counterparties, regular payments, credit, deposits
  counterparty  one outside counterparty: network, PNC clients, accounts / banks / names, month by month
Run:  streamlit run app.py      (reads Impala through dbi; logic in dc_core.py)
"""
from __future__ import annotations

import hashlib
import logging

import pandas as pd
import streamlit as st

import dc_core as dc
import dc_ui as ui

st.set_page_config(page_title="Data Center Network Explorer", layout="wide", initial_sidebar_state="collapsed")
logging.basicConfig(level=logging.INFO)
ui.inject_css()

# ------------------------------------------------------------------ state (survives a visit to another page)
DEFAULTS = {"f_tabs": [], "f_level": "Parents", "f_dc": [], "f_so": [], "o_len": dc.DEFAULT_LEN, "o_top": 25, "o_dir": "Both ways",
            "o_reg": "All", "o_kinds": ["PNC customers", "Outside counterparties"], "o_min": 0, "o_family": True, "o_people": True,
            "o_hubs": True, "o_between": False, "o_overlay": True, "o_shared": False, "q": ""}
for k, v in DEFAULTS.items():
    st.session_state[k] = st.session_state[k] if k in st.session_state else (list(v) if isinstance(v, list) else v)
st.session_state.setdefault("nav", 0)
st.session_state.setdefault("up_n", 0)
st.session_state.setdefault("extra", [])        # selections added by search or by a click on the network
MAX_FOCUS = 12


def qp_get(k):
    return (st.query_params.get(k) if ui.HAS_QP else st.session_state.get("_qp", {}).get(k)) or None


def go_to(**params) -> None:
    """Open a page: go_to() = the network, go_to(cust=mdm), go_to(fam=master power id), go_to(sel=key), go_to(cpty=ent)."""
    st.session_state["nav"] += 1
    if ui.HAS_QP:
        st.query_params.clear()
        for k, v in params.items():
            st.query_params[k] = str(v)
    else:
        st.session_state["_qp"] = {k: str(v) for k, v in params.items()}
    st.rerun()


def get_calendar():
    try:
        with st.spinner("Checking the payment data…"):
            return ui.calendar()
    except Exception as e:  # noqa: BLE001
        ui.fail("the payment data", e)
        st.stop()


def period(cal) -> tuple:
    return dc.period(cal["last"], int(st.session_state.o_len))


# ------------------------------------------------------------------ workbook
def load_workbook(name: str, data: bytes, cal) -> None:
    sig = hashlib.sha1(data).hexdigest()
    wb = st.session_state.get("wb")
    if wb and wb["sig"] == sig:
        return
    box = st.empty()
    try:
        with st.spinner("Reading the workbook…"):
            E = dc.read_workbook(data)
            P = dc.parties(E)
        with st.spinner("Matching the workbook to the payment data…"):
            M = dc.match_workbook(E, cal["last"], progress=lambda t: box.info(t))
    except ValueError as e:
        box.empty()
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001
        box.empty()
        ui.fail("the workbook", e)
        return
    box.empty()
    P["status"] = P.id.map(M.status)
    st.session_state["wb"] = {"name": name, "sig": sig, "E": E, "P": P, "M": M}
    for k in ("f_tabs", "f_dc", "f_so"):
        st.session_state[k] = []
    st.session_state["nav"] += 1
    st.rerun()


def current_focus(wb) -> list:
    """Selected workbook nodes plus anything added by search or by a click, in that order."""
    fam = bool(st.session_state.o_family)
    out = []
    if wb:
        E, P, M = wb["E"], wb["P"].set_index("id"), wb["M"]
        tabs = st.session_state.f_tabs
        Es = E[E.tab.isin(tabs)] if tabs else E
        for i in st.session_state.f_dc:
            nm = P.name.get(i, i)
            out.append(dc.focus_parent(i, Es, M, nm, fam) if st.session_state.f_level == "Parents" else dc.focus_entity(i, M, nm, fam))
        for i in st.session_state.f_so:
            out.append(dc.focus_party(i, M, P.name.get(i, i), fam))
    out += [dc.Focus(*t) for t in st.session_state.extra]
    seen, res = set(), []
    for f in out:
        if f.key not in seen:
            seen.add(f.key)
            res.append(f)
    return res


def node_focus(key: str, label: str, cal, members=None):
    """A selection built from a node of the network (a customer, a family or an outside counterparty)."""
    kind, _, ident = key.partition("|")
    if kind == "ent":
        return dc.Focus(key, label, (), (ident,))
    if kind == "cust":
        return dc.Focus(key, label, (ident,), ())
    if kind == "fam":
        F = ui.load_family(ident, cal["tm_last"])
        mdm = tuple(F.mdm_id) if len(F) else tuple(members or ())
        return dc.Focus(key, label, mdm[:dc.MAX_FOCUS_IDS], (), (), len(mdm) > dc.MAX_FOCUS_IDS)
    return None


def apply_pending() -> None:
    """Selection changes asked for by a button are applied here, before the widgets are drawn."""
    p = st.session_state.pop("pending", None)
    if not p:
        return
    if p.get("clear_q"):
        st.session_state.q = ""
    if p["do"] == "only":
        st.session_state.f_dc, st.session_state.f_so = [], []
        st.session_state.extra = [p["focus"]]
    elif p["do"] == "add":
        if p["focus"][0] not in {t[0] for t in st.session_state.extra}:
            st.session_state.extra = st.session_state.extra + [p["focus"]]
    elif p["do"] == "drop":
        st.session_state.extra = [t for t in st.session_state.extra if t[0] != p["key"]]
    elif p["do"] == "clear":
        st.session_state.f_dc, st.session_state.f_so, st.session_state.extra = [], [], []
    st.session_state["nav"] += 1


def ask(do: str, focus=None, key=None, clear_q: bool = False) -> None:
    st.session_state["pending"] = {"do": do, "focus": ui._key([focus])[0] if focus is not None else None, "key": key, "clear_q": clear_q}
    st.rerun()


# ------------------------------------------------------------------ network page
def workbook_box(cal):
    wb = st.session_state.get("wb")
    with st.expander("Workbook" + (f": {wb['name']}" if wb else ""), expanded=wb is None):
        up = st.file_uploader("Upload the data center workbook (.xlsx)", type=["xlsx", "xlsm"], key=f"up_{st.session_state.up_n}",
                              help="Every tab with 'Master Power ID' and 'Second Order' columns is read. The file stays in this session.")
        if up is not None:
            load_workbook(up.name, up.getvalue(), cal)
            wb = st.session_state.get("wb")
        if wb:
            P = wb["P"]
            n_dc = int(P.roles.map(lambda r: "Data center parent" in r).sum())
            n_so = int(P.roles.map(lambda r: "Second-order party" in r).sum())
            found = P.status != "Not in the payment data"
            ui.chips([(f"{wb['E'].tab.nunique()} tabs · {len(wb['E']):,} rows", ""), (f"{n_dc:,} data center parents", ""),
                      (f"{n_so:,} second-order parties", ""), (f"{int(found.sum()):,} of {len(P):,} ids found in the payment data", "ok")])
            V = pd.DataFrame({"name": P.name, "workbook id": P.id, "id type": P.kind, "role": P.roles.map(", ".join),
                              "in the payment data as": P.status, "relationship types": P.types.map(", ".join),
                              "tabs": P.tabs.map(", ".join), "workbook rows": P.rows})
            st.markdown('<div class="dc-h">Everything listed in the workbook</div>', unsafe_allow_html=True)
            ui.table(V, height=300)
            st.download_button("Download this list (CSV)", V.to_csv(index=False).encode(), "workbook_ids_matched.csv", "text/csv")
    return wb


def selection_row(wb, c1, c2, c3) -> None:
    E, P = wb["E"], wb["P"].set_index("id")
    tabs_all = list(dict.fromkeys(E.tab))
    st.session_state.f_tabs = [t for t in st.session_state.f_tabs if t in tabs_all]
    c1.multiselect("Workbook tabs", tabs_all, key="f_tabs", placeholder="All tabs")
    Es = E[E.tab.isin(st.session_state.f_tabs)] if st.session_state.f_tabs else E
    c1.radio("Data centers listed as", ["Parents", "Borrowing entities"], key="f_level", horizontal=True,
             help="Parents: one node per data center parent with every borrowing entity under it. Borrowing entities: one node each.")
    col = "master_id" if st.session_state.f_level == "Parents" else "dc_id"
    short = {"PNC customer": "PNC customer", "Outside account (name match)": "outside account", "PNC customer + outside account": "PNC customer + outside account",
             "Not in the payment data": "not in the payment data"}
    lab = lambda i: f"{P.name.get(i, i)}  ·  {short.get(P.status.get(i, ''), '')}"   # noqa: E731
    by_name = lambda ids: sorted(ids, key=lambda i: str(P.name.get(i, i)).upper())   # noqa: E731
    dcs = by_name(i for i in dict.fromkeys(Es[col].dropna()) if i in P.index)
    sos = by_name(i for i in dict.fromkeys(Es.so_id.dropna()) if i in P.index)
    st.session_state.f_dc = [i for i in st.session_state.f_dc if i in set(dcs)]
    st.session_state.f_so = [i for i in st.session_state.f_so if i in set(sos)]
    c2.multiselect(f"Data centers ({len(dcs):,})", dcs, key="f_dc", format_func=lab, placeholder="Choose one or more")
    c3.multiselect(f"Second-order parties ({len(sos):,})", sos, key="f_so", format_func=lab, placeholder="Choose one or more")


def search_row(cal, c1) -> None:
    q = c1.text_input("Find any PNC customer", key="q", placeholder="Part of the name, a Power ID or an mdm id")
    c2 = st.container()
    if not q.strip():
        return
    try:
        R = ui.search_customers(q.strip())
    except Exception as e:  # noqa: BLE001
        ui.fail("the customer search", e)
        return
    if R.empty:
        c2.info("No PNC customer matches. Type at least three letters or digits.")
        return
    opts = list(range(len(R)))
    lab = lambda i: f"{R.customer_name.iloc[i]}  ·  Power ID {R.customer_pwr_id.iloc[i] or '–'}  ·  {'person' if R.party_type.iloc[i] == 'P' else 'organisation'}"   # noqa: E731
    i = c2.selectbox(f"Matches ({len(R)} shown)", opts, format_func=lab, key=f"q_pick_{st.session_state.nav}")
    r = R.iloc[i]
    b1, b2, _ = c2.columns([1, 1, 3])
    if b1.button("Add to the selection", key=f"q_add_{st.session_state.nav}"):
        fam = r.master_pwr_id or r.customer_pwr_id
        if st.session_state.o_family and isinstance(fam, str):
            f = node_focus(f"fam|{fam}", r.customer_name, cal, [r.mdm_id])
        else:
            f = dc.Focus(f"cust|{r.mdm_id}", r.customer_name, (r.mdm_id,), ())
        ask("add", f, clear_q=True)
    if b2.button("Open its page", key=f"q_open_{st.session_state.nav}"):
        go_to(cust=r.mdm_id)


def extras_row() -> None:
    ex = st.session_state.extra
    if not ex:
        return
    cols = st.columns(min(len(ex), 5) + 1)
    for j, t in enumerate(ex[:5]):
        if cols[j].button(f"✕  {ui._short(t[1], 26)}", key=f"x_{t[0]}_{st.session_state.nav}", help="Remove from the selection"):
            ask("drop", key=t[0])
    if cols[-1].button("Clear the selection", key=f"x_all_{st.session_state.nav}"):
        ask("clear")


def options_rows(n_focus: int) -> dc.GraphOpts:
    c = st.columns([1.5, 1.5, 1.15, 1.15, 1.15, 1.15])
    c[0].radio("Period (months to the latest)", dc.PERIOD_LENGTHS, key="o_len", horizontal=True)
    c[1].slider("Neighbours shown", 5, 100, key="o_top", step=5, help="The largest by dollars; the rest are summed in one 'others' node.")
    c[2].toggle("Group by family", key="o_family", help="One node per corporate family (parent Power ID). Off: one node per customer.")
    c[3].toggle("Workbook links", key="o_overlay", help="Dotted violet lines for relationships listed in the workbook, with or without payments.")
    c[4].toggle("Neighbour links", key="o_between", help="Payments among the neighbours themselves, as thin grey lines.")
    if n_focus > 1:
        c[5].toggle("Shared only", key="o_shared", help="Only neighbours linked to two or more selected nodes.")
    ss = st.session_state
    changed = [n for n, k in (("direction", "o_dir"), ("regularity", "o_reg"), ("types", "o_kinds"), ("minimum $", "o_min"),
                              ("people shown", "o_people"), ("hubs shown", "o_hubs")) if ss[k] != DEFAULTS[k]]
    with st.expander("More filters" + (f" (changed: {', '.join(changed)})" if changed else "")):
        f = st.columns([1.2, 1.2, 1.7, 1.0, 0.9, 0.9])
        f[0].selectbox("Direction", ["Both ways", "The selection pays", "The selection is paid"], key="o_dir")
        f[1].selectbox("Regularity", ["All", "Regular only", "Not regular only"], key="o_reg")
        f[2].multiselect("Neighbour types", ["PNC customers", "Outside counterparties"], key="o_kinds")
        f[3].number_input("Minimum $", min_value=0, step=10_000, key="o_min", help="Neighbours with less money than this over the period are left out.")
        f[4].toggle("Hide people", key="o_people")
        f[5].toggle("Hide hubs", key="o_hubs", help="Outside counterparties that deal with a very large number of PNC customers (processors, utilities, tax). "
                                                    "Companies listed in the workbook are always kept.")
    kinds = tuple(k for k, lab in (("cust", "PNC customers"), ("ent", "Outside counterparties")) if lab in ss.o_kinds) or ("cust", "ent")
    return dc.GraphOpts(top_n=int(ss.o_top), family=bool(ss.o_family),
                        direction={"Both ways": "both", "The selection pays": "out", "The selection is paid": "in"}[ss.o_dir],
                        min_amt=float(ss.o_min or 0), kinds=kinds, hide_people=bool(ss.o_people), hide_hubs=bool(ss.o_hubs),
                        regular={"All": "all", "Regular only": "regular", "Not regular only": "casual"}[ss.o_reg],
                        shared_only=bool(ss.o_shared) and n_focus > 1, overlay=bool(ss.o_overlay))


def neighbour_table(G: dc.Graph, n_focus: int) -> pd.DataFrame:
    T = G.table
    V = pd.DataFrame({
        "neighbour": T.label, "type": T.kind.map(ui.KIND_LABEL),
        "paid to the selection": T.amt_in, "received from the selection": T.amt_out, "payments": T.n_txn,
        "months active": T.months, "last month": T.last_month,
        "regular stream": T.reg_state.fillna("not regular") if len(T) else [], "cadence": T.reg_cadence, "median payment": T.reg_amt_med,
        "last paid": T.reg_last_dt, "next expected": T.reg_next_dt,
        "Power ID / id": T.pwr.where(T.pwr.notna(), T.id), "DHE": T.dhe, "PD": T.pd_pct, "PD grade": T.pd_grade,
        "in the workbook as": T.wb_types, "customers linked": T.members, "drawn": T.drawn})
    if n_focus > 1:
        V.insert(2, "selected nodes linked", T.n_focus)
    return V


def workbook_rows(focus, wb, ego) -> pd.DataFrame:
    """Workbook lines of the selected nodes, with whether payments were seen with the other side."""
    E, M, P = wb["E"], wb["M"], wb["P"].set_index("id")
    tabs = st.session_state.f_tabs
    Es = E[E.tab.isin(tabs)] if tabs else E
    ids = {w for f in focus for w in f.wb_ids}
    if not ids:
        return pd.DataFrame()
    fm = {m for f in focus for m in f.mdm}
    fe = {e for f in focus for e in f.ent}
    R = ego.rows
    nb_m = set(R.loc[R.scope == "int", "other"]) | set(R.loc[~R.cust.isin(fm), "cust"]) | (set(R.cust) & fm)
    nb_e = set(R.loc[R.scope == "ext", "other"])
    L = Es[Es.dc_id.isin(ids) | Es.master_id.isin(ids) | Es.so_id.isin(ids)].copy()
    if L.empty:
        return L

    def seen(row):
        cp = row.dc_id if row.so_id in ids and row.dc_id not in ids else row.so_id
        if cp is None or cp != cp:
            return ""
        own = set(M.mdm_own.get(cp, ())) - fm
        return "yes" if (own & nb_m) or ((set(M.ent_of.get(cp, ())) - fe) & nb_e) else "no"
    L["seen"] = [seen(r) for r in L.itertuples()]
    return pd.DataFrame({"tab": L.tab, "data center parent": L.master_name, "borrowing entity": L.dc_name, "second-order party": L.so_name,
                         "relationship": L.type, "workbook figure": L.amount, "party in the payment data as": L.so_id.map(P.status).fillna(""),
                         "payments seen in the period": L.seen, "excel row": L.excel_row})


def node_panel(key: str, G: dc.Graph, focus, cal) -> None:
    """What can be done with the node that was clicked (or ticked in the table)."""
    N = G.nodes.set_index("key") if len(G.nodes) else None
    T = G.table.set_index("node") if len(G.table) else None
    nav = st.session_state.nav
    with st.container(border=True):
        if key.startswith("focus|"):
            f = focus[int(key.split("|")[1])]
            st.markdown(f"**{ui.esc(f.label)}** · selected")
            b = st.columns([1.2, 1.2, 4])
            if b[0].button("Open its page", key=f"np_open_{nav}"):
                go_to(sel=f.key) if f.mdm else go_to(cpty=f.ent[0]) if f.ent else st.warning("It is not in the payment data.")
            if len(focus) > 1 and b[1].button("Keep only this one", key=f"np_only_{nav}"):
                ask("only", f)
            return
        if key == "others":
            st.markdown("**Neighbours not drawn.** Raise *Neighbours shown*, or open *All neighbours and amounts* below and tick a row.")
            return
        if key.startswith("wb|"):
            r = N.loc[key]
            st.markdown(f"**{ui.esc(r.label)}** is listed in the workbook for the selection, with no payments between them in the period "
                        f"({ui.esc(getattr(r, 'wb_status', ''))}).")
            return
        r = T.loc[key] if T is not None and key in T.index else None
        if r is None:
            return
        kind = "PNC customer family" if key.startswith("fam|") else ui.KIND_LABEL[r.kind]
        st.markdown(f"**{ui.esc(r.label)}** · {kind} · paid the selection {ui.usd(r.amt_in)} · received {ui.usd(r.amt_out)}")
        b = st.columns([1.2, 1.3, 1.2, 3])
        mem = G.members.get(key, {}).get("mdm")
        if b[0].button("Show its network", key=f"np_ego_{nav}", help="Make it the only selected node"):
            ask("only", node_focus(key, r.label, cal, mem))
        if b[1].button("Add to the selection", key=f"np_add_{nav}"):
            ask("add", node_focus(key, r.label, cal, mem))
        if b[2].button("Open its page", key=f"np_page_{nav}"):
            kind_, _, ident = key.partition("|")
            go_to(**{"cpty" if kind_ == "ent" else kind_: ident})


def render_network() -> None:
    apply_pending()
    st.markdown('<div class="dc-title">Data Center Network Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="dc-sub">Who pays and is paid by the data centers and the companies around them, which of those flows are '
                'regular, and where PNC has credit exposure.</div>', unsafe_allow_html=True)
    cal = get_calendar()
    wb = workbook_box(cal)
    c = st.columns([1.0, 1.5, 1.5, 1.2]) if wb else [None, None, None, st.columns([1.2, 3])[0]]
    if wb:
        selection_row(wb, *c[:3])
    search_row(cal, c[3])
    extras_row()
    focus = current_focus(wb)
    if not focus:
        st.info("Upload the workbook and choose a data center or a second-order party, or find any PNC customer by name.")
        return
    if len(focus) > MAX_FOCUS:
        st.warning(f"The first {MAX_FOCUS} selected nodes are shown.")
        focus = focus[:MAX_FOCUS]
    opts = options_rows(len(focus))
    first, last = period(cal)
    try:
        with st.spinner(f"Reading payments for {first} to {last}…"):
            ego = ui.load_ego(focus, first, last, cal["tm_last"])
            streams = ui.load_streams(focus, last, int(st.session_state.o_len))
    except Exception as e:  # noqa: BLE001
        ui.fail("the payments", e)
        return
    Es = None
    if wb:
        Es = wb["E"][wb["E"].tab.isin(st.session_state.f_tabs)] if st.session_state.f_tabs else wb["E"]
    G = dc.build_graph(ego, streams, focus, opts, wb["M"] if wb else None, Es, wb["P"] if wb else None)
    if st.session_state.o_between and G.members:
        try:
            mk = tuple((k, tuple(v["mdm"]), tuple(v["ent"])) for k, v in sorted(G.members.items()))
            dc.add_between(G, ui.load_between(mk, first, last))
        except Exception as e:  # noqa: BLE001
            ui.fail("the links between neighbours", e)

    FN = G.nodes[G.nodes.kind == "focus"]
    T = G.table
    k = st.columns(6)
    k[0].metric("Selected nodes", f"{len(focus):,}")
    k[1].metric("Neighbours", f"{len(T):,}", help="After the filters; the network draws the largest.")
    k[2].metric("Paid by the selection", ui.usd(T.amt_out.sum() if len(T) else 0))
    k[3].metric("Received by the selection", ui.usd(T.amt_in.sum() if len(T) else 0))
    k[4].metric("Neighbours with a regular stream", f"{int(T.regular.sum()) if len(T) else 0:,}")
    k[5].metric("DHE of the selection", ui.usd(FN.dhe.sum(min_count=1)) if "dhe" in FN else "–",
                help=f"Direct hard exposure of the selected PNC customers, {cal['tm_last'] or 'latest month'}.")
    missing = [f.label for f in focus if not f.mdm and not f.ent]
    if missing:
        ui.chips([(f"Not in the payment data: {', '.join(missing[:6])}", "warn")])

    fig = ui.draw_graph(G, f"{first} to {last}", opts.family)
    clicked = None
    if ui.SELECTABLE:
        ev = st.plotly_chart(fig, **ui.WIDE, key=f"net_{st.session_state.nav}", on_select="rerun", selection_mode="points",
                             config={"displayModeBar": False, "scrollZoom": True})
        clicked = ui.clicked_key(ev)
    else:
        st.plotly_chart(fig, **ui.WIDE, config={"displayModeBar": False, "scrollZoom": True})
    st.markdown('<div class="dc-note">Arrows point the way the money moved; thicker lines and larger nodes mean more dollars '
                f'({first} to {last}). Hover a line or a node for the figures; click a node to act on it. Drag to move, scroll to zoom.'
                + ("".join(" " + ui.esc(n) for n in G.notes)) + "</div>", unsafe_allow_html=True)
    slot = st.container()

    V = neighbour_table(G, len(focus))
    with st.expander(f"All neighbours and amounts ({len(V):,})"):
        money = ["paid to the selection", "received from the selection", "median payment", "DHE"]
        picked = None
        if ui.SELECTABLE and len(V):
            ev2 = ui.table(V, money=money, pct_cols=["PD"], height=420, on_select="rerun", selection_mode="single-row",
                           key=f"nb_{st.session_state.nav}")
            rows = list(getattr(getattr(ev2, "selection", None), "rows", None) or [])
            picked = T.node.iloc[rows[0]] if rows and rows[0] < len(T) else None
            st.caption("Tick a row to act on that neighbour.")
        else:
            ui.table(V, money=money, pct_cols=["PD"], height=420)
        st.download_button("Download (CSV)", V.to_csv(index=False).encode(), f"neighbours_{first}_{last}.csv", "text/csv", key="dl_nb")
    target = clicked or picked
    if target:
        with slot:
            node_panel(target, G, focus, cal)
    if wb:
        W = workbook_rows(focus, wb, ego)
        if len(W):
            n_yes = int((W["payments seen in the period"] == "yes").sum())
            with st.expander(f"What the workbook says about the selection ({len(W):,} rows, payments seen for {n_yes:,})"):
                ui.table(W, height=360, skip=["excel row"])
                st.download_button("Download (CSV)", W.to_csv(index=False).encode(), "workbook_rows_selection.csv", "text/csv", key="dl_wb")


# ------------------------------------------------------------------ router
import dc_pages as pages  # noqa: E402  (after the helpers it uses)

if qp_get("cpty"):
    pages.render_cpty(qp_get("cpty"), go_to, get_calendar, ask)
elif qp_get("cust") or qp_get("fam") or qp_get("sel"):
    kind = "cust" if qp_get("cust") else "fam" if qp_get("fam") else "sel"
    sel = next((f for f in current_focus(st.session_state.get("wb")) if f.key == qp_get("sel")), None) if kind == "sel" else None
    pages.render_customer(kind, qp_get(kind), sel, go_to, get_calendar, ask)
else:
    render_network()
