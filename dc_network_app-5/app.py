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

import numpy as np
import pandas as pd
import streamlit as st

import dc_core as dc
import dc_ui as ui

st.set_page_config(page_title="Data Center Network Explorer", layout="wide", initial_sidebar_state="collapsed")
logging.basicConfig(level=logging.INFO)
ui.inject_css()

# ------------------------------------------------------------------ state (survives a visit to another page)
# The controls of the network page only edit a draft. Nothing is read from the database until "Update the network" is
# pressed: the pressed state is copied to st.session_state["applied"], and the picture is drawn from that copy.
DEFAULTS = {"f_tabs": [], "f_level": "Parents", "f_dc": [], "f_so": [], "o_len": dc.DEFAULT_LEN, "o_top": 25, "o_dir": "Both ways",
            "o_reg": "All", "o_kinds": ["PNC customers", "Outside counterparties"], "o_min": 0, "o_family": True, "o_people": True,
            "o_hubs": True, "o_between": False, "o_overlay": True, "o_shared": False, "q": ""}
APPLY_KEYS = [k for k in DEFAULTS if k != "q"] + ["extra"]
for k, v in DEFAULTS.items():
    st.session_state[k] = st.session_state[k] if k in st.session_state else (list(v) if isinstance(v, list) else v)
st.session_state.setdefault("nav", 0)
st.session_state.setdefault("up_n", 0)
st.session_state.setdefault("extra", [])        # selections added by search or by a click on the network
st.session_state.setdefault("applied", None)    # the state the picture is drawn from
MAX_FOCUS = 20                  # selected nodes drawn (a workbook company can take two: PNC side and outside side)


def snapshot() -> dict:
    return {k: (list(st.session_state[k]) if isinstance(st.session_state[k], list) else st.session_state[k]) for k in APPLY_KEYS}


ROUTE_KEYS = ("cust", "fam", "sel", "cpty", "osel")


def route() -> dict:
    """The page shown. Read from the address once, when the session starts (so links and a browser refresh work); after
    that the session decides. Some Streamlit versions keep sending an old address with each click, and following it
    would reopen the previous page."""
    if "route" not in st.session_state:
        qp = st.query_params if ui.HAS_QP else {}
        st.session_state["route"] = {k: str(qp.get(k)) for k in ROUTE_KEYS if qp.get(k)}
    return st.session_state["route"]


def qp_get(k):
    return route().get(k) or None


def go_to(**params) -> None:
    """Open a page: go_to() = the network, go_to(cust=mdm), go_to(fam=master power id), go_to(sel=key), go_to(cpty=ent),
    go_to(osel=key). The address is updated to match, so the page can be bookmarked."""
    st.session_state["nav"] += 1
    st.session_state["route"] = {k: str(v) for k, v in params.items()}
    if ui.HAS_QP:
        try:
            st.query_params.clear()
            for k, v in (params or {"view": "network"}).items():
                st.query_params[k] = str(v)
        except Exception:  # noqa: BLE001  (the address is only a mirror)
            pass
    st.rerun()


def get_calendar():
    try:
        with st.spinner("Checking the payment data…"):
            return ui.calendar()
    except Exception as e:  # noqa: BLE001
        ui.fail("the payment data", e)
        st.stop()


def period(cal, n) -> tuple:
    return dc.period(cal["last"], int(n))


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
    st.session_state["applied"] = None
    st.session_state["nav"] += 1
    st.rerun()


def wb_groups(Es: pd.DataFrame, P: pd.DataFrame, col: str) -> dict:
    """Companies listed in a workbook column, one entry per name: group key -> workbook ids (a company can carry
    several ids across tabs)."""
    out = {}
    for i in dict.fromkeys(Es[col].dropna()):
        if i in P.index:
            out.setdefault(P.gkey[i], []).append(i)
    return out


def current_focus(wb, S: dict) -> list:
    """Selected workbook nodes plus anything added by search or by a click, in that order, for a state S."""
    fam = bool(S["o_family"])
    out = []
    if wb:
        E, P, M = wb["E"], wb["P"].set_index("id"), wb["M"]
        Es = E[E.tab.isin(S["f_tabs"])] if S["f_tabs"] else E
        parents = S["f_level"] == "Parents"
        g_dc = wb_groups(Es, P, "master_id" if parents else "dc_id")
        g_so = wb_groups(Es, P, "so_id")
        for k in S["f_dc"]:
            ids = list(g_dc.get(k, []))
            if not ids:
                continue
            nm = P.name[ids[0]]
            if parents:
                ids += [d for d in dict.fromkeys(Es.loc[Es.master_id.isin(ids), "dc_id"].dropna())]
            out += dc.focus_split(f"{'wbp' if parents else 'wbe'}|{k}", nm, ids, M, fam)
        for k in S["f_so"]:
            ids = g_so.get(k, [])
            if ids:
                out += dc.focus_split(f"wbs|{k}", P.name[ids[0]], ids, M, fam)
    out += [dc.Focus(*t) for t in S["extra"]]
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
        return dc.Focus(key, label, (), (ident,), side="out")
    if kind == "cust":
        return dc.Focus(key, label, (ident,), (), side="pnc")
    if kind == "fam":
        F = ui.load_family(ident, cal["tm_last"])
        mdm = tuple(F.mdm_id) if len(F) else tuple(members or ())
        return dc.Focus(key, label, mdm[:dc.MAX_FOCUS_IDS], (), (), len(mdm) > dc.MAX_FOCUS_IDS, "pnc")
    return None


def apply_pending() -> None:
    """Selection changes asked for by a button are applied here, before the widgets are drawn."""
    p = st.session_state.pop("pending", None)
    if not p:
        return
    if p.get("clear_q"):
        st.session_state.q = ""
    if p["do"] == "only":                         # "Show its network" is itself a request to draw: applied at once
        st.session_state.f_dc, st.session_state.f_so = [], []
        st.session_state.extra = [p["focus"]]
        st.session_state.applied = snapshot()
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


def selection_rows(wb, cal) -> None:
    """Tabs, level and search on one row; the two pickers on their own wide row so long names stay readable."""
    top = st.columns([1.4, 1.1, 1.5]) if wb else [None, None, st.columns([1.5, 2.5])[0]]
    if wb:
        E, P = wb["E"], wb["P"].set_index("id")
        tabs_all = list(dict.fromkeys(E.tab))
        st.session_state.f_tabs = [t for t in st.session_state.f_tabs if t in tabs_all]
        top[0].multiselect("Workbook tabs", tabs_all, key="f_tabs", placeholder="All tabs")
        Es = E[E.tab.isin(st.session_state.f_tabs)] if st.session_state.f_tabs else E
        top[1].radio("Data centers listed as", ["Parents", "Borrowing entities"], key="f_level", horizontal=True,
                     help="Parents: one node per data center parent with every borrowing entity under it. Borrowing entities: one node each.")
    search_row(cal, top[2])
    if not wb:
        return
    g_dc = wb_groups(Es, P, "master_id" if st.session_state.f_level == "Parents" else "dc_id")
    g_so = wb_groups(Es, P, "so_id")

    def label(g):
        return lambda k: str(P.name[g[k][0]]) if k in g else str(k)
    order = lambda g: sorted(g, key=lambda k: str(P.name[g[k][0]]).upper())   # noqa: E731
    st.session_state.f_dc = [k for k in st.session_state.f_dc if k in g_dc]
    st.session_state.f_so = [k for k in st.session_state.f_so if k in g_so]
    c = st.columns(2)
    c[0].multiselect(f"Data centers ({len(g_dc):,})", order(g_dc), key="f_dc", format_func=label(g_dc), placeholder="Choose one or more")
    c[1].multiselect(f"Second-order parties ({len(g_so):,})", order(g_so), key="f_so", format_func=label(g_so), placeholder="Choose one or more")


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
            f = dc.Focus(f"cust|{r.mdm_id}", r.customer_name, (r.mdm_id,), (), side="pnc")
        ask("add", f, clear_q=True)
    if b2.button("Open its page", key=f"q_open_{st.session_state.nav}"):
        go_to(cust=r.mdm_id)


def extras_row() -> None:
    ex = st.session_state.extra
    if not ex:
        return
    st.markdown('<div class="dc-note">Added by search or from the network (press ✕ to take one out):</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(ex), 5) + 1)
    for j, t in enumerate(ex[:5]):
        if cols[j].button(f"✕  {ui._short(t[1], 40)}", key=f"x_{t[0]}_{st.session_state.nav}", help="Remove from the selection"):
            ask("drop", key=t[0])
    if cols[-1].button("Clear the selection", key=f"x_all_{st.session_state.nav}"):
        ask("clear")


def options_rows(n_focus: int) -> None:
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


def graph_opts(S: dict, n_focus: int) -> dc.GraphOpts:
    kinds = tuple(k for k, lab in (("cust", "PNC customers"), ("ent", "Outside counterparties")) if lab in S["o_kinds"]) or ("cust", "ent")
    return dc.GraphOpts(top_n=int(S["o_top"]), family=bool(S["o_family"]),
                        direction={"Both ways": "both", "The selection pays": "out", "The selection is paid": "in"}[S["o_dir"]],
                        min_amt=float(S["o_min"] or 0), kinds=kinds, hide_people=bool(S["o_people"]), hide_hubs=bool(S["o_hubs"]),
                        regular={"All": "all", "Regular only": "regular", "Not regular only": "casual"}[S["o_reg"]],
                        shared_only=bool(S["o_shared"]) and n_focus > 1, overlay=bool(S["o_overlay"]))


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


def workbook_rows(focus, wb, ego, tabs) -> pd.DataFrame:
    """Workbook lines of the selected nodes, with whether payments were seen with the other side."""
    E, M, P = wb["E"], wb["M"], wb["P"].set_index("id")
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


def open_focus(f) -> None:
    """The page of a selected node: a PNC side opens the customer page, an outside side the counterparty page."""
    if f.mdm and not f.ent:
        go_to(cust=f.mdm[0]) if len(f.mdm) == 1 and f.key.startswith("cust|") else go_to(sel=f.key)
    elif f.ent and not f.mdm:
        go_to(osel=f.key) if (len(f.ent) > 1 or f.key.endswith("|out")) else go_to(cpty=f.ent[0])
    else:
        go_to(sel=f.key)


def find_focus(key: str):
    """A selected node by its key, from the network as last drawn (or as currently set up)."""
    if not key:
        return None
    for S in (st.session_state.applied, snapshot()):
        if S:
            f = next((f for f in current_focus(st.session_state.get("wb"), S) if f.key == key), None)
            if f is not None:
                return f
    return None


def node_panel(key: str, G: dc.Graph, focus, cal) -> None:
    """What can be done with the node that was clicked (or ticked in the table)."""
    N = G.nodes.set_index("key") if len(G.nodes) else None
    T = G.table.set_index("node") if len(G.table) else None
    nav = st.session_state.nav
    with st.container(border=True):
        if key.startswith("focus|"):
            f = focus[int(key.split("|")[1])]
            what = {"pnc": "selected · PNC side", "out": "selected · outside side (accounts at other banks, matched by name)",
                    "none": "selected · not found in the payment data"}.get(f.side, "selected")
            st.markdown(f"**{ui.esc(f.label)}** · {what}")
            b = st.columns([1.3, 1.3, 4])
            if f.side != "none" and b[0].button("Open its page", key=f"np_open_{nav}"):
                open_focus(f)
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
        st.markdown(ui.md(f"**{ui.esc(r.label)}** · {kind} · paid the selection {ui.usd(r.amt_in)} · received {ui.usd(r.amt_out)}"))
        b = st.columns([1.2, 1.3, 1.2, 3])
        mem = G.members.get(key, {}).get("mdm")
        if b[0].button("Show its network", key=f"np_ego_{nav}", help="Make it the only selected node"):
            ask("only", node_focus(key, r.label, cal, mem))
        if b[1].button("Add to the selection", key=f"np_add_{nav}"):
            ask("add", node_focus(key, r.label, cal, mem))
        if b[2].button("Open its page", key=f"np_page_{nav}"):
            kind_, _, ident = key.partition("|")
            go_to(**{"cpty" if kind_ == "ent" else kind_: ident})


def risk_view(focus, ego, streams, wb, A, cal, first: str, last: str) -> None:
    """A first screen for contagion questions, on the data already loaded plus customer totals: who depends on the
    selection, what a stop of its payments would touch, how concentrated its own money is, and warning signs."""
    nav = st.session_state.nav
    st.markdown('<div class="dc-h" style="margin-top:1.2rem">Risk view <span class="dc-chip warn" style="margin-left:8px">first draft, to try</span></div>',
                unsafe_allow_html=True)
    M, P = (wb["M"], wb["P"]) if wb else (None, None)
    G_all = dc.build_graph(ego, streams, focus, dc.GraphOpts(top_n=10 ** 6, family=bool(A["o_family"]), hide_people=False, hide_hubs=False),
                           M, None, P)
    met = ui.all_cols().get("cust_met") or set()
    can = {"mdm_id", "month"} <= met and ({"amt_all_in", "amt_all_out"} <= met or {"amt_int_in", "amt_ext_in", "amt_int_out", "amt_ext_out"} <= met)
    sig = hashlib.sha1(repr((ui._key(focus), first, last)).encode()).hexdigest()
    ran = st.session_state.get("risk_sig") == sig
    dep = None
    if ran and can:
        try:
            with st.spinner("Reading the total payments of the selection's PNC neighbours…"):
                ids = dc.neighbour_customers(ego, focus)
                tot = ui.load_totals(tuple(ids), first, last)
                dep = dc.dependents(ego, focus, tot, streams, M, P, hide_people=bool(A["o_people"]))
                then = dc.add_months(cal["tm_last"], -12) if cal["tm_last"] else None
                pd_then = ui.load_pd_then(tuple(sorted(set(dep.pwr.dropna()))), then) if then and then in cal["tm"] else {}
                dep["pd_then"] = dep.pwr.map(pd_then) if pd_then else np.nan
        except Exception as e:  # noqa: BLE001
            ui.fail("the dependence analysis", e)
            dep = None

    def need_run(key: str) -> bool:
        if not can:
            st.info("Customer payment totals are not available, so shares of money in cannot be computed.")
            return True
        if dep is None:
            st.markdown('<div class="dc-note">This reads the total money in and out of every PNC customer that deals with the selection '
                        '(one or two more reads of the payment data).</div>', unsafe_allow_html=True)
            if st.button("Run the dependence analysis", key=f"{key}_{nav}"):
                st.session_state["risk_sig"] = sig
                st.rerun()
            return True
        return False

    t_dep, t_stop, t_conc, t_warn = st.tabs(["Who depends on the selection", "If the selection stops paying", "Concentration of the selection",
                                             "Warning signs"])
    with t_dep:
        if not need_run("risk_run_a"):
            c = st.columns([1.6, 4])
            thr = c[0].slider("Dependent from (share of its money in)", 5, 50, 20, step=5, format="%d%%", key="risk_thr") / 100
            paid = dep[dep.recv > 0]
            hit = paid[paid.dep_in >= thr]
            k = st.columns(5)
            k[0].metric("PNC customers the selection pays", f"{len(paid):,}", help="Organisations" + ("" if A["o_people"] else " and people") + f", {first} to {last}.")
            k[1].metric(f"Get {thr:.0%} or more of their money in from it", f"{len(hit):,}")
            k[2].metric("DHE of those customers", ui.usd(dc.dhe_sum(hit)), help="Direct hard exposure, each Power ID counted once.")
            k[3].metric("Of them, in default or past due", f"{int(((hit.is_default.fillna(0) > 0) | (hit.past_due.fillna(0) > 0)).sum()):,}")
            k[4].metric("Of them, PD at least doubled in 12 months", f"{int((hit.pd >= 2 * hit.pd_then.where(hit.pd_then > 0)).sum()):,}")
            V = pd.DataFrame({"customer": dep.name, "received from the selection": dep.recv, "its total money in": dep.tot_in,
                              "share of its money in": dep.dep_in, "paid to the selection": dep.paid, "share of its money out": dep.dep_out,
                              "DHE": dep.dhe, "PD": dep.pd, "PD 12 months ago": dep.pd_then, "regular stream from the selection": dep.stream,
                              "in the workbook as": dep.wb_types, "industry": dep.naics, "Power ID": dep.pwr, "mainly paid by": dep.source})
            money = ["received from the selection", "its total money in", "paid to the selection", "DHE"]
            pcts = ["share of its money in", "share of its money out", "PD", "PD 12 months ago"]
            st.caption(f"Each PNC customer that deals with the selection, {first} to {last}, most dependent first. The share is what it received from "
                       "the selection divided by all the money it received (own-account transfers left out). Outside counterparties are not listed: "
                       "only their payments with PNC customers are visible, so no share can be computed. Tick a row to open the customer.")
            if ui.SELECTABLE and len(V):
                ev = ui.table(V, money=money, pct_cols=pcts, height=460, on_select="rerun", selection_mode="single-row", key=f"risk_dep_{nav}")
                rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
                if rows and rows[0] < len(dep):
                    go_to(cust=dep.mdm_id.iloc[rows[0]])
            else:
                ui.table(V, money=money, pct_cols=pcts, height=460)
            st.download_button("Download (CSV)", V.to_csv(index=False).encode(), f"dependents_{first}_{last}.csv", "text/csv", key="dl_risk_dep")

    with t_stop:
        if not need_run("risk_run_b"):
            c = st.columns([1.5, 1.5, 1.2, 2])
            stop = c[0].select_slider("Share of its payments that stop", [25, 50, 75, 100], 100, format_func=lambda v: f"{v}%", key="risk_stop") / 100
            thr2 = c[1].slider("A customer is hit when it loses at least", 5, 50, 10, step=5, format="%d%%", key="risk_thr2") / 100
            second = c[2].toggle("Second round", value=True, key="risk_second",
                                 help="Follow the money one step further: customers paid by the customers that lost money in the first round.")
            P2 = T2 = A2 = None
            if second:
                try:
                    with st.spinner("Following the payments of the first-round customers…"):
                        r1 = dc.round1_ids(dep)
                        P2 = ui.load_round2(tuple(r1), first, last) if r1 else None
                        r2 = dc.round2_ids(dep, P2, focus) if P2 is not None else []
                        T2 = ui.load_totals(tuple(r2), first, last) if r2 else None
                        A2 = ui.load_customers(tuple(r2), cal["tm_last"]) if r2 else None
                except Exception as e:  # noqa: BLE001
                    ui.fail("the second round", e)
                    P2 = None
            Sc = dc.scenario(dep, stop, P2, T2, A2, focus)
            hit = Sc[Sc.share >= thr2]
            h1, h2 = hit[hit.lost1 > 0], hit[hit.lost1 <= 0]
            k = st.columns(5)
            k[0].metric("Customers hit, first round", f"{len(h1):,}", help="PNC customers paid by the selection that lose at least the chosen share of their money in.")
            k[1].metric("Customers hit, second round only", f"{len(h2):,}" if second else "–", help="Not paid by the selection, but paid by customers that are.")
            k[2].metric("DHE of the customers hit", ui.usd(dc.dhe_sum(hit)), help="Direct hard exposure, each Power ID counted once.")
            k[3].metric("Money in lost by PNC customers", ui.usd(Sc.lost.sum()), help="All customers in the table, hit or not.")
            k[4].metric("Largest share lost", ui.pct(Sc.share.max(), 0) if len(Sc) else "–")
            top = Sc[Sc.share.notna()].head(15).copy()
            if len(top):
                top["s1"] = top.share1.fillna(0)
                top["s2"] = (top.share - top.s1).clip(lower=0)
                parts = [("s1", "first round", ui.BLUE)] + ([("s2", "second round", ui.ORANGE)] if second else [])
                ui.show(ui.share_bars(top, "name", parts, title="Share of money in lost, the 15 most affected customers"))
            V = pd.DataFrame({"customer": Sc.name, "round": Sc["round"], "money in lost": Sc.lost, "of which first round": Sc.lost1,
                              "of which second round": Sc.lost2, "its total money in": Sc.tot_in, "share of its money in lost": Sc.share,
                              "DHE": Sc.dhe, "PD": Sc.pd, "in the workbook as": Sc.wb_types, "industry": Sc.naics, "Power ID": Sc.pwr})
            ui.table(V, money=["money in lost", "of which first round", "of which second round", "its total money in", "DHE"],
                     pct_cols=["share of its money in lost", "PD"], height=420)
            st.download_button("Download (CSV)", V.to_csv(index=False).encode(), f"stop_payment_{first}_{last}.csv", "text/csv", key="dl_risk_stop")
            st.caption("A simple screen, not a forecast. First round: a customer loses the chosen share of what the selection paid it. Second round: a "
                       "customer that lost x% of its money in is assumed to pay x% less to everyone it pays. No deposits, credit lines or new "
                       "customers cushion the loss, there is no third round, and only payments PNC can see count. "
                       f"The second round follows the {dc.RISK_R1_CAP:,} largest first-round customers.")

    with t_conc:
        K = dc.concentration(G_all.table)
        outside_only = all(f.side == "out" for f in focus if f.side != "none")
        g = st.columns(2)
        for col, d, head, who in ((g[0], "in", "Money the selection receives", "payer"), (g[1], "out", "Money the selection pays", "payee")):
            with col:
                st.markdown(f'<div class="dc-h">{head}</div>', unsafe_allow_html=True)
                kk = st.columns(3)
                kk[0].metric("Total", ui.usd(K[d]["total"]))
                kk[1].metric(f"From its largest {who}" if d == "in" else f"To its largest {who}", ui.pct(K[d]["top1"], 0),
                             help=str(K[d]["top_name"] or ""))
                kk[2].metric("Largest five", ui.pct(K[d]["top5"], 0), help=f"{K[d]['n']:,} {who}s in all.")
                if K[d]["top_name"]:
                    st.caption(f"Largest {who}: {K[d]['top_name']}")
        if wb and len(K["by_rel"]):
            st.markdown('<div class="dc-h">By relationship in the workbook</div>', unsafe_allow_html=True)
            R = K["by_rel"].rename(columns={"rel": "relationship", "amt_in": "paid to the selection", "amt_out": "received from the selection"})
            ui.table(R, money=["paid to the selection", "received from the selection"])
        st.caption(f"All neighbours, {first} to {last}, people and hubs included"
                   + (", customers grouped by family." if A["o_family"] else ".")
                   + (" The selection is seen through accounts at other banks, so only its payments with PNC customers count: its true "
                      "concentration can be lower." if outside_only else ""))

    with t_warn:
        AL = dc.stream_alerts(streams, ego, focus)
        st.markdown('<div class="dc-h">Regular payments with the selection that are late or have stopped</div>', unsafe_allow_html=True)
        if AL.empty:
            st.info("No regular payment stream with the selection is late, overdue or stopped in the months read.")
        else:
            n_stop, n_late = int((AL.state == "stopped").sum()), int((AL.state == "late or overdue").sum())
            ui.chips([(f"{n_stop:,} stopped", "bad" if n_stop else ""), (f"{n_late:,} late or overdue", "warn" if n_late else ""),
                      (f"12-month dollars on these streams: {ui.usd(AL.s12.sum(min_count=1))}", "")])
            ui.table(AL.rename(columns={"party": "counterparty", "amt_med": "median payment", "last_dt": "last paid", "next_dt": "next expected",
                                        "s12": "12-month dollars", "month": "latest month on record"}),
                     money=["median payment", "12-month dollars"], height=320)
        T = G_all.table
        Cx = T[(T.kind == "cust") & ((T.is_default == True) | (T.past_due.fillna(0) > 0))] if len(T) else T   # noqa: E712
        st.markdown('<div class="dc-h">PNC neighbours in default or past due</div>', unsafe_allow_html=True)
        if Cx.empty:
            st.info("No PNC neighbour of the selection is in default or past due in the latest credit month.")
        else:
            ui.table(pd.DataFrame({"customer": Cx.label, "paid to the selection": Cx.amt_in, "received from the selection": Cx.amt_out,
                                   "DHE": Cx.dhe, "PD": Cx.pd_pct, "days past due": Cx.past_due,
                                   "in default": np.where(Cx.is_default == True, "yes", ""), "in the workbook as": Cx.wb_types}),   # noqa: E712
                     money=["paid to the selection", "received from the selection", "DHE"], pct_cols=["PD"], height=300)


def render_network() -> None:
    apply_pending()
    st.markdown('<div class="dc-title">Data Center Network Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="dc-sub">Who pays and is paid by the data centers and the companies around them, which of those flows are '
                'regular, and where PNC has credit exposure.</div>', unsafe_allow_html=True)
    cal = get_calendar()
    wb = workbook_box(cal)
    selection_rows(wb, cal)
    extras_row()
    draft = snapshot()
    n_draft = len(draft["f_dc"]) + len(draft["f_so"]) + len(draft["extra"])
    options_rows(n_draft)
    draft = snapshot()
    A = st.session_state.applied
    b = st.columns([1.3, 5])
    if b[0].button("Update the network", type="primary", disabled=n_draft == 0, key="apply",
                   help="Choose data centers, parties and options first; the payment data is read only when you press this."):
        st.session_state.applied = draft
        st.session_state["nav"] += 1
        st.rerun()
    if n_draft == 0 and A is None:
        b[1].markdown('<div class="dc-note" style="padding-top:0.55rem">Upload the workbook and choose a data center or a second-order party, '
                      'or find any PNC customer by name. Then press <b>Update the network</b>.</div>', unsafe_allow_html=True)
    elif A is None:
        b[1].markdown(f'<div class="dc-note" style="padding-top:0.55rem">{n_draft:,} selected. Press <b>Update the network</b> to draw.</div>',
                      unsafe_allow_html=True)
    elif draft != A:
        b[1].markdown('<span class="dc-chip warn" style="margin-top:0.45rem">The selection or the options changed: press Update the network '
                      'to redraw. The picture below still shows the previous choice.</span>', unsafe_allow_html=True)
    if A is None:
        return
    focus = current_focus(wb, A)
    if not focus:
        st.info("Nothing is selected in the network shown. Choose something and press Update the network.")
        return
    if len(focus) > MAX_FOCUS:
        st.warning(f"The first {MAX_FOCUS} selected nodes are shown.")
        focus = focus[:MAX_FOCUS]
    opts = graph_opts(A, len(focus))
    first, last = period(cal, A["o_len"])
    try:
        with st.spinner(f"Reading payments for {first} to {last}…"):
            ego = ui.load_ego(focus, first, last, cal["tm_last"])
            streams = ui.load_streams(focus, last, int(A["o_len"]))
    except Exception as e:  # noqa: BLE001
        ui.fail("the payments", e)
        return
    Es = None
    if wb:
        Es = wb["E"][wb["E"].tab.isin(A["f_tabs"])] if A["f_tabs"] else wb["E"]
    G = dc.build_graph(ego, streams, focus, opts, wb["M"] if wb else None, Es, wb["P"] if wb else None)
    if A["o_between"] and G.members:
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
    missing = [f.label for f in focus if f.side == "none" or (not f.mdm and not f.ent)]
    if missing:
        ui.chips([(f"Not in the payment data: {', '.join(missing[:6])}", "warn")])

    fig = ui.draw_graph(G, f"{first} to {last}", opts.family)
    fname = f"data_center_network_{first}_{last}"
    clicked = None
    if ui.SELECTABLE:
        ev = st.plotly_chart(fig, **ui.WIDE, key=f"net_{st.session_state.nav}", on_select="rerun", selection_mode="points",
                             config=ui.net_config(fname))
        clicked = ui.clicked_key(ev)
    else:
        st.plotly_chart(fig, **ui.WIDE, config=ui.net_config(fname))
    st.markdown('<div class="dc-note">Arrows point the way the money moved; thicker lines and larger nodes mean more dollars '
                f'({first} to {last}). Hover a line or a node for the figures; click a node to act on it. Drag to move, scroll to zoom.'
                + ("".join(" " + ui.esc(n) for n in G.notes)) + "</div>", unsafe_allow_html=True)
    x = st.columns([1.5, 5])
    title = f"{', '.join(f.label for f in focus[:4])}{' and others' if len(focus) > 4 else ''} · payments {first} to {last}"
    try:
        x[0].download_button("Download as HTML (interactive)", ui.export_html(fig, title), fname + ".html", "text/html", key="dl_html")
    except Exception as e:  # noqa: BLE001
        ui.fail("the HTML export", e)
    x[1].markdown('<div class="dc-note" style="padding-top:0.55rem">For a PNG, use the camera button at the top right of the picture '
                  '(it saves what is on screen, at twice the resolution).</div>', unsafe_allow_html=True)
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
        W = workbook_rows(focus, wb, ego, A["f_tabs"])
        if len(W):
            n_yes = int((W["payments seen in the period"] == "yes").sum())
            with st.expander(f"What the workbook says about the selection ({len(W):,} rows, payments seen for {n_yes:,})"):
                ui.table(W, height=360, skip=["excel row"])
                st.download_button("Download (CSV)", W.to_csv(index=False).encode(), "workbook_rows_selection.csv", "text/csv", key="dl_wb")
    if any(f.side != "none" for f in focus):
        risk_view(focus, ego, streams, wb, A, cal, first, last)


# ------------------------------------------------------------------ router
import dc_pages as pages  # noqa: E402  (after the helpers it uses)

if qp_get("cpty"):
    pages.render_cpty([qp_get("cpty")], None, go_to, get_calendar, find_focus)
elif qp_get("osel"):
    f = find_focus(qp_get("osel"))
    pages.render_cpty(list(f.ent) if f is not None else [], f, go_to, get_calendar, find_focus)
elif qp_get("cust") or qp_get("fam") or qp_get("sel"):
    kind = "cust" if qp_get("cust") else "fam" if qp_get("fam") else "sel"
    pages.render_customer(kind, qp_get(kind), find_focus(qp_get("sel")) if kind == "sel" else None, go_to, get_calendar, find_focus)
else:
    render_network()
