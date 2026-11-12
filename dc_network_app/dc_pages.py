"""
Detail pages of the Data Center Network Explorer: one PNC customer (or family), and one outside counterparty.
The counterparty page keeps the tabs of the Prospect Finder (network, PNC clients, accounts / banks / names, month by
month) and adds regular payments and the workbook rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import dc_core as dc
import dc_ui as ui

BLUES = ["#184f95", "#3987e5", "#86b6ef"]          # ordinal blue steps for balances
ALL_OPTS = dict(top_n=10_000, hide_people=False, hide_hubs=False)


@st.cache_data(ttl=ui.TTL, show_spinner=False)
def _flows(mdm: tuple) -> pd.DataFrame:
    return dc.customer_flows(list(mdm), ui.all_cols().get("cust_met"))


@st.cache_data(ttl=ui.TTL, show_spinner=False)
def _credit(pwr: tuple) -> pd.DataFrame:
    return dc.customer_credit(list(pwr), ui.all_cols().get("tm"))


@st.cache_data(ttl=ui.TTL, show_spinner=False)
def _deposits(pwr: tuple) -> pd.DataFrame:
    c = ui.all_cols().get("dep")
    return dc.customer_deposits(list(pwr), c) if c else pd.DataFrame(columns=["month"])


@st.cache_data(ttl=ui.TTL, show_spinner=False)
def _trend(eid: str) -> pd.DataFrame:
    return dc.cpty_trend(eid, ui.all_cols().get("cpty_met"))


@st.cache_data(ttl=ui.TTL, show_spinner=False)
def _clients(eid: str, first: str, last: str) -> pd.DataFrame:
    return dc.cpty_clients(eid, first, last, ui.all_cols().get("cdim"))


@st.cache_data(ttl=ui.TTL, show_spinner=False)
def _accounts(eid: str, first: str, last: str) -> pd.DataFrame:
    return dc.cpty_accounts(eid, first, last, ui.all_cols())


def _period(cal) -> tuple:
    st.radio("Period (months to the latest)", dc.PERIOD_LENGTHS, key="o_len", horizontal=True)
    return dc.period(cal["last"], int(st.session_state.o_len))


def _state_chips(S: pd.DataFrame) -> None:
    vc = S.state.value_counts() if len(S) else {}
    cls = {"on schedule": "ok", "late or overdue": "warn", "stopped": "bad", "casual": ""}
    lab = {"on schedule": "on schedule", "late or overdue": "late or overdue", "stopped": "stopped", "casual": "no longer regular"}
    ui.chips([(f"{int(vc.get(s, 0)):,} {lab[s]}", cls[s]) for s in dc.STATES if vc.get(s, 0)])


def _wb_tab(wb, ids: set) -> None:
    E, P = wb["E"], wb["P"].set_index("id")
    L = E[E.dc_id.isin(ids) | E.master_id.isin(ids) | E.so_id.isin(ids)]
    if L.empty:
        st.info("It is not listed in the workbook.")
        return
    V = pd.DataFrame({"tab": L.tab, "data center parent": L.master_name, "borrowing entity": L.dc_name, "second-order party": L.so_name,
                      "relationship": L.type, "workbook figure": L.amount, "listed here as": np.where(L.so_id.isin(ids), "second-order party", "data center"),
                      "party in the payment data as": L.so_id.map(P.status).fillna(""), "excel row": L.excel_row})
    ui.table(V, height=min(38 + 35 * len(V), 460), skip=["excel row"])
    st.download_button("Download (CSV)", V.to_csv(index=False).encode(), "workbook_rows.csv", "text/csv", key="dl_wbrows")


# ====================================================================== customer page
def render_customer(kind: str, ident: str, sel, go_to, get_calendar, ask) -> None:
    nav = st.session_state.nav
    if st.button("← Back to the network"):
        go_to()
    cal = get_calendar()
    try:
        with st.spinner("Loading the customer…"):
            if kind == "fam":
                C = ui.load_family(ident, cal["tm_last"])
            elif kind == "sel":
                C = ui.load_customers(tuple(sel.mdm), cal["tm_last"]) if sel is not None else pd.DataFrame()
            else:
                C = ui.load_customers((ident,), cal["tm_last"])
    except Exception as e:  # noqa: BLE001
        ui.fail("the customer", e)
        return
    if C is None or C.empty:
        st.warning("This customer could not be found (a selection opened from the workbook is lost when the session restarts).")
        return
    head = C[C.customer_pwr_id == ident] if kind == "fam" and (C.customer_pwr_id == ident).any() else C
    name = sel.label if sel is not None else head.customer_name.iloc[0]
    mdm = tuple(C.mdm_id)
    pwr = tuple(dict.fromkeys(C.customer_pwr_id.dropna()))
    many = len(C) > 1
    st.markdown(f'<div class="dc-title">{ui.esc(name)}</div>', unsafe_allow_html=True)
    r0 = head.iloc[0]
    bits = [("PNC customer family" if kind == "fam" else "Selection from the workbook" if kind == "sel" else "PNC customer", ""),
            (f"{len(C):,} customers" if many else f"mdm id {r0.mdm_id}", ""),
            (f"Power ID {pwr[0]}" if len(pwr) == 1 else f"{len(pwr):,} Power IDs" if pwr else "no Power ID", ""),
            (f"parent {r0.master_pwr_id}" if isinstance(r0.master_pwr_id, str) and r0.master_pwr_id != r0.customer_pwr_id and not many else "", ""),
            (r0.naics_desc if isinstance(r0.naics_desc, str) else "", ""),
            ("person" if r0.party_type == "P" and not many else "", "")]
    if (C.is_default.fillna(0) > 0).any():
        bits.append(("in default", "bad"))
    ui.chips(bits)
    wb = st.session_state.get("wb")
    wb_ids = set()
    if wb:
        wb_ids = set().union(*[wb["M"].wb_of_mdm.get(m, set()) for m in mdm]) | (set(sel.wb_ids) if sel is not None else set())
    c = st.columns([2.2, 1.2, 1.2, 3])
    with c[0]:
        first, last = _period(cal)
    focus = [sel] if sel is not None else [dc.Focus(f"{kind}|{ident}", name, mdm[:dc.MAX_FOCUS_IDS], ())]
    if c[1].button("Show in the network", key=f"cu_net_{nav}"):
        st.session_state["pending"] = {"do": "only", "focus": ui._key(focus)[0], "key": None}
        go_to()
    if kind == "cust" and isinstance(r0.master_pwr_id, str) and c[2].button("Open its family", key=f"cu_fam_{nav}"):
        go_to(fam=r0.master_pwr_id)
    if sel is not None and sel.ent:
        st.caption("Also seen as outside accounts matched by name: open them from the network (click the selected node's neighbours) "
                   "or from the counterparty search of the Prospect Finder.")

    try:
        with st.spinner("Loading payments, credit and deposits…"):
            F = _flows(mdm)
            CR = _credit(pwr) if pwr else pd.DataFrame(columns=["month"])
            DP = _deposits(pwr) if pwr else pd.DataFrame(columns=["month"])
            ego = ui.load_ego(focus, first, last, cal["tm_last"])
            S = ui.load_streams(focus, last, int(st.session_state.o_len))
    except Exception as e:  # noqa: BLE001
        ui.fail("the customer's data", e)
        return
    Fp = F[(F.month >= first) & (F.month <= last)] if len(F) else F
    CM = dc.credit_by_month(CR) if len(CR) else CR
    lastc = CM.iloc[-1] if len(CM) else None
    k = st.columns(6)
    k[0].metric("Money in", ui.usd(Fp.amt_all_in.sum()) if "amt_all_in" in Fp else "–", help=f"All payments received, {first} to {last}.")
    k[1].metric("Money out", ui.usd(Fp.amt_all_out.sum()) if "amt_all_out" in Fp else "–", help=f"All payments made, {first} to {last}.")
    k[2].metric("DHE", ui.usd(lastc.dhe) if lastc is not None and "dhe" in CM else "–",
                help=f"Direct hard exposure, {lastc.month if lastc is not None else 'no credit row'}.")
    k[3].metric("PD", ui.pct(lastc.pd_pct_now) if lastc is not None and "pd_pct_now" in CM else "–",
                help="Probability of default (the worst grade when there are several Power IDs).")
    k[4].metric("Past due, days", ui.num(lastc.past_due_days) if lastc is not None and "past_due_days" in CM else "–")
    k[5].metric("Deposits, month end", ui.usd(DP.dep_bal_eom.iloc[-1]) if len(DP) and "dep_bal_eom" in DP else "–",
                help=f"{DP.month.iloc[-1]}" if len(DP) else "Deposit balances are not available.")

    names = ["Payments", "Counterparties", "Regular payments", "Credit"] + (["Deposits"] if len(DP) else []) + (["In the workbook"] if wb else [])
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Payments"]:
        if F.empty:
            st.info("No payment history for this customer.")
        else:
            ui.show(ui.line_chart(F, "month", [("amt_all_in", "money in", ui.C_IN), ("amt_all_out", "money out", ui.C_OUT)],
                                  title="All payments per month"))
            g = st.columns(2)
            if "amt_ext_in" in F or "amt_ext_out" in F:
                with g[0]:
                    ui.show(ui.line_chart(F, "month", [("amt_ext_in", "money in", ui.C_IN), ("amt_ext_out", "money out", ui.C_OUT)],
                                          title="With outside counterparties", height=250))
            if "amt_int_in" in F or "amt_int_out" in F:
                with g[1]:
                    ui.show(ui.line_chart(F, "month", [("amt_int_in", "money in", ui.C_IN), ("amt_int_out", "money out", ui.C_OUT)],
                                          title="With other PNC customers", height=250))
            with st.expander("The numbers"):
                ui.table(F.rename(columns={"amt_all_in": "money in", "amt_all_out": "money out", "amt_ext_in": "in, outside", "amt_ext_out": "out, outside",
                                           "amt_int_in": "in, PNC customers", "amt_int_out": "out, PNC customers", "ntxn_all_in": "payments in",
                                           "ntxn_all_out": "payments out", "ncpty_all_in": "payers", "ncpty_all_out": "payees"}),
                         money=["money in", "money out", "in, outside", "out, outside", "in, PNC customers", "out, PNC customers"])

    G = dc.build_graph(ego, S, focus, dc.GraphOpts(family=bool(st.session_state.o_family), **ALL_OPTS), wb["M"] if wb else None, None,
                       wb["P"] if wb else None)
    with tabs["Counterparties"]:
        T = G.table
        if T.empty:
            st.info(f"No payments with anyone in {first} to {last}.")
        else:
            V = pd.DataFrame({"counterparty": T.label, "type": T.kind.map(ui.KIND_LABEL), "paid to this customer": T.amt_in,
                              "received from this customer": T.amt_out, "payments": T.n_txn, "months active": T.months, "last month": T.last_month,
                              "regular stream": T.reg_state.fillna("not regular"), "cadence": T.reg_cadence, "median payment": T.reg_amt_med,
                              "DHE": T.dhe, "PD": T.pd_pct, "in the workbook as": T.wb_types})
            money = ["paid to this customer", "received from this customer", "median payment", "DHE"]
            st.caption(f"{len(V):,} counterparties in {first} to {last}, people and hubs included. Tick a row to open it.")
            if ui.SELECTABLE:
                ev = ui.table(V, money=money, pct_cols=["PD"], height=520, on_select="rerun", selection_mode="single-row", key=f"cu_cp_{nav}")
                rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
                if rows and rows[0] < len(T):
                    kind_, _, i = T.node.iloc[rows[0]].partition("|")
                    go_to(**{"cpty" if kind_ == "ent" else kind_: i})
            else:
                ui.table(V, money=money, pct_cols=["PD"], height=520)
            st.download_button("Download (CSV)", V.to_csv(index=False).encode(), "counterparties.csv", "text/csv", key="dl_cu_cp")

    with tabs["Regular payments"]:
        if S.empty:
            st.info("No regular payment streams on record for this customer in the months read.")
        else:
            _state_chips(S)
            cn = ego.cust.set_index("mdm_id").customer_name if len(ego.cust) else pd.Series(dtype=object)
            en = ego.ent.set_index("ent_id").ent_name if len(ego.ent) else pd.Series(dtype=object)
            inm = S.cust.isin(set(mdm))
            other = np.where(inm, S.other, S.cust)
            lab = pd.Series(other, index=S.index).map(lambda i: cn.get(i) or en.get(i) or i)
            V = pd.DataFrame({"counterparty": lab, "direction": np.where((S.dir == "out") == inm, "this customer pays", "this customer is paid"),
                              "cadence": S.cadence, "state": S.state.replace({"casual": "no longer regular"}), "median payment": S.amt_med,
                              "last paid": S.last_dt, "next expected": S.next_dt, "12-month dollars": S.s12, "latest month on record": S.month})
            V = V.sort_values("12-month dollars", ascending=False, na_position="last")
            ui.table(V, money=["median payment", "12-month dollars"], height=480)
            st.caption("From the stored regular-payment streams: at least 4 payment days, a known cadence and 70% of gaps on time.")

    with tabs["Credit"]:
        if CM.empty:
            st.info("No credit row for this customer.")
        else:
            if "is_default" in CM and (CM.is_default.fillna(0) > 0).any():
                d = CM[CM.is_default.fillna(0) > 0].month
                ui.chips([(f"In default in {len(d)} month{'s' if len(d) > 1 else ''}: {d.iloc[0]} to {d.iloc[-1]}", "bad")])
            g = st.columns(2)
            panels = [("dhe", "Direct hard exposure (DHE)", "usd"), ("pd_pct_now", "Probability of default", "pct"),
                      ("past_due_days", "Days past due", "days"), ("watchlist_exposure", "Watchlist exposure", "usd")]
            for j, (col, title, unit) in enumerate([p for p in panels if p[0] in CM and CM[p[0]].notna().any()]):
                with g[j % 2]:
                    ui.show(ui.line_chart(CM, "month", [(col, title, ui.BLUE)], unit=unit, title=title, height=250))
            more = [c for c in ("wrre", "mre", "soft_exposure") if c in CM and CM[c].notna().any()]
            if more:
                lab = {"wrre": "WRRE", "mre": "MRE", "soft_exposure": "soft exposure"}
                ui.show(ui.line_chart(CM, "month", [(c, lab[c], col) for c, col in zip(more, [ui.BLUE, ui.ORANGE, "#1baf7a"])],
                                      title="Other exposures", height=250))
            with st.expander("The numbers"):
                ui.table(CM.rename(columns={"dhe": "DHE", "pd_pct_now": "PD", "pd_grade_now": "PD grade", "past_due_days": "days past due",
                                            "watchlist_exposure": "watchlist exposure", "wrre": "WRRE", "mre": "MRE", "soft_exposure": "soft exposure",
                                            "is_default": "in default"}),
                         money=["DHE", "watchlist exposure", "WRRE", "MRE", "soft exposure"], pct_cols=["PD"])
            if many:
                L = C.dropna(subset=["customer_pwr_id"]).drop_duplicates("customer_pwr_id")
                st.markdown('<div class="dc-h">Latest credit snapshot per Power ID</div>', unsafe_allow_html=True)
                ui.table(pd.DataFrame({"customer": L.customer_name, "Power ID": L.customer_pwr_id, "DHE": L.dhe, "PD": L.pd_pct_now,
                                       "PD grade": L.pd_grade_now, "days past due": L.past_due_days, "watchlist exposure": L.watchlist_exposure})
                         .sort_values("DHE", ascending=False, na_position="last"), money=["DHE", "watchlist exposure"], pct_cols=["PD"])

    if "Deposits" in tabs:
        with tabs["Deposits"]:
            ser = [(c, lab, col) for (c, lab), col in zip([("dep_bal_eom", "month end"), ("dep_bal_mean", "daily average"), ("dep_bal_min", "lowest day")], BLUES)
                   if c in DP and (c != "dep_bal_min" or not many)]
            ui.show(ui.line_chart(DP, "month", ser, title="Deposit balances" + (" (summed over the Power IDs)" if len(pwr) > 1 else "")))
            with st.expander("The numbers"):
                ui.table(DP.rename(columns={"dep_bal_eom": "month end", "dep_bal_mean": "daily average", "dep_bal_min": "lowest day",
                                            "dep_bal_max": "highest day"}), money=["month end", "daily average", "lowest day", "highest day"])

    if "In the workbook" in tabs:
        with tabs["In the workbook"]:
            _wb_tab(wb, wb_ids)


# ====================================================================== counterparty page
def render_cpty(eid: str, go_to, get_calendar, ask) -> None:
    nav = st.session_state.nav
    if st.button("← Back to the network"):
        go_to()
    cal = get_calendar()
    try:
        with st.spinner("Loading the counterparty…"):
            TR = _trend(eid)
    except Exception as e:  # noqa: BLE001
        ui.fail("the counterparty", e)
        return
    name = next((v for v in TR.ent_name[::-1] if isinstance(v, str)), eid) if "ent_name" in TR and len(TR) else eid
    st.markdown(f'<div class="dc-title">{ui.esc(name)}</div>', unsafe_allow_html=True)
    lastr = TR.iloc[-1] if len(TR) else None
    hub = lastr is not None and "hub_tier" in TR and (TR.hub_tier.max() or 0) >= dc.HUB_TIER
    ui.chips([("Outside counterparty", ""), (str(lastr.name_class) if lastr is not None and "name_class" in TR and isinstance(lastr.name_class, str) else "", ""),
              ("hub: deals with a very large number of PNC customers" if hub else "", "warn"),
              (f"on record {TR.month.iloc[0]} to {TR.month.iloc[-1]}" if len(TR) else "no monthly record", "")])
    c = st.columns([2.2, 1.2, 4])
    with c[0]:
        first, last = _period(cal)
    focus = [dc.Focus(f"ent|{eid}", name, (), (eid,))]
    if c[1].button("Show in the network", key=f"cp_net_{nav}"):
        st.session_state["pending"] = {"do": "only", "focus": ui._key(focus)[0], "key": None}
        go_to()
    try:
        with st.spinner(f"Reading payments for {first} to {last}…"):
            ego = ui.load_ego(focus, first, last, cal["tm_last"])
            S = ui.load_streams(focus, last, int(st.session_state.o_len))
    except Exception as e:  # noqa: BLE001
        ui.fail("the counterparty's payments", e)
        return
    wb = st.session_state.get("wb")
    G_all = dc.build_graph(ego, S, focus, dc.GraphOpts(family=False, **ALL_OPTS), wb["M"] if wb else None, None, wb["P"] if wb else None)
    T = G_all.table
    k = st.columns(5)
    k[0].metric("PNC customers", f"{len(T):,}", help=f"Customers that paid it or were paid by it, {first} to {last}.")
    k[1].metric("Received from PNC customers", ui.usd(T.amt_in.sum() if len(T) else 0))
    k[2].metric("Paid to PNC customers", ui.usd(T.amt_out.sum() if len(T) else 0))
    k[3].metric("Customers with a regular stream", f"{int(T.regular.sum()) if len(T) else 0:,}")
    k[4].metric("DHE of those customers", ui.usd(T.dhe.sum(min_count=1)) if len(T) else "–", help="Direct hard exposure of the PNC customers linked to it.")

    names = ["Network", "PNC clients", "Accounts, banks and names", "Month by month", "Regular payments"] + (["In the workbook"] if wb else [])
    tabs = dict(zip(names, st.tabs(names)))

    with tabs["Network"]:
        n = st.slider("PNC customers shown", 5, 60, 25, step=5, key="cp_top")
        G = dc.build_graph(ego, S, focus, dc.GraphOpts(top_n=int(n), family=False, hide_people=False, hide_hubs=False), wb["M"] if wb else None, None,
                           wb["P"] if wb else None)
        fig = ui.draw_graph(G, f"{first} to {last}", family=False, height=640)
        if ui.SELECTABLE:
            ev = st.plotly_chart(fig, **ui.WIDE, key=f"cpnet_{nav}", on_select="rerun", selection_mode="points", config={"displayModeBar": False})
            key = ui.clicked_key(ev)
            if key and key.startswith("cust|"):
                go_to(cust=key.split("|", 1)[1])
            st.caption("Click a customer to open its page.")
        else:
            st.plotly_chart(fig, **ui.WIDE, config={"displayModeBar": False})

    with tabs["PNC clients"]:
        try:
            CL = _clients(eid, first, last)
        except Exception as e:  # noqa: BLE001
            ui.fail("its PNC clients", e)
            CL = pd.DataFrame()
        if CL.empty:
            st.info(f"No PNC customer paid it or was paid by it in {first} to {last}.")
        else:
            V = pd.DataFrame({"customer": CL.customer_name, "mdm id": CL.mdm_id, "Power ID": CL.customer_pwr_id})
            if "party_type" in CL:
                V["type"] = CL.party_type.map({"O": "organisation", "P": "person"})
            if "naics_desc" in CL:
                V["industry"] = CL.naics_desc
            V["paid to it"], V["received from it"] = CL.paid_to_it, CL.received_from_it
            V["payments"], V["months active"] = CL.n_paid + CL.n_received, CL.months
            reg = T.set_index("id").reg_state if len(T) else pd.Series(dtype=object)
            V["regular stream"] = CL.mdm_id.map(reg).fillna("not regular")
            st.caption(f"{len(V):,} PNC customers, largest first. Tick a row to open the customer.")
            if ui.SELECTABLE:
                ev = ui.table(V, money=["paid to it", "received from it"], skip=["mdm id"], height=520, on_select="rerun",
                              selection_mode="single-row", key=f"cp_cl_{nav}")
                rows = list(getattr(getattr(ev, "selection", None), "rows", None) or [])
                if rows and rows[0] < len(CL):
                    go_to(cust=CL.mdm_id.iloc[rows[0]])
            else:
                ui.table(V, money=["paid to it", "received from it"], skip=["mdm id"], height=520)
            st.download_button("Download (CSV)", V.to_csv(index=False).encode(), "pnc_clients.csv", "text/csv", key="dl_cp_cl")

    with tabs["Accounts, banks and names"]:
        try:
            AC = _accounts(eid, first, last)
        except Exception as e:  # noqa: BLE001
            ui.fail("its accounts", e)
            AC = pd.DataFrame()
        if AC.empty:
            st.info("Account detail is not available for this counterparty in the period.")
        else:
            B = AC.groupby("bank", as_index=False).agg(accounts=("account", "nunique"), dollars=("amt", "sum"), payments=("n_txn", "sum")).sort_values("dollars", ascending=False)
            g = st.columns([1, 1.8])
            with g[0]:
                st.markdown('<div class="dc-h">Banks</div>', unsafe_allow_html=True)
                ui.table(B, money=["dollars"])
            with g[1]:
                st.markdown('<div class="dc-h">Accounts and the names on the payments</div>', unsafe_allow_html=True)
                ui.table(pd.DataFrame({"account": AC.account, "names": AC.names, "bank": AC.bank, "dollars": AC.amt, "payments": AC.n_txn,
                                       "first month": AC.first_month, "last month": AC.last_month}), money=["dollars"], height=420)
            st.caption("Account numbers are masked. One counterparty can hold several accounts, at several banks, under several names.")

    with tabs["Month by month"]:
        if TR.empty:
            st.info("No monthly record for this counterparty.")
        else:
            panels = [(t, u, [(c, l) for c, l in ser if c in TR and TR[c].notna().any()], why) for t, u, ser, why in dc.TREND_PANELS]
            panels = [p for p in panels if p[2]]
            g = st.columns(2)
            for j, (title, unit, ser, why) in enumerate(panels):
                with g[j % 2]:
                    ui.show(ui.line_chart(TR, "month", [(c, l, col) for (c, l), col in zip(ser, [ui.C_IN, ui.C_OUT])], unit=unit, title=title, height=250))
                    st.caption(why)

    with tabs["Regular payments"]:
        if S.empty:
            st.info("No regular payment streams with PNC customers on record in the months read.")
        else:
            _state_chips(S)
            cn = ego.cust.set_index("mdm_id").customer_name if len(ego.cust) else pd.Series(dtype=object)
            V = pd.DataFrame({"customer": S.cust.map(lambda i: cn.get(i) or i), "direction": np.where(S.dir == "out", "the customer pays it", "it pays the customer"),
                              "cadence": S.cadence, "state": S.state.replace({"casual": "no longer regular"}), "median payment": S.amt_med,
                              "last paid": S.last_dt, "next expected": S.next_dt, "12-month dollars": S.s12, "latest month on record": S.month})
            ui.table(V.sort_values("12-month dollars", ascending=False, na_position="last"), money=["median payment", "12-month dollars"], height=480)

    if "In the workbook" in tabs:
        with tabs["In the workbook"]:
            _wb_tab(wb, set(wb["M"].wb_of_ent.get(eid, set())))
