"""One-hop payment network around a customer: a flow diagram (payers → client → payees) and a network map in which
PNC clients are clickable. A simplified stand-in for the full ego-network explorer, which plugs in here later."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import nav, state, ui
from .fmt import cust_key, money

TYPE_COLOR = {"PNC client": ui.BLUE, "other bank": "#9aa5b4", "own account": ui.ORANGE, "own entity": ui.ORANGE}
ROLE_COLOR = {"pays in": ui.TEAL, "receives": ui.ORANGE, "both ways": ui.VIOLET}


@st.cache_data(ttl=3600, show_spinner=False)
def latest_month(mode):
    return state.source().latest_month()


@st.cache_data(ttl=3600, show_spinner="Reading payments…")
def _cp(mode, mdm_ids, end_month, n_months):
    return state.source().counterparties(list(mdm_ids), end_month, n_months)


@st.cache_data(ttl=3600, show_spinner=False)
def _ties(mode, mdm_ids, end_month, n_months):
    return state.source().internal_ties(list(mdm_ids), end_month, n_months)


def _agg(d, col):
    cols = ["cpty_id", col, f"n_{col}", "name", "bank", "kind", "customer_pwr_id", "is_self"]
    if d is None or not len(d): return pd.DataFrame(columns=cols)
    d = d.copy()
    for c in ["name", "bank", "kind", "customer_pwr_id"]:
        if c not in d: d[c] = None
    d["is_self"] = d["is_self"].fillna(False).astype(bool) if "is_self" in d else False
    d["amount"] = pd.to_numeric(d.amount, errors="coerce").fillna(0.0); d["n_txn"] = pd.to_numeric(d.n_txn, errors="coerce").fillna(0.0)
    return (d.groupby("cpty_id", as_index=False)
             .agg(**{col: ("amount", "sum"), f"n_{col}": ("n_txn", "sum"), "name": ("name", "first"), "bank": ("bank", "first"),
                     "kind": ("kind", "first"), "customer_pwr_id": ("customer_pwr_id", "first"), "is_self": ("is_self", "max")}))


def counterparties(mdm_ids, end_month, n_months):
    """One row per counterparty: money in, money out, type, and a customer key for PNC clients."""
    inn, out = _cp(state.MODE, tuple(sorted(mdm_ids)), end_month, n_months)
    a = _agg(inn, "amt_in").merge(_agg(out, "amt_out"), on="cpty_id", how="outer", suffixes=("", "_o"))
    for c in ["name", "bank", "kind", "customer_pwr_id", "is_self"]:
        if f"{c}_o" in a: a[c] = a[c].where(a[c].notna(), a[f"{c}_o"]); a = a.drop(columns=f"{c}_o")
    for c in ["amt_in", "amt_out", "n_amt_in", "n_amt_out"]: a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0.0)
    if not len(a): return a.assign(key=None, type=None, role=None, total=0.0, label=None)
    own = set(mdm_ids)
    a["is_self"] = a.is_self.fillna(False).astype(bool)
    a["type"] = np.where(a.cpty_id.isin(own), "own entity", np.where(a.is_self, "own account",
                         np.where(a.kind.eq("CUST"), "PNC client", "other bank")))
    a["key"] = [cust_key(p, m) if t == "PNC client" else None for p, m, t in zip(a.customer_pwr_id, a.cpty_id, a.type)]
    a["total"] = a.amt_in + a.amt_out
    a["role"] = np.where((a.amt_in > 0) & (a.amt_out > 0), "both ways", np.where(a.amt_in > 0, "pays in", "receives"))
    a["label"] = [n if ui.is_txt(n) else (f"own account at {b}" if t == "own account" else f"{b if ui.is_txt(b) else 'Counterparty'} …{str(c)[-4:]}")
                  for n, b, c, t in zip(a.name, a.bank, a.cpty_id, a.type)]
    return a[a.total > 0].sort_values("total", ascending=False).reset_index(drop=True)


def flags():
    """Customer key → short marks from the modules this viewer can see (never shows a module they can't open)."""
    f = {}
    if state.can("fraud"):
        from ..fraud import service as FS
        for k, (tier, _, prim) in FS.flag_map().items(): f.setdefault(k, []).append(f"{tier} · {FS.T.name(prim)}")
    if state.can("attrition"):
        from ..attrition import service as AS
        for k, rank in AS.list_map().items(): f.setdefault(k, []).append(f"attrition #{rank}")
    return f


def sankey(a, center, top, fl):
    P = a[a.amt_in > 0].nlargest(top, "amt_in"); Q = a[a.amt_out > 0].nlargest(top, "amt_out")
    rest_in, rest_out = a.amt_in.sum() - P.amt_in.sum(), a.amt_out.sum() - Q.amt_out.sum()
    labels, colors, src, dst, val, lcol = [], [], [], [], [], []
    def node(lbl, col):
        labels.append(lbl); colors.append(col); return len(labels) - 1
    c = node(center, ui.NAVY)
    for r in P.itertuples():
        i = node(r.label[:34], ui.RED if r.key in fl else TYPE_COLOR[r.type]); src.append(i); dst.append(c); val.append(r.amt_in)
        lcol.append("rgba(18,163,154,.35)")
    if rest_in > 0:
        i = node(f"{int((a.amt_in > 0).sum()) - len(P):,} other payers", "#c4ccd6"); src.append(i); dst.append(c); val.append(rest_in); lcol.append("rgba(18,163,154,.2)")
    for r in Q.itertuples():
        i = node(r.label[:34], ui.RED if r.key in fl else TYPE_COLOR[r.type]); src.append(c); dst.append(i); val.append(r.amt_out)
        lcol.append("rgba(242,140,40,.4)" if r.type != "own account" else "rgba(242,140,40,.75)")
    if rest_out > 0:
        i = node(f"{int((a.amt_out > 0).sum()) - len(Q):,} other payees", "#c4ccd6"); src.append(c); dst.append(i); val.append(rest_out); lcol.append("rgba(242,140,40,.2)")
    fig = go.Figure(go.Sankey(arrangement="snap", node=dict(label=labels, color=colors, pad=10, thickness=14, line=dict(width=0)),
                              link=dict(source=src, target=dst, value=val, color=lcol,
                                        hovertemplate="%{source.label} → %{target.label}: $%{value:,.0f}<extra></extra>")))
    return ui.base_fig(fig, max(380, 24 * max(len(P), len(Q)) + 120))


def netmap(a, center, top, fl, ties, own_ids):
    """Payers on the left, payees on the right, two-way counterparties above and below; PNC clients clickable."""
    A = a.head(top).copy()
    pos = {}
    for role, side in (("pays in", -1), ("receives", 1)):              # payers bulge out on the left, payees on the right
        g = list(A[A.role == role].index)
        ys = np.linspace(.95, -.95, len(g)) if len(g) > 1 else np.zeros(len(g))
        for idx, y in zip(g, ys): pos[idx] = (side * (1.25 + .4 * (1 - y * y)), float(y))
    g = list(A[A.role == "both ways"].index)                           # two-way counterparties: a row above (and below)
    rows = [g[::2], g[1::2]] if len(g) > 4 else [g, []]
    for part, y in zip(rows, (1.12, -1.12)):
        xs = np.linspace(-.75, .75, len(part)) if len(part) > 1 else np.zeros(len(part))
        for idx, x in zip(part, xs): pos[idx] = (float(x), y)
    mx = max(float(A.total.max()), 1.0)
    fig = go.Figure()
    for idx, r in A.iterrows():                                         # center → counterparty
        x, y = pos[idx]
        fig.add_scatter(x=[0, x], y=[0, y], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color=ROLE_COLOR[r.role], width=1 + 9 * np.sqrt(r.total / mx)), opacity=.55)
    if ties is not None and len(ties):                                   # PNC counterparties that also pay each other
        by_id = {r.cpty_id: pos[i] for i, r in A.iterrows() if r.type == "PNC client"}
        for t in ties.itertuples():
            if t.src_id in by_id and t.dst_id in by_id:
                (x0, y0), (x1, y1) = by_id[t.src_id], by_id[t.dst_id]
                fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=ui.GREY, width=1.5, dash="dot"),
                                hovertemplate=f"between PNC clients: {money(t.amount)}<extra></extra>", showlegend=False)
    for typ, col in TYPE_COLOR.items():
        g = A[A.type == typ]
        if not len(g): continue
        xs = [pos[i][0] for i in g.index]; ys = [pos[i][1] for i in g.index]
        fig.add_scatter(x=xs, y=ys, mode="markers+text", name=typ, text=[s[:26] for s in g.label],
                        textposition=["middle left" if x < -.05 else "middle right" if x > .05 else ("top center" if y > 0 else "bottom center")
                                      for x, y in zip(xs, ys)],
                        textfont=dict(size=11), customdata=np.c_[g.key.fillna("").to_numpy(), g.label.to_numpy()],
                        marker=dict(size=12 + 30 * np.sqrt(g.total / mx), color=col, line=dict(color="white", width=1)),
                        hovertext=[f"<b>{r.label}</b><br>{r.type}{' · ' + r.bank if ui.is_txt(r.bank) else ''}<br>in {money(r.amt_in)} · out {money(r.amt_out)}"
                                   + ("".join(f"<br>⚑ {x}" for x in fl.get(r.key, []))) + ("<br><i>click, then open below</i>" if r.key else "")
                                   for r in g.itertuples()], hoverinfo="text")
    fg = A[A.key.isin(list(fl))]
    if len(fg):                                                          # red rings: flagged in a module this viewer can see
        fig.add_scatter(x=[pos[i][0] for i in fg.index], y=[pos[i][1] for i in fg.index], mode="markers", name="flagged elsewhere",
                        marker=dict(size=20 + 30 * np.sqrt(fg.total / mx), color="rgba(0,0,0,0)", line=dict(color=ui.RED, width=3)),
                        hoverinfo="skip")
    fig.add_scatter(x=[0], y=[0], mode="markers+text", text=[center[:30]], textposition="bottom center", name="this client",
                    marker=dict(size=46, color=ui.NAVY, symbol="circle"), textfont=dict(size=13, color=ui.NAVY), hoverinfo="skip")
    fig.update_xaxes(visible=False, range=[-2.9, 2.9]); fig.update_yaxes(visible=False, range=[-1.3, 1.3])
    fig.add_annotation(x=-1.45, y=1.24, text="◀ pays this client", showarrow=False, font=dict(color=ui.TEAL))
    fig.add_annotation(x=1.45, y=1.24, text="receives from it ▶", showarrow=False, font=dict(color=ui.ORANGE))
    fig.update_layout(dragmode="pan", clickmode="event+select")
    return ui.base_fig(fig, 600)


def ego(info, key, end_month=None, lens="network", default_months=6):
    """The network block used by the network page and the customer page's network lens."""
    mdm = info.get("mdm_ids") or []
    if not mdm:
        ui.empty_state("No payment entity is linked to this client, so there is no payment network to draw.")
        return
    end_month = end_month or latest_month(state.MODE)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 2.5], vertical_alignment="bottom")
    n_m = c1.segmented_control("Months", [3, 6, 12], default=default_months, key=f"{key}_m", required=True)
    top = c2.segmented_control("Show top", [10, 20, 40], default=20, key=f"{key}_n", required=True)
    view = c3.segmented_control("View", ["Network map", "Flow diagram"], default="Network map", key=f"{key}_v", required=True)
    c4.caption(f"Payments over the {n_m} months to {end_month}. Blue = PNC client (click to open) · grey = client of another bank · "
               f"orange = its own accounts · red ring = flagged in a module you can see.")
    a = counterparties(mdm, end_month, int(n_m))
    if not len(a):
        ui.empty_state("No payments recorded in this window."); return
    fl = flags(); fl.pop(info.get("key"), None)
    center = ui.first_txt(info.get("name"), default=f"Client {info.get('key')}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Money in", money(a.amt_in.sum())); k2.metric("Money out", money(a.amt_out.sum()))
    k3.metric("Payers", f"{int((a.amt_in > 0).sum()):,}"); k4.metric("Payees", f"{int((a.amt_out > 0).sum()):,}")
    k5.metric("PNC clients among them", f"{int((a.type == 'PNC client').sum()):,}",
              help="Counterparties that are themselves PNC clients — the part of the network we can follow further.")
    pnc_ids = list(a.loc[a.type == "PNC client", "cpty_id"].head(int(top)))
    ties = _ties(state.MODE, tuple(sorted(pnc_ids)), end_month, int(n_m)) if len(pnc_ids) > 1 else None
    picked = []
    if view == "Flow diagram":
        st.plotly_chart(sankey(a, center, int(top), fl), key=f"{key}_sk", config={"displayModeBar": False})
    else:
        ev = st.plotly_chart(netmap(a, center, int(top), fl, ties, mdm), key=f"{key}_map", on_select="rerun", selection_mode="points",
                             config={"displayModeBar": False, "scrollZoom": True})
        try:
            picked = [p for p in (ev.selection.points if ev else []) if p.get("customdata") and p["customdata"][0]]
        except Exception:
            picked = []
    pnc = a[a.type == "PNC client"]
    if len(pnc):
        opts = {r.key: f"{r.label} · in {money(r.amt_in)} · out {money(r.amt_out)}" + (" · ⚑ " + ", ".join(fl[r.key]) if r.key in fl else "")
                for r in pnc.itertuples() if r.key}
        pre = picked[0]["customdata"][0] if picked else None
        o1, o2 = st.columns([4, 1.2], vertical_alignment="bottom")
        sel = o1.selectbox("Follow the money to a PNC counterparty", list(opts), format_func=opts.get,
                           index=list(opts).index(pre) if pre in opts else 0, key=f"{key}_follow")
        if o2.button("Open its page →", key=f"{key}_go", type="primary", width="stretch"):
            nav.open_customer(sel, lens=lens)
    with st.expander(f"All {len(a):,} counterparties", expanded=False):
        t = a.assign(flags=[", ".join(fl.get(k, [])) if k else "" for k in a.key])[["label", "type", "bank", "role", "amt_in", "amt_out", "n_amt_in", "n_amt_out", "flags"]]
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"label": "counterparty", "amt_in": st.column_config.NumberColumn("money in", format="dollar"),
                                    "amt_out": st.column_config.NumberColumn("money out", format="dollar"),
                                    "n_amt_in": st.column_config.NumberColumn("payments in", format="%d"),
                                    "n_amt_out": st.column_config.NumberColumn("payments out", format="%d")})
    st.caption("One hop around the client. The full ego-network explorer (several hops, time slider, communities) is being built "
               "separately and will replace this view.")
