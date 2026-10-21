"""One-hop payment networks: around a customer, or around an industry (every customer with one NAICS code). Each comes as a
flow diagram (payers → center → payees) and a network map with directed arrows (payer → payee) in which PNC clients, or
other industries, are clickable. A simplified stand-in for the full ego-network explorer, which plugs in here later."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import industry, nav, state, ui
from .fmt import cust_key, money

TYPE_COLOR = {"PNC client": ui.BLUE, "other bank": "#9aa5b4", "own account": ui.ORANGE, "own entity": ui.ORANGE,
              "industry": ui.BLUE, "NAICS unknown": "#8fb3dd", "outside PNC": "#9aa5b4"}
ROLE_COLOR = {"pays in": ui.TEAL, "receives": ui.ORANGE, "both ways": ui.VIOLET}
IN_COL, OUT_COL, TIE_COL = "rgba(18,163,154,.6)", "rgba(242,140,40,.65)", "rgba(110,122,140,.8)"   # money in · out · between others


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


def _offset(p0, p1, d):
    """Shift the segment p0 → p1 sideways by d (data units), so the two directions of a two-way pair sit side by side."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0; n = float(np.hypot(dx, dy)) or 1.0
    ox, oy = -dy / n * d, dx / n * d
    return (x0 + ox, y0 + oy), (x1 + ox, y1 + oy)


def _arrow(fig, p0, p1, color, width, dash=False, hover=None, at=.6):
    """Directed edge p0 → p1: a line from node to node, with the arrowhead `at` of the way along it (mid-edge heads stay
    readable where many edges meet one node). The head is an annotation, so it pans and zooms with the map. dash: dotted line.
    The line trace carries the hover text, if any."""
    (x0, y0), (x1, y1) = p0, p1
    fig.add_scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=color, width=float(width), dash="dot" if dash else "solid"),
                    showlegend=False, hovertemplate=(hover + "<extra></extra>") if hover else None, hoverinfo=None if hover else "skip")
    b, h = max(at - .12, 0.0), at
    fig.add_annotation(x=x0 + h * (x1 - x0), y=y0 + h * (y1 - y0), ax=x0 + b * (x1 - x0), ay=y0 + b * (y1 - y0),
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, text="",
                       arrowhead=2, arrowsize=max(.75, 3.0 / max(float(width), .1)), arrowwidth=float(width), arrowcolor=color)


def netmap(a, center, top, fl, ties, own_ids, subject="client", layout="sides"):
    """Arrows point payer → payee: teal = money into the center, orange = money out of it, dotted grey = between the others;
    width follows the dollars. Rows with a key are clickable (PNC clients, or other industries).
    layout "sides": payers on the left, payees on the right, two-way counterparties in rows above and below (clients, where
    most counterparties go one way). layout "ring": everyone on a ring, net payers on the left half and net receivers on the
    right, biggest at the middle of each side (industries, where almost every pair goes both ways)."""
    A = a.head(top).copy()
    pos = {}
    if layout == "ring":
        net = A.amt_in - A.amt_out
        for side, g in ((-1, A[net >= 0]), (1, A[net < 0])):
            g = list(g.sort_values("total", ascending=False).index)
            order = [g[i] for i in sorted(range(len(g)), key=lambda i: (i % 2, i if i % 2 == 0 else -i))]   # big ones mid-side
            # both halves tilted the same way (−62°…78°): two nodes facing each other never line up through the center
            th = np.linspace(-62, 78, len(order)) if len(order) > 1 else np.full(len(order), 8.0)
            for idx, t in zip(order, np.radians(th)): pos[idx] = (side * 1.75 * np.cos(t), 1.05 * np.sin(t))
    else:
        for role, side in (("pays in", -1), ("receives", 1)):          # payers bulge out on the left, payees on the right
            g = list(A[A.role == role].index)
            ys = np.linspace(.95, -.95, len(g)) if len(g) > 1 else np.zeros(len(g))
            for idx, y in zip(g, ys): pos[idx] = (side * (1.25 + .4 * (1 - y * y)), float(y))
        g = list(A[A.role == "both ways"].index)                       # two-way counterparties: a row above (and below)
        rows = [g[::2], g[1::2]] if len(g) > 4 else [g, []]
        for part, y in zip(rows, (1.12, -1.12)):
            xs = np.linspace(-.75, .75, len(part)) if len(part) > 1 else np.zeros(len(part))
            for idx, x in zip(part, xs): pos[idx] = (float(x), y)
    mx = max(float(A.total.max()), 1.0)
    size = {i: 12 + 30 * np.sqrt(r.total / mx) for i, r in A.iterrows()}
    amx = max(float(A.amt_in.max()), float(A.amt_out.max()), 1.0)
    C = (0.0, 0.0)                                                      # the center node
    fig = go.Figure()
    for idx, r in A.iterrows():                                         # one arrow per direction, side by side when two-way
        p = pos[idx]; d = .035 if (r.amt_in > 0 and r.amt_out > 0) else 0.0
        if r.amt_in > 0:
            s0, s1 = _offset(p, C, d); _arrow(fig, s0, s1, IN_COL, 1 + 8 * np.sqrt(r.amt_in / amx))
        if r.amt_out > 0:
            s0, s1 = _offset(C, p, d); _arrow(fig, s0, s1, OUT_COL, 1 + 8 * np.sqrt(r.amt_out / amx))
    if ties is not None and len(ties):                                   # payments between the others (PNC clients / industries)
        by_id = {r.cpty_id: (pos[i], r.label) for i, r in A.iterrows() if r.type in ("PNC client", "industry")}
        T = ties[ties.src_id.isin(list(by_id)) & ties.dst_id.isin(list(by_id)) & (ties.src_id != ties.dst_id)]
        tmx = max(float(T.amount.max()), 1.0) if len(T) else 1.0
        pairs = set(zip(T.src_id, T.dst_id))
        for t in T.itertuples():
            (p0, l0), (p1, l1) = by_id[t.src_id], by_id[t.dst_id]
            if (t.dst_id, t.src_id) in pairs: p0, p1 = _offset(p0, p1, .03)
            _arrow(fig, p0, p1, TIE_COL, 1 + 2.5 * np.sqrt(t.amount / tmx), dash=True,
                   hover=f"{l0[:30]} → {l1[:30]}: {money(t.amount)}")
    for typ, col in TYPE_COLOR.items():
        g = A[A.type == typ]
        if not len(g): continue
        xs = [pos[i][0] for i in g.index]; ys = [pos[i][1] for i in g.index]
        hover = (list(g.hover) if "hover" in g else
                 [f"<b>{r.label}</b><br>{r.type}{' · ' + r.bank if ui.is_txt(r.bank) else ''}<br>in {money(r.amt_in)} · out {money(r.amt_out)}"
                  + ("".join(f"<br>⚑ {x}" for x in fl.get(r.key, []))) + ("<br><i>click, then open below</i>" if r.key else "")
                  for r in g.itertuples()])
        fig.add_scatter(x=xs, y=ys, mode="markers+text", name=typ, text=[s[:34 if layout == "ring" else 26] for s in g.label],
                        textposition=["middle left" if x < -.05 else "middle right" if x > .05 else ("top center" if y > 0 else "bottom center")
                                      for x, y in zip(xs, ys)],
                        textfont=dict(size=11), customdata=np.c_[g.key.fillna("").to_numpy(), g.label.to_numpy()],
                        marker=dict(size=[size[i] for i in g.index], color=col, line=dict(color="white", width=1)),
                        hovertext=hover, hoverinfo="text")
    fg = A[A.key.isin(list(fl))]
    if len(fg):                                                          # red rings: flagged in a module this viewer can see
        fig.add_scatter(x=[pos[i][0] for i in fg.index], y=[pos[i][1] for i in fg.index], mode="markers", name="flagged elsewhere",
                        marker=dict(size=[size[i] + 8 for i in fg.index], color="rgba(0,0,0,0)", line=dict(color=ui.RED, width=3)),
                        hoverinfo="skip")
    fig.add_scatter(x=[0], y=[0], mode="markers+text", text=[center[:30]], textposition="bottom center", name=f"this {subject}",
                    marker=dict(size=46, color=ui.NAVY, symbol="circle"), textfont=dict(size=13, color=ui.NAVY), hoverinfo="skip")
    fig.update_xaxes(visible=False, range=[-2.9, 2.9]); fig.update_yaxes(visible=False, range=[-1.3, 1.3])
    left, right = ((f"◀ pay this {subject} more than it pays them", "receive more from it than they pay it ▶") if layout == "ring"
                   else (f"◀ pays this {subject}", "receives from it ▶"))
    fig.add_annotation(x=-1.45, y=1.24, text=left, showarrow=False, font=dict(color=ui.TEAL))
    fig.add_annotation(x=1.45, y=1.24, text=right, showarrow=False, font=dict(color=ui.ORANGE))
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
               f"orange = its own accounts · red ring = flagged in a module you can see. Arrows point payer → payee: teal = "
               f"money in, orange = money out, dotted = between PNC clients.")
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


# ── industry (NAICS) ego network ────────────────────────────────────────────────────────────────────────────────────────
def industry_frame(F, show_ext=True, show_unk=True):
    """Flows of one industry → one row per other group (another NAICS, PNC customers without NAICS, outside PNC), in the
    shape the map and the flow diagram use."""
    X = F[F.dir.isin(["IN", "OUT"])]
    if not show_ext: X = X[X.grp != "EXTERNAL"]
    if not show_unk: X = X[X.grp != "UNKNOWN"]
    cols = ["cpty_id", "label", "type", "key", "amt_in", "amt_out", "n_amt_in", "n_amt_out", "parties_in", "parties_out",
            "members_in", "members_out", "total", "role", "bank", "hover"]
    if not len(X): return pd.DataFrame(columns=cols)
    v = lambda col, d: X[X.dir == d].groupby("grp")[col].sum()
    a = pd.DataFrame(index=pd.Index(sorted(X.grp.unique()), name="cpty_id"))
    for out, col, d in [("amt_in", "amount", "IN"), ("amt_out", "amount", "OUT"), ("n_amt_in", "n_txn", "IN"), ("n_amt_out", "n_txn", "OUT"),
                        ("parties_in", "n_parties", "IN"), ("parties_out", "n_parties", "OUT"),
                        ("members_in", "n_members", "IN"), ("members_out", "n_members", "OUT")]:
        a[out] = v(col, d).reindex(a.index).fillna(0.0)
    desc = X.dropna(subset=["grp_desc"]).groupby("grp").grp_desc.first()
    a = a.reset_index()
    a["type"] = np.where(a.cpty_id == "EXTERNAL", "outside PNC", np.where(a.cpty_id == "UNKNOWN", "NAICS unknown", "industry"))
    a["key"] = [f"NAICS:{c}" if t == "industry" else None for c, t in zip(a.cpty_id, a.type)]
    a["label"] = [industry.GROUPS.get(c) or (f"{c} · {desc[c]}" if ui.is_txt(desc.get(c)) else c) for c in a.cpty_id]
    a["total"] = a.amt_in + a.amt_out
    a["role"] = np.where((a.amt_in > 0) & (a.amt_out > 0), "both ways", np.where(a.amt_in > 0, "pays in", "receives"))
    a["bank"] = None
    who = {"industry": "customers", "NAICS unknown": "customers", "outside PNC": "accounts"}
    a["hover"] = [f"<b>{r.label}</b><br>pays this industry {money(r.amt_in)} ({int(r.parties_in):,} {who[r.type]})"
                  f"<br>receives from it {money(r.amt_out)} ({int(r.parties_out):,} {who[r.type]})"
                  + ("<br><i>click, then draw its network below</i>" if r.key else "") for r in a.itertuples()]
    return a[a.total > 0].sort_values("total", ascending=False).reset_index(drop=True)[cols]


def industry_ego(code, key, desc=None, n_customers=None, next_key="_net_naics_next"):
    """The network block for one NAICS: every customer holding the code is the center. Other industries are clickable — the
    follow button redraws the page around the one picked (through session_state[next_key])."""
    end_month = latest_month(state.MODE)
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 2.5], vertical_alignment="bottom")
    n_m = c1.segmented_control("Months", [3, 6, 12], default=6, key=f"{key}_m", required=True)
    top = c2.segmented_control("Show top", [10, 20, 40], default=20, key=f"{key}_n", required=True)
    view = c3.segmented_control("View", ["Network map", "Flow diagram"], default="Network map", key=f"{key}_v", required=True)
    c4.caption(f"Payments over the {n_m} months to {end_month} between the customers with this NAICS and everyone else. Blue = "
               f"another industry (click to follow) · light blue = PNC customers without an industry code · grey = all accounts "
               f"outside PNC, as one node. Arrows point payer → payee: teal = money in, orange = money out, dotted = between the "
               f"other industries.")
    t1, t2, t3 = st.columns(3)
    show_ext = t1.toggle("Accounts outside PNC (one node)", value=True, key=f"{key}_ext")
    show_unk = t2.toggle("PNC customers without NAICS (one node)", value=True, key=f"{key}_unk")
    show_ties = t3.toggle("Payments between the other industries", value=True, key=f"{key}_ties")
    try:
        F = industry.flows(state.MODE, code, end_month, int(n_m))
    except Exception as e:
        st.error(f"The industry's payments could not be read ({type(e).__name__}: {str(e)[:200]})."); return
    if not len(F):
        ui.empty_state("No payments recorded for customers with this NAICS in this window."); return
    a = industry_frame(F, show_ext, show_unk)
    within = F[F.dir == "WITHIN"]
    center = ui.first_txt(desc, default=f"NAICS {code}")
    k = st.columns(6)
    k[0].metric("Money in", money(a.amt_in.sum()), help="Paid to customers with this NAICS by everyone outside it.")
    k[1].metric("Money out", money(a.amt_out.sum()), help="Paid by customers with this NAICS to everyone outside it.")
    k[2].metric("Within the industry", money(within.amount.sum()),
                help="Payments between customers with this NAICS (not drawn; it would be a loop on the center).")
    k[3].metric("Paying industries", f"{int(((a.type == 'industry') & (a.amt_in > 0)).sum()):,}")
    k[4].metric("Industries paid", f"{int(((a.type == 'industry') & (a.amt_out > 0)).sum()):,}")
    k[5].metric("Customers with this NAICS", f"{int(n_customers):,}" if n_customers is not None else "—",
                help="PNC customers whose valid NAICS is this code, active or not in the window.")
    if not len(a):
        ui.empty_state("Nothing left to draw with these options."); return
    codes = [c for c, t in zip(a.cpty_id.head(int(top)), a.type.head(int(top))) if t == "industry"]
    T = None
    if show_ties and len(codes) > 1:
        try:
            T = industry.ties(state.MODE, tuple(sorted(codes)), end_month, int(n_m))
        except Exception as e:
            st.caption(f"Payments between the other industries could not be read ({type(e).__name__}).")
    picked = []
    if view == "Flow diagram":
        st.plotly_chart(sankey(a, center, int(top), {}), key=f"{key}_sk", config={"displayModeBar": False})
    else:
        ev = st.plotly_chart(netmap(a, center, int(top), {}, T, [], subject="industry", layout="ring"), key=f"{key}_map", on_select="rerun",
                             selection_mode="points", config={"displayModeBar": False, "scrollZoom": True})
        try:
            picked = [p for p in (ev.selection.points if ev else []) if p.get("customdata") and p["customdata"][0]]
        except Exception:
            picked = []
    ind = a[a.type == "industry"]
    if len(ind):
        opts = {r.cpty_id: f"{r.label} · in {money(r.amt_in)} · out {money(r.amt_out)}" for r in ind.itertuples()}
        pre = picked[0]["customdata"][0].split(":", 1)[-1] if picked else None
        o1, o2 = st.columns([4, 1.2], vertical_alignment="bottom")
        sel = o1.selectbox("Follow the money to another industry", list(opts), format_func=opts.get,
                           index=list(opts).index(pre) if pre in opts else 0, key=f"{key}_follow")
        if o2.button("Draw its network →", key=f"{key}_go", type="primary", width="stretch"):
            st.session_state[next_key] = sel; st.rerun()
    try:
        M = industry.members(state.MODE, code, end_month, int(n_m))
    except Exception as e:
        M = None; st.caption(f"The industry's customers could not be read ({type(e).__name__}).")
    if M is not None and len(M):
        M = M.assign(key=[cust_key(p, m) for p, m in zip(M.customer_pwr_id, M.mdm_id)],
                     name=[ui.first_txt(n, default=f"Entity {m}") for n, m in zip(M.customer_name, M.mdm_id)])
        mo = {r.key: f"{r.name} · in {money(r.amt_in)} · out {money(r.amt_out)}" for r in M.itertuples() if r.key}
        m1, m2 = st.columns([4, 1.2], vertical_alignment="bottom")
        msel = m1.selectbox(f"Open one of its customers (top {len(mo):,} by payments)", list(mo), format_func=mo.get, key=f"{key}_mem")
        if m2.button("Open its page →", key=f"{key}_mgo", width="stretch"):
            nav.open_customer(msel, lens="network")
    with st.expander(f"All {len(a):,} counterpart groups", expanded=False):
        st.dataframe(a[["label", "type", "amt_in", "amt_out", "n_amt_in", "n_amt_out", "parties_in", "parties_out", "members_in", "members_out"]],
                     hide_index=True, width="stretch",
                     column_config={"label": "industry / group", "amt_in": st.column_config.NumberColumn("money in", format="dollar"),
                                    "amt_out": st.column_config.NumberColumn("money out", format="dollar"),
                                    "n_amt_in": st.column_config.NumberColumn("payments in", format="%d"),
                                    "n_amt_out": st.column_config.NumberColumn("payments out", format="%d"),
                                    "parties_in": st.column_config.NumberColumn("payers", format="%d"),
                                    "parties_out": st.column_config.NumberColumn("payees", format="%d"),
                                    "members_in": st.column_config.NumberColumn("its customers paid", format="%d"),
                                    "members_out": st.column_config.NumberColumn("its customers paying", format="%d")})
    if M is not None and len(M):
        with st.expander(f"Its customers (top {len(M):,})", expanded=False):
            st.dataframe(M[["name", "mdm_id", "customer_pwr_id", "amt_in", "amt_out", "from_members", "to_members"]], hide_index=True,
                         width="stretch",
                         column_config={"name": "customer", "mdm_id": "payment entity", "customer_pwr_id": "power id",
                                        "amt_in": st.column_config.NumberColumn("money in", format="dollar"),
                                        "amt_out": st.column_config.NumberColumn("money out", format="dollar"),
                                        "from_members": st.column_config.NumberColumn("from the same industry", format="dollar"),
                                        "to_members": st.column_config.NumberColumn("to the same industry", format="dollar")})
    st.caption("Industry = the NAICS code of PNC customers in the customer dimension (valid codes only). Accounts outside PNC "
               "have no known industry, so they form one node. Payer and payee counts are approximate for very large groups.")
