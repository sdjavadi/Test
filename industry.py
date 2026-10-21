"""Industry (NAICS) payment network: which industries pay a chosen NAICS and which ones it pays, built from the monthly edges
(`pkg_edge_monthly`) and the customer dimension (`pkg_cust_dim`, valid NAICS only). Every PNC customer with the chosen code
is a member of the target; the other side of each payment is grouped as
  * another NAICS code (PNC customers with a valid code),
  * UNKNOWN   — PNC customers without a valid NAICS (placeholder or missing),
  * EXTERNAL  — accounts outside PNC (no line of business is known for them).
Same function names in both modes: Impala reads the tables through the source's dbi helper, demo mode derives the same frames
from the synthetic world. Results are cached by their arguments, so changing display options never re-queries."""
import re

import numpy as np
import pandas as pd
import streamlit as st

from . import state
from .data import ids_sql, ym, ym_add

CODE_RE = re.compile(r"^\d{2,6}$")
GROUPS = {"UNKNOWN": "PNC customers without NAICS", "EXTERNAL": "Counterparties outside PNC"}
FLOW_COLS = ["dir", "grp", "grp_desc", "amount", "n_txn", "n_members", "n_parties"]
TIE_COLS = ["src_id", "dst_id", "amount", "n_txn"]
MEMBER_COLS = ["mdm_id", "customer_pwr_id", "customer_name", "amt_in", "amt_out", "from_members", "to_members"]


def _code(c):
    c = str(c).strip()
    if not CODE_RE.match(c): raise ValueError(f"bad NAICS code {c!r}")
    return c


def _num(d, cols):
    for c in cols:
        if c in d: d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    return d


# ── Impala ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _target_edges(src, code, end_month, n_months):
    """CTEs tg (the NAICS's customers) and te (every monthly edge touching one of them, marked with which end(s) are
    members). Self-transfers are left out, as in the client view."""
    dim, edge = src.t("cust_dim_table"), src.t("edge_table")
    m1 = ym(end_month); m0 = ym_add(m1, -(int(n_months) - 1))
    return f"""tg AS (SELECT DISTINCT mdm_id FROM {dim} WHERE naics_status = 'valid' AND naics_cd = '{_code(code)}'),
te AS (
    SELECT g.src_id, g.src_kind, g.dst_id, g.dst_kind, CAST(g.amount AS DOUBLE) AS amount, CAST(g.n_txn AS DOUBLE) AS n_txn,
           tp.mdm_id AS t_src, tr.mdm_id AS t_dst
    FROM {edge} g
    LEFT JOIN tg tp ON g.src_id = tp.mdm_id AND g.src_kind = 'CUST'
    LEFT JOIN tg tr ON g.dst_id = tr.mdm_id AND g.dst_kind = 'CUST'
    WHERE g.month BETWEEN '{m0}' AND '{m1}' AND NOT COALESCE(g.is_self, FALSE) AND g.src_id <> g.dst_id
      AND (tp.mdm_id IS NOT NULL OR tr.mdm_id IS NOT NULL))"""


def _impala_list(src):
    return src.q(f"SELECT naics_cd, MAX(naics_desc) AS naics_desc, COUNT(DISTINCT mdm_id) AS n_customers "
                 f"FROM {src.t('cust_dim_table')} WHERE naics_status = 'valid' AND naics_cd IS NOT NULL GROUP BY naics_cd")


def _impala_flows(src, code, end_month, n_months):
    dim = src.t("cust_dim_table")
    return src.q(f"""WITH {_target_edges(src, code, end_month, n_months)},
od AS (SELECT mdm_id, MAX(naics_cd) AS naics_cd, MAX(naics_desc) AS naics_desc FROM {dim} WHERE naics_status = 'valid' GROUP BY mdm_id),
x AS (
    SELECT CASE WHEN t_src IS NOT NULL AND t_dst IS NOT NULL THEN 'WITHIN' WHEN t_src IS NOT NULL THEN 'OUT' ELSE 'IN' END AS dir,
           COALESCE(t_src, t_dst) AS mem_id,
           CASE WHEN t_src IS NOT NULL THEN dst_id ELSE src_id END AS other_id,
           CASE WHEN t_src IS NOT NULL THEN dst_kind ELSE src_kind END AS other_kind, amount, n_txn
    FROM te)
SELECT x.dir,
       CASE WHEN x.dir = 'WITHIN' THEN 'TARGET' WHEN x.other_kind <> 'CUST' THEN 'EXTERNAL'
            WHEN od.mdm_id IS NULL THEN 'UNKNOWN' ELSE od.naics_cd END AS grp,
       MAX(od.naics_desc) AS grp_desc, SUM(x.amount) AS amount, SUM(x.n_txn) AS n_txn,
       NDV(x.mem_id) AS n_members, NDV(x.other_id) AS n_parties
FROM x LEFT JOIN od ON x.other_id = od.mdm_id AND x.other_kind = 'CUST'
GROUP BY 1, 2""")


def _impala_ties(src, codes, end_month, n_months):
    """Payments between the given other industries (both ends PNC customers with those codes, different codes)."""
    codes = [_code(c) for c in codes]
    if len(codes) < 2: return pd.DataFrame(columns=TIE_COLS)
    dim, edge = src.t("cust_dim_table"), src.t("edge_table")
    m1 = ym(end_month); m0 = ym_add(m1, -(int(n_months) - 1))
    return src.q(f"""WITH od AS (SELECT mdm_id, MAX(naics_cd) AS naics_cd FROM {dim}
            WHERE naics_status = 'valid' AND naics_cd IN ({ids_sql(codes)}) GROUP BY mdm_id)
SELECT a.naics_cd AS src_id, b.naics_cd AS dst_id, SUM(CAST(e.amount AS DOUBLE)) AS amount, SUM(e.n_txn) AS n_txn
FROM {edge} e JOIN od a ON e.src_id = a.mdm_id JOIN od b ON e.dst_id = b.mdm_id
WHERE e.month BETWEEN '{m0}' AND '{m1}' AND e.src_kind = 'CUST' AND e.dst_kind = 'CUST'
  AND NOT COALESCE(e.is_self, FALSE) AND a.naics_cd <> b.naics_cd
GROUP BY 1, 2""")


def _impala_members(src, code, end_month, n_months, limit):
    dim = src.t("cust_dim_table")
    return src.q(f"""WITH {_target_edges(src, code, end_month, n_months)},
u AS (
    SELECT t_dst AS mdm_id, CASE WHEN t_src IS NULL THEN amount ELSE 0 END AS amt_in, CAST(0 AS DOUBLE) AS amt_out,
           CASE WHEN t_src IS NOT NULL THEN amount ELSE 0 END AS from_members, CAST(0 AS DOUBLE) AS to_members
    FROM te WHERE t_dst IS NOT NULL
    UNION ALL
    SELECT t_src AS mdm_id, CAST(0 AS DOUBLE) AS amt_in, CASE WHEN t_dst IS NULL THEN amount ELSE 0 END AS amt_out,
           CAST(0 AS DOUBLE) AS from_members, CASE WHEN t_dst IS NOT NULL THEN amount ELSE 0 END AS to_members
    FROM te WHERE t_src IS NOT NULL),
m AS (SELECT mdm_id, SUM(amt_in) AS amt_in, SUM(amt_out) AS amt_out, SUM(from_members) AS from_members,
             SUM(to_members) AS to_members FROM u GROUP BY mdm_id),
d AS (SELECT mdm_id, MAX(customer_pwr_id) AS customer_pwr_id, MAX(customer_name) AS customer_name FROM {dim} GROUP BY mdm_id)
SELECT m.mdm_id, d.customer_pwr_id, d.customer_name, m.amt_in, m.amt_out, m.from_members, m.to_members
FROM m LEFT JOIN d ON m.mdm_id = d.mdm_id
ORDER BY m.amt_in + m.amt_out + m.from_members + m.to_members DESC LIMIT {int(limit)}""")


# ── demo: the same frames from the synthetic world ──────────────────────────────────────────────────────────────────────
def _demo_naics(D):
    h = D.H[D.H.t == D.M - 1]
    return dict(zip(h.cust_pwr_id, h.naics_cd)), dict(zip(h.naics_cd, h.naics_desc)), h


def _demo_window(D, end_month, n_months):
    e = D.all_months.index(end_month) if end_month in D.all_months else D.M - 1
    return list(range(max(0, e - int(n_months) + 1), e + 1))


def _demo_list(D):
    _, _, h = _demo_naics(D)
    return (h.groupby("naics_cd").agg(naics_desc=("naics_desc", "first"), n_customers=("mdm_id", "nunique")).reset_index())


def _demo_rows(D, code, end_month, n_months):
    """Member-side rows: (member, direction, other id, other kind, other customer, amount, n_txn). A payment between two members
    is counted once, from the payer's side."""
    nc, _, _ = _demo_naics(D); ts = _demo_window(D, end_month, n_months)
    rows = []
    for c in [c for c, k in nc.items() if str(k) == code]:
        inn, out = D._flows(c, ts)
        for d, f in (("IN", inn), ("OUT", out)):
            for r in f.itertuples():
                oc = D.cust_of.get(r.cpty_id) if r.kind == "CUST" else None
                if oc == c: continue
                inside = oc is not None and str(nc.get(oc)) == code
                if inside and d == "IN": continue                       # counted from the payer's side
                rows.append(dict(member=D.mdm[c][0], dir="WITHIN" if inside else d, other_id=r.cpty_id, other_cust=oc,
                                 amount=float(r.amount), n_txn=float(r.n_txn)))
    return pd.DataFrame(rows, columns=["member", "dir", "other_id", "other_cust", "amount", "n_txn"]), nc


def _demo_flows(D, code, end_month, n_months):
    R, nc = _demo_rows(D, code, end_month, n_months)
    if not len(R): return pd.DataFrame(columns=FLOW_COLS)
    _, desc, _ = _demo_naics(D)
    R["grp"] = np.where(R.dir == "WITHIN", "TARGET", [str(nc.get(o)) if o is not None and nc.get(o) is not None
                                                      else ("UNKNOWN" if o is not None else "EXTERNAL") for o in R.other_cust])
    R["grp_desc"] = R.grp.map(desc)
    return (R.groupby(["dir", "grp"], as_index=False)
             .agg(grp_desc=("grp_desc", "first"), amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                  n_members=("member", "nunique"), n_parties=("other_id", "nunique")))


def _demo_ties(D, codes, end_month, n_months):
    nc, _, _ = _demo_naics(D); ts = _demo_window(D, end_month, n_months); codes = set(map(str, codes)); rows = []
    for c in [c for c, k in nc.items() if str(k) in codes]:
        _, out = D._flows(c, ts)
        for r in out[out.kind == "CUST"].itertuples() if len(out) else []:
            oc = D.cust_of.get(r.cpty_id)
            if oc is not None and str(nc.get(oc)) in codes and nc.get(oc) != nc.get(c):
                rows.append(dict(src_id=str(nc[c]), dst_id=str(nc[oc]), amount=float(r.amount), n_txn=float(r.n_txn)))
    d = pd.DataFrame(rows, columns=TIE_COLS)
    return d.groupby(["src_id", "dst_id"], as_index=False)[["amount", "n_txn"]].sum() if len(d) else d


def _demo_members(D, code, end_month, n_months, limit):
    R, _ = _demo_rows(D, code, end_month, n_months)
    if not len(R): return pd.DataFrame(columns=MEMBER_COLS)
    mem_of = {D.mdm[c][0]: c for c in D.ids}
    parts = [R[R.dir == "IN"].assign(mdm_id=R.member, amt_in=R.amount),
             R[R.dir == "OUT"].assign(mdm_id=R.member, amt_out=R.amount),
             R[R.dir == "WITHIN"].assign(mdm_id=R.member, to_members=R.amount),
             R[R.dir == "WITHIN"].assign(mdm_id=[D.mdm[o][0] for o in R.loc[R.dir == "WITHIN", "other_cust"]], from_members=R.amount)]
    m = pd.concat(parts, ignore_index=True)
    m = _num(m, ["amt_in", "amt_out", "from_members", "to_members"]).groupby("mdm_id", as_index=False)[
        ["amt_in", "amt_out", "from_members", "to_members"]].sum()
    h = D.H[D.H.t == D.M - 1].set_index("mdm_id")
    m["customer_pwr_id"] = m.mdm_id.map(lambda x: mem_of[x].lstrip("0"))
    m["customer_name"] = m.mdm_id.map(h.customer_name)
    m["tot"] = m[["amt_in", "amt_out", "from_members", "to_members"]].sum(axis=1)
    return m.sort_values("tot", ascending=False).head(int(limit))[MEMBER_COLS]


# ── public, cached ──────────────────────────────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner="Reading industries…")
def naics_list(mode):
    """All valid NAICS codes with a description and the number of customers holding them."""
    src = state.source()
    d = _demo_list(src) if mode == "demo" else _impala_list(src)
    d["naics_cd"] = d.naics_cd.astype(str).str.strip()
    d = d[d.naics_cd.str.match(CODE_RE)]
    d["n_customers"] = pd.to_numeric(d.n_customers, errors="coerce").fillna(0).astype(int)
    return d.sort_values("naics_cd").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner="Reading payments of the industry…")
def flows(mode, code, end_month, n_months):
    """One row per (direction, group): IN = paid to the industry from outside it, OUT = paid by it, WITHIN = between its
    own customers (grp = TARGET)."""
    src = state.source()
    d = _demo_flows(src, code, end_month, n_months) if mode == "demo" else _impala_flows(src, code, end_month, n_months)
    d = _num(d.copy(), ["amount", "n_txn", "n_members", "n_parties"])
    d["grp"] = d.grp.astype(str)
    return d[FLOW_COLS]


@st.cache_data(ttl=3600, show_spinner=False)
def ties(mode, codes, end_month, n_months):
    src = state.source()
    d = _demo_ties(src, codes, end_month, n_months) if mode == "demo" else _impala_ties(src, codes, end_month, n_months)
    d = _num(d.copy(), ["amount", "n_txn"])
    d["src_id"] = d.src_id.astype(str); d["dst_id"] = d.dst_id.astype(str)
    return d[TIE_COLS]


@st.cache_data(ttl=3600, show_spinner=False)
def members(mode, code, end_month, n_months, limit=300):
    """The industry's most active customers in the window (money in and out, and with each other)."""
    src = state.source()
    d = _demo_members(src, code, end_month, n_months, limit) if mode == "demo" else _impala_members(src, code, end_month, n_months, limit)
    d = _num(d.copy(), ["amt_in", "amt_out", "from_members", "to_members"])
    d["mdm_id"] = d.mdm_id.astype(str)
    return d[MEMBER_COLS]
