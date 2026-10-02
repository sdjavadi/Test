"""
PKG · Ego Network Explorer (Streamlit)

    streamlit run pkg_ego_app.py

Look up a PNC deposit account or an MDM id and see its ego network from the payment staging
table: every PNC customer and external counterparty that paid it or was paid by it, the payments
between those, and summaries of both ends (rails, time, banks, concentration).

Data comes from Impala through `dbi.db_get_query` (pandas out). Three queries per lookup, each
cached by its SQL text, so the display options never re-query:

    Q1  ego cube       rows touching the target, grouped to (month, rail, both ends)
    Q2  amount histo   target rows bucketed by log10(amount), for transaction-size percentiles
    Q3  ties           payments between the target's top connected parties, target excluded

Identity rules carried over from the PKG edge build:
  * a side is a PNC customer when it has an mdm_id or a PNC deposit account; otherwise external
  * counterparty id = unq_cpty_id for 3B/3C/7B/9B (unq_cpty_acct_id is only a routing number or
    null there), unq_cpty_acct_id elsewhere, the other as fallback
  * 6C twins listed in pkg_dedup_5c6c are removed

The network is a small custom component (vis-network, the library pyvis wraps) so that clicking a
PNC customer can send its account / MDM id back to the lookup field.
"""
import glob
import hashlib
import inspect
import io
import json
import os
import re
import tempfile
import time
import traceback
import zipfile
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import dbi

# =============================================================================
# CONFIGURATION
# =============================================================================
PAYMENTS_TABLE = "dsihd01p_dsi.neo4j_payments"
DEDUP_TABLE = "bdahd01p_dlcdi1_cdi_tm.pkg_dedup_5c6c"
DATA_START = date(2023, 7, 1)            # first trans_dt partition
DEFAULT_WINDOW_MONTHS = 3                # each lookup scans the window twice; widen deliberately
TIE_SEARCH_CAP = 500                     # per side: top PNC customers and top counterparties by $
MAX_VIS_DEFAULT = 150
NET_HEIGHT_PX = 780
CACHE_TTL_S = 6 * 3600


def get_query(sql):
    df = dbi.db_get_query(
        sql,
        dsn="DSN=bdpimp04-impala;",
        pool="root.CIB-AMG_Impala",
        conn_options={"SocketTimeout": 0},
    )
    return df


# ---- payment rails ---------------------------------------------------------------------------
# value of payment_rail -> (display label, category numbers carrying it). The rail filter is applied
# on the category number because category is a partition column (Impala prunes on it) and
# payment_rail is not; the mapping is 1:1 (1-3 ACH, 4 wire, 5 RTP PRT, 6 RTP P2P, 7 check,
# 8 debit card, 9 PCARD).
RAIL_DEF = {
    "ACH": ("ACH", "123"),
    "WIRE": ("Wire", "4"),
    "RTP_PRT": ("RTP (PRT)", "5"),
    "RTP_P2P": ("RTP (P2P)", "6"),
    "CHECK": ("Check", "7"),
    "DEBIT_CARD_SIGNATURE": ("Debit card", "8"),
    "PCARD": ("PCARD", "9"),
}
# Debit card is ~42% of rows and ~0.5% of dollars, and its merchants flood the picture: off by
# default, one click to add back.
DEFAULT_RAILS = [r for r in RAIL_DEF if r != "DEBIT_CARD_SIGNATURE"]


def rail_label(r):
    return RAIL_DEF.get(r, (r if isinstance(r, str) and r else "Unknown",))[0]


CODE_EXPR = "regexp_extract(p.category, '^([0-9]+[A-Z][0-9]*)', 1)"
CPTY_ID_EXPR = (f"CASE WHEN {CODE_EXPR} IN ('3B','3C','7B','9B') "
                "THEN COALESCE(NULLIF(p.unq_cpty_id, ''), NULLIF(p.unq_cpty_acct_id, '')) "
                "ELSE COALESCE(NULLIF(p.unq_cpty_acct_id, ''), NULLIF(p.unq_cpty_id, '')) END")

# ---- visual vocabulary -----------------------------------------------------------------------
C_EGO, C_INT, C_EXT, C_EDGE = "#F58025", "#1F5AA6", "#2E9E6B", "#8A8A8A"
TYPE_LABEL = {"internal": "PNC customer", "external": "Counterparty"}
TYPE_COLOR = {TYPE_LABEL["internal"]: C_INT, TYPE_LABEL["external"]: C_EXT}
DIR_LABEL = {"IN": "Received", "OUT": "Sent"}
RAIL_COLORS = {"ACH": "#1F77B4", "Wire": "#D62728", "RTP (PRT)": "#9467BD", "RTP (P2P)": "#C5B0D5",
               "Check": "#8C564B", "Debit card": "#E377C2", "PCARD": "#17BECF", "Unknown": "#BBBBBB"}
_SPARE = ["#BCBD22", "#FF7F0E", "#AEC7E8", "#98DF8A", "#FFBB78", "#9EDAE5"]

MODES = ["Deposit account", "MDM id"]
ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
ID_COLS = ["mdm_id_pays", "mdm_id_receives", "pnc_dep_acct_pays", "pnc_dep_acct_receives", "cpty_id"]


def rail_color(r):
    if r not in RAIL_COLORS:
        RAIL_COLORS[r] = next((c for c in _SPARE if c not in RAIL_COLORS.values()), "#999999")
    return RAIL_COLORS[r]


def rail_scale(rails):
    rails = list(rails)
    return alt.Scale(domain=rails, range=[rail_color(r) for r in rails])


MONEY_AXIS = alt.Axis(labelExpr="replace(format(datum.value, '$~s'), 'G', 'B')")


def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    a, s = abs(x), "-" if x < 0 else ""
    for d, u in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= d:
            return f"{s}${a / d:,.1f}{u}"
    return f"{s}${a:,.0f}"


def _safe(s):
    """Names are data: keep '<' and '>' out of anything that ends up inside the page's script."""
    return re.sub(r"[<>]", " ", "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s))


def trunc(s, n=28):
    s = _safe(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _wide(fn):
    """Full-width kwarg for this Streamlit version. Releases that deprecate use_container_width keep
    it as a None-default shim and take width='stretch'; older releases default it to False."""
    p = inspect.signature(fn).parameters.get("use_container_width")
    if p is None:
        return {}
    return {"width": "stretch"} if p.default is None else {"use_container_width": True}


def _rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


# =============================================================================
# SQL
# =============================================================================
def _q(v):
    return "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _in(vals):
    return ", ".join(_q(v) for v in vals)


def _month_shift(d, k):
    m = d.year * 12 + d.month - 1 + k
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


def _scope(P):
    """FROM/JOIN + WHERE shared by every query: window, rails, removal of listed 6C twins."""
    where = [f"p.trans_dt BETWEEN {_q(P['start'])} AND {_q(P['end'])}"]
    if set(P["rails"]) != set(RAIL_DEF):
        fams = sorted("".join(RAIL_DEF[r][1] for r in P["rails"]))
        where.append(f"substr(p.category, 1, 1) IN ({_in(fams)})")
    s, e = date.fromisoformat(P["start"]), date.fromisoformat(P["end"])
    join = (f"LEFT ANTI JOIN (SELECT trans_id_drop FROM {DEDUP_TABLE} "
            f"WHERE `month` BETWEEN '{_month_shift(s, -1)}' AND '{_month_shift(e, 1)}') d "
            f"ON p.trans_id = d.trans_id_drop")
    return f"FROM {PAYMENTS_TABLE} p {join}", where


def _ego_cols(P):
    return (("p.pnc_dep_acct_pays", "p.pnc_dep_acct_receives") if P["mode"] == "acct"
            else ("p.mdm_id_pays", "p.mdm_id_receives"))


_INNER = f"""
    SELECT p.trans_dt, CAST(p.trans_amt AS DOUBLE) AS amt,
           COALESCE(NULLIF(p.payment_rail, ''), 'UNKNOWN') AS pay_rail, p.cpty_type,
           NULLIF(p.mdm_id_pays, '') AS mdm_id_pays, NULLIF(p.mdm_id_receives, '') AS mdm_id_receives,
           NULLIF(p.pnc_dep_acct_pays, '') AS pnc_dep_acct_pays,
           NULLIF(p.pnc_dep_acct_receives, '') AS pnc_dep_acct_receives,
           p.customer_name_pays, p.customer_name_receives, p.cpty_name, p.cpty_fin_entity_name,
           {CPTY_ID_EXPR} AS cpty_id"""

_CUBE_SELECT = """
SELECT {month} b.pay_rail, b.cpty_type,
       b.mdm_id_pays, b.mdm_id_receives, b.pnc_dep_acct_pays, b.pnc_dep_acct_receives, b.cpty_id,
       MAX(b.customer_name_pays) AS name_pays, MAX(b.customer_name_receives) AS name_receives,
       MAX(b.cpty_name) AS cpty_name, MAX(b.cpty_fin_entity_name) AS cpty_bank,
       SUM(b.amt) AS amount, COUNT(*) AS n_txn, MIN(b.trans_dt) AS first_dt, MAX(b.trans_dt) AS last_dt"""


def sql_ego_cube(P):
    frm, where = _scope(P)
    cp, cr = _ego_cols(P)
    where.append(f"({cp} = {_q(P['id'])} OR {cr} = {_q(P['id'])})")
    return (_CUBE_SELECT.format(month="substr(b.trans_dt, 1, 7) AS txn_month,")
            + f"\nFROM ({_INNER}\n    {frm}\n    WHERE " + "\n      AND ".join(where) + "\n) b"
            + "\nGROUP BY 1, 2, 3, 4, 5, 6, 7, 8")


def sql_amount_hist(P):
    """20 log-buckets per decade (bucket width ×1.12) -> percentiles good to about ±6%.
    Impala has no general percentile function; the histogram also feeds the size chart.
    other_pnc lets the 'counterparties off' view drop external transactions here too."""
    frm, where = _scope(P)
    cp, cr = _ego_cols(P)
    x = _q(P["id"])
    where.append(f"({cp} = {x} OR {cr} = {x})")
    return f"""
SELECT x.dir, x.pay_rail, x.other_pnc, x.amt_bucket, COUNT(*) AS n_txn, SUM(x.amt) AS amount
FROM (
    SELECT CASE WHEN {cp} = {x} AND {cr} = {x} THEN 'SELF' WHEN {cp} = {x} THEN 'OUT'
                WHEN {cr} = {x} THEN 'IN' ELSE 'OTHER' END AS dir,
           CASE WHEN {cp} = {x}
                THEN (NULLIF(p.mdm_id_receives, '') IS NOT NULL OR NULLIF(p.pnc_dep_acct_receives, '') IS NOT NULL)
                ELSE (NULLIF(p.mdm_id_pays, '') IS NOT NULL OR NULLIF(p.pnc_dep_acct_pays, '') IS NOT NULL) END AS other_pnc,
           COALESCE(NULLIF(p.payment_rail, ''), 'UNKNOWN') AS pay_rail,
           CAST(p.trans_amt AS DOUBLE) AS amt,
           CAST(FLOOR(LOG10(GREATEST(CAST(p.trans_amt AS DOUBLE), 0.01)) * 20) AS INT) AS amt_bucket
    {frm}
    WHERE {' AND '.join(where)}
) x
GROUP BY 1, 2, 3, 4"""


def sql_ties(P, int_ids, ext_ids):
    """Payments between connected parties. Every observable tie has a PNC end (external->external
    is not in the table), so the scan is pre-filtered on the PNC parties; the outer filter keeps
    rows whose other end is also a connected party. The target's own rows are excluded."""
    frm, where = _scope(P)
    ip, ir = (("pnc_dep_acct_pays", "pnc_dep_acct_receives") if P["level"] == "account"
              else ("mdm_id_pays", "mdm_id_receives"))
    cp, cr = _ego_cols(P)
    I = _in(int_ids)
    where += [f"(p.{ip} IN ({I}) OR p.{ir} IN ({I}))",
              f"COALESCE({cp}, '') <> {_q(P['id'])}", f"COALESCE({cr}, '') <> {_q(P['id'])}"]
    keep = [f"(b.{ip} IN ({I}) AND b.{ir} IN ({I}))"]
    if ext_ids:
        E = _in(ext_ids)
        keep += [f"(b.{ip} IN ({I}) AND b.mdm_id_receives IS NULL AND b.cpty_id IN ({E}))",
                 f"(b.{ir} IN ({I}) AND b.mdm_id_pays IS NULL AND b.cpty_id IN ({E}))"]
    return (_CUBE_SELECT.format(month="")
            + f"\nFROM ({_INNER}\n    {frm}\n    WHERE " + "\n      AND ".join(where) + "\n) b"
            + "\nWHERE " + "\n   OR ".join(keep)
            + "\nGROUP BY 1, 2, 3, 4, 5, 6, 7")


def sql_near_match(P):
    """Zero rows is usually formatting (leading zeros, wrong id type), not an unused id.
    One scan over the window, all rails, on every id column."""
    t = _q(P["id"].strip().lstrip("0"))
    cols = ["pnc_dep_acct_pays", "pnc_dep_acct_receives", "mdm_id_pays", "mdm_id_receives",
            "unq_cpty_acct_id", "unq_cpty_id"]
    norm = lambda c: f"regexp_replace(trim(p.{c}), '^0+', '')"
    which = "CASE " + " ".join(f"WHEN {norm(c)} = {t} THEN '{c}'" for c in cols) + " END"
    stored = "CASE " + " ".join(f"WHEN {norm(c)} = {t} THEN p.{c}" for c in cols) + " END"
    return f"""
SELECT x.id_column, x.stored_value, COUNT(*) AS n_txn FROM (
    SELECT {which} AS id_column, {stored} AS stored_value
    FROM {PAYMENTS_TABLE} p
    WHERE p.trans_dt BETWEEN {_q(P['start'])} AND {_q(P['end'])}
      AND ({' OR '.join(f'{norm(c)} = {t}' for c in cols)})
) x GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20"""


@st.cache_data(ttl=CACHE_TTL_S, max_entries=64, show_spinner=False)
def run_sql(sql):
    t0 = time.time()
    df = get_query(sql)
    df.columns = [str(c).lower().split(".")[-1] for c in df.columns]
    return df, time.time() - t0


# =============================================================================
# TRANSFORM (pandas, on the small query results)
# =============================================================================
def _clean_ids(df):
    for c in ID_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object").map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                                               or str(v).strip() == "" else str(v))
    for c in ["amount", "n_txn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if "pay_rail" in df.columns:
        df["rail"] = df["pay_rail"].map(rail_label)
    return df


def _prefixed(prefix, s):
    return s.map(lambda v: None if v is None else prefix + v)


def node_keys(is_pnc, mdm, acct, cpty, level):
    """Node key of one end. PNC end: 'PNC:<acct>' (account level) or 'MDM:<mdm>' (customer level),
    each falling back to the other id when missing. Counterparty end: 'EXT:<cpty_id>' or None."""
    if level == "account":
        k_int = _prefixed("PNC:", acct).where(acct.notna(), _prefixed("MDM:", mdm))
    else:
        k_int = _prefixed("MDM:", mdm).where(mdm.notna(), _prefixed("PNC:", acct))
    return k_int.where(is_pnc, _prefixed("EXT:", cpty))


def lookup_of(key):
    """Node key -> (lookup mode, id) for click-to-lookup; None for counterparties."""
    if key.startswith("PNC:"):
        return "acct", key[4:]
    if key.startswith("MDM:"):
        return "mdm", key[4:]
    return None


def ego_key(P):
    return ("PNC:" if P["mode"] == "acct" else "MDM:") + P["id"]


def orient(cube, P):
    """Raw ego cube -> target-centric rows: direction, the other end's key and attributes."""
    c = _clean_ids(cube.copy())
    if P["mode"] == "acct":
        out, inn = c["pnc_dep_acct_pays"].eq(P["id"]), c["pnc_dep_acct_receives"].eq(P["id"])
    else:
        out, inn = c["mdm_id_pays"].eq(P["id"]), c["mdm_id_receives"].eq(P["id"])
    c["direction"] = np.select([out & inn, out, inn], ["SELF", "OUT", "IN"], "OTHER")
    other_is_rec = c["direction"].ne("IN")          # for OUT the other end is the receiver

    def other(pay_col, rec_col):
        return c[rec_col].where(other_is_rec, c[pay_col])

    def mine(pay_col, rec_col):
        return c[pay_col].where(other_is_rec, c[rec_col])

    c["cp_mdm"], c["cp_pnc_acct"] = other("mdm_id_pays", "mdm_id_receives"), other("pnc_dep_acct_pays", "pnc_dep_acct_receives")
    c["ego_mdm"], c["ego_acct"] = mine("mdm_id_pays", "mdm_id_receives"), mine("pnc_dep_acct_pays", "pnc_dep_acct_receives")
    c["ego_name"] = mine("name_pays", "name_receives")
    is_pnc = c["cp_mdm"].notna() | c["cp_pnc_acct"].notna()
    c["cp_type"] = np.where(is_pnc, "internal", "external")
    c["cp_key"] = node_keys(is_pnc, c["cp_mdm"], c["cp_pnc_acct"], c["cpty_id"], P["level"])
    c["cp_name"] = other("name_pays", "name_receives").where(is_pnc, c["cpty_name"])
    c["cp_bank"] = c["cpty_bank"].where(~is_pnc, "PNC")
    c["cp_id"] = (c["cp_pnc_acct"] if P["level"] == "account" else c["cp_mdm"]).where(is_pnc, c["cpty_id"])
    return c


def _mode_by(df, key, col, weight="n_txn"):
    d = df[df[col].notna()].groupby([key, col], as_index=False)[weight].sum()
    if d.empty:
        return pd.Series(dtype=object)
    return d.sort_values([key, weight], ascending=[True, False]).drop_duplicates(key).set_index(key)[col]


def hist_percentiles(h, by, qs=(0.10, 0.50, 0.90, 0.99)):
    """Percentiles from the log-bucket histogram (geometric bucket midpoint)."""
    out = []
    for keys, g in h.groupby(by):
        g = g.groupby("amt_bucket", as_index=False)["n_txn"].sum().sort_values("amt_bucket")
        cum = g["n_txn"].cumsum() / g["n_txn"].sum()
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        for q in qs:
            b = g.loc[cum >= q, "amt_bucket"].iloc[0]
            row[f"p{int(q * 100)}"] = 10 ** ((b + 0.5) / 20)
        out.append(row)
    return pd.DataFrame(out)


def _bool(s):
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "t", "yes"))


def build_results(P, cube, hist, ties_raw, include_cp=True):
    R = {"P": P, "ego_key": ego_key(P), "include_cp": include_cp}
    c = orient(cube, P)
    no_cp = c["cp_key"].isna()
    R["qa"] = (c.assign(n_no_cp=np.where(no_cp, c["n_txn"], 0), amt_no_cp=np.where(no_cp, c["amount"], 0.0))
                 .groupby("direction").agg(n_txn=("n_txn", "sum"), amount=("amount", "sum"),
                                           n_txn_no_cpty_id=("n_no_cp", "sum"), amount_no_cpty_id=("amt_no_cp", "sum")))
    R["self"] = c[c["direction"] == "SELF"]
    ex = c[c["direction"].isin(["IN", "OUT"]) & c["cp_key"].notna()].copy()
    if not include_cp:
        ex = ex[ex["cp_type"] == "internal"]
    if ex.empty:
        raise LookupError("No payments with " + ("other parties" if include_cp else "other PNC customers")
                          + " in this window and rail selection.")
    ex["type"] = ex["cp_type"].map(TYPE_LABEL)
    R["ex"] = ex
    R["ego_names"] = sorted({_safe(v) for v in c["ego_name"].dropna()})
    R["ego_mdms"] = sorted(c["ego_mdm"].dropna().unique().tolist())
    if P["mode"] == "mdm":   # every account on the customer's side, incl. ones seen only in own transfers
        own = pd.concat([c.loc[c["mdm_id_pays"].eq(P["id"]), "pnc_dep_acct_pays"],
                         c.loc[c["mdm_id_receives"].eq(P["id"]), "pnc_dep_acct_receives"]])
    else:
        own = pd.Series([P["id"]])
    R["ego_accts"] = sorted(own.dropna().unique().tolist())

    # ---- per connected party -------------------------------------------------------------------
    R["cp_dir"] = ex.groupby(["direction", "cp_key"], as_index=False).agg(
        amount=("amount", "sum"), n_txn=("n_txn", "sum"), first_dt=("first_dt", "min"),
        last_dt=("last_dt", "max"), months_active=("txn_month", "nunique"))
    R["cp_rail"] = ex.groupby(["direction", "cp_key", "rail"], as_index=False)[["amount", "n_txn"]].sum()
    attr = ex.groupby("cp_key").agg(cp_type=("cp_type", "first"), cp_mdm=("cp_mdm", "first"),
                                    cp_id=("cp_id", "first"), n_pnc_accts=("cp_pnc_acct", "nunique"))
    attr["cp_name"] = _mode_by(ex, "cp_key", "cp_name").map(_safe)
    attr["cp_bank"] = _mode_by(ex, "cp_key", "cp_bank").map(_safe)
    attr["cpty_type"] = _mode_by(ex, "cp_key", "cpty_type")
    attr["rails"] = ex.groupby("cp_key")["rail"].agg(lambda s: ", ".join(sorted(set(s.dropna()))))
    attr["same_customer"] = (attr["cp_type"] == "internal") & attr["cp_mdm"].isin(R["ego_mdms"])
    attr["label"] = attr["cp_name"].where(attr["cp_name"].fillna("").str.len() > 0,
                                          "(" + attr["cpty_type"].fillna("no name").astype(str) + " "
                                          + attr["cp_id"].astype(str).str[-6:] + ")")
    R["attr"] = attr

    flow = R["cp_dir"].pivot_table(index="cp_key", columns="direction", values="amount",
                                   aggfunc="sum", fill_value=0.0).reindex(columns=["IN", "OUT"], fill_value=0.0)
    flow["total"] = flow["IN"] + flow["OUT"]
    R["flow"] = flow.sort_values("total", ascending=False)

    # ---- slices for the summaries ----------------------------------------------------------------
    g = lambda keys: ex.groupby(keys, as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                                                          parties=("cp_key", "nunique"))
    R["dir_type"], R["rail"] = g(["direction", "type"]), g(["direction", "rail"])
    R["rail_type"] = g(["direction", "rail", "type"])
    R["month"], R["month_type"] = g(["direction", "txn_month", "rail"]), g(["direction", "txn_month", "type"])
    R["bank"] = g(["direction", "cp_bank"])[lambda d: d["cp_bank"] != "PNC"]
    R["cpty_types"] = (ex[ex["cp_type"] == "external"].assign(cpty_type=lambda d: d["cpty_type"].fillna("(none)"))
                         .groupby(["direction", "cpty_type"], as_index=False)
                         .agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"), parties=("cp_key", "nunique")))
    R["own_accts"] = (ex.groupby(["ego_acct", "direction"], as_index=False)
                        .agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"), parties=("cp_key", "nunique")))

    # ---- reconciliation: every slice carries the same dollars -------------------------------------
    ref = float(ex["amount"].sum())
    for name in ["cp_dir", "cp_rail", "dir_type", "rail", "month", "own_accts"]:
        got = float(R[name]["amount"].sum())
        if abs(got - ref) > 1e-6 * max(ref, 1.0):
            raise AssertionError(f"Reconciliation failed: {name} = {got:,.2f}, expected {ref:,.2f}")

    # ---- transaction size -------------------------------------------------------------------------
    R["hist"] = R["pct_dir"] = R["pct_rail"] = None
    if hist is not None and len(hist):
        h = hist.copy()
        h["amt_bucket"] = pd.to_numeric(h["amt_bucket"]).astype(int)
        h["n_txn"] = pd.to_numeric(h["n_txn"])
        h["rail"] = h["pay_rail"].map(rail_label)
        h = h[h["dir"].isin(["IN", "OUT"])]
        if not include_cp:
            h = h[_bool(h["other_pnc"])]
        if len(h):
            R["hist"] = h
            R["pct_dir"] = hist_percentiles(h, ["dir"]).rename(columns={"dir": "direction"})
            R["pct_rail"] = hist_percentiles(h, ["dir", "rail"]).rename(columns={"dir": "direction"})

    # ---- ties between connected parties ------------------------------------------------------------
    R["ties"] = pd.DataFrame(columns=["src", "dst", "kind", "rail", "amount", "n_txn", "first_dt", "last_dt"])
    if ties_raw is not None and len(ties_raw):
        t = _clean_ids(ties_raw.copy())
        s_pnc = t["mdm_id_pays"].notna() | t["pnc_dep_acct_pays"].notna()
        d_pnc = t["mdm_id_receives"].notna() | t["pnc_dep_acct_receives"].notna()
        t["src"] = node_keys(s_pnc, t["mdm_id_pays"], t["pnc_dep_acct_pays"], t["cpty_id"], P["level"])
        t["dst"] = node_keys(d_pnc, t["mdm_id_receives"], t["pnc_dep_acct_receives"], t["cpty_id"], P["level"])
        t["kind"] = np.where(s_pnc & d_pnc, "pnc_pnc", "pnc_cpty")
        alters = set(R["flow"].index)
        t = t[t["src"].isin(alters) & t["dst"].isin(alters) & (t["src"] != t["dst"])
              & (t["src"] != R["ego_key"]) & (t["dst"] != R["ego_key"])]
        if len(t):
            R["ties"] = t.groupby(["src", "dst", "kind", "rail"], as_index=False).agg(
                amount=("amount", "sum"), n_txn=("n_txn", "sum"), first_dt=("first_dt", "min"), last_dt=("last_dt", "max"))
    return R


def visible_ties(R, show_pp, show_pc):
    t = R["ties"]
    kinds = ([("pnc_pnc")] if show_pp else []) + (["pnc_cpty"] if show_pc and R["include_cp"] else [])
    return t[t["kind"].isin(kinds)]


def counterparty_table(R, direction):
    d = R["cp_dir"][R["cp_dir"]["direction"] == direction].merge(R["attr"], left_on="cp_key", right_index=True)
    if d.empty:
        return d
    cr = R["cp_rail"][R["cp_rail"]["direction"] == direction].copy()
    cr["pct"] = cr["amount"] / cr.groupby("cp_key")["amount"].transform("sum")
    cr = cr.sort_values(["cp_key", "amount"], ascending=[True, False])
    cr["s"] = cr["rail"] + " " + (cr["pct"] * 100).round().fillna(0).astype(int).astype(str) + "%"
    mix = cr.groupby("cp_key").head(3).groupby("cp_key")["s"].agg(", ".join)
    other = set(R["cp_dir"].loc[R["cp_dir"]["direction"] != direction, "cp_key"])
    d = d.sort_values("amount", ascending=False).reset_index(drop=True)
    d["share"] = d["amount"] / d["amount"].sum()
    d["cum_share"] = d["share"].cumsum()
    d["avg_txn"] = d["amount"] / d["n_txn"]
    d["type"] = d["cp_type"].map(TYPE_LABEL)
    d["rail_mix"] = d["cp_key"].map(mix)
    d["two_way"] = d["cp_key"].isin(other)
    d.index = np.arange(1, len(d) + 1)
    d.index.name = "rank"
    return d.rename(columns={"label": "name", "cp_id": "id", "cp_mdm": "mdm_id", "cp_bank": "bank"})


def concentration(t):
    if t is None or t.empty:
        return {}
    s = t["share"].values
    hhi = float((s ** 2).sum())
    return {"parties": len(s), "top 1": s[:1].sum(), "top 5": s[:5].sum(), "top 10": s[:10].sum(),
            "HHI": hhi, "effective n": 1 / hhi if hhi else np.nan}


# =============================================================================
# NETWORK — data for the component (and for the standalone export)
# =============================================================================
def _edge_title(sn, dn, g):
    lines = [f"{sn}  →  {dn}", f"{money(g['amount'].sum())} in {int(g['n_txn'].sum()):,} txns"]
    by = g.groupby("rail")[["amount", "n_txn"]].sum().sort_values("amount", ascending=False)
    lines += [f"   {r}: {money(v['amount'])} ({int(v['n_txn']):,})" for r, v in by.iterrows()]
    lines.append(f"{g['first_dt'].min()} → {g['last_dt'].max()}")
    return "\n".join(lines)


def build_graph(R, max_nodes, show_pp, show_pc):
    """Three node types (target, PNC customer, counterparty) and one edge type: all rails between two
    parties summed into one directed edge whose width is proportional to its total dollars."""
    P, A, flow, ek = R["P"], R["attr"], R["flow"], R["ego_key"]
    vis = flow.head(max_nodes)
    keep = set(vis.index)
    font = lambda size, bold=False: {"size": size, "face": "arial", "color": "#222222", "bold": bold}
    ego_name = R["ego_names"][0] if R["ego_names"] else "(no name)"

    tip = [f"TARGET {'account' if P['mode'] == 'acct' else 'customer (MDM)'} {P['id']}", ego_name]
    tip.append(f"MDM: {', '.join(R['ego_mdms'])}" if P["mode"] == "acct" else f"{len(R['ego_accts'])} PNC account(s)")
    tip += [f"Received {money(flow['IN'].sum())} from {int((flow['IN'] > 0).sum()):,} parties",
            f"Sent {money(flow['OUT'].sum())} to {int((flow['OUT'] > 0).sum()):,} parties"]
    nodes = [dict(id=ek, label=f"{trunc(ego_name, 32)}\n{P['id']}", title="\n".join(tip), shape="star", size=40,
                  color={"background": C_EGO, "border": "#8A3F06",
                         "highlight": {"background": C_EGO, "border": "#000000"}},
                  font=font(16, True), x=0, y=0, fixed=True, search=f"★ {ego_name} · {P['id']} (target)")]

    fmax = max(float(vis["total"].max()), 1.0)
    for k, row in vis.iterrows():
        a = A.loc[k]
        pnc = a["cp_type"] == "internal"
        t = [a["label"], TYPE_LABEL[a["cp_type"]]]
        if pnc:
            if k.startswith("PNC:"):
                t += [f"Account: {k[4:]}", f"MDM: {a['cp_mdm']}"]
            else:
                t.append(f"MDM: {k[4:]}  ({int(a['n_pnc_accts'])} account(s) seen)")
        else:
            t += [f"Counterparty id: {a['cp_id']}",
                  f"Bank: {a['cp_bank'] if isinstance(a['cp_bank'], str) and a['cp_bank'] else '(not recorded)'}",
                  f"Type: {a['cpty_type'] if isinstance(a['cpty_type'], str) else '—'}"]
        t.append(f"Rails: {a['rails']}")
        if row["IN"] > 0:
            t.append(f"Paid the target: {money(row['IN'])}")
        if row["OUT"] > 0:
            t.append(f"Paid by the target: {money(row['OUT'])}")
        if pnc:
            t.append("Click to load into the lookup")
        bg, border = (C_INT, "#123A6E") if pnc else (C_EXT, "#1D6B48")
        n = dict(id=k, label=trunc(a["label"], 26), title="\n".join(t), shape="dot" if pnc else "diamond",
                 size=float(10 + 28 * np.sqrt(row["total"] / fmax)), borderWidth=1.2, font=font(11),
                 color={"background": bg, "border": border, "highlight": {"background": bg, "border": "#000000"}},
                 search=f"{a['label']} · {a['cp_id']} · {TYPE_LABEL[a['cp_type']]}")
        lk = lookup_of(k) if pnc else None
        if lk:
            n["pnc_kind"], n["pnc_id"] = lk
        nodes.append(n)

    # ---- edges: target <-> party, then party <-> party; one edge per direction per pair -------------
    raw = []
    ex = R["ex"][R["ex"]["cp_key"].isin(keep)]
    for (d, k), g in ex.groupby(["direction", "cp_key"]):
        nm = A.loc[k, "label"]
        src, dst, sn, dn = (k, ek, nm, ego_name) if d == "IN" else (ek, k, ego_name, nm)
        raw.append((src, dst, float(g["amount"].sum()), _edge_title(sn, dn, g)))
    tv = visible_ties(R, show_pp, show_pc)
    tv = tv[tv["src"].isin(keep) & tv["dst"].isin(keep)]
    for (s_, d_), g in tv.groupby(["src", "dst"]):
        raw.append((s_, d_, float(g["amount"].sum()), _edge_title(A.loc[s_, "label"], A.loc[d_, "label"], g)))
    amax = max([e[2] for e in raw] + [1.0])
    edges = [dict(**{"from": s_}, to=d_, title=title, width=round(1.0 + 11.0 * amt / amax, 2),
                  color={"color": C_EDGE, "highlight": "#222222", "hover": "#222222", "opacity": 0.85})
             for s_, d_, amt, title in raw]

    options = {
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.55}}, "arrowStrikethrough": False,
                  # curvedCW bends A->B and B->A to opposite sides, so two-way pairs stay readable
                  "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.18}},
        "physics": {"solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -90, "centralGravity": 0.012,
                                         "springLength": 170, "springConstant": 0.06, "avoidOverlap": 0.5},
                    "stabilization": {"enabled": True, "iterations": 300}},
        "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 60},
    }
    shown = vis["total"].sum() / max(flow["total"].sum(), 1.0)
    caption = (f"Showing {len(vis):,} of {len(flow):,} connected parties ({shown:.0%} of the target's dollar flow) "
               f"· {len(tv.groupby(['src', 'dst'])) if len(tv) else 0:,} payment links between them")
    return nodes, edges, options, caption


_NET_JS = r"""
const send = (type, extra) => window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, extra || {}), "*");
let net = null, nodes = null, lastSig = null, orig = {}, byLabel = {};
const DIM = {color: {background: "#ECECEC", border: "#D9D9D9"}, font: {color: "#C8C8C8"}};
function fitHeight() { send("streamlit:setFrameHeight", {height: document.body.scrollHeight + 4}); }
function highlight(id) {
  if (!nodes || !net) return;
  const keep = id === null ? null : new Set([id].concat(net.getConnectedNodes(id)));
  nodes.update(nodes.getIds().map(k => (keep === null || keep.has(k))
      ? {id: k, color: orig[k].color, font: orig[k].font}
      : Object.assign({id: k}, DIM)));
}
function render(args) {
  document.getElementById("net").style.height = (args.height || 780) + "px";
  if (args.sig !== lastSig) {
    lastSig = args.sig;
    nodes = new vis.DataSet(args.nodes);
    const edges = new vis.DataSet(args.edges);
    orig = {}; byLabel = {};
    const dl = document.getElementById("names");
    dl.innerHTML = "";
    args.nodes.forEach(n => {
      orig[n.id] = {color: n.color, font: n.font};
      byLabel[n.search] = n.id;
      const o = document.createElement("option"); o.value = n.search; dl.appendChild(o);
    });
    if (net) net.destroy();
    net = new vis.Network(document.getElementById("net"), {nodes: nodes, edges: edges}, args.options);
    net.on("click", p => {
      if (!p.nodes.length) { highlight(null); return; }
      const n = nodes.get(p.nodes[0]);
      highlight(n.id);
      if (n.pnc_id) {
        send("streamlit:setComponentValue", {value: {id: n.pnc_id, kind: n.pnc_kind, ts: Date.now()}, dataType: "json"});
        document.getElementById("hint").textContent =
          (n.pnc_kind === "acct" ? "Account " : "MDM id ") + n.pnc_id + " loaded into the lookup. Press Build ego network.";
      }
    });
  }
  fitHeight();
}
document.getElementById("find").addEventListener("change", e => {
  const id = byLabel[e.target.value];
  if (id !== undefined && net) { net.selectNodes([id]); net.focus(id, {scale: 1.1, animation: true}); highlight(id); }
});
window.addEventListener("message", e => { if (e.data && e.data.type === "streamlit:render") render(e.data.args); });
window.addEventListener("resize", fitHeight);
send("streamlit:componentReady", {apiVersion: 1});
"""

_NET_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
__VIS__
<style>
  html, body { margin: 0; padding: 0; font-family: Arial, sans-serif; background: #FFFFFF; }
  #bar { display: flex; gap: 12px; align-items: center; padding: 2px 2px 6px 2px; font-size: 12px; color: #555; }
  #find { width: 320px; padding: 4px 6px; font-size: 12px; border: 1px solid #CCC; border-radius: 4px; }
  #hint { color: #1F5AA6; }
  #net { width: 100%; border: 1px solid #DDD; border-radius: 4px; }
  div.vis-tooltip { white-space: pre-line !important; font-family: Arial, sans-serif; font-size: 12px; max-width: 520px; }
</style></head>
<body>
<div id="bar"><input id="find" list="names" placeholder="Find a party by name or id"><datalist id="names"></datalist>
<span id="hint">Click a PNC customer to load it into the lookup.</span></div>
<div id="net"></div>
<script>__JS__</script>
__DATA__
</body></html>"""


def _vis_js_path():
    import pyvis   # only for its bundled copy of vis-network; nothing is loaded from a CDN
    found = sorted(glob.glob(os.path.join(os.path.dirname(pyvis.__file__), "lib", "vis-*", "vis-network.min.js")))
    if not found:
        raise FileNotFoundError("vis-network.min.js not found inside the pyvis package")
    return found[-1]


def _write_if_changed(path, data):
    if os.path.exists(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return
    with open(path, "wb") as f:
        f.write(data)


@st.cache_resource(show_spinner=False)
def _component_dir():
    """Materialise the component (index.html + vis-network.min.js) once per server process: next to
    the app if writable, else in the temp dir."""
    with open(_vis_js_path(), "rb") as f:
        vis_js = f.read()
    page = _NET_HTML.replace("__VIS__", '<script src="vis-network.min.js"></script>') \
                    .replace("__JS__", _NET_JS).replace("__DATA__", "").encode("utf-8")
    for base in [os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ego_net_component"),
                 os.path.join(tempfile.gettempdir(), "pkg_ego_net_component")]:
        try:
            os.makedirs(base, exist_ok=True)
            _write_if_changed(os.path.join(base, "index.html"), page)
            _write_if_changed(os.path.join(base, "vis-network.min.js"), vis_js)
            return base
        except OSError:
            continue
    raise OSError("No writable location for the network component")


def standalone_html(nodes, edges, options):
    """Same page with vis-network and the data inlined: opens anywhere, no server, no network."""
    with open(_vis_js_path(), "r", encoding="utf-8") as f:
        vis_js = f.read()
    payload = json.dumps({"nodes": nodes, "edges": edges, "options": options, "height": NET_HEIGHT_PX,
                          "sig": "export"}).replace("</", "<\\/")
    return (_NET_HTML.replace("__VIS__", "<script>" + vis_js.replace("</script", "<\\/script") + "</script>")
                     .replace("__JS__", _NET_JS)
                     .replace("__DATA__", f"<script>render({payload});</script>")
                     .replace("Click a PNC customer to load it into the lookup.",
                              "Click a node to isolate its neighbourhood."))


_ego_net = components.declare_component("pkg_ego_net", path=_component_dir())


def legend_html():
    sw = {"star": f"<span style='color:{C_EGO};font-size:17px'>★</span>",
          "int": f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;background:{C_INT}'></span>",
          "ext": f"<span style='display:inline-block;width:10px;height:10px;background:{C_EXT};transform:rotate(45deg)'></span>"}
    it = lambda s, t: f"<span style='margin-right:18px;white-space:nowrap'>{s} {t}</span>"
    edge = (f"<span style='display:inline-block;width:30px;border-top:4px solid {C_EDGE};vertical-align:middle'></span>")
    return ("<div style='font-size:13px;line-height:1.9'>" + it(sw["star"], "target") + it(sw["int"], "PNC customer")
            + it(sw["ext"], "counterparty")
            + it(edge, "payment link: all rails combined, width proportional to total $, arrow = payer → payee")
            + "</div>")


# =============================================================================
# CHARTS (Altair ships with Streamlit)
# =============================================================================
def ch_monthly(R):
    m = R["month"].copy()
    m["signed"] = np.where(m["direction"] == "IN", m["amount"], -m["amount"])
    m["flow"] = m["direction"].map(DIR_LABEL)
    rails = m.groupby("rail")["amount"].sum().sort_values(ascending=False).index.tolist()
    net = m.groupby("txn_month", as_index=False)["signed"].sum().rename(columns={"signed": "net"})
    bars = alt.Chart(m).mark_bar().encode(
        x=alt.X("txn_month:O", title=None),
        y=alt.Y("sum(signed):Q", title="received (+) / sent (−)", axis=MONEY_AXIS),
        color=alt.Color("rail:N", scale=rail_scale(rails), legend=alt.Legend(title="rail", orient="top")),
        tooltip=["txn_month", "flow", "rail", alt.Tooltip("amount:Q", format="$,.0f"), alt.Tooltip("n_txn:Q", format=",")])
    line = alt.Chart(net).mark_line(color="black", point=True).encode(
        x="txn_month:O", y="net:Q", tooltip=["txn_month", alt.Tooltip("net:Q", format="$,.0f", title="net in − out")])
    return alt.layer(bars, line).properties(height=360)


def ch_rail_mix(R):
    rows = []
    for d in ["IN", "OUT"]:
        sub = R["rail"][R["rail"]["direction"] == d]
        for col, lab in [("amount", "$"), ("n_txn", "# txns")]:
            for _, r in sub.iterrows():
                rows.append({"row": f"{DIR_LABEL[d]} · {lab}", "rail": r["rail"], "value": float(r[col])})
    df = pd.DataFrame(rows)
    rails = R["rail"].groupby("rail")["amount"].sum().sort_values(ascending=False).index.tolist()
    order = [f"{DIR_LABEL[d]} · {l}" for d in ["IN", "OUT"] for l in ["$", "# txns"]]
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("row:N", sort=order, title=None),
        x=alt.X("value:Q", stack="normalize", axis=alt.Axis(format="%"), title="share"),
        color=alt.Color("rail:N", scale=rail_scale(rails), legend=alt.Legend(title="rail", orient="top")),
        tooltip=["row", "rail", alt.Tooltip("value:Q", format=",.0f")]).properties(height=190)


def ch_top(t, title, n=15):
    d = t.head(n).copy()
    d["who"] = [f"{trunc(nm, 34)} · …{str(i)[-4:]}" for nm, i in zip(d["name"], d["id"])]
    return alt.Chart(d).mark_bar().encode(
        y=alt.Y("who:N", sort=list(d["who"]), title=None),
        x=alt.X("amount:Q", title=None, axis=MONEY_AXIS),
        color=alt.Color("type:N", scale=alt.Scale(domain=list(TYPE_COLOR), range=list(TYPE_COLOR.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["name", "type", "id", "bank", alt.Tooltip("amount:Q", format="$,.0f"),
                 alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("n_txn:Q", format=","), "rail_mix"]
    ).properties(title=title, height=max(220, 22 * len(d)))


def ch_concentration(tin, tout):
    parts = [pd.DataFrame({"rank": np.arange(1, len(t) + 1), "cum_share": t["cum_share"].values, "flow": DIR_LABEL[d]})
             for t, d in [(tin, "IN"), (tout, "OUT")] if len(t)]
    return alt.Chart(pd.concat(parts)).mark_line().encode(
        x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="party rank (log)"),
        y=alt.Y("cum_share:Q", axis=alt.Axis(format="%"), title="cumulative share of dollars"),
        color=alt.Color("flow:N", scale=alt.Scale(range=["#1F5AA6", "#C0392B"])),
        tooltip=["flow", "rank", alt.Tooltip("cum_share:Q", format=".1%")]).properties(height=300)


def ch_hist(R, d):
    h = R["hist"][R["hist"]["dir"] == d].groupby(["rail", "amt_bucket"], as_index=False)["n_txn"].sum()
    h["amount_mid"] = 10 ** ((h["amt_bucket"] + 0.5) / 20)
    rails = R["rail"].groupby("rail")["amount"].sum().sort_values(ascending=False).index.tolist()
    return alt.Chart(h).mark_line(interpolate="step-after").encode(
        x=alt.X("amount_mid:Q", scale=alt.Scale(type="log"), axis=MONEY_AXIS, title="amount per transaction (log)"),
        y=alt.Y("n_txn:Q", title="transactions"),
        color=alt.Color("rail:N", scale=rail_scale(rails), legend=alt.Legend(orient="top", title=None)),
        tooltip=["rail", alt.Tooltip("amount_mid:Q", format="$,.0f", title="≈ amount"), alt.Tooltip("n_txn:Q", format=",")]
    ).properties(title=f"Transaction size — {DIR_LABEL[d].lower()}", height=280)


def ch_bars(df, cat, title, color, n=12):
    d = df.sort_values("amount", ascending=False).head(n).copy()
    d[cat] = d[cat].fillna("(not recorded)")
    return alt.Chart(d).mark_bar(color=color).encode(
        y=alt.Y(f"{cat}:N", sort=list(d[cat]), title=None), x=alt.X("amount:Q", axis=MONEY_AXIS, title=None),
        tooltip=[cat, alt.Tooltip("amount:Q", format="$,.0f"), alt.Tooltip("n_txn:Q", format=","),
                 alt.Tooltip("parties:Q", format=",")]).properties(title=title, height=max(180, 24 * len(d)))


def ch_active(R):
    m = R["month_type"].copy()
    m["series"] = m["direction"].map({"IN": "senders", "OUT": "receivers"}) + " — " + m["type"]
    dom = [f"{a} — {TYPE_LABEL[t]}" for a in ["senders", "receivers"] for t in ["internal", "external"]]
    return alt.Chart(m).mark_line(point=True).encode(
        x=alt.X("txn_month:O", title=None), y=alt.Y("parties:Q", title="active parties"),
        color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=[C_INT, C_EXT, "#7FA6D6", "#8FD0B0"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("direction:N", legend=None),
        tooltip=["txn_month", "series", "parties"]).properties(height=280)


# =============================================================================
# TABLE HELPERS
# =============================================================================
def show(df, money_cols=(), pct_cols=(), int_cols=()):
    if df is None or len(df) == 0:
        st.caption("none")
        return
    fmt = {c: "${:,.0f}" for c in money_cols if c in df.columns}
    fmt.update({c: "{:.1%}" for c in pct_cols if c in df.columns})
    fmt.update({c: "{:,.0f}" for c in int_cols if c in df.columns})
    st.dataframe(df.style.format(fmt, na_rep=""), **_wide(st.dataframe))


def with_shares(df, by="direction"):
    d = df.copy()
    d["share_amt"] = d["amount"] / d.groupby(by)["amount"].transform("sum")
    d["share_txn"] = d["n_txn"] / d.groupby(by)["n_txn"].transform("sum")
    d["avg_txn"] = d["amount"] / d["n_txn"]
    d[by] = d[by].map(DIR_LABEL)
    return d


def zip_bytes(tables, network_html):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is not None and len(df):
                z.writestr(f"{name}.csv", df.to_csv(index=not isinstance(df.index, pd.RangeIndex)))
        z.writestr("ego_network.html", network_html)
    return buf.getvalue()


# =============================================================================
# APP
# =============================================================================
st.set_page_config(page_title="PKG · Ego network", page_icon="🕸️", layout="wide")
st.markdown("<style>div.block-container{padding-top:1.6rem}</style>", unsafe_allow_html=True)

# a click on a PNC customer in the network arrives as a pending lookup; apply it before the lookup
# widgets exist (a widget's value cannot be changed after it is drawn in the same run)
st.session_state.setdefault("lookup_mode", MODES[0])
st.session_state.setdefault("lookup_id", "")
_pending = st.session_state.pop("_pending_lookup", None)
if _pending:
    kind, pid = _pending
    st.session_state["lookup_mode"] = MODES[0] if kind == "acct" else MODES[1]
    st.session_state["lookup_id"] = pid
    msg = f"{'Account' if kind == 'acct' else 'MDM id'} {pid} loaded into the lookup. Press Build ego network."
    st.toast(msg) if hasattr(st, "toast") else st.sidebar.info(msg)

_today = date.today()
_end_default = _today.replace(day=1) - timedelta(days=1)                       # end of last month
_start_default = date(*(int(x) for x in _month_shift(_end_default, -(DEFAULT_WINDOW_MONTHS - 1)).split("-")), 1)

with st.sidebar:
    st.header("Lookup")
    with st.form("lookup"):
        st.radio("Look up by", MODES, horizontal=True, key="lookup_mode")
        st.text_input("Id", placeholder="PNC deposit account or MDM id", key="lookup_id")
        window = st.date_input("Window", value=(_start_default, _end_default), min_value=DATA_START,
                               max_value=_today, help="Each lookup scans the window; start narrow.")
        level_lbl = st.radio("PNC customers drawn as", ["Same as lookup", "Accounts", "Customers (MDM)"],
                             help="Accounts: one node per deposit account. Customers: accounts of the same MDM id merged.")
        rails = st.multiselect("Payment rails", list(RAIL_DEF), default=DEFAULT_RAILS, format_func=rail_label)
        submitted = st.form_submit_button("Build ego network", type="primary", **_wide(st.form_submit_button))

    st.header("Show")
    include_cp = st.checkbox("Include counterparties", value=True,
                             help="Off: PNC customers only, in the network and in every table.")
    show_pp = st.checkbox("Connections between PNC customers", value=True)
    show_pc = st.checkbox("Connections between PNC customers and counterparties", value=True,
                          disabled=not include_cp)
    max_vis = st.slider("Parties drawn", 10, 600, MAX_VIS_DEFAULT, step=10)
    top_n = st.slider("Rows in ranked tables", 10, 500, 25, step=5)

if submitted:
    ego_id = st.session_state["lookup_id"].strip()
    errs = []
    if not ID_RE.match(ego_id):
        errs.append("Id must be 1–64 letters, digits, '.', '_' or '-'.")
    if not (isinstance(window, (tuple, list)) and len(window) == 2):
        errs.append("Pick both a start and an end date.")
    if not rails:
        errs.append("Select at least one payment rail.")
    if errs:
        for e in errs:
            st.sidebar.error(e)
    else:
        mode = "acct" if st.session_state["lookup_mode"] == MODES[0] else "mdm"
        level = {"Accounts": "account", "Customers (MDM)": "customer"}.get(
            level_lbl, "account" if mode == "acct" else "customer")
        st.session_state["P"] = dict(mode=mode, id=ego_id, start=window[0].isoformat(), end=window[1].isoformat(),
                                     level=level, rails=tuple(r for r in RAIL_DEF if r in rails))

P = st.session_state.get("P")
st.title("PKG · Ego network")
if P is None:
    st.info("Enter a deposit account or an MDM id in the sidebar and press **Build ego network**.")
    st.stop()

# ---- queries (cached by SQL text, so the Show options never re-query) ------------------------------
log = []
try:
    with st.spinner(f"Q1/3 · transactions touching {P['id']} ({P['start']} → {P['end']}) …"):
        s1 = sql_ego_cube(P)
        cube, t1 = run_sql(s1)
        log.append(("Q1 transactions touching the target", s1, t1, len(cube)))
    if cube.empty:
        with st.spinner("No rows — searching every id column for near matches (one more scan) …"):
            near, _ = run_sql(sql_near_match(P))
        st.warning(f"No transactions for {'account' if P['mode'] == 'acct' else 'MDM id'} **{P['id']}** "
                   f"in {P['start']} → {P['end']} on the selected rails.")
        if len(near):
            st.write("Near matches (whitespace / leading zeros ignored, all rails):")
            show(near, int_cols=["n_txn"])
        else:
            st.write("No near matches on any account, MDM or counterparty id column in this window.")
        st.stop()

    with st.spinner("Q2/3 · transaction sizes …"):
        s2 = sql_amount_hist(P)
        hist, t2 = run_sql(s2)
        log.append(("Q2 transaction-size histogram", s2, t2, len(hist)))

    # the ties query depends only on the lookup, never on the Show options: search the top PNC
    # customers and the top counterparties separately, so toggling counterparties needs no new query
    R_all = build_results(P, cube, hist, None, include_cp=True)
    A0, f0 = R_all["attr"], R_all["flow"]
    top_int = [k for k in f0.index if A0.loc[k, "cp_type"] == "internal"][:TIE_SEARCH_CAP]
    top_ext = [k for k in f0.index if A0.loc[k, "cp_type"] == "external"][:TIE_SEARCH_CAP]
    int_ids = sorted({lookup_of(k)[1] for k in top_int
                      if lookup_of(k)[0] == ("acct" if P["level"] == "account" else "mdm")})
    ext_ids = sorted(A0.loc[top_ext, "cp_id"].dropna().unique().tolist())
    ties_raw = None
    if int_ids:
        with st.spinner("Q3/3 · payments between the connected parties …"):
            s3 = sql_ties(P, int_ids, ext_ids)
            ties_raw, t3 = run_sql(s3)
            log.append(("Q3 payments between connected parties", s3, t3, len(ties_raw)))
    R = build_results(P, cube, hist, ties_raw, include_cp=include_cp)
except Exception as ex:
    st.error(f"{type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
    st.stop()

# ---- header ---------------------------------------------------------------------------------------
fl = R["flow"]
ego_name = R["ego_names"][0] if R["ego_names"] else "(no name)"
st.markdown(f"**{'Account' if P['mode'] == 'acct' else 'Customer'} {P['id']}** · {ego_name}"
            + (f" · MDM {', '.join(R['ego_mdms'])}" if P["mode"] == "acct" else f" · {len(R['ego_accts'])} PNC account(s)")
            + f" · {P['start']} → {P['end']} · {', '.join(rail_label(r) for r in P['rails'])}"
            + f" · PNC customers drawn as {'accounts' if P['level'] == 'account' else 'customers'}"
            + ("" if include_cp else " · **PNC customers only**"))
m = st.columns(6)
m[0].metric("Received", money(fl["IN"].sum()))
m[1].metric("Sent", money(fl["OUT"].sum()))
m[2].metric("Net (in − out)", money(fl["IN"].sum() - fl["OUT"].sum()))
m[3].metric("Transactions", f"{int(R['ex']['n_txn'].sum()):,}")
m[4].metric("PNC customers", f"{int((R['attr']['cp_type'] == 'internal').sum()):,}")
m[5].metric("Counterparties", f"{int((R['attr']['cp_type'] == 'external').sum()):,}" if include_cp else "hidden")

T_in, T_out = counterparty_table(R, "IN"), counterparty_table(R, "OUT")
tab_names = ["Network", "Senders & receivers", "Rails", "Over time"]
tab_names += (["Counterparty banks"] if include_cp else []) + ["Connections"]
tab_names += (["Own accounts"] if P["mode"] == "mdm" else []) + ["Summary & QA", "Queries"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

nodes, edges, options, caption = build_graph(R, max_vis, show_pp, show_pc)
with tabs["Network"]:
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption(caption + " · hover for details · click a node to isolate its neighbourhood")
    sig = hashlib.md5(json.dumps([nodes, edges], sort_keys=True, default=str).encode()).hexdigest()
    clicked = _ego_net(nodes=nodes, edges=edges, options=options, height=NET_HEIGHT_PX, sig=sig,
                       key="ego_net", default=None)
    if isinstance(clicked, dict) and clicked.get("id") and clicked.get("ts") != st.session_state.get("_click_ts"):
        st.session_state["_click_ts"] = clicked.get("ts")
        st.session_state["_pending_lookup"] = (clicked.get("kind"), str(clicked["id"]))
        _rerun()

CP_COLS = ["name", "type", "id", "mdm_id", "bank", "cpty_type", "rails", "amount", "share", "cum_share",
           "n_txn", "avg_txn", "rail_mix", "months_active", "first_dt", "last_dt", "same_customer", "two_way"]
CP_FMT = dict(money_cols=["amount", "avg_txn"], pct_cols=["share", "cum_share"], int_cols=["n_txn", "months_active"])


def _cp_view(t):
    cols = [c for c in CP_COLS if c in t.columns and not (c in ("bank", "cpty_type") and not include_cp)]
    if P["mode"] == "mdm":
        cols = [c for c in cols if c != "same_customer"]
    return t[cols].head(top_n) if len(t) else t


with tabs["Senders & receivers"]:
    c1, c2 = st.columns(2)
    if len(T_in):
        c1.altair_chart(ch_top(T_in, "Top senders to the target"), **_wide(st.altair_chart))
    if len(T_out):
        c2.altair_chart(ch_top(T_out, "Top receivers from the target"), **_wide(st.altair_chart))
    st.subheader(f"Senders — paid the target ({len(T_in):,})")
    show(_cp_view(T_in), **CP_FMT)
    st.subheader(f"Receivers — paid by the target ({len(T_out):,})")
    show(_cp_view(T_out), **CP_FMT)
    both = sorted(set(T_in["cp_key"]) & set(T_out["cp_key"])) if len(T_in) and len(T_out) else []
    st.subheader(f"Two-way parties — both sent to and received from the target ({len(both):,})")
    tw = pd.DataFrame()
    if both:
        a, b = T_in.set_index("cp_key").loc[both], T_out.set_index("cp_key").loc[both]
        tw = pd.DataFrame({"name": a["name"], "type": a["type"], "id": a["id"], "received_from": a["amount"],
                           "sent_to": b["amount"]})
        tw["net_to_target"] = tw["received_from"] - tw["sent_to"]
        tw["gross"] = tw["received_from"] + tw["sent_to"]
        tw = tw.sort_values("gross", ascending=False).reset_index(drop=True)
        show(tw.head(top_n), money_cols=["received_from", "sent_to", "net_to_target", "gross"])
    else:
        st.caption("none")

with tabs["Rails"]:
    st.altair_chart(ch_rail_mix(R), **_wide(st.altair_chart))
    rails_t = with_shares(R["rail"])
    if R["pct_rail"] is not None:
        rails_t = rails_t.merge(R["pct_rail"].assign(direction=lambda d: d["direction"].map(DIR_LABEL)),
                                on=["direction", "rail"], how="left")
    rails_t = rails_t.sort_values(["direction", "amount"], ascending=[True, False]).set_index(["direction", "rail"])
    st.subheader("Rails by direction")
    show(rails_t, money_cols=["amount", "avg_txn", "p10", "p50", "p90", "p99"], pct_cols=["share_amt", "share_txn"],
         int_cols=["n_txn", "parties"])
    st.caption("p10 … p99: per-transaction amount percentiles from a log-bucket histogram (±6%).")
    if include_cp:
        rt = R["rail_type"].pivot_table(index=["direction", "rail"], columns="type", values="amount",
                                        aggfunc="sum", fill_value=0.0).reindex(columns=list(TYPE_COLOR), fill_value=0.0)
        rt["counterparty_share"] = rt[TYPE_LABEL["external"]] / rt.sum(axis=1)
        rt = rt.reset_index().assign(direction=lambda d: d["direction"].map(DIR_LABEL)).set_index(["direction", "rail"])
        st.subheader("Rail × party type (dollars)")
        show(rt, money_cols=list(TYPE_COLOR), pct_cols=["counterparty_share"])
    if R["hist"] is not None:
        c1, c2 = st.columns(2)
        if (R["hist"]["dir"] == "IN").any():
            c1.altair_chart(ch_hist(R, "IN"), **_wide(st.altair_chart))
        if (R["hist"]["dir"] == "OUT").any():
            c2.altair_chart(ch_hist(R, "OUT"), **_wide(st.altair_chart))

with tabs["Over time"]:
    st.altair_chart(ch_monthly(R), **_wide(st.altair_chart))
    st.altair_chart(ch_active(R), **_wide(st.altair_chart))
    mo = R["month_type"].pivot_table(index="txn_month", columns=["direction", "type"],
                                     values=["amount", "parties"], aggfunc="sum", fill_value=0)
    monthly = pd.DataFrame(index=mo.index)
    for d in ["IN", "OUT"]:
        cols = [("amount", d, TYPE_LABEL[t]) for t in ["internal", "external"] if ("amount", d, TYPE_LABEL[t]) in mo.columns]
        monthly[f"{DIR_LABEL[d].lower()} $"] = mo[cols].sum(axis=1) if cols else 0.0
    monthly["net $"] = monthly["received $"] - monthly["sent $"]
    for d, who in [("IN", "senders"), ("OUT", "receivers")]:
        for t, s in [("internal", "PNC"), ("external", "cpty")]:
            if t == "external" and not include_cp:
                continue
            col = ("parties", d, TYPE_LABEL[t])
            monthly[f"{who} ({s})"] = mo[col] if col in mo.columns else 0
    st.subheader("Monthly")
    show(monthly, money_cols=["received $", "sent $", "net $"], int_cols=[c for c in monthly.columns if "$" not in c])

bank_t = pd.DataFrame()
if "Counterparty banks" in tabs:
    with tabs["Counterparty banks"]:
        bk = R["bank"]
        c1, c2 = st.columns(2)
        for col, d in [(c1, "IN"), (c2, "OUT")]:
            sub = bk[bk["direction"] == d]
            if len(sub):
                col.altair_chart(ch_bars(sub, "cp_bank", f"Counterparty banks — {DIR_LABEL[d].lower()}", C_EXT),
                                 **_wide(st.altair_chart))
        if len(bk):
            bank_t = bk.assign(share_of_counterparty_amt=bk["amount"] / bk.groupby("direction")["amount"].transform("sum"),
                               direction=bk["direction"].map(DIR_LABEL)) \
                       .sort_values(["direction", "amount"], ascending=[True, False]).set_index(["direction", "cp_bank"])
        st.subheader("Banks of counterparties")
        show(bank_t, money_cols=["amount"], pct_cols=["share_of_counterparty_amt"], int_cols=["n_txn", "parties"])
        ct = R["cpty_types"]
        if len(ct):
            ct = ct.assign(direction=ct["direction"].map(DIR_LABEL)).set_index(["direction", "cpty_type"])
        st.subheader("Counterparty types")
        show(ct, money_cols=["amount"], int_cols=["n_txn", "parties"])

with tabs["Connections"]:
    tv = visible_ties(R, show_pp, show_pc)
    st.caption(f"Payments between the target's connected parties, the target excluded. Searched among the top "
               f"{TIE_SEARCH_CAP:,} PNC customers and the top {TIE_SEARCH_CAP:,} counterparties by dollars with the "
               "target. Counterparty-to-counterparty payments are not in the source table. "
               "The connection types shown follow the Show options in the sidebar.")
    tp_named, emb = pd.DataFrame(), pd.DataFrame()
    if len(tv):
        tp = tv.groupby(["src", "dst", "kind"], as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"))
        tp["kind"] = tp["kind"].map({"pnc_pnc": "PNC ↔ PNC", "pnc_cpty": "PNC ↔ counterparty"})
        st.subheader("Payment links between connected parties")
        show(tp.groupby("kind").agg(links=("amount", "size"), amount=("amount", "sum"), n_txn=("n_txn", "sum")),
             money_cols=["amount"], int_cols=["links", "n_txn"])
        deg = pd.concat([tp[["src", "dst", "amount"]].rename(columns={"src": "k", "dst": "o"}),
                         tp[["dst", "src", "amount"]].rename(columns={"dst": "k", "src": "o"})])
        emb = deg.groupby("k").agg(partners=("o", "nunique"), link_amount=("amount", "sum"))
        emb = emb.join(R["attr"][["label", "cp_type", "cp_id"]]).join(fl["total"].rename("flow_with_target"))
        emb["type"] = emb["cp_type"].map(TYPE_LABEL)
        emb = emb.sort_values(["partners", "link_amount"], ascending=False).reset_index(drop=True)
        emb = emb[["label", "type", "cp_id", "partners", "link_amount", "flow_with_target"]].rename(
            columns={"label": "name", "cp_id": "id"})
        st.subheader("Most connected parties inside the network")
        show(emb.head(top_n), money_cols=["link_amount", "flow_with_target"], int_cols=["partners"])
        lab = R["attr"]["label"]
        tp_named = tp.assign(payer=tp["src"].map(lab), payee=tp["dst"].map(lab)) \
                     .sort_values("amount", ascending=False)[["payer", "payee", "kind", "amount", "n_txn"]]
        st.subheader("Largest links")
        show(tp_named.head(top_n).reset_index(drop=True), money_cols=["amount"], int_cols=["n_txn"])
    else:
        st.caption("No payments between connected parties for the selected connection types.")

if "Own accounts" in tabs:
    with tabs["Own accounts"]:
        oa = R["own_accts"].pivot_table(index="ego_acct", columns="direction", values=["amount", "n_txn", "parties"],
                                        aggfunc="sum", fill_value=0)
        oa.columns = [f"{DIR_LABEL[d].lower()} {v}" for v, d in oa.columns]
        oa = oa.sort_values([c for c in oa.columns if "amount" in c], ascending=False)
        st.subheader(f"Which of the customer's accounts carry the flow ({len(oa):,})")
        show(oa, money_cols=[c for c in oa.columns if "amount" in c], int_cols=[c for c in oa.columns if "amount" not in c])
        sf = R["self"]
        st.caption(f"Transfers between the customer's own accounts (not drawn): "
                   f"{int(sf['n_txn'].sum()):,} txns, {money(sf['amount'].sum())}.")

with tabs["Summary & QA"]:
    ht = with_shares(R["dir_type"]).sort_values(["direction", "type"]).set_index(["direction", "type"])
    st.subheader("PNC customers vs counterparties" if include_cp else "PNC customers")
    show(ht, money_cols=["amount", "avg_txn"], pct_cols=["share_amt", "share_txn"], int_cols=["n_txn", "parties"])
    if R["pct_dir"] is not None:
        st.subheader("Transaction size")
        show(R["pct_dir"].assign(direction=lambda d: d["direction"].map(DIR_LABEL)).set_index("direction"),
             money_cols=["p10", "p50", "p90", "p99"])
    st.subheader("Concentration")
    conc = pd.DataFrame({DIR_LABEL["IN"]: concentration(T_in), DIR_LABEL["OUT"]: concentration(T_out)}).T
    show(conc, pct_cols=["top 1", "top 5", "top 10"], int_cols=["parties"])
    st.caption("effective n = 1 / HHI: the number of equal-sized parties giving the same concentration.")
    if len(T_in) or len(T_out):
        st.altair_chart(ch_concentration(T_in, T_out), **_wide(st.altair_chart))
    st.subheader("Data quality")
    show(R["qa"], money_cols=["amount", "amount_no_cpty_id"], int_cols=["n_txn", "n_txn_no_cpty_id"])
    st.caption("All transactions touching the target, before the Show options. SELF = the target paying itself "
               "(own-account transfers for an MDM lookup), not drawn. no_cpty_id = counterparty side with no id; "
               "counted here, not drawn.")

with tabs["Queries"]:
    for name, sql, secs, n in log:
        with st.expander(f"{name} · {n:,} rows · {secs:.1f}s (cached results show the original time)"):
            st.code(sql, language="sql")

with st.sidebar:
    st.header("Export")
    tables = {"senders": T_in, "receivers": T_out, "two_way": tw, "rails": rails_t, "monthly": monthly,
              "banks": bank_t, "links": tp_named, "most_connected": emb}
    st.download_button("Download tables + network (zip)", zip_bytes(tables, standalone_html(nodes, edges, options)),
                       file_name=f"ego_{P['mode']}_{P['id']}_{P['start']}_{P['end']}.zip",
                       mime="application/zip", **_wide(st.download_button))
