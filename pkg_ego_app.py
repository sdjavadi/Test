"""
PKG · Ego Network Explorer (Streamlit)

    streamlit run pkg_ego_app.py

Look up a PNC deposit account or an MDM id and see its ego network from the payment staging
table: every account / customer / external counterparty that paid it or was paid by it, the
payments between those, and summaries of both ends (rails, categories, time, banks, concentration).

All data comes from Impala through `dbi.db_get_query` (pandas out). Three queries per lookup,
each cached by its SQL text, so changing display options never re-queries:

    Q1  ego cube       rows touching the ego, grouped to (month, category, rail, both ends)
    Q2  amount histo   ego rows bucketed by log10(amount), for size percentiles (optional)
    Q3  alter ties     payments between the ego's top-N alters, ego excluded

Identity rules carried over from the PKG edge build:
  * a side is PNC (internal) when it has an mdm_id or a PNC deposit account; otherwise external
  * external counterparty id = unq_cpty_id for 3B/3C/7B/9B (unq_cpty_acct_id is only a routing
    number or null there), unq_cpty_acct_id elsewhere, the other as fallback
  * 5C/6C inbound RTP double-recording is removed by anti-joining pkg_dedup_5c6c (6C twin dropped)
"""
import io
import json
import re
import time
import traceback
import zipfile
import html as _html
from datetime import date, timedelta

import inspect

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

import dbi

# =============================================================================
# CONFIGURATION
# =============================================================================
PAYMENTS_TABLE = "dsihd01p_dsi.neo4j_payments"
DEDUP_TABLE = "bdahd01p_dlcdi1_cdi_tm.pkg_dedup_5c6c"   # trans_id_drop = 6C twin of a 5C payment
DATA_START = date(2023, 7, 1)                          # first trans_dt partition
DEFAULT_WINDOW_MONTHS = 3      # each lookup scans the window twice; widen deliberately
ALTER_CAP_DEFAULT = 500        # alters searched for alter<->alter ties (top by $ with the ego)
ALTER_CAP_MAX = 2000           # IN-lists beyond this make Impala planning slow
MAX_VIS_DEFAULT = 150
CACHE_TTL_S = 6 * 3600


def _wide(fn):
    """Full-width kwarg for this Streamlit version. Releases that deprecate use_container_width keep
    it as a None-default shim and take width='stretch'; older releases default it to False."""
    p = inspect.signature(fn).parameters.get("use_container_width")
    if p is None:
        return {}
    return {"width": "stretch"} if p.default is None else {"use_container_width": True}


def get_query(sql):
    df = dbi.db_get_query(
        sql,
        dsn="DSN=bdpimp04-impala;",
        pool="root.CIB-AMG_Impala",
        conn_options={"SocketTimeout": 0},
    )
    return df


# ---- category dictionary: letter = direction, number = rail family --------------------------
FAMILY = {"1": "ACH orig. via PNC, with TPO", "2": "ACH orig. via PNC, no TPO", "3": "ACH orig. via non-PNC",
          "4": "Wire", "5": "RTP (PRT)", "6": "RTP (P2P)", "7": "Check", "8": "Debit card", "9": "PCARD"}
LETTER = {"A": "internal", "B": "outbound", "C": "inbound"}
CODES = ["1A1", "1A2", "1B", "1C", "2A", "2B", "2C", "3B", "3C", "4A", "4B", "4C", "5A", "5B", "5C",
         "6A", "6B", "6C", "7A", "7B", "7C", "8B", "8C", "9B"]
# Debit card is ~42% of rows and ~0.5% of dollars; its merchants flood the picture. Off by default,
# matching the edge build. Tick them back on for a consumer-style account.
DEFAULT_OFF = {"8B", "8C"}


def code_label(c):
    extra = {"1A1": " (originator → customer)", "1A2": " (customer → originator)"}.get(c, "")
    return f"{c} · {FAMILY.get(c[0], '?')} · {LETTER.get(c[1], '?')}{extra}"


CODE_EXPR = "regexp_extract(p.category, '^([0-9]+[A-Z][0-9]*)', 1)"
CPTY_ID_EXPR = (f"CASE WHEN {CODE_EXPR} IN ('3B','3C','7B','9B') "
                "THEN COALESCE(NULLIF(p.unq_cpty_id, ''), NULLIF(p.unq_cpty_acct_id, '')) "
                "ELSE COALESCE(NULLIF(p.unq_cpty_acct_id, ''), NULLIF(p.unq_cpty_id, '')) END")

# ---- visual vocabulary -----------------------------------------------------------------------
C_EGO, C_INT, C_EXT, C_SIB = "#F58025", "#1F5AA6", "#2E9E6B", "#F9C89B"
TYPE_LABEL = {"internal": "PNC customer", "external": "External counterparty"}
TYPE_COLOR = {TYPE_LABEL["internal"]: C_INT, TYPE_LABEL["external"]: C_EXT}
DIR_LABEL = {"IN": "Received", "OUT": "Sent"}
RAIL_COLORS = {"ACH": "#1F77B4", "WIRE": "#D62728", "RTP_PRT": "#9467BD", "RTP_P2P": "#C5B0D5",
               "CHECK": "#8C564B", "DEBIT_CARD_SIGNATURE": "#E377C2", "PCARD": "#17BECF",
               "UNKNOWN": "#BBBBBB"}
_SPARE = ["#BCBD22", "#FF7F0E", "#AEC7E8", "#98DF8A", "#FFBB78", "#9EDAE5"]


def rail_color(r):
    r = str(r)
    if r not in RAIL_COLORS:
        RAIL_COLORS[r] = next((c for c in _SPARE if c not in RAIL_COLORS.values()), "#999999")
    return RAIL_COLORS[r]


def rail_scale(rails):
    rails = list(rails)
    return alt.Scale(domain=rails, range=[rail_color(r) for r in rails])


MONEY_AXIS = alt.Axis(labelExpr="replace(format(datum.value, '$~s'), 'G', 'B')")
ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
ID_COLS = ["mdm_id_pays", "mdm_id_receives", "pnc_dep_acct_pays", "pnc_dep_acct_receives", "cpty_id"]


def money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    a, s = abs(x), "-" if x < 0 else ""
    for d, u in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= d:
            return f"{s}${a / d:,.1f}{u}"
    return f"{s}${a:,.0f}"


def _safe(s):
    return re.sub(r"[<>]", " ", "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s))


def trunc(s, n=28):
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


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
    """FROM/JOIN + WHERE shared by every query: window, categories, 5C/6C dedup."""
    where = [f"p.trans_dt BETWEEN {_q(P['start'])} AND {_q(P['end'])}"]
    if set(P["codes"]) != set(CODES):
        # expression on the partition column -> Impala still prunes category partitions
        where.append(f"{CODE_EXPR} IN ({_in(P['codes'])})")
    join = ""
    if P["dedup"]:
        s, e = date.fromisoformat(P["start"]), date.fromisoformat(P["end"])
        # one month of slack each side: a tier-2 6C twin is dated a day after its 5C
        join = (f"LEFT ANTI JOIN (SELECT trans_id_drop FROM {DEDUP_TABLE} "
                f"WHERE `month` BETWEEN '{_month_shift(s, -1)}' AND '{_month_shift(e, 1)}') d "
                f"ON p.trans_id = d.trans_id_drop")
    return f"FROM {PAYMENTS_TABLE} p {join}", where


def _ego_cols(P):
    return (("p.pnc_dep_acct_pays", "p.pnc_dep_acct_receives") if P["mode"] == "acct"
            else ("p.mdm_id_pays", "p.mdm_id_receives"))


_INNER = f"""
    SELECT p.trans_dt, CAST(p.trans_amt AS DOUBLE) AS amt, {CODE_EXPR} AS cat_code,
           COALESCE(NULLIF(p.payment_rail, ''), 'UNKNOWN') AS pay_rail, p.cpty_type,
           NULLIF(p.mdm_id_pays, '') AS mdm_id_pays, NULLIF(p.mdm_id_receives, '') AS mdm_id_receives,
           NULLIF(p.pnc_dep_acct_pays, '') AS pnc_dep_acct_pays,
           NULLIF(p.pnc_dep_acct_receives, '') AS pnc_dep_acct_receives,
           p.customer_name_pays, p.customer_name_receives, p.cpty_name, p.cpty_fin_entity_name,
           {CPTY_ID_EXPR} AS cpty_id"""

_CUBE_SELECT = """
SELECT {month} b.cat_code, b.pay_rail, b.cpty_type,
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
            + "\nGROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9")


def sql_amount_hist(P):
    """20 log-buckets per decade (bucket width ×1.12) -> percentiles good to ~±6%.
    Impala has no general percentile function; a histogram also gives the distribution chart."""
    frm, where = _scope(P)
    cp, cr = _ego_cols(P)
    x = _q(P["id"])
    where.append(f"({cp} = {x} OR {cr} = {x})")
    return f"""
SELECT x.dir, x.pay_rail, x.amt_bucket, COUNT(*) AS n_txn, SUM(x.amt) AS amount
FROM (
    SELECT CASE WHEN {cp} = {x} AND {cr} = {x} THEN 'SELF' WHEN {cp} = {x} THEN 'OUT'
                WHEN {cr} = {x} THEN 'IN' ELSE 'OTHER' END AS dir,
           COALESCE(NULLIF(p.payment_rail, ''), 'UNKNOWN') AS pay_rail,
           CAST(p.trans_amt AS DOUBLE) AS amt,
           CAST(FLOOR(LOG10(GREATEST(CAST(p.trans_amt AS DOUBLE), 0.01)) * 20) AS INT) AS amt_bucket
    {frm}
    WHERE {' AND '.join(where)}
) x
GROUP BY 1, 2, 3"""


def sql_ties(P, int_ids, ext_ids):
    """Payments between alters. Every observable tie has a PNC end (external->external is not in
    the table), so the scan is pre-filtered on the internal alters; the outer filter keeps only
    rows whose other end is also an alter. The ego's own rows are excluded."""
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
            + "\nGROUP BY 1, 2, 3, 4, 5, 6, 7, 8")


def sql_near_match(P):
    """Zero rows is usually formatting (leading zeros, wrong id type), not an unused id.
    One scan over the window, all categories, on every id column."""
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
            df[c] = df[c].astype("object").where(df[c].notna(), None)
            df[c] = df[c].map(lambda v: None if v is None or str(v).strip() == "" else str(v))
    for c in ["amount", "n_txn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _prefixed(prefix, s):
    return s.map(lambda v: None if v is None else prefix + v)


def node_keys(pays_internal, mdm, acct, cpty, level):
    """Node key of one end. PNC end: 'PNC:<acct>' (account level) or 'MDM:<mdm>' (customer level),
    each falling back to the other id when missing. External end: 'EXT:<cpty_id>' or None."""
    if level == "account":
        k_int = _prefixed("PNC:", acct).where(acct.notna(), _prefixed("MDM:", mdm))
    else:
        k_int = _prefixed("MDM:", mdm).where(mdm.notna(), _prefixed("PNC:", acct))
    return k_int.where(pays_internal, _prefixed("EXT:", cpty))


def ego_key(P):
    return ("PNC:" if P["mode"] == "acct" else "MDM:") + P["id"]


def orient(cube, P):
    """Turn the raw ego cube into ego-centric rows: direction, the other end's key and attributes."""
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
    internal = c["cp_mdm"].notna() | c["cp_pnc_acct"].notna()
    c["cp_type"] = np.where(internal, "internal", "external")
    c["cp_key"] = node_keys(internal, c["cp_mdm"], c["cp_pnc_acct"], c["cpty_id"], P["level"])
    c["cp_name"] = other("name_pays", "name_receives").where(internal, c["cpty_name"])
    c["cp_bank"] = c["cpty_bank"].where(~internal, "PNC")
    c["cp_id"] = (c["cp_pnc_acct"] if P["level"] == "account" else c["cp_mdm"]).where(internal, c["cpty_id"])
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
        g = g.sort_values("amt_bucket")
        cum = g["n_txn"].cumsum() / g["n_txn"].sum()
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        for q in qs:
            b = g.loc[cum >= q, "amt_bucket"].iloc[0]
            row[f"p{int(q * 100)}"] = 10 ** ((b + 0.5) / 20)
        out.append(row)
    return pd.DataFrame(out)


def build_results(P, cube, hist, ties_raw):
    R = {"P": P, "ego_key": ego_key(P)}
    c = orient(cube, P)
    no_cp = c["cp_key"].isna()
    R["qa"] = (c.assign(n_no_cp=np.where(no_cp, c["n_txn"], 0), amt_no_cp=np.where(no_cp, c["amount"], 0.0))
                 .groupby("direction").agg(n_txn=("n_txn", "sum"), amount=("amount", "sum"),
                                           n_txn_no_cpty_id=("n_no_cp", "sum"), amount_no_cpty_id=("amt_no_cp", "sum")))
    R["self"] = c[c["direction"] == "SELF"]
    ex = c[c["direction"].isin(["IN", "OUT"]) & c["cp_key"].notna()].copy()
    if ex.empty:
        raise LookupError("Only self-transfers or unidentified counterparties in this window and category selection.")
    ex["type"] = ex["cp_type"].map(TYPE_LABEL)
    R["ex"] = ex
    R["ego_names"] = sorted({_safe(v) for v in c["ego_name"].dropna()})
    R["ego_mdms"] = sorted(c["ego_mdm"].dropna().unique().tolist())
    if P["mode"] == "mdm":
        own = pd.concat([c.loc[c["mdm_id_pays"].eq(P["id"]), "pnc_dep_acct_pays"],
                         c.loc[c["mdm_id_receives"].eq(P["id"]), "pnc_dep_acct_receives"]])
    else:
        own = pd.Series([P["id"]])
    R["ego_accts"] = sorted(own.dropna().unique().tolist())

    # ---- per counterparty -------------------------------------------------------------------
    R["cp_dir"] = ex.groupby(["direction", "cp_key"], as_index=False).agg(
        amount=("amount", "sum"), n_txn=("n_txn", "sum"), first_dt=("first_dt", "min"),
        last_dt=("last_dt", "max"), months_active=("txn_month", "nunique"))
    R["cp_rail"] = ex.groupby(["direction", "cp_key", "pay_rail"], as_index=False)[["amount", "n_txn"]].sum()
    attr = ex.groupby("cp_key").agg(cp_type=("cp_type", "first"), cp_mdm=("cp_mdm", "first"),
                                    cp_id=("cp_id", "first"),
                                    n_pnc_accts=("cp_pnc_acct", "nunique"))
    attr["cp_name"] = _mode_by(ex, "cp_key", "cp_name")
    attr["cp_bank"] = _mode_by(ex, "cp_key", "cp_bank")
    attr["cpty_type"] = _mode_by(ex, "cp_key", "cpty_type")
    attr["categories"] = ex.groupby("cp_key")["cat_code"].agg(lambda s: ", ".join(sorted(set(s.dropna()))))
    attr["same_customer"] = (attr["cp_type"] == "internal") & attr["cp_mdm"].isin(R["ego_mdms"])
    attr["cp_name"] = attr["cp_name"].map(lambda v: _safe(v) if isinstance(v, str) else v)
    attr["cp_bank"] = attr["cp_bank"].map(lambda v: _safe(v) if isinstance(v, str) else v)
    attr["label"] = attr["cp_name"].where(attr["cp_name"].notna(),
                                          "(" + attr["cpty_type"].fillna("no name").astype(str) + " " + attr["cp_id"].astype(str).str[-6:] + ")")
    R["attr"] = attr

    flow = R["cp_dir"].pivot_table(index="cp_key", columns="direction", values="amount",
                                   aggfunc="sum", fill_value=0.0).reindex(columns=["IN", "OUT"], fill_value=0.0)
    flow["total"] = flow["IN"] + flow["OUT"]
    R["flow"] = flow.sort_values("total", ascending=False)

    # ---- slices for the summaries -----------------------------------------------------------
    g = lambda keys: ex.groupby(keys, as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                                                          accounts=("cp_key", "nunique"))
    R["dir_type"], R["rail"] = g(["direction", "type"]), g(["direction", "pay_rail"])
    R["rail_type"], R["cat"] = g(["direction", "pay_rail", "type"]), g(["direction", "cat_code", "pay_rail"])
    R["month"], R["month_type"] = g(["direction", "txn_month", "pay_rail"]), g(["direction", "txn_month", "type"])
    R["bank"] = g(["direction", "cp_bank"])[lambda d: d["cp_bank"] != "PNC"]
    R["cpty_types"] = ex[ex["cp_type"] == "external"].assign(cpty_type=lambda d: d["cpty_type"].fillna("(none)")) \
                        .groupby(["direction", "cpty_type"], as_index=False) \
                        .agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"), accounts=("cp_key", "nunique"))
    R["own_accts"] = (ex.groupby(["ego_acct", "direction"], as_index=False)
                        .agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"), accounts=("cp_key", "nunique")))

    # ---- reconciliation: every slice carries the same dollars -------------------------------
    ref = float(ex["amount"].sum())
    for name in ["cp_dir", "cp_rail", "dir_type", "rail", "cat", "month", "own_accts"]:
        got = float(R[name]["amount"].sum())
        if abs(got - ref) > 1e-6 * max(ref, 1.0):
            raise AssertionError(f"Reconciliation failed: {name} = {got:,.2f}, expected {ref:,.2f}")

    # ---- amount distribution ------------------------------------------------------------------
    if hist is not None and len(hist):
        h = hist.copy()
        h["amt_bucket"] = pd.to_numeric(h["amt_bucket"]).astype(int)
        h = h[h["dir"].isin(["IN", "OUT"])]
        h["n_txn"] = pd.to_numeric(h["n_txn"])
        R["hist"] = h
        R["pct_dir"] = hist_percentiles(h, ["dir"]).rename(columns={"dir": "direction"})
        R["pct_rail"] = hist_percentiles(h, ["dir", "pay_rail"]).rename(columns={"dir": "direction"})
    else:
        R["hist"] = R["pct_dir"] = R["pct_rail"] = None

    # ---- ties between alters ---------------------------------------------------------------------
    R["ties"] = pd.DataFrame(columns=["src", "dst", "pay_rail", "cat_code", "amount", "n_txn", "first_dt", "last_dt"])
    if ties_raw is not None and len(ties_raw):
        t = _clean_ids(ties_raw.copy())
        s_int = t["mdm_id_pays"].notna() | t["pnc_dep_acct_pays"].notna()
        d_int = t["mdm_id_receives"].notna() | t["pnc_dep_acct_receives"].notna()
        t["src"] = node_keys(s_int, t["mdm_id_pays"], t["pnc_dep_acct_pays"], t["cpty_id"], P["level"])
        t["dst"] = node_keys(d_int, t["mdm_id_receives"], t["pnc_dep_acct_receives"], t["cpty_id"], P["level"])
        alters = set(R["flow"].index)
        t = t[t["src"].isin(alters) & t["dst"].isin(alters) & (t["src"] != t["dst"])
              & (t["src"] != R["ego_key"]) & (t["dst"] != R["ego_key"])]
        R["ties"] = t.groupby(["src", "dst", "pay_rail", "cat_code"], as_index=False).agg(
            amount=("amount", "sum"), n_txn=("n_txn", "sum"), first_dt=("first_dt", "min"), last_dt=("last_dt", "max"))
    return R


def counterparty_table(R, direction):
    d = R["cp_dir"][R["cp_dir"]["direction"] == direction].merge(R["attr"], left_on="cp_key", right_index=True)
    if d.empty:
        return d
    cr = R["cp_rail"][R["cp_rail"]["direction"] == direction].copy()
    cr["pct"] = cr["amount"] / cr.groupby("cp_key")["amount"].transform("sum")
    cr = cr.sort_values(["cp_key", "amount"], ascending=[True, False])
    cr["s"] = cr["pay_rail"] + " " + (cr["pct"] * 100).round().fillna(0).astype(int).astype(str) + "%"
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
    return {"accounts": len(s), "top 1": s[:1].sum(), "top 5": s[:5].sum(), "top 10": s[:10].sum(),
            "HHI": hhi, "effective n": 1 / hhi if hhi else np.nan}


# =============================================================================
# NETWORK (pyvis, self-contained HTML)
# =============================================================================
def _edge_title(sn, dn, g, first_dt, last_dt, prefix=""):
    lines = [prefix + f"{sn}  →  {dn}", f"{money(g['amount'].sum())} in {int(g['n_txn'].sum()):,} txns"]
    by = g.groupby("pay_rail")[["amount", "n_txn"]].sum().sort_values("amount", ascending=False)
    lines += [f"  {r}: {money(v['amount'])} ({int(v['n_txn']):,})" for r, v in by.iterrows()]
    lines.append(f"{first_dt} → {last_dt}")
    return "\n".join(lines)


def build_network(R, max_nodes, show_ties, height_px=780):
    P, A, flow, ek = R["P"], R["attr"], R["flow"], R["ego_key"]
    vis = flow.head(max_nodes)
    keep = set(vis.index)
    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False,
                  cdn_resources="in_line", bgcolor="#FFFFFF", font_color="#222222",
                  neighborhood_highlight=True, select_menu=True)

    ego_name = R["ego_names"][0] if R["ego_names"] else "(no name)"
    kind = "account" if P["mode"] == "acct" else "customer (MDM)"
    tip = [f"EGO {kind} {P['id']}", ego_name]
    if P["mode"] == "acct":
        tip.append(f"MDM: {', '.join(R['ego_mdms'])}")
    else:
        tip.append(f"{len(R['ego_accts'])} PNC account(s) active")
    tip += [f"Received {money(flow['IN'].sum())} from {int((flow['IN'] > 0).sum()):,}",
            f"Sent {money(flow['OUT'].sum())} to {int((flow['OUT'] > 0).sum()):,}"]
    net.add_node(ek, label=f"{trunc(ego_name, 32)}\n{P['id']}", title="\n".join(tip), shape="star", size=40,
                 color={"background": C_EGO, "border": "#8A3F06"}, x=0, y=0, fixed=True,
                 font={"size": 16, "face": "arial", "bold": True})

    fmax = max(float(vis["total"].max()), 1.0)
    for k, row in vis.iterrows():
        a = A.loc[k]
        internal = a["cp_type"] == "internal"
        same = bool(a["same_customer"])
        t = [a["label"], TYPE_LABEL[a["cp_type"]] + ("  — same customer as ego" if same else "")]
        if internal:
            t.append(f"Account: {a['cp_id']}" if P["level"] == "account"
                     else f"MDM: {a['cp_mdm']}  ({int(a['n_pnc_accts'])} account(s) seen)")
            if P["level"] == "account":
                t.append(f"MDM: {a['cp_mdm']}")
        else:
            t += [f"Counterparty id: {a['cp_id']}", f"Bank: {a['cp_bank'] if isinstance(a['cp_bank'], str) else '(not recorded)'}",
                  f"Type: {a['cpty_type'] if isinstance(a['cpty_type'], str) else '—'}"]
        t.append(f"Categories: {a['categories']}")
        if row["IN"] > 0:
            t.append(f"Paid the ego: {money(row['IN'])}")
        if row["OUT"] > 0:
            t.append(f"Paid by the ego: {money(row['OUT'])}")
        bg = C_SIB if same else (C_INT if internal else C_EXT)
        border = C_EGO if same else ("#123A6E" if internal else "#1D6B48")
        label = trunc(a["label"], 26) + (f"\n({int(a['n_pnc_accts'])} accts)" if internal and P["level"] == "customer" and a["n_pnc_accts"] > 1 else "")
        net.add_node(k, label=label, title="\n".join(t), shape="dot" if internal else "diamond",
                     size=float(10 + 28 * np.sqrt(row["total"] / fmax)), borderWidth=2.5 if same else 1.2,
                     color={"background": bg, "border": border, "highlight": {"background": bg, "border": "#000"}},
                     font={"size": 11, "face": "arial"})

    edges = []
    ex = R["ex"][R["ex"]["cp_key"].isin(keep)]
    for (d, k), g in ex.groupby(["direction", "cp_key"]):
        nm = A.loc[k, "label"]
        src, dst, sn, dn = (k, ek, nm, ego_name) if d == "IN" else (ek, k, ego_name, nm)
        rail = g.groupby("pay_rail")["amount"].sum().idxmax()
        edges.append((src, dst, g["amount"].sum(), rail, False,
                      _edge_title(sn, dn, g, g["first_dt"].min(), g["last_dt"].max())))
    n_ties = 0
    if show_ties and len(R["ties"]):
        tv = R["ties"][R["ties"]["src"].isin(keep) & R["ties"]["dst"].isin(keep)]
        for (s_, d_), g in tv.groupby(["src", "dst"]):
            rail = g.groupby("pay_rail")["amount"].sum().idxmax()
            edges.append((s_, d_, g["amount"].sum(), rail, True,
                          _edge_title(A.loc[s_, "label"], A.loc[d_, "label"], g, g["first_dt"].min(),
                                      g["last_dt"].max(), prefix="Between connected parties\n")))
            n_ties += 1
    if edges:
        la = np.log1p(np.array([max(e[2], 0) for e in edges]))
        lo, hi = la.min(), la.max()
        for (s_, d_, amt, rail, tie, title), v in zip(edges, la):
            w = 1.0 + 7.0 * ((v - lo) / (hi - lo) if hi > lo else 0.5)
            col = rail_color(rail)
            net.add_edge(s_, d_, title=title, width=float(w * (0.7 if tie else 1.0)), dashes=tie,
                         color={"color": col, "highlight": col, "opacity": 0.55 if tie else 0.9})

    net.set_options(json.dumps({
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.55}}, "arrowStrikethrough": False,
                  "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.18}},   # two-way pairs split
        "physics": {"solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -90, "centralGravity": 0.012,
                                         "springLength": 170, "springConstant": 0.06, "avoidOverlap": 0.5},
                    "stabilization": {"enabled": True, "iterations": 300}},
        "interaction": {"hover": True, "navigationButtons": True, "multiselect": True, "tooltipDelay": 60}}))
    doc = net.generate_html(notebook=False)
    # search drop-down lists raw node ids; relabel with names
    for n in net.nodes:
        k = n["id"]
        lab = f"★ {ego_name} · {P['id']} (ego)" if k == ek else \
            f"{A.loc[k, 'label']} · {A.loc[k, 'cp_id']} · {TYPE_LABEL[A.loc[k, 'cp_type']]}"
        doc = doc.replace(f'<option value="{k}">{k}</option>',
                          f'<option value="{_html.escape(k)}">{_html.escape(lab)}</option>', 1)
    doc = re.sub(r"<(link|script)[^>]*bootstrap[^>]*>(\s*</script>)?", "", doc)   # cosmetic CDN only
    css = ("<style>div.vis-tooltip{white-space:pre-line !important;font-family:arial;font-size:12px;"
           "max-width:520px}body{margin:0}</style>")
    doc = doc.replace("<body>", "<body>" + css, 1)
    shown = vis["total"].sum() / max(flow["total"].sum(), 1.0)
    caption = (f"Showing {len(vis):,} of {len(flow):,} connected parties ({shown:.0%} of the ego's dollar flow)"
               + (f" · {n_ties:,} edges between them" if show_ties else " · ties hidden")
               + (f" · ties searched among the top {R.get('n_searched', 0):,}" if R.get("n_searched", len(flow)) < len(flow) else ""))
    return doc, caption, sorted({e[3] for e in edges})


def legend_html(rails):
    sw = {"star": f"<span style='color:{C_EGO};font-size:17px'>★</span>",
          "int": f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;background:{C_INT}'></span>",
          "ext": f"<span style='display:inline-block;width:10px;height:10px;background:{C_EXT};transform:rotate(45deg)'></span>",
          "sib": f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;background:{C_SIB};border:2px solid {C_EGO}'></span>"}
    it = lambda s, t: f"<span style='margin-right:16px;white-space:nowrap'>{s} {t}</span>"
    rail_items = "".join(it(f"<span style='display:inline-block;width:22px;height:4px;background:{rail_color(r)};vertical-align:middle'></span>", r) for r in rails)
    return ("<div style='font-size:13px;line-height:1.9'>"
            + it(sw["star"], "ego") + it(sw["int"], "PNC customer") + it(sw["ext"], "external counterparty")
            + it(sw["sib"], "same customer as ego")
            + it("<span style='display:inline-block;width:26px;border-top:3px solid #555;vertical-align:middle'></span>", "payment with the ego")
            + it("<span style='display:inline-block;width:26px;border-top:3px dashed #999;vertical-align:middle'></span>", "payment between connected parties")
            + "<br><b>edge colour = main rail:</b> " + rail_items + "</div>")


# =============================================================================
# CHARTS (Altair ships with Streamlit)
# =============================================================================
def ch_monthly(R):
    m = R["month"].copy()
    m["signed"] = np.where(m["direction"] == "IN", m["amount"], -m["amount"])
    m["flow"] = m["direction"].map(DIR_LABEL)
    rails = m.groupby("pay_rail")["amount"].sum().sort_values(ascending=False).index.tolist()
    net = m.groupby("txn_month", as_index=False)["signed"].sum().rename(columns={"signed": "net"})
    bars = alt.Chart(m).mark_bar().encode(
        x=alt.X("txn_month:O", title=None),
        y=alt.Y("sum(signed):Q", title="received (+) / sent (−)", axis=MONEY_AXIS),
        color=alt.Color("pay_rail:N", scale=rail_scale(rails), legend=alt.Legend(title="rail", orient="top")),
        tooltip=["txn_month", "flow", "pay_rail", alt.Tooltip("amount:Q", format="$,.0f"), alt.Tooltip("n_txn:Q", format=",")])
    line = alt.Chart(net).mark_line(color="black", point=True).encode(
        x="txn_month:O", y="net:Q", tooltip=["txn_month", alt.Tooltip("net:Q", format="$,.0f", title="net in − out")])
    return alt.layer(bars, line).properties(height=360)


def ch_rail_mix(R):
    rows = []
    for d in ["IN", "OUT"]:
        sub = R["rail"][R["rail"]["direction"] == d]
        for col, lab in [("amount", "$"), ("n_txn", "# txns")]:
            for _, r in sub.iterrows():
                rows.append({"row": f"{DIR_LABEL[d]} · {lab}", "pay_rail": r["pay_rail"], "value": float(r[col])})
    df = pd.DataFrame(rows)
    rails = R["rail"].groupby("pay_rail")["amount"].sum().sort_values(ascending=False).index.tolist()
    order = [f"{DIR_LABEL[d]} · {l}" for d in ["IN", "OUT"] for l in ["$", "# txns"]]
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("row:N", sort=order, title=None),
        x=alt.X("value:Q", stack="normalize", axis=alt.Axis(format="%"), title="share"),
        color=alt.Color("pay_rail:N", scale=rail_scale(rails), legend=alt.Legend(title="rail", orient="top")),
        tooltip=["row", "pay_rail", alt.Tooltip("value:Q", format=",.0f")]).properties(height=190)


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


def ch_concentration(R, tin, tout):
    parts = []
    for t, d in [(tin, "IN"), (tout, "OUT")]:
        if len(t):
            parts.append(pd.DataFrame({"rank": np.arange(1, len(t) + 1), "cum_share": t["cum_share"].values,
                                       "flow": DIR_LABEL[d]}))
    df = pd.concat(parts)
    return alt.Chart(df).mark_line().encode(
        x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="counterparty rank (log)"),
        y=alt.Y("cum_share:Q", axis=alt.Axis(format="%"), title="cumulative share of dollars"),
        color=alt.Color("flow:N", scale=alt.Scale(range=["#1F5AA6", "#C0392B"])),
        tooltip=["flow", "rank", alt.Tooltip("cum_share:Q", format=".1%")]).properties(height=300)


def ch_hist(R, d):
    h = R["hist"][R["hist"]["dir"] == d].copy()
    h["amount_mid"] = 10 ** ((h["amt_bucket"] + 0.5) / 20)
    rails = R["rail"].groupby("pay_rail")["amount"].sum().sort_values(ascending=False).index.tolist()
    return alt.Chart(h).mark_line(interpolate="step-after", point=False).encode(
        x=alt.X("amount_mid:Q", scale=alt.Scale(type="log"), axis=MONEY_AXIS, title="amount per transaction (log)"),
        y=alt.Y("n_txn:Q", title="transactions"),
        color=alt.Color("pay_rail:N", scale=rail_scale(rails), legend=alt.Legend(orient="top", title=None)),
        tooltip=["pay_rail", alt.Tooltip("amount_mid:Q", format="$,.0f", title="≈ amount"), alt.Tooltip("n_txn:Q", format=",")]
    ).properties(title=f"Transaction size — {DIR_LABEL[d].lower()}", height=280)


def ch_bars(df, cat, title, color, n=12):
    d = df.sort_values("amount", ascending=False).head(n).copy()
    d[cat] = d[cat].fillna("(not recorded)")
    return alt.Chart(d).mark_bar(color=color).encode(
        y=alt.Y(f"{cat}:N", sort=list(d[cat]), title=None), x=alt.X("amount:Q", axis=MONEY_AXIS, title=None),
        tooltip=[cat, alt.Tooltip("amount:Q", format="$,.0f"), alt.Tooltip("n_txn:Q", format=","),
                 alt.Tooltip("accounts:Q", format=",")]).properties(title=title, height=max(180, 24 * len(d)))


def ch_active(R):
    m = R["month_type"].copy()
    m["series"] = m["direction"].map({"IN": "senders", "OUT": "receivers"}) + " — " + m["type"]
    return alt.Chart(m).mark_line(point=True).encode(
        x=alt.X("txn_month:O", title=None), y=alt.Y("accounts:Q", title="active counterparties"),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=[f"{a} — {TYPE_LABEL[t]}" for a in ["senders", "receivers"] for t in ["internal", "external"]],
            range=[C_INT, C_EXT, "#7FA6D6", "#8FD0B0"]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("direction:N", legend=None),
        tooltip=["txn_month", "series", "accounts"]).properties(height=280)


# =============================================================================
# TABLE HELPERS
# =============================================================================
def show(df, money_cols=(), pct_cols=(), int_cols=(), height=None):
    if df is None or len(df) == 0:
        st.caption("none")
        return
    fmt = {c: "${:,.0f}" for c in money_cols if c in df.columns}
    fmt.update({c: "{:.1%}" for c in pct_cols if c in df.columns})
    fmt.update({c: "{:,.0f}" for c in int_cols if c in df.columns})
    kw = {"height": height} if height else {}
    st.dataframe(df.style.format(fmt, na_rep=""), **_wide(st.dataframe), **kw)


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

_today = date.today()
_end_default = _today.replace(day=1) - timedelta(days=1)                       # end of last month
_start_default = date(*(int(x) for x in _month_shift(_end_default, -(DEFAULT_WINDOW_MONTHS - 1)).split("-")), 1)

with st.sidebar:
    st.header("Lookup")
    with st.form("lookup"):
        mode_lbl = st.radio("Look up by", ["Deposit account", "MDM id"], horizontal=True)
        ego_id = st.text_input("Id", placeholder="PNC deposit account or MDM id").strip()
        window = st.date_input("Window", value=(_start_default, _end_default), min_value=DATA_START,
                               max_value=_today, help="Each lookup scans the window; start narrow.")
        level_lbl = st.radio("PNC parties drawn as", ["Same as lookup", "Accounts", "Customers (MDM)"],
                             help="Accounts: one node per deposit account. Customers: accounts of the same MDM id merged.")
        codes = st.multiselect("Categories", CODES, default=[c for c in CODES if c not in DEFAULT_OFF],
                               format_func=code_label)
        dedup = st.checkbox("Remove 5C/6C RTP duplicates", value=True,
                            help=f"Anti-join on {DEDUP_TABLE}: the 6C twin of a 5C inbound RTP payment is dropped.")
        with st.expander("Advanced"):
            alter_cap = st.number_input("Parties searched for ties between them", 0, ALTER_CAP_MAX,
                                        ALTER_CAP_DEFAULT, step=100,
                                        help="Top parties by $ with the ego. 0 skips the ties query.")
            ext_ties = st.checkbox("Include ties to external counterparties", value=True)
            want_hist = st.checkbox("Transaction-size distribution (one extra scan)", value=True)
        submitted = st.form_submit_button("Build ego network", type="primary", **_wide(st.form_submit_button))

    st.header("Display")
    max_vis = st.slider("Parties drawn", 10, 600, MAX_VIS_DEFAULT, step=10)
    show_ties = st.checkbox("Show payments between connected parties", value=True)
    top_n = st.slider("Rows in ranked tables", 10, 500, 25, step=5)

if submitted:
    errs = []
    if not ID_RE.match(ego_id or ""):
        errs.append("Id must be 1–64 letters, digits, '.', '_' or '-'.")
    if not (isinstance(window, (tuple, list)) and len(window) == 2):
        errs.append("Pick both a start and an end date.")
    if not codes:
        errs.append("Select at least one category.")
    if errs:
        for e in errs:
            st.sidebar.error(e)
    else:
        mode = "acct" if mode_lbl == "Deposit account" else "mdm"
        level = {"Accounts": "account", "Customers (MDM)": "customer"}.get(
            level_lbl, "account" if mode == "acct" else "customer")
        st.session_state["P"] = dict(mode=mode, id=ego_id, start=window[0].isoformat(), end=window[1].isoformat(),
                                     level=level, codes=tuple(sorted(codes)), dedup=dedup,
                                     alter_cap=int(alter_cap), ext_ties=ext_ties, hist=want_hist)

P = st.session_state.get("P")
st.title("PKG · Ego network")
if P is None:
    st.info("Enter a deposit account or an MDM id in the sidebar and press **Build ego network**.")
    st.stop()

# ---- queries (cached by SQL text) -----------------------------------------------------------------
log = []
try:
    with st.spinner(f"Q1/3 · transactions touching {P['id']} ({P['start']} → {P['end']}) …"):
        s1 = sql_ego_cube(P)
        cube, t1 = run_sql(s1)
        log.append(("Q1 ego cube", s1, t1, len(cube)))
    if cube.empty:
        with st.spinner("No rows — searching every id column for near matches (one more scan) …"):
            s0 = sql_near_match(P)
            near, t0 = run_sql(s0)
        st.warning(f"No transactions for {'account' if P['mode'] == 'acct' else 'MDM id'} **{P['id']}** "
                   f"in {P['start']} → {P['end']} with the selected categories.")
        if len(near):
            st.write("Near matches (whitespace / leading zeros ignored, all categories):")
            show(near, int_cols=["n_txn"])
        else:
            st.write("No near matches on any account, MDM or counterparty id column in this window.")
        st.stop()

    hist = None
    if P["hist"]:
        with st.spinner("Q2/3 · transaction-size histogram …"):
            s2 = sql_amount_hist(P)
            hist, t2 = run_sql(s2)
            log.append(("Q2 amount histogram", s2, t2, len(hist)))

    R = build_results(P, cube, hist, None)
    search = R["flow"].index[: P["alter_cap"]]
    A = R["attr"].loc[search]
    int_ids = (A.loc[A["cp_type"] == "internal", "cp_id" if P["level"] == "account" else "cp_mdm"]
                .dropna().unique().tolist())
    ext_ids = A.loc[A["cp_type"] == "external", "cp_id"].dropna().unique().tolist() if P["ext_ties"] else []
    R["n_searched"] = len(search)
    ties_raw = None
    if int_ids:
        with st.spinner(f"Q3/3 · payments between the top {len(search):,} connected parties …"):
            s3 = sql_ties(P, sorted(int_ids), sorted(ext_ids))
            ties_raw, t3 = run_sql(s3)
            log.append(("Q3 alter ties", s3, t3, len(ties_raw)))
        R = build_results(P, cube, hist, ties_raw)
        R["n_searched"] = len(search)
except Exception as ex:
    st.error(f"{type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
    st.stop()

# ---- header metrics -------------------------------------------------------------------------------
fl, dt = R["flow"], R["dir_type"]
n_of = lambda d, t: int(dt.loc[(dt["direction"] == d) & (dt["type"] == TYPE_LABEL[t]), "accounts"].sum())
ego_name = R["ego_names"][0] if R["ego_names"] else "(no name)"
kind = "Account" if P["mode"] == "acct" else "Customer"
st.markdown(f"**{kind} {P['id']}** · {ego_name}"
            + (f" · MDM {', '.join(R['ego_mdms'])}" if P["mode"] == "acct" else f" · {len(R['ego_accts'])} PNC account(s)")
            + f" · {P['start']} → {P['end']} · nodes = {'accounts' if P['level'] == 'account' else 'customers'}"
            + (" · 5C/6C deduplicated" if P["dedup"] else ""))
m = st.columns(6)
m[0].metric("Received", money(fl["IN"].sum()))
m[1].metric("Sent", money(fl["OUT"].sum()))
m[2].metric("Net (in − out)", money(fl["IN"].sum() - fl["OUT"].sum()))
m[3].metric("Transactions", f"{int(R['ex']['n_txn'].sum()):,}")
m[4].metric("PNC parties", f"{int((R['attr']['cp_type'] == 'internal').sum()):,}")
m[5].metric("External parties", f"{int((R['attr']['cp_type'] == 'external').sum()):,}")

T_in, T_out = counterparty_table(R, "IN"), counterparty_table(R, "OUT")
tab_names = ["Network", "Senders & receivers", "Rails & categories", "Over time", "External banks",
             "Ties", "Summary & QA", "Queries"]
if P["mode"] == "mdm":
    tab_names.insert(6, "Own accounts")
tabs = dict(zip(tab_names, st.tabs(tab_names)))

with tabs["Network"]:
    doc, caption, rails_drawn = build_network(R, max_vis, show_ties)
    st.markdown(legend_html(rails_drawn), unsafe_allow_html=True)
    st.caption(caption + " · arrows point payer → payee · click a node to isolate its neighbourhood · "
               "use the drop-down to find a party")
    if hasattr(st, "iframe"):            # newer Streamlit; components.html is deprecated there
        st.iframe(doc, height=860)
    else:
        components.html(doc, height=860, scrolling=False)

CP_COLS = ["name", "type", "id", "mdm_id", "bank", "cpty_type", "categories", "amount", "share", "cum_share",
           "n_txn", "avg_txn", "rail_mix", "months_active", "first_dt", "last_dt", "same_customer", "two_way"]
CP_FMT = dict(money_cols=["amount", "avg_txn"], pct_cols=["share", "cum_share"], int_cols=["n_txn", "months_active"])

with tabs["Senders & receivers"]:
    c1, c2 = st.columns(2)
    if len(T_in):
        c1.altair_chart(ch_top(T_in, "Top senders to the ego"), **_wide(st.altair_chart))
    if len(T_out):
        c2.altair_chart(ch_top(T_out, "Top receivers from the ego"), **_wide(st.altair_chart))
    st.subheader(f"Senders — paid the ego ({len(T_in):,})")
    show(T_in[CP_COLS].head(top_n) if len(T_in) else T_in, **CP_FMT)
    st.subheader(f"Receivers — paid by the ego ({len(T_out):,})")
    show(T_out[CP_COLS].head(top_n) if len(T_out) else T_out, **CP_FMT)
    both = sorted(set(T_in["cp_key"]) & set(T_out["cp_key"])) if len(T_in) and len(T_out) else []
    st.subheader(f"Two-way parties — both sent to and received from the ego ({len(both):,})")
    if both:
        a, b = T_in.set_index("cp_key").loc[both], T_out.set_index("cp_key").loc[both]
        tw = pd.DataFrame({"name": a["name"], "type": a["type"], "id": a["id"], "bank": a["bank"],
                           "received_from": a["amount"], "sent_to": b["amount"]})
        tw["net_to_ego"] = tw["received_from"] - tw["sent_to"]
        tw["gross"] = tw["received_from"] + tw["sent_to"]
        tw = tw.sort_values("gross", ascending=False).reset_index(drop=True)
        show(tw.head(top_n), money_cols=["received_from", "sent_to", "net_to_ego", "gross"])
    else:
        tw = pd.DataFrame()
        st.caption("none")

with tabs["Rails & categories"]:
    st.altair_chart(ch_rail_mix(R), **_wide(st.altair_chart))
    rails_t = with_shares(R["rail"])
    if R["pct_rail"] is not None:
        rails_t = rails_t.merge(R["pct_rail"].assign(direction=lambda d: d["direction"].map(DIR_LABEL)),
                                on=["direction", "pay_rail"], how="left")
    rails_t = rails_t.sort_values(["direction", "amount"], ascending=[True, False]).set_index(["direction", "pay_rail"])
    st.subheader("Rails by direction")
    show(rails_t, money_cols=["amount", "avg_txn", "p10", "p50", "p90", "p99"], pct_cols=["share_amt", "share_txn"],
         int_cols=["n_txn", "accounts"])
    st.caption("p10 … p99: per-transaction amount percentiles from a log-bucket histogram (±6%).")
    rt = R["rail_type"].pivot_table(index=["direction", "pay_rail"], columns="type", values="amount",
                                    aggfunc="sum", fill_value=0.0)
    rt = rt.reindex(columns=list(TYPE_COLOR), fill_value=0.0)
    rt["external_share"] = rt[TYPE_LABEL["external"]] / rt.sum(axis=1)
    rt = rt.reset_index().assign(direction=lambda d: d["direction"].map(DIR_LABEL)).set_index(["direction", "pay_rail"])
    st.subheader("Rail × party type (dollars)")
    show(rt, money_cols=list(TYPE_COLOR), pct_cols=["external_share"])
    cat_t = with_shares(R["cat"])
    cat_t["category"] = cat_t["cat_code"].map(code_label)
    cat_t = cat_t.sort_values(["direction", "amount"], ascending=[True, False]) \
                 .set_index(["direction", "category"])[["pay_rail", "amount", "share_amt", "n_txn", "share_txn", "avg_txn", "accounts"]]
    st.subheader("Categories")
    show(cat_t, money_cols=["amount", "avg_txn"], pct_cols=["share_amt", "share_txn"], int_cols=["n_txn", "accounts"])
    if R["hist"] is not None:
        c1, c2 = st.columns(2)
        c1.altair_chart(ch_hist(R, "IN"), **_wide(st.altair_chart))
        c2.altair_chart(ch_hist(R, "OUT"), **_wide(st.altair_chart))

with tabs["Over time"]:
    st.altair_chart(ch_monthly(R), **_wide(st.altair_chart))
    st.altair_chart(ch_active(R), **_wide(st.altair_chart))
    mo = R["month_type"].pivot_table(index="txn_month", columns=["direction", "type"],
                                     values=["amount", "accounts"], aggfunc="sum", fill_value=0)
    monthly = pd.DataFrame(index=mo.index)
    for d in ["IN", "OUT"]:
        monthly[f"{DIR_LABEL[d].lower()} $"] = sum(mo[("amount", d, TYPE_LABEL[t])] for t in ["internal", "external"]
                                                    if ("amount", d, TYPE_LABEL[t]) in mo.columns)
    monthly["net $"] = monthly.get("received $", 0) - monthly.get("sent $", 0)
    for d, who in [("IN", "senders"), ("OUT", "receivers")]:
        for t, s in [("internal", "PNC"), ("external", "ext")]:
            col = ("accounts", d, TYPE_LABEL[t])
            monthly[f"{who} ({s})"] = mo[col] if col in mo.columns else 0
    st.subheader("Monthly")
    show(monthly, money_cols=["received $", "sent $", "net $"],
         int_cols=[c for c in monthly.columns if "$" not in c])

with tabs["External banks"]:
    bk = R["bank"]
    c1, c2 = st.columns(2)
    for col, d in [(c1, "IN"), (c2, "OUT")]:
        sub = bk[bk["direction"] == d]
        if len(sub):
            col.altair_chart(ch_bars(sub, "cp_bank", f"Counterparty banks — {DIR_LABEL[d].lower()}", C_EXT), **_wide(st.altair_chart))
    bank_t = bk.copy()
    if len(bank_t):
        bank_t["share_of_external"] = bank_t["amount"] / bank_t.groupby("direction")["amount"].transform("sum")
        bank_t["direction"] = bank_t["direction"].map(DIR_LABEL)
        bank_t = bank_t.sort_values(["direction", "amount"], ascending=[True, False]).set_index(["direction", "cp_bank"])
    st.subheader("Banks of external counterparties")
    show(bank_t, money_cols=["amount"], pct_cols=["share_of_external"], int_cols=["n_txn", "accounts"])
    ct = R["cpty_types"].copy()
    if len(ct):
        ct["direction"] = ct["direction"].map(DIR_LABEL)
        ct = ct.set_index(["direction", "cpty_type"])
    st.subheader("External counterparty types")
    show(ct, money_cols=["amount"], int_cols=["n_txn", "accounts"])
    st.caption("Inbound checks (7C) carry the drawer bank but no payer name; 6C rows never carry a bank.")

with tabs["Ties"]:
    ties = R["ties"]
    st.caption(f"Searched the top {R['n_searched']:,} of {len(fl):,} connected parties by dollars with the ego"
               + ("" if P["ext_ties"] else "; PNC ↔ PNC only")
               + ". External → external payments are not in the source table, so those ties are never observable.")
    if len(ties):
        tp = ties.groupby(["src", "dst"], as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"))
        tp["kind"] = np.where(tp["src"].str.startswith("EXT:") | tp["dst"].str.startswith("EXT:"),
                              "PNC ↔ external", "PNC ↔ PNC")
        st.subheader("Payments between connected parties")
        show(tp.groupby("kind").agg(directed_pairs=("amount", "size"), amount=("amount", "sum"), n_txn=("n_txn", "sum")),
             money_cols=["amount"], int_cols=["directed_pairs", "n_txn"])
        deg = pd.concat([tp[["src", "dst", "amount"]].rename(columns={"src": "k", "dst": "o"}),
                         tp[["dst", "src", "amount"]].rename(columns={"dst": "k", "src": "o"})])
        emb = deg.groupby("k").agg(tie_partners=("o", "nunique"), tie_amount=("amount", "sum"))
        emb = emb.join(R["attr"][["label", "cp_type", "cp_id"]]).join(fl["total"].rename("flow_with_ego"))
        emb["type"] = emb["cp_type"].map(TYPE_LABEL)
        emb = emb.sort_values(["tie_partners", "tie_amount"], ascending=False).reset_index(drop=True)
        emb = emb[["label", "type", "cp_id", "tie_partners", "tie_amount", "flow_with_ego"]].rename(
            columns={"label": "name", "cp_id": "id"})
        st.subheader("Most embedded parties (most tie partners inside the ego network)")
        show(emb.head(top_n), money_cols=["tie_amount", "flow_with_ego"], int_cols=["tie_partners"])
        lab = R["attr"]["label"]
        tp_named = tp.assign(src_name=tp["src"].map(lab), dst_name=tp["dst"].map(lab)) \
                     .sort_values("amount", ascending=False)[["src_name", "dst_name", "kind", "amount", "n_txn", "src", "dst"]]
        st.subheader("Largest ties")
        show(tp_named.head(top_n).reset_index(drop=True), money_cols=["amount"], int_cols=["n_txn"])
    else:
        tp_named = emb = pd.DataFrame()
        st.caption("No payments between connected parties in this window.")

if "Own accounts" in tabs:
    with tabs["Own accounts"]:
        oa = R["own_accts"].pivot_table(index="ego_acct", columns="direction", values=["amount", "n_txn", "accounts"],
                                        aggfunc="sum", fill_value=0)
        oa.columns = [f"{DIR_LABEL[d].lower()} {v}" for v, d in oa.columns]
        oa = oa.sort_values([c for c in oa.columns if "amount" in c], ascending=False)
        st.subheader(f"Which of the customer's accounts carry the flow ({len(oa):,})")
        show(oa, money_cols=[c for c in oa.columns if "amount" in c], int_cols=[c for c in oa.columns if "amount" not in c])
        sf = R["self"]
        st.caption(f"Transfers between the customer's own accounts (excluded from the network): "
                   f"{int(sf['n_txn'].sum()):,} txns, {money(sf['amount'].sum())}.")

with tabs["Summary & QA"]:
    ht = with_shares(R["dir_type"])
    ht = ht.sort_values(["direction", "type"]).set_index(["direction", "type"])
    st.subheader("PNC customers vs external counterparties")
    show(ht, money_cols=["amount", "avg_txn"], pct_cols=["share_amt", "share_txn"], int_cols=["n_txn", "accounts"])
    if R["pct_dir"] is not None:
        st.subheader("Transaction size")
        show(R["pct_dir"].assign(direction=lambda d: d["direction"].map(DIR_LABEL)).set_index("direction"),
             money_cols=["p10", "p50", "p90", "p99"])
    st.subheader("Concentration")
    conc = pd.DataFrame({DIR_LABEL["IN"]: concentration(T_in), DIR_LABEL["OUT"]: concentration(T_out)}).T
    show(conc, pct_cols=["top 1", "top 5", "top 10"], int_cols=["accounts"])
    st.caption("effective n = 1 / HHI: the number of equal-sized counterparties giving the same concentration.")
    if len(T_in) or len(T_out):
        st.altair_chart(ch_concentration(R, T_in, T_out), **_wide(st.altair_chart))
    st.subheader("Data quality")
    show(R["qa"], money_cols=["amount", "amount_no_cpty_id"], int_cols=["n_txn", "n_txn_no_cpty_id"])
    st.caption("SELF = the ego paying itself (own-account transfers in MDM mode), kept out of the network. "
               "no_cpty_id = external side with neither counterparty id; counted here, not drawn."
               + (" 5C/6C dedup map covers the months it was built for; duplicates outside it remain."
                  if P["dedup"] else " 5C/6C dedup is OFF: inbound RTP may be double-counted."))

with tabs["Queries"]:
    for name, sql, secs, n in log:
        with st.expander(f"{name} · {n:,} rows · {secs:.1f}s (cached results show the original time)"):
            st.code(sql, language="sql")

with st.sidebar:
    st.header("Export")
    tables = {"senders": T_in, "receivers": T_out, "two_way": tw, "rails": rails_t, "categories": cat_t,
              "monthly": monthly, "banks": bank_t, "ties": tp_named, "embedded": emb}
    st.download_button("Download tables + network (zip)", zip_bytes(tables, doc),
                       file_name=f"ego_{P['mode']}_{P['id']}_{P['start']}_{P['end']}.zip",
                       mime="application/zip", **_wide(st.download_button))
