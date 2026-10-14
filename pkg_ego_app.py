"""
PKG · Ego Network Explorer (Streamlit)

    streamlit run pkg_ego_app.py

Three ways to pick the target, all drawn from the payment staging table:

  * Customer / account  one deposit account or MDM id
  * Upload list         a file of deposit accounts or MDM ids; every id is a target
  * NAICS               every PNC customer whose NAICS (6 digits, valid) matches; the network is
                        drawn between industries: other NAICS codes, one node for PNC customers
                        without a valid NAICS, one node for all external counterparties

Data comes from Impala through `dbi.db_get_query` (pandas out). Each query is cached by its SQL
text, so the Show options never re-query.

    customer / list   Q1 pairs (target x party x rail)   Q2 flow (month x direction x rail x size)
                      Q3 payments between connected parties
    NAICS             Q1 groups (direction x other NAICS) Q2 flow   Q3 member customers
                      Q4 payments between the top other NAICS (and with counterparties)

Identity rules carried over from the PKG edge build:
  * a side is a PNC customer when it has an mdm_id or a PNC deposit account; otherwise external
  * counterparty id = unq_cpty_id for 3B/3C/7B/9B (unq_cpty_acct_id is only a routing number or
    null there), unq_cpty_acct_id elsewhere, the other as fallback
  * 6C twins listed in pkg_dedup_5c6c are removed
  * NAICS comes from pkg_cust_dim (one row per mdm_id, current state); only naics_status = 'valid'

The network is a small custom component (vis-network, the library pyvis wraps) so that clicking a
node can send it back to the lookup.
"""
import csv
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
CUST_DIM_TABLE = "bdahd01p_dlcdi1_cdi_tm.pkg_cust_dim"   # mdm_id, customer_name, party_type, naics_*
DATA_START = date(2023, 7, 1)            # first trans_dt partition
DEFAULT_WINDOW_MONTHS = 3                # every query scans the window; widen deliberately
LIST_MAX_IDS = 2000                      # uploaded ids go into SQL IN-lists
TIE_SEARCH_CAP = 500                     # per side: top PNC customers and top counterparties by $
TIE_NAICS_CAP = 40                       # NAICS mode: other NAICS searched for links between them
MAX_TARGETS_DRAWN = 300                  # list mode: targets drawn (top by $), all kept in the tables
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
# payment_rail value -> (label, category numbers carrying it). The rail filter is applied on the
# category number because category is a partition column (Impala prunes on it) and payment_rail
# is not; the mapping is 1:1 (1-3 ACH, 4 wire, 5 RTP PRT, 6 RTP P2P, 7 check, 8 debit card, 9 PCARD).
RAIL_DEF = {
    "ACH": ("ACH", "123"),
    "WIRE": ("Wire", "4"),
    "RTP_PRT": ("RTP (PRT)", "5"),
    "RTP_P2P": ("RTP (P2P)", "6"),
    "CHECK": ("Check", "7"),
    "DEBIT_CARD_SIGNATURE": ("Debit card", "8"),
    "PCARD": ("PCARD", "9"),
}
# Debit card is ~42% of rows and ~0.5% of dollars, and its merchants flood the picture.
DEFAULT_RAILS = [r for r in RAIL_DEF if r != "DEBIT_CARD_SIGNATURE"]


def rail_label(r):
    return RAIL_DEF.get(r, (r if isinstance(r, str) and r else "Unknown",))[0]


CODE_EXPR = "regexp_extract(p.category, '^([0-9]+[A-Z][0-9]*)', 1)"
CPTY_ID_EXPR = (f"CASE WHEN {CODE_EXPR} IN ('3B','3C','7B','9B') "
                "THEN COALESCE(NULLIF(p.unq_cpty_id, ''), NULLIF(p.unq_cpty_acct_id, '')) "
                "ELSE COALESCE(NULLIF(p.unq_cpty_acct_id, ''), NULLIF(p.unq_cpty_id, '')) END")
PNC_PAYS = "(NULLIF(p.mdm_id_pays, '') IS NOT NULL OR NULLIF(p.pnc_dep_acct_pays, '') IS NOT NULL)"
PNC_RECS = "(NULLIF(p.mdm_id_receives, '') IS NOT NULL OR NULLIF(p.pnc_dep_acct_receives, '') IS NOT NULL)"
BUCKET_EXPR = "CAST(FLOOR(LOG10(GREATEST(CAST(p.trans_amt AS DOUBLE), 0.01)) * 20) AS INT)"
RAIL_EXPR = "COALESCE(NULLIF(p.payment_rail, ''), 'UNKNOWN')"

# ---- visual vocabulary -----------------------------------------------------------------------
C_EGO, C_INT, C_EXT, C_EDGE = "#F58025", "#1F5AA6", "#2E9E6B", "#8A8A8A"
KIND_LABEL = {"pnc": "PNC customer", "naics": "Other NAICS", "unknown": "PNC, NAICS unknown",
              "cpty": "Counterparty", "target": "Target"}
KIND_COLOR = {"PNC customer": C_INT, "Other NAICS": C_INT, "PNC, NAICS unknown": "#7FA6D6",
              "Counterparty": C_EXT, "Target": C_EGO}
RAIL_COLORS = {"ACH": "#1F77B4", "Wire": "#D62728", "RTP (PRT)": "#9467BD", "RTP (P2P)": "#C5B0D5",
               "Check": "#8C564B", "Debit card": "#E377C2", "PCARD": "#17BECF", "Unknown": "#BBBBBB"}
_SPARE = ["#BCBD22", "#FF7F0E", "#AEC7E8", "#98DF8A", "#FFBB78", "#9EDAE5"]

SEARCH_MODES = ["Customer / account", "NAICS", "Upload list"]
MODE_OF = {"Customer / account": "single", "NAICS": "naics", "Upload list": "list"}
ID_KINDS = ["Deposit account", "MDM id"]
LIST_KINDS = ["Deposit accounts", "MDM ids"]
ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")
NAICS_RE = re.compile(r"^\d{6}$")
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


def dir_label(d, mode):
    return {"IN": "Received", "OUT": "Sent",
            "WITHIN": "Within the NAICS" if mode == "naics" else "Between targets"}.get(d, d)


# =============================================================================
# ID FILE PARSING
# =============================================================================
_HINTS = {"acct": ("acct", "account", "dep"), "mdm": ("mdm",)}


def _idlike(s):
    return bool(ID_RE.match(s)) and any(ch.isdigit() for ch in s)


def parse_id_file(raw, kind):
    """Ids from a .csv / .txt / .tsv upload, whatever its layout:
    one id per line; ids separated by commas, semicolons, tabs, pipes or spaces (on one line or
    many); or a table with a header, where the id column is picked by name (mdm / account / acct /
    dep / id) or, failing that, as the column that looks most like ids. Values stay strings, so
    leading zeros survive; quotes and blank lines are ignored; duplicates are dropped in order."""
    rep = {}
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    lines = [l for l in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
    if not lines:
        return [], {"error": "The file is empty."}
    sample = lines[:200]
    delim, best = None, 0
    for d in [",", "\t", ";", "|"]:
        n_lines = sum(1 for l in sample if d in l)
        if n_lines >= max(1, int(0.8 * len(sample))) and sum(l.count(d) for l in sample) > best:
            delim, best = d, sum(l.count(d) for l in sample)
    if delim:
        rows = list(csv.reader(lines, delimiter=delim))
    else:
        rows = [l.split() for l in lines]
    rows = [[c.strip().strip('"').strip("'").strip() for c in r] for r in rows]
    rep["layout"] = {",": "comma separated", "\t": "tab separated", ";": "semicolon separated",
                     "|": "pipe separated", None: "one or more ids per line"}[delim]
    first = [c for c in rows[0] if c]
    header = bool(first) and not any(_idlike(c) for c in first)
    body = rows[1:] if header else rows
    ncol = max(len(r) for r in rows)
    col = None
    if ncol > 1:
        cells = [[(r[j] if j < len(r) else "") for r in body] for j in range(ncol)]
        share = [np.mean([_idlike(v) for v in cells[j] if v] or [0]) for j in range(ncol)]
        if header:
            names = [(rows[0][j] if j < len(rows[0]) else "").lower() for j in range(ncol)]
            for hints in (_HINTS[kind], ("id",)):
                hit = [j for j, n in enumerate(names) if any(h in n for h in hints)]
                if hit:
                    col = max(hit, key=lambda j: share[j])
                    break
        filled = [s for j, s in enumerate(share) if any(cells[j])]
        # a headerless block where every filled column is ids is a wrapped list; otherwise one column holds the ids
        if col is None and not (all(s >= 0.9 for s in filled) and not header):
            col = int(np.argmax(share))
    if col is None:      # a list of ids, possibly several per line
        values = [c for r in body for c in r if c]
        rep["column"] = "all values"
    else:
        values = [r[col] for r in body if col < len(r) and r[col]]
        rep["column"] = rows[0][col] if header else f"column {col + 1}"
    rep["header"] = rows[0] if header else None
    valid = [v for v in values if _idlike(v)]
    invalid = [v for v in values if not _idlike(v)]
    seen, ids = set(), []
    for v in valid:
        if v not in seen:
            seen.add(v)
            ids.append(v)
    rep.update(n_values=len(values), n_ids=len(ids), n_duplicates=len(valid) - len(ids),
               n_invalid=len(invalid), invalid_sample=invalid[:5])
    return ids, rep


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


def _tcols(P):
    """Columns that identify a target on the paying / receiving side (customer and list modes)."""
    return (("pnc_dep_acct_pays", "pnc_dep_acct_receives") if P["kind"] == "acct"
            else ("mdm_id_pays", "mdm_id_receives"))


_INNER = f"""
    SELECT p.trans_dt, CAST(p.trans_amt AS DOUBLE) AS amt, {RAIL_EXPR} AS pay_rail, p.cpty_type,
           NULLIF(p.mdm_id_pays, '') AS mdm_id_pays, NULLIF(p.mdm_id_receives, '') AS mdm_id_receives,
           NULLIF(p.pnc_dep_acct_pays, '') AS pnc_dep_acct_pays,
           NULLIF(p.pnc_dep_acct_receives, '') AS pnc_dep_acct_receives,
           p.customer_name_pays, p.customer_name_receives, p.cpty_name, p.cpty_fin_entity_name,
           {CPTY_ID_EXPR} AS cpty_id"""

_CUBE_SELECT = """
SELECT b.pay_rail, b.cpty_type,
       b.mdm_id_pays, b.mdm_id_receives, b.pnc_dep_acct_pays, b.pnc_dep_acct_receives, b.cpty_id,
       MAX(b.customer_name_pays) AS name_pays, MAX(b.customer_name_receives) AS name_receives,
       MAX(b.cpty_name) AS cpty_name, MAX(b.cpty_fin_entity_name) AS cpty_bank,
       SUM(b.amt) AS amount, COUNT(*) AS n_txn, MIN(b.trans_dt) AS first_dt, MAX(b.trans_dt) AS last_dt"""


# ---- customer / list ---------------------------------------------------------------------------
def sql_pairs(P):
    frm, where = _scope(P)
    a, b = _tcols(P)
    T = _in(P["ids"])
    where.append(f"(p.{a} IN ({T}) OR p.{b} IN ({T}))")
    return (_CUBE_SELECT + f"\nFROM ({_INNER}\n    {frm}\n    WHERE " + "\n      AND ".join(where) + "\n) b"
            + "\nGROUP BY 1, 2, 3, 4, 5, 6, 7")


def sql_flow_ids(P):
    """Month x direction x kind of the other party x rail x amount bucket. Feeds monthly flow, rail
    mix, transaction-size percentiles and the totals check. 20 log-buckets per decade (×1.12)."""
    frm, where = _scope(P)
    a, b = _tcols(P)
    T = _in(P["ids"])
    tp, tr = f"p.{a} IN ({T})", f"p.{b} IN ({T})"
    where.append(f"({tp} OR {tr})")
    other = lambda pnc: (f"(CASE WHEN {pnc} THEN 'pnc' WHEN {CPTY_ID_EXPR} IS NULL THEN 'cpty_noid' "
                         f"ELSE 'cpty' END)")
    return f"""
SELECT x.txn_month, x.dir, x.other_kind, x.pay_rail, x.amt_bucket, COUNT(*) AS n_txn, SUM(x.amt) AS amount
FROM (
    SELECT substr(p.trans_dt, 1, 7) AS txn_month,
           CASE WHEN {tp} AND {tr} AND p.{a} = p.{b} THEN 'SELF' WHEN {tp} AND {tr} THEN 'WITHIN'
                WHEN {tp} THEN 'OUT' ELSE 'IN' END AS dir,
           CASE WHEN {tp} AND {tr} THEN 'target' WHEN {tp} THEN {other(PNC_RECS)} ELSE {other(PNC_PAYS)} END AS other_kind,
           {RAIL_EXPR} AS pay_rail, CAST(p.trans_amt AS DOUBLE) AS amt, {BUCKET_EXPR} AS amt_bucket
    {frm}
    WHERE {' AND '.join(where)}
) x
GROUP BY 1, 2, 3, 4, 5"""


def sql_ties_ids(P, int_ids, ext_ids):
    """Payments between connected parties, every target excluded. Each observable tie has a PNC end
    (counterparty->counterparty is not in the table), so the scan is pre-filtered on the PNC
    parties; the outer filter keeps rows whose other end is also a connected party."""
    frm, where = _scope(P)
    ip, ir = (("pnc_dep_acct_pays", "pnc_dep_acct_receives") if P["level"] == "account"
              else ("mdm_id_pays", "mdm_id_receives"))
    a, b = _tcols(P)
    I, T = _in(int_ids), _in(P["ids"])
    where += [f"(p.{ip} IN ({I}) OR p.{ir} IN ({I}))",
              f"COALESCE(p.{a}, '') NOT IN ({T})", f"COALESCE(p.{b}, '') NOT IN ({T})"]
    keep = [f"(b.{ip} IN ({I}) AND b.{ir} IN ({I}))"]
    if ext_ids:
        E = _in(ext_ids)
        keep += [f"(b.{ip} IN ({I}) AND b.mdm_id_receives IS NULL AND b.pnc_dep_acct_receives IS NULL AND b.cpty_id IN ({E}))",
                 f"(b.{ir} IN ({I}) AND b.mdm_id_pays IS NULL AND b.pnc_dep_acct_pays IS NULL AND b.cpty_id IN ({E}))"]
    return (_CUBE_SELECT + f"\nFROM ({_INNER}\n    {frm}\n    WHERE " + "\n      AND ".join(where) + "\n) b"
            + "\nWHERE " + "\n   OR ".join(keep) + "\nGROUP BY 1, 2, 3, 4, 5, 6, 7")


def sql_near_match(P):
    """Zero rows is usually formatting (leading zeros, wrong id type), not an unused id."""
    t = _q(P["ids"][0].strip().lstrip("0"))
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


# ---- NAICS -------------------------------------------------------------------------------------
def sql_naics_list():
    return f"""
SELECT naics_cd, MAX(naics_desc) AS naics_desc, COUNT(*) AS n_customers
FROM {CUST_DIM_TABLE}
WHERE naics_status = 'valid' AND naics_cd IS NOT NULL
GROUP BY 1 ORDER BY 1"""


def _naics_rows(P):
    """One row per transaction touching a member of the target NAICS, with the other side's group:
    its NAICS code, 'UNKNOWN' (PNC customer without a valid NAICS) or 'EXTERNAL'. Target members are
    a small broadcast set; the other side is joined to the valid-NAICS slice of the customer dim."""
    frm, where = _scope(P)
    code = _q(P["naics"])
    tgt = f"(SELECT mdm_id FROM {CUST_DIM_TABLE} WHERE naics_cd = {code} AND naics_status = 'valid')"
    dim = f"(SELECT mdm_id, naics_cd, naics_desc FROM {CUST_DIM_TABLE} WHERE naics_status = 'valid')"
    where.append("(tp.mdm_id IS NOT NULL OR tr.mdm_id IS NOT NULL)")
    return f"""(
    SELECT b.trans_dt, b.amt, b.pay_rail, b.dir, b.target_mdm, b.other_mdm,
           CASE WHEN b.dir IN ('WITHIN', 'SELF') THEN 'target' WHEN NOT b.other_pnc THEN 'cpty'
                WHEN od.mdm_id IS NOT NULL THEN 'naics' ELSE 'unknown' END AS other_kind,
           CASE WHEN b.dir IN ('WITHIN', 'SELF') THEN 'TARGET' WHEN NOT b.other_pnc THEN 'EXTERNAL'
                WHEN od.mdm_id IS NOT NULL THEN od.naics_cd ELSE 'UNKNOWN' END AS other_group,
           od.naics_desc AS other_desc,
           CASE WHEN b.other_pnc THEN b.other_mdm ELSE b.cpty_id END AS other_party
    FROM (
        SELECT p.trans_dt, CAST(p.trans_amt AS DOUBLE) AS amt, {RAIL_EXPR} AS pay_rail,
               CASE WHEN tp.mdm_id IS NOT NULL AND tr.mdm_id IS NOT NULL
                    THEN (CASE WHEN p.mdm_id_pays = p.mdm_id_receives THEN 'SELF' ELSE 'WITHIN' END)
                    WHEN tp.mdm_id IS NOT NULL THEN 'OUT' ELSE 'IN' END AS dir,
               CASE WHEN tp.mdm_id IS NOT NULL THEN p.mdm_id_pays ELSE p.mdm_id_receives END AS target_mdm,
               CASE WHEN tp.mdm_id IS NOT NULL THEN NULLIF(p.mdm_id_receives, '') ELSE NULLIF(p.mdm_id_pays, '') END AS other_mdm,
               CASE WHEN tp.mdm_id IS NOT NULL THEN {PNC_RECS} ELSE {PNC_PAYS} END AS other_pnc,
               {CPTY_ID_EXPR} AS cpty_id
        {frm}
        LEFT JOIN {tgt} tp ON p.mdm_id_pays = tp.mdm_id
        LEFT JOIN {tgt} tr ON p.mdm_id_receives = tr.mdm_id
        WHERE {' AND '.join(where)}
    ) b
    LEFT JOIN {dim} od ON b.other_mdm = od.mdm_id
) r"""


def _rail_sums(prefix="r."):
    """Per-rail amount and count columns, so a group's rail mix comes back on the same row as its
    exact distinct-party counts (NDV is not additive across rails)."""
    cols = []
    for i, k in enumerate(RAIL_DEF):
        cols += [f"SUM(CASE WHEN {prefix}pay_rail = '{k}' THEN {prefix}amt ELSE 0 END) AS amt_{i}",
                 f"SUM(CASE WHEN {prefix}pay_rail = '{k}' THEN 1 ELSE 0 END) AS n_{i}"]
    known = _in(RAIL_DEF)
    cols += [f"SUM(CASE WHEN {prefix}pay_rail NOT IN ({known}) THEN {prefix}amt ELSE 0 END) AS amt_x",
             f"SUM(CASE WHEN {prefix}pay_rail NOT IN ({known}) THEN 1 ELSE 0 END) AS n_x"]
    return ",\n       ".join(cols)


def sql_naics_groups(P):
    return f"""
SELECT r.dir, r.other_group, MAX(r.other_desc) AS other_desc,
       SUM(r.amt) AS amount, COUNT(*) AS n_txn, MIN(r.trans_dt) AS first_dt, MAX(r.trans_dt) AS last_dt,
       NDV(r.target_mdm) AS n_targets, NDV(r.other_party) AS n_parties,
       {_rail_sums()}
FROM {_naics_rows(P)}
WHERE r.dir <> 'SELF'
GROUP BY 1, 2"""


def sql_naics_flow(P):
    return f"""
SELECT substr(r.trans_dt, 1, 7) AS txn_month, r.dir, r.other_kind, r.pay_rail,
       CAST(FLOOR(LOG10(GREATEST(r.amt, 0.01)) * 20) AS INT) AS amt_bucket,
       COUNT(*) AS n_txn, SUM(r.amt) AS amount
FROM {_naics_rows(P)}
GROUP BY 1, 2, 3, 4, 5"""


def sql_naics_members(P):
    names = (f"(SELECT mdm_id, customer_name, party_type FROM {CUST_DIM_TABLE} "
             f"WHERE naics_cd = {_q(P['naics'])} AND naics_status = 'valid')")
    return f"""
SELECT r.target_mdm, r.dir, r.other_kind, MAX(n.customer_name) AS customer_name, MAX(n.party_type) AS party_type,
       SUM(r.amt) AS amount, COUNT(*) AS n_txn, NDV(r.other_party) AS n_parties
FROM {_naics_rows(P)}
LEFT JOIN {names} n ON r.target_mdm = n.mdm_id
WHERE r.dir <> 'SELF'
GROUP BY 1, 2, 3"""


def sql_naics_ties(P, codes):
    """Payments between the top other NAICS (and between them and counterparties). Members of the
    target NAICS are not in the set, so their own payments drop out automatically."""
    frm, where = _scope(P)
    alt_ = (f"(SELECT mdm_id, naics_cd FROM {CUST_DIM_TABLE} "
            f"WHERE naics_status = 'valid' AND naics_cd IN ({_in(codes)}))")
    where.append("(ap.mdm_id IS NOT NULL OR ar.mdm_id IS NOT NULL)")
    return f"""
SELECT b.pay_rail, b.src_group, b.dst_group, SUM(b.amt) AS amount, COUNT(*) AS n_txn,
       MIN(b.trans_dt) AS first_dt, MAX(b.trans_dt) AS last_dt
FROM (
    SELECT p.trans_dt, CAST(p.trans_amt AS DOUBLE) AS amt, {RAIL_EXPR} AS pay_rail,
           COALESCE(ap.naics_cd, CASE WHEN NOT {PNC_PAYS} THEN 'EXTERNAL' END) AS src_group,
           COALESCE(ar.naics_cd, CASE WHEN NOT {PNC_RECS} THEN 'EXTERNAL' END) AS dst_group
    {frm}
    LEFT JOIN {alt_} ap ON p.mdm_id_pays = ap.mdm_id
    LEFT JOIN {alt_} ar ON p.mdm_id_receives = ar.mdm_id
    WHERE {' AND '.join(where)}
) b
WHERE b.src_group IS NOT NULL AND b.dst_group IS NOT NULL AND b.src_group <> b.dst_group
GROUP BY 1, 2, 3"""


@st.cache_data(ttl=CACHE_TTL_S, max_entries=64, show_spinner=False)
def run_sql(sql):
    t0 = time.time()
    df = get_query(sql)
    df.columns = [str(c).lower().split(".")[-1] for c in df.columns]
    return df, time.time() - t0


# =============================================================================
# TRANSFORM — every mode ends in the same shapes:
#   pairs   target x party x direction x rail   (IN/OUT only)      A   party attributes
#   within  target -> target x rail                                 TA  target attributes
#   flow    month x direction x other kind x rail x amount bucket   ties party -> party x rail
# =============================================================================
def _num(df, cols=("amount", "n_txn")):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _clean_ids(df):
    for c in ID_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object").map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                                               or str(v).strip() == "" else str(v))
    if "pay_rail" in df.columns:
        df["rail"] = df["pay_rail"].map(rail_label)
    return _num(df)


def _prefixed(prefix, s):
    return s.map(lambda v: None if v is None else prefix + v)


def node_keys(is_pnc, mdm, acct, cpty, level):
    """Key of one end. PNC end: 'PNC:<acct>' (account level) or 'MDM:<mdm>' (customer level), each
    falling back to the other id when missing. Counterparty end: 'EXT:<cpty_id>' or None."""
    if level == "account":
        k_int = _prefixed("PNC:", acct).where(acct.notna(), _prefixed("MDM:", mdm))
    else:
        k_int = _prefixed("MDM:", mdm).where(mdm.notna(), _prefixed("PNC:", acct))
    return k_int.where(is_pnc, _prefixed("EXT:", cpty))


def _mode_by(df, key, col, weight="n_txn"):
    d = df[df[col].notna()].groupby([key, col], as_index=False)[weight].sum()
    if d.empty:
        return pd.Series(dtype=object)
    return d.sort_values([key, weight], ascending=[True, False]).drop_duplicates(key).set_index(key)[col]


def _clean_flow(flow):
    f = _num(flow.copy())
    f["amt_bucket"] = pd.to_numeric(f["amt_bucket"]).astype(int)
    f["rail"] = f["pay_rail"].map(rail_label)
    return f


def _agg_pairs(df, keys):
    return df.groupby(keys, as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                                               first_dt=("first_dt", "min"), last_dt=("last_dt", "max"))


def build_ids(P, cube, flow):
    """Customer and list modes: orient the pair cube around the target set."""
    c = _clean_ids(cube.copy())
    a, b = _tcols(P)
    T = set(P["ids"])
    tp, tr = c[a].isin(T), c[b].isin(T)
    same = c[a].notna() & (c[a] == c[b])
    c["direction"] = np.select([tp & tr & same, tp & tr, tp, tr], ["SELF", "WITHIN", "OUT", "IN"], "OTHER")
    pref = "PNC:" if P["kind"] == "acct" else "MDM:"
    c["t_src"] = _prefixed(pref, c[a].where(tp, None))
    c["t_dst"] = _prefixed(pref, c[b].where(tr, None))
    other_is_rec = c["direction"].ne("IN")

    def other(pay_col, rec_col):
        return c[rec_col].where(other_is_rec, c[pay_col])

    c["target"] = c["t_src"].where(other_is_rec, c["t_dst"])
    c["t_acct"] = c["pnc_dep_acct_pays"].where(other_is_rec, c["pnc_dep_acct_receives"])
    c["cp_mdm"], c["cp_pnc_acct"] = other("mdm_id_pays", "mdm_id_receives"), other("pnc_dep_acct_pays", "pnc_dep_acct_receives")
    is_pnc = c["cp_mdm"].notna() | c["cp_pnc_acct"].notna()
    c["kind"] = np.where(is_pnc, "pnc", "cpty")
    c["party"] = node_keys(is_pnc, c["cp_mdm"], c["cp_pnc_acct"], c["cpty_id"], P["level"])
    c["cp_name"] = other("name_pays", "name_receives").where(is_pnc, c["cpty_name"])
    c["cp_bank"] = c["cpty_bank"].where(~is_pnc, "PNC")
    c["cp_id"] = (c["cp_pnc_acct"] if P["level"] == "account" else c["cp_mdm"]).where(is_pnc, c["cpty_id"])

    R = {"P": P}
    io_ = c[c["direction"].isin(["IN", "OUT"]) & c["party"].notna()]
    R["pairs"] = _agg_pairs(io_, ["target", "party", "direction", "kind", "rail"])

    # ---- targets ---------------------------------------------------------------------------------
    tside = pd.concat([
        c.loc[tp, ["t_src", "name_pays", "mdm_id_pays", "n_txn"]].set_axis(["key", "name", "mdm", "n_txn"], axis=1),
        c.loc[tr, ["t_dst", "name_receives", "mdm_id_receives", "n_txn"]].set_axis(["key", "name", "mdm", "n_txn"], axis=1)])
    TA = pd.DataFrame(index=pd.Index(sorted(tside["key"].unique()), name="key"))
    TA["name"] = _mode_by(tside, "key", "name").map(_safe)
    TA["mdm"] = _mode_by(tside, "key", "mdm")
    TA["id"] = TA.index.str[4:]
    TA["label"] = TA["name"].where(TA["name"].fillna("").str.len() > 0, TA["id"])
    TA["lookup"] = [("single", P["kind"], i) for i in TA["id"]]
    R["TA"] = TA
    R["not_found"] = [i for i in P["ids"] if pref + i not in TA.index]
    R["target_mdms"] = set(TA["mdm"].dropna())

    # ---- parties ---------------------------------------------------------------------------------
    A = io_.groupby("party").agg(cp_type=("kind", "first"), cp_mdm=("cp_mdm", "first"), cp_id=("cp_id", "first"),
                                 n_pnc_accts=("cp_pnc_acct", "nunique"))
    A["kind"] = A["cp_type"]
    A["cp_type"] = np.where(A["kind"] == "pnc", "internal", "external")
    A["cp_name"] = _mode_by(io_, "party", "cp_name").map(_safe)
    A["bank"] = _mode_by(io_, "party", "cp_bank").map(_safe)
    A["cpty_type"] = _mode_by(io_, "party", "cpty_type")
    A["same_customer"] = (A["kind"] == "pnc") & A["cp_mdm"].isin(R["target_mdms"])
    A["label"] = A["cp_name"].where(A["cp_name"].fillna("").str.len() > 0,
                                    "(" + A["cpty_type"].fillna("no name").astype(str) + " "
                                    + A["cp_id"].astype(str).str[-6:] + ")")
    A["lookup"] = [(("single", "acct" if k.startswith("PNC:") else "mdm", k[4:]) if t == "pnc" else None)
                   for k, t in zip(A.index, A["kind"])]
    R["A"] = A

    w = c[c["direction"] == "WITHIN"].rename(columns={"t_src": "src", "t_dst": "dst"})
    R["within"] = _agg_pairs(w, ["src", "dst", "rail"])
    R["own_accts"] = (io_.groupby(["t_acct", "direction"], as_index=False)
                         .agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"), parties=("party", "nunique")))
    R["flow"] = _clean_flow(flow)
    R["members"] = None
    R["ties"] = _empty_ties()
    return R


def _melt_rails(g, keys):
    """Per-rail columns (amt_i, n_i) -> long rows (keys..., rail, amount, n_txn)."""
    parts = []
    for i, k in list(enumerate(RAIL_DEF)) + [("x", "UNKNOWN")]:
        d = g[keys + ["first_dt", "last_dt"]].copy()
        d["rail"] = rail_label(k)
        d["amount"] = pd.to_numeric(g[f"amt_{i}"], errors="coerce").fillna(0)
        d["n_txn"] = pd.to_numeric(g[f"n_{i}"], errors="coerce").fillna(0)
        parts.append(d[d["n_txn"] > 0])
    return pd.concat(parts, ignore_index=True)


def _group_key(g):
    return ("NAICS:" + g) if g not in ("UNKNOWN", "EXTERNAL", "TARGET") else "GRP:" + g


def build_naics(P, groups, flow, members, naics_info):
    R = {"P": P}
    tkey = "NAICS:" + P["naics"]
    g = _num(groups.copy(), ("amount", "n_txn", "n_targets", "n_parties"))
    long = _melt_rails(g, ["dir", "other_group"])
    io_ = long[long["dir"].isin(["IN", "OUT"])].copy()
    io_["target"] = tkey
    io_["party"] = io_["other_group"].map(_group_key)
    io_["kind"] = io_["other_group"].map(lambda s: {"UNKNOWN": "unknown", "EXTERNAL": "cpty"}.get(s, "naics"))
    io_ = io_.rename(columns={"dir": "direction"})
    R["pairs"] = _agg_pairs(io_, ["target", "party", "direction", "kind", "rail"])
    w = long[long["dir"] == "WITHIN"].assign(src=tkey, dst=tkey)
    R["within"] = _agg_pairs(w, ["src", "dst", "rail"])

    info = naics_info.get(P["naics"], {})
    desc = _safe(info.get("desc") or g.loc[g["other_group"] == "TARGET", "other_desc"].dropna().head(1).squeeze()
                 if (g["other_group"] == "TARGET").any() else info.get("desc", ""))
    R["TA"] = pd.DataFrame({"name": [desc], "mdm": [None], "id": [P["naics"]],
                            "label": [f"{P['naics']} · {desc}" if desc else P["naics"]], "lookup": [None]},
                           index=pd.Index([tkey], name="key"))
    R["not_found"], R["target_mdms"] = [], set()

    gi = g[g["dir"].isin(["IN", "OUT"])].copy()
    gi["party"] = gi["other_group"].map(_group_key)
    per = gi.pivot_table(index="party", columns="dir", values=["n_parties", "n_targets"], aggfunc="sum", fill_value=0)
    A = pd.DataFrame(index=pd.Index(sorted(gi["party"].unique()), name="party"))
    A["kind"] = [("unknown" if k == "GRP:UNKNOWN" else "cpty" if k == "GRP:EXTERNAL" else "naics") for k in A.index]
    A["cp_type"] = np.where(A["kind"] == "cpty", "external", "internal")
    A["cp_id"] = [k[6:] if k.startswith("NAICS:") else "" for k in A.index]
    A["desc"] = gi.groupby("party")["other_desc"].agg(lambda s: next((v for v in s if isinstance(v, str) and v), "")).map(_safe)
    for col, d in [("n_parties", "IN"), ("n_parties", "OUT"), ("n_targets", "IN"), ("n_targets", "OUT")]:
        A[f"{col}_{d.lower()}"] = per[(col, d)].reindex(A.index).fillna(0).astype(int) if (col, d) in per.columns else 0
    A["label"] = [("External counterparties" if k == "GRP:EXTERNAL" else "PNC customers, NAICS unknown"
                   if k == "GRP:UNKNOWN" else f"{cid} · {ds}" if ds else cid)
                  for k, cid, ds in zip(A.index, A["cp_id"], A["desc"])]
    A["cp_name"], A["cp_mdm"], A["bank"], A["cpty_type"], A["same_customer"] = A["label"], None, None, None, False
    A["lookup"] = [("naics", None, k[6:]) if k.startswith("NAICS:") else None for k in A.index]
    R["A"] = A

    m = _num(members.copy(), ("amount", "n_txn", "n_parties"))
    m["customer_name"] = m["customer_name"].map(_safe)
    R["members"] = m
    R["n_members"] = int(info.get("n", 0) or 0)
    R["own_accts"] = None
    R["flow"] = _clean_flow(flow)
    R["ties"] = _empty_ties()
    return R


def _empty_ties():
    return pd.DataFrame(columns=["src", "dst", "kind", "rail", "amount", "n_txn", "first_dt", "last_dt"])


def ties_ids(R, ties_raw):
    P = R["P"]
    if ties_raw is None or not len(ties_raw):
        return _empty_ties()
    t = _clean_ids(ties_raw.copy())
    s_pnc = t["mdm_id_pays"].notna() | t["pnc_dep_acct_pays"].notna()
    d_pnc = t["mdm_id_receives"].notna() | t["pnc_dep_acct_receives"].notna()
    t["src"] = node_keys(s_pnc, t["mdm_id_pays"], t["pnc_dep_acct_pays"], t["cpty_id"], P["level"])
    t["dst"] = node_keys(d_pnc, t["mdm_id_receives"], t["pnc_dep_acct_receives"], t["cpty_id"], P["level"])
    t["kind"] = np.where(s_pnc & d_pnc, "pnc_pnc", "pnc_cpty")
    parties = set(R["A"].index)
    t = t[t["src"].isin(parties) & t["dst"].isin(parties) & (t["src"] != t["dst"])]
    return _agg_pairs(t, ["src", "dst", "kind", "rail"]) if len(t) else _empty_ties()


def ties_naics(R, ties_raw):
    if ties_raw is None or not len(ties_raw):
        return _empty_ties()
    t = _clean_ids(ties_raw.copy())
    t["src"], t["dst"] = t["src_group"].map(_group_key), t["dst_group"].map(_group_key)
    t["kind"] = np.where((t["src_group"] == "EXTERNAL") | (t["dst_group"] == "EXTERNAL"), "pnc_cpty", "pnc_pnc")
    parties = set(R["A"].index)
    t = t[t["src"].isin(parties) & t["dst"].isin(parties)]
    return _agg_pairs(t, ["src", "dst", "kind", "rail"]) if len(t) else _empty_ties()


# ---- the view: what the Show options keep ---------------------------------------------------------
def view(R, o):
    """Apply the Show options. Normal view: flows crossing the target boundary with the allowed kinds
    of party (+ within-target links in the picture). 'Only between targets': within-target flows only.
    The pair totals must equal the flow totals; a mismatch stops the page."""
    P = R["P"]
    if o["only_between"]:
        kinds = set()
        pairs = R["pairs"].iloc[0:0]
        flow = R["flow"][R["flow"]["dir"] == "WITHIN"]
        ref = float(R["within"]["amount"].sum())
    else:
        kinds = ({"naics"} | ({"unknown"} if o["include_unknown"] else set())) if P["mode"] == "naics" else {"pnc"}
        if o["include_cp"]:
            kinds.add("cpty")
        pairs = R["pairs"][R["pairs"]["kind"].isin(kinds)]
        flow = R["flow"][R["flow"]["dir"].isin(["IN", "OUT"]) & R["flow"]["other_kind"].isin(kinds)]
        ref = float(pairs["amount"].sum())
    got = float(flow["amount"].sum())
    if abs(got - ref) > 1e-6 * max(abs(ref), 1.0):
        raise AssertionError(f"Totals check failed: flow {got:,.2f} vs parties {ref:,.2f}")
    pf = pairs.pivot_table(index="party", columns="direction", values="amount", aggfunc="sum",
                           fill_value=0.0).reindex(columns=["IN", "OUT"], fill_value=0.0)
    pf["total"] = pf["IN"] + pf["OUT"]
    pf = pf.sort_values("total", ascending=False)
    t = R["ties"]
    tk = ([] if o["only_between"] else (["pnc_pnc"] if o["show_pp"] else [])
          + (["pnc_cpty"] if o["show_pc"] and o["include_cp"] else []))
    keep = set(pf.index)
    ties = t[t["kind"].isin(tk) & t["src"].isin(keep) & t["dst"].isin(keep)]
    return {"pairs": pairs, "flow": flow, "party_flow": pf, "ties": ties, "kinds": kinds}


# ---- tables ---------------------------------------------------------------------------------------
def party_table(R, V, direction):
    p = V["pairs"][V["pairs"]["direction"] == direction]
    if p.empty:
        return pd.DataFrame()
    d = p.groupby("party", as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                                               first_dt=("first_dt", "min"), last_dt=("last_dt", "max"),
                                               targets=("target", "nunique"))
    d = d.merge(R["A"], left_on="party", right_index=True, how="left")
    cr = p.groupby(["party", "rail"], as_index=False)["amount"].sum()
    cr["pct"] = cr["amount"] / cr.groupby("party")["amount"].transform("sum")
    cr = cr.sort_values(["party", "amount"], ascending=[True, False])
    cr["s"] = cr["rail"] + " " + (cr["pct"] * 100).round().fillna(0).astype(int).astype(str) + "%"
    mix = cr.groupby("party").head(3).groupby("party")["s"].agg(", ".join)
    other = set(V["pairs"].loc[V["pairs"]["direction"] != direction, "party"])
    d = d.sort_values("amount", ascending=False).reset_index(drop=True)
    d["share"] = d["amount"] / d["amount"].sum()
    d["cum_share"] = d["share"].cumsum()
    d["avg_txn"] = d["amount"] / d["n_txn"]
    d["type"] = d["kind"].map(KIND_LABEL)
    d["rail_mix"] = d["party"].map(mix)
    d["two_way"] = d["party"].isin(other)
    d.index = np.arange(1, len(d) + 1)
    d.index.name = "rank"
    return d.rename(columns={"label": "name", "cp_id": "id", "cp_mdm": "mdm_id"})


def concentration(t):
    if t is None or t.empty:
        return {}
    s = t["share"].values
    hhi = float((s ** 2).sum())
    return {"parties": len(s), "top 1": s[:1].sum(), "top 5": s[:5].sum(), "top 10": s[:10].sum(),
            "HHI": hhi, "effective n": 1 / hhi if hhi else np.nan}


def hist_percentiles(h, by, qs=(0.10, 0.50, 0.90, 0.99)):
    """Percentiles from the log-bucket histogram (geometric bucket midpoint, ±6%)."""
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


# =============================================================================
# NETWORK — data for the component (and for the standalone export)
# =============================================================================
def _edge_title(sn, dn, g):
    lines = [f"{sn}  →  {dn}", f"{money(g['amount'].sum())} in {int(g['n_txn'].sum()):,} txns"]
    by = g.groupby("rail")[["amount", "n_txn"]].sum().sort_values("amount", ascending=False)
    lines += [f"   {r}: {money(v['amount'])} ({int(v['n_txn']):,})" for r, v in by.iterrows()]
    lines.append(f"{g['first_dt'].min()} → {g['last_dt'].max()}")
    return "\n".join(lines)


def _lk(lookup, label):
    if not lookup:
        return {}
    mode, kind, value = lookup
    what = "NAICS" if mode == "naics" else ("Account" if kind == "acct" else "MDM id")
    return {"lk": {"mode": mode, "kind": kind, "id": value}, "lk_text": f"{what} {value} ({label})"}


def build_graph(R, V, max_nodes, o):
    """Three node types (target, PNC customer, counterparty) and one edge type: all rails between two
    nodes summed into one directed edge whose width is proportional to its total dollars."""
    P, A, TA = R["P"], R["A"], R["TA"]
    font = lambda size, bold=False: {"size": size, "face": "arial", "color": "#222222", "bold": bold}
    mode = P["mode"]

    # ---- targets ---------------------------------------------------------------------------------
    gross = V["pairs"].groupby("target")["amount"].sum()
    w = R["within"]
    gross = gross.add(w.groupby("src")["amount"].sum(), fill_value=0).add(w.groupby("dst")["amount"].sum(), fill_value=0)
    if o["only_between"]:
        gross = gross[gross.index.isin(set(w["src"]) | set(w["dst"]))]
    tkeys = list(gross.sort_values(ascending=False).index[:MAX_TARGETS_DRAWN]) if mode == "list" else list(TA.index)
    gmax = max(float(gross.max()) if len(gross) else 1.0, 1.0)
    nodes = []
    for k in tkeys:
        t = TA.loc[k]
        fixed = mode != "list"
        tip = [f"TARGET {'NAICS' if mode == 'naics' else 'account' if P['kind'] == 'acct' else 'customer (MDM)'} {t['id']}",
               t["label"] if t["label"] != t["id"] else ""]
        if mode == "naics":
            tip.append(f"{R['n_members']:,} customers with this NAICS")
        elif P["kind"] == "acct" and isinstance(t["mdm"], str):
            tip.append(f"MDM: {t['mdm']}")
        tp = V["pairs"][V["pairs"]["target"] == k]
        tip += [f"Received {money(tp.loc[tp['direction'] == 'IN', 'amount'].sum())}",
                f"Sent {money(tp.loc[tp['direction'] == 'OUT', 'amount'].sum())}"]
        if mode == "list":
            tip.append("Click to load into the customer lookup")
        size = 40 if mode != "list" else float(20 + 20 * np.sqrt(gross.get(k, 0) / gmax))
        n = dict(id=k, label=(f"{trunc(t['label'], 32)}\n{t['id']}" if t["label"] != t["id"] else t["id"])
                 if mode != "naics" else trunc(t["label"], 40),
                 title="\n".join(x for x in tip if x), shape="star", size=size,
                 color={"background": C_EGO, "border": "#8A3F06", "highlight": {"background": C_EGO, "border": "#000000"}},
                 font=font(16 if mode != "list" else 12, True), search=f"★ {t['label']} · {t['id']} (target)")
        if fixed:
            n.update(x=0, y=0, fixed=True)
        if mode == "list":
            n.update(_lk(t["lookup"], t["label"]))
        nodes.append(n)
    tset = set(tkeys)

    # ---- parties -----------------------------------------------------------------------------------
    vis = V["party_flow"].head(max_nodes) if not o["only_between"] else V["party_flow"].iloc[0:0]
    keep = set(vis.index)
    fmax = max(float(vis["total"].max()) if len(vis) else 1.0, 1.0)
    for k, row in vis.iterrows():
        a = A.loc[k]
        pnc = a["cp_type"] == "internal"
        t = [a["label"], KIND_LABEL[a["kind"]]]
        if a["kind"] == "pnc":
            t += ([f"Account: {k[4:]}", f"MDM: {a['cp_mdm']}"] if k.startswith("PNC:")
                  else [f"MDM: {k[4:]}  ({int(a['n_pnc_accts'])} account(s) seen)"])
        elif a["kind"] == "cpty" and mode != "naics":
            t += [f"Counterparty id: {a['cp_id']}",
                  f"Bank: {a['bank'] if isinstance(a['bank'], str) and a['bank'] else '(not recorded)'}",
                  f"Type: {a['cpty_type'] if isinstance(a['cpty_type'], str) else '—'}"]
        if mode == "naics":
            t.append(f"Parties paying the target: {int(a['n_parties_in']):,} · paid by the target: {int(a['n_parties_out']):,}")
        rails = sorted(V["pairs"].loc[V["pairs"]["party"] == k, "rail"].unique())
        t.append(f"Rails: {', '.join(rails)}")
        if row["IN"] > 0:
            t.append(f"Paid the target{'s' if mode == 'list' else ''}: {money(row['IN'])}")
        if row["OUT"] > 0:
            t.append(f"Paid by the target{'s' if mode == 'list' else ''}: {money(row['OUT'])}")
        if a["lookup"]:
            t.append("Click to load into the " + ("NAICS" if mode == "naics" else "customer") + " lookup")
        bg, border = (C_INT, "#123A6E") if pnc else (C_EXT, "#1D6B48")
        n = dict(id=k, label=trunc(a["label"], 26 if mode != "naics" else 34), title="\n".join(t),
                 shape="dot" if pnc else "diamond", size=float(10 + 28 * np.sqrt(row["total"] / fmax)),
                 borderWidth=1.2, font=font(11),
                 color={"background": bg, "border": border, "highlight": {"background": bg, "border": "#000000"}},
                 search=f"{a['label']} · {a['cp_id']} · {KIND_LABEL[a['kind']]}")
        n.update(_lk(a["lookup"], a["label"]))
        nodes.append(n)

    # ---- edges -------------------------------------------------------------------------------------
    raw = []
    lab = lambda k: TA.loc[k, "label"] if k in TA.index else A.loc[k, "label"]
    pr = V["pairs"][V["pairs"]["party"].isin(keep) & V["pairs"]["target"].isin(tset)]
    for (tg, k, d), g in pr.groupby(["target", "party", "direction"]):
        src, dst = (k, tg) if d == "IN" else (tg, k)
        raw.append((src, dst, float(g["amount"].sum()), _edge_title(lab(src), lab(dst), g)))
    ww = w[w["src"].isin(tset) & w["dst"].isin(tset)]
    for (s_, d_), g in ww.groupby(["src", "dst"]):
        raw.append((s_, d_, float(g["amount"].sum()),
                    ("Within the NAICS\n" if s_ == d_ else "Between targets\n") + _edge_title(lab(s_), lab(d_), g)))
    tv = V["ties"][V["ties"]["src"].isin(keep) & V["ties"]["dst"].isin(keep)]
    for (s_, d_), g in tv.groupby(["src", "dst"]):
        raw.append((s_, d_, float(g["amount"].sum()), _edge_title(lab(s_), lab(d_), g)))
    amax = max([e[2] for e in raw] + [1.0])
    edges = [dict(**{"from": s_}, to=d_, title=title, width=round(1.0 + 11.0 * amt / amax, 2),
                  color={"color": C_EDGE, "highlight": "#222222", "hover": "#222222", "opacity": 0.85})
             for s_, d_, amt, title in raw]

    options = {
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.55}}, "arrowStrikethrough": False,
                  # curvedCW bends A->B and B->A to opposite sides, so two-way pairs stay readable
                  "smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.18},
                  "selfReference": {"size": 34, "angle": 0.8}},
        "physics": {"solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -90, "centralGravity": 0.012,
                                         "springLength": 170, "springConstant": 0.06, "avoidOverlap": 0.5},
                    "stabilization": {"enabled": True, "iterations": 300}},
        "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 60},
    }
    pf = V["party_flow"]
    if o["only_between"]:
        caption = f"Showing {len(tkeys):,} targets · {len(ww.groupby(['src', 'dst'])) if len(ww) else 0:,} payment links between them"
    else:
        shown = vis["total"].sum() / max(pf["total"].sum(), 1.0)
        caption = (f"Showing {len(vis):,} of {len(pf):,} connected parties ({shown:.0%} of the dollars)"
                   + (f" · {len(tkeys):,} of {len(gross):,} targets" if mode == "list" else "")
                   + f" · {len(tv.groupby(['src', 'dst'])) if len(tv) else 0:,} links between parties")
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
  if (args.hint) document.getElementById("hint").textContent = args.hint;
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
      if (n.lk) {
        send("streamlit:setComponentValue", {value: Object.assign({ts: Date.now()}, n.lk), dataType: "json"});
        document.getElementById("hint").textContent = n.lk_text + " loaded into the lookup. Press Build ego network.";
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
<div id="bar"><input id="find" list="names" placeholder="Find a node by name or id"><datalist id="names"></datalist>
<span id="hint"></span></div>
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
                          "sig": "export", "hint": "Click a node to isolate its neighbourhood."}).replace("</", "<\\/")
    return (_NET_HTML.replace("__VIS__", "<script>" + vis_js.replace("</script", "<\\/script") + "</script>")
                     .replace("__JS__", _NET_JS)
                     .replace("__DATA__", f"<script>render({payload});</script>"))


_ego_net = components.declare_component("pkg_ego_net", path=_component_dir())


def legend_html(mode):
    sw = {"star": f"<span style='color:{C_EGO};font-size:17px'>★</span>",
          "int": f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;background:{C_INT}'></span>",
          "ext": f"<span style='display:inline-block;width:10px;height:10px;background:{C_EXT};transform:rotate(45deg)'></span>"}
    it = lambda s, t: f"<span style='margin-right:18px;white-space:nowrap'>{s} {t}</span>"
    edge = f"<span style='display:inline-block;width:30px;border-top:4px solid {C_EDGE};vertical-align:middle'></span>"
    names = {"single": ("target", "PNC customer", "counterparty"),
             "list": ("target (from the list)", "PNC customer", "counterparty"),
             "naics": ("target NAICS", "other NAICS / PNC customers without NAICS", "all external counterparties")}[mode]
    return ("<div style='font-size:13px;line-height:1.9'>" + it(sw["star"], names[0]) + it(sw["int"], names[1])
            + it(sw["ext"], names[2])
            + it(edge, "payment link: all rails combined, width proportional to total $, arrow = payer → payee")
            + "</div>")


# =============================================================================
# CHARTS (Altair ships with Streamlit)
# =============================================================================
def _rails_order(flow):
    return flow.groupby("rail")["amount"].sum().sort_values(ascending=False).index.tolist()


def ch_monthly(flow, mode):
    m = flow.groupby(["dir", "txn_month", "rail"], as_index=False)[["amount", "n_txn"]].sum()
    m["signed"] = np.where(m["dir"] == "OUT", -m["amount"], m["amount"])
    m["flow"] = m["dir"].map(lambda d: dir_label(d, mode))
    rails = _rails_order(flow)
    within = (m["dir"] == "WITHIN").all()
    net = m.groupby("txn_month", as_index=False)["signed"].sum().rename(columns={"signed": "net"})
    bars = alt.Chart(m).mark_bar().encode(
        x=alt.X("txn_month:O", title=None),
        y=alt.Y("sum(signed):Q", title=dir_label("WITHIN", mode) if within else "received (+) / sent (−)", axis=MONEY_AXIS),
        color=alt.Color("rail:N", scale=rail_scale(rails), legend=alt.Legend(title="rail", orient="top")),
        tooltip=["txn_month", "flow", "rail", alt.Tooltip("amount:Q", format="$,.0f"), alt.Tooltip("n_txn:Q", format=",")])
    if within:
        return bars.properties(height=360)
    line = alt.Chart(net).mark_line(color="black", point=True).encode(
        x="txn_month:O", y="net:Q", tooltip=["txn_month", alt.Tooltip("net:Q", format="$,.0f", title="net in − out")])
    return alt.layer(bars, line).properties(height=360)


def ch_rail_mix(flow, mode):
    r = flow.groupby(["dir", "rail"], as_index=False)[["amount", "n_txn"]].sum()
    rows = []
    for d in [x for x in ["IN", "OUT", "WITHIN"] if x in set(r["dir"])]:
        for col, lab in [("amount", "$"), ("n_txn", "# txns")]:
            for _, x in r[r["dir"] == d].iterrows():
                rows.append({"row": f"{dir_label(d, mode)} · {lab}", "rail": x["rail"], "value": float(x[col])})
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["row"]))
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("row:N", sort=order, title=None),
        x=alt.X("value:Q", stack="normalize", axis=alt.Axis(format="%"), title="share"),
        color=alt.Color("rail:N", scale=rail_scale(_rails_order(flow)), legend=alt.Legend(title="rail", orient="top")),
        tooltip=["row", "rail", alt.Tooltip("value:Q", format=",.0f")]).properties(height=60 + 34 * len(order))


def ch_top(t, title, n=15):
    d = t.head(n).copy()
    d["who"] = [f"{trunc(nm, 34)} · …{str(i)[-4:]}" if i else trunc(nm, 40) for nm, i in zip(d["name"], d["id"])]
    kinds = [k for k in KIND_COLOR if k in set(d["type"])]
    return alt.Chart(d).mark_bar().encode(
        y=alt.Y("who:N", sort=list(d["who"]), title=None),
        x=alt.X("amount:Q", title=None, axis=MONEY_AXIS),
        color=alt.Color("type:N", scale=alt.Scale(domain=kinds, range=[KIND_COLOR[k] for k in kinds]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["name", "type", "id", alt.Tooltip("amount:Q", format="$,.0f"),
                 alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("n_txn:Q", format=","), "rail_mix"]
    ).properties(title=title, height=max(220, 22 * len(d)))


def ch_concentration(tin, tout):
    parts = [pd.DataFrame({"rank": np.arange(1, len(t) + 1), "cum_share": t["cum_share"].values, "flow": lab})
             for t, lab in [(tin, "Received"), (tout, "Sent")] if len(t)]
    return alt.Chart(pd.concat(parts)).mark_line().encode(
        x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="party rank (log)"),
        y=alt.Y("cum_share:Q", axis=alt.Axis(format="%"), title="cumulative share of dollars"),
        color=alt.Color("flow:N", scale=alt.Scale(range=["#1F5AA6", "#C0392B"])),
        tooltip=["flow", "rank", alt.Tooltip("cum_share:Q", format=".1%")]).properties(height=300)


def ch_hist(flow, d, mode):
    h = flow[flow["dir"] == d].groupby(["rail", "amt_bucket"], as_index=False)["n_txn"].sum()
    h["amount_mid"] = 10 ** ((h["amt_bucket"] + 0.5) / 20)
    return alt.Chart(h).mark_line(interpolate="step-after").encode(
        x=alt.X("amount_mid:Q", scale=alt.Scale(type="log"), axis=MONEY_AXIS, title="amount per transaction (log)"),
        y=alt.Y("n_txn:Q", title="transactions"),
        color=alt.Color("rail:N", scale=rail_scale(_rails_order(flow)), legend=alt.Legend(orient="top", title=None)),
        tooltip=["rail", alt.Tooltip("amount_mid:Q", format="$,.0f", title="≈ amount"), alt.Tooltip("n_txn:Q", format=",")]
    ).properties(title=f"Transaction size — {dir_label(d, mode).lower()}", height=280)


def ch_bars(df, cat, title, color, n=12):
    d = df.sort_values("amount", ascending=False).head(n).copy()
    d[cat] = d[cat].fillna("(not recorded)")
    return alt.Chart(d).mark_bar(color=color).encode(
        y=alt.Y(f"{cat}:N", sort=list(d[cat]), title=None), x=alt.X("amount:Q", axis=MONEY_AXIS, title=None),
        tooltip=[cat, alt.Tooltip("amount:Q", format="$,.0f"), alt.Tooltip("n_txn:Q", format=","),
                 alt.Tooltip("parties:Q", format=",")]).properties(title=title, height=max(180, 24 * len(d)))


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

# A click in the network arrives as a pending lookup. Apply it before the lookup widgets exist
# (a widget's value cannot be changed after it is drawn in the same run).
for k, v in [("search_mode", SEARCH_MODES[0]), ("lookup_mode", ID_KINDS[0]), ("lookup_id", ""),
             ("naics_code", ""), ("naics_txt", ""), ("list_kind", LIST_KINDS[0])]:
    st.session_state.setdefault(k, v)
_pending = st.session_state.pop("_pending_lookup", None)
if _pending:
    pmode, pkind, pid = _pending
    if pmode == "naics":
        st.session_state.update(search_mode="NAICS", naics_code=pid, naics_txt=pid)
        msg = f"NAICS {pid} loaded into the lookup. Press Build ego network."
    else:
        st.session_state.update(search_mode=SEARCH_MODES[0], lookup_mode=ID_KINDS[0] if pkind == "acct" else ID_KINDS[1],
                                lookup_id=pid)
        msg = f"{'Account' if pkind == 'acct' else 'MDM id'} {pid} loaded into the lookup. Press Build ego network."
    st.toast(msg) if hasattr(st, "toast") else st.sidebar.info(msg)

_today = date.today()
_end_default = _today.replace(day=1) - timedelta(days=1)                       # end of last month
_start_default = date(*(int(x) for x in _month_shift(_end_default, -(DEFAULT_WINDOW_MONTHS - 1)).split("-")), 1)

naics_info, naics_err = {}, None
with st.sidebar:
    st.header("Lookup")
    smode = st.radio("Search by", SEARCH_MODES, key="search_mode")
    mode = MODE_OF[smode]
    if mode == "naics":
        try:
            nl, _ = run_sql(sql_naics_list())
            naics_info = {str(r.naics_cd): {"desc": r.naics_desc, "n": int(r.n_customers)} for r in nl.itertuples()}
        except Exception as ex:          # keep the page usable with a plain text box
            naics_err = f"NAICS list unavailable ({type(ex).__name__}); type the code instead."
    with st.form("lookup"):
        upload = None
        if mode == "single":
            st.radio("Look up by", ID_KINDS, horizontal=True, key="lookup_mode")
            st.text_input("Id", placeholder="PNC deposit account or MDM id", key="lookup_id")
        elif mode == "naics":
            if naics_info:
                opts = [""] + sorted(naics_info)
                if st.session_state["naics_code"] not in opts:
                    st.session_state["naics_code"] = ""
                st.selectbox("NAICS (6 digits)", opts, key="naics_code",
                             format_func=lambda c: "Choose or type a NAICS code…" if not c else
                             f"{c} · {naics_info[c]['desc']} ({naics_info[c]['n']:,} customer{'s' if naics_info[c]['n'] != 1 else ''})")
            else:
                if naics_err:
                    st.caption(naics_err)
                st.text_input("NAICS (6 digits)", key="naics_txt", max_chars=6)
        else:
            st.radio("The file holds", LIST_KINDS, horizontal=True, key="list_kind")
            upload = st.file_uploader("Ids file", type=["csv", "txt", "tsv"],
                                      help="One id per line, or ids separated by commas, semicolons, tabs, pipes or "
                                           "spaces, or a table with a header (the id column is found by name).")
        window = st.date_input("Window", value=(_start_default, _end_default), min_value=DATA_START,
                               max_value=_today, help="Every query scans the window; start narrow.")
        level_lbl = "Same as lookup"
        if mode != "naics":
            level_lbl = st.radio("PNC customers drawn as", ["Same as lookup", "Accounts", "Customers (MDM)"],
                                 help="Accounts: one node per deposit account. Customers: accounts of the same MDM id merged.")
        rails = st.multiselect("Payment rails", list(RAIL_DEF), default=DEFAULT_RAILS, format_func=rail_label)
        submitted = st.form_submit_button("Build ego network", type="primary", **_wide(st.form_submit_button))

if submitted:
    errs, base = [], {}
    if not (isinstance(window, (tuple, list)) and len(window) == 2):
        errs.append("Pick both a start and an end date.")
    if not rails:
        errs.append("Select at least one payment rail.")
    if mode == "single":
        kind = "acct" if st.session_state["lookup_mode"] == ID_KINDS[0] else "mdm"
        ego_id = st.session_state["lookup_id"].strip()
        if not ID_RE.match(ego_id):
            errs.append("Id must be 1–64 letters, digits, '.', '_' or '-'.")
        base = dict(kind=kind, ids=(ego_id,), naics=None)
    elif mode == "naics":
        code = (st.session_state["naics_code"] if naics_info else st.session_state["naics_txt"]).strip()
        if not NAICS_RE.match(code):
            errs.append("Choose a 6-digit NAICS code.")
        base = dict(kind=None, ids=(), naics=code)
    else:
        kind = "acct" if st.session_state["list_kind"] == LIST_KINDS[0] else "mdm"
        if upload is None:
            errs.append("Upload a file of ids.")
        else:
            ids, rep = parse_id_file(upload.getvalue(), kind)
            rep["file"] = upload.name
            st.session_state["list_report"] = rep
            if rep.get("error") or not ids:
                errs.append(rep.get("error") or "No ids found in the file.")
            elif len(ids) > LIST_MAX_IDS:
                errs.append(f"The file holds {len(ids):,} ids; the limit is {LIST_MAX_IDS:,}.")
            base = dict(kind=kind, ids=tuple(ids), naics=None)
    if errs:
        for e in errs:
            st.sidebar.error(e)
    else:
        level = "customer" if mode == "naics" else {"Accounts": "account", "Customers (MDM)": "customer"}.get(
            level_lbl, "account" if base["kind"] == "acct" else "customer")
        st.session_state["P"] = dict(mode=mode, start=window[0].isoformat(), end=window[1].isoformat(),
                                     level=level, rails=tuple(r for r in RAIL_DEF if r in rails), **base)

P = st.session_state.get("P")
with st.sidebar:
    st.header("Show")
    pm = P["mode"] if P else mode
    only_between = False
    if pm == "list":
        only_between = st.checkbox("Only transactions between targets", value=False,
                                   help="Only payments where both ends are ids from the list.")
    include_cp = st.checkbox("Include external counterparties (one node)" if pm == "naics" else "Include counterparties",
                             value=True, disabled=only_between,
                             help="Off: PNC customers only, in the network and in every table.")
    include_unknown = True
    if pm == "naics":
        include_unknown = st.checkbox("Include PNC customers without a valid NAICS (one node)", value=True)
    show_pp = st.checkbox("Connections between other NAICS" if pm == "naics" else "Connections between PNC customers",
                          value=True, disabled=only_between)
    show_pc = st.checkbox("Connections between other NAICS and counterparties" if pm == "naics"
                          else "Connections between PNC customers and counterparties",
                          value=True, disabled=only_between or not include_cp)
    max_vis = st.slider("Parties drawn", 10, 600, MAX_VIS_DEFAULT, step=10)
    top_n = st.slider("Rows in ranked tables", 10, 500, 25, step=5)
O = dict(only_between=only_between, include_cp=include_cp and not only_between, include_unknown=include_unknown,
         show_pp=show_pp, show_pc=show_pc)

st.title("PKG · Ego network")
if P is None:
    st.info("Pick a customer, a NAICS code or a file of ids in the sidebar and press **Build ego network**.")
    st.stop()

# ---- queries (cached by SQL text, so the Show options never re-query) ------------------------------
try:
    if P["mode"] in ("single", "list"):
        what = P["ids"][0] if P["mode"] == "single" else f"{len(P['ids']):,} listed ids"
        with st.spinner(f"Q1/3 · transactions touching {what} ({P['start']} → {P['end']}) …"):
            cube, _ = run_sql(sql_pairs(P))
        if cube.empty:
            st.warning(f"No transactions for {what} in {P['start']} → {P['end']} on the selected rails.")
            if P["mode"] == "single":
                with st.spinner("Searching every id column for near matches (one more scan) …"):
                    near, _ = run_sql(sql_near_match(P))
                if len(near):
                    st.write("Near matches (whitespace / leading zeros ignored, all rails):")
                    show(near, int_cols=["n_txn"])
                else:
                    st.write("No near matches on any account, MDM or counterparty id column in this window.")
            else:
                st.write("Check the id type (deposit accounts vs MDM ids) and leading zeros in the file.")
            st.stop()
        with st.spinner("Q2/3 · monthly flow and transaction sizes …"):
            flow, _ = run_sql(sql_flow_ids(P))
        R = build_ids(P, cube, flow)
        # the ties query depends only on the lookup: the top PNC customers and the top counterparties are
        # searched separately, so toggling the Show options needs no new query
        A0 = R["A"]
        tot = R["pairs"].groupby("party")["amount"].sum().sort_values(ascending=False)
        top_int = [k for k in tot.index if A0.loc[k, "kind"] == "pnc"][:TIE_SEARCH_CAP]
        top_ext = [k for k in tot.index if A0.loc[k, "kind"] == "cpty"][:TIE_SEARCH_CAP]
        want = "PNC:" if P["level"] == "account" else "MDM:"
        int_ids = sorted({k[4:] for k in top_int if k.startswith(want)})
        ext_ids = sorted(A0.loc[top_ext, "cp_id"].dropna().unique().tolist())
        if int_ids:
            with st.spinner("Q3/3 · payments between the connected parties …"):
                ties_raw, _ = run_sql(sql_ties_ids(P, int_ids, ext_ids))
            R["ties"] = ties_ids(R, ties_raw)
    else:
        info = naics_info.get(P["naics"], {})
        with st.spinner(f"Q1/4 · flows of NAICS {P['naics']} by counterparty industry …"):
            groups, _ = run_sql(sql_naics_groups(P))
        if groups.empty:
            st.warning(f"No transactions for customers with NAICS {P['naics']} in {P['start']} → {P['end']} "
                       "on the selected rails.")
            st.stop()
        with st.spinner("Q2/4 · monthly flow and transaction sizes …"):
            flow, _ = run_sql(sql_naics_flow(P))
        with st.spinner("Q3/4 · member customers …"):
            members, _ = run_sql(sql_naics_members(P))
        R = build_naics(P, groups, flow, members, naics_info)
        tot = R["pairs"][R["pairs"]["kind"] == "naics"].groupby("party")["amount"].sum().sort_values(ascending=False)
        codes = [k[6:] for k in tot.index[:TIE_NAICS_CAP]]
        if codes:
            with st.spinner(f"Q4/4 · payments between the top {len(codes)} other NAICS …"):
                ties_raw, _ = run_sql(sql_naics_ties(P, codes))
            R["ties"] = ties_naics(R, ties_raw)
    V = view(R, O)
except Exception as ex:
    st.error(f"{type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ''}")
    with st.expander("Traceback"):
        st.code(traceback.format_exc())
    st.stop()

# ---- header ---------------------------------------------------------------------------------------
M = P["mode"]
TA = R["TA"]
pf, fv = V["party_flow"], V["flow"]
win_amt = float(R["within"]["amount"].sum())
if M == "single":
    t = TA.iloc[0]
    head = (f"**{'Account' if P['kind'] == 'acct' else 'Customer'} {t['id']}** · {t['label'] if t['label'] != t['id'] else '(no name)'}"
            + (f" · MDM {t['mdm']}" if P["kind"] == "acct" and isinstance(t["mdm"], str) else ""))
elif M == "list":
    rep = st.session_state.get("list_report", {})
    head = (f"**{len(P['ids']):,} {'accounts' if P['kind'] == 'acct' else 'MDM ids'}** from {rep.get('file', 'the file')}"
            f" · {len(TA):,} active in the window · {len(R['not_found']):,} not found")
else:
    head = (f"**NAICS {TA.iloc[0]['label']}** · {R['n_members']:,} customers with this NAICS · "
            f"{R['members']['target_mdm'].nunique():,} active in the window")
st.markdown(head + f" · {P['start']} → {P['end']} · {', '.join(rail_label(r) for r in P['rails'])}"
            + ("" if M == "naics" else f" · PNC customers drawn as {'accounts' if P['level'] == 'account' else 'customers'}")
            + (" · **only transactions between targets**" if O["only_between"] else
               "" if O["include_cp"] else " · **counterparties hidden**"))

if O["only_between"]:
    cards = [("Between targets", money(win_amt)), ("Transactions", f"{int(fv['n_txn'].sum()):,}"),
             ("Targets linked", f"{len(set(R['within']['src']) | set(R['within']['dst'])):,}"),
             ("Links", f"{len(R['within'].groupby(['src', 'dst'])) if len(R['within']) else 0:,}")]
else:
    rin = float(fv.loc[fv["dir"] == "IN", "amount"].sum())
    rout = float(fv.loc[fv["dir"] == "OUT", "amount"].sum())
    nk = lambda kinds: int(sum(1 for k in pf.index if R["A"].loc[k, "kind"] in kinds))
    cards = [("Received", money(rin)), ("Sent", money(rout)), ("Net (in − out)", money(rin - rout)),
             ("Transactions", f"{int(fv['n_txn'].sum()):,}")]
    cards += ([("Other NAICS", f"{nk({'naics'}):,}")] if M == "naics" else [("PNC customers", f"{nk({'pnc'}):,}")])
    cards += [("Counterparties", (f"{nk({'cpty'}):,}" if M != "naics" else "1 node") if O["include_cp"] else "hidden")]
    if M != "single":
        cards += [(dir_label("WITHIN", M), money(win_amt))]
for col, (lab_, val) in zip(st.columns(len(cards)), cards):
    col.metric(lab_, val)

T_in = party_table(R, V, "IN") if not O["only_between"] else pd.DataFrame()
T_out = party_table(R, V, "OUT") if not O["only_between"] else pd.DataFrame()
tab_names = ["Network"]
tab_names += ([] if O["only_between"] else ["Senders & receivers"])
tab_names += (["Targets"] if M == "list" else ["Members"] if M == "naics" else [])
tab_names += ["Rails", "Over time"]
tab_names += (["Counterparty banks"] if O["include_cp"] and M != "naics" else [])
tab_names += ([] if O["only_between"] else ["Connections"])
tab_names += (["Own accounts"] if M == "single" and P["kind"] == "mdm" else []) + ["Summary & QA"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

nodes, edges, options, caption = build_graph(R, V, max_vis, O)
with tabs["Network"]:
    st.markdown(legend_html(M), unsafe_allow_html=True)
    st.caption(caption + " · hover for details · click a node to isolate its neighbourhood")
    hint = {"single": "Click a PNC customer to load it into the lookup.",
            "list": "Click a target or a PNC customer to load it into the customer lookup.",
            "naics": "Click another NAICS to load it into the lookup."}[M]
    sig = hashlib.md5(json.dumps([nodes, edges], sort_keys=True, default=str).encode()).hexdigest()
    clicked = _ego_net(nodes=nodes, edges=edges, options=options, height=NET_HEIGHT_PX, sig=sig, hint=hint,
                       key="ego_net", default=None)
    if isinstance(clicked, dict) and clicked.get("id") and clicked.get("ts") != st.session_state.get("_click_ts"):
        st.session_state["_click_ts"] = clicked.get("ts")
        st.session_state["_pending_lookup"] = (clicked.get("mode"), clicked.get("kind"), str(clicked["id"]))
        _rerun()

# columns of the party tables, by mode
if M == "naics":
    CP_COLS = ["name", "type", "amount", "share", "cum_share", "n_txn", "avg_txn", "rail_mix",
               "n_parties_in", "n_parties_out", "n_targets_in", "n_targets_out", "first_dt", "last_dt", "two_way"]
else:
    CP_COLS = ["name", "type", "id", "mdm_id", "bank", "cpty_type", "amount", "share", "cum_share", "n_txn", "avg_txn",
               "rail_mix"] + (["targets"] if M == "list" else []) + ["first_dt", "last_dt", "two_way"] \
              + (["same_customer"] if P["kind"] == "acct" else [])
CP_FMT = dict(money_cols=["amount", "avg_txn"], pct_cols=["share", "cum_share"],
              int_cols=["n_txn", "targets", "n_parties_in", "n_parties_out", "n_targets_in", "n_targets_out"])


def _cp_view(t):
    if t.empty:
        return t
    cols = [c for c in CP_COLS if c in t.columns and not (c in ("bank", "cpty_type") and not O["include_cp"])]
    return t[cols].head(top_n)


tw = pd.DataFrame()
if "Senders & receivers" in tabs:
    with tabs["Senders & receivers"]:
        tgt_word = "the targets" if M == "list" else "the NAICS" if M == "naics" else "the target"
        c1, c2 = st.columns(2)
        if len(T_in):
            c1.altair_chart(ch_top(T_in, f"Top senders to {tgt_word}"), **_wide(st.altair_chart))
        if len(T_out):
            c2.altair_chart(ch_top(T_out, f"Top receivers from {tgt_word}"), **_wide(st.altair_chart))
        st.subheader(f"Senders — paid {tgt_word} ({len(T_in):,})")
        show(_cp_view(T_in), **CP_FMT)
        st.subheader(f"Receivers — paid by {tgt_word} ({len(T_out):,})")
        show(_cp_view(T_out), **CP_FMT)
        if M == "naics":
            st.caption("n_parties_*: distinct customers / counterparties in that group on each side; "
                       "n_targets_*: members of the target NAICS involved.")
        both = sorted(set(T_in["party"]) & set(T_out["party"])) if len(T_in) and len(T_out) else []
        st.subheader(f"Two-way — both sent to and received from {tgt_word} ({len(both):,})")
        if both:
            a, b = T_in.set_index("party").loc[both], T_out.set_index("party").loc[both]
            tw = pd.DataFrame({"name": a["name"], "type": a["type"], "id": a["id"], "received_from": a["amount"],
                               "sent_to": b["amount"]})
            tw["net_to_target"] = tw["received_from"] - tw["sent_to"]
            tw["gross"] = tw["received_from"] + tw["sent_to"]
            tw = tw.sort_values("gross", ascending=False).reset_index(drop=True)
            show(tw.head(top_n), money_cols=["received_from", "sent_to", "net_to_target", "gross"])
        else:
            st.caption("none")

within_t, targets_t = pd.DataFrame(), pd.DataFrame()
lab_of = lambda k: TA.loc[k, "label"] if k in TA.index else k
if R["within"] is not None and len(R["within"]):
    wg = R["within"].groupby(["src", "dst"], as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                                                                first_dt=("first_dt", "min"), last_dt=("last_dt", "max"))
    wr = R["within"].groupby(["src", "dst", "rail"], as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    wg["rails"] = wg.apply(lambda r: ", ".join(wr.loc[(wr["src"] == r["src"]) & (wr["dst"] == r["dst"]), "rail"]), axis=1)
    within_t = wg.assign(payer=wg["src"].map(lab_of), payer_id=wg["src"].str[4:], payee=wg["dst"].map(lab_of),
                         payee_id=wg["dst"].str[4:]).sort_values("amount", ascending=False)[
        ["payer", "payer_id", "payee", "payee_id", "amount", "n_txn", "rails", "first_dt", "last_dt"]].reset_index(drop=True)

if "Targets" in tabs:
    with tabs["Targets"]:
        rep = st.session_state.get("list_report", {})
        st.caption(f"File {rep.get('file', '')}: {rep.get('layout', '')}, ids taken from {rep.get('column', '')}; "
                   f"{rep.get('n_ids', 0):,} ids, {rep.get('n_duplicates', 0):,} duplicates dropped, "
                   f"{rep.get('n_invalid', 0):,} values that are not ids skipped"
                   + (f" (e.g. {', '.join(rep['invalid_sample'])})" if rep.get("invalid_sample") else "") + ".")
        pr = V["pairs"]
        tt = pd.DataFrame(index=TA.index)
        tt["name"], tt["id"], tt["mdm_id"] = TA["label"], TA["id"], TA["mdm"]
        tt["received"] = pr[pr["direction"] == "IN"].groupby("target")["amount"].sum()
        tt["sent"] = pr[pr["direction"] == "OUT"].groupby("target")["amount"].sum()
        tt["from_targets"] = R["within"].groupby("dst")["amount"].sum()
        tt["to_targets"] = R["within"].groupby("src")["amount"].sum()
        tt["senders"] = pr[pr["direction"] == "IN"].groupby("target")["party"].nunique()
        tt["receivers"] = pr[pr["direction"] == "OUT"].groupby("target")["party"].nunique()
        tt = tt.fillna({c: 0 for c in ["received", "sent", "from_targets", "to_targets", "senders", "receivers"]})
        tt["net"] = tt["received"] - tt["sent"]
        targets_t = tt.sort_values(["received", "sent"], ascending=False).reset_index(drop=True)
        st.subheader(f"Targets active in the window ({len(targets_t):,})")
        show(targets_t.head(top_n), money_cols=["received", "sent", "from_targets", "to_targets", "net"],
             int_cols=["senders", "receivers"])
        st.subheader(f"Payments between targets ({len(within_t):,} links, {money(win_amt)})")
        show(within_t.head(top_n), money_cols=["amount"], int_cols=["n_txn"])
        st.subheader(f"Ids with no transactions in the window ({len(R['not_found']):,})")
        if R["not_found"]:
            show(pd.DataFrame({"id": R["not_found"]}).head(top_n))
            st.caption("Check the id type chosen for the file and leading zeros (spreadsheets often drop them).")
        else:
            st.caption("none")

if "Members" in tabs:
    with tabs["Members"]:
        m = R["members"]
        kinds = V["kinds"]
        mm = m[m["other_kind"].isin(kinds) | (m["dir"] == "WITHIN")]
        piv = mm.pivot_table(index="target_mdm", columns="dir", values=["amount", "n_parties"], aggfunc="sum", fill_value=0)
        tt = pd.DataFrame(index=piv.index)
        names = m.groupby("target_mdm").agg(name=("customer_name", "first"), party_type=("party_type", "first"))
        tt = tt.join(names)
        for col, d, out in [("amount", "IN", "received"), ("amount", "OUT", "sent"), ("amount", "WITHIN", "paid_to_members"),
                            ("n_parties", "IN", "senders"), ("n_parties", "OUT", "receivers")]:
            tt[out] = piv[(col, d)] if (col, d) in piv.columns else 0
        tt["net"] = tt["received"] - tt["sent"]
        targets_t = tt.reset_index().rename(columns={"target_mdm": "mdm_id"}).sort_values(
            ["received", "sent"], ascending=False).reset_index(drop=True)
        st.subheader(f"Customers with NAICS {P['naics']} active in the window ({len(targets_t):,} of {R['n_members']:,})")
        show(targets_t.head(top_n), money_cols=["received", "sent", "paid_to_members", "net"],
             int_cols=["senders", "receivers"])
        st.caption("Received / sent follow the Show options. paid_to_members: payments to other customers of the same "
                   "NAICS. senders / receivers: distinct parties per kind, summed (approximate for very large counts).")
        wf = R["flow"][R["flow"]["dir"] == "WITHIN"]
        st.subheader(f"Within the NAICS ({money(win_amt)}, {int(wf['n_txn'].sum()):,} txns)")
        if len(wf):
            wr = wf.groupby("rail", as_index=False)[["amount", "n_txn"]].sum().sort_values("amount", ascending=False)
            wr["share_amt"] = wr["amount"] / wr["amount"].sum()
            show(wr.set_index("rail"), money_cols=["amount"], pct_cols=["share_amt"], int_cols=["n_txn"])
        else:
            st.caption("none")

with tabs["Rails"]:
    st.altair_chart(ch_rail_mix(fv, M), **_wide(st.altair_chart))
    r = fv.groupby(["dir", "rail"], as_index=False)[["amount", "n_txn"]].sum()
    r["share_amt"] = r["amount"] / r.groupby("dir")["amount"].transform("sum")
    r["share_txn"] = r["n_txn"] / r.groupby("dir")["n_txn"].transform("sum")
    r["avg_txn"] = r["amount"] / r["n_txn"]
    r = r.merge(hist_percentiles(fv, ["dir", "rail"]), on=["dir", "rail"], how="left")
    r["direction"] = r["dir"].map(lambda d: dir_label(d, M))
    rails_t = r.sort_values(["direction", "amount"], ascending=[True, False]).set_index(["direction", "rail"]) \
               [["amount", "share_amt", "n_txn", "share_txn", "avg_txn", "p10", "p50", "p90", "p99"]]
    st.subheader("Rails by direction")
    show(rails_t, money_cols=["amount", "avg_txn", "p10", "p50", "p90", "p99"], pct_cols=["share_amt", "share_txn"],
         int_cols=["n_txn"])
    st.caption("p10 … p99: per-transaction amount percentiles from a log-bucket histogram (±6%).")
    if not O["only_between"] and len(V["kinds"]) > 1:
        rt = fv.assign(type=fv["other_kind"].map(KIND_LABEL)).pivot_table(
            index=["dir", "rail"], columns="type", values="amount", aggfunc="sum", fill_value=0.0)
        rt = rt.reset_index().assign(direction=lambda d: d["dir"].map(lambda x: dir_label(x, M))) \
               .drop(columns="dir").set_index(["direction", "rail"])
        st.subheader("Rail × party type (dollars)")
        show(rt, money_cols=list(rt.columns))
    dirs = [d for d in ["IN", "OUT", "WITHIN"] if (fv["dir"] == d).any()]
    for col, d in zip(st.columns(max(len(dirs), 1)), dirs):
        col.altair_chart(ch_hist(fv, d, M), **_wide(st.altair_chart))

with tabs["Over time"]:
    st.altair_chart(ch_monthly(fv, M), **_wide(st.altair_chart))
    mo = fv.pivot_table(index="txn_month", columns="dir", values=["amount", "n_txn"], aggfunc="sum", fill_value=0)
    monthly = pd.DataFrame(index=mo.index)
    for d in [x for x in ["IN", "OUT", "WITHIN"] if ("amount", x) in mo.columns]:
        monthly[f"{dir_label(d, M).lower()} $"] = mo[("amount", d)]
        monthly[f"{dir_label(d, M).lower()} txns"] = mo[("n_txn", d)]
    if "received $" in monthly and "sent $" in monthly:
        monthly["net $"] = monthly["received $"] - monthly["sent $"]
    st.subheader("Monthly")
    show(monthly, money_cols=[c for c in monthly.columns if "$" in c], int_cols=[c for c in monthly.columns if "txns" in c])

bank_t = pd.DataFrame()
if "Counterparty banks" in tabs:
    with tabs["Counterparty banks"]:
        ext = V["pairs"][V["pairs"]["kind"] == "cpty"].merge(R["A"][["bank", "cpty_type"]], left_on="party", right_index=True)
        if len(ext):
            ext["bank"] = ext["bank"].replace("", np.nan).fillna("(not recorded)")
            bk = ext.groupby(["direction", "bank"], as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"),
                                                                        parties=("party", "nunique"))
            c1, c2 = st.columns(2)
            for col, d in [(c1, "IN"), (c2, "OUT")]:
                sub = bk[bk["direction"] == d]
                if len(sub):
                    col.altair_chart(ch_bars(sub, "bank", f"Counterparty banks — {dir_label(d, M).lower()}", C_EXT),
                                     **_wide(st.altair_chart))
            bank_t = bk.assign(share_of_counterparty_amt=bk["amount"] / bk.groupby("direction")["amount"].transform("sum"),
                               direction=bk["direction"].map(lambda d: dir_label(d, M))) \
                       .sort_values(["direction", "amount"], ascending=[True, False]).set_index(["direction", "bank"])
            st.subheader("Banks of counterparties")
            show(bank_t, money_cols=["amount"], pct_cols=["share_of_counterparty_amt"], int_cols=["n_txn", "parties"])
            ct = ext.assign(cpty_type=ext["cpty_type"].fillna("(none)")).groupby(["direction", "cpty_type"], as_index=False) \
                    .agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"), parties=("party", "nunique"))
            ct["direction"] = ct["direction"].map(lambda d: dir_label(d, M))
            st.subheader("Counterparty types")
            show(ct.set_index(["direction", "cpty_type"]), money_cols=["amount"], int_cols=["n_txn", "parties"])
        else:
            st.caption("No counterparties in this view.")

tp_named, emb = pd.DataFrame(), pd.DataFrame()
if "Connections" in tabs:
    with tabs["Connections"]:
        tv = V["ties"]
        if M == "naics":
            st.caption(f"Payments between the top {TIE_NAICS_CAP} other NAICS by dollars with the target NAICS, and "
                       "between them and external counterparties; the target NAICS's own customers are excluded. "
                       "Links with the 'NAICS unknown' node are not computed.")
        else:
            st.caption(f"Payments between the connected parties, every target excluded. Searched among the top "
                       f"{TIE_SEARCH_CAP:,} PNC customers and the top {TIE_SEARCH_CAP:,} counterparties by dollars with "
                       "the target(s). Counterparty-to-counterparty payments are not in the source table.")
        if len(tv):
            tp = tv.groupby(["src", "dst", "kind"], as_index=False).agg(amount=("amount", "sum"), n_txn=("n_txn", "sum"))
            tp["kind"] = tp["kind"].map({"pnc_pnc": "NAICS ↔ NAICS" if M == "naics" else "PNC ↔ PNC",
                                         "pnc_cpty": ("NAICS" if M == "naics" else "PNC") + " ↔ counterparty"})
            st.subheader("Payment links between connected parties")
            show(tp.groupby("kind").agg(links=("amount", "size"), amount=("amount", "sum"), n_txn=("n_txn", "sum")),
                 money_cols=["amount"], int_cols=["links", "n_txn"])
            deg = pd.concat([tp[["src", "dst", "amount"]].rename(columns={"src": "k", "dst": "o"}),
                             tp[["dst", "src", "amount"]].rename(columns={"dst": "k", "src": "o"})])
            emb = deg.groupby("k").agg(partners=("o", "nunique"), link_amount=("amount", "sum"))
            emb = emb.join(R["A"][["label", "kind", "cp_id"]]).join(pf["total"].rename("flow_with_target"))
            emb["type"] = emb["kind"].map(KIND_LABEL)
            emb = emb.sort_values(["partners", "link_amount"], ascending=False).reset_index(drop=True)
            emb = emb[["label", "type", "cp_id", "partners", "link_amount", "flow_with_target"]].rename(
                columns={"label": "name", "cp_id": "id"})
            st.subheader("Most connected parties inside the network")
            show(emb.head(top_n), money_cols=["link_amount", "flow_with_target"], int_cols=["partners"])
            lab = R["A"]["label"]
            tp_named = tp.assign(payer=tp["src"].map(lab), payee=tp["dst"].map(lab)) \
                         .sort_values("amount", ascending=False)[["payer", "payee", "kind", "amount", "n_txn"]]
            st.subheader("Largest links")
            show(tp_named.head(top_n).reset_index(drop=True), money_cols=["amount"], int_cols=["n_txn"])
        else:
            st.caption("No payments between connected parties for the selected connection types.")

if "Own accounts" in tabs:
    with tabs["Own accounts"]:
        oa = R["own_accts"].pivot_table(index="t_acct", columns="direction", values=["amount", "n_txn", "parties"],
                                        aggfunc="sum", fill_value=0)
        oa.columns = [f"{dir_label(d, M).lower()} {v}" for v, d in oa.columns]
        oa = oa.sort_values([c for c in oa.columns if "amount" in c], ascending=False)
        st.subheader(f"Which of the customer's accounts carry the flow ({len(oa):,})")
        show(oa, money_cols=[c for c in oa.columns if "amount" in c], int_cols=[c for c in oa.columns if "amount" not in c])
        sf = R["flow"][R["flow"]["dir"] == "SELF"]
        st.caption(f"All parties, before the Show options. Transfers between the customer's own accounts (not drawn): "
                   f"{int(sf['n_txn'].sum()):,} txns, {money(sf['amount'].sum())}.")

with tabs["Summary & QA"]:
    if not O["only_between"]:
        ty = fv.groupby(["dir", "other_kind"], as_index=False)[["amount", "n_txn"]].sum()
        ty["share_amt"] = ty["amount"] / ty.groupby("dir")["amount"].transform("sum")
        ty["avg_txn"] = ty["amount"] / ty["n_txn"]
        ty["direction"] = ty["dir"].map(lambda d: dir_label(d, M))
        ty["type"] = ty["other_kind"].map(KIND_LABEL)
        st.subheader("By party type")
        show(ty.set_index(["direction", "type"])[["amount", "share_amt", "n_txn", "avg_txn"]],
             money_cols=["amount", "avg_txn"], pct_cols=["share_amt"], int_cols=["n_txn"])
    st.subheader("Transaction size")
    show(hist_percentiles(fv, ["dir"]).assign(direction=lambda d: d["dir"].map(lambda x: dir_label(x, M)))
         .drop(columns="dir").set_index("direction"), money_cols=["p10", "p50", "p90", "p99"])
    if not O["only_between"]:
        st.subheader("Concentration")
        conc = pd.DataFrame({"Received": concentration(T_in), "Sent": concentration(T_out)}).T
        show(conc, pct_cols=["top 1", "top 5", "top 10"], int_cols=["parties"])
        st.caption("effective n = 1 / HHI: the number of equal-sized parties giving the same concentration.")
        if len(T_in) or len(T_out):
            st.altair_chart(ch_concentration(T_in, T_out), **_wide(st.altair_chart))
    st.subheader("Data quality")
    qa = R["flow"].groupby(["dir", "other_kind"]).agg(n_txn=("n_txn", "sum"), amount=("amount", "sum"))
    show(qa, money_cols=["amount"], int_cols=["n_txn"])
    st.caption("Every transaction touching the target(s), before the Show options. SELF = a target paying itself "
               "(own-account transfers for an MDM id), not drawn. cpty_noid = counterparty side with no id; counted "
               "here, not drawn. Party totals are checked against these flows on every view.")

with st.sidebar:
    st.header("Export")
    tables = {"senders": T_in, "receivers": T_out, "two_way": tw, "targets": targets_t, "between_targets": within_t,
              "rails": rails_t, "monthly": monthly, "banks": bank_t, "links": tp_named, "most_connected": emb}
    tag = P["naics"] if M == "naics" else (P["ids"][0] if M == "single" else f"list{len(P['ids'])}")
    st.download_button("Download tables + network (zip)", zip_bytes(tables, standalone_html(nodes, edges, options)),
                       file_name=f"ego_{M}_{tag}_{P['start']}_{P['end']}.zip",
                       mime="application/zip", **_wide(st.download_button))
