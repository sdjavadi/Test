"""
Prospect Finder: data access, filters and graph layout, with no Streamlit inside, so the dev team can wrap it in
FastAPI as is.

A counterparty is an outside company (an ent_l2 entity: accounts merged across names and banks) that PNC clients
pay or are paid by.

  universe_sql          one row per counterparty for the period: size, reach, first month seen, change, bank
  cpty_trend_sql        one counterparty, month by month
  cpty_clients_sql      its PNC clients (TM detail) and the money each way
  cpty_accounts_sql     its accounts, the bank of each (canonical name) and the names it appeared under
  cpty_account_edges_sql  money between its accounts and PNC clients (account-level graph)

Every list filter runs in pandas on the cached universe. Column names that are reserved words in Impala (scope,
dir, month ...) are always written in backticks via Q. No free text from the UI reaches SQL: ids are checked
against a fixed pattern.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("prospect_finder")

# ================================================================== configuration (check on the first run)
DB = "bdahd01p_dlcdi1_cdi_tm"
T_CPTY_MET = f"{DB}.pkg_cpty_metrics_monthly"   # entity x month; _in = the counterparty receives from PNC clients
T_ENT_DIM = f"{DB}.pkg_cpty_entity_dim"         # pnc_match_n = same cleaned name as a PNC organisation
T_PAIR = f"{DB}.pkg_cust_pair_month_v3"         # customer x dir x scope x other; dir is the CUSTOMER's view
T_CDIM = f"{DB}.pkg_cust_dim"                   # names, power ids, is_tm, NAICS
T_MAP = f"{DB}.pkg_cpty_entity_map_v2"          # account id (cpty_id) -> ent_l2
T_CPTY_MONTH = f"{DB}.pkg_cpty_month"           # account id x month: cpty_name, fi, amount, n_txn
T_EDGE_M = f"{DB}.pkg_edge_monthly"             # src/dst x month, CUST or CPTY on each side
T_FI_ROUTING = f"{DB}.pkg_fi_routing"           # routing number -> fi_canonical
T_FI_NAMES = f"{DB}.pkg_fi_name_map"            # fi_raw -> fi_canonical
IMPALA = {"dsn": "DSN=bdpimp04-impala;", "pool": "root.CIB-AMG_Impala", "conn_options": {"SocketTimeout": 0}}

FIRST_MONTH, LAST_MONTH = "2024-01", "2026-08"
DEFAULT_END, DEFAULT_LEN = "2026-06", 6     # Jul-Aug 2026 carry a dollar jump in the source; amounts stable to June
PERIOD_LENGTHS = [3, 6, 12]
UNIVERSE_MIN_CLIENTS = 3                    # loaded once per period; every other size filter runs in pandas
CLASSES = ("business", "unknown")           # typer v2 name classes kept
MAX_ROWS = 400_000                          # safety cap on the universe pull (largest by PNC clients kept)
AUDIT_PATH = os.environ.get("PROSPECT_AUDIT", "audit/prospect_finder_audit.csv")

# Impala reserved words used as column names: always quote them.
Q = {c: f"`{c}`" for c in ("month", "dir", "scope", "category", "source")}

# ================================================================== reference lists
SECTORS = {
    "11": "Agriculture", "21": "Mining, oil and gas", "22": "Utilities", "23": "Construction",
    "31-33": "Manufacturing", "42": "Wholesale trade", "44-45": "Retail trade", "48-49": "Transportation, warehousing",
    "51": "Information", "52": "Finance and insurance", "53": "Real estate, rental, leasing",
    "54": "Professional, scientific, technical", "55": "Management of companies", "56": "Administrative, support, waste",
    "61": "Educational services", "62": "Health care, social assistance", "71": "Arts, entertainment, recreation",
    "72": "Accommodation, food services", "81": "Other services", "92": "Public administration"}
NAICS2_SECTOR = {c: v for k, v in SECTORS.items()
                 for c in (map(str, range(int(k[:2]), int(k[-2:]) + 1)) if "-" in k else [k])}
HIDE_RE = (r"\bPNC\b|\b(?:IRS|US TREASURY|TREASURY|TREAS|SSA|STATE OF|DEPT OF|DEPARTMENT OF|COUNTY OF|CITY OF|"
           r"COMMONWEALTH OF|SALES ?TAX|USE ?TAX|TAXPYMT|DEPT REV|VISA|MASTERCARD|AMERICAN EXPRESS|AMEX|DISCOVER)\b")
GENERIC_TOKENS = {"BNKCD", "BANKCARD", "BANKCD", "BNKCRD", "MERCH", "MERCHANT", "MERCHANTS", "MRCH", "SVC", "SVCS",
                  "SERVICE", "SERVICES", "SETTLE", "SETTLEMENT", "STLMT", "DEP", "DEPOSIT", "EMPLOYEE", "BENEFIT",
                  "BENEFITS", "PAYMENT", "PAYMENTS", "PAY", "CARD", "CARDS", "DEBIT", "CREDIT", "ACH", "TRANSFER",
                  "FEE", "FEES", "DISC", "DISCOUNT", "BATCH", "PAYROLL", "REFUND", "PROCESSING", "DEPOSITS", "CHGBK",
                  "MER", "BK", "BNK", "MRCHNT", "MERCHNT", "BANKCRD", "STLMNT", "SETL", "CRD", "DEPST", "SERV",
                  "SERVS", "SRV", "SRVS", "NSD", "SD", "STL", "STLM", "CHGBCK", "DISCNT", "ACCT", "ACC", "ONLINE",
                  "WEB", "PPD", "CCD", "WIRE", "BILL", "INV", "INVOICE", "TAX", "COLLECTION", "RETURN", "ORIG"}
FLOWS = ["Receives from PNC clients", "Pays PNC clients", "Both ways"]
FIRST_SEEN = {"Any time": None, "In the last month": 1, "In the last 3 months": 3, "In the last 6 months": 6,
              "In the last 12 months": 12}
CHANGE = {"Any": None, "Growing 10% or more": 0.10, "Growing 25% or more": 0.25, "Growing 50% or more": 0.50,
          "Doubling or more": 1.00, "Shrinking 25% or more": -0.25}
SORTS = {"Money with PNC clients": "amt", "PNC clients": "clients", "TM clients": "tm_all",
         "Change in PNC clients": "change", "Newest first": "first_month"}

# ================================================================== Impala access
_ID_RE = re.compile(r"^[A-Za-z0-9:._|\-]{1,80}$")


def run_sql(sql: str, label: str = "") -> pd.DataFrame:
    """One Impala query through the in-house dbi module; time and size go to the server log only."""
    import dbi
    t0 = time.time()
    df = dbi.db_get_query(sql, **IMPALA)
    df.columns = [str(c).lower().split(".")[-1] for c in df.columns]
    log.info("query %s: %.1fs, %d rows", label, time.time() - t0, len(df))
    return df


def table_columns(table: str) -> set[str] | None:
    """Lower-case column names, or None when the table cannot be read."""
    try:
        return set(run_sql(f"SELECT * FROM {table} LIMIT 0", f"columns {table}").columns)
    except Exception:
        log.warning("cannot read %s", table, exc_info=True)
        return None


def _lit(values) -> str:
    return ", ".join("'" + str(v) + "'" for v in values)


def _check_id(x: str) -> str:
    if not _ID_RE.match(str(x)):
        raise ValueError("unexpected id format")
    return str(x)


def add_months(m: str, k: int) -> str:
    y, mo = int(m[:4]), int(m[5:]) + k
    y, mo = y + (mo - 1) // 12, (mo - 1) % 12 + 1
    return f"{y:04d}-{mo:02d}"


def month_range(a: str, b: str) -> list[str]:
    out, m = [], a
    while m <= b:
        out.append(m)
        m = add_months(m, 1)
    return out


def period(end: str, n: int) -> tuple[str, ...]:
    return tuple(month_range(max(FIRST_MONTH, add_months(end, 1 - n)), end))


def mask_account(cpty_id: str) -> str:
    """Routing number and the last 4 digits of the account only."""
    s = str(cpty_id)
    if "-" in s:
        rt, acct = s.split("-", 1)
        return f"{rt}-…{acct[-4:]}"
    return f"…{s[-4:]}"


# ================================================================== universe
def universe_sql(per, floor, cols: set[str], dim_cols: set[str] | None, max_rows: int = MAX_ROWS) -> str:
    """
    One row per counterparty with eligible metrics and at least `floor` PNC clients per active month in the period.
      per-month figures  = averages over the months it was active (n_cust_all > 0) in the period
      clients_last3/prev3 = average PNC clients over the last 3 months of the period and the 3 before
                            (absent months count as zero), for the change filter
      first_month        = first month with PNC clients since FIRST_MONTH (history starts there)
      bank today         = fi_dom in its last active month of the period
    """
    need = {"ent_id", "month", "ent_name", "name_class", "fi_dom", "eligible", "amt_in", "amt_out", "n_cust_in",
            "n_cust_out", "n_cust_all"}
    miss = need - cols
    if miss:
        raise ValueError(f"counterparty metrics lack {sorted(miss)}")
    M, end, p0 = Q["month"], per[-1], per[0]
    r0, q0 = add_months(end, -2), add_months(end, -5)
    lo = min(p0, q0)
    inp = f"{M} >= '{p0}'"
    act = f"{inp} AND n_cust_all > 0"
    avg = lambda c, a: (f"SUM(CASE WHEN {act} THEN CAST({c} AS DOUBLE) ELSE 0 END) / "
                        f"NULLIF(SUM(CASE WHEN {act} THEN 1 ELSE 0 END), 0) AS {a}")
    aggs = [f"SUM(CASE WHEN {act} THEN 1 ELSE 0 END) AS months_active",
            f"MAX(CASE WHEN {act} THEN {M} END) AS last_month",
            avg("amt_in", "amt_in"), avg("amt_out", "amt_out"), avg("n_cust_in", "clients_in"),
            avg("n_cust_out", "clients_out"), avg("n_cust_all", "clients"),
            f"SUM(CASE WHEN {M} >= '{r0}' THEN CAST(n_cust_all AS DOUBLE) ELSE 0 END) / 3 AS clients_last3",
            f"SUM(CASE WHEN {M} >= '{q0}' AND {M} < '{r0}' THEN CAST(n_cust_all AS DOUBLE) ELSE 0 END) / 3 "
            f"AS clients_prev3"]
    if {"n_cust_tm_in", "n_cust_tm_out"} <= cols:
        aggs += [avg("n_cust_tm_in", "tm_in"), avg("n_cust_tm_out", "tm_out")]
    last = ["t.ent_name", "t.fi_dom"]
    last += ["CAST(t.n_fi AS DOUBLE) AS n_fi"] if "n_fi" in cols else []
    dim = [f"d.{c}" for c in ("pnc_match_n",) if dim_cols and c in dim_cols]
    a_out = [s.rsplit(" AS ", 1)[1] for s in aggs if not s.endswith("last_month")]
    l_out = [c.rsplit(" AS ", 1)[-1].replace("t.", "") for c in last]
    sel = ", ".join(["a.ent_id"] + [f"a.{c}" for c in a_out] + [f"l.{c}" for c in l_out] + ["f.first_month"] + dim)
    base = f"eligible AND name_class IN ({_lit(CLASSES)})"
    return f"""
WITH a AS (
  SELECT ent_id, {', '.join(aggs)}
  FROM {T_CPTY_MET}
  WHERE {M} BETWEEN '{lo}' AND '{end}' AND {base}
  GROUP BY ent_id
  HAVING SUM(CASE WHEN {act} THEN CAST(n_cust_all AS DOUBLE) ELSE 0 END)
         >= {float(floor)} * SUM(CASE WHEN {act} THEN 1 ELSE 0 END)
     AND SUM(CASE WHEN {act} THEN 1 ELSE 0 END) > 0
),
l AS (
  SELECT t.ent_id, {', '.join(last)}
  FROM {T_CPTY_MET} t JOIN a ON t.ent_id = a.ent_id AND t.{M} = a.last_month
  WHERE t.{M} BETWEEN '{p0}' AND '{end}'
),
f AS (
  SELECT t.ent_id, MIN(t.{M}) AS first_month
  FROM {T_CPTY_MET} t JOIN a ON t.ent_id = a.ent_id
  WHERE t.{M} <= '{end}' AND t.eligible AND t.n_cust_all > 0
  GROUP BY t.ent_id
)
SELECT {sel}
FROM a JOIN l ON a.ent_id = l.ent_id
LEFT JOIN f ON a.ent_id = f.ent_id
{f"LEFT JOIN {T_ENT_DIM} d ON a.ent_id = d.ent_id" if dim else ""}
ORDER BY a.clients DESC
LIMIT {int(max_rows)}"""


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def is_generic(name: str) -> bool:
    toks = re.findall(r"[A-Z0-9]+", name)
    rest = [t for t in toks if t not in GENERIC_TOKENS and not re.fullmatch(r"[A-Z]*\d+[A-Z\d]*", t) and len(t) > 2]
    return len(rest) == 0


def prepare_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Typed columns, derived measures and flags; run once per cached universe."""
    df = df.drop_duplicates("ent_id").reset_index(drop=True)
    df = _num(df, ["months_active", "amt_in", "amt_out", "clients_in", "clients_out", "clients", "clients_last3",
                   "clients_prev3", "tm_in", "tm_out", "n_fi", "pnc_match_n"])
    df["ent_name"] = df.ent_name.fillna("").astype(str).str.upper().str.strip()
    df["first_month"] = df.first_month.astype(str).str[:7].where(df.first_month.notna())
    df["amt"] = df.amt_in.fillna(0) + df.amt_out.fillna(0)
    share_in = df.amt_in.fillna(0) / df.amt.where(df.amt > 0)
    df["flow"] = np.select([share_in >= 0.8, share_in <= 0.2], FLOWS[:2], FLOWS[2])
    prev, last = df.clients_prev3.fillna(0), df.clients_last3.fillna(0)
    df["change"] = np.where(prev > 0, last / prev.where(prev > 0, 1) - 1, np.where(last > 0, np.inf, np.nan))
    if {"tm_in", "tm_out"} <= set(df.columns):
        df["tm_all"] = df.tm_in.fillna(0) + df.tm_out.fillna(0)
    df["pnc_account"] = (np.where(df.pnc_match_n.fillna(0) > 0, "Yes", "No") if "pnc_match_n" in df
                         else "Unknown")
    df["hidden_kind"] = df.ent_name.str.contains(HIDE_RE, regex=True)
    df["generic"] = df.ent_name.map(is_generic)
    return df


# ================================================================== filters, presets, sorting
@dataclass
class Filters:
    name_contains: str = ""
    pnc_account: str = "Any"              # Any / Yes / No
    flows: list = field(default_factory=list)
    banks: list = field(default_factory=list)
    exclude_banks: list = field(default_factory=list)
    min_banks: int = 0
    min_clients: float = 10.0
    min_tm: float = 0.0
    min_amt_k: float = 0.0
    min_months: int = 0
    first_seen: str = "Any time"
    change: str = "Any"
    hide_pnc_gov_card: bool = True
    hide_generic: bool = True
    sort: str = "Money with PNC clients"


FILTER_DEFAULTS = asdict(Filters())

PRESETS = {  # each preset resets the filters to their defaults, then sets these; "all" = every month of the period
    "None": {"desc": "No preset: set the filters yourself.", "set": {}},
    "Largest by money": {
        "desc": "At least $1M a month with PNC clients and active every month of the period; sorted by money.",
        "set": {"min_amt_k": 1000.0, "min_months": "all", "sort": "Money with PNC clients"}},
    "Widest reach": {
        "desc": "At least 100 PNC clients a month and active every month of the period; sorted by PNC clients.",
        "set": {"min_clients": 100.0, "min_months": "all", "sort": "PNC clients"}},
    "New counterparties": {
        "desc": "First seen with PNC clients in the last 3 months of the period, with at least 10 PNC clients a "
                "month; newest first.",
        "set": {"first_seen": "In the last 3 months", "min_clients": 10.0, "sort": "Newest first"}},
    "Growing fast": {
        "desc": "PNC clients up 50% or more (last 3 months of the period vs the 3 before), with at least 20 PNC "
                "clients a month; sorted by the change.",
        "set": {"change": "Growing 50% or more", "min_clients": 20.0, "sort": "Change in PNC clients"}},
    "No PNC account": {
        "desc": "No PNC account under its name, at least 50 PNC clients a month; sorted by money.",
        "set": {"pnc_account": "No", "min_clients": 50.0, "sort": "Money with PNC clients"}},
}


def apply_filters(df: pd.DataFrame, f: Filters, end: str) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if f.name_contains.strip():
        m &= df.ent_name.str.contains(re.escape(f.name_contains.strip().upper()), regex=True)
    if f.pnc_account in ("Yes", "No"):
        m &= df.pnc_account == f.pnc_account
    if f.flows:
        m &= df.flow.isin(f.flows)
    if f.banks:
        m &= df.fi_dom.isin(f.banks)
    if f.exclude_banks:
        m &= ~df.fi_dom.isin(f.exclude_banks)
    if f.min_banks and "n_fi" in df:
        m &= df.n_fi.fillna(0) >= f.min_banks
    m &= df.clients >= f.min_clients
    if f.min_tm and "tm_all" in df:
        m &= df.tm_all >= f.min_tm
    m &= df.amt >= 1e3 * f.min_amt_k
    m &= df.months_active >= f.min_months
    k = FIRST_SEEN.get(f.first_seen)
    if k:
        m &= df.first_month.fillna("") >= add_months(end, 1 - k)
    c = CHANGE.get(f.change)
    if c is not None:
        m &= (df.change >= c) if c > 0 else (df.change <= c)
    if f.hide_pnc_gov_card:
        m &= ~df.hidden_kind
    if f.hide_generic:
        m &= ~df.generic
    return m


def sort_frame(df: pd.DataFrame, sort: str) -> pd.DataFrame:
    col = SORTS.get(sort, "amt")
    col = col if col in df else "amt"
    return df.sort_values([col, "amt"], ascending=False, na_position="last")


# ================================================================== one counterparty
RAILS = [("ach", "ACH"), ("wire", "Wire"), ("rtp_prt", "RTP"), ("rtp_p2p", "RTP P2P"), ("check", "Check"),
         ("pcard", "Card")]
# Month-by-month charts: (title, unit, [(column, series label)], what it means). A chart shows the columns that exist.
TREND_PANELS = [
    ("PNC clients", "count", [("n_cust_in", "paying it"), ("n_cust_out", "it pays")],
     "Distinct PNC clients in the month."),
    ("TM clients", "count", [("n_cust_tm_in", "paying it"), ("n_cust_tm_out", "it pays")],
     "Distinct TM clients in the month."),
    ("Money with PNC clients, $K", "usd_k", [("amt_in", "in"), ("amt_out", "out")],
     "Money it received from and paid to PNC clients."),
    ("Average payment, $", "usd", [("avg_ticket_in", "in"), ("avg_ticket_out", "out")],
     "Money divided by the number of payments."),
    ("Days with payments", "count", [("active_days_in", "in"), ("active_days_out", "out")],
     "Days in the month with at least one payment: daily payers look like processors or billers."),
    ("Recurring clients", "count", [("rec_standing_in", "paying it"), ("rec_standing_out", "it pays")],
     "Clients seen in at least 4 of the previous 6 months: the steady base of the relationship."),
    ("Client retention, %", "pct", [("retention_in", "paying it"), ("retention_out", "it pays")],
     "Share of last month's clients still there this month."),
    ("Money from new clients, %", "pct", [("new_amt_share_in", "in"), ("new_amt_share_out", "out")],
     "Share of the month's money with clients not seen in the previous 12 months: high means it is winning clients."),
    ("Largest client's share, %", "pct", [("top1_share_in", "in"), ("top1_share_out", "out")],
     "Share of the month's money with its single largest PNC client: high means it depends on one client."),
    ("Its weight for its clients, %", "pct", [("dep_wavg_in", "clients paying it"), ("dep_wavg_out", "clients it pays")],
     "For its PNC clients, its share of their outside payments, weighted by money: high means they rely on it."),
    ("Money with organisations, %", "pct", [("org_share_in", "in"), ("org_share_out", "out")],
     "Share of its money with organisations rather than individuals."),
    ("Payment rails, money in, %", "pct", [(f"share_{r}_in", lab) for r, lab in RAILS],
     "How PNC clients pay it."),
    ("Payment rails, money out, %", "pct", [(f"share_{r}_out", lab) for r, lab in RAILS],
     "How it pays PNC clients."),
    ("Banks and account ids", "count", [("n_fi", "banks"), ("n_ids_active", "active account ids")],
     "Banks holding its accounts, and its account ids active in the month."),
]


def trend_columns(cols: set[str]) -> list[str]:
    return [c for _, _, series, _ in TREND_PANELS for c, _ in series if c in cols]


def cpty_trend_sql(ent_id: str, months, cols: set[str]) -> str:
    M = Q["month"]
    sel = ", ".join(f"CAST({c} AS DOUBLE) AS {c}" for c in trend_columns(cols))
    return (f"SELECT {M} AS mon, {sel} FROM {T_CPTY_MET} "
            f"WHERE {M} BETWEEN '{months[0]}' AND '{months[-1]}' AND ent_id = '{_check_id(ent_id)}' ORDER BY {M}")


def search_names(U: pd.DataFrame, q: str, limit: int = 20) -> pd.DataFrame:
    """Counterparties whose name contains every word typed (any order); names starting with it first."""
    words = re.sub(r"[^A-Z0-9&]+", " ", str(q).upper()).split()
    if not words or sum(map(len, words)) < 2:
        return U.iloc[0:0]
    m = pd.Series(True, index=U.index)
    for w in words:
        m &= U.ent_name.str.contains(re.escape(w), regex=True)
    R = U[m].assign(_starts=lambda d: d.ent_name.str.startswith(" ".join(words)))
    return R.sort_values(["_starts", "amt"], ascending=False).drop(columns="_starts").head(limit)


def cpty_clients_sql(ent_id: str, per, limit: int = 2000) -> str:
    """Every PNC client with money to or from this counterparty in the period (dir is the client's view)."""
    M, D, S = Q["month"], Q["dir"], Q["scope"]
    amt = "CAST(p.amount AS DOUBLE)"
    ntx = "CAST(p.n_txn AS DOUBLE)"
    return f"""
SELECT p.cust AS mdm_id, c.customer_name, c.customer_pwr_id, c.is_tm, c.party_type, c.naics_cd, c.naics_desc,
       CAST(c.naics2 AS STRING) AS naics2, c.naics_status,
       SUM(CASE WHEN p.{D} = 'out' THEN {amt} ELSE 0 END) AS paid_to_it,
       SUM(CASE WHEN p.{D} = 'in' THEN {amt} ELSE 0 END) AS received_from_it,
       SUM(CASE WHEN p.{D} = 'out' THEN {ntx} ELSE 0 END) AS n_paid,
       SUM(CASE WHEN p.{D} = 'in' THEN {ntx} ELSE 0 END) AS n_received,
       COUNT(DISTINCT p.{M}) AS months
FROM {T_PAIR} p JOIN {T_CDIM} c ON p.cust = c.mdm_id
WHERE p.{M} BETWEEN '{per[0]}' AND '{per[-1]}' AND p.{S} = 'ext' AND p.other = '{_check_id(ent_id)}'
GROUP BY p.cust, c.customer_name, c.customer_pwr_id, c.is_tm, c.party_type, c.naics_cd, c.naics_desc, c.naics2,
         c.naics_status
ORDER BY SUM({amt}) DESC
LIMIT {int(limit)}"""


def cpty_accounts_sql(ent_id: str, per, routing_col: str | None, have_names: bool, limit: int = 5000) -> str:
    """Accounts merged into the counterparty, with name and bank as they appear on the payments in the period."""
    M = Q["month"]
    joins, bank = [], ["m.fi"]
    if routing_col:
        joins.append(f"LEFT JOIN (SELECT CAST({routing_col} AS STRING) AS rt, MAX(fi_canonical) AS fi_c "
                     f"FROM {T_FI_ROUTING} GROUP BY CAST({routing_col} AS STRING)) r "
                     f"ON r.rt = regexp_extract(m.cpty_id, '^([0-9]{{9}})-', 1)")
        bank.insert(0, "r.fi_c")
    if have_names:
        joins.append(f"LEFT JOIN (SELECT fi_raw, MAX(fi_canonical) AS fi_c FROM {T_FI_NAMES} GROUP BY fi_raw) nm "
                     f"ON nm.fi_raw = m.fi")
        bank.insert(1 if routing_col else 0, "nm.fi_c")
    return f"""
WITH acct_ids AS (SELECT DISTINCT cpty_id FROM {T_MAP} WHERE ent_l2 = '{_check_id(ent_id)}'),
m AS (
  SELECT t.cpty_id, t.cpty_name, t.fi, SUM(CAST(t.amount AS DOUBLE)) AS amt, SUM(CAST(t.n_txn AS DOUBLE)) AS n_txn,
         MIN(t.{M}) AS first_month, MAX(t.{M}) AS last_month
  FROM {T_CPTY_MONTH} t JOIN acct_ids ON t.cpty_id = acct_ids.cpty_id
  WHERE t.{M} BETWEEN '{per[0]}' AND '{per[-1]}'
  GROUP BY t.cpty_id, t.cpty_name, t.fi
)
SELECT m.cpty_id, m.cpty_name, m.fi, COALESCE({', '.join(bank)}) AS bank, m.amt, m.n_txn, m.first_month, m.last_month
FROM m
{chr(10).join(joins)}
ORDER BY m.amt DESC
LIMIT {int(limit)}"""


def cpty_account_edges_sql(ent_id: str, per) -> str:
    """Money between each of its accounts and each PNC client in the period (self-transfers left out)."""
    M = Q["month"]
    w = (f"e.{M} BETWEEN '{per[0]}' AND '{per[-1]}' AND NOT COALESCE(e.is_self, FALSE)")
    return f"""
WITH acct_ids AS (SELECT DISTINCT cpty_id FROM {T_MAP} WHERE ent_l2 = '{_check_id(ent_id)}')
SELECT e.src_id AS mdm_id, e.dst_id AS cpty_id, 'paid_to_it' AS flow_dir,
       SUM(CAST(e.amount AS DOUBLE)) AS amt, SUM(CAST(e.n_txn AS DOUBLE)) AS n_txn
FROM {T_EDGE_M} e JOIN acct_ids ON e.dst_id = acct_ids.cpty_id
WHERE {w} AND e.src_kind = 'CUST' AND e.dst_kind = 'CPTY'
GROUP BY e.src_id, e.dst_id
UNION ALL
SELECT e.dst_id AS mdm_id, e.src_id AS cpty_id, 'received_from_it' AS flow_dir,
       SUM(CAST(e.amount AS DOUBLE)) AS amt, SUM(CAST(e.n_txn AS DOUBLE)) AS n_txn
FROM {T_EDGE_M} e JOIN acct_ids ON e.src_id = acct_ids.cpty_id
WHERE {w} AND e.src_kind = 'CPTY' AND e.dst_kind = 'CUST'
GROUP BY e.dst_id, e.src_id"""


def prepare_clients(df: pd.DataFrame) -> pd.DataFrame:
    df = _num(df.copy(), ["paid_to_it", "received_from_it", "n_paid", "n_received", "months"])
    df["is_tm"] = df.is_tm.astype(str).str.lower().isin(["true", "1"])
    df["total"] = df.paid_to_it.fillna(0) + df.received_from_it.fillna(0)
    valid = df.naics_status.astype(str) == "valid"
    df["sector"] = np.where(valid, df.naics2.astype(str).map(NAICS2_SECTOR).fillna("Unknown"), "Unknown")
    df["industry"] = np.where(valid, (df.naics_cd.fillna("").astype(str) + " "
                                      + df.naics_desc.fillna("").astype(str)).str.strip(), "")
    return df


def prepare_accounts(df: pd.DataFrame) -> pd.DataFrame:
    df = _num(df.copy(), ["amt", "n_txn"])
    df["bank"] = df.bank.fillna(df.fi).fillna("(unknown)").astype(str).str.upper().str.strip()
    df["cpty_name"] = df.cpty_name.fillna("(no name)").astype(str)
    df["account"] = df.cpty_id.map(mask_account)
    return df


# ================================================================== graph layout (pure functions)
NODE_COLS = ["id", "label", "kind", "x", "y", "amt", "hover"]
EDGE_COLS = ["src", "dst", "flow", "amt", "n_txn"]


def _arc(n: int, a0: float, a1: float, r: float = 1.0):
    if n == 0:
        return []
    ang = [a0 + (a1 - a0) * (i + 0.5) / n for i in range(n)]
    return [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a))) for a in ang]


def _client_hover(r) -> str:
    bits = [str(r.customer_name), f"power id {r.customer_pwr_id}" if pd.notna(r.get("customer_pwr_id")) else "",
            r.industry if isinstance(r.get("industry"), str) and r.industry else "",
            f"pays it ${r.paid_to_it / 1e3:,.0f}K · receives ${r.received_from_it / 1e3:,.0f}K"]
    return "<br>".join(b for b in bits if b)


def graph_consolidated(name: str, CL: pd.DataFrame, top_n: int):
    """The counterparty in the middle, its largest TM clients around it; other PNC clients as one node.
    Clients whose money mostly goes to it sit on the left, clients it mostly pays on the right."""
    tm = CL[CL.is_tm].sort_values("total", ascending=False)
    top = tm.head(top_n)
    rest = pd.concat([tm.iloc[top_n:], CL[~CL.is_tm]])
    nodes = [{"id": "CPTY", "label": name, "kind": "counterparty", "x": 0.0, "y": 0.0,
              "amt": CL.total.sum(), "hover": f"{name}<br>${CL.total.sum() / 1e3:,.0f}K with PNC clients"}]
    edges = []
    ring = [(r.paid_to_it / r.total if r.total else 0.5, r.total, "c", r) for _, r in top.iterrows()]
    rows = [(r.mdm_id, r.paid_to_it, r.n_paid, r.received_from_it, r.n_received) for _, r in top.iterrows()]
    if len(rest) and rest.total.sum() > 0:
        rp, rr = rest.paid_to_it.sum(), rest.received_from_it.sum()
        ring.append((rp / (rp + rr), rp + rr, "o", None))
        rows.append(("OTHER", rp, rest.n_paid.sum(), rr, rest.n_received.sum()))
    # around the full circle from the top, counter-clockwise: clients mostly paying it first (left side),
    # clients it mostly pays last (right side)
    ring.sort(key=lambda t: (-t[0], -t[1]))
    for i, (_, amt, kind, r) in enumerate(ring):
        ang = math.radians(90 + 360 * (i + 0.5) / len(ring))
        x, y = math.cos(ang), math.sin(ang)
        if kind == "c":
            nodes.append({"id": r.mdm_id, "label": str(r.customer_name), "kind": "client", "x": x, "y": y,
                          "amt": r.total, "hover": _client_hover(r)})
        else:
            nodes.append({"id": "OTHER", "label": f"Other PNC clients ({len(rest):,})", "kind": "other",
                          "x": x, "y": y, "amt": amt,
                          "hover": f"{len(rest):,} other PNC clients<br>pay it ${rp / 1e3:,.0f}K · receive "
                                   f"${rr / 1e3:,.0f}K"})
    for cid, pa, npd, ra, nrc in rows:
        if pa > 0:
            edges.append({"src": cid, "dst": "CPTY", "flow": "paid_to_it", "amt": pa, "n_txn": npd})
        if ra > 0:
            edges.append({"src": "CPTY", "dst": cid, "flow": "received_from_it", "amt": ra, "n_txn": nrc})
    return pd.DataFrame(nodes, columns=NODE_COLS), pd.DataFrame(edges, columns=EDGE_COLS)


def graph_accounts(AE: pd.DataFrame, ACC: pd.DataFrame, CL: pd.DataFrame, top_n: int, top_acc: int):
    """Accounts on an inner ring grouped by bank, banks in the middle, the largest TM clients outside, each client
    placed near the accounts it deals with. Other accounts and other PNC clients are one node each."""
    AE = _num(AE.copy(), ["amt", "n_txn"])
    info = CL.set_index("mdm_id")
    AE["is_tm"] = AE.mdm_id.map(info.is_tm).fillna(False).astype(bool)
    acct_bank = (ACC.sort_values("amt").drop_duplicates("cpty_id", keep="last").set_index("cpty_id").bank.to_dict()
                 if len(ACC) else {})
    acct_names = ACC.groupby("cpty_id").cpty_name.agg(lambda s: "; ".join(sorted(set(s))[:3])).to_dict() if len(ACC) else {}
    by_acct = AE.groupby("cpty_id").amt.sum().sort_values(ascending=False)
    keep_acct = list(by_acct.index[:top_acc])
    AE["acct"] = np.where(AE.cpty_id.isin(keep_acct), AE.cpty_id, "OTHER_ACCTS")
    by_cli = AE[AE.is_tm].groupby("mdm_id").amt.sum().sort_values(ascending=False)
    keep_cli = list(by_cli.index[:top_n])
    AE["cli"] = np.where(AE.mdm_id.isin(keep_cli), AE.mdm_id, "OTHER")
    E = AE.groupby(["cli", "acct", "flow_dir"], as_index=False).agg(amt=("amt", "sum"), n_txn=("n_txn", "sum"))

    accts = [a for a in keep_acct] + (["OTHER_ACCTS"] if (AE.acct == "OTHER_ACCTS").any() else [])
    banks_of = {a: acct_bank.get(a, "(unknown)") for a in keep_acct}
    banks_of["OTHER_ACCTS"] = "various"
    accts.sort(key=lambda a: (banks_of[a] == "various", banks_of[a], -by_acct.get(a, 0)))
    nodes, angle = [], {}
    for i, a in enumerate(accts):
        ang = 90 - 360 * (i + 0.5) / len(accts)
        angle[a] = ang
        x, y = 0.56 * math.cos(math.radians(ang)), 0.56 * math.sin(math.radians(ang))
        amt = by_acct.get(a, AE[AE.acct == a].amt.sum())
        if a == "OTHER_ACCTS":
            n_other = len(by_acct) - len(keep_acct)
            lab, hov = f"{n_other:,} other accounts", f"{n_other:,} other accounts<br>${amt / 1e3:,.0f}K"
        else:
            lab = mask_account(a)
            hov = (f"account {mask_account(a)}<br>bank {banks_of[a]}<br>names: {acct_names.get(a, '')}"
                   f"<br>${amt / 1e3:,.0f}K with PNC clients")
        nodes.append({"id": a, "label": lab, "kind": "account", "x": x, "y": y, "amt": amt, "hover": hov})
    bank_list = list(dict.fromkeys(banks_of[a] for a in accts if banks_of[a] != "various"))
    bank_edges = []
    for j, b in enumerate(bank_list):
        members = [a for a in accts if banks_of[a] == b]
        ma = sum(angle[a] for a in members) / len(members)
        r = 0.0 if len(bank_list) == 1 else 0.24
        nodes.append({"id": f"BANK::{b}", "label": b, "kind": "bank", "x": r * math.cos(math.radians(ma)),
                      "y": r * math.sin(math.radians(ma)), "amt": float(by_acct.reindex(members).fillna(0).sum()),
                      "hover": f"{b}<br>{len(members)} of the accounts shown"})
        bank_edges += [{"src": f"BANK::{b}", "dst": a, "flow": "holds", "amt": 0.0, "n_txn": 0.0} for a in members]
    # clients: preferred angle = money-weighted circular mean of their accounts' angles, then spread evenly
    pref = {}
    for c, d in E.groupby("cli"):
        w = d.groupby("acct").amt.sum()
        sx = sum(v * math.cos(math.radians(angle[a])) for a, v in w.items())
        sy = sum(v * math.sin(math.radians(angle[a])) for a, v in w.items())
        pref[c] = math.degrees(math.atan2(sy, sx))
    order = sorted(pref, key=lambda c: (90 - pref[c]) % 360)
    for i, c in enumerate(order):
        ang = 90 - 360 * (i + 0.5) / len(order)
        x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        d = E[E.cli == c]
        pa, ra = d[d.flow_dir == "paid_to_it"].amt.sum(), d[d.flow_dir == "received_from_it"].amt.sum()
        if c == "OTHER":
            n_other = AE[AE.cli == "OTHER"].mdm_id.nunique()
            nodes.append({"id": c, "label": f"Other PNC clients ({n_other:,})", "kind": "other", "x": x, "y": y,
                          "amt": pa + ra, "hover": f"{n_other:,} other PNC clients<br>pay it ${pa / 1e3:,.0f}K · "
                                                   f"receive ${ra / 1e3:,.0f}K"})
        else:
            r = info.loc[c] if c in info.index else None
            nm = str(r.customer_name) if r is not None else c
            if r is not None:
                r = r.copy()
                r["paid_to_it"], r["received_from_it"] = pa, ra
            hov = _client_hover(r) if r is not None else f"{c}<br>pays it ${pa / 1e3:,.0f}K · receives ${ra / 1e3:,.0f}K"
            nodes.append({"id": c, "label": nm, "kind": "client", "x": x, "y": y, "amt": pa + ra, "hover": hov})
    edges = [{"src": r.cli, "dst": r.acct, "flow": "paid_to_it", "amt": r.amt, "n_txn": r.n_txn}
             if r.flow_dir == "paid_to_it" else
             {"src": r.acct, "dst": r.cli, "flow": "received_from_it", "amt": r.amt, "n_txn": r.n_txn}
             for _, r in E.iterrows()]
    return pd.DataFrame(nodes, columns=NODE_COLS), pd.DataFrame(edges + bank_edges, columns=EDGE_COLS)


# ================================================================== audit
def log_event(user: str, action: str, detail: str, rows: int = 0) -> None:
    """Append who did what (counterparties opened, exports). Never breaks the app."""
    try:
        os.makedirs(os.path.dirname(AUDIT_PATH) or ".", exist_ok=True)
        new = not os.path.exists(AUDIT_PATH)
        with open(AUDIT_PATH, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["at", "user", "action", "detail", "rows"])
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), user, action, detail, rows])
    except Exception:
        pass
