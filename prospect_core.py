"""
Prospect Finder: data access and ranking, with no Streamlit inside, so the dev team can wrap it in FastAPI as is.

Prospects are outside companies (counterparty entities, ent_id = ent_l2) that PNC clients pay or are paid by.
The app runs four kinds of Impala query, all built here:
  universe_sql         one row per company over the window: size, reach, growth, bank, name match to PNC
  segment_sql          per company: TM clients in a chosen industry and/or state (on demand)
  profile_trend_sql    one company, month by month
  profile_clients_sql  one company's TM clients (warm introductions)
Ranking runs in pandas on the cached universe: every measure becomes a percentile within the filtered set and a
lens is a set of weights over those percentiles, so a score always means "how high it sits on what this lens
values, among the companies on screen".

No free text from the UI reaches SQL: ids, NAICS codes and states are checked against fixed patterns or lists.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ================================================================== configuration (check on the first run)
DB = "bdahd01p_dlcdi1_cdi_tm"
T_CPTY_MET = f"{DB}.pkg_cpty_metrics_monthly"   # entity x month; direction is the entity's view (_in = receives from PNC)
T_ENT_DIM = f"{DB}.pkg_cpty_entity_dim"         # pnc_match_n / pnc_tm_match_n = same cleaned name as a PNC organisation
T_PAIR = f"{DB}.pkg_cust_pair_month_v3"         # customer x dir x scope x other; dir is the CUSTOMER's view
T_CDIM = f"{DB}.pkg_cust_dim"                   # names, power ids, is_tm, naics_cd / naics2 / naics_status
TPPP_TABLES = [f"{DB}.pkg_tppp_prospects_v8", f"{DB}.pkg_tppp_prospects_v7", f"{DB}.pkg_tppp_prospects_v6"]
# Address of PNC clients (client footprint). Column names come from the 2026-08 address profile: verify.
ADDR = {"table": "dsihd01p_dsi.neo4j_address", "id": "mdm_id", "state": "state_or_province",
        "snap_col": "load_dt", "snapshot": None}      # snapshot None = latest load_dt within ADDR_LOOKBACK_DAYS
ADDR_LOOKBACK_DAYS = 45
IMPALA = {"dsn": "DSN=bdpimp04-impala;", "pool": "root.CIB-AMG_Impala", "conn_options": {"SocketTimeout": 0}}

FIRST_MONTH, LAST_MONTH = "2024-01", "2026-08"
DEFAULT_END, DEFAULT_LEN = "2026-06", 6     # Jul-Aug 2026 carry a dollar jump in the source; amounts stable to June
WINDOW_LENGTHS = [3, 6, 12]
MIN_CLIENT_CHOICES = [3, 5, 10, 25, 50]     # universe floor: average PNC clients per month, either direction
DEFAULT_MIN_CLIENTS = 10
CLASSES = ("business", "unknown")           # typer v2 name classes kept as prospects
MAX_ROWS = 300_000                          # safety cap on the universe pull
AUDIT_PATH = os.environ.get("PROSPECT_AUDIT", "audit/prospect_finder_audit.csv")

# ================================================================== reference lists
SECTORS = {  # NAICS 2-digit sectors; ranges are several 2-digit codes
    "11": "Agriculture", "21": "Mining, oil and gas", "22": "Utilities", "23": "Construction",
    "31-33": "Manufacturing", "42": "Wholesale trade", "44-45": "Retail trade", "48-49": "Transportation, warehousing",
    "51": "Information", "52": "Finance and insurance", "53": "Real estate, rental, leasing",
    "54": "Professional, scientific, technical", "55": "Management of companies", "56": "Administrative, support, waste",
    "61": "Educational services", "62": "Health care, social assistance", "71": "Arts, entertainment, recreation",
    "72": "Accommodation, food services", "81": "Other services", "92": "Public administration"}
NAICS2_SECTOR = {c: f"{k} {v}" for k, v in SECTORS.items()
                 for c in (map(str, range(int(k[:2]), int(k[-2:]) + 1)) if "-" in k else [k])}
STATES = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY",
          "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
          "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"]

# What the name says (a hint, not a classification); first match wins for the label, all matches are filterable.
NAME_TYPES = {
    "Payroll and HR": r"\b(?:PAYROLL|PAYCHEX|ADP|GUSTO|PAYCOM|PAYLOCITY|PAYCOR|ISOLVED|TRINET|INSPERITY|JUSTWORKS|PEO|HR)\b",
    "Payments and processing": r"\b(?:PAYMENTS?|PROCESSING|PROCESSORS?|MERCHANT SERVICES|STRIPE|PAYPAL|FISERV|WORLDPAY|"
                               r"ELAVON|TSYS|SQUARE|SHIFT4|ADYEN|REPAY|DWOLLA|BILL COM|AVIDXCHANGE|PAYMENTECH)\b",
    "Staffing": r"\b(?:STAFFING|PERSONNEL|TEMPS?|WORKFORCE|EMPLOYMENT|RECRUIT\w*)\b",
    "Insurance and benefits": r"\b(?:INSURANCE|ASSURANCE|INS CO|CASUALTY|UNDERWRIT\w*|BENEFITS?|ANNUITY|REINSURANCE)\b",
    "Real estate and property": r"\b(?:PROPERTY|PROPERTIES|REALTY|REAL ESTATE|APARTMENTS?|HOA|CONDOMINIUM|LEASING|RENTALS?)\b",
    "Freight and logistics": r"\b(?:FREIGHT|LOGISTICS|TRUCKING|TRANSPORT\w*|CARRIERS?|SHIPPING|EXPRESS|DRAYAGE|MOVING)\b",
    "Health care": r"\b(?:HEALTH\w*|MEDICAL|HOSPITAL|CLINIC|DENTAL|PHARMACY|PHYSICIANS?|ORTHO\w*|THERAPY|SURGERY|LABS?)\b",
    "Construction and trades": r"\b(?:CONSTRUCTION|BUILDERS?|CONTRACTING|CONTRACTORS?|ROOFING|PLUMBING|ELECTRICAL|"
                               r"MECHANICAL|PAVING|CONCRETE|EXCAVATING|HVAC)\b",
    "Food and distribution": r"\b(?:FOODS?|PRODUCE|MEATS?|BEVERAGES?|DISTRIBUT\w*|WHOLESALE|SUPPLY|SUPPLIES)\b",
    "Software and IT": r"\b(?:SOFTWARE|TECHNOLOG\w*|SYSTEMS|DIGITAL|CLOUD|DATA|SAAS|NETWORKS?|COMPUTER\w*)\b",
    "Legal and accounting": r"\b(?:LAW|LEGAL|ATTORNEYS?|CPAS?|ACCOUNTING|ACCOUNTANTS|AUDIT\w*)\b",
    "Lending and leasing": r"\b(?:CAPITAL|FINANCIAL|FINANCE|FUNDING|LENDING|LOANS?|CREDIT|MORTGAGE|LEASE)\b",
    "Utilities and telecom": r"\b(?:ENERGY|ELECTRIC|GAS|WATER|POWER|UTILIT\w*|TELECOM\w*|WIRELESS|COMMUNICATIONS|BROADBAND)\b",
    "Education": r"\b(?:SCHOOLS?|UNIVERSITY|COLLEGE|ACADEMY|EDUCATION|LEARNING)\b",
}
KNOWN_PROC_RE = (r"\b(?:ADP|PAYCHEX|PAYCOM|PAYLOCITY|DAYFORCE|CERIDIAN|GUSTO|PAYCOR|ISOLVED|TRINET|JUSTWORKS|INSPERITY|"
                 r"RIPPLING|STRIPE|PAYPAL|BRAINTREE|WORLDPAY|FISERV|FIRST DATA|GLOBAL PAYMENTS|HEARTLAND PAYMENT|TSYS|"
                 r"ELAVON|PAYMENTECH|AUTHORIZE NET|SHIFT4|NUVEI|ADYEN|PAYSAFE|REPAY|DWOLLA|BILL COM|AVIDXCHANGE|"
                 r"CLICKPAY|APPFOLIO|YARDI|PAYLEASE|ZEGO|REALPAGE)\b")   # same seed list as the TPPP notebooks
HIDE_RE = (r"\bPNC\b|\b(?:IRS|US TREASURY|TREASURY|TREAS|SSA|STATE OF|DEPT OF|DEPARTMENT OF|COUNTY OF|CITY OF|"
           r"COMMONWEALTH OF|SALES ?TAX|USE ?TAX|TAXPYMT|DEPT REV|VISA|MASTERCARD|AMERICAN EXPRESS|AMEX|DISCOVER)\b")
GENERIC_TOKENS = {"BNKCD", "BANKCARD", "BANKCD", "BNKCRD", "MERCH", "MERCHANT", "MERCHANTS", "MRCH", "SVC", "SVCS",
                  "SERVICE", "SERVICES", "SETTLE", "SETTLEMENT", "STLMT", "DEP", "DEPOSIT", "EMPLOYEE", "BENEFIT",
                  "BENEFITS", "PAYMENT", "PAYMENTS", "PAY", "CARD", "CARDS", "DEBIT", "CREDIT", "ACH", "TRANSFER",
                  "FEE", "FEES", "DISC", "DISCOUNT", "BATCH", "PAYROLL", "REFUND", "PROCESSING", "DEPOSITS", "CHGBK",
                  "MER", "BK", "BNK", "MRCHNT", "MERCHNT", "BANKCRD", "STLMNT", "SETL", "CRD", "DEPST", "SERV",
                  "SERVS", "SRV", "SRVS", "NSD", "SD", "STL", "STLM", "CHGBCK", "DISCNT", "ACCT", "ACC", "ONLINE",
                  "WEB", "PPD", "CCD", "WIRE", "BILL", "INV", "INVOICE", "TAX", "COLLECTION", "RETURN", "ORIG"}

NAME_TYPE_COL = {k: f"nt_{i}" for i, k in enumerate(NAME_TYPES)}   # one boolean column per name type
RELATIONSHIPS = ["New to bank", "PNC client (same name)", "TM client (same name)", "Unknown"]
ROLES = ["Receives from PNC clients", "Pays PNC clients", "Both ways"]

# ================================================================== Impala access
QUERY_LOG: list[dict] = []
_ID_RE = re.compile(r"^[A-Za-z0-9:._|\-]{1,80}$")


def run_sql(sql: str, label: str = "") -> pd.DataFrame:
    """One Impala query through the in-house dbi module; every call is timed into QUERY_LOG."""
    import dbi
    t0 = time.time()
    df = dbi.db_get_query(sql, **IMPALA)
    df.columns = [str(c).lower().split(".")[-1] for c in df.columns]
    QUERY_LOG.append({"at": dt.datetime.now().strftime("%H:%M:%S"), "query": label,
                      "seconds": round(time.time() - t0, 1), "rows": len(df), "sql": sql})
    del QUERY_LOG[:-40]
    return df


def table_columns(table: str) -> set[str] | None:
    """Lower-case column names, or None when the table cannot be read."""
    try:
        return set(run_sql(f"SELECT * FROM {table} LIMIT 0", f"columns: {table}").columns)
    except Exception:
        return None


def _lit(values) -> str:
    return ", ".join("'" + str(v) + "'" for v in values)


def _check_id(x: str) -> str:
    if not _ID_RE.match(str(x)):
        raise ValueError(f"unexpected id format: {x!r}")
    return str(x)


def month_range(a: str, b: str) -> list[str]:
    y, m = int(a[:4]), int(a[5:])
    out = []
    while f"{y:04d}-{m:02d}" <= b:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def window(end: str, n: int) -> tuple[str, ...]:
    return tuple(month_range(FIRST_MONTH, end)[-n:])


# ================================================================== universe
def _base_where(win) -> str:
    return (f"month BETWEEN '{win[0]}' AND '{win[-1]}' AND eligible "
            f"AND name_class IN ({_lit(CLASSES)})")


def _universe_ids_sql(win, min_clients) -> str:
    return (f"SELECT ent_id FROM {T_CPTY_MET} WHERE {_base_where(win)} GROUP BY ent_id "
            f"HAVING SUM(CAST(n_cust_all AS DOUBLE)) / {len(win)} >= {float(min_clients)}")


def universe_sql(win, min_clients, cols: set[str], dim_cols: set[str] | None, max_rows: int = MAX_ROWS) -> str:
    """Window averages per company (absent months count as zero), its latest name and bank, PNC name match."""
    need = {"ent_id", "month", "ent_name", "name_class", "fi_dom", "eligible", "amt_in", "amt_out", "n_cust_in",
            "n_cust_out", "n_cust_all"}
    miss = need - cols
    if miss:
        raise ValueError(f"{T_CPTY_MET} lacks {sorted(miss)}")
    n, half = len(win), len(win) // 2
    split = win[half]
    avg = lambda c, a: f"SUM(CAST({c} AS DOUBLE)) / {n} AS {a}"
    sums = [avg("amt_in", "amt_in"), avg("amt_out", "amt_out"), avg("n_cust_in", "clients_in"),
            avg("n_cust_out", "clients_out"), avg("n_cust_all", "clients"),
            f"SUM(CASE WHEN month >= '{split}' THEN CAST(n_cust_all AS DOUBLE) ELSE 0 END) / {n - half} AS clients_recent",
            f"SUM(CASE WHEN month < '{split}' THEN CAST(n_cust_all AS DOUBLE) ELSE 0 END) / {half} AS clients_early"]
    if {"n_cust_tm_in", "n_cust_tm_out"} <= cols:
        sums += [avg("n_cust_tm_in", "tm_in"), avg("n_cust_tm_out", "tm_out")]
    if "cpty_type" in cols:
        sums.append("AVG(CASE WHEN cpty_type = 'NON_PNC_ACH_ORIGINATOR' THEN 1.0 ELSE 0.0 END) AS ach_orig")
    last = ["t.ent_name", "t.name_class", "t.fi_dom"]
    last += ["CAST(t.n_fi AS DOUBLE) AS n_fi"] if "n_fi" in cols else []
    last += ["CAST(t.hub_tier AS INT) AS hub_tier"] if "hub_tier" in cols else []
    dim = [f"d.{c}" for c in ("pnc_match_n", "pnc_tm_match_n") if dim_cols and c in dim_cols]
    a_cols = ["months_active"] + [s.rsplit(" AS ", 1)[1] for s in sums]
    l_cols = [c.rsplit(" AS ", 1)[-1].replace("t.", "") for c in last]
    sel = ", ".join(["a.ent_id"] + [f"a.{c}" for c in a_cols] + [f"l.{c}" for c in l_cols] + dim)
    dim_join = f"LEFT JOIN {T_ENT_DIM} d ON a.ent_id = d.ent_id" if dim else ""
    return f"""
WITH a AS (
  SELECT ent_id, COUNT(*) AS months_active, MAX(month) AS last_month,
         {', '.join(sums)}
  FROM {T_CPTY_MET}
  WHERE {_base_where(win)}
  GROUP BY ent_id
  HAVING SUM(CAST(n_cust_all AS DOUBLE)) / {n} >= {float(min_clients)}
),
l AS (
  SELECT t.ent_id, {', '.join(last)}
  FROM {T_CPTY_MET} t JOIN a ON t.ent_id = a.ent_id AND t.month = a.last_month
  WHERE t.month BETWEEN '{win[0]}' AND '{win[-1]}'
)
SELECT {sel}
FROM a JOIN l ON a.ent_id = l.ent_id
{dim_join}
ORDER BY a.clients DESC
LIMIT {int(max_rows)}"""


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def name_types(names: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """First matching name type per company, plus one boolean column per type."""
    hits = pd.DataFrame({k: names.str.contains(p, regex=True) for k, p in NAME_TYPES.items()}, index=names.index)
    first = hits.idxmax(axis=1).where(hits.any(axis=1), "")
    return first, hits


def is_generic(name: str) -> bool:
    toks = re.findall(r"[A-Z0-9]+", name)
    rest = [t for t in toks if t not in GENERIC_TOKENS and not re.fullmatch(r"[A-Z]*\d+[A-Z\d]*", t) and len(t) > 2]
    return len(rest) == 0


def prepare_universe(df: pd.DataFrame, n_months: int) -> pd.DataFrame:
    """Typed columns, derived measures and flags; run once per cached universe."""
    df = df.drop_duplicates("ent_id").reset_index(drop=True)
    df = _num(df, ["months_active", "amt_in", "amt_out", "clients_in", "clients_out", "clients", "clients_recent",
                   "clients_early", "tm_in", "tm_out", "ach_orig", "n_fi", "hub_tier", "pnc_match_n", "pnc_tm_match_n"])
    df["ent_name"] = df.ent_name.fillna("").astype(str).str.upper().str.strip()
    df["amt"] = df.amt_in.fillna(0) + df.amt_out.fillna(0)
    df["in_share"] = np.where(df.amt > 0, df.amt_in.fillna(0) / df.amt.where(df.amt > 0, 1), np.nan)
    df["role"] = np.select([df.in_share >= 0.8, df.in_share <= 0.2], ROLES[:2], ROLES[2])
    df["growth"] = (df.clients_recent + 1) / (df.clients_early + 1) - 1
    df["steady"] = df.months_active / n_months
    if {"tm_in", "tm_out"} <= set(df.columns):
        df["tm_all"] = df.tm_in.fillna(0) + df.tm_out.fillna(0)
    if "pnc_match_n" in df:
        tm_match = df.get("pnc_tm_match_n", pd.Series(0, index=df.index)).fillna(0) > 0
        df["relationship"] = np.select([tm_match, df.pnc_match_n.fillna(0) > 0],
                                       ["TM client (same name)", "PNC client (same name)"], "New to bank")
    else:
        df["relationship"] = "Unknown"
    df["name_type"], hits = name_types(df.ent_name)
    for k, c in NAME_TYPE_COL.items():
        df[c] = hits[k].to_numpy()
    df["known_proc"] = df.ent_name.str.contains(KNOWN_PROC_RE, regex=True)
    df["hidden_kind"] = df.ent_name.str.contains(HIDE_RE, regex=True)
    df["generic"] = df.ent_name.map(is_generic)
    return df


# ================================================================== optional joins
def tppp_sql(table: str, cols: set[str]) -> str:
    keep = [c for c in ("score", "known_proc", "reasons", "most_similar_example", "relationship") if c in cols]
    alias = {"score": "tppp_score", "known_proc": "tppp_known_proc", "reasons": "tppp_reasons",
             "most_similar_example": "tppp_similar", "relationship": "tppp_relationship"}
    return f"SELECT ent_id, {', '.join(f'{c} AS {alias[c]}' for c in keep)} FROM {table}"


def latest_addr_snapshot_sql() -> str:
    since = (dt.date.today() - dt.timedelta(days=ADDR_LOOKBACK_DAYS)).isoformat()
    return f"SELECT MAX({ADDR['snap_col']}) AS snap FROM {ADDR['table']} WHERE {ADDR['snap_col']} >= '{since}'"


@dataclass(frozen=True)
class Segment:
    """TM clients a company deals with, by industry (NAICS sector) and/or state of the client's address."""
    sectors: tuple = ()
    states: tuple = ()
    side: str = "sells to"          # sells to = the segment's clients pay it; buys from = it pays them; either
    min_clients: int = 2

    def label(self) -> str:
        what = " + ".join(filter(None, [", ".join(SECTORS[s] for s in self.sectors), ", ".join(self.states)]))
        return f"{self.side} TM clients in {what}"


def segment_sql(win, min_clients, seg: Segment, addr_snapshot: str | None) -> str:
    codes = [c for s in seg.sectors for c in NAICS2_SECTOR if NAICS2_SECTOR[c].startswith(s + " ")]
    states = [s for s in seg.states if s in STATES]
    if not codes and not states:
        raise ValueError("choose at least one industry or state")
    preds = []
    if codes:
        preds.append(f"(c.naics_status = 'valid' AND CAST(c.naics2 AS STRING) IN ({_lit(codes)}))")
    addr_join = ""
    if states:
        if not addr_snapshot or not re.fullmatch(r"[0-9: \-T]+", str(addr_snapshot)):
            raise ValueError(f"no usable address snapshot ({addr_snapshot!r}); set ADDR['snapshot']")
        preds.append(f"ad.st IN ({_lit(states)})")
        addr_join = (f"LEFT JOIN (SELECT CAST({ADDR['id']} AS STRING) AS mdm_id, upper(trim({ADDR['state']})) AS st "
                     f"FROM {ADDR['table']} WHERE {ADDR['snap_col']} = '{addr_snapshot}' "
                     f"AND upper(trim({ADDR['state']})) IN ({_lit(states)})) ad "
                     f"ON CAST(p.cust AS STRING) = ad.mdm_id")
    dirs = {"sells to": ["out"], "buys from": ["in"], "either way": ["out", "in"]}[seg.side]
    n = len(win)
    return f"""
WITH u AS ({_universe_ids_sql(win, min_clients)}),
p AS (
  SELECT p.other AS ent_id, p.cust,
         MAX(CASE WHEN {' AND '.join(preds)} THEN 1 ELSE 0 END) AS in_seg,
         SUM(CAST(p.amount AS DOUBLE)) AS amt
  FROM {T_PAIR} p
  JOIN u ON p.other = u.ent_id
  JOIN {T_CDIM} c ON p.cust = c.mdm_id
  {addr_join}
  WHERE p.month BETWEEN '{win[0]}' AND '{win[-1]}' AND p.scope = 'ext' AND p.dir IN ({_lit(dirs)}) AND c.is_tm
  GROUP BY p.other, p.cust
)
SELECT ent_id, COUNT(*) AS seg_base, SUM(in_seg) AS seg_clients,
       SUM(CASE WHEN in_seg = 1 THEN amt ELSE 0 END) / {n} AS seg_amt
FROM p
GROUP BY ent_id
HAVING SUM(in_seg) >= {int(seg.min_clients)}"""


def prepare_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = _num(df.copy(), ["seg_base", "seg_clients", "seg_amt"])
    df["seg_share"] = df.seg_clients / df.seg_base.where(df.seg_base > 0)
    return df


# ================================================================== one company
def profile_trend_sql(ent_id: str, months) -> str:
    return (f"SELECT month, CAST(amt_in AS DOUBLE) AS amt_in, CAST(amt_out AS DOUBLE) AS amt_out, "
            f"n_cust_in, n_cust_out, n_cust_all FROM {T_CPTY_MET} "
            f"WHERE month BETWEEN '{months[0]}' AND '{months[-1]}' AND ent_id = '{_check_id(ent_id)}' ORDER BY month")


def profile_clients_sql(ent_id: str, win, limit: int = 300) -> str:
    return f"""
SELECT p.cust AS mdm_id, c.customer_name, c.customer_pwr_id, c.naics_cd, c.naics_desc,
       CAST(c.naics2 AS STRING) AS naics2, c.naics_status,
       SUM(CASE WHEN p.dir = 'out' THEN CAST(p.amount AS DOUBLE) ELSE 0 END) AS paid_to_it,
       SUM(CASE WHEN p.dir = 'in' THEN CAST(p.amount AS DOUBLE) ELSE 0 END) AS received_from_it,
       COUNT(DISTINCT p.month) AS months
FROM {T_PAIR} p JOIN {T_CDIM} c ON p.cust = c.mdm_id
WHERE p.month BETWEEN '{win[0]}' AND '{win[-1]}' AND p.scope = 'ext' AND p.other = '{_check_id(ent_id)}' AND c.is_tm
GROUP BY p.cust, c.customer_name, c.customer_pwr_id, c.naics_cd, c.naics_desc, c.naics2, c.naics_status
ORDER BY SUM(CAST(p.amount AS DOUBLE)) DESC
LIMIT {int(limit)}"""


# ================================================================== filters
@dataclass
class Filters:
    name_contains: str = ""
    name_types: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    roles: list = field(default_factory=list)
    banks: list = field(default_factory=list)
    exclude_banks: list = field(default_factory=list)
    min_banks: int = 0
    min_clients: float = 0.0
    min_tm: float = 0.0
    min_amt_k: float = 0.0
    min_months: int = 0
    hide_pnc_gov_card: bool = True
    hide_generic: bool = True
    known_proc_only: bool = False
    tppp_scored_only: bool = False


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if f.name_contains.strip():
        m &= df.ent_name.str.contains(re.escape(f.name_contains.strip().upper()), regex=True)
    if f.name_types:
        m &= df[[NAME_TYPE_COL[t] for t in f.name_types]].any(axis=1)
    if f.relationships:
        m &= df.relationship.isin(f.relationships)
    if f.roles:
        m &= df.role.isin(f.roles)
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
    if f.hide_pnc_gov_card:
        m &= ~df.hidden_kind
    if f.hide_generic:
        m &= ~df.generic
    if f.known_proc_only:
        m &= df.known_proc
    if f.tppp_scored_only and "tppp_score" in df:
        m &= df.tppp_score.notna()
    return m


# ================================================================== lenses and scores
MEASURES = {  # key: (label in the UI, column)
    "amt": ("Money with PNC clients", "amt"),
    "clients": ("PNC clients", "clients"),
    "tm_clients": ("TM clients", "tm_all"),
    "growth": ("Growth in PNC clients", "growth"),
    "steady": ("Months active", "steady"),
    "seg_clients": ("TM clients in segment", "seg_clients"),
    "seg_share": ("Share of its TM clients in segment", "seg_share"),
    "tppp": ("TPPP model score", "tppp_score"),
}
LENSES = {
    "Largest wallet": {"desc": "Most money moving with PNC clients, steady over the window.",
                       "w": {"amt": 3, "clients": 1, "steady": 1}},
    "Widest reach": {"desc": "Deals with the most PNC and TM clients: the most warm introductions.",
                     "w": {"clients": 2, "tm_clients": 2, "steady": 1}},
    "Rising": {"desc": "PNC clients growing fastest from the first to the second half of the window.",
               "w": {"growth": 3, "clients": 1}},
    "Segment fit": {"desc": "Most TM clients in the chosen industry or state (set it under Who they deal with).",
                    "w": {"seg_clients": 2, "seg_share": 2, "amt": 1}},
    "TPPP-like processors": {"desc": "Ranked by the TPPP prospect model (notebook v8); companies it did not score "
                                     "are hidden.", "w": {"tppp": 4, "clients": 1}},
}


def usable_measures(df: pd.DataFrame) -> list[str]:
    return [k for k, (_, c) in MEASURES.items() if c in df and df[c].notna().any()]


def score(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Weighted mean of within-set percentiles (0-100). Missing values count as the bottom."""
    out = df.copy()
    use = {k: float(w) for k, w in weights.items() if w and k in usable_measures(df)}
    if not use or not len(df):
        out["score"], out["_pct"] = np.nan, None
        return out
    pct = pd.DataFrame({k: df[MEASURES[k][1]].rank(pct=True, method="average").fillna(0.0) for k in use})
    out["score"] = 100 * (pct * pd.Series(use)).sum(axis=1) / sum(use.values())
    for k in use:
        out[f"_pct_{k}"] = pct[k]
    return out.sort_values(["score", "amt"], ascending=False)


def _reason(k: str, row: pd.Series) -> str:
    top = f" (top {max(1, round(100 * (1 - row[f'_pct_{k}'])))}%)"
    if k == "amt":
        return f"${row.amt / 1e3:,.0f}K a month with PNC clients" + top
    if k == "clients":
        return f"{row.clients:,.0f} PNC clients a month" + top
    if k == "tm_clients":
        return f"{row.tm_all:,.0f} TM clients a month" + top
    if k == "growth":
        return f"PNC clients {row.growth:+.0%}, second half of the window vs first" + top
    if k == "steady":
        return f"active {row.months_active:.0f} of {round(row.months_active / row.steady)} months"
    if k == "seg_clients":
        return f"{row.seg_clients:,.0f} TM clients in the segment" + top
    if k == "seg_share":
        return f"{row.seg_share:.0%} of its TM clients in the segment" + top
    return f"TPPP model score {row.tppp_score:.0f}"


def reasons(row: pd.Series, weights: dict, top: int = 3) -> str:
    """The measures that lift this company most (weight x percentile), in words."""
    items = [(w * row[f"_pct_{k}"], k) for k, w in weights.items()
             if w and f"_pct_{k}" in row and pd.notna(row[f"_pct_{k}"]) and pd.notna(row.get(MEASURES[k][1]))]
    return " · ".join(_reason(k, row) for _, k in sorted(items, reverse=True)[:top])


# ================================================================== audit
def log_event(user: str, action: str, detail: str, rows: int = 0) -> None:
    """Append who did what (views of lists, exports). Never breaks the app."""
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
