"""
Data Center Network Explorer: workbook reader, matching to the payment data, queries and graph building.
No Streamlit inside, so the dev team can wrap it in FastAPI as is.

  read_workbook        the relationship tabs of the data center workbook -> one row per workbook line
  match_workbook       workbook Power IDs -> PNC customers; workbook names -> outside counterparties (name key)
  load_ego             payments around a selection: who pays it, who it pays (pair table), names, credit snapshot
  load_streams         regular payment streams of the same pairs (stored periodicity)
  build_graph          nodes, edges and the full neighbour table for the network picture
  customer_* / cpty_*  queries behind the customer page and the counterparty page

A selection ("focus") is a set of PNC customers (mdm_id) and / or outside counterparties (ent_id). The payment data
sees a company three ways: through its PNC account (ids), as an outside account matched by name, or both.

Column names that are reserved words in Impala (month, dir, scope) are always written in backticks. No free text
from the UI reaches SQL: ids are checked against a fixed pattern and search words are reduced to letters and digits.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import math
import os
import pickle
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("dc_network")

# ================================================================== configuration (check on the first run)
DB = "bdahd01p_dlcdi1_cdi_tm"
T_CDIM = f"{DB}.pkg_cust_dim"                    # mdm_id, power ids, name, party type, NAICS
T_PAIR = f"{DB}.pkg_cust_pair_month_v4"          # customer x dir x scope x other; dir is the CUSTOMER's view
T_CUST_MET = f"{DB}.pkg_cust_metrics_monthly_v4"  # customer x month payment profile
T_CPTY_MET = f"{DB}.pkg_cpty_metrics_monthly"    # counterparty entity x month; _in = it receives from PNC customers
T_MAP = f"{DB}.pkg_cpty_entity_map_v2"           # account id (cpty_id) -> ent_l2
T_CPTY_MONTH = f"{DB}.pkg_cpty_month"            # account id x month: cpty_name, fi, amount, n_txn
T_FI_ROUTING = f"{DB}.pkg_fi_routing"            # routing number -> fi_canonical
T_FI_NAMES = f"{DB}.pkg_fi_name_map"             # fi_raw -> fi_canonical
T_TM = f"{DB}.pkg_tm_cust_month"                 # power id x month: DHE, PD, past due, watchlist ...
T_DEP = f"{DB}.pkg_dep_cust_month"               # power id x month: deposit balances (optional)
T_REG = f"{DB}.pkg_cust_reg_stream"              # regular payment streams: customer x dir x side x other x month
IMPALA = {"dsn": "DSN=bdpimp04-impala;", "pool": "root.CIB-AMG_Impala", "conn_options": {"SocketTimeout": 0}}

PERIOD_LENGTHS = [3, 6, 12, 24]
DEFAULT_LEN = 12
NAME_MONTHS = 12            # counterparty names are searched in the last N months of counterparty metrics
REG_LOOKBACK = 12           # months of stored streams read back from the end of the period (lower it if this read is slow)
MAX_IDS = 4000              # ids per IN list; longer lists are split into several queries
MAX_FOCUS_IDS = 4000        # customers / counterparties behind one selection (larger selections are cut, with a note)
MAX_EGO_ROWS = 250_000      # safety cap on pairs pulled for one selection (largest by dollars kept)
NAME_CAP = 12_000           # neighbours looked up by name (largest by dollars); the rest show their id
HUB_TIER = 2                # counterparties at this hub tier or above count as hubs (P99.9 of reach or dollars)
PIN_CAP = 60                # workbook parties always shown, beyond the top-N neighbours
WB_ONLY_CAP = 25            # workbook parties with no payments drawn as hollow nodes
CACHE_DIR = os.environ.get("DC_APP_CACHE", "cache")   # name-match results per workbook (holds client names)

DEFAULT_DC_IDS = {"DCM9999999", "DCC9999999"}   # the workbook's "Default DC" bucket
REAL = "Power ID"
LEGAL = ["INC", "INCORPORATED", "LLC", "LP", "LLP", "LTD", "LIMITED", "CORP", "CORPORATION", "CO", "COMPANY", "PLC",
         "NA", "SA", "GMBH"]

STATES = ["on schedule", "late or overdue", "stopped", "casual"]
CADENCE_GAP = {"daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30, "quarterly": 91}

_SAFE_ID = re.compile(r"^[^'\"\\;\x00-\x1f]{1,200}$")
_MONTH = re.compile(r"^\d{4}-\d{2}$")


# ================================================================== Impala access
def run_sql(sql: str, tag: str = "") -> pd.DataFrame:
    """One read-only Impala query through dbi; column names come back lower case."""
    import dbi
    t0 = time.time()
    df = dbi.db_get_query(sql, **IMPALA)
    df.columns = [str(c).lower().split(".")[-1] for c in df.columns]
    log.info("sql %-22s %6.1fs %8d rows", tag, time.time() - t0, len(df))
    return df


def table_columns(table: str):
    """Column names of a table, or None when it cannot be read."""
    try:
        d = run_sql(f"DESCRIBE {table}", "describe")
    except Exception as e:  # noqa: BLE001
        log.warning("describe %s failed: %s", table, e)
        return None
    names = [str(v).strip().lower() for v in d.iloc[:, 0].tolist()]
    return {n for n in names if n and not n.startswith("#")}


def table_months(table: str) -> list:
    """Months present in a table partitioned by month (row counts come from file footers, so this is quick)."""
    d = run_sql(f"SELECT `month` AS m, COUNT(*) AS n FROM {table} GROUP BY `month`", "months")
    return sorted(str(m) for m in d.m.dropna().tolist() if _MONTH.match(str(m)))


def safe_ids(values) -> list:
    """Distinct ids that are safe to place in a SQL string literal (others are dropped and logged)."""
    out, bad = [], 0
    for v in dict.fromkeys(str(x) for x in values if x is not None and x == x):
        if _SAFE_ID.match(v):
            out.append(v)
        else:
            bad += 1
    if bad:
        log.warning("%d ids left out of a query (unexpected characters)", bad)
    return out


def _in(values) -> str:
    return ", ".join("'" + v + "'" for v in values)


def _chunks(values, n=MAX_IDS):
    values = list(values)
    return [values[i:i + n] for i in range(0, len(values), n)]


def _m(month: str) -> str:
    if not _MONTH.match(str(month)):
        raise ValueError(f"bad month {month!r}")
    return str(month)


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _str(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: None if v is None or v != v else str(v)).astype(object)
    return df


def _pick(cols, wanted) -> list:
    """The wanted columns that exist."""
    return [c for c in wanted if cols and c in cols]


# ================================================================== months
def add_months(month: str, k: int) -> str:
    y, m = int(month[:4]), int(month[5:7])
    i = y * 12 + (m - 1) + k
    return f"{i // 12}-{i % 12 + 1:02d}"


def month_range(first: str, last: str) -> list:
    out, m = [], first
    while m <= last:
        out.append(m)
        m = add_months(m, 1)
    return out


def period(end: str, n: int) -> tuple:
    """(first month, last month) of an n-month period ending at `end`."""
    return add_months(end, -(n - 1)), end


# ================================================================== workbook
def _txt(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if np.isnan(v):
            return None
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()[:10]
    s = str(v).strip()
    return s if s else None


def _norm(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else re.sub(r"\s+", " ", str(v)).strip().lower()


def clean_id(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    if not s:
        return None
    return s.zfill(10) if (s.isdigit() and len(s) < 10) else s


def id_kind(s) -> str:
    if not isinstance(s, str):
        return "blank"
    if s in DEFAULT_DC_IDS:
        return "Default DC"
    if re.fullmatch(r"\d{10}", s):
        return REAL
    for p in ("DCM", "DCC", "XX"):
        if s.startswith(p):
            return p
    return "other"


def parse_amt(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = re.sub(r"[$,\s]", "", str(v))
    if s in ("", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


_LEGAL_RX = " (?:" + "|".join(LEGAL) + ")(?= )"


def name_key(v):
    """Company-name key: upper case, brackets / punctuation / legal words removed, no spaces."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"\(.*?\)", " ", str(v).upper())
    s = s.replace(".", "").replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = " " + re.sub(r" +", " ", s) + " "
    s = re.sub(_LEGAL_RX, "", s).replace(" ", "")
    return s if s else None


def sql_name_key(col: str) -> str:
    """The same key as name_key, as an Impala expression (no look-ahead: spaces are doubled before legal words go)."""
    s = f"upper({col})"
    s = f"regexp_replace({s}, '[(][^)]*[)]', ' ')"
    s = f"regexp_replace({s}, '[.]', '')"
    s = f"regexp_replace({s}, '&', ' AND ')"
    s = f"regexp_replace({s}, '[^A-Z0-9 ]', ' ')"
    s = f"concat(' ', regexp_replace({s}, ' +', ' '), ' ')"
    s = f"regexp_replace({s}, ' ', '  ')"
    s = f"regexp_replace({s}, ' ({'|'.join(LEGAL)}) ', '')"
    return f"regexp_replace({s}, ' ', '')"


_FIELD_RULES = [("master_id", lambda h: "master power" in h),
                ("master_name", lambda h: "master name" in h),
                ("dc_id", lambda h: h.startswith("datacenter customer power")),
                ("dc_name", lambda h: h.startswith("datacenter customer name")),
                ("so_id", lambda h: h.startswith("second order power")),
                ("so_name", lambda h: h.startswith("second order customer name")),
                ("so_type", lambda h: h.startswith("second order relationship type")),
                ("so_amount", lambda h: h.startswith("second order loan amount"))]


def _find_header(df: pd.DataFrame, keys, max_rows: int = 15):
    for i in range(min(max_rows, len(df))):
        cells = [_norm(v) for v in df.iloc[i].tolist()]
        if all(any(k in c for c in cells) for k in keys):
            return i
    return None


def read_workbook(src) -> pd.DataFrame:
    """Every relationship tab of the workbook (a path or the file's bytes), one row per workbook line.
    Columns are found by header text, so their order may differ per tab."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, data_only=True)
    frames = []
    for ws in wb.worksheets:
        used = [(r, c) for (r, c), cell in getattr(ws, "_cells", {}).items() if cell.value is not None]
        if not used:
            continue
        nr, nc = max(r for r, _ in used), max(c for _, c in used)
        df = pd.DataFrame([[_txt(v) for v in r] for r in ws.iter_rows(min_row=1, max_row=nr, max_col=nc, values_only=True)])
        hdr = _find_header(df, ["master power", "second order"])
        if hdr is None:
            continue
        heads = [_norm(v) for v in df.iloc[hdr].tolist()]
        body = df.iloc[hdr + 1:].dropna(how="all")
        out = pd.DataFrame({"tab": ws.title, "excel_row": (body.index + 1).astype(int)}, index=body.index)
        for f, test in _FIELD_RULES:
            j = next((j for j, h in enumerate(heads) if test(h)), None)
            out[f] = body.iloc[:, j].astype(object) if j is not None else None
        frames.append(out)
    if not frames:
        raise ValueError("No relationship tab found (looked for 'Master Power ID' and 'Second Order' headers).")
    E = pd.concat(frames, ignore_index=True)
    for c in ("master_id", "dc_id", "so_id"):
        E[c] = E[c].map(clean_id).astype(object)
    for c in ("master_name", "dc_name", "so_name"):
        E[c] = E[c].map(lambda v: v if isinstance(v, str) and v.strip() else None).astype(object)
    E["amount"] = E.so_amount.map(parse_amt)
    E["type"] = E.so_type.map(lambda v: str(v).strip().title() if isinstance(v, str) and v.strip() else "(blank)")
    return E.drop(columns=["so_type", "so_amount"])


ROLES = [("Data center parent", "master_id", "master_name"), ("Data center entity", "dc_id", "dc_name"),
         ("Second-order party", "so_id", "so_name")]


def parties(E: pd.DataFrame) -> pd.DataFrame:
    """One row per workbook id (the 'Default DC' bucket left out): name, id kind, roles, tabs, relationship types."""
    parts = []
    for role, ic, nc in ROLES:
        d = E[[ic, nc, "tab", "type"]].rename(columns={ic: "id", nc: "name"}).dropna(subset=["id"])
        d = d.assign(role=role, type=d.type if role == "Second-order party" else None)
        parts.append(d)
    N = pd.concat(parts, ignore_index=True)
    N["kind"] = N.id.map(id_kind)
    N = N[N.kind != "Default DC"]
    if N.empty:
        return pd.DataFrame(columns=["id", "name", "kind", "roles", "tabs", "types", "rows", "gkey"])
    g = N.groupby("id", sort=False)
    P = pd.DataFrame({
        "name": g.name.agg(lambda s: next((v for v in s if isinstance(v, str)), None)),
        "kind": g.kind.first(),
        "roles": g.role.agg(lambda s: tuple(r for r, _, _ in ROLES if r in set(s))),
        "tabs": g.tab.agg(lambda s: tuple(dict.fromkeys(s))),
        "types": g.type.agg(lambda s: tuple(sorted({v for v in s if isinstance(v, str)}))),
        "rows": g.size()}).reset_index()
    P["name"] = P.name.where(P.name.notna(), P.id)
    # the same company can carry several ids across tabs (a Power ID in one, a made-up id in another): gkey groups them
    P["gkey"] = [name_key(n) or i for n, i in zip(P.name, P.id)]
    return P


# ================================================================== matching the workbook to the payment data
@dataclass
class Match:
    """Workbook ids resolved against the payment data."""
    idmap: pd.DataFrame          # pid, level (customer | master | family), mdm_id, customer_pwr_id, master_pwr_id, customer_name
    ent: pd.DataFrame            # name_key, ent_id, ent_name, amt, n_cust, hub_tier, name_class
    keys: pd.DataFrame           # id, key, src (workbook name | PKG name)
    name_months: tuple           # months of counterparty metrics searched for names

    def __post_init__(self):
        im = self.idmap
        tup = lambda s: tuple(dict.fromkeys(s))   # noqa: E731
        # customer: the Power ID itself; own: plus customers that name it as their parent; family: plus its siblings
        self.mdm_cust = im[im.level == "customer"].groupby("pid").mdm_id.agg(tup).to_dict()
        self.mdm_own = im[im.level != "family"].groupby("pid").mdm_id.agg(tup).to_dict()
        self.mdm_family = im.groupby("pid").mdm_id.agg(tup).to_dict()
        k2e = self.ent.groupby("name_key").ent_id.agg(lambda s: tuple(dict.fromkeys(s))).to_dict()
        self.ent_of = {}
        for i, k in zip(self.keys.id, self.keys.key):
            if k in k2e:
                self.ent_of.setdefault(i, {}).update(dict.fromkeys(k2e[k]))
        self.ent_of = {i: tuple(v) for i, v in self.ent_of.items()}
        self.wb_of_mdm, self.wb_of_ent = {}, {}
        for pid, ms in self.mdm_own.items():
            for m in ms:
                self.wb_of_mdm.setdefault(m, set()).add(pid)
        for i, es in self.ent_of.items():
            for e in es:
                self.wb_of_ent.setdefault(e, set()).add(i)

    def status(self, wb_id: str) -> str:
        a, b = wb_id in self.mdm_own, wb_id in self.ent_of
        if a and b:
            return "PNC customer + outside account"
        if a:
            return "PNC customer"
        if b:
            return "Outside account (name match)"
        return "Not in the payment data"


def power_id_sql(pids, cols) -> str:
    extra = "".join(f", {c}" for c in _pick(cols, ["party_type", "naics_desc", "is_tm"]))
    return f"""
SELECT CAST(mdm_id AS STRING) AS mdm_id, CAST(customer_pwr_id AS STRING) AS customer_pwr_id,
       CAST(master_pwr_id AS STRING) AS master_pwr_id, customer_name{extra}
FROM {T_CDIM}
WHERE customer_pwr_id IN ({_in(pids)}) OR master_pwr_id IN ({_in(pids)})"""


def name_match_sql(keys, first: str, last: str, cols) -> str:
    opt = _pick(cols, ["n_cust_all", "hub_tier", "name_class"])
    amt = " + ".join(f"COALESCE(CAST({c} AS DOUBLE), 0)" for c in _pick(cols, ["amt_in", "amt_out"])) or "0"
    inner = "".join(f", {c}" for c in opt)
    outer = "".join(f", MAX({c}) AS {c}" for c in opt)
    return f"""
SELECT ent_id, ent_name, k AS name_key, SUM(amt) AS amt{outer}
FROM (SELECT CAST(ent_id AS STRING) AS ent_id, ent_name, {sql_name_key('ent_name')} AS k, {amt} AS amt{inner}
      FROM {T_CPTY_MET} WHERE `month` BETWEEN '{_m(first)}' AND '{_m(last)}') t
WHERE k IN ({_in(keys)})
GROUP BY ent_id, ent_name, k"""


def _workbook_keys(E: pd.DataFrame, idmap: pd.DataFrame) -> pd.DataFrame:
    """Name keys for every workbook id: the name typed in the workbook, plus the PKG customer names of the same
    Power ID (whole family for data centers, the customer itself for second-order parties). A company with a
    Power ID may still be paid through an account at another bank, and then the payment data knows it by name."""
    rows = []
    for _, ic, nc in ROLES:
        d = E[[ic, nc]].dropna(subset=[ic]).drop_duplicates()
        rows.append(pd.DataFrame({"id": d[ic].values, "key": d[nc].map(name_key).values, "src": "workbook name"}))
    dc_ids = set(E.dc_id.dropna()) | set(E.master_id.dropna())
    so_ids = set(E.so_id.dropna())
    if len(idmap):
        a = idmap[idmap.pid.isin(dc_ids) & (idmap.level != "family")]
        b = idmap[idmap.pid.isin(so_ids) & (idmap.level == "customer")]
        for d in (a, b):
            rows.append(pd.DataFrame({"id": d.pid.values, "key": d.customer_name.map(name_key).values, "src": "PKG name"}))
    K = pd.concat(rows, ignore_index=True)
    K = K[~K.id.isin(DEFAULT_DC_IDS) & K.key.map(lambda k: isinstance(k, str) and len(k) >= 3)]
    return K.sort_values("src", ascending=False).drop_duplicates(["id", "key"]).reset_index(drop=True)


def match_workbook(E: pd.DataFrame, last_month: str, progress=None, use_cache: bool = True) -> Match:
    """Resolve every workbook id. The name search reads NAME_MONTHS months of counterparty metrics, which is the
    slow step, so its result is kept on disk per workbook content and month."""
    say = progress or (lambda *_: None)
    ids = sorted(set(E.master_id.dropna()) | set(E.dc_id.dropna()) | set(E.so_id.dropna()))
    pids = safe_ids(i for i in ids if id_kind(i) == REAL)
    first = add_months(last_month, -(NAME_MONTHS - 1))
    sig = hashlib.sha1(repr((sorted(zip(E.master_id.astype(str), E.master_name.astype(str), E.dc_id.astype(str),
                                         E.dc_name.astype(str), E.so_id.astype(str), E.so_name.astype(str))),
                             first, last_month, LEGAL, T_CDIM, T_CPTY_MET)).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"match_{sig}.pkl")
    if use_cache and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            return Match(d["idmap"], d["ent"], d["keys"], (first, last_month))
        except Exception as e:  # noqa: BLE001
            log.warning("match cache unreadable (%s): rebuilding", e)

    say(f"Looking up {len(pids):,} Power IDs")
    cdim_cols = table_columns(T_CDIM) or set()
    parts = [run_sql(power_id_sql(ch, cdim_cols), "power ids") for ch in _chunks(pids)]
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["mdm_id", "customer_pwr_id", "master_pwr_id", "customer_name"])
    raw = _str(raw, ["mdm_id", "customer_pwr_id", "master_pwr_id", "customer_name"]).drop_duplicates("mdm_id")
    pset = set(pids)
    # families: customers under the same parent as a workbook customer (the parent itself may not be in the workbook)
    masters = safe_ids(set(raw.loc[raw.customer_pwr_id.isin(pset), "master_pwr_id"].dropna()) - pset)
    if masters:
        say(f"Reading the corporate families of {len(masters):,} parents")
        more = pd.concat([run_sql(power_id_sql(ch, cdim_cols), "families") for ch in _chunks(masters)], ignore_index=True)
        raw = pd.concat([raw, _str(more, ["mdm_id", "customer_pwr_id", "master_pwr_id", "customer_name"])], ignore_index=True).drop_duplicates("mdm_id")
    a = raw[raw.customer_pwr_id.isin(pset)].assign(pid=lambda d: d.customer_pwr_id, level="customer")
    b = raw[raw.master_pwr_id.isin(pset) & (raw.master_pwr_id != raw.customer_pwr_id)].assign(pid=lambda d: d.master_pwr_id, level="master")
    sib = a[["pid", "master_pwr_id"]].dropna().drop_duplicates().merge(raw, on="master_pwr_id").assign(level="family")
    idmap = pd.concat([a, b, sib], ignore_index=True).drop_duplicates(["pid", "mdm_id"]).reset_index(drop=True)

    K = _workbook_keys(E, idmap)
    keys = safe_ids(sorted(K.key.unique()))
    say(f"Searching {len(keys):,} company names among outside counterparties ({first} to {last_month}). "
        "The first time for a workbook this can take a few minutes; it is remembered afterwards")
    met_cols = table_columns(T_CPTY_MET) or set()
    parts = [run_sql(name_match_sql(ch, first, last_month, met_cols), "name match") for ch in _chunks(keys, 6000)]
    ent = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ent_id", "ent_name", "name_key", "amt"])
    ent = _num(_str(ent, ["ent_id", "ent_name", "name_key", "name_class"]), ["amt", "n_cust_all", "hub_tier"])
    for c in ("n_cust_all", "hub_tier", "name_class"):
        if c not in ent:
            ent[c] = np.nan if c != "name_class" else None
    ent = ent.rename(columns={"n_cust_all": "n_cust"}).sort_values("amt", ascending=False).reset_index(drop=True)
    if use_cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({"idmap": idmap, "ent": ent, "keys": K}, f)
        except Exception as e:  # noqa: BLE001
            log.warning("match cache not written: %s", e)
    return Match(idmap, ent, K, (first, last_month))


# ================================================================== a selection
@dataclass(frozen=True)
class Focus:
    key: str                 # stable id of the selection ("wbp|<master id>", "cust|<mdm>", "fam|<master pwr>", "ent|<id>")
    label: str
    mdm: tuple = ()          # PNC customers behind it
    ent: tuple = ()          # outside counterparties behind it
    wb_ids: tuple = ()       # workbook ids it stands for
    cut: bool = False        # True when the id lists were cut at MAX_FOCUS_IDS


def _cap(values):
    v = tuple(dict.fromkeys(values))
    return v[:MAX_FOCUS_IDS], len(v) > MAX_FOCUS_IDS


def _focus(key: str, label: str, ids, M: Match, family: bool) -> Focus:
    src = M.mdm_family if family else M.mdm_cust
    mdm, c1 = _cap(m for i in ids for m in src.get(i, ()))
    ent, c2 = _cap(e for i in ids for e in M.ent_of.get(i, ()))
    return Focus(key, label, mdm, ent, tuple(dict.fromkeys(ids)), c1 or c2)


def focus_ids(key: str, label: str, ids, M: Match, family: bool = True) -> Focus:
    """One selected node standing for several workbook ids (the same company listed under more than one id)."""
    return _focus(key, label, [i for i in dict.fromkeys(ids) if i not in DEFAULT_DC_IDS], M, family)


def focus_parent(master_id: str, E: pd.DataFrame, M: Match, label: str, family: bool = True) -> Focus:
    """A data center parent with every borrowing entity listed under it in the workbook."""
    ids = [master_id] + [i for i in dict.fromkeys(E.loc[E.master_id == master_id, "dc_id"].dropna()) if i not in DEFAULT_DC_IDS]
    return _focus(f"wbp|{master_id}", label, ids, M, family)


def focus_entity(dc_id: str, M: Match, label: str, family: bool = True) -> Focus:
    return _focus(f"wbe|{dc_id}", label, [dc_id], M, family)


def focus_party(so_id: str, M: Match, label: str, family: bool = True) -> Focus:
    return _focus(f"wbs|{so_id}", label, [so_id], M, family)


# ================================================================== customers: lookups and search
def cust_attr_sql(where: str, cdim_cols, tm_cols, tm_month) -> str:
    d_opt = _pick(cdim_cols, ["party_type", "naics_desc", "is_tm"])
    t_opt = _pick(tm_cols, ["dhe", "pd_grade_now", "pd_pct_now", "is_default", "past_due_days", "watchlist_exposure"]) if tm_month else []
    sel = "".join(f", d.{c}" for c in d_opt) + "".join(f", t.{c}" for c in t_opt)
    join = (f"LEFT JOIN (SELECT CAST(customer_pwr_id AS STRING) AS pwr{''.join(', ' + c for c in t_opt)} FROM {T_TM} "
            f"WHERE `month` = '{_m(tm_month)}') t ON t.pwr = CAST(d.customer_pwr_id AS STRING)") if t_opt else ""
    return f"""
SELECT CAST(d.mdm_id AS STRING) AS mdm_id, d.customer_name, CAST(d.customer_pwr_id AS STRING) AS customer_pwr_id,
       CAST(d.master_pwr_id AS STRING) AS master_pwr_id{sel}
FROM {T_CDIM} d {join}
WHERE {where}"""


_CUST_COLS = ["mdm_id", "customer_name", "customer_pwr_id", "master_pwr_id", "party_type", "naics_desc", "is_tm", "dhe",
              "pd_grade_now", "pd_pct_now", "is_default", "past_due_days", "watchlist_exposure"]


def _prep_cust(df: pd.DataFrame) -> pd.DataFrame:
    for c in _CUST_COLS:
        if c not in df:
            df[c] = None if c in ("party_type", "naics_desc", "pd_grade_now") else np.nan
    df = _str(df, ["mdm_id", "customer_name", "customer_pwr_id", "master_pwr_id", "party_type", "naics_desc", "pd_grade_now"])
    df = _num(df, ["dhe", "pd_pct_now", "past_due_days", "watchlist_exposure", "is_default", "is_tm"])
    return df.drop_duplicates("mdm_id").reset_index(drop=True)


def cust_attrs(mdm_ids, cdim_cols, tm_cols, tm_month) -> pd.DataFrame:
    """Name, power ids, party type and the latest credit snapshot for a list of customers."""
    parts = [run_sql(cust_attr_sql(f"d.mdm_id IN ({_in(ch)})", cdim_cols, tm_cols, tm_month), "customers")
             for ch in _chunks(safe_ids(mdm_ids))]
    return _prep_cust(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["mdm_id"]))


def family_members(master_pwr_id: str, cdim_cols, tm_cols, tm_month, limit: int = MAX_FOCUS_IDS) -> pd.DataFrame:
    (pid,) = safe_ids([master_pwr_id]) or [None]
    if pid is None:
        return _prep_cust(pd.DataFrame(columns=["mdm_id"]))
    sql = cust_attr_sql(f"d.master_pwr_id = '{pid}' OR d.customer_pwr_id = '{pid}'", cdim_cols, tm_cols, tm_month)
    return _prep_cust(run_sql(sql + f"\nLIMIT {int(limit)}", "family"))


def pwr_names(pwr_ids) -> dict:
    """customer_pwr_id -> customer name (used to name a family by its parent)."""
    out = {}
    for ch in _chunks(safe_ids(pwr_ids)):
        d = run_sql(f"SELECT CAST(customer_pwr_id AS STRING) AS pwr, MAX(customer_name) AS name FROM {T_CDIM} "
                    f"WHERE customer_pwr_id IN ({_in(ch)}) GROUP BY CAST(customer_pwr_id AS STRING)", "family names")
        out.update(dict(zip(d.pwr.astype(str), d.name)))
    return out


def search_customers(q: str, cdim_cols, limit: int = 60) -> pd.DataFrame:
    """PNC customers by part of the name, or by Power ID / mdm id when the text is a number."""
    q = str(q or "").strip()
    words = re.sub(r"[^A-Z0-9]+", " ", q.upper()).split()
    if not words or sum(map(len, words)) < 3:
        return _prep_cust(pd.DataFrame(columns=["mdm_id"]))
    if len(words) == 1 and words[0].isdigit():
        w = words[0]
        where = f"d.customer_pwr_id = '{w.zfill(10)}' OR d.master_pwr_id = '{w.zfill(10)}' OR CAST(d.mdm_id AS STRING) = '{w}'"
    else:
        where = " AND ".join(f"upper(d.customer_name) LIKE '%{w}%'" for w in words[:6])
    order = "CASE WHEN d.party_type = 'O' THEN 0 ELSE 1 END, " if cdim_cols and "party_type" in cdim_cols else ""
    sql = cust_attr_sql(where, cdim_cols, None, None) + f"\nORDER BY {order}d.customer_name\nLIMIT {int(limit)}"
    return _prep_cust(run_sql(sql, "customer search"))


# ================================================================== payments around a selection
def ego_sql(mdm, ent, first: str, last: str, pair_cols, limit: int = MAX_EGO_ROWS) -> str:
    opt = "".join(f", MAX(p.{c}) AS {c}" for c in _pick(pair_cols, ["o_class", "o_fi"]))
    conds = []
    if mdm:
        conds.append(f"p.cust IN ({_in(mdm)})")
    if ent:
        conds.append(f"(p.`scope` = 'ext' AND p.other IN ({_in(ent)}))")
    return f"""
SELECT CAST(p.cust AS STRING) AS cust, p.`dir` AS dir, p.`scope` AS scope, CAST(p.other AS STRING) AS other,
       SUM(CAST(p.amount AS DOUBLE)) AS amt, SUM(CAST(p.n_txn AS DOUBLE)) AS n_txn,
       COUNT(DISTINCT p.`month`) AS months, MIN(p.`month`) AS first_month, MAX(p.`month`) AS last_month{opt}
FROM {T_PAIR} p
WHERE p.`month` BETWEEN '{_m(first)}' AND '{_m(last)}' AND p.`scope` IN ('int', 'ext') AND ({' OR '.join(conds)})
GROUP BY CAST(p.cust AS STRING), p.`dir`, p.`scope`, CAST(p.other AS STRING)
ORDER BY SUM(CAST(p.amount AS DOUBLE)) DESC
LIMIT {int(limit)}"""


_EGO_COLS = ["cust", "dir", "scope", "other", "amt", "n_txn", "months", "first_month", "last_month", "o_class", "o_fi"]


def ego_rows(focus, first: str, last: str, pair_cols) -> pd.DataFrame:
    """Pairs touching the selection, summed over the period (one row per customer x direction x other side)."""
    mdm = safe_ids(m for f in focus for m in f.mdm)
    ent = safe_ids(e for f in focus for e in f.ent)
    parts = []
    for ch in _chunks(mdm):
        parts.append(run_sql(ego_sql(ch, None, first, last, pair_cols), "pairs by customer"))
    for ch in _chunks(ent):
        parts.append(run_sql(ego_sql(None, ch, first, last, pair_cols), "pairs by counterparty"))
    R = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_EGO_COLS)
    for c in _EGO_COLS:
        if c not in R:
            R[c] = None
    R = _num(_str(R, ["cust", "dir", "scope", "other", "first_month", "last_month", "o_class", "o_fi"]), ["amt", "n_txn", "months"])
    return R[_EGO_COLS].drop_duplicates(["cust", "dir", "scope", "other"]).reset_index(drop=True)


def ent_attrs(ent_ids, first: str, last: str, met_cols) -> pd.DataFrame:
    """Name, class, hub tier and reach of outside counterparties, from their metrics in the period."""
    opt = _pick(met_cols, ["name_class", "hub_tier", "n_cust_all"])
    sel = "".join(f", MAX({c}) AS {c}" for c in opt)
    parts = []
    for ch in _chunks(safe_ids(ent_ids)):
        parts.append(run_sql(f"""
SELECT CAST(ent_id AS STRING) AS ent_id, MAX(ent_name) AS ent_name{sel}
FROM {T_CPTY_MET}
WHERE `month` BETWEEN '{_m(first)}' AND '{_m(last)}' AND ent_id IN ({_in(ch)})
GROUP BY CAST(ent_id AS STRING)""", "counterparty names"))
    A = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ent_id", "ent_name"])
    for c in ("name_class", "hub_tier", "n_cust_all"):
        if c not in A:
            A[c] = None if c == "name_class" else np.nan
    return _num(_str(A, ["ent_id", "ent_name", "name_class"]), ["hub_tier", "n_cust_all"]).drop_duplicates("ent_id")


@dataclass
class Ego:
    rows: pd.DataFrame       # pairs (see ego_rows)
    cust: pd.DataFrame       # attributes of every customer seen (focus members and neighbours)
    ent: pd.DataFrame        # attributes of outside counterparties seen
    fam_names: dict          # master power id -> name
    capped: bool             # True when the pair pull hit MAX_EGO_ROWS


def load_ego(focus, first: str, last: str, cols: dict, tm_month) -> Ego:
    """Everything the graph needs about the payments around a selection."""
    R = ego_rows(focus, first, last, cols.get("pair"))
    fm = {m for f in focus for m in f.mdm}
    nb_c = R.loc[(R.scope == "int"), ["other", "amt"]].rename(columns={"other": "id"})
    nb_c = pd.concat([nb_c, R.loc[~R.cust.isin(fm), ["cust", "amt"]].rename(columns={"cust": "id"})])
    top_c = nb_c.groupby("id").amt.sum().sort_values(ascending=False).head(NAME_CAP).index.tolist() if len(nb_c) else []
    C = cust_attrs(list(fm) + top_c, cols.get("cdim"), cols.get("tm"), tm_month)
    ents = R.loc[R.scope == "ext"].groupby("other").amt.sum().sort_values(ascending=False).head(NAME_CAP).index.tolist() if len(R) else []
    A = ent_attrs(ents, first, last, cols.get("cpty_met")) if cols.get("cpty_met") else ent_attrs([], first, last, None)
    masters = sorted({m for m in C.master_pwr_id.dropna()} - set(C.customer_pwr_id.dropna()))
    names = dict(zip(C.customer_pwr_id, C.customer_name))
    names.update(pwr_names(masters[:MAX_IDS]) if masters else {})
    return Ego(R, C, A, names, len(R) >= MAX_EGO_ROWS)


# ================================================================== regular payment streams
def reg_sql(mdm, ent, first: str, last: str, reg_cols) -> str:
    opt = _pick(reg_cols, ["status", "late", "overdue", "stopped", "last_dt", "next_dt", "med_gap", "amt_med", "s12", "rail"])
    sel = "".join(f", {c}" for c in opt)
    conds = []
    if mdm:
        conds.append(f"cust IN ({_in(mdm)})")
    if ent:
        conds.append(f"(side = 'ext' AND other IN ({_in(ent)}))")
    return f"""
SELECT CAST(cust AS STRING) AS cust, `dir` AS dir, side, CAST(other AS STRING) AS other, `month` AS month, cadence{sel}
FROM {T_REG}
WHERE `month` BETWEEN '{_m(first)}' AND '{_m(last)}' AND ({' OR '.join(conds)})"""


_STREAM_COLS = ["cust", "dir", "other", "side", "month", "cadence", "state", "amt_med", "last_dt", "next_dt", "s12", "rail"]


def stream_states(R: pd.DataFrame, last_month: str) -> pd.DataFrame:
    """Latest stored row of each stream, turned into one of: on schedule, late or overdue, stopped, casual.
    A stream the table marks 'not regular', or that dropped out of the table before the last month without a stop,
    counts as casual (it is no longer regular)."""
    if R is None or R.empty:
        return pd.DataFrame(columns=_STREAM_COLS)
    R = R.copy()
    for c in ("status", "late", "overdue", "stopped", "last_dt", "next_dt", "med_gap", "amt_med", "s12", "rail"):
        if c not in R:
            R[c] = None if c in ("status", "last_dt", "next_dt", "rail") else np.nan
    R = _num(_str(R, ["cust", "dir", "side", "other", "month", "cadence", "status", "rail"]), ["late", "overdue", "stopped", "amt_med", "s12", "med_gap"])
    L = R.sort_values("month").drop_duplicates(["cust", "dir", "other"], keep="last").reset_index(drop=True)
    st = L.status.fillna("").str.strip().str.lower()
    flag = lambda c: L[c].fillna(0).astype(float) > 0   # noqa: E731
    L["state"] = np.select(
        [st == "not regular", st == "stopped", st == "overdue", flag("stopped"), L.month < last_month, flag("overdue") | flag("late")],
        ["casual", "stopped", "late or overdue", "stopped", "casual", "late or overdue"], default="on schedule")
    L.loc[L.cadence.isna() | (L.cadence.astype(str).str.lower() == "irregular"), "state"] = "casual"
    for c in ("last_dt", "next_dt"):
        L[c] = pd.to_datetime(L[c], errors="coerce").dt.strftime("%Y-%m-%d")
    return L[_STREAM_COLS]


def load_streams(focus, last: str, n_months: int, reg_cols, reg_months) -> pd.DataFrame:
    """Stream states for the pairs of a selection, read from the last REG_LOOKBACK months that exist up to `last`."""
    if not reg_cols or not {"cust", "dir", "side", "other", "month", "cadence"} <= set(reg_cols) or not reg_months:
        return pd.DataFrame(columns=_STREAM_COLS)
    have = [m for m in reg_months if m <= last]
    if not have:
        return pd.DataFrame(columns=_STREAM_COLS)
    end = have[-1]
    first = max(add_months(end, -(min(REG_LOOKBACK, max(n_months, 1)) - 1)), have[0])
    mdm = safe_ids(m for f in focus for m in f.mdm)
    ent = safe_ids(e for f in focus for e in f.ent)
    parts = [run_sql(reg_sql(ch, None, first, end, reg_cols), "streams by customer") for ch in _chunks(mdm)]
    parts += [run_sql(reg_sql(None, ch, first, end, reg_cols), "streams by counterparty") for ch in _chunks(ent)]
    return stream_states(pd.concat(parts, ignore_index=True) if parts else None, end)


# ================================================================== the graph
@dataclass(frozen=True)
class GraphOpts:
    top_n: int = 25
    family: bool = True              # PNC customers grouped by corporate family (master power id)
    direction: str = "both"          # both | out (the selection pays) | in (the selection is paid)
    min_amt: float = 0.0
    kinds: tuple = ("cust", "ent")   # PNC customers / outside counterparties
    hide_people: bool = True
    hide_hubs: bool = True
    regular: str = "all"             # all | regular | casual
    shared_only: bool = False        # only neighbours linked to 2+ selected nodes
    overlay: bool = False            # workbook relationships drawn over the payments


@dataclass
class Graph:
    nodes: pd.DataFrame      # key, kind, label, x, y, size, amt_in, amt_out, ... one row per drawn node
    edges: pd.DataFrame      # src, dst, amt, style, width, hover fields ... one row per drawn edge
    table: pd.DataFrame      # every neighbour after the filters (not only the drawn ones)
    notes: list              # plain sentences about what was left out
    members: dict            # node key -> {"mdm": [...], "ent": [...]} for drawn nodes


def _first(s):
    return next((v for v in s if isinstance(v, str) and v), None)


def classify(rows: pd.DataFrame, focus) -> tuple:
    """Split pairs into edges selection <-> neighbour and edges between selected nodes.
    flow is seen from the selection: 'out' = the selection pays the neighbour, 'in' = the neighbour pays it."""
    m2f, e2f = {}, {}
    for i, f in enumerate(focus):
        for m in f.mdm:
            m2f.setdefault(m, i)
        for e in f.ent:
            e2f.setdefault(e, i)
    R = rows.copy()
    a = R.cust.map(m2f)
    ext = R.scope == "ext"
    b = R.other.map(e2f).where(ext, R.other.map(m2f))
    R["a"], R["b"] = a, b
    A = R[a.notna() & b.isna()].assign(f=lambda d: d.a.astype(int), nb_kind=lambda d: np.where(d.scope == "ext", "ent", "cust"),
                                       nb_id=lambda d: d.other, flow=lambda d: d.dir)
    B = R[a.isna() & b.notna()].assign(f=lambda d: d.b.astype(int), nb_kind="cust", nb_id=lambda d: d.cust,
                                       flow=lambda d: np.where(d.dir == "out", "in", "out"))
    I = R[a.notna() & b.notna() & (a != b)]
    I = I[(I.scope == "ext") | (I.dir == "out")]
    I = I.assign(src=lambda d: np.where(d.dir == "out", d.a, d.b).astype(int), dst=lambda d: np.where(d.dir == "out", d.b, d.a).astype(int))
    inside = float(R.loc[a.notna() & b.notna() & (a == b) & ((R.scope == "ext") | (R.dir == "out")), "amt"].sum())
    keep = ["f", "nb_kind", "nb_id", "flow", "amt", "n_txn", "months", "first_month", "last_month", "cust", "dir", "other", "o_class", "o_fi"]
    return pd.concat([A[keep], B[keep]], ignore_index=True), I, inside


def _edge_agg(X: pd.DataFrame, by) -> pd.DataFrame:
    """Sum pair rows into drawn edges; the largest regular stream on an edge gives its state and schedule."""
    if X.empty:
        return pd.DataFrame(columns=list(by) + ["amt", "n_txn", "n_streams", "months", "last_month", "n_reg", "reg_amt",
                                               "state", "cadence", "amt_med", "last_dt", "next_dt"])
    g = X.groupby(list(by), sort=False)
    E = g.agg(amt=("amt", "sum"), n_txn=("n_txn", "sum"), n_streams=("amt", "size"), months=("months", "max"),
              last_month=("last_month", "max")).reset_index()
    reg = X[X.state.isin(STATES[:3])].sort_values("amt", ascending=False)
    top = reg.drop_duplicates(list(by))[list(by) + ["state", "cadence", "amt_med", "last_dt", "next_dt"]]
    cnt = reg.groupby(list(by), sort=False).agg(n_reg=("amt", "size"), reg_amt=("amt", "sum")).reset_index()
    E = E.merge(cnt, on=list(by), how="left").merge(top, on=list(by), how="left")
    E["n_reg"] = E.n_reg.fillna(0).astype(int)
    E["reg_amt"] = E.reg_amt.fillna(0.0)
    E["state"] = E.state.where(E.state.notna(), "casual")
    return E


def build_graph(ego: Ego, streams: pd.DataFrame, focus, opts: GraphOpts, match: Match = None, wb_links: pd.DataFrame = None,
                wb_parties: pd.DataFrame = None) -> Graph:
    """Nodes and edges for the picture, plus the full neighbour table.
    wb_links: workbook rows (tab, master_id, dc_id, so_id, type, amount) of the selected tabs, for the overlay."""
    notes = []
    X, inner, inside = classify(ego.rows, focus)
    C = ego.cust.set_index("mdm_id") if len(ego.cust) else pd.DataFrame(columns=_CUST_COLS[1:])
    A = ego.ent.set_index("ent_id") if len(ego.ent) else pd.DataFrame(columns=["ent_name", "name_class", "hub_tier", "n_cust_all"])

    # ---- stream state per pair row
    if streams is not None and len(streams):
        s = streams[["cust", "dir", "other", "state", "cadence", "amt_med", "last_dt", "next_dt"]]
        X = X.merge(s, on=["cust", "dir", "other"], how="left")
        inner = inner.merge(s, on=["cust", "dir", "other"], how="left")
    else:
        for d in (X, inner):
            for c in ("state", "cadence", "amt_med", "last_dt", "next_dt"):
                d[c] = None
    for d in (X, inner):
        d["state"] = d.state.where(d.state.notna(), "casual")

    # ---- neighbour node of each pair row
    is_c = X.nb_kind == "cust"
    pwr = X.nb_id.map(C.customer_pwr_id) if len(C) else pd.Series(None, index=X.index, dtype=object)
    mas = X.nb_id.map(C.master_pwr_id) if len(C) else pd.Series(None, index=X.index, dtype=object)
    fam = mas.where(mas.notna(), pwr)
    if opts.family:
        ckey = ("fam|" + fam.astype(str)).where(fam.notna(), "cust|" + X.nb_id.astype(str))
    else:
        ckey = "cust|" + X.nb_id.astype(str)
    X["node"] = np.where(is_c, ckey, "ent|" + X.nb_id.astype(str))
    X["party_type"] = X.nb_id.map(C.party_type) if len(C) else None
    X["name_class"] = X.nb_id.map(A.name_class) if len(A) else None
    X["name_class"] = X.name_class.where(X.name_class.notna(), X.o_class)
    X["hub_tier"] = X.nb_id.map(A.hub_tier) if len(A) else np.nan
    wb_m = match.wb_of_mdm if match else {}
    wb_e = match.wb_of_ent if match else {}
    X["in_wb"] = np.where(is_c, X.nb_id.isin(wb_m), X.nb_id.isin(wb_e))

    # ---- row filters (workbook parties are never filtered out as people or hubs)
    n0 = X.node.nunique()
    keep = X.nb_kind.isin(opts.kinds)
    if opts.direction in ("in", "out"):
        keep &= X.flow == opts.direction
    if opts.hide_people:
        person = np.where(is_c, X.party_type.astype(str).str.upper() == "P", X.name_class.astype(str).str.lower() == "person")
        keep &= ~person | X.in_wb
    if opts.hide_hubs:
        keep &= ~(X.hub_tier.fillna(0) >= HUB_TIER) | X.in_wb | is_c
    X = X[keep]

    # ---- node table (every neighbour after the filters)
    if len(X):
        g = X.groupby("node", sort=False)
        N = g.agg(kind=("nb_kind", "first"), amt=("amt", "sum"), n_txn=("n_txn", "sum"), months=("months", "max"),
                  last_month=("last_month", "max"), n_focus=("f", "nunique"), in_wb=("in_wb", "max"),
                  hub_tier=("hub_tier", "max"), name_class=("name_class", _first), bank=("o_fi", _first)).reset_index()
        N = N.merge(X[X.flow == "out"].groupby("node").amt.sum().rename("amt_out"), on="node", how="left")
        N = N.merge(X[X.flow == "in"].groupby("node").amt.sum().rename("amt_in"), on="node", how="left")
        N[["amt_out", "amt_in"]] = N[["amt_out", "amt_in"]].fillna(0.0)
    else:
        N = pd.DataFrame(columns=["node", "kind", "amt", "n_txn", "months", "last_month", "n_focus", "in_wb", "hub_tier",
                                  "name_class", "bank", "amt_out", "amt_in"])
    mem = X.drop_duplicates(["node", "nb_id"]).groupby("node").nb_id.agg(list).to_dict() if len(X) else {}

    def describe(node, kind):
        ids = mem.get(node, [])
        if kind == "ent":
            e = ids[0]
            nm = A.ent_name.get(e) if len(A) else None
            return {"label": nm or e, "id": e, "members": 1, "wb": set().union(*[wb_e.get(i, set()) for i in ids]),
                    "n_cust_all": A.n_cust_all.get(e) if len(A) else np.nan}
        rows = C.reindex([i for i in ids if i in C.index])
        pw = rows.drop_duplicates("customer_pwr_id") if len(rows) else rows
        out = {"members": len(ids), "wb": set().union(*[wb_m.get(i, set()) for i in ids]) if ids else set(),
               "pwr": _first(rows.customer_pwr_id) if len(rows) else None,
               "master": _first(rows.master_pwr_id) if len(rows) else None,
               "party_type": _first(rows.party_type) if len(rows) else None,
               "naics": _first(rows.naics_desc) if len(rows) else None,
               "dhe": float(pw.dhe.sum()) if len(pw) and pw.dhe.notna().any() else np.nan,
               "pd_pct": float(pw.pd_pct_now.max()) if len(pw) and pw.pd_pct_now.notna().any() else np.nan,
               "pd_grade": (pw.sort_values("pd_pct_now", ascending=False).pd_grade_now.iloc[0] if len(pw) and pw.pd_pct_now.notna().any() else None),
               "is_default": bool((pw.is_default.fillna(0) > 0).any()) if len(pw) else False,
               "past_due": float(pw.past_due_days.max()) if len(pw) and pw.past_due_days.notna().any() else np.nan}
        if node.startswith("fam|"):
            fid = node[4:]
            nm = ego.fam_names.get(fid) or (_first(rows.customer_name) if len(rows) else None) or fid
            out.update(label=nm, id=fid)
        else:
            out.update(label=(_first(rows.customer_name) if len(rows) else None) or (ids[0] if ids else node), id=ids[0] if ids else node)
        return out

    info = {r.node: describe(r.node, r.kind) for r in N.itertuples()}
    for c, dflt in (("label", None), ("id", None), ("members", 1), ("pwr", None), ("master", None), ("party_type", None),
                    ("naics", None), ("dhe", np.nan), ("pd_pct", np.nan), ("pd_grade", None), ("is_default", False),
                    ("past_due", np.nan), ("n_cust_all", np.nan)):
        N[c] = [info[k].get(c, dflt) for k in N.node] if len(N) else []
    N["wb_ids"] = [tuple(sorted(info[k]["wb"])) for k in N.node] if len(N) else []

    # ---- edges selection <-> neighbour, and node-level filters
    EG = _edge_agg(X, ["f", "node", "flow"])
    reg_nodes = set(EG.loc[EG.state != "casual", "node"])
    N["regular"] = N.node.isin(reg_nodes)
    keepn = N.amt >= float(opts.min_amt or 0)
    if opts.regular == "regular":
        keepn &= N.regular
    elif opts.regular == "casual":
        keepn &= ~N.regular
    if opts.shared_only and len(focus) > 1:
        keepn &= N.n_focus >= 2
    N = N[keepn].sort_values("amt", ascending=False).reset_index(drop=True)
    if n0 - len(N) > 0:
        notes.append(f"{n0 - len(N):,} neighbours are hidden by the filters.")
    if inside > 0:
        notes.append(f"${inside:,.0f} moved between accounts inside a selected node and is not drawn.")
    if ego.capped:
        notes.append(f"Only the {MAX_EGO_ROWS:,} largest pairs were read for this selection.")
    if any(f.cut for f in focus):
        notes.append(f"A selected node has more than {MAX_FOCUS_IDS:,} customers or accounts behind it; the list was cut.")

    # ---- workbook counterparts of the selection (for pins and the overlay)
    PT = wb_parties.set_index("id") if wb_parties is not None and len(wb_parties) else None
    wb_cp = []        # (focus index, workbook id of the other side, types, tabs, workbook amount)
    if opts.overlay and wb_links is not None and len(wb_links):
        for i, f in enumerate(focus):
            ids = set(f.wb_ids)
            if not ids:
                continue
            L = wb_links
            d1 = L[L.dc_id.isin(ids) | L.master_id.isin(ids)][["so_id", "type", "tab", "amount"]].rename(columns={"so_id": "cp"})
            d2 = L[L.so_id.isin(ids)][["dc_id", "type", "tab", "amount"]].rename(columns={"dc_id": "cp"})
            d = pd.concat([d1, d2]).dropna(subset=["cp"])
            d = d[~d.cp.isin(ids) & ~d.cp.isin(DEFAULT_DC_IDS)]
            for cp, gg in d.groupby("cp", sort=False):
                wb_cp.append((i, cp, tuple(sorted(set(gg.type))), tuple(dict.fromkeys(gg.tab)), float(gg.amount.sum(min_count=1)) if gg.amount.notna().any() else np.nan))
    cp_ids = {cp for _, cp, _, _, _ in wb_cp}
    node_of_wb = {}
    for k, ids in zip(N.node, N.wb_ids):
        for w in ids:
            node_of_wb.setdefault(w, []).append(k)
    focus_of_wb = {w: i for i, f in enumerate(focus) for w in f.wb_ids}

    # ---- which neighbours are drawn
    top = list(N.node.head(int(opts.top_n)))
    pins = [k for k in N.node.iloc[int(opts.top_n):] if set(N.wb_ids[N.node == k].iloc[0]) & cp_ids][:PIN_CAP] if cp_ids else []
    shown = top + pins
    hidden = N[~N.node.isin(shown)]
    V = N[N.node.isin(shown)].copy()

    fl = []
    for i, f in enumerate(focus):
        fx = X[X.f == i] if len(X) else X
        rows = C.reindex([m for m in f.mdm if m in C.index]).drop_duplicates("customer_pwr_id") if len(C) else C
        fl.append({"node": f"focus|{i}", "kind": "focus", "label": f.label, "id": f.key,
                   "amt": float(fx.amt.sum()) if len(fx) else 0.0,
                   "amt_out": float(fx.loc[fx.flow == "out", "amt"].sum()) if len(fx) else 0.0,
                   "amt_in": float(fx.loc[fx.flow == "in", "amt"].sum()) if len(fx) else 0.0,
                   "members": len(f.mdm) + len(f.ent), "n_mdm": len(f.mdm), "n_ent": len(f.ent),
                   "dhe": float(rows.dhe.sum()) if len(rows) and rows.dhe.notna().any() else np.nan,
                   "pd_pct": float(rows.pd_pct_now.max()) if len(rows) and rows.pd_pct_now.notna().any() else np.nan,
                   "pd_grade": rows.sort_values("pd_pct_now", ascending=False).pd_grade_now.iloc[0] if len(rows) and rows.pd_pct_now.notna().any() else None,
                   "is_default": bool((rows.is_default.fillna(0) > 0).any()) if len(rows) else False,
                   "wb_ids": tuple(f.wb_ids), "in_wb": bool(f.wb_ids)})
    FN = pd.DataFrame(fl)

    edges = []
    Es = EG[EG.node.isin(shown)]
    for r in Es.itertuples():
        src, dst = (f"focus|{r.f}", r.node) if r.flow == "out" else (r.node, f"focus|{r.f}")
        edges.append({"src": src, "dst": dst, "amt": r.amt, "n_txn": r.n_txn, "style": r.state, "cadence": r.cadence,
                      "amt_med": r.amt_med, "last_dt": r.last_dt, "next_dt": r.next_dt, "n_streams": r.n_streams,
                      "n_reg": r.n_reg, "months": r.months, "layer": "pay"})
    if len(hidden):
        Eo = EG[EG.node.isin(set(hidden.node))].groupby(["f", "flow"]).agg(amt=("amt", "sum"), n_txn=("n_txn", "sum"), n_streams=("node", "nunique")).reset_index()
        for r in Eo.itertuples():
            src, dst = (f"focus|{r.f}", "others") if r.flow == "out" else ("others", f"focus|{r.f}")
            edges.append({"src": src, "dst": dst, "amt": r.amt, "n_txn": r.n_txn, "style": "others", "n_streams": r.n_streams, "layer": "pay"})
    if len(inner):
        Ei = _edge_agg(inner.assign(f=inner.src, node=inner.dst.astype(str), flow="out"), ["f", "node", "flow"])
        for r in Ei.itertuples():
            edges.append({"src": f"focus|{r.f}", "dst": f"focus|{r.node}", "amt": r.amt, "n_txn": r.n_txn, "style": r.state,
                          "cadence": r.cadence, "amt_med": r.amt_med, "last_dt": r.last_dt, "next_dt": r.next_dt,
                          "n_streams": r.n_streams, "n_reg": r.n_reg, "months": r.months, "layer": "pay"})

    extra = []
    if len(hidden):
        extra.append({"node": "others", "kind": "others", "label": f"{len(hidden):,} others", "id": "others",
                      "amt": float(hidden.amt.sum()), "amt_in": float(hidden.amt_in.sum()), "amt_out": float(hidden.amt_out.sum()),
                      "members": len(hidden)})

    # ---- workbook overlay: dotted links, and hollow nodes for parties with no payments
    wb_missing = 0
    if wb_cp:
        drawn = set(V.node)
        paid_any = {(e["src"], e["dst"]) for e in edges} | {(e["dst"], e["src"]) for e in edges}
        only = []
        for i, cp, types, tabs, amt in wb_cp:
            tgt = [k for k in node_of_wb.get(cp, []) if k in drawn]
            if cp in focus_of_wb and focus_of_wb[cp] != i:
                tgt = [f"focus|{focus_of_wb[cp]}"]
            nm = PT.name.get(cp, cp) if PT is not None else cp
            if tgt:
                for k in tgt:
                    edges.append({"src": f"focus|{i}", "dst": k, "amt": amt, "style": "workbook", "layer": "wb", "wb_types": ", ".join(types),
                                  "wb_tabs": ", ".join(tabs), "wb_name": nm, "wb_paid": (f"focus|{i}", k) in paid_any})
            elif not node_of_wb.get(cp):
                only.append((i, cp, types, tabs, amt, nm))
        only.sort(key=lambda t: -(t[4] if t[4] == t[4] else 0))
        wb_missing = max(0, len({o[1] for o in only}) - WB_ONLY_CAP)
        seen = {}
        for i, cp, types, tabs, amt, nm in only:
            if cp not in seen:
                if len(seen) >= WB_ONLY_CAP:
                    continue
                seen[cp] = True
                extra.append({"node": f"wb|{cp}", "kind": "wb_only", "label": nm, "id": cp, "amt": 0.0, "amt_in": 0.0, "amt_out": 0.0,
                              "members": 1, "wb_ids": (cp,), "in_wb": True, "wb_status": match.status(cp) if match else ""})
            edges.append({"src": f"focus|{i}", "dst": f"wb|{cp}", "amt": amt, "style": "workbook", "layer": "wb", "wb_types": ", ".join(types),
                          "wb_tabs": ", ".join(tabs), "wb_name": nm, "wb_paid": False})
        if wb_missing:
            notes.append(f"{wb_missing:,} more workbook parties with no payments are not drawn (the largest {WB_ONLY_CAP} are).")

    nodes = pd.concat([FN, V, pd.DataFrame(extra)], ignore_index=True) if len(FN) else pd.DataFrame()
    nodes = nodes.rename(columns={"node": "key"})
    Ed = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["src", "dst", "amt", "style", "layer"])
    if len(nodes):
        _layout(nodes, Ed, len(focus))
        _sizes(nodes, Ed)
    members = {}
    for k in V.node:
        ids = mem.get(k, [])
        members[k] = {"mdm": ids if not k.startswith("ent|") else [], "ent": ids if k.startswith("ent|") else []}

    T = N.copy()
    regc = EG.sort_values("amt", ascending=False).drop_duplicates("node").set_index("node") if len(EG) else None
    T["state"] = T.node.map(regc.state) if regc is not None else None
    reg_only = EG[EG.state != "casual"].sort_values("amt", ascending=False).drop_duplicates("node").set_index("node") if len(EG) else None
    for c in ("state", "cadence", "amt_med", "last_dt", "next_dt"):
        T["reg_" + c] = T.node.map(reg_only[c]) if reg_only is not None and len(reg_only) else None
    T["drawn"] = T.node.isin(shown)
    if PT is not None and len(T):
        T["wb_types"] = [", ".join(sorted({t for w in ids if w in PT.index for t in PT.types[w]})) for ids in T.wb_ids]
    else:
        T["wb_types"] = ""
    return Graph(nodes, Ed, T, notes, members)


def _layout(nodes: pd.DataFrame, edges: pd.DataFrame, k: int) -> None:
    """Positions: selected nodes in the middle, each with its own neighbours on the arc of the ring facing it (the
    arc grows with the number of neighbours). Neighbours shared by several selected nodes sit on an inner ring.
    Largest first, clockwise; crowded stretches are staggered over two or three radii so labels do not collide."""
    link = {}
    for e in edges.itertuples():
        for a, b in ((e.src, e.dst), (e.dst, e.src)):
            if a.startswith("focus|") and not b.startswith("focus|"):
                link.setdefault(b, set()).add(int(a.split("|")[1]))
    nb = nodes[~nodes.key.str.startswith("focus|")]
    order = {"cust": 0, "ent": 1, "others": 2, "wb_only": 3}
    nb = nb.assign(_o=nb.kind.map(order).fillna(4)).sort_values(["_o", "amt"], ascending=[True, False])
    shared = [n for n in nb.key if len(link.get(n, ())) >= 2] if k > 1 else []
    sset = set(shared)
    groups = {i: [] for i in range(k)}
    for n in nb.key:
        if n not in sset:
            groups[min(link.get(n, {0})) if k > 1 else 0].append(n)
    x, y = {}, {}
    w = {i: max(len(groups[i]), 6) for i in range(k)}
    tot = float(sum(w.values()))
    start = math.pi / 2 + (math.pi * w[0] / tot if k > 1 else 0.0)     # the first selected node's arc is centred at the top
    for i in range(k):
        span = 2 * math.pi * w[i] / tot
        mid = start - span / 2
        r0 = 0.0 if k == 1 else 0.30
        x[f"focus|{i}"], y[f"focus|{i}"] = r0 * math.cos(mid), r0 * math.sin(mid)
        items = groups[i]
        m = len(items)
        pad = 0.0 if k == 1 else span * 0.07
        per_rad = m / max(span - 2 * pad, 1e-6)
        for j, n in enumerate(items):
            t = j / m if k == 1 else (j + 0.5) / m
            ang = start - pad - t * (span - 2 * pad)
            flat = abs(math.cos(ang)) < 0.62                               # top and bottom of the ring: labels run into each other
            levels = (3 if per_rad > 4.5 else 2 if per_rad > 2.2 else 1) if flat else 1
            r = 1.0 - 0.15 * (j % levels)
            x[n], y[n] = r * math.cos(ang), r * math.sin(ang)
        start -= span
    for j, n in enumerate(shared):
        ang = math.pi / 2 - 2 * math.pi * (j + 0.5) / max(len(shared), 1)
        x[n], y[n] = 0.60 * math.cos(ang), 0.60 * math.sin(ang)
    nodes["x"] = nodes.key.map(x).fillna(0.0) * 1.45
    nodes["y"] = nodes.key.map(y).fillna(0.0)


def _sizes(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    """Marker size and line width grow with the square root of the dollars (floor and cap keep small ones visible)."""
    nb = nodes[nodes.kind.isin(["cust", "ent"])]
    mx = float(nb.amt.max()) if len(nb) and nb.amt.max() > 0 else 1.0
    fmx = float(nodes.loc[nodes.kind == "focus", "amt"].max() or 0) or 1.0
    size = []
    for r in nodes.itertuples():
        if r.kind == "focus":
            size.append(34 + 18 * math.sqrt(max(r.amt, 0) / fmx))
        elif r.kind == "others":
            size.append(26.0)
        elif r.kind == "wb_only":
            size.append(13.0)
        else:
            size.append(9 + 33 * math.sqrt(max(r.amt, 0) / mx))
    nodes["size"] = size
    if len(edges):
        pay = edges[(edges.layer == "pay") & (edges["style"] != "between")]
        emx = float(pay.amt.max()) if len(pay) and pay.amt.max() > 0 else 1.0
        btw = edges[edges["style"] == "between"]
        bmx = float(btw.amt.max()) if len(btw) and btw.amt.max() > 0 else 1.0
        edges["width"] = [1.6 if e.layer == "wb" else (0.7 + 1.1 * math.sqrt(max(e.amt, 0) / bmx) if e.style == "between"
                                                       else 1.0 + 8.0 * math.sqrt(max(e.amt or 0, 0) / emx)) for e in edges.itertuples()]


def between_sql(mdm, ent, first: str, last: str) -> str:
    conds = [f"(p.`scope` = 'int' AND p.`dir` = 'out' AND p.other IN ({_in(mdm)}))"] if mdm else []
    if ent:
        conds.append(f"(p.`scope` = 'ext' AND p.other IN ({_in(ent)}))")
    return f"""
SELECT CAST(p.cust AS STRING) AS cust, p.`dir` AS dir, p.`scope` AS scope, CAST(p.other AS STRING) AS other,
       SUM(CAST(p.amount AS DOUBLE)) AS amt, SUM(CAST(p.n_txn AS DOUBLE)) AS n_txn
FROM {T_PAIR} p
WHERE p.`month` BETWEEN '{_m(first)}' AND '{_m(last)}' AND p.cust IN ({_in(mdm)}) AND ({' OR '.join(conds)})
GROUP BY CAST(p.cust AS STRING), p.`dir`, p.`scope`, CAST(p.other AS STRING)"""


def between_edges(members: dict, first: str, last: str) -> pd.DataFrame:
    """Payments among the drawn neighbours themselves (not through the selection): src, dst, amt, n_txn."""
    n_of_m = {m: k for k, v in members.items() for m in v["mdm"]}
    n_of_e = {e: k for k, v in members.items() for e in v["ent"]}
    mdm, ent = safe_ids(n_of_m)[:MAX_IDS], safe_ids(n_of_e)[:MAX_IDS]
    if not mdm or (len(mdm) < 2 and not ent):
        return pd.DataFrame(columns=["src", "dst", "amt", "n_txn"])
    R = _num(_str(run_sql(between_sql(mdm, ent, first, last), "between neighbours"), ["cust", "dir", "scope", "other"]), ["amt", "n_txn"])
    if R.empty:
        return pd.DataFrame(columns=["src", "dst", "amt", "n_txn"])
    a = R.cust.map(n_of_m)
    b = R.other.map(n_of_e).where(R.scope == "ext", R.other.map(n_of_m))
    out = R.dir == "out"
    R = R.assign(src=a.where(out, b), dst=b.where(out, a)).dropna(subset=["src", "dst"])
    R = R[R.src != R.dst]
    return R.groupby(["src", "dst"], sort=False).agg(amt=("amt", "sum"), n_txn=("n_txn", "sum")).reset_index()


def add_between(G: Graph, B: pd.DataFrame) -> None:
    """Put neighbour-to-neighbour payments on the graph as thin muted lines on their own width scale."""
    if B is None or B.empty:
        return
    B = B.assign(style="between", layer="pay")
    G.edges = pd.concat([G.edges.drop(columns=["width"], errors="ignore"), B], ignore_index=True)
    _sizes(G.nodes, G.edges)


# ================================================================== customer page
def customer_flows(mdm_ids, met_cols) -> pd.DataFrame:
    """Money in and out per month for a set of customers (all months held)."""
    cols = _pick(met_cols, ["amt_all_in", "amt_all_out", "amt_ext_in", "amt_ext_out", "amt_int_in", "amt_int_out",
                            "ntxn_all_in", "ntxn_all_out", "ncpty_all_in", "ncpty_all_out"])
    if not cols:
        return pd.DataFrame(columns=["month"])
    sel = ", ".join(f"SUM(CAST({c} AS DOUBLE)) AS {c}" for c in cols)
    parts = [run_sql(f"SELECT `month` AS month, {sel} FROM {T_CUST_MET} WHERE mdm_id IN ({_in(ch)}) GROUP BY `month`", "customer flows")
             for ch in _chunks(safe_ids(mdm_ids))]
    if not parts:
        return pd.DataFrame(columns=["month"] + cols)
    D = _num(pd.concat(parts, ignore_index=True), cols)
    return D.groupby("month", as_index=False)[cols].sum().sort_values("month")


def customer_credit(pwr_ids, tm_cols) -> pd.DataFrame:
    """Credit rows per power id per month: DHE, PD, default, past due, watchlist and other exposures."""
    cols = _pick(tm_cols, ["dhe", "pd_pct_now", "pd_grade_now", "is_default", "past_due_days", "watchlist_exposure", "wrre", "mre",
                           "soft_exposure"])
    if not cols or not pwr_ids:
        return pd.DataFrame(columns=["month", "customer_pwr_id"])
    sel = ", ".join(cols)
    parts = [run_sql(f"SELECT `month` AS month, CAST(customer_pwr_id AS STRING) AS customer_pwr_id, {sel} FROM {T_TM} "
                     f"WHERE customer_pwr_id IN ({_in(ch)})", "customer credit") for ch in _chunks(safe_ids(pwr_ids))]
    D = pd.concat(parts, ignore_index=True)
    return _num(_str(D, ["month", "customer_pwr_id", "pd_grade_now"]), [c for c in cols if c != "pd_grade_now"]).sort_values("month")


def credit_by_month(D: pd.DataFrame) -> pd.DataFrame:
    """One row per month over the power ids: exposures summed, PD and past-due as the worst value."""
    if D.empty:
        return D
    agg = {}
    for c in ("dhe", "watchlist_exposure", "wrre", "mre", "soft_exposure"):
        if c in D:
            agg[c] = (c, lambda s: s.sum(min_count=1))
    for c in ("pd_pct_now", "past_due_days", "is_default"):
        if c in D:
            agg[c] = (c, "max")
    out = D.groupby("month").agg(**agg).reset_index()
    if "pd_grade_now" in D and "pd_pct_now" in D:
        w = D.sort_values("pd_pct_now", ascending=False).drop_duplicates("month").set_index("month").pd_grade_now
        out["pd_grade_now"] = out.month.map(w)
    return out.sort_values("month")


def customer_deposits(pwr_ids, dep_cols) -> pd.DataFrame:
    """Deposit balances per month (summed over the power ids); empty when the table is not there."""
    cols = _pick(dep_cols, ["dep_bal_eom", "dep_bal_mean", "dep_bal_min", "dep_bal_max"])
    if not cols or not dep_cols or not {"customer_pwr_id", "month"} <= set(dep_cols) or not pwr_ids:
        return pd.DataFrame(columns=["month"])
    sel = ", ".join(f"SUM(CAST({c} AS DOUBLE)) AS {c}" for c in cols)
    parts = [run_sql(f"SELECT `month` AS month, {sel} FROM {T_DEP} WHERE customer_pwr_id IN ({_in(ch)}) GROUP BY `month`", "deposits")
             for ch in _chunks(safe_ids(pwr_ids))]
    D = _num(pd.concat(parts, ignore_index=True), cols)
    return D.groupby("month", as_index=False)[cols].sum().sort_values("month")


# ================================================================== counterparty page (as in the Prospect Finder)
RAILS = [("ach", "ACH"), ("wire", "Wire"), ("rtp_prt", "RTP"), ("rtp_p2p", "RTP P2P"), ("check", "Check"), ("pcard", "Card")]
# Month-by-month charts: (title, unit, [(column, series label)], what it means). A chart shows the columns that exist.
TREND_PANELS = [
    ("PNC clients", "count", [("n_cust_in", "paying it"), ("n_cust_out", "it pays")], "Distinct PNC clients in the month."),
    ("Money with PNC clients", "flow", [("amt_in", "received"), ("amt_out", "paid")], "Money it received from and paid to PNC clients."),
    ("TM clients", "count", [("n_cust_tm_in", "paying it"), ("n_cust_tm_out", "it pays")], "Distinct TM clients in the month."),
    ("Average payment", "usd", [("avg_ticket_in", "received"), ("avg_ticket_out", "paid")], "Money divided by the number of payments."),
    ("Days with payments", "count", [("active_days_in", "received"), ("active_days_out", "paid")],
     "Days in the month with at least one payment."),
    ("Share of its largest PNC client", "pct", [("top1_share_in", "received"), ("top1_share_out", "paid")],
     "How much of its money with PNC clients comes from (goes to) a single client."),
    ("Clients gained and lost", "count", [("new_cust_in", "new payers"), ("lost_cust_in", "lost payers")],
     "PNC clients paying it for the first time in 12 months, and those that stopped."),
    ("Clients that depend on it", "count", [("n_cust_dep50_in", "pay it 50%+ of their outflow"), ("n_cust_dep50_out", "get 50%+ of their inflow from it")],
     "PNC clients for whom it is at least half of their money out (or in)."),
    ("Banks it uses", "count", [("n_fi", "banks")], "Banks its accounts sit at, as seen on payments with PNC clients."),
]


def cpty_trend(ent_id: str, met_cols) -> pd.DataFrame:
    """One counterparty month by month, with every trend column that exists."""
    (eid,) = safe_ids([ent_id]) or [None]
    want = ["ent_name", "name_class", "hub_tier", "n_cust_all"] + [c for _, _, ser, _ in TREND_PANELS for c, _ in ser]
    cols = _pick(met_cols, list(dict.fromkeys(want)))
    if eid is None or not met_cols:
        return pd.DataFrame(columns=["month"])
    sel = "".join(f", {c}" for c in cols)
    D = run_sql(f"SELECT `month` AS month{sel} FROM {T_CPTY_MET} WHERE ent_id = '{eid}'", "counterparty trend")
    return _num(D, [c for c in cols if c not in ("ent_name", "name_class")]).sort_values("month").reset_index(drop=True)


def cpty_clients_sql(ent_id: str, first: str, last: str, cdim_cols, limit: int = 5000) -> str:
    opt = _pick(cdim_cols, ["is_tm", "party_type", "naics_desc"])
    sel = "".join(f", c.{c}" for c in opt)
    amt, ntx = "CAST(p.amount AS DOUBLE)", "CAST(p.n_txn AS DOUBLE)"
    return f"""
SELECT CAST(p.cust AS STRING) AS mdm_id, c.customer_name, CAST(c.customer_pwr_id AS STRING) AS customer_pwr_id{sel},
       SUM(CASE WHEN p.`dir` = 'out' THEN {amt} ELSE 0 END) AS paid_to_it,
       SUM(CASE WHEN p.`dir` = 'in' THEN {amt} ELSE 0 END) AS received_from_it,
       SUM(CASE WHEN p.`dir` = 'out' THEN {ntx} ELSE 0 END) AS n_paid,
       SUM(CASE WHEN p.`dir` = 'in' THEN {ntx} ELSE 0 END) AS n_received,
       COUNT(DISTINCT p.`month`) AS months
FROM {T_PAIR} p JOIN {T_CDIM} c ON p.cust = c.mdm_id
WHERE p.`month` BETWEEN '{_m(first)}' AND '{_m(last)}' AND p.`scope` = 'ext' AND p.other = '{ent_id}'
GROUP BY CAST(p.cust AS STRING), c.customer_name, CAST(c.customer_pwr_id AS STRING){sel}
ORDER BY SUM({amt}) DESC
LIMIT {int(limit)}"""


def cpty_clients(ent_id: str, first: str, last: str, cdim_cols) -> pd.DataFrame:
    (eid,) = safe_ids([ent_id]) or [None]
    if eid is None:
        return pd.DataFrame()
    D = run_sql(cpty_clients_sql(eid, first, last, cdim_cols), "counterparty clients")
    return _num(_str(D, ["mdm_id", "customer_name", "customer_pwr_id", "party_type", "naics_desc"]),
                ["paid_to_it", "received_from_it", "n_paid", "n_received", "months", "is_tm"])


def cpty_accounts_sql(ent_id: str, first: str, last: str, routing_col, have_names: bool, limit: int = 5000) -> str:
    """Accounts merged into the counterparty, with name and bank as they appear on the payments in the period."""
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
    nl = "\n"
    return f"""
WITH acct_ids AS (SELECT DISTINCT cpty_id FROM {T_MAP} WHERE ent_l2 = '{ent_id}'),
m AS (
  SELECT t.cpty_id, t.cpty_name, t.fi, SUM(CAST(t.amount AS DOUBLE)) AS amt, SUM(CAST(t.n_txn AS DOUBLE)) AS n_txn,
         MIN(t.`month`) AS first_month, MAX(t.`month`) AS last_month
  FROM {T_CPTY_MONTH} t JOIN acct_ids ON t.cpty_id = acct_ids.cpty_id
  WHERE t.`month` BETWEEN '{_m(first)}' AND '{_m(last)}'
  GROUP BY t.cpty_id, t.cpty_name, t.fi
)
SELECT m.cpty_id, m.cpty_name, m.fi, COALESCE({', '.join(bank)}) AS bank, m.amt, m.n_txn, m.first_month, m.last_month
FROM m
{nl.join(joins)}
ORDER BY m.amt DESC
LIMIT {int(limit)}"""


def mask_account(cpty_id) -> str:
    """Account ids are never shown in full: bank routing part kept, the rest reduced to its last 4 characters."""
    s = str(cpty_id or "")
    m = re.match(r"^(\d{9})-(.*)$", s)
    tail = m.group(2) if m else s
    return (m.group(1) + "-" if m else "") + "•••" + tail[-4:]


def cpty_accounts(ent_id: str, first: str, last: str, cols: dict) -> pd.DataFrame:
    """Accounts of a counterparty: masked id, names seen, bank, dollars and payments. Empty when tables are missing."""
    (eid,) = safe_ids([ent_id]) or [None]
    need = cols.get("map") and {"cpty_id", "ent_l2"} <= set(cols["map"]) and cols.get("cpty_month") and \
        {"cpty_id", "cpty_name", "fi", "amount", "n_txn", "month"} <= set(cols["cpty_month"])
    if eid is None or not need:
        return pd.DataFrame()
    rc = cols.get("fi_routing") or set()
    rcol = next((c for c in sorted(rc) if "rout" in c and "fi" not in c), None) if "fi_canonical" in rc else None
    names = {"fi_raw", "fi_canonical"} <= (cols.get("fi_names") or set())
    D = run_sql(cpty_accounts_sql(eid, first, last, rcol, names), "counterparty accounts")
    D = _num(_str(D, ["cpty_id", "cpty_name", "fi", "bank", "first_month", "last_month"]), ["amt", "n_txn"])
    if D.empty:
        return D
    D["bank"] = D.bank.where(D.bank.notna(), "(not known)")
    g = D.groupby("cpty_id", sort=False)
    A = g.agg(amt=("amt", "sum"), n_txn=("n_txn", "sum"), first_month=("first_month", "min"), last_month=("last_month", "max"),
              bank=("bank", "first"), names=("cpty_name", lambda s: " / ".join(dict.fromkeys(v for v in s if isinstance(v, str)))),
              n_names=("cpty_name", "nunique")).reset_index()
    A["account"] = A.cpty_id.map(mask_account)
    return A.sort_values("amt", ascending=False).reset_index(drop=True)


def cpty_streams(ent_id: str, last: str, n_months: int, reg_cols, reg_months) -> pd.DataFrame:
    return load_streams([Focus(f"ent|{ent_id}", "", (), (ent_id,))], last, n_months, reg_cols, reg_months)
