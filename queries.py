"""
queries.py — Neo4j Cypher queries for the Rail Shift Group Analyzer.
PNC Treasury Management · Payment Knowledge Network

Schema (pkn_data_model.md):

  Nodes:
    PncCustomer   — customerId, naics ("code|label"), geoLocation
    Counterparty  — unqCptyId ("routing-account"), cptyNamePrmry (sparse)

  Relationships:
    PAYS          PncCustomer → PncCustomer       (complete Jan 2024 – Dec 2025)
    CPTY_PAYS     Counterparty → PncCustomer      (ingested)
    PAYS_CPTY     PncCustomer → Counterparty      (upcoming)

  Edge properties (same across all three):
    time_key           Date     YYYY-MM-01 (Neo4j Date)
    totalTransAmt      Float    sum of all 5 rail amounts
    transFreq          Integer  transaction count
    totalTransAmtACH   Float
    totalTransAmtWire  Float
    totalTransAmtR2P   Float
    totalTransAmtCheck Float
    totalTransAmtDebit Float

Query strategy — 2 queries total (down from 4):

  QUERY_CUSTOMER_EDGES:
    Matches PAYS undirected: (c:PncCustomer)-[r:PAYS]-(t:PncCustomer)
    Uses startNode(r) = c to derive direction per row.
    One query covers both outgoing and incoming customer-to-customer flows.

  QUERY_COUNTERPARTY_EDGES:
    Matches (c:PncCustomer)-[r:CPTY_PAYS|PAYS_CPTY]-(cp:Counterparty)
    undirected with the | operator.
    Uses type(r) to derive direction: CPTY_PAYS → 'in', PAYS_CPTY → 'out'.
    One query covers both counterparty directions.
    PAYS_CPTY half is a no-op until that rel is ingested.

  time_key: pass ISO date strings ("2024-01-01") → date($param) in Cypher.
  Hub exclusion: inject hub customerId list as $hub_ids parameter.
"""

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase

# ── Schema constants ──────────────────────────────────────────────────────────

NODE_CUSTOMER      = "PncCustomer"
NODE_COUNTERPARTY  = "Counterparty"

REL_PAYS           = "PAYS"
REL_PAYS_CPTY      = "PAYS_CPTY"
REL_CPTY_PAYS      = "CPTY_PAYS"

PROP_CUSTOMER_ID   = "customerId"
PROP_NAICS         = "naics"
PROP_CPTY_ID       = "unqCptyId"
PROP_TIME_KEY      = "time_key"
PROP_TOTAL_AMT     = "totalTransAmt"
PROP_FREQ          = "transFreq"
PROP_AMT_ACH       = "totalTransAmtACH"
PROP_AMT_WIRE      = "totalTransAmtWire"
PROP_AMT_R2P       = "totalTransAmtR2P"
PROP_AMT_CHECK     = "totalTransAmtCheck"
PROP_AMT_DEBIT     = "totalTransAmtDebit"

RAILS = ["ACH", "Wire", "R2P", "Check", "Debit"]
RAIL_COLORS = {
    "ACH":   "#1D9E75",
    "Wire":  "#378ADD",
    "R2P":   "#EF9F27",
    "Check": "#888780",
    "Debit": "#D85A30",
}

# Maps wide DataFrame column names → rail display names (used in _melt_rails)
RAIL_AMT_COLS = {
    "amt_ach":   "ACH",
    "amt_wire":  "Wire",
    "amt_r2p":   "R2P",
    "amt_check": "Check",
    "amt_debit": "Debit",
}

NAICS_EXCLUDE_PREFIXES = ("-1|", "******|")

# ── NAICS loader query ────────────────────────────────────────────────────────

QUERY_ALL_NAICS = f"""
MATCH (c:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS} IS NOT NULL
  AND NOT c.{PROP_NAICS} STARTS WITH '-1|'
  AND NOT c.{PROP_NAICS} STARTS WITH '******|'
  AND trim(c.{PROP_NAICS}) <> ''
WITH c.{PROP_NAICS} AS naics_raw, count(c) AS n_customers
ORDER BY n_customers DESC
RETURN naics_raw, n_customers
"""

# ── Cohort size check ─────────────────────────────────────────────────────────

QUERY_COHORT_COUNT = f"""
MATCH (c:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS} STARTS WITH $naics_prefix
RETURN count(c) AS n
"""

# ── Query 1: Customer ↔ Customer (PAYS, both directions in one query) ─────────
#
# Undirected match — no arrow on the relationship.
# startNode(r) = c  →  c is the sender  →  direction = 'out'
# startNode(r) <> c →  c is the receiver →  direction = 'in'
#
# For NAICS-based cohort: filter on c.naics.
# For ID-based cohort:    filter on c.customerId IN $ids.
#
# Both hub nodes (c and t) are excluded via $hub_ids.
# counterpart_naics is returned so the NAICS graph section can use it.

QUERY_CUSTOMER_EDGES = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_PAYS}]-(t:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS} STARTS WITH $naics_prefix
  AND r.{PROP_TIME_KEY} >= date($time_start)
  AND r.{PROP_TIME_KEY} <= date($time_end)
  AND NOT c.{PROP_CUSTOMER_ID} IN $hub_ids
  AND NOT t.{PROP_CUSTOMER_ID} IN $hub_ids
RETURN
  c.{PROP_CUSTOMER_ID}              AS customerId,
  t.{PROP_CUSTOMER_ID}              AS counterpart_id,
  t.{PROP_NAICS}                    AS counterpart_naics,
  toString(r.{PROP_TIME_KEY})       AS time_key,
  r.{PROP_TOTAL_AMT}                AS total_amt,
  r.{PROP_FREQ}                     AS trans_freq,
  r.{PROP_AMT_ACH}                  AS amt_ach,
  r.{PROP_AMT_WIRE}                 AS amt_wire,
  r.{PROP_AMT_R2P}                  AS amt_r2p,
  r.{PROP_AMT_CHECK}                AS amt_check,
  r.{PROP_AMT_DEBIT}                AS amt_debit,
  CASE WHEN startNode(r) = c
       THEN 'out' ELSE 'in' END     AS direction,
  'customer'                        AS node_type
"""

QUERY_CUSTOMER_EDGES_IDS = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_PAYS}]-(t:{NODE_CUSTOMER})
WHERE c.{PROP_CUSTOMER_ID} IN $ids
  AND r.{PROP_TIME_KEY} >= date($time_start)
  AND r.{PROP_TIME_KEY} <= date($time_end)
  AND NOT c.{PROP_CUSTOMER_ID} IN $hub_ids
  AND NOT t.{PROP_CUSTOMER_ID} IN $hub_ids
RETURN
  c.{PROP_CUSTOMER_ID}              AS customerId,
  t.{PROP_CUSTOMER_ID}              AS counterpart_id,
  t.{PROP_NAICS}                    AS counterpart_naics,
  toString(r.{PROP_TIME_KEY})       AS time_key,
  r.{PROP_TOTAL_AMT}                AS total_amt,
  r.{PROP_FREQ}                     AS trans_freq,
  r.{PROP_AMT_ACH}                  AS amt_ach,
  r.{PROP_AMT_WIRE}                 AS amt_wire,
  r.{PROP_AMT_R2P}                  AS amt_r2p,
  r.{PROP_AMT_CHECK}                AS amt_check,
  r.{PROP_AMT_DEBIT}                AS amt_debit,
  CASE WHEN startNode(r) = c
       THEN 'out' ELSE 'in' END     AS direction,
  'customer'                        AS node_type
"""

# ── Query 2: Customer ↔ Counterparty (CPTY_PAYS|PAYS_CPTY, both directions) ──
#
# The | operator matches either relationship type in one pass.
# type(r) = 'CPTY_PAYS'  →  Counterparty paid PncCustomer  →  direction = 'in'
# type(r) = 'PAYS_CPTY'  →  PncCustomer paid Counterparty  →  direction = 'out'
#
# Until PAYS_CPTY is ingested, the PAYS_CPTY half simply returns zero rows —
# no guard needed; the | match is a no-op for a rel that doesn't exist yet.
#
# Counterparty nodes have no naics → counterpart_naics is null.

QUERY_COUNTERPARTY_EDGES = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_CPTY_PAYS}|{REL_PAYS_CPTY}]-(cp:{NODE_COUNTERPARTY})
WHERE c.{PROP_NAICS} STARTS WITH $naics_prefix
  AND r.{PROP_TIME_KEY} >= date($time_start)
  AND r.{PROP_TIME_KEY} <= date($time_end)
  AND NOT c.{PROP_CUSTOMER_ID} IN $hub_ids
RETURN
  c.{PROP_CUSTOMER_ID}              AS customerId,
  cp.{PROP_CPTY_ID}                 AS counterpart_id,
  null                              AS counterpart_naics,
  toString(r.{PROP_TIME_KEY})       AS time_key,
  r.{PROP_TOTAL_AMT}                AS total_amt,
  r.{PROP_FREQ}                     AS trans_freq,
  r.{PROP_AMT_ACH}                  AS amt_ach,
  r.{PROP_AMT_WIRE}                 AS amt_wire,
  r.{PROP_AMT_R2P}                  AS amt_r2p,
  r.{PROP_AMT_CHECK}                AS amt_check,
  r.{PROP_AMT_DEBIT}                AS amt_debit,
  CASE WHEN type(r) = '{REL_CPTY_PAYS}'
       THEN 'in' ELSE 'out' END     AS direction,
  'counterparty'                    AS node_type
"""

QUERY_COUNTERPARTY_EDGES_IDS = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_CPTY_PAYS}|{REL_PAYS_CPTY}]-(cp:{NODE_COUNTERPARTY})
WHERE c.{PROP_CUSTOMER_ID} IN $ids
  AND r.{PROP_TIME_KEY} >= date($time_start)
  AND r.{PROP_TIME_KEY} <= date($time_end)
  AND NOT c.{PROP_CUSTOMER_ID} IN $hub_ids
RETURN
  c.{PROP_CUSTOMER_ID}              AS customerId,
  cp.{PROP_CPTY_ID}                 AS counterpart_id,
  null                              AS counterpart_naics,
  toString(r.{PROP_TIME_KEY})       AS time_key,
  r.{PROP_TOTAL_AMT}                AS total_amt,
  r.{PROP_FREQ}                     AS trans_freq,
  r.{PROP_AMT_ACH}                  AS amt_ach,
  r.{PROP_AMT_WIRE}                 AS amt_wire,
  r.{PROP_AMT_R2P}                  AS amt_r2p,
  r.{PROP_AMT_CHECK}                AS amt_check,
  r.{PROP_AMT_DEBIT}                AS amt_debit,
  CASE WHEN type(r) = '{REL_CPTY_PAYS}'
       THEN 'in' ELSE 'out' END     AS direction,
  'counterparty'                    AS node_type
"""

# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_driver(uri: str, user: str, password: str):
    """Cached Neo4j driver — one connection pool for the app lifetime."""
    return GraphDatabase.driver(uri, auth=(user, password))


def run_query(driver, query: str, params: dict) -> pd.DataFrame:
    """Run a Cypher query and return results as a DataFrame."""
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.data() for r in result])


# ── NAICS helpers ─────────────────────────────────────────────────────────────

def parse_naics(raw: str) -> tuple:
    """
    Parse "561710|Exterminating and Pest Control Services" → ("561710", "Exterminating…")
    Returns (None, None) for dirty values.
    """
    if not raw or not isinstance(raw, str):
        return None, None
    raw = raw.strip()
    for prefix in NAICS_EXCLUDE_PREFIXES:
        if raw.startswith(prefix):
            return None, None
    parts = raw.split("|", 1)
    if len(parts) == 2:
        code, label = parts[0].strip(), parts[1].strip()
        return (code, label) if code and label else (None, None)
    return (raw, raw)


@st.cache_data(ttl=3600)
def load_naics_from_graph(_driver) -> pd.DataFrame:
    """
    Query all distinct NAICS values from PncCustomer nodes, parse them,
    and return a clean DataFrame for the dropdown.
    Cached 1 hour. Underscore prefix avoids Streamlit hashing the driver.

    Columns: naics_raw, naics_code, naics_label, n_customers,
             naics_prefix, dropdown_label
    """
    df = run_query(_driver, QUERY_ALL_NAICS, {})
    if df.empty:
        return pd.DataFrame(
            columns=["naics_raw", "naics_code", "naics_label",
                     "n_customers", "naics_prefix", "dropdown_label"]
        )
    parsed = df["naics_raw"].apply(
        lambda r: pd.Series(parse_naics(r), index=["naics_code", "naics_label"])
    )
    df = pd.concat([df, parsed], axis=1).dropna(subset=["naics_code", "naics_label"])
    df["naics_prefix"]   = df["naics_code"]
    df["dropdown_label"] = df["naics_code"] + " · " + df["naics_label"]
    return df.reset_index(drop=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

_EMPTY_LONG = pd.DataFrame(columns=[
    "customerId", "counterpart_id", "counterpart_naics",
    "time_key", "trans_freq", "direction", "node_type", "rail", "amount",
])


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types on a raw wide-format Neo4j result."""
    if df.empty:
        return df
    for col in ["total_amt", "amt_ach", "amt_wire", "amt_r2p", "amt_check", "amt_debit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "trans_freq" in df.columns:
        df["trans_freq"] = pd.to_numeric(df["trans_freq"], errors="coerce").fillna(0).astype(int)
    return df


def _melt_rails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unpivot wide edge DataFrame (5 rail amount columns) → long format.
    Drops zero-amount rows so only active rails produce edges.

    Output columns:
        customerId, counterpart_id, counterpart_naics, time_key,
        trans_freq, direction, node_type, rail, amount
    """
    if df.empty:
        return _EMPTY_LONG.copy()

    id_vars = ["customerId", "counterpart_id", "counterpart_naics",
               "time_key", "trans_freq", "direction", "node_type"]
    long = df.melt(
        id_vars=id_vars,
        value_vars=list(RAIL_AMT_COLS.keys()),
        var_name="rail_col",
        value_name="amount",
    )
    long["rail"]   = long["rail_col"].map(RAIL_AMT_COLS)
    long["amount"] = pd.to_numeric(long["amount"], errors="coerce").fillna(0.0)
    return long[long["amount"] > 0].drop(columns=["rail_col"]).reset_index(drop=True)


def _load(driver, q_customer, q_counterparty, params) -> pd.DataFrame:
    """Run the two queries, coerce, concatenate, and melt to long format."""
    frames = [
        run_query(driver, q_customer,      params),
        run_query(driver, q_counterparty,  params),
    ]
    wide = pd.concat([_coerce(f) for f in frames if not f.empty], ignore_index=True)
    if wide.empty:
        return _EMPTY_LONG.copy()
    return _melt_rails(wide)


# ── Public loaders ────────────────────────────────────────────────────────────

def load_cohort_by_naics(
    driver,
    naics_prefix: str,
    time_start: str,
    time_end: str,
    hub_ids: list = None,
) -> pd.DataFrame:
    """
    Load all payment edges for a NAICS cohort within [time_start, time_end].

    Args:
        naics_prefix: NAICS code string, e.g. "5221". Matched with STARTS WITH
                      against the "code|label" naics property.
        time_start:   ISO date string "YYYY-MM-DD" (first day of month).
        time_end:     ISO date string "YYYY-MM-DD" (first day of last month).
        hub_ids:      List of customerId strings to exclude. Fetch from
                      Impala precomputed degree table. Defaults to [].

    Returns long-format DataFrame:
        customerId, counterpart_id, counterpart_naics, time_key,
        trans_freq, direction ('in'|'out'), node_type ('customer'|'counterparty'),
        rail, amount

    One row per (edge × rail) where amount > 0.
    Direction and node_type filtering happens downstream in pandas — no re-query needed.
    """
    params = {
        "naics_prefix": naics_prefix,
        "time_start":   time_start,
        "time_end":     time_end,
        "hub_ids":      hub_ids or [],
    }
    return _load(driver, QUERY_CUSTOMER_EDGES, QUERY_COUNTERPARTY_EDGES, params)


def load_cohort_by_ids(
    driver,
    customer_ids: list,
    time_start: str,
    time_end: str,
    hub_ids: list = None,
) -> pd.DataFrame:
    """
    Load all payment edges for an explicit list of customerIds.
    Same return format as load_cohort_by_naics().
    Used when the analyst uploads a custom customer file.
    """
    params = {
        "ids":        customer_ids,
        "time_start": time_start,
        "time_end":   time_end,
        "hub_ids":    hub_ids or [],
    }
    return _load(driver, QUERY_CUSTOMER_EDGES_IDS, QUERY_COUNTERPARTY_EDGES_IDS, params)
