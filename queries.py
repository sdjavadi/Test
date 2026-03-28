"""
queries.py — Neo4j Cypher queries for the Rail Shift Group Analyzer.
PNC Treasury Management · Payment Knowledge Network

Schema assumptions — edit the constants block if your PKN differs:

  Nodes:
    Customer      — mdmId (str), naicsCode (str), geoLocation (str)
    Counterparty  — cptyId (str), cptyNamePrmry (str)

  Relationships and direction:
    PAYS          Customer  → Customer       (customer-to-customer payments)
    CPTY_PAYS     Counterparty → Customer    (counterparty pays customer — inbound)
    PAYS_CPTY     Customer  → Counterparty   (customer pays counterparty — outbound, upcoming)

  Relationship properties (same across all three rel types):
    rail          str   "ACH" | "Wire" | "R2P" | "Check"
    time_key      int   YYYYMM  e.g. 202401
    totalAmount   float sum of payment amounts in that month/rail bucket
    txnCount      int   number of transactions in that bucket

counterpart_naics column:
    Populated for PAYS edges (Customer ↔ Customer).
    NULL for CPTY_PAYS / PAYS_CPTY edges (Counterparty nodes have no NAICS).
    The app uses this column to build the NAICS neighbor graph.
"""

import pandas as pd
import streamlit as st
from neo4j import GraphDatabase

# ── Schema constants ──────────────────────────────────────────────────────────

NODE_CUSTOMER      = "Customer"
NODE_COUNTERPARTY  = "Counterparty"

REL_PAYS           = "PAYS"
REL_PAYS_CPTY      = "PAYS_CPTY"
REL_CPTY_PAYS      = "CPTY_PAYS"

PROP_MDM_ID        = "mdmId"
PROP_NAICS         = "naicsCode"
PROP_CPTY_ID       = "cptyId"
PROP_RAIL          = "rail"
PROP_TIME_KEY      = "time_key"
PROP_AMOUNT        = "totalAmount"
PROP_TXN_COUNT     = "txnCount"

RAILS = ["ACH", "Wire", "R2P", "Check"]

# ── Fast cohort size check ────────────────────────────────────────────────────

QUERY_COHORT_COUNT = f"""
MATCH (c:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS} = $naics
RETURN count(c) AS n
"""

# ── NAICS-based queries ───────────────────────────────────────────────────────
# counterpart_naics is returned for PAYS edges so the app can build the
# NAICS neighbor graph without a second round-trip.

QUERY_PAYS_OUT = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_PAYS}]->(t:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS}    = $naics
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  t.{PROP_MDM_ID}    AS counterpart_id,
  t.{PROP_NAICS}     AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'out'              AS direction,
  'customer'         AS node_type
"""

QUERY_PAYS_IN = f"""
MATCH (s:{NODE_CUSTOMER})-[r:{REL_PAYS}]->(c:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS}    = $naics
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  s.{PROP_MDM_ID}    AS counterpart_id,
  s.{PROP_NAICS}     AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'in'               AS direction,
  'customer'         AS node_type
"""

# CPTY_PAYS has no naicsCode — counterpart_naics is NULL (None in pandas)
QUERY_CPTY_PAYS_IN = f"""
MATCH (cp:{NODE_COUNTERPARTY})-[r:{REL_CPTY_PAYS}]->(c:{NODE_CUSTOMER})
WHERE c.{PROP_NAICS}    = $naics
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  cp.{PROP_CPTY_ID}  AS counterpart_id,
  null               AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'in'               AS direction,
  'counterparty'     AS node_type
"""

QUERY_PAYS_CPTY_OUT = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_PAYS_CPTY}]->(cp:{NODE_COUNTERPARTY})
WHERE c.{PROP_NAICS}    = $naics
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  cp.{PROP_CPTY_ID}  AS counterpart_id,
  null               AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'out'              AS direction,
  'counterparty'     AS node_type
"""

# ── mdmId-list-based queries (file upload path) ───────────────────────────────

QUERY_PAYS_OUT_IDS = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_PAYS}]->(t:{NODE_CUSTOMER})
WHERE c.{PROP_MDM_ID}   IN $ids
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  t.{PROP_MDM_ID}    AS counterpart_id,
  t.{PROP_NAICS}     AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'out'              AS direction,
  'customer'         AS node_type
"""

QUERY_PAYS_IN_IDS = f"""
MATCH (s:{NODE_CUSTOMER})-[r:{REL_PAYS}]->(c:{NODE_CUSTOMER})
WHERE c.{PROP_MDM_ID}   IN $ids
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  s.{PROP_MDM_ID}    AS counterpart_id,
  s.{PROP_NAICS}     AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'in'               AS direction,
  'customer'         AS node_type
"""

QUERY_CPTY_PAYS_IN_IDS = f"""
MATCH (cp:{NODE_COUNTERPARTY})-[r:{REL_CPTY_PAYS}]->(c:{NODE_CUSTOMER})
WHERE c.{PROP_MDM_ID}   IN $ids
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  cp.{PROP_CPTY_ID}  AS counterpart_id,
  null               AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'in'               AS direction,
  'counterparty'     AS node_type
"""

QUERY_PAYS_CPTY_OUT_IDS = f"""
MATCH (c:{NODE_CUSTOMER})-[r:{REL_PAYS_CPTY}]->(cp:{NODE_COUNTERPARTY})
WHERE c.{PROP_MDM_ID}   IN $ids
  AND r.{PROP_TIME_KEY} >= $time_start
  AND r.{PROP_TIME_KEY} <= $time_end
RETURN
  c.{PROP_MDM_ID}    AS mdmId,
  cp.{PROP_CPTY_ID}  AS counterpart_id,
  null               AS counterpart_naics,
  r.{PROP_RAIL}      AS rail,
  r.{PROP_TIME_KEY}  AS time_key,
  r.{PROP_AMOUNT}    AS amount,
  r.{PROP_TXN_COUNT} AS txn_count,
  'out'              AS direction,
  'counterparty'     AS node_type
"""

# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_driver(uri: str, user: str, password: str):
    """Cached Neo4j driver — one connection pool for the Streamlit app lifetime."""
    return GraphDatabase.driver(uri, auth=(user, password))


def run_query(driver, query: str, params: dict) -> pd.DataFrame:
    """Run a Cypher query and return results as a DataFrame."""
    with driver.session() as session:
        result = session.run(query, params)
        return pd.DataFrame([r.data() for r in result])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types on a raw Neo4j result frame."""
    if df.empty:
        return df
    df["amount"]    = pd.to_numeric(df["amount"],    errors="coerce").fillna(0.0)
    df["txn_count"] = pd.to_numeric(df["txn_count"], errors="coerce").fillna(0).astype(int)
    df["time_key"]  = df["time_key"].astype(int)
    df["rail"]      = df["rail"].fillna("Unknown")
    # counterpart_naics may be None for counterparty rows — leave as-is (NaN in pandas)
    return df


_EMPTY = pd.DataFrame(
    columns=["mdmId", "counterpart_id", "counterpart_naics", "rail",
             "time_key", "amount", "txn_count", "direction", "node_type"]
)


def _concat(frames: list) -> pd.DataFrame:
    non_empty = [f for f in frames if not f.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else _EMPTY.copy()


# ── Public loaders ────────────────────────────────────────────────────────────

def load_cohort_by_naics(
    driver,
    naics: str,
    time_start: int,
    time_end: int,
    include_pays_cpty: bool = False,
) -> pd.DataFrame:
    """
    Load all payment edges for a NAICS cohort within [time_start, time_end].

    time_start / time_end are YYYYMM integers (e.g. 202401, 202512).
    The time window is pushed into Cypher — only the requested months are fetched.

    Direction and node_type filtering happens downstream in pandas (app.py).

    Returns a DataFrame with columns:
        mdmId, counterpart_id, counterpart_naics, rail, time_key,
        amount, txn_count, direction, node_type
    """
    p = {"naics": naics, "time_start": time_start, "time_end": time_end}

    frames = [
        run_query(driver, QUERY_PAYS_OUT,     p),
        run_query(driver, QUERY_PAYS_IN,      p),
        run_query(driver, QUERY_CPTY_PAYS_IN, p),
    ]
    if include_pays_cpty:
        frames.append(run_query(driver, QUERY_PAYS_CPTY_OUT, p))

    return _coerce(_concat(frames))


def load_cohort_by_ids(
    driver,
    mdm_ids: list,
    time_start: int,
    time_end: int,
    include_pays_cpty: bool = False,
) -> pd.DataFrame:
    """
    Load all payment edges for an explicit list of mdmIds within [time_start, time_end].
    Used when the analyst uploads a custom customer file instead of selecting NAICS.
    """
    p = {"ids": mdm_ids, "time_start": time_start, "time_end": time_end}

    frames = [
        run_query(driver, QUERY_PAYS_OUT_IDS,     p),
        run_query(driver, QUERY_PAYS_IN_IDS,      p),
        run_query(driver, QUERY_CPTY_PAYS_IN_IDS, p),
    ]
    if include_pays_cpty:
        frames.append(run_query(driver, QUERY_PAYS_CPTY_OUT_IDS, p))

    return _coerce(_concat(frames))
