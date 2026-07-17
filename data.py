"""
data.py — PKG Explorer data access layer (v2: Impala-first).

Framework-free (no streamlit imports) so this module and logic.py can be
lifted into the FastAPI service at dev-team handoff.

Backends (PKG_BACKEND env var)
------------------------------
impala (default)   unified Hive tables via the bank-internal `dbi` helper
parquet            per-month files from pkg_pipeline / pkg_roles (local dev)

Tables (override via env)
-------------------------
PKG_METRICS_TABLE   default bdahd01p_dlcdi1_cdi_tm.cust_c2c_metrics
PKG_ROLES_TABLE     default bdahd01p_dlcdi1_cdi_tm.cust_c2c_roles

Interface contract (both backends):
    months(version)                      -> list[str]
    node_month(version, time_key)        -> DataFrame (NODE_COLS)
    roles_month(version, time_key)       -> DataFrame (all role cols)
    customer_history(node, version)      -> metrics+roles joined, all months
    display_name(node)                   -> str

Every Impala query is filtered on version + time_key (or node) and
column-pruned — the app never pulls a full panel.
"""

from __future__ import annotations

import glob
import os
import re
from functools import lru_cache

import pandas as pd

METRICS_DIR = os.environ.get("PKG_METRICS_DIR", "../metrics")
BACKEND = os.environ.get("PKG_BACKEND", "impala")
METRICS_TABLE = os.environ.get("PKG_METRICS_TABLE",
                               "bdahd01p_dlcdi1_cdi_tm.cust_c2c_metrics")
ROLES_TABLE = os.environ.get("PKG_ROLES_TABLE",
                             "bdahd01p_dlcdi1_cdi_tm.cust_c2c_roles")

# Prototyping row cap: keep only the top-N customers BY STRENGTH per
# (version, month). 0 disables. Roles are semi-joined against the same
# top-N so metrics and roles always cover identical customers.
ROW_LIMIT = int(os.environ.get("PKG_ROW_LIMIT", "50000"))

NODE_COLS = [
    "time_key", "version", "node", "strength", "in_strength", "out_strength",
    "net_flow", "flow_ratio", "throughflow", "degree", "in_degree",
    "out_degree", "hhi_in", "hhi_out", "top1_in_share",
    "lost_payer_amount_share", "new_payer_amount_share", "payer_jaccard",
    "n_payer_lost", "n_payer_new", "top_payer_same", "hub_in_share",
    "months_active", "naics2", "naics4",
]

_SAFE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _lit(value: str) -> str:
    """Quote a value for SQL after strict whitelist validation. The dbi
    helper has no parameter binding, so nothing outside [A-Za-z0-9_-.]
    is ever interpolated."""
    if not _SAFE.match(str(value)):
        raise ValueError(f"unsafe literal rejected: {value!r}")
    return f"'{value}'"


# ---------------------------------------------------------------------------
# Impala backend
# ---------------------------------------------------------------------------

class ImpalaSource:
    """Unified Hive tables via the bank-internal `dbi` helper."""

    DSN = "DSN=bdpimp04-impala;"
    POOL = "root.CIB-AMG_Impala"

    def __init__(self):
        try:
            import dbi  # bank-internal
        except ImportError as e:
            raise ImportError(
                "the `dbi` helper is not importable — run inside the bank "
                "environment or set PKG_BACKEND=parquet for local dev"
            ) from e
        self._dbi = dbi

    def _q(self, sql: str) -> pd.DataFrame:
        return self._dbi.db_get_query(
            sql,
            dsn=self.DSN,
            pool=self.POOL,
            conn_options={"SocketTimeout": 0},
        )

    def months(self, version: str) -> list[str]:
        df = self._q(
            f"SELECT DISTINCT time_key FROM {METRICS_TABLE} "
            f"WHERE version = {_lit(version)}")
        return sorted(df["time_key"].astype(str))

    def node_month(self, version: str, time_key: str) -> pd.DataFrame:
        cols = ", ".join(NODE_COLS)
        sql = (f"SELECT {cols} FROM {METRICS_TABLE} "
               f"WHERE version = {_lit(version)} "
               f"AND time_key = {_lit(time_key)}")
        if ROW_LIMIT > 0:
            sql += f" ORDER BY strength DESC LIMIT {ROW_LIMIT}"
        return self._q(sql)

    def roles_month(self, version: str, time_key: str) -> pd.DataFrame:
        if ROW_LIMIT > 0:
            # semi-join so roles cover EXACTLY the same top-N customers
            return self._q(
                f"SELECT r.* FROM {ROLES_TABLE} r "
                f"JOIN (SELECT node FROM {METRICS_TABLE} "
                f"      WHERE version = {_lit(version)} "
                f"      AND time_key = {_lit(time_key)} "
                f"      ORDER BY strength DESC LIMIT {ROW_LIMIT}) t "
                f"ON r.node = t.node "
                f"WHERE r.version = {_lit(version)} "
                f"AND r.time_key = {_lit(time_key)}")
        return self._q(
            f"SELECT * FROM {ROLES_TABLE} "
            f"WHERE version = {_lit(version)} "
            f"AND time_key = {_lit(time_key)}")

    def customer_history(self, node: str, version: str) -> pd.DataFrame:
        cols = ", ".join(NODE_COLS)
        n = self._q(
            f"SELECT {cols} FROM {METRICS_TABLE} "
            f"WHERE version = {_lit(version)} AND node = {_lit(node)}")
        r = self._q(
            f"SELECT * FROM {ROLES_TABLE} "
            f"WHERE version = {_lit(version)} AND node = {_lit(node)}")
        return _join_history(n, r)


# ---------------------------------------------------------------------------
# Parquet backend (local dev)
# ---------------------------------------------------------------------------

class ParquetSource:
    def __init__(self, metrics_dir: str = METRICS_DIR):
        self.dir = metrics_dir

    def _read(self, sub: str, pattern: str, columns=None) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.dir, sub, pattern)))
        if not files:
            raise FileNotFoundError(f"no {pattern} under {self.dir}/{sub}")
        return pd.concat(
            (pd.read_parquet(f, columns=columns) for f in files),
            ignore_index=True)

    def months(self, version: str) -> list[str]:
        files = sorted(glob.glob(os.path.join(self.dir, "node",
                                              "node_*.parquet")))
        return [re.search(r"node_(\d{4}-\d{2})", f).group(1) for f in files]

    def node_month(self, version: str, time_key: str) -> pd.DataFrame:
        f = os.path.join(self.dir, "node", f"node_{time_key}.parquet")
        df = pd.read_parquet(f, columns=NODE_COLS)
        df = df[df["version"] == version]
        if ROW_LIMIT > 0:
            df = df.nlargest(ROW_LIMIT, "strength")
        return df.reset_index(drop=True)

    def roles_month(self, version: str, time_key: str) -> pd.DataFrame:
        f = os.path.join(self.dir, "roles", f"roles_{time_key}.parquet")
        df = pd.read_parquet(f)
        df = df[df["version"] == version]
        if ROW_LIMIT > 0:
            keep = set(self.node_month(version, time_key)["node"])
            df = df[df["node"].isin(keep)]
        return df.reset_index(drop=True)

    def customer_history(self, node: str, version: str) -> pd.DataFrame:
        n = self._read("node", "node_*.parquet", NODE_COLS)
        n = n[(n["node"] == node) & (n["version"] == version)]
        r = self._read("roles", "roles_*.parquet")
        r = r[(r["node"] == node) & (r["version"] == version)]
        return _join_history(n, r)


def _join_history(n: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ("naics2", "naics3", "naics4") if c in r.columns]
    return (n.merge(r.drop(columns=drop),
                    on=["time_key", "version", "node"],
                    how="outer", suffixes=("", "_r"))
            .sort_values("time_key").reset_index(drop=True))


# ---------------------------------------------------------------------------
# module-level cached accessors (interface used by logic.py / app.py)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_source():
    return ImpalaSource() if BACKEND == "impala" else ParquetSource()


@lru_cache(maxsize=8)
def months(version: str = "V0") -> tuple:
    return tuple(get_source().months(version))


@lru_cache(maxsize=32)
def node_month(version: str, time_key: str) -> pd.DataFrame:
    return get_source().node_month(version, time_key)


@lru_cache(maxsize=32)
def roles_month(version: str, time_key: str) -> pd.DataFrame:
    return get_source().roles_month(version, time_key)


def customer_history(node: str, version: str) -> pd.DataFrame:
    return get_source().customer_history(node, version)


def display_name(node: str) -> str:
    """Hook: join customer names from your dimension table here."""
    return node
