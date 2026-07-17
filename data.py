"""
data.py — PKG Explorer data access layer.

Framework-free (no streamlit imports) so this module and logic.py can be
lifted into the FastAPI service at dev-team handoff.

Backends
--------
PKG_BACKEND=parquet (default)   reads the per-month files produced by
                                pkg_pipeline / pkg_roles under PKG_METRICS_DIR
PKG_BACKEND=impala              queries the unified Hive tables; fill in
                                the two SQL methods in ImpalaSource with
                                your table names (one metrics table, one
                                roles table, both keyed by
                                time_key/version/node).

All loaders return pandas DataFrames with the schemas documented in
PKG_MONTHLY_METRICS_MANIFEST.md and PKG_ROLES_MANIFEST.md.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache

import pandas as pd

METRICS_DIR = os.environ.get("PKG_METRICS_DIR", "../metrics")
BACKEND = os.environ.get("PKG_BACKEND", "parquet")

# columns the app actually needs — keep the panel slim in memory
NODE_COLS = [
    "time_key", "version", "node", "strength", "in_strength", "out_strength",
    "net_flow", "flow_ratio", "throughflow", "degree", "in_degree",
    "out_degree", "hhi_in", "hhi_out", "top1_in_share",
    "lost_payer_amount_share", "new_payer_amount_share", "payer_jaccard",
    "n_payer_lost", "n_payer_new", "top_payer_same", "hub_in_share",
    "months_active", "naics2", "naics4",
]
ROLE_COLS = None  # roles files are already slim; take all


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

    def node_panel(self, version: str) -> pd.DataFrame:
        df = self._read("node", "node_*.parquet", NODE_COLS)
        return df[df["version"] == version].reset_index(drop=True)

    def roles_panel(self, version: str) -> pd.DataFrame:
        df = self._read("roles", "roles_*.parquet", ROLE_COLS)
        return df[df["version"] == version].reset_index(drop=True)

    def customer_history(self, node: str, version: str) -> pd.DataFrame:
        """Full metric + role history for one customer (deep dive)."""
        n = self.node_panel(version)
        r = self.roles_panel(version)
        h = (n[n["node"] == node]
             .merge(r[r["node"] == node],
                    on=["time_key", "version", "node"],
                    how="outer", suffixes=("", "_r"))
             .sort_values("time_key").reset_index(drop=True))
        return h

    def graph_panel(self) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self.dir, "graph",
                                              "graph_*.csv")))
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


class ImpalaSource:
    """Production backend against the unified Hive tables.

    Wire up with impyla/pyodbc and set:
        PKG_BACKEND=impala
        PKG_METRICS_TABLE=<db.customer_metrics>
        PKG_ROLES_TABLE=<db.customer_roles>
    Keep queries column-pruned (NODE_COLS) and always filtered on
    version + time_key or node — never SELECT * on the full table.
    """

    def __init__(self):
        raise NotImplementedError(
            "fill in Impala connection + the two SELECTs for your "
            "unified tables; interface must match ParquetSource")


def get_source():
    return ImpalaSource() if BACKEND == "impala" else ParquetSource()


# ---- cached module-level accessors (lru: keyed by version) ----------------

@lru_cache(maxsize=8)
def node_panel(version: str) -> pd.DataFrame:
    return get_source().node_panel(version)


@lru_cache(maxsize=8)
def roles_panel(version: str) -> pd.DataFrame:
    return get_source().roles_panel(version)


@lru_cache(maxsize=1)
def graph_panel() -> pd.DataFrame:
    return get_source().graph_panel()


def customer_history(node: str, version: str) -> pd.DataFrame:
    return get_source().customer_history(node, version)


def months(version: str = "V0") -> list[str]:
    return sorted(node_panel(version)["time_key"].unique())


def display_name(node: str) -> str:
    """Hook: join customer names from your Hive dimension table here.
    Metrics parquet intentionally carries only the node id."""
    return node
