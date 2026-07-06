"""
pkg_supergraph.py
=================
Collapse a monthly PKG snapshot into a supergraph where each group
(Louvain community, full NAICS code, or 2-digit NAICS sector) becomes one
node. Intra-group payments become SELF-LOOP edges; inter-group payments
become directed edges. amount and volume are summed.

Output CSVs reuse the EXACT base snapshot schema

    source,source_name,source_naics,amount,volume,dest,dest_name,dest_naics

so the entire node/graph metric suite re-runs on supergraphs unchanged.
Files land in ../snapshot/ as  super_{partition}_{version}_{YYYY-MM}.csv
Amount conservation is asserted after every build.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_supergraph")

SCHEMA = ["source", "source_name", "source_naics", "amount", "volume",
          "dest", "dest_name", "dest_naics"]


def _group_labels(edges: pd.DataFrame, partition: pd.DataFrame | None,
                  mode: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return (source_group, dest_group, group_meta[group, name, naics])."""
    if mode == "community":
        if partition is None:
            raise ValueError("community mode requires a partition frame")
        pmap = partition.set_index("node")["community_id"]
        gs = pmap.reindex(edges["source"]).to_numpy()
        gd = pmap.reindex(edges["dest"]).to_numpy()
        # group meta: dominant NAICS of members (by node count)
        nodes = pd.concat([
            edges[["source", "source_naics"]].rename(
                columns={"source": "node", "source_naics": "naics"}),
            edges[["dest", "dest_naics"]].rename(
                columns={"dest": "node", "dest_naics": "naics"}),
        ]).drop_duplicates("node")
        nodes["group"] = pmap.reindex(nodes["node"]).to_numpy()
        dom = (nodes.dropna(subset=["group"])
               .groupby(["group", "naics"]).size().rename("n").reset_index()
               .sort_values("n", ascending=False)
               .drop_duplicates("group"))
        meta = pd.DataFrame({
            "group": dom["group"],
            "name": "COMM_" + dom["group"].astype("Int64").astype(str),
            "naics": dom["naics"],
        })
        return pd.Series(gs), pd.Series(gd), meta

    if mode in ("naics", "naics2"):
        gs = edges["source_naics"].astype(str)
        gd = edges["dest_naics"].astype(str)
        if mode == "naics2":
            gs = gs.str.slice(0, 2)
            gd = gd.str.slice(0, 2)
        groups = pd.Index(pd.unique(pd.concat([gs, gd], ignore_index=True)))
        meta = pd.DataFrame({"group": groups,
                             "name": ("SECTOR_" if mode == "naics2"
                                      else "NAICS_") + groups.astype(str),
                             "naics": groups})
        return gs.reset_index(drop=True), gd.reset_index(drop=True), meta

    raise ValueError(f"unknown mode {mode!r}")


def build_supergraph(edges: pd.DataFrame, mode: str,
                     partition: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collapse `edges` into a supergraph edge list in the base schema.
    Intra-group flow -> self-loop (source == dest)."""
    gs, gd, meta = _group_labels(edges, partition, mode)
    df = pd.DataFrame({"gs": gs.to_numpy(), "gd": gd.to_numpy(),
                       "amount": edges["amount"].to_numpy(float),
                       "volume": edges["volume"].to_numpy(float)})
    df = df.dropna(subset=["gs", "gd"])
    agg = df.groupby(["gs", "gd"], as_index=False)[["amount", "volume"]].sum()

    m = meta.set_index("group")
    out = pd.DataFrame({
        "source": agg["gs"],
        "source_name": m["name"].reindex(agg["gs"]).to_numpy(),
        "source_naics": m["naics"].reindex(agg["gs"]).to_numpy(),
        "amount": agg["amount"],
        "volume": agg["volume"],
        "dest": agg["gd"],
        "dest_name": m["name"].reindex(agg["gd"]).to_numpy(),
        "dest_naics": m["naics"].reindex(agg["gd"]).to_numpy(),
    })[SCHEMA]

    # --- conservation smoke test -------------------------------------------
    src_total = df["amount"].sum()
    if not np.isclose(out["amount"].sum(), src_total, rtol=1e-9):
        raise AssertionError(
            f"amount conservation violated: {out['amount'].sum()} vs {src_total}")
    return out


def write_supergraph(edges: pd.DataFrame, mode: str, version: str,
                     time_key: str, out_dir: str = "../snapshot",
                     partition: pd.DataFrame | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sg = build_supergraph(edges, mode, partition)
    path = os.path.join(out_dir, f"super_{mode}_{version}_{time_key}.csv")
    sg.to_csv(path, index=False)
    log.info("wrote %s  (%d supernodes, %d edges, %.0f amount)",
             path, pd.unique(pd.concat([sg['source'], sg['dest']])).size,
             len(sg), sg["amount"].sum())
    return path


def sector_flow_matrix(supergraph: pd.DataFrame) -> pd.DataFrame:
    """Pivot a NAICS supergraph edge list into a square sector-to-sector
    input-output amount matrix (rows=payer, cols=payee; diagonal=intra)."""
    return (supergraph.pivot_table(index="source", columns="dest",
                                   values="amount", aggfunc="sum",
                                   fill_value=0.0))
