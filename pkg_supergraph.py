"""
pkg_supergraph.py
=================
Community-as-node and NAICS-as-node supergraph builders for PKG monthly
snapshots. Intra-group payments become self-loops on the group node.

Outputs per snapshot (CSV):
    community_supergraph_edges.csv   src_comm, dst_comm, amount, volume,
                                     edge_count, distinct_pairs, is_self
    community_supergraph_nodes.csv   comm_id, n_nodes, internal_amount,
                                     internal_volume, ext_out_amount,
                                     ext_in_amount, mixing_ratio_w,
                                     mixing_ratio_u, density_directed,
                                     conductance, net_flow,
                                     naics_top_code, naics_top_share,
                                     naics_entropy, strength_hhi_internal,
                                     top_node_share
    naics_supergraph_edges.csv       src_naics, dst_naics, amount, volume,
                                     edge_count, distinct_pairs, is_self
    naics_supergraph_nodes.csv       naics, n_nodes, internal_amount,
                                     internal_volume, ext_out_amount,
                                     ext_in_amount, mixing_ratio_w,
                                     self_loop_share, net_flow

GPU-first (cuDF), pandas fallback. Weight = amount; volume descriptive.
The supergraph edge CSVs are themselves valid PKG-format edge lists
(source=src_*, dest=dst_*), so the full node-metric suite in
pkg_custom_metrics.py (PageRank, weighted HITS, trophic levels, ...) can be
re-run directly on the supergraphs — including self-loop stripping, which the
loaders below keep as explicit `is_self` rows instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import cudf
    GPU = True
except Exception:
    cudf = None
    GPU = False

log = logging.getLogger("pkg_supergraph")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

NAICS_UNKNOWN_TOKENS = {"-1|UNKNOWN", "-1", "******", "", "nan", "None", "NULL"}


def _host(df):
    return df.to_pandas() if GPU and hasattr(df, "to_pandas") else df


def clean_naics(series: pd.Series, level: int | None = None) -> pd.Series:
    """
    Normalize NAICS codes: strip, map unknown tokens -> 'UNKNOWN',
    optionally truncate to `level` digits (e.g. 2 for sector supergraph).
    """
    s = series.astype("string").str.strip()
    s = s.where(~s.isin(NAICS_UNKNOWN_TOKENS) & s.notna(), "UNKNOWN")
    if level is not None:
        known = s != "UNKNOWN"
        s = s.where(~known, s.str.slice(0, level))
    return s


# --------------------------------------------------------------------------- #
# Core aggregation
# --------------------------------------------------------------------------- #
def _aggregate_supergraph(e: pd.DataFrame, src_grp: str, dst_grp: str,
                          out_src: str, out_dst: str) -> pd.DataFrame:
    """Group edge frame by (src group, dst group); intra-group -> is_self."""
    agg = (
        e.groupby([src_grp, dst_grp], dropna=False, observed=True)
        .agg(
            amount=("amount", "sum"),
            volume=("volume", "sum"),
            edge_count=("amount", "size"),
            distinct_pairs=("pair", "nunique"),
        )
        .reset_index()
        .rename(columns={src_grp: out_src, dst_grp: out_dst})
    )
    agg["is_self"] = agg[out_src] == agg[out_dst]
    return agg.sort_values("amount", ascending=False).reset_index(drop=True)


def _node_side_sums(edges: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    """internal / external in / external out amounts + edge counts per group."""
    internal = edges[edges["is_self"]]
    ext = edges[~edges["is_self"]]
    parts = {
        "internal_amount": internal.set_index(src)["amount"],
        "internal_volume": internal.set_index(src)["volume"],
        "internal_edges": internal.set_index(src)["edge_count"],
        "ext_out_amount": ext.groupby(src)["amount"].sum(),
        "ext_in_amount": ext.groupby(dst)["amount"].sum(),
        "ext_out_edges": ext.groupby(src)["edge_count"].sum(),
        "ext_in_edges": ext.groupby(dst)["edge_count"].sum(),
    }
    df = pd.DataFrame(parts).fillna(0.0)
    df.index.name = "group"
    return df.reset_index()


# --------------------------------------------------------------------------- #
# Community supergraph
# --------------------------------------------------------------------------- #
def build_community_supergraph(
    edges,
    membership,
    out_edges_csv: str = "community_supergraph_edges.csv",
    out_nodes_csv: str = "community_supergraph_nodes.csv",
    unassigned_label: str = "UNASSIGNED",
):
    """
    edges: PKG edge frame (source, dest, amount, volume[, source_naics,
           dest_naics]); self-loops already removed at the customer level.
    membership: [node, community_id] from matched Louvain.
    Returns (edges_df, nodes_df); also writes both CSVs.
    """
    e = _host(edges).copy()
    m = _host(membership).set_index("node")["community_id"]
    e["c_src"] = e["source"].map(m).fillna(unassigned_label)
    e["c_dst"] = e["dest"].map(m).fillna(unassigned_label)
    e["pair"] = e["source"].astype(str) + "\x1f" + e["dest"].astype(str)
    if "volume" not in e:
        e["volume"] = 1

    super_edges = _aggregate_supergraph(e, "c_src", "c_dst", "src_comm", "dst_comm")

    # ---- node table ----
    sides = _node_side_sums(super_edges, "src_comm", "dst_comm").rename(
        columns={"group": "comm_id"}
    )
    sizes = m.groupby(m).size().rename("n_nodes").rename_axis("comm_id").reset_index()
    nodes = sizes.merge(sides, on="comm_id", how="outer").fillna(0.0)

    tot_amt = nodes["internal_amount"] + nodes["ext_out_amount"] + nodes["ext_in_amount"]
    tot_edg = nodes["internal_edges"] + nodes["ext_out_edges"] + nodes["ext_in_edges"]
    nodes["mixing_ratio_w"] = np.where(tot_amt > 0, nodes["internal_amount"] / tot_amt, np.nan)
    nodes["mixing_ratio_u"] = np.where(tot_edg > 0, nodes["internal_edges"] / tot_edg, np.nan)
    nn = nodes["n_nodes"]
    denom = np.where(nn > 1, nn * (nn - 1), np.nan)
    nodes["density_directed"] = nodes["internal_edges"] / denom
    nodes["net_flow"] = nodes["ext_in_amount"] - nodes["ext_out_amount"]

    vol_c = 2 * nodes["internal_amount"] + nodes["ext_out_amount"] + nodes["ext_in_amount"]
    total_vol = float(2 * e["amount"].sum())
    cut = nodes["ext_out_amount"] + nodes["ext_in_amount"]
    nodes["conductance"] = cut / np.maximum(np.minimum(vol_c, total_vol - vol_c), 1e-300)

    # NAICS composition of members (from edge-observed NAICS, majority per node)
    if {"source_naics", "dest_naics"} <= set(e.columns):
        n1 = e[["source", "source_naics"]].rename(
            columns={"source": "node", "source_naics": "naics"})
        n2 = e[["dest", "dest_naics"]].rename(columns={"dest": "node", "dest_naics": "naics"})
        node_naics = pd.concat([n1, n2], ignore_index=True)
        node_naics["naics"] = clean_naics(node_naics["naics"])
        node_naics = (
            node_naics.groupby(["node", "naics"]).size().reset_index(name="k")
            .sort_values("k", ascending=False).drop_duplicates("node")
        )
        node_naics["comm_id"] = node_naics["node"].map(m).fillna(unassigned_label)
        comp = node_naics.groupby(["comm_id", "naics"]).size().rename("cnt").reset_index()
        comp["tot"] = comp.groupby("comm_id")["cnt"].transform("sum")
        comp["p"] = comp["cnt"] / comp["tot"]
        ent = comp.assign(h=lambda d: -d["p"] * np.log(d["p"])) \
                  .groupby("comm_id")["h"].sum().rename("naics_entropy")
        top = comp.sort_values("p", ascending=False).drop_duplicates("comm_id") \
                  .set_index("comm_id")[["naics", "p"]] \
                  .rename(columns={"naics": "naics_top_code", "p": "naics_top_share"})
        nodes = nodes.merge(ent, on="comm_id", how="left") \
                     .merge(top, on="comm_id", how="left")
    else:
        nodes["naics_entropy"] = np.nan
        nodes["naics_top_code"] = pd.NA
        nodes["naics_top_share"] = np.nan

    # Internal concentration: HHI / top-node share of member strengths (intra edges)
    intra = e[e["c_src"] == e["c_dst"]]
    ns = pd.concat(
        [intra.groupby(["c_src", "source"])["amount"].sum().rename("s"),
         intra.groupby(["c_src", "dest"])["amount"].sum().rename("s")]
    )
    ns.index.names = ["comm_id", "node"]
    ns = ns.groupby(level=[0, 1]).sum().reset_index()
    ns["tot"] = ns.groupby("comm_id")["s"].transform("sum")
    ns["sh2"] = (ns["s"] / ns["tot"]) ** 2
    hhi = ns.groupby("comm_id")["sh2"].sum().rename("strength_hhi_internal")
    tshare = (ns.sort_values("s", ascending=False).drop_duplicates("comm_id")
                .set_index("comm_id")
                .assign(top_node_share=lambda d: d["s"] / d["tot"])["top_node_share"])
    nodes = nodes.merge(hhi, on="comm_id", how="left").merge(tshare, on="comm_id", how="left")

    cols_edges = ["src_comm", "dst_comm", "amount", "volume",
                  "edge_count", "distinct_pairs", "is_self"]
    cols_nodes = ["comm_id", "n_nodes", "internal_amount", "internal_volume",
                  "ext_out_amount", "ext_in_amount", "mixing_ratio_w",
                  "mixing_ratio_u", "density_directed", "conductance", "net_flow",
                  "naics_top_code", "naics_top_share", "naics_entropy",
                  "strength_hhi_internal", "top_node_share"]
    super_edges[cols_edges].to_csv(out_edges_csv, index=False)
    nodes[cols_nodes].to_csv(out_nodes_csv, index=False)
    log.info("community supergraph: %d comms, %d edges -> %s / %s",
             nodes.shape[0], super_edges.shape[0], out_edges_csv, out_nodes_csv)
    return super_edges[cols_edges], nodes[cols_nodes]


# --------------------------------------------------------------------------- #
# NAICS supergraph
# --------------------------------------------------------------------------- #
def build_naics_supergraph(
    edges,
    naics_level: int | None = None,
    out_edges_csv: str = "naics_supergraph_edges.csv",
    out_nodes_csv: str = "naics_supergraph_nodes.csv",
):
    """
    Same construction keyed on cleaned source_naics / dest_naics.
    naics_level: None = full code as stored; 2 = sector-level supergraph.
    Returns (edges_df, nodes_df); also writes both CSVs.
    """
    e = _host(edges).copy()
    e["n_src"] = clean_naics(e["source_naics"], naics_level)
    e["n_dst"] = clean_naics(e["dest_naics"], naics_level)
    e["pair"] = e["source"].astype(str) + "\x1f" + e["dest"].astype(str)
    if "volume" not in e:
        e["volume"] = 1

    super_edges = _aggregate_supergraph(e, "n_src", "n_dst", "src_naics", "dst_naics")

    sides = _node_side_sums(super_edges, "src_naics", "dst_naics").rename(
        columns={"group": "naics"}
    )
    # distinct customers observed per NAICS
    n1 = e[["source", "n_src"]].rename(columns={"source": "node", "n_src": "naics"})
    n2 = e[["dest", "n_dst"]].rename(columns={"dest": "node", "n_dst": "naics"})
    sizes = (pd.concat([n1, n2]).drop_duplicates()
             .groupby("naics")["node"].nunique().rename("n_nodes").reset_index())

    nodes = sizes.merge(sides, on="naics", how="outer").fillna(0.0)
    tot_amt = nodes["internal_amount"] + nodes["ext_out_amount"] + nodes["ext_in_amount"]
    nodes["mixing_ratio_w"] = np.where(tot_amt > 0, nodes["internal_amount"] / tot_amt, np.nan)
    tot_out = nodes["internal_amount"] + nodes["ext_out_amount"]
    nodes["self_loop_share"] = np.where(tot_out > 0, nodes["internal_amount"] / tot_out, np.nan)
    nodes["net_flow"] = nodes["ext_in_amount"] - nodes["ext_out_amount"]

    cols_edges = ["src_naics", "dst_naics", "amount", "volume",
                  "edge_count", "distinct_pairs", "is_self"]
    cols_nodes = ["naics", "n_nodes", "internal_amount", "internal_volume",
                  "ext_out_amount", "ext_in_amount", "mixing_ratio_w",
                  "self_loop_share", "net_flow"]
    super_edges[cols_edges].to_csv(out_edges_csv, index=False)
    nodes[cols_nodes].to_csv(out_nodes_csv, index=False)
    log.info("NAICS supergraph: %d codes, %d edges -> %s / %s",
             nodes.shape[0], super_edges.shape[0], out_edges_csv, out_nodes_csv)
    return super_edges[cols_edges], nodes[cols_nodes]


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _smoke():
    rng = np.random.default_rng(11)
    n_nodes, n_edges = 500, 4000
    naics_pool = ["522110", "722511", "423430", "-1|UNKNOWN", "541611", "******"]
    node_naics = {f"N{i}": naics_pool[rng.integers(0, len(naics_pool))]
                  for i in range(n_nodes)}
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    keep = src != dst
    e = pd.DataFrame({
        "source": [f"N{i}" for i in src[keep]],
        "dest": [f"N{i}" for i in dst[keep]],
        "amount": np.round(rng.lognormal(10, 1.5, keep.sum()), 2),
        "volume": rng.integers(1, 25, keep.sum()),
    })
    e["source_naics"] = e["source"].map(node_naics)
    e["dest_naics"] = e["dest"].map(node_naics)
    e = e.groupby(["source", "dest", "source_naics", "dest_naics"], as_index=False) \
         .agg(amount=("amount", "sum"), volume=("volume", "sum"))
    membership = pd.DataFrame({"node": [f"N{i}" for i in range(n_nodes)],
                               "community_id": [f"C{i % 10}" for i in range(n_nodes)]})

    ce, cn = build_community_supergraph(e, membership,
                                        "/tmp/community_supergraph_edges.csv",
                                        "/tmp/community_supergraph_nodes.csv")
    ne, nn = build_naics_supergraph(e, None,
                                    "/tmp/naics_supergraph_edges.csv",
                                    "/tmp/naics_supergraph_nodes.csv")
    ne2, nn2 = build_naics_supergraph(e, 2,
                                      "/tmp/naics2_supergraph_edges.csv",
                                      "/tmp/naics2_supergraph_nodes.csv")

    assert abs(ce["amount"].sum() - e["amount"].sum()) < 1e-6, "community amount conservation"
    assert abs(ne["amount"].sum() - e["amount"].sum()) < 1e-6, "naics amount conservation"
    assert abs(ne2["amount"].sum() - e["amount"].sum()) < 1e-6, "naics2 amount conservation"
    assert ce["is_self"].any(), "self-loops present in community supergraph"
    print("\ncommunity edges head:\n", ce.head(5))
    print("\ncommunity nodes head:\n", cn.head(5))
    print("\nnaics nodes:\n", nn)
    print("\nSMOKE TEST PASSED — amount conserved across all supergraphs")


if __name__ == "__main__":
    _smoke()
