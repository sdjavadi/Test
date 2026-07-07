"""
pkg_pipeline.py
===============
Monthly orchestrator for the PKG metric framework (v2 — node-focused).

Graph versions (ablation ladder, built once from ALL snapshot aggregate):
    V0  raw
    V1  'de-hubbed'  : exclude degree > P99.9  OR strength > P99.9
    V2  'mega-only'  : exclude degree > P99.99 OR strength > P99.99

Per snapshot (../data/cust_YYYY-MM.csv) x version:
  1. NODE-LEVEL metrics — flow/balance/concentration (raw amount),
     PageRank on BOTH raw and log1p weights, weighted HITS (log1p),
     core number, k-sample betweenness, trophic level, dyad reciprocity,
     plus COMMUNITY-RELATED NODE metrics: community_id, within-module z,
     participation coefficient, Guimerà–Amaral role, weighted/unweighted
     intra-community flow fractions, NAICS participation.
  2. GRAPH-LEVEL metrics for the snapshot, incl. NMI/ARI partition
     stability vs. the previous month.

Finally: GRAPH-LEVEL metrics for the ALL-SNAPSHOT AGGREGATED graph
(time_key='AGG'), also at V0/V1/V2.

Community-level tables and community/NAICS supergraph exports are handled
by separate modules (to be added), not by this pipeline.

Weight policy: amount is the sole weight. log1p for spectral/iterative
metrics; raw amount for flow/accounting metrics. SCC/WCC excluded.
"""

from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np
import pandas as pd

import pkg_custom_metrics as cm

log = logging.getLogger("pkg_pipeline")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

DATA_DIR = "../data"
OUT_DIR = "../metrics"
VERSIONS = ["V0", "V1", "V2"]
BETWEENNESS_K = 128          # sampled sources; set 0 to disable

try:
    import cudf
    import cugraph
    HAS_GPU = True
except ImportError:  # pragma: no cover
    HAS_GPU = False
    log.warning("cuGraph unavailable — CPU fallbacks in use (dev mode only)")


# ---------------------------------------------------------------------------
# GPU wrappers  (np.log1p ufunc dispatches on both cuDF and pandas)
# ---------------------------------------------------------------------------

def _cu_graph(edges: pd.DataFrame, log_weight: bool):
    gdf = cudf.from_pandas(edges[["source", "dest", "amount"]])
    gdf["w"] = np.log1p(gdf["amount"]) if log_weight else gdf["amount"]
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source="source", destination="dest",
                         edge_attr="w", renumber=True)
    return G


def gpu_pagerank(edges: pd.DataFrame, log_weight: bool) -> pd.DataFrame:
    col = "pagerank_logw" if log_weight else "pagerank_raw"
    if HAS_GPU:
        pr = cugraph.pagerank(_cu_graph(edges, log_weight), alpha=0.85)
        return pr.to_pandas().rename(columns={"vertex": "node",
                                              "pagerank": col})
    s = edges.groupby("dest")["amount"].sum()
    s = np.log1p(s) if log_weight else s
    return (s / s.sum()).rename(col).rename_axis("node").reset_index()


def gpu_louvain(edges: pd.DataFrame) -> pd.DataFrame:
    if HAS_GPU:
        gdf = cudf.from_pandas(edges[["source", "dest", "amount"]])
        gdf["w"] = np.log1p(gdf["amount"])
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             edge_attr="w", renumber=True)
        parts, mod = cugraph.louvain(G)
        out = parts.to_pandas().rename(columns={"vertex": "node",
                                                "partition": "community_id"})
        out.attrs["modularity"] = float(mod)
        return out
    nodes = pd.unique(pd.concat([edges["source"], edges["dest"]]))
    out = pd.DataFrame({"node": nodes,
                        "community_id": pd.factorize(nodes)[0] % 50})
    out.attrs["modularity"] = np.nan
    return out


def gpu_core_number(edges: pd.DataFrame) -> pd.DataFrame:
    if HAS_GPU:
        gdf = cudf.from_pandas(edges[["source", "dest"]]).drop_duplicates()
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             renumber=True)
        return cugraph.core_number(G).to_pandas().rename(
            columns={"vertex": "node"})
    return pd.DataFrame(columns=["node", "core_number"])


def gpu_betweenness(edges: pd.DataFrame, k: int = BETWEENNESS_K
                    ) -> pd.DataFrame:
    if HAS_GPU and k > 0:
        bc = cugraph.betweenness_centrality(_cu_graph(edges, True), k=k)
        return bc.to_pandas().rename(
            columns={"vertex": "node",
                     "betweenness_centrality": "betweenness_approx"})
    return pd.DataFrame(columns=["node", "betweenness_approx"])


# ---------------------------------------------------------------------------
# node-level flow metrics (raw amount)
# ---------------------------------------------------------------------------

def node_flow_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    g_out = edges.groupby("source").agg(
        out_degree=("dest", "nunique"), out_strength=("amount", "sum"),
        out_volume=("volume", "sum"))
    g_in = edges.groupby("dest").agg(
        in_degree=("source", "nunique"), in_strength=("amount", "sum"),
        in_volume=("volume", "sum"))
    nf = g_out.join(g_in, how="outer").fillna(0.0).rename_axis("node")
    nf["degree"] = nf["in_degree"] + nf["out_degree"]
    nf["strength"] = nf["in_strength"] + nf["out_strength"]
    nf["net_flow"] = nf["in_strength"] - nf["out_strength"]
    nf["flow_ratio"] = np.where(nf["strength"] > 0,
                                nf["net_flow"] / nf["strength"], 0.0)
    nf["throughflow"] = np.minimum(nf["in_strength"], nf["out_strength"])
    nf["log_strength"] = np.log1p(nf["strength"])
    nf["avg_in_ticket"] = np.where(nf["in_volume"] > 0,
                                   nf["in_strength"] / nf["in_volume"], np.nan)
    nf["avg_out_ticket"] = np.where(nf["out_volume"] > 0,
                                    nf["out_strength"] / nf["out_volume"],
                                    np.nan)
    for direction, grp in (("out", "source"), ("in", "dest")):
        other = "dest" if grp == "source" else "source"
        sh = edges.groupby([grp, other])["amount"].sum()
        p = sh / sh.groupby(level=0).sum()
        nf[f"hhi_{direction}"] = (p ** 2).groupby(level=0).sum()
        nf[f"top1_{direction}_share"] = p.groupby(level=0).max()
        d = p.rename("p").reset_index().sort_values("p", ascending=False,
                                                    kind="stable")
        nf[f"top3_{direction}_share"] = (
            d.groupby(grp, sort=False).head(3).groupby(grp)["p"].sum())
    return nf.reset_index()


# ---------------------------------------------------------------------------
# node-level community metrics: intra-community flow fractions
# ---------------------------------------------------------------------------

def node_intra_community_fractions(edges: pd.DataFrame,
                                   partition: pd.DataFrame) -> pd.DataFrame:
    """Per node: share of its edge COUNT and its AMOUNT (both directions
    combined) that stays inside its own community.

        frac_intra_uw = intra edges / all edges
        frac_intra_w  = intra amount / all amount
    """
    pmap = partition.set_index("node")["community_id"]
    gs = pmap.reindex(edges["source"]).to_numpy()
    gd = pmap.reindex(edges["dest"]).to_numpy()
    intra = (gs == gd) & pd.notna(gs) & pd.notna(gd)
    amt = edges["amount"].to_numpy(float)
    st = pd.DataFrame({
        "node": pd.concat([edges["source"], edges["dest"]],
                          ignore_index=True),
        "amount": np.concatenate([amt, amt]),
        "intra": np.concatenate([intra, intra]),
    })
    g = st.groupby("node").agg(tot_w=("amount", "sum"), tot_uw=("intra", "size"))
    gi = st[st["intra"]].groupby("node").agg(int_w=("amount", "sum"),
                                             int_uw=("intra", "size"))
    g = g.join(gi, how="left").fillna(0.0)
    return pd.DataFrame({
        "node": g.index,
        "frac_intra_edges_uw": g["int_uw"] / g["tot_uw"],
        "frac_intra_edges_w": np.where(g["tot_w"] > 0,
                                       g["int_w"] / g["tot_w"], np.nan),
    }).reset_index(drop=True)


def _naics_partition(edges: pd.DataFrame) -> pd.DataFrame:
    nodes = pd.concat([
        edges[["source", "source_naics"]].rename(
            columns={"source": "node", "source_naics": "community_id"}),
        edges[["dest", "dest_naics"]].rename(
            columns={"dest": "node", "dest_naics": "community_id"}),
    ]).drop_duplicates("node")
    nodes["community_id"] = nodes["community_id"].astype(str)
    return nodes.reset_index(drop=True)


def node_metrics(edges: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full node-level table + the Louvain partition (reused downstream)."""
    nf = node_flow_metrics(edges)
    log.info("node flow metrics done (%d nodes)", len(nf))
    part = gpu_louvain(edges)
    naics_part = _naics_partition(edges)
    blocks = [
        gpu_pagerank(edges, log_weight=False),
        gpu_pagerank(edges, log_weight=True),
        cm.weighted_hits(edges),
        gpu_core_number(edges),
        gpu_betweenness(edges),
        cm.trophic_levels(edges),
        cm.node_reciprocity(edges),
        part,
        cm.participation_and_roles(edges, part),
        node_intra_community_fractions(edges, part),
        cm.participation_and_roles(edges, naics_part)[
            ["node", "participation_coef"]].rename(
            columns={"participation_coef": "naics_participation"}),
    ]
    for extra in blocks:
        if "node" in extra.columns and len(extra):
            nf = nf.merge(extra, on="node", how="left")
    return nf, part


# ---------------------------------------------------------------------------
# graph-level metrics
# ---------------------------------------------------------------------------

def graph_metrics(edges: pd.DataFrame, partition: pd.DataFrame,
                  node_flow: pd.DataFrame,
                  trophic_lv: pd.DataFrame) -> pd.DataFrame:
    n_nodes = len(node_flow)
    row = {
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "total_amount": edges["amount"].sum(),
        "total_volume": edges["volume"].sum(),
        "avg_ticket": edges["amount"].sum() / max(edges["volume"].sum(), 1),
        "density": len(edges) / (n_nodes * (n_nodes - 1)),
        "n_communities": partition["community_id"].nunique(),
        "modularity_Q": partition.attrs.get("modularity", np.nan),
    }
    row.update(cm.graph_reciprocity(edges).iloc[0].to_dict())
    row.update(cm.directed_assortativity(edges).iloc[0].to_dict())
    if len(trophic_lv):
        row["trophic_incoherence_F0"] = \
            cm.trophic_incoherence(edges, trophic_lv).iloc[0, 0]
    ts = cm.tail_stats(node_flow["strength"])
    row.update({"gini_strength": ts["gini"].iloc[0],
                "hill_alpha_strength": ts["hill_alpha"].iloc[0],
                "top_0.1pct_amount_share": ts["top_share"].iloc[0]})
    row["gini_degree"] = cm.tail_stats(node_flow["degree"])["gini"].iloc[0]
    for _, r in cm.weighted_rich_club(edges).iterrows():
        row[f"rich_club_w_{r['rank_frac']}"] = r["rich_club_w"]
    csz = partition.groupby("community_id").size()
    row["community_size_gini"] = cm.tail_stats(csz)["gini"].iloc[0]
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR,
        versions=VERSIONS):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "cust_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no cust_*.csv under {data_dir}")
    log.info("building ablation ladder from %d snapshots", len(paths))
    ladder = cm.build_ladder(paths)
    pd.DataFrame([ladder.thresholds]).to_csv(
        os.path.join(out_dir, "ladder_thresholds.csv"), index=False)
    pd.DataFrame({
        "node": sorted(ladder.v1_dehubbed),
        "in_v2_mega": [n in ladder.v2_mega
                       for n in sorted(ladder.v1_dehubbed)],
    }).to_csv(os.path.join(out_dir, "ladder_exclusions.csv"), index=False)

    prev_partition: dict[str, pd.DataFrame] = {}
    node_rows, graph_rows = [], []

    # -- per snapshot ------------------------------------------------------
    for path in paths:
        time_key = re.search(r"cust_(\d{4}-\d{2})\.csv", path).group(1)
        raw = pd.read_csv(path)
        for version in versions:
            edges = cm.apply_version(raw, ladder, version)
            if edges.empty:
                continue
            log.info("%s %s: %d edges", time_key, version, len(edges))

            nf, part = node_metrics(edges)
            nf.insert(0, "version", version)
            nf.insert(0, "time_key", time_key)
            node_rows.append(nf)

            trophic_lv = (nf[["node", "trophic_level"]].dropna()
                          if "trophic_level" in nf else
                          pd.DataFrame(columns=["node", "trophic_level"]))
            gm = graph_metrics(edges, part, nf, trophic_lv)
            if version in prev_partition:
                cp = cm.compare_partitions(prev_partition[version], part)
                gm["nmi_vs_prev"] = cp["nmi"].iloc[0]
                gm["ari_vs_prev"] = cp["ari"].iloc[0]
            prev_partition[version] = part
            gm.insert(0, "version", version)
            gm.insert(0, "time_key", time_key)
            graph_rows.append(gm)

    # -- aggregated graph (all snapshots combined) --------------------------
    log.info("building aggregated graph")
    agg = None
    for path in paths:
        e = pd.read_csv(path)
        part_sum = (e.groupby(
            ["source", "source_naics", "dest", "dest_naics"],
            as_index=False)[["amount", "volume"]].sum())
        agg = part_sum if agg is None else (
            pd.concat([agg, part_sum])
            .groupby(["source", "source_naics", "dest", "dest_naics"],
                     as_index=False)[["amount", "volume"]].sum())
    for version in versions:
        edges = cm.apply_version(agg, ladder, version)
        log.info("AGG %s: %d edges", version, len(edges))
        nf, part = node_metrics(edges)
        trophic_lv = (nf[["node", "trophic_level"]].dropna()
                      if "trophic_level" in nf else
                      pd.DataFrame(columns=["node", "trophic_level"]))
        gm = graph_metrics(edges, part, nf, trophic_lv)
        gm.insert(0, "version", version)
        gm.insert(0, "time_key", "AGG")
        graph_rows.append(gm)
        nf.insert(0, "version", version)
        nf.insert(0, "time_key", "AGG")
        node_rows.append(nf)

    # -- write panels --------------------------------------------------------
    def _write(rows, name, str_cols):
        df = pd.concat(rows, ignore_index=True)
        for c in str_cols:
            if c in df.columns:
                df[c] = df[c].astype("string")
        df.to_parquet(os.path.join(out_dir, name))
        log.info("wrote %s (%d rows)", name, len(df))

    _write(node_rows, "node_panel.parquet",
           ["node", "community_id", "ga_role"])
    _write(graph_rows, "graph_panel.parquet", [])
    log.info("done — panels written to %s", out_dir)


if __name__ == "__main__":
    run()
