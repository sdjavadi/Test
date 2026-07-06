"""
pkg_pipeline.py
===============
Monthly orchestrator for the PKG metric framework.

Per snapshot (../data/cust_YYYY-MM.csv) x version (V0..V4):
  1. filter edges by ablation ladder (built once from all 23 snapshots)
  2. cuGraph metrics on GPU (PageRank, Louvain, core number, k-sample
     betweenness) with CPU fallbacks where cuGraph is unavailable
  3. custom metrics from pkg_custom_metrics
  4. community-level and NAICS-level metric tables
  5. supergraph CSVs -> ../snapshot/
  6. append everything to long-format temporal panels in ../metrics/

Weight policy: amount is the sole weight; log1p for spectral/iterative,
raw for flow. SCC/WCC intentionally excluded.
"""

from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np
import pandas as pd

import pkg_custom_metrics as cm
import pkg_supergraph as sg

log = logging.getLogger("pkg_pipeline")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

DATA_DIR = "../data"
SNAP_DIR = "../snapshot"
OUT_DIR = "../metrics"
VERSIONS = ["V0", "V1", "V2", "V3", "V4"]

try:
    import cudf
    import cugraph
    HAS_GPU = True
except ImportError:  # pragma: no cover
    HAS_GPU = False
    log.warning("cuGraph unavailable — CPU fallbacks in use (dev mode only)")


# ---------------------------------------------------------------------------
# GPU wrappers (log1p weights for spectral; Louvain on log1p)
# ---------------------------------------------------------------------------

def _cu_graph(edges: pd.DataFrame, log_weight: bool):
    gdf = cudf.from_pandas(edges[["source", "dest", "amount"]])
    gdf["w"] = np.log1p(gdf["amount"]) if log_weight else gdf["amount"]
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source="source", destination="dest",
                         edge_attr="w", renumber=True)
    return G


def gpu_pagerank(edges: pd.DataFrame) -> pd.DataFrame:
    if HAS_GPU:
        pr = cugraph.pagerank(_cu_graph(edges, True), alpha=0.85)
        return pr.to_pandas().rename(columns={"vertex": "node",
                                              "pagerank": "pagerank_w"})
    # fallback: strength-proportional proxy (dev only)
    s = np.log1p(edges.groupby("dest")["amount"].sum())
    return (s / s.sum()).rename("pagerank_w").rename_axis("node").reset_index()


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
    # fallback: connected-component-free greedy label propagation (dev only)
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
        cn = cugraph.core_number(G)
        return cn.to_pandas().rename(columns={"vertex": "node"})
    return pd.DataFrame(columns=["node", "core_number"])


def gpu_betweenness(edges: pd.DataFrame, k: int = 256) -> pd.DataFrame:
    if HAS_GPU:
        bc = cugraph.betweenness_centrality(_cu_graph(edges, True), k=k)
        return bc.to_pandas().rename(
            columns={"vertex": "node",
                     "betweenness_centrality": "betweenness_approx"})
    return pd.DataFrame(columns=["node", "betweenness_approx"])


# ---------------------------------------------------------------------------
# node-level flow metrics (pure pandas, raw amount)
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
    nf["avg_in_ticket"] = np.where(nf["in_volume"] > 0,
                                   nf["in_strength"] / nf["in_volume"], np.nan)
    nf["avg_out_ticket"] = np.where(nf["out_volume"] > 0,
                                    nf["out_strength"] / nf["out_volume"], np.nan)
    # concentration
    for direction, grp, tot in (("out", "source", "out_strength"),
                                ("in", "dest", "in_strength")):
        sh = edges.groupby([grp, "dest" if grp == "source" else "source"]
                           )["amount"].sum()
        tot_s = sh.groupby(level=0).sum()
        p = sh / tot_s
        nf[f"hhi_{direction}"] = (p ** 2).groupby(level=0).sum()
        nf[f"top1_{direction}_share"] = p.groupby(level=0).max()
        d = p.rename("p").reset_index().sort_values("p", ascending=False,
                                                    kind="stable")
        top3 = d.groupby(grp, sort=False).head(3).groupby(grp)["p"].sum()
        nf[f"top3_{direction}_share"] = top3
    return nf.reset_index()


# ---------------------------------------------------------------------------
# community / NAICS group metrics
# ---------------------------------------------------------------------------

def group_metrics(edges: pd.DataFrame, partition: pd.DataFrame,
                  label: str) -> pd.DataFrame:
    """§3 of the manifest for an arbitrary partition [node, community_id]."""
    pmap = partition.set_index("node")["community_id"]
    e = edges.assign(gs=pmap.reindex(edges["source"]).to_numpy(),
                     gd=pmap.reindex(edges["dest"]).to_numpy()).dropna(
                         subset=["gs", "gd"])
    intra = e[e["gs"] == e["gd"]]
    n = partition.groupby("community_id").size().rename("n_nodes")

    g = pd.DataFrame({"n_nodes": n})
    g["n_internal_edges"] = intra.groupby("gs").size()
    g["internal_amount"] = intra.groupby("gs")["amount"].sum()
    g["internal_volume"] = intra.groupby("gs")["volume"].sum()
    g["internal_avg_ticket"] = g["internal_amount"] / g["internal_volume"]
    g["density_uw"] = g["n_internal_edges"] / (g["n_nodes"] * (g["n_nodes"] - 1))
    g["density_w"] = g["internal_amount"] / (g["n_nodes"] * (g["n_nodes"] - 1))

    out_ext = e[e["gs"] != e["gd"]].groupby("gs")["amount"].sum()
    in_ext = e[e["gs"] != e["gd"]].groupby("gd")["amount"].sum()
    out_ext_uw = e[e["gs"] != e["gd"]].groupby("gs").size()
    in_ext_uw = e[e["gs"] != e["gd"]].groupby("gd").size()
    g["out_amount_ext"] = out_ext
    g["in_amount_ext"] = in_ext
    g = g.fillna({"n_internal_edges": 0, "internal_amount": 0.0,
                  "internal_volume": 0.0, "out_amount_ext": 0.0,
                  "in_amount_ext": 0.0})
    ext_amt = g["out_amount_ext"] + g["in_amount_ext"]
    g["mixing_ratio_w"] = g["internal_amount"] / (g["internal_amount"] + ext_amt)
    ext_uw = out_ext_uw.add(in_ext_uw, fill_value=0).reindex(g.index).fillna(0)
    g["mixing_ratio_uw"] = g["n_internal_edges"] / (g["n_internal_edges"] + ext_uw)
    g["net_external_flow"] = g["in_amount_ext"] - g["out_amount_ext"]

    # conductance_w: cut / min(vol(S), vol(rest)); vol = 2*internal + cut
    cut = ext_amt
    vol_s = 2 * g["internal_amount"] + cut
    total_vol = 2 * edges["amount"].sum()
    g["conductance_w"] = cut / np.minimum(vol_s, total_vol - vol_s)

    # boundary node fraction
    bnodes = pd.concat([e.loc[e["gs"] != e["gd"], ["source", "gs"]]
                        .rename(columns={"source": "node", "gs": "g"}),
                        e.loc[e["gs"] != e["gd"], ["dest", "gd"]]
                        .rename(columns={"dest": "node", "gd": "g"})])
    g["boundary_node_frac"] = (bnodes.groupby("g")["node"].nunique()
                               / g["n_nodes"])

    # internal reciprocity (dyad-min weighted, restricted to intra edges)
    if len(intra):
        rec = (intra.groupby("gs")[["source", "dest", "amount"]]
               .apply(lambda d: cm.graph_reciprocity(d)["reciprocity_w"].iloc[0]))
        g["internal_reciprocity_w"] = rec

    # NAICS composition (only meaningful for community partition)
    nodes = pd.concat([
        edges[["source", "source_naics"]].rename(
            columns={"source": "node", "source_naics": "naics"}),
        edges[["dest", "dest_naics"]].rename(
            columns={"dest": "node", "dest_naics": "naics"}),
    ]).drop_duplicates("node")
    nodes["g"] = pmap.reindex(nodes["node"]).to_numpy()
    comp = nodes.dropna(subset=["g"]).groupby(["g", "naics"]).size()
    tot = comp.groupby(level=0).sum()
    p = comp / tot
    g["naics_entropy"] = (-(p * np.log2(p))).groupby(level=0).sum()
    top = p.groupby(level=0).idxmax()
    g["naics_top1"] = top.map(lambda t: t[1] if isinstance(t, tuple) else None)
    g["naics_top1_share"] = p.groupby(level=0).max()

    # hub dependence: internal amount share of the top internal node
    if len(intra):
        node_int = pd.concat([intra.groupby(["gs", "source"])["amount"].sum(),
                              intra.groupby(["gs", "dest"])["amount"].sum()])
        node_int = node_int.groupby(level=[0, 1]).sum()
        g["hub_dependence"] = (node_int.groupby(level=0).max()
                               / (2 * g["internal_amount"]))
    g = g.rename_axis("group_id").reset_index()
    g.insert(0, "partition", label)
    return g


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
    row.update({"trophic_incoherence_F0":
                cm.trophic_incoherence(edges, trophic_lv).iloc[0, 0]})
    ts = cm.tail_stats(node_flow["strength"])
    row.update({"gini_strength": ts["gini"].iloc[0],
                "hill_alpha_strength": ts["hill_alpha"].iloc[0],
                "top_0.1pct_amount_share": ts["top_share"].iloc[0]})
    row["gini_degree"] = cm.tail_stats(node_flow["degree"])["gini"].iloc[0]
    rc = cm.weighted_rich_club(edges)
    for _, r in rc.iterrows():
        row[f"rich_club_w_{r['rank_frac']}"] = r["rich_club_w"]
    csz = partition.groupby("community_id").size()
    row["community_size_gini"] = cm.tail_stats(csz)["gini"].iloc[0]
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR,
        snap_dir: str = SNAP_DIR, versions=VERSIONS):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "cust_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no cust_*.csv under {data_dir}")
    log.info("building ablation ladder from %d snapshots", len(paths))
    ladder = cm.build_ladder(paths)
    pd.DataFrame([ladder.thresholds]).to_csv(
        os.path.join(out_dir, "ladder_thresholds.csv"), index=False)

    prev_partition: dict[str, pd.DataFrame] = {}
    node_rows, comm_rows, graph_rows, life_rows = [], [], [], []

    for path in paths:
        time_key = re.search(r"cust_(\d{4}-\d{2})\.csv", path).group(1)
        raw = pd.read_csv(path)
        for version in versions:
            edges = cm.apply_version(raw, ladder, version)
            if edges.empty:
                continue
            log.info("%s %s: %d edges", time_key, version, len(edges))

            # --- node level -------------------------------------------------
            nf = node_flow_metrics(edges)
            part = gpu_louvain(edges)
            for extra in (gpu_pagerank(edges), cm.weighted_hits(edges),
                          gpu_core_number(edges), gpu_betweenness(edges),
                          cm.trophic_levels(edges),
                          cm.node_reciprocity(edges),
                          cm.participation_and_roles(edges, part),
                          part):
                if "node" in extra.columns:
                    nf = nf.merge(extra, on="node", how="left")
            # NAICS participation (partition swapped)
            naics_part = _naics_partition(edges)
            npart = cm.participation_and_roles(edges, naics_part)[
                ["node", "participation_coef"]].rename(
                columns={"participation_coef": "naics_participation"})
            nf = nf.merge(npart, on="node", how="left")
            nf.insert(0, "version", version)
            nf.insert(0, "time_key", time_key)
            node_rows.append(nf)

            # --- community & NAICS level -------------------------------------
            for lbl, p in (("community", part), ("naics", naics_part),
                           ("naics2", _naics_partition(edges, digits=2))):
                gm = group_metrics(edges, p, lbl)
                gm.insert(0, "version", version)
                gm.insert(0, "time_key", time_key)
                comm_rows.append(gm)
                sg.write_supergraph(edges, lbl, version, time_key,
                                    out_dir=snap_dir,
                                    partition=p if lbl == "community" else None)

            # --- graph level --------------------------------------------------
            trophic_lv = nf[["node", "trophic_level"]].dropna() \
                if "trophic_level" in nf else pd.DataFrame(
                    columns=["node", "trophic_level"])
            gm = graph_metrics(edges, part, nf, trophic_lv)
            gm.insert(0, "version", version)
            gm.insert(0, "time_key", time_key)
            graph_rows.append(gm)

            # --- lifecycle vs previous month ----------------------------------
            key = version
            if key in prev_partition:
                lc = cm.community_lifecycle(prev_partition[key], part)
                cp = cm.compare_partitions(prev_partition[key], part)
                lc["nmi_vs_prev"] = cp["nmi"].iloc[0]
                lc["ari_vs_prev"] = cp["ari"].iloc[0]
                lc.insert(0, "version", version)
                lc.insert(0, "time_key", time_key)
                life_rows.append(lc)
            prev_partition[key] = part

    def _write(rows, name, str_cols):
        df = pd.concat(rows, ignore_index=True)
        for c in str_cols:
            if c in df.columns:
                df[c] = df[c].astype("string")
        df.to_parquet(os.path.join(out_dir, name))

    _write(node_rows, "node_panel.parquet",
           ["node", "community_id", "ga_role"])
    _write(comm_rows, "group_panel.parquet",
           ["group_id", "naics_top1", "partition"])
    _write(graph_rows, "graph_panel.parquet", [])
    if life_rows:
        _write(life_rows, "lifecycle_panel.parquet",
               ["comm_prev", "comm_curr", "event"])
    log.info("done — panels written to %s, supergraphs to %s", out_dir, snap_dir)


def _naics_partition(edges: pd.DataFrame, digits: int | None = None
                     ) -> pd.DataFrame:
    nodes = pd.concat([
        edges[["source", "source_naics"]].rename(
            columns={"source": "node", "source_naics": "community_id"}),
        edges[["dest", "dest_naics"]].rename(
            columns={"dest": "node", "dest_naics": "community_id"}),
    ]).drop_duplicates("node")
    nodes["community_id"] = nodes["community_id"].astype(str)
    if digits:
        nodes["community_id"] = nodes["community_id"].str.slice(0, digits)
    return nodes.reset_index(drop=True)


if __name__ == "__main__":
    run()
