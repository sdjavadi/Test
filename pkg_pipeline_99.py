"""
pkg_pipeline.py
===============
Monthly orchestrator for the PKG metric framework (v3 — streaming output).

Changes vs v2
-------------
- NO aggregated-graph section (OOM at production scale).
- Output is written PER MONTH, immediately after that month is computed,
  then freed. Nothing accumulates in memory. Files:

      ../metrics/node/node_{YYYY-MM}_{version}.parquet
      ../metrics/graph/graph_{YYYY-MM}.csv          (one row per version)
      ../metrics/ladder_thresholds.csv
      ../metrics/ladder_exclusions.csv

  Combine at the end with e.g.
      pd.concat(map(pd.read_parquet, glob('../metrics/node/*.parquet')))

- RESUMABLE: a (month, version) whose node file already exists is skipped.
  (After a resume, nmi_vs_prev / turnover metrics restart NaN on the first
  processed month, since cross-month state isn't persisted.)

- NEW node metrics (churn/deposit-model feature pool):
    counterparty turnover  : payer/payee new/lost/retained counts, Jaccard,
                             lost_payer_amount_share, new_payer_amount_share
    top-payer stability    : top_payer_same, top_payer_share_delta
    tenure & recency       : months_since_first_seen, months_active,
                             activity_gap
    neighborhood contagion : nbr_strength_trend (inflow-weighted log MoM
                             ratio of payers' strength),
                             inflow_from_shrinking_share
    hub exposure           : hub_in_share, hub_out_share (share of raw
                             in/out amount exchanged with ladder-registry
                             nodes; computed on RAW edges, attached to all
                             versions)

Graph versions: V0 raw | V1 de-hubbed (deg OR strength > P99.9)
              | V2 mega-only (deg OR strength > P99.99)
Weight policy: amount only; log1p for spectral, raw for flow. SCC excluded.
"""

from __future__ import annotations

import gc
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
# versions are derived from the ladder tiers at runtime
BETWEENNESS_K = 128           # sampled sources; 0 disables
SHRINKING_RATIO = 0.8         # payer counts as 'shrinking' below this MoM

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
    # UNWEIGHTED by design: BC weights act as distances (high amount would
    # mean a costly path — inverted semantics), and BFS is far cheaper.
    if HAS_GPU and k > 0:
        gdf = cudf.from_pandas(edges[["source", "dest"]]).drop_duplicates()
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             renumber=True)
        bc = cugraph.betweenness_centrality(G, k=k)
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
    g = st.groupby("node").agg(tot_w=("amount", "sum"),
                               tot_uw=("intra", "size"))
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


# ---------------------------------------------------------------------------
# NAICS hierarchy + counterparty NAICS-mix metrics
# ---------------------------------------------------------------------------

def naics_hierarchy(edges: pd.DataFrame) -> pd.DataFrame:
    """Per node: naics2..naics6 (first K digits when available) + naics_known.
    Unknowns (-1|UNKNOWN, ******, non-digit) -> <NA> at every level."""
    nodes = pd.concat([
        edges[["source", "source_naics"]].rename(
            columns={"source": "node", "source_naics": "naics"}),
        edges[["dest", "dest_naics"]].rename(
            columns={"dest": "node", "dest_naics": "naics"}),
    ]).drop_duplicates("node")
    digits = (nodes["naics"].astype(str).str.strip()
              .str.extract(r"^(\d{2,6})")[0])
    out = pd.DataFrame({"node": nodes["node"].to_numpy()})
    for k in range(2, 7):
        col = digits.str[:k]
        out[f"naics{k}"] = col.where(digits.str.len() >= k).to_numpy()
    out["naics_known"] = out["naics2"].notna().astype(float)
    return out


def node_naics_mix_metrics(edges: pd.DataFrame,
                           hier: pd.DataFrame) -> pd.DataFrame:
    """Counterparty industry-mix per node, on the 2-digit sector level
    (best coverage/stability; finer levels are too sparse per node):

      payer_naics2_entropy   Shannon entropy (bits) of inflow amount by
                             payer sector — revenue-base industry diversity
      payee_naics2_entropy   same for outflow — spend diversity
      top_payer_naics2_share largest sector's share of inflow
      top_payee_naics2_share largest sector's share of outflow
      same_naics2_in_share   inflow share coming from the node's own sector
      same_naics2_out_share  outflow share going to the node's own sector

    Edges whose counterpart sector is unknown are excluded from the mix;
    shares are of KNOWN-sector amount."""
    nmap = hier.set_index("node")["naics2"]
    e = pd.DataFrame({
        "src": edges["source"], "dst": edges["dest"],
        "amount": edges["amount"].to_numpy(float),
        "src_n2": nmap.reindex(edges["source"]).to_numpy(),
        "dst_n2": nmap.reindex(edges["dest"]).to_numpy(),
    })

    def _mix(df, node_col, cp_sector_col, own_sector_col, prefix):
        d = df.dropna(subset=[cp_sector_col])
        g = d.groupby([node_col, cp_sector_col])["amount"].sum()
        tot = g.groupby(level=0).sum()
        p = g / tot
        ent = (-(p * np.log2(p))).groupby(level=0).sum()
        top = p.groupby(level=0).max()
        same_amt = (d.loc[d[cp_sector_col] == d[own_sector_col]]
                    .groupby(node_col)["amount"].sum())
        same = (same_amt / tot).reindex(ent.index).fillna(0.0)
        out = pd.DataFrame({
            f"{prefix}_naics2_entropy": ent,
            f"top_{prefix}_naics2_share": top,
            f"same_naics2_{'in' if prefix == 'payer' else 'out'}_share":
                same,
        })
        out.index.name = "node"
        return out.reset_index()

    inflow = _mix(e, "dst", "src_n2", "dst_n2", "payer")
    outflow = _mix(e, "src", "dst_n2", "src_n2", "payee")
    return inflow.merge(outflow, on="node", how="outer")


# ---------------------------------------------------------------------------
# distribution summary of node metrics (network description)
# ---------------------------------------------------------------------------

_DIST_QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def distribution_summary(nf: pd.DataFrame) -> pd.DataFrame:
    """Long-format summary stats for every numeric node metric:
    count, nan_share, zero_share, mean, std, skew, min, p1..p99, max, gini.
    One row per (metric, stat)."""
    rows = []
    num = nf.select_dtypes(include=[np.number])
    for col in num.columns:
        x = num[col].to_numpy(dtype=np.float64)
        n = len(x)
        nan_share = float(np.isnan(x).mean())
        v = x[~np.isnan(x)]
        if len(v) == 0:
            continue
        stats = {"count": float(len(v)), "nan_share": nan_share,
                 "zero_share": float((v == 0).mean()),
                 "mean": float(v.mean()), "std": float(v.std()),
                 "min": float(v.min()), "max": float(v.max()),
                 "skew": float(pd.Series(v).skew())}
        qs = np.quantile(v, _DIST_QUANTILES)
        for q, val in zip(_DIST_QUANTILES, qs):
            stats[f"p{int(q * 100)}"] = float(val)
        pos = np.sort(v[v > 0])
        if len(pos) > 1:
            k = np.arange(1, len(pos) + 1)
            stats["gini"] = float(((2 * k - len(pos) - 1) @ pos)
                                  / (len(pos) * pos.sum()))
        rows += [{"metric": col, "stat": s, "value": val}
                 for s, val in stats.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# community-level metrics (per snapshot x version) with lifecycle
# ---------------------------------------------------------------------------

def community_metrics(edges: pd.DataFrame, part: pd.DataFrame,
                      nf: pd.DataFrame, hier: pd.DataFrame,
                      prev_part: pd.DataFrame | None) -> pd.DataFrame:
    """One row per Louvain community: size, intra/inter links & weight,
    density, NAICS composition at 2/4/6 digits, entropy, flow direction,
    hierarchy (trophic), importance shares, hub dependence, internal
    reciprocity, and lifecycle event vs. the previous month."""
    pmap = part.set_index("node")["community_id"]
    total_amount = edges["amount"].sum()
    e = edges.assign(gs=pmap.reindex(edges["source"]).to_numpy(),
                     gd=pmap.reindex(edges["dest"]).to_numpy()).dropna(
                         subset=["gs", "gd"])
    intra = e[e["gs"] == e["gd"]]
    ext = e[e["gs"] != e["gd"]]

    g = pd.DataFrame({"n_nodes": part.groupby("community_id").size()})
    g["n_internal_edges"] = intra.groupby("gs").size()
    g["internal_amount"] = intra.groupby("gs")["amount"].sum()
    g["internal_volume"] = intra.groupby("gs")["volume"].sum()
    g["out_edges_ext"] = ext.groupby("gs").size()
    g["in_edges_ext"] = ext.groupby("gd").size()
    g["out_amount_ext"] = ext.groupby("gs")["amount"].sum()
    g["in_amount_ext"] = ext.groupby("gd")["amount"].sum()
    g = g.fillna(0.0)
    g["internal_avg_ticket"] = np.where(
        g["internal_volume"] > 0,
        g["internal_amount"] / g["internal_volume"], np.nan)
    npairs = g["n_nodes"] * (g["n_nodes"] - 1)
    g["density_uw"] = np.where(npairs > 0,
                               g["n_internal_edges"] / npairs, np.nan)
    g["density_w"] = np.where(npairs > 0,
                              g["internal_amount"] / npairs, np.nan)
    ext_amt = g["out_amount_ext"] + g["in_amount_ext"]
    ext_uw = g["out_edges_ext"] + g["in_edges_ext"]
    g["mixing_ratio_w"] = np.where(
        g["internal_amount"] + ext_amt > 0,
        g["internal_amount"] / (g["internal_amount"] + ext_amt), np.nan)
    g["mixing_ratio_uw"] = np.where(
        g["n_internal_edges"] + ext_uw > 0,
        g["n_internal_edges"] / (g["n_internal_edges"] + ext_uw), np.nan)
    g["net_external_flow"] = g["in_amount_ext"] - g["out_amount_ext"]
    vol_s = 2 * g["internal_amount"] + ext_amt
    g["conductance_w"] = np.where(
        np.minimum(vol_s, 2 * total_amount - vol_s) > 0,
        ext_amt / np.minimum(vol_s, 2 * total_amount - vol_s), np.nan)
    g["internal_amount_share"] = g["internal_amount"] / total_amount
    g["touch_amount_share"] = (g["internal_amount"] + ext_amt) / total_amount

    # boundary nodes
    b = pd.concat([ext[["source", "gs"]].rename(
                       columns={"source": "node", "gs": "c"}),
                   ext[["dest", "gd"]].rename(
                       columns={"dest": "node", "gd": "c"})])
    g["boundary_node_frac"] = (b.groupby("c")["node"].nunique()
                               / g["n_nodes"])

    # internal reciprocity (dyad-min weighted), fully vectorized
    if len(intra):
        ip = intra.groupby(["gs", "source", "dest"], as_index=False)[
            "amount"].sum()
        rev = ip.rename(columns={"source": "dest", "dest": "source",
                                 "amount": "amount_rev"})
        m = ip.merge(rev, on=["gs", "source", "dest"], how="left")
        m["mn"] = np.minimum(m["amount"], m["amount_rev"].fillna(0.0))
        rec = m.groupby("gs")[["mn", "amount"]].sum()
        g["internal_reciprocity_w"] = rec["mn"] / rec["amount"]

    # hub dependence: top internal node's share of internal touch
    if len(intra):
        ni = pd.concat([intra.groupby(["gs", "source"])["amount"].sum(),
                        intra.groupby(["gs", "dest"])["amount"].sum()]
                       ).groupby(level=[0, 1]).sum()
        g["hub_dependence"] = (ni.groupby(level=0).max()
                               / (2 * g["internal_amount"]))

    # NAICS composition at 2 / 4 / 6 digits + entropy + unknown share
    nodes = part.merge(hier, on="node", how="left")
    for k in (2, 4, 6):
        col = f"naics{k}"
        comp = (nodes.dropna(subset=[col])
                .groupby(["community_id", col]).size())
        tot = comp.groupby(level=0).sum()
        p = comp / tot
        top = p.groupby(level=0).idxmax()
        g[f"{col}_top"] = top.map(
            lambda t: t[1] if isinstance(t, tuple) else None)
        g[f"{col}_top_share"] = p.groupby(level=0).max()
        if k == 2:
            g["naics2_entropy"] = (-(p * np.log2(p))).groupby(level=0).sum()
    g["unknown_naics_share"] = 1.0 - nodes.groupby(
        "community_id")["naics_known"].mean()

    # hierarchy & importance from node metrics
    aux_cols = [c for c in ("trophic_level", "pagerank_logw", "net_flow")
                if c in nf.columns]
    if aux_cols:
        aux = nf[["node"] + aux_cols].merge(part, on="node")
        ga = aux.groupby("community_id")
        if "trophic_level" in aux_cols:
            g["mean_trophic"] = ga["trophic_level"].mean()
            g["trophic_span"] = (ga["trophic_level"].quantile(0.9)
                                 - ga["trophic_level"].quantile(0.1))
        if "pagerank_logw" in aux_cols:
            g["pagerank_mass"] = ga["pagerank_logw"].sum()

    # lifecycle vs previous month
    g = g.rename_axis("community_id").reset_index()
    if prev_part is not None:
        lc = cm.community_lifecycle(prev_part, part)
        if "event" in lc.columns:
            cur = (lc.dropna(subset=["comm_curr"])
                   .sort_values("jaccard", ascending=False)
                   .drop_duplicates("comm_curr"))
            cur = cur.rename(columns={"comm_curr": "community_id",
                                      "comm_prev": "prev_community_id",
                                      "jaccard": "prev_jaccard"})
            g = g.merge(cur[["community_id", "prev_community_id",
                             "prev_jaccard", "event"]],
                        on="community_id", how="left")
            g["event"] = g["event"].fillna("birth")
    return g


# ---------------------------------------------------------------------------
# hub exposure (computed on RAW edges, attached to every version)
# ---------------------------------------------------------------------------

def hub_exposure(raw_edges: pd.DataFrame, hub_set: set) -> pd.DataFrame:
    """Share of a node's raw in/out amount exchanged with ladder-registry
    (V1) nodes. High hub_in_share = revenue fed by one mega-hub =
    structurally fragile deposit relationship."""
    e = raw_edges
    in_tot = e.groupby("dest")["amount"].sum()
    out_tot = e.groupby("source")["amount"].sum()
    in_hub = e[e["source"].isin(hub_set)].groupby("dest")["amount"].sum()
    out_hub = e[e["dest"].isin(hub_set)].groupby("source")["amount"].sum()
    out = pd.DataFrame({
        "hub_in_share": (in_hub / in_tot),
        "hub_out_share": (out_hub / out_tot),
    })
    out.index.name = "node"
    return out.fillna(0.0).reset_index()


# ---------------------------------------------------------------------------
# largest weakly-connected component (de-hubbed versions; V0 skipped)
# ---------------------------------------------------------------------------

def wcc_stats(edges: pd.DataFrame, node_flow: pd.DataFrame) -> dict:
    """n_wcc, lwcc_node_share, lwcc_strength_share. Answers whether a
    residual connected economy exists after de-hubbing or the graph is
    pure fragments."""
    if HAS_GPU:
        gdf = cudf.from_pandas(edges[["source", "dest"]]).drop_duplicates()
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             renumber=True)
        lab = cugraph.weakly_connected_components(G).to_pandas().rename(
            columns={"vertex": "node", "labels": "wcc"})
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        nodes = pd.unique(pd.concat([edges["source"], edges["dest"]]))
        idx = pd.Series(np.arange(len(nodes)), index=nodes)
        A = coo_matrix((np.ones(len(edges)),
                        (idx.loc[edges["source"]], idx.loc[edges["dest"]])),
                       shape=(len(nodes), len(nodes)))
        _, labels = connected_components(A, directed=False)
        lab = pd.DataFrame({"node": nodes, "wcc": labels})
    m = lab.merge(node_flow[["node", "strength"]], on="node", how="left")
    g = m.groupby("wcc")["strength"].agg(["size", "sum"])
    top = g.sort_values("size", ascending=False).iloc[0]
    return {"n_wcc": float(len(g)),
            "lwcc_node_share": float(top["size"] / len(m)),
            "lwcc_strength_share": float(top["sum"]
                                         / max(m["strength"].sum(), 1e-9))}


# ---------------------------------------------------------------------------
# per-hub monthly summary (taxonomy base + shared-hub co-payment scale)
# ---------------------------------------------------------------------------

def hub_summary(raw: pd.DataFrame, hub_set: set) -> pd.DataFrame:
    """One row per registry hub active this month, from RAW edges:
    identity (name, naics) for taxonomy work, in/out distinct counterparts
    (n_payers/n_payees), strengths, amount share, and co_pay_pairs =
    C(n_payers,2) — the number of customer pairs this hub connects, i.e.
    the clique size a hub-projection graph would create."""
    e = raw
    is_h_src = e["source"].isin(hub_set)
    is_h_dst = e["dest"].isin(hub_set)
    outg = e[is_h_src].groupby("source").agg(
        n_payees=("dest", "nunique"), out_strength=("amount", "sum"))
    inc = e[is_h_dst].groupby("dest").agg(
        n_payers=("source", "nunique"), in_strength=("amount", "sum"))
    h = outg.join(inc, how="outer").fillna(0.0).rename_axis("node")
    names = pd.concat([
        e.loc[is_h_src, ["source", "source_name", "source_naics"]].rename(
            columns={"source": "node", "source_name": "name",
                     "source_naics": "naics"}),
        e.loc[is_h_dst, ["dest", "dest_name", "dest_naics"]].rename(
            columns={"dest": "node", "dest_name": "name",
                     "dest_naics": "naics"}),
    ]).drop_duplicates("node").set_index("node")
    h = h.join(names, how="left")
    h["strength"] = h["in_strength"] + h["out_strength"]
    h["amount_share"] = h["strength"] / (2 * e["amount"].sum())
    h["co_pay_pairs"] = h["n_payers"] * (h["n_payers"] - 1) / 2
    return h.reset_index().sort_values("strength", ascending=False)


# ---------------------------------------------------------------------------
# cross-month temporal tracker (one instance per version)
# ---------------------------------------------------------------------------

class TemporalTracker:
    """Holds ONLY the previous month's aggregated pair list, previous node
    strengths, and a running tenure table — small, bounded memory."""

    def __init__(self):
        self.prev_pairs: pd.DataFrame | None = None     # source,dest,amount
        self.prev_strength: pd.Series | None = None     # node -> strength
        self.tenure: pd.DataFrame | None = None         # node state
        self.t: int = -1

    # -- turnover, top-payer stability ------------------------------------
    def _turnover_one_side(self, curr, prev, key, other, prefix):
        m = curr.merge(prev, on=[key, other], how="outer",
                       suffixes=("", "_prev"), indicator=True)
        cnt = (m.groupby([key, "_merge"], observed=True).size()
               .unstack(fill_value=0))
        stat = pd.DataFrame(index=cnt.index)
        stat[f"n_{prefix}_new"] = cnt.get("left_only", 0)
        stat[f"n_{prefix}_lost"] = cnt.get("right_only", 0)
        stat[f"n_{prefix}_retained"] = cnt.get("both", 0)
        stat[f"{prefix}_jaccard"] = stat[f"n_{prefix}_retained"] / (
            stat[f"n_{prefix}_new"] + stat[f"n_{prefix}_lost"]
            + stat[f"n_{prefix}_retained"])
        # amount shares
        lost_amt = (m.loc[m["_merge"] == "right_only"]
                    .groupby(key)["amount_prev"].sum())
        prev_tot = prev.groupby(key)["amount"].sum().rename("amount_prev_tot")
        new_amt = (m.loc[m["_merge"] == "left_only"]
                   .groupby(key)["amount"].sum())
        curr_tot = curr.groupby(key)["amount"].sum()
        stat[f"lost_{prefix}_amount_share"] = (
            lost_amt / prev_tot).reindex(stat.index)
        stat[f"new_{prefix}_amount_share"] = (
            new_amt / curr_tot).reindex(stat.index)
        stat.index.name = "node"
        return stat.reset_index()

    def _top_payer(self, pairs):
        d = pairs.sort_values("amount", ascending=False, kind="stable")
        top = d.drop_duplicates("dest")[["dest", "source", "amount"]]
        tot = pairs.groupby("dest")["amount"].sum()
        top = top.set_index("dest")
        top["share"] = top["amount"] / tot
        return top  # index dest: source, amount, share

    def update(self, edges: pd.DataFrame
               ) -> tuple[pd.DataFrame, dict]:
        """Advance one month; return (per-node temporal metrics frame,
        graph-level turnover dict: edge_jaccard_vs_prev,
        node_jaccard_vs_prev, retained_edge_amount_share,
        new_edge_amount_share)."""
        self.t += 1
        pairs = edges.groupby(["source", "dest"], as_index=False)[
            "amount"].sum()
        strength = (edges.groupby("source")["amount"].sum()
                    .add(edges.groupby("dest")["amount"].sum(),
                         fill_value=0.0))
        nodes = strength.index

        # -- tenure / recency ---------------------------------------------
        gap = pd.Series(np.nan, index=nodes)  # months since last active
        if self.tenure is None:
            self.tenure = pd.DataFrame(
                {"first_t": self.t, "last_t": self.t, "n_active": 1},
                index=nodes)
        else:
            known = self.tenure.index.intersection(nodes)
            newbies = nodes.difference(self.tenure.index)
            gap.loc[known] = self.t - self.tenure.loc[known, "last_t"]
            self.tenure.loc[known, "n_active"] += 1
            self.tenure.loc[known, "last_t"] = self.t
            if len(newbies):
                self.tenure = pd.concat([self.tenure, pd.DataFrame(
                    {"first_t": self.t, "last_t": self.t, "n_active": 1},
                    index=newbies)])
        ten = self.tenure.loc[nodes]
        out = pd.DataFrame({
            "node": nodes,
            "months_since_first_seen": self.t - ten["first_t"].to_numpy(),
            "months_active": ten["n_active"].to_numpy(),
            "activity_gap": gap.to_numpy(),
        })

        if self.prev_pairs is None:
            self.prev_pairs, self.prev_strength = pairs, strength
            gstats = {"edge_jaccard_vs_prev": np.nan,
                      "node_jaccard_vs_prev": np.nan,
                      "retained_edge_amount_share": np.nan,
                      "new_edge_amount_share": np.nan}
            for c in ("n_payer_new", "n_payer_lost", "n_payer_retained",
                      "payer_jaccard", "lost_payer_amount_share",
                      "new_payer_amount_share",
                      "n_payee_new", "n_payee_lost", "n_payee_retained",
                      "payee_jaccard", "lost_payee_amount_share",
                      "new_payee_amount_share",
                      "top_payer_same", "top_payer_share_delta",
                      "nbr_strength_trend", "inflow_from_shrinking_share"):
                out[c] = np.nan
            return out, gstats

        # -- graph-level edge/node turnover ---------------------------------
        em = pairs.merge(self.prev_pairs, on=["source", "dest"],
                         how="outer", suffixes=("", "_prev"),
                         indicator=True)
        n_both = int((em["_merge"] == "both").sum())
        gstats = {
            "edge_jaccard_vs_prev": n_both / len(em),
            "node_jaccard_vs_prev":
                len(nodes.intersection(self.prev_strength.index))
                / len(nodes.union(self.prev_strength.index)),
            "retained_edge_amount_share":
                float(em.loc[em["_merge"] == "both", "amount"].sum()
                      / pairs["amount"].sum()),
            "new_edge_amount_share":
                float(em.loc[em["_merge"] == "left_only", "amount"].sum()
                      / pairs["amount"].sum()),
        }

        # -- counterparty turnover -----------------------------------------
        payer = self._turnover_one_side(
            pairs.rename(columns={"dest": "node"}),
            self.prev_pairs.rename(columns={"dest": "node"}),
            "node", "source", "payer")
        payee = self._turnover_one_side(
            pairs.rename(columns={"source": "node"}),
            self.prev_pairs.rename(columns={"source": "node"}),
            "node", "dest", "payee")
        out = out.merge(payer, on="node", how="left").merge(
            payee, on="node", how="left")

        # -- top-payer stability --------------------------------------------
        tp_c, tp_p = self._top_payer(pairs), self._top_payer(self.prev_pairs)
        both = tp_c.join(tp_p, how="inner", lsuffix="_c", rsuffix="_p")
        tps = pd.DataFrame({
            "node": both.index,
            "top_payer_same": (both["source_c"] == both["source_p"]
                               ).astype(float),
            "top_payer_share_delta": (both["share_c"] - both["share_p"]),
        })
        out = out.merge(tps, on="node", how="left")

        # -- neighborhood contagion ------------------------------------------
        ratio = np.log((strength.reindex(self.prev_strength.index)
                        .fillna(0.0) + 1.0)
                       / (self.prev_strength + 1.0))
        r_src = ratio.reindex(pairs["source"]).to_numpy()
        amt = pairs["amount"].to_numpy(float)
        ok = ~np.isnan(r_src)
        contag = pd.DataFrame({"dest": pairs["dest"][ok],
                               "w": amt[ok], "wr": amt[ok] * r_src[ok],
                               "shrinking": (np.exp(r_src[ok])
                                             < SHRINKING_RATIO)})
        g = contag.groupby("dest")
        nbr = pd.DataFrame({
            "node": g.size().index,
            "nbr_strength_trend": g["wr"].sum() / g["w"].sum(),
            "inflow_from_shrinking_share":
                contag[contag["shrinking"]].groupby("dest")["w"].sum()
                .reindex(g.size().index).fillna(0.0) / g["w"].sum(),
        })
        out = out.merge(nbr, on="node", how="left")

        self.prev_pairs, self.prev_strength = pairs, strength
        return out, gstats


# ---------------------------------------------------------------------------
# node metric assembly
# ---------------------------------------------------------------------------

def node_metrics(edges: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nf = node_flow_metrics(edges)
    log.info("node flow metrics done (%d nodes)", len(nf))
    part = gpu_louvain(edges)
    naics_part = _naics_partition(edges)
    hier = naics_hierarchy(edges)
    blocks = [
        hier,
        node_naics_mix_metrics(edges, hier),
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
    return nf, part, hier


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
# main loop — streams one month at a time, writes, frees
# ---------------------------------------------------------------------------

def _downcast(df):
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
        elif df[c].dtype == np.int64:
            df[c] = df[c].astype(np.int32)
    return df


def run(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR):
    node_dir = os.path.join(out_dir, "node")
    graph_dir = os.path.join(out_dir, "graph")
    dist_dir = os.path.join(out_dir, "dist")
    comm_dir = os.path.join(out_dir, "community")
    hub_dir = os.path.join(out_dir, "hub")
    for d in (node_dir, graph_dir, dist_dir, comm_dir, hub_dir):
        os.makedirs(d, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(data_dir, "cust_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no cust_*.csv under {data_dir}")
    log.info("building ablation ladder from %d snapshots", len(paths))
    ladder = cm.build_ladder(paths)          # default tiers 99 / 99.9 / 99.99
    versions = ladder.versions               # ['V0','P99','P99_9','P99_99']
    log.info("graph versions: %s", versions)
    pd.DataFrame([ladder.thresholds]).to_csv(
        os.path.join(out_dir, "ladder_thresholds.csv"), index=False)
    # exclusion registry: broadest tier, flagged per stricter tier
    tiers = list(ladder.exclusion_sets)
    broad = sorted(ladder.exclusion_sets[tiers[0]])
    reg = pd.DataFrame({"node": broad})
    for t in tiers[1:]:
        reg[f"in_{t}"] = [n in ladder.exclusion_sets[t] for n in broad]
    reg.to_csv(os.path.join(out_dir, "ladder_exclusions.csv"), index=False)

    # hub registry for hub-exposure metrics: P99_9 tier if present,
    # otherwise the strictest available tier
    hub_set = ladder.exclusion_sets.get(
        "P99_9", ladder.exclusion_sets[tiers[-1]])

    prev_partition: dict[str, pd.DataFrame] = {}
    trackers = {v: TemporalTracker() for v in versions}

    for path in paths:
        time_key = re.search(r"cust_(\d{4}-\d{2})\.csv", path).group(1)
        node_path = os.path.join(node_dir, f"node_{time_key}.parquet")
        if os.path.exists(node_path):
            log.info("%s: output exists, skipping month", time_key)
            continue
        raw = pd.read_csv(path)
        hub_exp = hub_exposure(raw, hub_set)
        hub_summary(raw, hub_set).to_csv(
            os.path.join(hub_dir, f"hub_{time_key}.csv"), index=False)
        n_rows, g_rows, d_rows, c_rows = [], [], [], []

        for version in versions:
            edges = cm.apply_version(raw, ladder, version)
            if edges.empty:
                continue
            log.info("%s %s: %d edges", time_key, version, len(edges))

            nf, part, hier = node_metrics(edges)
            temporal, gstats = trackers[version].update(edges)
            nf = nf.merge(temporal, on="node", how="left")
            nf = nf.merge(hub_exp, on="node", how="left")
            nf.insert(0, "version", version)
            nf.insert(0, "time_key", time_key)
            for c in ("node", "community_id", "ga_role",
                      "naics2", "naics3", "naics4", "naics5", "naics6"):
                if c in nf.columns:
                    nf[c] = nf[c].astype("string")
            n_rows.append(_downcast(nf))

            ds = distribution_summary(nf)
            ds.insert(0, "version", version)
            ds.insert(0, "time_key", time_key)
            d_rows.append(ds)

            comm = community_metrics(edges, part, nf, hier,
                                     prev_partition.get(version))
            comm.insert(0, "version", version)
            comm.insert(0, "time_key", time_key)
            for c in ("community_id", "prev_community_id", "event",
                      "naics2_top", "naics4_top", "naics6_top"):
                if c in comm.columns:
                    comm[c] = comm[c].astype("string")
            c_rows.append(_downcast(comm))

            trophic_lv = (nf[["node", "trophic_level"]].dropna()
                          if "trophic_level" in nf else
                          pd.DataFrame(columns=["node", "trophic_level"]))
            gm = graph_metrics(edges, part, nf, trophic_lv)
            for k, v in gstats.items():
                gm[k] = v
            if version != "V0":          # WCC on de-hubbed graphs only
                for k, v in wcc_stats(edges, nf).items():
                    gm[k] = v
            if version in prev_partition:
                cp = cm.compare_partitions(prev_partition[version], part)
                gm["nmi_vs_prev"] = cp["nmi"].iloc[0]
                gm["ari_vs_prev"] = cp["ari"].iloc[0]
            prev_partition[version] = part
            gm.insert(0, "version", version)
            gm.insert(0, "time_key", time_key)
            g_rows.append(gm)

            del edges, part
            gc.collect()

        if n_rows:
            month = pd.concat(n_rows, ignore_index=True)
            month.to_parquet(node_path)
            log.info("wrote %s (%d rows, %d cols, %d versions)",
                     node_path, len(month), month.shape[1], len(n_rows))
            del month
        if g_rows:
            pd.concat(g_rows, ignore_index=True).to_csv(
                os.path.join(graph_dir, f"graph_{time_key}.csv"),
                index=False)
        if d_rows:
            pd.concat(d_rows, ignore_index=True).to_csv(
                os.path.join(dist_dir, f"dist_{time_key}.csv"), index=False)
        if c_rows:
            pd.concat(c_rows, ignore_index=True).to_parquet(
                os.path.join(comm_dir, f"community_{time_key}.parquet"))
        del raw, hub_exp, n_rows, g_rows, d_rows, c_rows
        gc.collect()

    log.info("done — per-month outputs in %s and %s", node_dir, graph_dir)


if __name__ == "__main__":
    run()
