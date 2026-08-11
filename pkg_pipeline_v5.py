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
import pickle
import re
import warnings

import numpy as np
import pandas as pd

import pkg_custom_metrics as cm
import pkg_customers as pc
import pkg_geo_metrics as pgm
import pkg_runtime as rt

log = logging.getLogger("pkg_pipeline")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

DATA_DIR = "../data"
OUT_DIR = "../metrics"
CUSTOMERS_CSV = None       # None -> "<data_dir>/customers.csv"
TIME_BUDGET_H = 8.0        # GPU session cap; drives the ETA projector
REUSE_LADDER = True        # load thresholds/exclusions from disk
GEO_METRICS = True         # per-version geographic block, inline
GEO_ONLY = False           # skip graph metrics; emit ../metrics/geo/ only
PERSIST_TRACKERS = True    # checkpoint cross-month state for clean resume
# Snapshot columns actually consumed. Identity / NAICS now come from
# customers.csv, so name+naics columns are never parsed: less IO, less RAM.
SNAPSHOT_COLS = ["source", "dest", "amount", "volume"]
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


def _sdiv(num, den, default=np.nan):
    """Elementwise divide that never evaluates the 0-denominator entries.

    `np.where(cond, a / b, x)` still computes a / b for EVERY element and
    only then discards the masked ones — which is where every "invalid value
    encountered in divide" RuntimeWarning in this pipeline came from. Here
    the division is only performed where the denominator is positive and
    finite; everything else takes `default` untouched.
    """
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    out = np.full(np.broadcast(n, d).shape, float(default), dtype=np.float64)
    np.divide(n, d, out=out,
              where=(d > 0) & np.isfinite(d) & np.isfinite(n))
    return out


# ---------------------------------------------------------------------------
# GPU wrappers  (np.log1p ufunc dispatches on both cuDF and pandas)
# ---------------------------------------------------------------------------

def _cu_graph(edges: pd.DataFrame, log_weight: bool,
              store_transposed: bool = False):
    gdf = cudf.from_pandas(edges[["source", "dest", "amount"]])
    gdf["w"] = np.log1p(gdf["amount"]) if log_weight else gdf["amount"]
    G = cugraph.Graph(directed=True)
    # PageRank consumes the transposed CSR; building it directly removes
    # both the per-call UserWarning and an internal re-transpose.
    try:
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             edge_attr="w", renumber=True,
                             store_transposed=store_transposed)
    except TypeError:      # older cuGraph without the kwarg
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             edge_attr="w", renumber=True)
    return G


def gpu_pagerank(edges: pd.DataFrame, log_weight: bool) -> pd.DataFrame:
    col = "pagerank_logw" if log_weight else "pagerank_raw"
    if HAS_GPU:
        pr = cugraph.pagerank(
            _cu_graph(edges, log_weight, store_transposed=True), alpha=0.85)
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


def gpu_clustering(edges: pd.DataFrame) -> pd.DataFrame:
    """Local clustering coefficient on the undirected projection:
    C_i = 2*T_i / (k_i*(k_i-1)); NaN where undirected degree k < 2."""
    und = pd.concat([edges[["source", "dest"]],
                     edges[["dest", "source"]].rename(
                         columns={"dest": "source", "source": "dest"})]
                    ).drop_duplicates()
    und = und[und["source"] != und["dest"]]
    if HAS_GPU:
        gdf = cudf.from_pandas(und)
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(gdf, source="source", destination="dest",
                             renumber=True)
        tri = cugraph.triangle_count(G).to_pandas().rename(
            columns={"vertex": "node", "counts": "tri"})
        k = und.groupby("source").size().rename("k")
        tri = tri.merge(k, left_on="node", right_index=True, how="left")
        tri["clustering_coef"] = _sdiv(2.0 * tri["tri"],
                                       tri["k"] * (tri["k"] - 1))
        return tri[["node", "clustering_coef"]]
    # CPU fallback (dev scale only): A^3 diagonal
    nodes = pd.unique(pd.concat([und["source"], und["dest"]]))
    if len(nodes) > 100_000:
        return pd.DataFrame(columns=["node", "clustering_coef"])
    from scipy import sparse as sp
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    A = sp.coo_matrix(
        (np.ones(len(und)),
         (idx.loc[und["source"]], idx.loc[und["dest"]])),
        shape=(len(nodes), len(nodes))).tocsr()
    A.data[:] = 1.0
    tri = (A @ A).multiply(A).sum(axis=1).A.ravel() / 2.0
    k = np.asarray(A.sum(axis=1)).ravel()
    cc = _sdiv(2.0 * tri, k * (k - 1))
    return pd.DataFrame({"node": nodes, "clustering_coef": cc})


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
    # distinct counterparties regardless of direction: <= degree, with
    # equality only when no counterpart appears on both sides
    nbr = pd.concat([
        edges[["source", "dest"]].rename(columns={"source": "node",
                                                  "dest": "nbr"}),
        edges[["dest", "source"]].rename(columns={"dest": "node",
                                                  "source": "nbr"}),
    ]).drop_duplicates()
    nf["n_neighbors"] = nbr.groupby("node").size()
    nf["strength"] = nf["in_strength"] + nf["out_strength"]
    nf["net_flow"] = nf["in_strength"] - nf["out_strength"]
    nf["flow_ratio"] = _sdiv(nf["net_flow"], nf["strength"], 0.0)
    nf["throughflow"] = np.minimum(nf["in_strength"], nf["out_strength"])
    nf["log_strength"] = np.log1p(nf["strength"])
    nf["avg_in_ticket"] = _sdiv(nf["in_strength"], nf["in_volume"])
    nf["avg_out_ticket"] = _sdiv(nf["out_strength"], nf["out_volume"])
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
        "frac_intra_edges_w": _sdiv(g["int_w"], g["tot_w"]),
    }).reset_index(drop=True)


def _naics_partition(edges: pd.DataFrame,
                     attrs: pd.DataFrame) -> pd.DataFrame:
    """NAICS2 pseudo-partition restricted to nodes present this month.
    Sourced from the customer dimension, not the snapshot."""
    nodes = pd.unique(pd.concat([edges["source"], edges["dest"]],
                                ignore_index=True))
    part = pc.naics_partition(attrs)
    return part[part["node"].isin(nodes)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# NAICS hierarchy + counterparty NAICS-mix metrics
# ---------------------------------------------------------------------------

def naics_hierarchy(edges: pd.DataFrame,
                    attrs: pd.DataFrame) -> pd.DataFrame:
    """naics2..naics6 + naics_known for this month's nodes, taken from
    the customer dimension. naics_cd and naics_desc arrive as separate
    columns now, so there is no '|' composite to split."""
    nodes = pd.unique(pd.concat([edges["source"], edges["dest"]],
                                ignore_index=True))
    cols = ["node", "naics2", "naics3", "naics4", "naics5", "naics6",
            "naics_known"]
    h = attrs.loc[attrs["node"].isin(nodes), cols].copy()
    missing = pd.Index(nodes).difference(pd.Index(h["node"]))
    if len(missing):
        pad = pd.DataFrame({"node": missing})
        for c in cols[1:-1]:
            pad[c] = pd.NA
        pad["naics_known"] = 0.0
        h = pd.concat([h, pad], ignore_index=True)
    return h.reset_index(drop=True)


def node_naics_homophily(edges: pd.DataFrame,
                         hier: pd.DataFrame) -> pd.DataFrame:
    """same_naics2_in/out_share: share of known-sector inflow/outflow
    exchanged with the node's OWN 2-digit sector (industry homophily).
    Sector entropy/top-share/n now come from composition_metrics (guarded,
    valid-NAICS only) and are no longer computed here."""
    nmap = hier.set_index("node")["naics2"]
    e = pd.DataFrame({
        "src": edges["source"], "dst": edges["dest"],
        "amount": edges["amount"].to_numpy(float),
        "src_n2": nmap.reindex(edges["source"]).to_numpy(),
        "dst_n2": nmap.reindex(edges["dest"]).to_numpy(),
    })
    outs = []
    for node_col, cp_sector, own_sector, col in (
            ("dst", "src_n2", "dst_n2", "same_naics2_in_share"),
            ("src", "dst_n2", "src_n2", "same_naics2_out_share")):
        d = e.dropna(subset=[cp_sector])
        tot = d.groupby(node_col)["amount"].sum()
        same = (d.loc[d[cp_sector] == d[own_sector]]
                .groupby(node_col)["amount"].sum())
        outs.append((same / tot).reindex(tot.index).fillna(0.0)
                    .rename(col).rename_axis("node").reset_index())
    return outs[0].merge(outs[1], on="node", how="outer")


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
    g["internal_avg_ticket"] = _sdiv(g["internal_amount"],
                                     g["internal_volume"])
    npairs = g["n_nodes"] * (g["n_nodes"] - 1)
    g["density_uw"] = _sdiv(g["n_internal_edges"], npairs)
    g["density_w"] = _sdiv(g["internal_amount"], npairs)
    ext_amt = g["out_amount_ext"] + g["in_amount_ext"]
    ext_uw = g["out_edges_ext"] + g["in_edges_ext"]
    g["mixing_ratio_w"] = _sdiv(g["internal_amount"],
                                g["internal_amount"] + ext_amt)
    g["mixing_ratio_uw"] = _sdiv(g["n_internal_edges"],
                                 g["n_internal_edges"] + ext_uw)
    g["net_external_flow"] = g["in_amount_ext"] - g["out_amount_ext"]
    vol_s = 2 * g["internal_amount"] + ext_amt
    g["conductance_w"] = _sdiv(
        ext_amt, np.minimum(vol_s, 2 * total_amount - vol_s))
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


# Full customer passthrough carried into the node table. Everything in
# customers.csv lands here so downstream apps need no dimension join.
IDENTITY_COLS = [
    "node", "cust_name", "naics_cd", "naics_desc", "addr_loc_rec_type",
    "lat", "lon", "zip5", "zip3", "state", "city", "geo_status",
    "shared_structure",
    # typing, merged onto attrs in run() so the node table is self-contained
    "entity_type", "naics_status", "node_type",
]


def node_identity(edges: pd.DataFrame,
                  attrs: pd.DataFrame) -> pd.DataFrame:
    """Customer passthrough for this month's nodes. Nodes absent from
    customers.csv are emitted with NA attributes and geo_status
    'not_in_customers' so coverage stays visible instead of silent."""
    nodes = pd.unique(pd.concat([edges["source"], edges["dest"]],
                                ignore_index=True))
    cols = [c for c in IDENTITY_COLS if c in attrs.columns]
    a = attrs.loc[attrs["node"].isin(nodes), cols]
    missing = pd.Index(nodes).difference(pd.Index(a["node"]))
    if len(missing):
        pad = pd.DataFrame({"node": missing})
        for c in cols[1:]:
            pad[c] = pd.NA
        pad["geo_status"] = "not_in_customers"
        a = pd.concat([a, pad], ignore_index=True)
    return a.reset_index(drop=True)


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

def hub_summary(raw: pd.DataFrame, hub_set: set,
                attrs: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per registry hub active this month, from RAW edges:
    identity (name, naics, location) for taxonomy work, in/out distinct
    counterparts (n_payers/n_payees), strengths, amount share, and
    co_pay_pairs = C(n_payers,2) — the number of customer pairs this hub
    connects, i.e. the clique size a hub-projection graph would create.

    Identity now comes from the customer dimension, not the snapshot: the
    snapshot carries only source/dest/amount/volume.
    """
    e = raw
    is_h_src = e["source"].isin(hub_set)
    is_h_dst = e["dest"].isin(hub_set)
    outg = e[is_h_src].groupby("source").agg(
        n_payees=("dest", "nunique"), out_strength=("amount", "sum"))
    inc = e[is_h_dst].groupby("dest").agg(
        n_payers=("source", "nunique"), in_strength=("amount", "sum"))
    h = outg.join(inc, how="outer").fillna(0.0).rename_axis("node")
    if attrs is not None and len(attrs):
        keep = [c for c in ("cust_name", "naics_cd", "naics_desc", "state",
                            "city", "zip3", "geo_status")
                if c in attrs.columns]
        h = h.join(attrs.set_index("node")[keep], how="left")
    h["strength"] = h["in_strength"] + h["out_strength"]
    h["amount_share"] = h["strength"] / (2 * e["amount"].sum())
    h["co_pay_pairs"] = h["n_payers"] * (h["n_payers"] - 1) / 2
    return h.reset_index().sort_values("strength", ascending=False)


# ---------------------------------------------------------------------------
# cross-month temporal tracker (one instance per version)
# ---------------------------------------------------------------------------

# Counterparty-memory design (multi-window):
#   Comparing only consecutive snapshots over-counts churn — a counterpart
#   can skip one month and reappear. Each tracker therefore keeps a
#   per-relationship PRESENCE BITMASK over the last MEM_BITS months
#   (bit0 = current month after update). Current counterparts classify as:
#     retained   — also present last month
#     returning  — absent last month, but seen within the last W_NEW months
#     new        — first appearance within the W_NEW-month memory window
#   plus persistent = counterparts present in >= PERSIST_MIN of the last
#   W_NEW months (incl. current) — the stable relationship core.
#   'lost' stays the 1-month definition (present t-1, absent t) and is
#   documented as short-memory churn.
W_NEW = 6          # months of memory before an appearance counts as 'new'
PERSIST_MIN = 4    # months seen (of last W_NEW) to count as persistent
MEM_BITS = 16
_POPCOUNT = np.array([bin(i).count("1") for i in range(1 << MEM_BITS)],
                     dtype=np.uint8)


class TemporalTracker:
    """Bounded cross-month state per version: relationship presence
    bitmasks, previous pairs (for lost/top-payer/graph turnover),
    previous strengths, running tenure."""

    def __init__(self):
        self.mem: pd.DataFrame | None = None      # source,dest,presence
        self.prev_pairs: pd.DataFrame | None = None
        self.prev_strength: pd.Series | None = None
        self.tenure: pd.DataFrame | None = None
        self.t: int = -1

    # -- side aggregation of cohort classes --------------------------------
    @staticmethod
    def _cohorts_one_side(cl: pd.DataFrame, node_col: str,
                          prefix: str) -> pd.DataFrame:
        g = cl.groupby(node_col)
        tot_amt = g["amount"].sum()
        out = pd.DataFrame(index=tot_amt.index)
        for cls in ("retained", "returning", "new"):
            m = cl[cl["cls"] == cls]
            gm = m.groupby(node_col)
            out[f"n_{prefix}_{cls}"] = gm.size()
            out[f"{cls}_{prefix}_amount_share"] = (
                gm["amount"].sum() / tot_amt)
        pm = cl[cl["persistent"]].groupby(node_col)["amount"].sum()
        out[f"persistent_{prefix}_amount_share"] = pm / tot_amt
        cnt_cols = [c for c in out.columns if c.startswith("n_")]
        out[cnt_cols] = out[cnt_cols].fillna(0)
        share_cols = [c for c in out.columns if c.endswith("share")]
        out[share_cols] = out[share_cols].fillna(0.0)
        out.index.name = "node"
        return out.reset_index()

    def _lost_one_side(self, curr, prev, key, other, prefix):
        m = curr.merge(prev, on=[key, other], how="outer",
                       suffixes=("", "_prev"), indicator=True)
        cnt = (m.groupby([key, "_merge"], observed=True).size()
               .unstack(fill_value=0))
        stat = pd.DataFrame(index=cnt.index)
        stat[f"n_{prefix}_lost"] = cnt.get("right_only", 0)
        n_both = cnt.get("both", 0)
        n_left = cnt.get("left_only", 0)
        stat[f"{prefix}_jaccard"] = n_both / (
            n_both + n_left + stat[f"n_{prefix}_lost"])
        lost_amt = (m.loc[m["_merge"] == "right_only"]
                    .groupby(key)["amount_prev"].sum())
        prev_tot = prev.groupby(key)["amount"].sum()
        stat[f"lost_{prefix}_amount_share"] = (
            lost_amt / prev_tot).reindex(stat.index)
        stat.index.name = "node"
        return stat.reset_index()

    def _top_payer(self, pairs):
        d = pairs.sort_values("amount", ascending=False, kind="stable")
        top = d.drop_duplicates("dest")[["dest", "source", "amount"]]
        tot = pairs.groupby("dest")["amount"].sum()
        top = top.set_index("dest")
        top["share"] = top["amount"] / tot
        return top

    _TEMPORAL_COLS = (
        [f"n_{p}_{c}" for p in ("payer", "payee")
         for c in ("retained", "returning", "new", "lost")]
        + [f"{c}_{p}_amount_share" for p in ("payer", "payee")
           for c in ("retained", "returning", "new", "lost", "persistent")]
        + ["payer_jaccard", "payee_jaccard",
           "top_payer_same", "top_payer_share_delta",
           "nbr_strength_trend", "inflow_from_shrinking_share"])

    def update(self, edges: pd.DataFrame
               ) -> tuple[pd.DataFrame, dict]:
        self.t += 1
        pairs = edges.groupby(["source", "dest"], as_index=False)[
            "amount"].sum()
        strength = (edges.groupby("source")["amount"].sum()
                    .add(edges.groupby("dest")["amount"].sum(),
                         fill_value=0.0))
        nodes = strength.index

        # -- tenure / recency ---------------------------------------------
        gap = pd.Series(np.nan, index=nodes)
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

        # -- relationship memory: shift, classify, upsert --------------------
        mask_all = (1 << MEM_BITS) - 1
        if self.mem is None:
            self.mem = pd.DataFrame(
                {"source": pd.Series(dtype=object),
                 "dest": pd.Series(dtype=object),
                 "presence": pd.Series(dtype=np.uint32)})
        else:
            self.mem["presence"] = (
                (self.mem["presence"].to_numpy(np.uint32) << 1)
                & mask_all)
        m = pairs.merge(self.mem, on=["source", "dest"], how="left")
        hist = m["presence"].fillna(0).to_numpy(np.uint32)
        prev_bit = (hist & 0b10) != 0
        win_mask = ((1 << W_NEW) - 1) << 1          # bits 1..W (t-1..t-W)
        seen_before = (hist & win_mask) != 0
        cls = np.select([prev_bit, seen_before],
                        ["retained", "returning"], default="new")
        pres_now = (hist | 1) & ((1 << W_NEW) - 1)  # incl current month
        persistent = _POPCOUNT[pres_now] >= PERSIST_MIN
        cl = pairs.assign(cls=cls, persistent=persistent)

        # first month: cohorts undefined (everything trivially 'new')
        if self.t == 0:
            for c in self._TEMPORAL_COLS:
                out[c] = np.nan
        else:
            payer = self._cohorts_one_side(
                cl.rename(columns={"dest": "node"}), "node", "payer")
            payee = self._cohorts_one_side(
                cl.rename(columns={"source": "node"}), "node", "payee")
            out = out.merge(payer, on="node", how="left").merge(
                payee, on="node", how="left")
            lp = self._lost_one_side(
                pairs.rename(columns={"dest": "node"}),
                self.prev_pairs.rename(columns={"dest": "node"}),
                "node", "source", "payer")
            le = self._lost_one_side(
                pairs.rename(columns={"source": "node"}),
                self.prev_pairs.rename(columns={"source": "node"}),
                "node", "dest", "payee")
            out = out.merge(lp, on="node", how="left").merge(
                le, on="node", how="left")

            tp_c, tp_p = self._top_payer(pairs), self._top_payer(
                self.prev_pairs)
            both = tp_c.join(tp_p, how="inner", lsuffix="_c", rsuffix="_p")
            out = out.merge(pd.DataFrame({
                "node": both.index,
                "top_payer_same": (both["source_c"] == both["source_p"]
                                   ).astype(float),
                "top_payer_share_delta": both["share_c"] - both["share_p"],
            }), on="node", how="left")

            ratio = np.log((strength.reindex(self.prev_strength.index)
                            .fillna(0.0) + 1.0)
                           / (self.prev_strength + 1.0))
            r_src = ratio.reindex(pairs["source"]).to_numpy()
            amt = pairs["amount"].to_numpy(float)
            ok = ~np.isnan(r_src)
            contag = pd.DataFrame({
                "dest": pairs["dest"][ok], "w": amt[ok],
                "wr": amt[ok] * r_src[ok],
                "shrinking": np.exp(r_src[ok]) < SHRINKING_RATIO})
            g = contag.groupby("dest")
            out = out.merge(pd.DataFrame({
                "node": g.size().index,
                "nbr_strength_trend": g["wr"].sum() / g["w"].sum(),
                "inflow_from_shrinking_share":
                    contag[contag["shrinking"]].groupby("dest")["w"].sum()
                    .reindex(g.size().index).fillna(0.0) / g["w"].sum(),
            }), on="node", how="left")

        # graph-level turnover
        if self.prev_pairs is None:
            gstats = {k: np.nan for k in
                      ("edge_jaccard_vs_prev", "node_jaccard_vs_prev",
                       "retained_edge_amount_share",
                       "new_edge_amount_share")}
        else:
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
                    float(em.loc[em["_merge"] == "left_only",
                                 "amount"].sum() / pairs["amount"].sum()),
            }

        # upsert memory: shifted old rows OUTER current pairs with bit0 set
        up = self.mem.merge(pairs[["source", "dest"]],
                            on=["source", "dest"], how="outer",
                            indicator=True)
        up["presence"] = up["presence"].fillna(0).astype(np.uint32)
        up.loc[up["_merge"] != "left_only", "presence"] |= 1
        up = up[up["presence"] > 0].drop(columns="_merge")
        self.mem = up.reset_index(drop=True)

        self.prev_pairs, self.prev_strength = pairs, strength
        return out, gstats


# ---------------------------------------------------------------------------
# node metric assembly
# ---------------------------------------------------------------------------

def node_metrics(edges: pd.DataFrame, attrs: pd.DataFrame,
                 coords: pd.DataFrame, month: str, version: str
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_e = len(edges)
    with rt.step(month, version, "flow", n_e):
        nf = node_flow_metrics(edges)
    with rt.step(month, version, "louvain", n_e):
        part = gpu_louvain(edges)
    with rt.step(month, version, "naics_dims", n_e):
        naics_part = _naics_partition(edges, attrs)
        hier = naics_hierarchy(edges, attrs)

    steps = [
        ("identity",      lambda: node_identity(edges, attrs)),
        ("naics_hier",    lambda: hier),
        ("naics_homoph",  lambda: node_naics_homophily(edges, hier)),
        ("pagerank_raw",  lambda: gpu_pagerank(edges, log_weight=False)),
        ("pagerank_logw", lambda: gpu_pagerank(edges, log_weight=True)),
        ("hits",          lambda: cm.weighted_hits(edges)),
        ("core_number",   lambda: gpu_core_number(edges)),
        ("betweenness",   lambda: gpu_betweenness(edges)),
        ("trophic",       lambda: cm.trophic_levels(edges)),
        ("clustering",    lambda: gpu_clustering(edges)),
        ("reciprocity",   lambda: cm.node_reciprocity(edges).rename(
            columns={"reciprocity_node_w": "reciprocity_amount_share"})),
        ("community_id",  lambda: part),
        ("roles",         lambda: cm.participation_and_roles(edges, part)),
        ("intra_comm",    lambda: node_intra_community_fractions(edges, part)),
        ("naics_particip", lambda: cm.participation_and_roles(
            edges, naics_part)[["node", "participation_coef"]].rename(
            columns={"participation_coef": "naics_participation"})),
    ]
    if GEO_METRICS and coords is not None and len(coords):
        # per version, not once on RAW: hub removal changes the
        # counterparty set, and that reversed six geographic findings.
        # `coords` here is a prepared GeoIndex built once per month — see
        # the month loop. Passing the raw 20M-row frame instead costs ~290s
        # per call in table construction alone.
        steps.append(("geo", lambda: pgm.geo_node_metrics(edges, coords)))

    for name, fn in steps:
        with rt.step(month, version, name, n_e):
            extra = fn()
        if extra is not None and "node" in extra.columns and len(extra):
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


def _tracker_path(out_dir: str) -> str:
    return os.path.join(out_dir, "state", "trackers.pkl")


def _save_trackers(out_dir, trackers, prev_partition, done_months):
    """Checkpoint cross-month state so a resumed run does not restart
    turnover / tenure / nmi_vs_prev at NaN."""
    if not PERSIST_TRACKERS:
        return
    p = _tracker_path(out_dir)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "wb") as fh:
        pickle.dump({"trackers": trackers, "prev_partition": prev_partition,
                     "done_months": done_months}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(p + ".tmp", p)      # atomic: never a half-written checkpoint


def _load_trackers(out_dir):
    p = _tracker_path(out_dir)
    if not (PERSIST_TRACKERS and os.path.exists(p)):
        return None
    try:
        with open(p, "rb") as fh:
            st = pickle.load(fh)
        log.info("resumed tracker state after %d months",
                 len(st.get("done_months", [])))
        return st
    except Exception as exc:
        log.warning("tracker checkpoint unreadable (%s) — starting cold", exc)
        return None


# ---------------------------------------------------------------------------
# geo-only pass
# ---------------------------------------------------------------------------

def _run_geo_only(paths, out_dir, ladder, versions, coords, attrs):
    """Compute ONLY the geographic block, one file per month.

    Split from the main run because the two have very different costs and
    very different iteration rates: graph metrics are stable and expensive,
    geo is new and cheap. Running them separately means geo can be re-run
    after a fix without repeating hours of centrality work.

    Output keys on (time_key, version, node) — identical to the node table —
    so the two are combined with a plain merge, not a reconciliation:

        node = pd.read_parquet(".../node/node_2024-01.parquet")
        geo  = pd.read_parquet(".../geo/geo_2024-01.parquet")
        full = node.merge(geo, on=["time_key", "version", "node"], how="left")
    """
    geo_dir = os.path.join(out_dir, "geo")
    os.makedirs(geo_dir, exist_ok=True)
    n_total, n_done = len(paths), 0
    for path in paths:
        time_key = re.search(r"cust_(\d{4}-\d{2})\.csv", path).group(1)
        gpath = os.path.join(geo_dir, f"geo_{time_key}.parquet")
        if os.path.exists(gpath):
            n_done += 1
            log.info("%s: geo exists, skipping (%d/%d)", time_key, n_done,
                     n_total)
            continue
        with rt.step(time_key, "-", "read_snapshot"):
            raw = pd.read_csv(path, usecols=SNAPSHOT_COLS,
                              dtype={"source": str, "dest": str})
        # ONE GeoIndex for the month: the coordinate table is ~20M rows and
        # a month touches ~2M nodes, so restricting once and reusing across
        # versions is where the fixed cost goes away.
        with rt.step(time_key, "-", "geo_index", len(raw)):
            gi = pgm.GeoIndex(coords, pd.concat(
                [raw["source"], raw["dest"]], ignore_index=True).to_numpy())
        rows = []
        for version in versions:
            edges = cm.apply_version(raw, ladder, version)
            if edges.empty:
                continue
            with rt.step(time_key, version, "geo", len(edges)):
                g = pgm.geo_node_metrics(edges, gi)
            if g.empty:
                continue
            g.insert(0, "version", version)
            g.insert(0, "time_key", time_key)
            for c in ("node", "geo_locality_class"):
                if c in g.columns:
                    g[c] = g[c].astype("string")
            rows.append(_downcast(g))
            del edges
            gc.collect()
        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.to_parquet(gpath)
            log.info("wrote %s (%d rows, %d cols)", gpath, len(out),
                     out.shape[1])
            del out
        n_done += 1
        rt.month_done(time_key, n_done, n_total)
        del raw, gi, rows
        gc.collect()
    rt.flush()
    rt.log_summary()
    log.info("geo-only pass done — %s", geo_dir)


def run(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR,
        customers_csv: str | None = None):
    rt.configure(budget_hours=TIME_BUDGET_H,
                 out_path=os.path.join(out_dir, "run_timings.csv"))
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
    # ---- customer dimension (replaces two full snapshot passes) --------
    with rt.step("setup", "-", "load_customers"):
        cpath = (customers_csv or CUSTOMERS_CSV
                 or os.path.join(data_dir, "customers.csv"))
        customers = pc.load_customers(cpath)
    coords = customers.coords
    typing = cm.typing_from_customers(customers)
    typing.to_csv(os.path.join(out_dir, "node_typing.csv"), index=False)
    # one customer frame carrying identity + NAICS hierarchy + geo + typing,
    # so every downstream consumer reads a single table
    attrs = customers.attrs.merge(
        typing[["node", "entity_type", "naics_status", "node_type"]],
        on="node", how="left")
    attrs.to_parquet(os.path.join(out_dir, "customers_dim.parquet"),
                     index=False)
    log.info("customer dimension: %d nodes, %d attribute columns",
             len(attrs), attrs.shape[1])

    # ---- ablation ladder ----------------------------------------------
    ladder = None
    if REUSE_LADDER:
        try:
            with rt.step("setup", "-", "load_ladder"):
                ladder = cm.load_ladder(out_dir)
        except FileNotFoundError as exc:
            log.warning("ladder reuse failed (%s) — rebuilding", exc)
    if ladder is None:
        log.info("building ablation ladder from %d snapshots", len(paths))
        with rt.step("setup", "-", "build_ladder", len(paths)):
            ladder = cm.build_ladder(paths)
        pd.DataFrame([ladder.thresholds]).to_csv(
            os.path.join(out_dir, "ladder_thresholds.csv"), index=False)
        tiers = list(ladder.exclusion_sets)
        broad = sorted(ladder.exclusion_sets[tiers[0]])
        reg = pd.DataFrame({"node": broad})
        for t in tiers[1:]:
            reg[f"in_{t}"] = [n in ladder.exclusion_sets[t] for n in broad]
        reg.to_csv(os.path.join(out_dir, "ladder_exclusions.csv"),
                   index=False)
    versions = ladder.versions
    log.info("graph versions: %s", versions)

    # hub registry for hub-exposure metrics: P99_9 tier if present,
    # otherwise the strictest available tier
    tiers = list(ladder.exclusion_sets)
    if not tiers:
        raise RuntimeError("ladder has no exclusion tiers — check "
                           "ladder_exclusions.csv")
    hub_set = ladder.exclusion_sets.get("P99_9",
                                        ladder.exclusion_sets[tiers[-1]])
    log.info("hub registry: %d nodes (tier %s)", len(hub_set),
             "P99_9" if "P99_9" in ladder.exclusion_sets else tiers[-1])

    if GEO_ONLY:
        _run_geo_only(paths, out_dir, ladder, versions, coords, attrs)
        return

    state = _load_trackers(out_dir)
    if state:
        trackers = state["trackers"]
        prev_partition = state["prev_partition"]
        done_months = list(state["done_months"])
        for v in versions:
            trackers.setdefault(v, TemporalTracker())
    else:
        trackers = {v: TemporalTracker() for v in versions}
        prev_partition: dict[str, pd.DataFrame] = {}
        done_months = []

    n_total = len(paths)
    n_done = 0
    reported_cov = False
    for path in paths:
        time_key = re.search(r"cust_(\d{4}-\d{2})\.csv", path).group(1)
        node_path = os.path.join(node_dir, f"node_{time_key}.parquet")
        if os.path.exists(node_path):
            n_done += 1
            log.info("%s: output exists, skipping month (%d/%d)",
                     time_key, n_done, n_total)
            continue
        # SINGLE ID DTYPE POLICY: node ids are strings from ingestion on.
        # (int-parsed ids collided with the str-typed typing table twice;
        # one canonical dtype ends the class of bug.)
        with rt.step(time_key, "-", "read_snapshot"):
            raw = pd.read_csv(path, usecols=SNAPSHOT_COLS,
                              dtype={"source": str, "dest": str})
        with rt.step(time_key, "-", "hub_exposure", len(raw)):
            hub_exp = hub_exposure(raw, hub_set)
        # composition on the RAW graph (node-local; attached to all versions)
        with rt.step(time_key, "-", "composition", len(raw)):
            comp = cm.composition_metrics(raw, typing, hub_set=hub_set)
        with rt.step(time_key, "-", "hub_summary", len(raw)):
            hub_summary(raw, hub_set, attrs).to_csv(
                os.path.join(hub_dir, f"hub_{time_key}.csv"), index=False)
        if not reported_cov:
            nodes = pd.unique(pd.concat([raw["source"], raw["dest"]],
                                        ignore_index=True))
            log.info("customer attribute coverage on %s graph nodes: %s",
                     time_key, customers.coverage(nodes))
            reported_cov = True
        # ONE GeoIndex per month, shared by every version: restricting the
        # ~20M-row coordinate table to the ~2M nodes this month touches, and
        # factorizing ids to int codes, is what removes geo's fixed cost.
        month_geo = coords
        if GEO_METRICS and coords is not None and len(coords):
            with rt.step(time_key, "-", "geo_index", len(raw)):
                month_geo = pgm.GeoIndex(coords, pd.concat(
                    [raw["source"], raw["dest"]],
                    ignore_index=True).to_numpy())
        n_rows, g_rows, d_rows, c_rows = [], [], [], []

        for version in versions:
            edges = cm.apply_version(raw, ladder, version)
            if edges.empty:
                continue
            log.info("%s %s: %d edges", time_key, version, len(edges))

            nf, part, hier = node_metrics(edges, attrs, month_geo,
                                          time_key, version)
            with rt.step(time_key, version, "temporal", len(edges)):
                temporal, gstats = trackers[version].update(edges)
            nf = nf.merge(temporal, on="node", how="left")
            nf = nf.merge(hub_exp, on="node", how="left")
            nf = nf.merge(comp, on="node", how="left")
            nf.insert(0, "version", version)
            nf.insert(0, "time_key", time_key)
            for c in ("node", "cust_name", "naics_cd", "naics_desc",
                      "addr_loc_rec_type", "zip5", "zip3", "state", "city",
                      "geo_status", "geo_locality_class",
                      "community_id", "ga_role",
                      "naics2", "naics3", "naics4", "naics5", "naics6"):
                if c in nf.columns:
                    nf[c] = nf[c].astype("string")
            n_rows.append(_downcast(nf))

            ds = distribution_summary(nf)
            ds.insert(0, "version", version)
            ds.insert(0, "time_key", time_key)
            d_rows.append(ds)

            with rt.step(time_key, version, "community", len(edges)):
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
            with rt.step(time_key, version, "graph_metrics", len(edges)):
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
        del raw, hub_exp, comp, n_rows, g_rows, d_rows, c_rows, month_geo
        gc.collect()

        n_done += 1
        done_months.append(time_key)
        _save_trackers(out_dir, trackers, prev_partition, done_months)
        rt.month_done(time_key, n_done, n_total)

    rt.log_summary()
    rt.flush()
    log.info("done — per-month outputs in %s and %s", node_dir, graph_dir)


if __name__ == "__main__":
    run()
