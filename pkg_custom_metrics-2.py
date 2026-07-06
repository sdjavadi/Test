"""
pkg_custom_metrics.py
=====================
Custom graph metrics for the PKG monthly snapshot framework — algorithms
NOT available in cuGraph, implemented on CPU (scipy/numpy/pandas) against
edge lists. All functions accept a pandas edge DataFrame with columns:

    source, dest, amount   (volume optional, never used as weight)

Conventions
-----------
- amount is the SOLE edge weight; log1p transform applied internally where
  the metric is spectral/iterative (HITS, trophic); raw amount for flow math.
- Every public function is wrapped by @safe(): on any exception it logs and
  returns a NaN-filled frame with the documented schema so a single metric
  failure never kills the monthly run.
- Feature flags in FLAGS let you disable expensive metrics per run.
"""

from __future__ import annotations

import functools
import logging
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

log = logging.getLogger("pkg_custom_metrics")

# ----------------------------------------------------------------------------
# feature flags & safe() isolation
# ----------------------------------------------------------------------------

FLAGS = {
    "weighted_hits": True,
    "trophic": True,
    "reciprocity": True,
    "participation": True,
    "assortativity": True,
    "tail_stats": True,
    "rich_club": True,
    "partition_compare": True,
    "lifecycle": True,
}


def safe(flag: str, empty_schema: dict):
    """Decorator: feature-flag gate + exception isolation.

    On disabled flag or any exception, returns a single-row NaN frame with
    `empty_schema` columns (dtype inferred), never raises.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not FLAGS.get(flag, True):
                log.info("metric %s disabled by flag", fn.__name__)
                return _nan_frame(empty_schema)
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.error("metric %s FAILED:\n%s", fn.__name__, traceback.format_exc())
                return _nan_frame(empty_schema)

        return wrapper

    return deco


def _nan_frame(schema: dict) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series([np.nan], dtype="float64") if t == float
                         else pd.Series([None], dtype="object")
                         for c, t in schema.items()})


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------

def _index_nodes(edges: pd.DataFrame):
    nodes = pd.Index(pd.unique(pd.concat([edges["source"], edges["dest"]],
                                         ignore_index=True)))
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    src = idx.loc[edges["source"]].to_numpy()
    dst = idx.loc[edges["dest"]].to_numpy()
    return nodes, src, dst


def _adj(edges: pd.DataFrame, log_weight: bool):
    nodes, src, dst = _index_nodes(edges)
    w = edges["amount"].to_numpy(dtype=np.float64)
    if log_weight:
        w = np.log1p(w)
    n = len(nodes)
    A = sparse.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return nodes, A


# ============================================================================
# 1. WEIGHTED HITS (power iteration)  — cuGraph HITS is unweighted
# ============================================================================

@safe("weighted_hits", {"node": object, "hits_hub_w": float, "hits_auth_w": float})
def weighted_hits(edges: pd.DataFrame, max_iter: int = 200,
                  tol: float = 1e-9) -> pd.DataFrame:
    """Weighted HITS via alternating power iteration on W = log1p(amount).

        h <- W a ;  a <- W^T h ;  L2-normalize each step.

    Returns per-node hub and authority scores (L2-normalized).
    """
    nodes, A = _adj(edges, log_weight=True)
    n = A.shape[0]
    h = np.full(n, 1.0 / np.sqrt(n))
    a = h.copy()
    for _ in range(max_iter):
        a_new = A.T @ h
        na = np.linalg.norm(a_new)
        a_new = a_new / na if na > 0 else a_new
        h_new = A @ a_new
        nh = np.linalg.norm(h_new)
        h_new = h_new / nh if nh > 0 else h_new
        if (np.abs(h_new - h).max() < tol) and (np.abs(a_new - a).max() < tol):
            h, a = h_new, a_new
            break
        h, a = h_new, a_new
    return pd.DataFrame({"node": nodes, "hits_hub_w": h, "hits_auth_w": a})


# ============================================================================
# 2. TROPHIC LEVELS — MacKay–Johnson–Rodgers (2020), conjugate-gradient solve
# ============================================================================

@safe("trophic", {"node": object, "trophic_level": float})
def trophic_levels(edges: pd.DataFrame, tol: float = 1e-8,
                   max_iter: int = 2000) -> pd.DataFrame:
    """MJR-2020 trophic levels on W = log1p(amount).

    Solve  Λ h = v  with  Λ = diag(u) − W − W^T,  u = in+out strength,
    v = in_strength − out_strength.  Λ is singular (constant nullspace on
    each weakly-connected piece); CG with zero-mean shift yields the
    minimum-norm solution; levels shifted so min = 0.
    """
    nodes, A = _adj(edges, log_weight=True)
    s_out = np.asarray(A.sum(axis=1)).ravel()
    s_in = np.asarray(A.sum(axis=0)).ravel()
    u = s_in + s_out
    v = s_in - s_out
    n = A.shape[0]
    L = sparse.diags(u) - A - A.T          # symmetric PSD (weighted Laplacian)
    # tiny regularization to pin nullspace, keeps CG stable at 5M nodes
    L = L + sparse.identity(n, format="csr") * (1e-10 * max(u.max(), 1.0))
    h, info = cg(L.tocsr(), v, rtol=tol, maxiter=max_iter)
    if info > 0:
        log.warning("trophic CG hit max_iter (info=%d); returning best iterate", info)
    h = h - h.min()
    return pd.DataFrame({"node": nodes, "trophic_level": h})


@safe("trophic", {"trophic_incoherence_F0": float})
def trophic_incoherence(edges: pd.DataFrame,
                        levels: pd.DataFrame) -> pd.DataFrame:
    """Global incoherence F0 = Σ w_ij (h_j − h_i − 1)^2 / Σ w_ij  (log1p weights).
    F0→0: perfectly hierarchical (feed-forward); F0→1: maximally loopy."""
    lv = levels.set_index("node")["trophic_level"]
    hi = lv.reindex(edges["source"]).to_numpy()
    hj = lv.reindex(edges["dest"]).to_numpy()
    w = np.log1p(edges["amount"].to_numpy(dtype=np.float64))
    ok = ~(np.isnan(hi) | np.isnan(hj))
    F0 = float((w[ok] * (hj[ok] - hi[ok] - 1.0) ** 2).sum() / w[ok].sum())
    return pd.DataFrame({"trophic_incoherence_F0": [F0]})


# ============================================================================
# 3. DYAD-BASED WEIGHTED RECIPROCITY  (graph-, node-, and subset-level)
# ============================================================================

@safe("reciprocity", {"reciprocity_uw": float, "reciprocity_w": float})
def graph_reciprocity(edges: pd.DataFrame) -> pd.DataFrame:
    """Unweighted: fraction of edges whose reverse exists.
    Weighted (dyad-min): Σ_dyads 2·min(w_ij, w_ji) / Σ w  — the share of
    total amount that is 'matched' by an opposing flow. Raw amounts."""
    e = edges.groupby(["source", "dest"], as_index=False)["amount"].sum()
    rev = e.rename(columns={"source": "dest", "dest": "source",
                            "amount": "amount_rev"})
    m = e.merge(rev, on=["source", "dest"], how="left")
    has_rev = m["amount_rev"].notna()
    r_uw = float(has_rev.mean())
    matched = np.minimum(m["amount"], m["amount_rev"].fillna(0.0)).sum()
    r_w = float(matched / m["amount"].sum())
    return pd.DataFrame({"reciprocity_uw": [r_uw], "reciprocity_w": [r_w]})


@safe("reciprocity", {"node": object, "reciprocity_node_w": float})
def node_reciprocity(edges: pd.DataFrame) -> pd.DataFrame:
    """Per node i: Σ_j min(w_ij, w_ji) / Σ_j (w_ij + w_ji), raw amounts."""
    e = edges.groupby(["source", "dest"], as_index=False)["amount"].sum()
    rev = e.rename(columns={"source": "dest", "dest": "source",
                            "amount": "amount_rev"})
    m = e.merge(rev, on=["source", "dest"], how="outer")
    m[["amount", "amount_rev"]] = m[["amount", "amount_rev"]].fillna(0.0)
    m["mn"] = np.minimum(m["amount"], m["amount_rev"])
    m["tot"] = m["amount"] + m["amount_rev"]
    g = m.groupby("source")[["mn", "tot"]].sum()
    out = (g["mn"] / g["tot"]).rename("reciprocity_node_w").reset_index()
    return out.rename(columns={"source": "node"})


# ============================================================================
# 4. PARTICIPATION COEFFICIENT, WITHIN-MODULE Z, GUIMERÀ–AMARAL ROLES
# ============================================================================

_GA_SCHEMA = {"node": object, "within_module_z": float,
              "participation_coef": float, "ga_role": object}


@safe("participation", _GA_SCHEMA)
def participation_and_roles(edges: pd.DataFrame,
                            partition: pd.DataFrame,
                            weighted: bool = False) -> pd.DataFrame:
    """partition: DataFrame [node, community_id].

    - k_i,c : node i's (weighted) degree into community c (both directions)
    - P_i   = 1 − Σ_c (k_i,c / k_i)^2
    - z_i   = (k_i,in-own − mean_own) / std_own   (within own community)
    - Guimerà–Amaral roles R1–R7 from the (z, P) plane (Nature 2005 cutoffs).
    """
    part = partition.set_index("node")["community_id"]
    wcol = edges["amount"].to_numpy(float) if weighted else np.ones(len(edges))
    # both directions: stack (node, neighbor_comm, w)
    a = pd.DataFrame({"node": edges["source"],
                      "nbr_comm": part.reindex(edges["dest"]).to_numpy(),
                      "w": wcol})
    b = pd.DataFrame({"node": edges["dest"],
                      "nbr_comm": part.reindex(edges["source"]).to_numpy(),
                      "w": wcol})
    st = pd.concat([a, b], ignore_index=True).dropna(subset=["nbr_comm"])
    k_ic = st.groupby(["node", "nbr_comm"])["w"].sum()
    k_i = k_ic.groupby("node").sum()
    P = 1.0 - ((k_ic / k_i) ** 2).groupby("node").sum()

    own = part.reindex(k_ic.index.get_level_values("node")).to_numpy()
    is_own = own == k_ic.index.get_level_values("nbr_comm").to_numpy()
    k_own = k_ic[is_own].groupby("node").sum().reindex(k_i.index, fill_value=0.0)
    comm = part.reindex(k_i.index)
    stats = k_own.groupby(comm).agg(["mean", "std"])
    mu = stats["mean"].reindex(comm).to_numpy()
    sd = stats["std"].reindex(comm).to_numpy()
    z = np.where(np.nan_to_num(sd) > 0, (k_own.to_numpy() - mu) / sd, 0.0)

    Pv = P.reindex(k_i.index).to_numpy()
    role = np.empty(len(k_i), dtype=object)
    nh = z < 2.5
    role[nh & (Pv <= 0.05)] = "R1_ultra_peripheral"
    role[nh & (Pv > 0.05) & (Pv <= 0.62)] = "R2_peripheral"
    role[nh & (Pv > 0.62) & (Pv <= 0.80)] = "R3_non_hub_connector"
    role[nh & (Pv > 0.80)] = "R4_non_hub_kinless"
    hb = ~nh
    role[hb & (Pv <= 0.30)] = "R5_provincial_hub"
    role[hb & (Pv > 0.30) & (Pv <= 0.75)] = "R6_connector_hub"
    role[hb & (Pv > 0.75)] = "R7_kinless_hub"
    return pd.DataFrame({"node": k_i.index, "within_module_z": z,
                         "participation_coef": Pv, "ga_role": role})


# ============================================================================
# 5. FOUR-WAY DIRECTED WEIGHTED ASSORTATIVITY
# ============================================================================

_ASSORT_SCHEMA = {"assort_in_in": float, "assort_in_out": float,
                  "assort_out_in": float, "assort_out_out": float}


@safe("assortativity", _ASSORT_SCHEMA)
def directed_assortativity(edges: pd.DataFrame) -> pd.DataFrame:
    """Amount-weighted Pearson correlation, over edges (i→j), of the four
    endpoint-strength pairings (s^α_i, s^β_j), α,β ∈ {in,out}. Raw amounts,
    log10-transformed strengths (heavy tails would otherwise swamp Pearson).
    """
    s_out = edges.groupby("source")["amount"].sum()
    s_in = edges.groupby("dest")["amount"].sum()

    def s(series, keys):
        return np.log10(series.reindex(keys).fillna(0.0).to_numpy() + 1.0)

    w = edges["amount"].to_numpy(float)

    def wcorr(x, y):
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.average((x - mx) * (y - my), weights=w)
        vx = np.average((x - mx) ** 2, weights=w)
        vy = np.average((y - my) ** 2, weights=w)
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan

    si_src = s(s_in, edges["source"]); so_src = s(s_out, edges["source"])
    si_dst = s(s_in, edges["dest"]);  so_dst = s(s_out, edges["dest"])
    return pd.DataFrame({
        "assort_in_in":  [wcorr(si_src, si_dst)],
        "assort_in_out": [wcorr(si_src, so_dst)],
        "assort_out_in": [wcorr(so_src, si_dst)],
        "assort_out_out": [wcorr(so_src, so_dst)],
    })


# ============================================================================
# 6. TAIL STATISTICS — Gini, Hill estimator, top-share
# ============================================================================

_TAIL_SCHEMA = {"gini": float, "hill_alpha": float, "top_share": float,
                "top_frac": float}


@safe("tail_stats", _TAIL_SCHEMA)
def tail_stats(values: pd.Series, hill_tail_frac: float = 0.05,
               top_frac: float = 0.001) -> pd.DataFrame:
    """Gini coefficient, Hill tail index (top `hill_tail_frac`), and the
    share of total held by the top `top_frac` of entities."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[x > 0]
    n = len(x)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))
    k = max(int(n * hill_tail_frac), 10)
    tail = x[-k:]
    mean_log = np.mean(np.log(tail / tail[0])) if tail[0] > 0 else 0.0
    hill = float(1.0 / mean_log) if mean_log > 0 else np.nan
    m = max(int(np.ceil(n * top_frac)), 1)
    top_share = float(x[-m:].sum() / x.sum())
    return pd.DataFrame({"gini": [gini], "hill_alpha": [hill],
                         "top_share": [top_share], "top_frac": [top_frac]})


# ============================================================================
# 7. WEIGHTED RICH-CLUB COEFFICIENT
# ============================================================================

@safe("rich_club", {"rank_frac": float, "rich_club_w": float, "n_rich": float})
def weighted_rich_club(edges: pd.DataFrame,
                       rank_fracs=(0.01, 0.001)) -> pd.DataFrame:
    """φ_w(r) = W_rich / Σ(top E_rich edge weights), where 'rich' = top r
    fraction of nodes by total strength, W_rich = amount on edges among them,
    E_rich = # of edges among them (Opsahl et al. normalization). φ→1 means
    the elite reserve their strongest ties for each other."""
    s = (edges.groupby("source")["amount"].sum()
         .add(edges.groupby("dest")["amount"].sum(), fill_value=0.0))
    w_sorted = np.sort(edges["amount"].to_numpy(float))[::-1]
    rows = []
    for r in rank_fracs:
        k = max(int(len(s) * r), 2)
        rich = set(s.nlargest(k).index)
        mask = edges["source"].isin(rich) & edges["dest"].isin(rich)
        e_rich = int(mask.sum())
        if e_rich == 0:
            rows.append((r, np.nan, k)); continue
        phi = float(edges.loc[mask, "amount"].sum() / w_sorted[:e_rich].sum())
        rows.append((r, phi, k))
    return pd.DataFrame(rows, columns=["rank_frac", "rich_club_w", "n_rich"])


# ============================================================================
# 8. PARTITION COMPARISON — NMI (arithmetic) & ARI
# ============================================================================

@safe("partition_compare", {"nmi": float, "ari": float})
def compare_partitions(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    """NMI (arithmetic normalization) and Adjusted Rand Index between two
    partitions [node, community_id], computed on the node intersection."""
    m = p1.merge(p2, on="node", suffixes=("_1", "_2"))
    n = len(m)
    ct = m.groupby(["community_id_1", "community_id_2"]).size()
    nij = ct.to_numpy(float)
    ai = ct.groupby(level=0).sum().to_numpy(float)
    bj = ct.groupby(level=1).sum().to_numpy(float)
    pi, pj, pij = ai / n, bj / n, nij / n
    Hi = -(pi * np.log(pi)).sum()
    Hj = -(pj * np.log(pj)).sum()
    pi_of = (ct.groupby(level=0).sum() / n).reindex(
        ct.index.get_level_values(0)).to_numpy()
    pj_of = (ct.groupby(level=1).sum() / n).reindex(
        ct.index.get_level_values(1)).to_numpy()
    I = (pij * np.log(pij / (pi_of * pj_of))).sum()
    nmi = float(I / ((Hi + Hj) / 2)) if (Hi + Hj) > 0 else 1.0

    def c2(x): return (x * (x - 1) / 2).sum()
    sum_ij, sum_a, sum_b, C = c2(nij), c2(ai), c2(bj), n * (n - 1) / 2
    exp = sum_a * sum_b / C
    mx = (sum_a + sum_b) / 2
    ari = float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0
    return pd.DataFrame({"nmi": [nmi], "ari": [ari]})


# ============================================================================
# 9. COMMUNITY LIFECYCLE — Jaccard best-match events
# ============================================================================

_LIFE_SCHEMA = {"comm_prev": object, "comm_curr": object, "jaccard": float,
                "event": object}


@safe("lifecycle", _LIFE_SCHEMA)
def community_lifecycle(prev: pd.DataFrame, curr: pd.DataFrame,
                        tau: float = 0.3,
                        growth_thresh: float = 0.2) -> pd.DataFrame:
    """Match communities across consecutive months by membership Jaccard.
    Events: continuation / growth / shrinkage / merge / split / birth / death.
    prev, curr: [node, community_id]."""
    ov = (prev.merge(curr, on="node", suffixes=("_p", "_c"))
          .groupby(["community_id_p", "community_id_c"]).size()
          .rename("inter").reset_index())
    sz_p = prev.groupby("community_id").size()
    sz_c = curr.groupby("community_id").size()
    ov["jaccard"] = ov["inter"] / (
        sz_p.reindex(ov["community_id_p"]).to_numpy()
        + sz_c.reindex(ov["community_id_c"]).to_numpy() - ov["inter"])
    ov = ov[ov["jaccard"] >= tau].copy()

    fwd = ov.groupby("community_id_p")["community_id_c"].nunique()
    bwd = ov.groupby("community_id_c")["community_id_p"].nunique()
    rows = []
    for _, r in ov.iterrows():
        p, c, j = r["community_id_p"], r["community_id_c"], r["jaccard"]
        if fwd[p] > 1:
            ev = "split"
        elif bwd[c] > 1:
            ev = "merge"
        else:
            ratio = sz_c[c] / sz_p[p]
            ev = ("growth" if ratio > 1 + growth_thresh
                  else "shrinkage" if ratio < 1 - growth_thresh
                  else "continuation")
        rows.append((p, c, j, ev))
    matched_p = set(ov["community_id_p"]); matched_c = set(ov["community_id_c"])
    rows += [(p, None, np.nan, "death") for p in sz_p.index if p not in matched_p]
    rows += [(None, c, np.nan, "birth") for c in sz_c.index if c not in matched_c]
    return pd.DataFrame(rows, columns=["comm_prev", "comm_curr",
                                       "jaccard", "event"])


# ============================================================================
# 10. ABLATION LADDER BUILDER — exclusion lists from 23-snapshot aggregate
# ============================================================================

@dataclass
class Ladder:
    """Exclusion sets for V1–V4 plus the thresholds used (for audit)."""
    v1_degree_broad: set = field(default_factory=set)
    v2_degree_strict: set = field(default_factory=set)
    v3_strength_broad: set = field(default_factory=set)
    v4_combined: set = field(default_factory=set)
    thresholds: dict = field(default_factory=dict)

    def exclusions(self, version: str) -> set:
        return {"V0": set(), "V1": self.v1_degree_broad,
                "V2": self.v2_degree_strict, "V3": self.v3_strength_broad,
                "V4": self.v4_combined}[version]


def build_ladder(snapshot_paths: list[str],
                 p_broad: float = 99.9, p_strict: float = 99.99) -> Ladder:
    """Aggregate all snapshots, compute per-node total degree (distinct
    counterparts, both directions) and total strength (in+out amount), then
    derive data-driven P99.9 / P99.99 exclusion sets. Streams file-by-file
    to keep memory flat."""
    deg_parts, str_parts = [], []
    for p in snapshot_paths:
        e = pd.read_csv(p, usecols=["source", "dest", "amount"])
        d = pd.concat([
            e.groupby("source")["dest"].nunique().rename("deg"),
            e.groupby("dest")["source"].nunique().rename("deg")])
        deg_parts.append(d.groupby(level=0).sum())
        s = pd.concat([e.groupby("source")["amount"].sum(),
                       e.groupby("dest")["amount"].sum()])
        str_parts.append(s.groupby(level=0).sum())
    deg = pd.concat(deg_parts).groupby(level=0).sum()
    stg = pd.concat(str_parts).groupby(level=0).sum()

    t_deg_b = np.percentile(deg, p_broad)
    t_deg_s = np.percentile(deg, p_strict)
    t_str_b = np.percentile(stg, p_broad)
    lad = Ladder(
        v1_degree_broad=set(deg[deg > t_deg_b].index),
        v2_degree_strict=set(deg[deg > t_deg_s].index),
        v3_strength_broad=set(stg[stg > t_str_b].index),
        thresholds={"degree_p_broad": float(t_deg_b),
                    "degree_p_strict": float(t_deg_s),
                    "strength_p_broad": float(t_str_b)},
    )
    lad.v4_combined = lad.v1_degree_broad | lad.v3_strength_broad
    log.info("ladder: |V1|=%d |V2|=%d |V3|=%d |V4|=%d, thresholds=%s",
             len(lad.v1_degree_broad), len(lad.v2_degree_strict),
             len(lad.v3_strength_broad), len(lad.v4_combined), lad.thresholds)
    return lad


def apply_version(edges: pd.DataFrame, ladder: Ladder,
                  version: str) -> pd.DataFrame:
    """Filter a snapshot edge list to graph version V0..V4."""
    excl = ladder.exclusions(version)
    if not excl:
        return edges
    return edges[~edges["source"].isin(excl)
                 & ~edges["dest"].isin(excl)].reset_index(drop=True)
