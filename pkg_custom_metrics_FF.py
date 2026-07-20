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

    fwd = ov.groupby("community_id_p")["community_id_c"].transform("nunique")
    bwd = ov.groupby("community_id_c")["community_id_p"].transform("nunique")
    ratio = (sz_c.reindex(ov["community_id_c"]).to_numpy()
             / sz_p.reindex(ov["community_id_p"]).to_numpy())
    event = np.select(
        [fwd.to_numpy() > 1, bwd.to_numpy() > 1,
         ratio > 1 + growth_thresh, ratio < 1 - growth_thresh],
        ["split", "merge", "growth", "shrinkage"],
        default="continuation")
    res = pd.DataFrame({"comm_prev": ov["community_id_p"].to_numpy(),
                        "comm_curr": ov["community_id_c"].to_numpy(),
                        "jaccard": ov["jaccard"].to_numpy(),
                        "event": event})
    matched_p = set(ov["community_id_p"]); matched_c = set(ov["community_id_c"])
    deaths = pd.DataFrame({"comm_prev": [p for p in sz_p.index
                                         if p not in matched_p]})
    deaths["comm_curr"], deaths["jaccard"], deaths["event"] = \
        None, np.nan, "death"
    births = pd.DataFrame({"comm_curr": [c for c in sz_c.index
                                         if c not in matched_c]})
    births["comm_prev"], births["jaccard"], births["event"] = \
        None, np.nan, "birth"
    return pd.concat([res, deaths, births], ignore_index=True)[
        ["comm_prev", "comm_curr", "jaccard", "event"]]


# ============================================================================
# 10. ABLATION LADDER BUILDER — exclusion lists from 23-snapshot aggregate
# ============================================================================

@dataclass
class Ladder:
    """Exclusion sets keyed by version label (plus implicit 'V0' = raw).
    Each set is the UNION of a degree criterion and a strength criterion at
    one percentile — the two populations overlap only partially (~60% in
    the 2024–2025 PKG aggregate), so single-criterion exclusion misses a
    material share of dominant nodes.

    Default tiers:  P99  |  P99_9  |  P99_99
    (exclude nodes with aggregate degree OR strength above that pctile)
    """
    exclusion_sets: dict = field(default_factory=dict)   # label -> set
    thresholds: dict = field(default_factory=dict)

    def exclusions(self, version: str) -> set:
        return self.exclusion_sets.get(version, set())

    @property
    def versions(self) -> list:
        return ["V0"] + list(self.exclusion_sets)


def _pct_label(pct: float) -> str:
    return "P" + ("%g" % pct).replace(".", "_")


def build_ladder(snapshot_paths: list[str],
                 percentiles=(99.0, 99.9, 99.99)) -> Ladder:
    """Aggregate all snapshots, compute per-node total degree (distinct
    counterparts, both directions) and total strength (in+out amount), then
    derive one union exclusion set per percentile tier. Streams
    file-by-file to keep memory flat. Logs each tier's degree/strength
    overlap so the union-criterion choice stays auditable."""
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

    lad = Ladder()
    for pct in sorted(percentiles):
        label = _pct_label(pct)
        td, ts = np.percentile(deg, pct), np.percentile(stg, pct)
        dset = set(deg[deg > td].index)
        sset = set(stg[stg > ts].index)
        lad.exclusion_sets[label] = dset | sset
        lad.thresholds[f"degree_{label}"] = float(td)
        lad.thresholds[f"strength_{label}"] = float(ts)
        log.info("ladder %s: |deg|=%d |str|=%d overlap=%d union=%d",
                 label, len(dset), len(sset), len(dset & sset),
                 len(dset | sset))
    log.info("thresholds=%s", lad.thresholds)
    return lad


def apply_version(edges: pd.DataFrame, ladder: Ladder,
                  version: str) -> pd.DataFrame:
    """Filter a snapshot edge list to graph version V0..V4."""
    excl = ladder.exclusions(version)
    if not excl:
        return edges
    return edges[~edges["source"].isin(excl)
                 & ~edges["dest"].isin(excl)].reset_index(drop=True)


# ============================================================================
# 11. COUNTERPARTY COMPOSITION — entity typing & NAICS-status decomposition
#     (see "PKG — Counterparty Composition Metrics" spec)
# ============================================================================

# Placeholder NAICS values: CONFIGURABLE — enumerate via EDA on distinct
# NAICS values before the production run and extend this list.
PLACEHOLDER_NAICS = {"-1", "-1|UNKNOWN", "******", ""}

_BUSINESS_TOKENS = {
    "LLC", "INC", "CORP", "CORPORATION", "LTD", "LP", "LLP", "PLLC", "PC",
    "CO", "COMPANY", "TRUST", "DBA", "ASSOC", "ASSOCIATES", "ASSOCIATION",
    "GROUP", "HOLDINGS", "ENTERPRISES", "ENTERPRISE", "FOUNDATION",
    "CHURCH", "SCHOOL", "UNIVERSITY", "CITY", "COUNTY", "BANK", "PARTNERS",
    "SERVICES", "SOLUTIONS", "CONSTRUCTION", "PROPERTIES", "REALTY",
    "MEDICAL", "DENTAL", "LAW", "CLINIC", "FARMS", "&",
}
_PUNCT_RE = None  # compiled lazily


def _tokenize_names(names: pd.Series) -> pd.Series:
    import re as _re
    global _PUNCT_RE
    if _PUNCT_RE is None:
        _PUNCT_RE = _re.compile(r"[^A-Z0-9& ]+")
    up = names.fillna("").astype(str).str.upper()
    return up.map(lambda s: _PUNCT_RE.sub(" ", s).split())


def build_node_typing(snapshot_paths: list[str],
                      placeholder_naics: set | None = None) -> pd.DataFrame:
    """Node typing table, computed ONCE from the full node universe across
    all snapshots. Streams file-by-file. Returns one row per node:

        node, entity_type, naics_status, node_type, naics_clean

    naics_status: valid | placeholder | missing
    entity_type : business | individual | unknown  (token-based name typer;
                  no fuzzy per-pair scoring at 5M-node scale)
    node_type   : precedence per spec — placeholder wins over name because
                  a populated NAICS field implies business onboarding.
    naics_observed stays the source of truth; NEVER feed imputed NAICS back
    into composition metrics (circularity).
    """
    ph = placeholder_naics or PLACEHOLDER_NAICS
    parts = []
    for p in snapshot_paths:
        e = pd.read_csv(p, usecols=["source", "source_name", "source_naics",
                                    "dest", "dest_name", "dest_naics"],
                        dtype=str)
        parts.append(pd.concat([
            e[["source", "source_name", "source_naics"]].rename(columns={
                "source": "node", "source_name": "name",
                "source_naics": "naics"}),
            e[["dest", "dest_name", "dest_naics"]].rename(columns={
                "dest": "node", "dest_name": "name",
                "dest_naics": "naics"}),
        ], ignore_index=True).drop_duplicates("node"))
    nodes = (pd.concat(parts, ignore_index=True)
             .drop_duplicates("node").reset_index(drop=True))

    # --- naics_status -------------------------------------------------------
    naics = nodes["naics"].astype("string").str.strip()
    # NOTE: ops on nullable string dtype return nullable BooleanDtype whose
    # .to_numpy() is an OBJECT array on some pandas versions — np.select
    # then raises "invalid entry in condlist". Coerce every condition to a
    # plain numpy bool array explicitly.
    def _b(cond) -> np.ndarray:
        return cond.fillna(False).to_numpy(dtype=bool)

    is_missing = _b(naics.isna() | (naics == ""))
    is_ph = _b(naics.fillna("").isin(ph)) & ~is_missing
    # a valid NAICS must start with >=2 digits and not be a placeholder
    looks_code = _b(naics.fillna("").str.match(r"^\d{2}"))
    naics_status = np.select(
        [is_missing, is_ph, looks_code],
        ["missing", "placeholder", "valid"], default="placeholder")
    ph_count = int((naics_status == "placeholder").sum())
    log.info("node typing: %d nodes | placeholder NAICS matched %d "
             "(sanity check the PLACEHOLDER_NAICS list via EDA)",
             len(nodes), ph_count)

    # --- entity_type (vectorized token typer) --------------------------------
    toks = _tokenize_names(nodes["name"])
    has_biz = toks.map(
        lambda t: any(w in _BUSINESS_TOKENS for w in t)).to_numpy(dtype=bool)
    n_alpha = toks.map(lambda t: sum(w.isalpha() for w in t)).to_numpy(int)
    n_tok = toks.map(len).to_numpy(int)
    person_shaped = (~has_biz) & (n_alpha == n_tok) & (n_tok >= 2) \
        & (n_tok <= 3)
    entity_type = np.select(
        [has_biz, person_shaped],
        ["business", "individual"], default="unknown")

    # --- node_type precedence -------------------------------------------------
    node_type = np.select(
        [naics_status == "valid",
         naics_status == "placeholder",
         (naics_status == "missing") & (entity_type == "business"),
         (naics_status == "missing") & (entity_type == "individual")],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual"],
        default="unknown")

    out = pd.DataFrame({
        "node": nodes["node"], "entity_type": entity_type,
        "naics_status": naics_status, "node_type": node_type,
        "naics_clean": naics.where(naics_status == "valid"),
    })
    log.info("node_type counts: %s",
             out["node_type"].value_counts().to_dict())
    return out


_CAT_SHORT = {"individual": "individual",
              "business_naics_valid": "biz_valid",
              "business_naics_placeholder": "biz_placeholder",
              "business_naics_missing": "biz_missing",
              "unknown": "unknown", "hub": "hub"}


def composition_metrics(edges: pd.DataFrame, typing: pd.DataFrame,
                        hub_set: set | None = None,
                        min_valid_naics: int = 5) -> pd.DataFrame:
    """Per-node counterparty composition on the RAW (V0) graph.
    Self-edges excluded. For d in {in, out} and w in {cp, amt}:

      share_{d}_{w}_{cat}   — composition shares over node_type categories
                              (sixth 'hub' category when hub_set given, so
                              processor-intermediated flow is visible rather
                              than miscounted as biz)
      naics_coverage_biz_{d} — biz_valid / all business counterparties
                              (cp-counted; DATA QUALITY, not behavior)
      naics2_entropy_{d}, naics2_top_share_{d}, n_naics2_{d}
                            — amount-weighted industry mix over valid-NAICS
                              counterparties; NaN when < min_valid_naics.

    Zero in- or out-edges in the month -> NaN for that direction's columns.
    """
    e = edges.loc[edges["source"] != edges["dest"],
                  ["source", "dest", "amount"]].copy()
    tmap = typing.set_index("node")
    ntype = tmap["node_type"]
    n2 = tmap["naics_clean"].str[:2]

    frames = []
    for d, node_col, cp_col in (("in", "dest", "source"),
                                ("out", "source", "dest")):
        cat = ntype.reindex(e[cp_col]).fillna("unknown").to_numpy(object)
        if hub_set:
            cat = np.where(e[cp_col].isin(hub_set), "hub", cat)
        df = pd.DataFrame({"node": e[node_col].to_numpy(),
                           "cp": e[cp_col].to_numpy(),
                           "cat": cat,
                           "amount": e["amount"].to_numpy(float)})
        # cp-weighted (distinct counterparties) and amt-weighted pivots
        cpd = df.drop_duplicates(["node", "cp"])
        cnt = (cpd.groupby(["node", "cat"]).size()
               .unstack(fill_value=0))
        amt = (df.groupby(["node", "cat"])["amount"].sum()
               .unstack(fill_value=0.0))
        out = pd.DataFrame(index=cnt.index)
        for w, piv in (("cp", cnt), ("amt", amt)):
            tot = piv.sum(axis=1)
            for cat_name, short in _CAT_SHORT.items():
                if cat_name == "hub" and not hub_set:
                    continue
                col = (piv[cat_name] if cat_name in piv.columns
                       else pd.Series(0.0, index=piv.index))
                out[f"share_{d}_{w}_{short}"] = col / tot
        # enrichment coverage (cp-counted, business categories only)
        biz_cols = [c for c in ("business_naics_valid",
                                "business_naics_placeholder",
                                "business_naics_missing")
                    if c in cnt.columns]
        biz_tot = cnt[biz_cols].sum(axis=1) if biz_cols else 0
        valid = (cnt["business_naics_valid"]
                 if "business_naics_valid" in cnt.columns else 0)
        out[f"naics_coverage_biz_{d}"] = np.where(
            np.asarray(biz_tot) > 0, np.asarray(valid) / np.asarray(biz_tot),
            np.nan)
        # industry mix over valid-NAICS counterparties, amount-weighted
        dfv = df.assign(n2=n2.reindex(df["cp"]).to_numpy()).dropna(
            subset=["n2"])
        nv = dfv.drop_duplicates(["node", "cp"]).groupby("node").size()
        g = dfv.groupby(["node", "n2"])["amount"].sum()
        totv = g.groupby(level=0).sum()
        p = g / totv
        ent = (-(p * np.log2(p))).groupby(level=0).sum()
        top = p.groupby(level=0).max()
        ncodes = g.groupby(level=0).size()
        ok = nv[nv >= min_valid_naics].index
        out[f"naics2_entropy_{d}"] = ent.where(ent.index.isin(ok)).reindex(
            out.index)
        out[f"naics2_top_share_{d}"] = top.where(
            top.index.isin(ok)).reindex(out.index)
        out[f"n_naics2_{d}"] = ncodes.where(
            ncodes.index.isin(ok)).reindex(out.index)
        frames.append(out)
    # outer join: node with zero edges in a direction -> NaN for that side
    res = frames[0].join(frames[1], how="outer")
    res.index.name = "node"
    return res.reset_index()
