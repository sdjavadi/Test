"""
pkg_custom_metrics.py
=====================
Custom network metrics for the PKG monthly snapshot suite — everything the
manifest requires that cuGraph does not provide.

GPU-first (cuDF / CuPy sparse), transparent CPU fallback (pandas / NumPy /
SciPy sparse). Edge weight is AMOUNT only; `volume` is descriptive.

Edge frame schema (post pair-aggregation, self-loops removed):
    source, dest, amount, volume, edge_count
Optional columns: source_naics, dest_naics, source_name, dest_name.

All public functions are wrapped by `safe()` — a failure returns a
NaN/empty result and logs, never aborts the monthly run. Feature flags in
FLAGS gate expensive blocks.

Smoke test: `python pkg_custom_metrics.py` (runs on CPU fallback).
"""

from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass, field
from functools import wraps

log = logging.getLogger("pkg_custom_metrics")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #
try:  # GPU stack
    import cudf as xdf
    import cupy as xp
    from cupyx.scipy import sparse as xsp
    from cupyx.scipy.sparse.linalg import cg as _cg
    GPU = True
except Exception:  # CPU fallback
    import pandas as xdf
    import numpy as xp
    from scipy import sparse as xsp
    from scipy.sparse.linalg import cg as _cg
    GPU = False

import numpy as np  # always available, host-side
import pandas as pd

log.info("backend: %s", "GPU (cuDF/CuPy)" if GPU else "CPU (pandas/NumPy/SciPy)")


def to_host(df):
    """cuDF -> pandas passthrough."""
    return df.to_pandas() if GPU and hasattr(df, "to_pandas") else df


def to_host_arr(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


# --------------------------------------------------------------------------- #
# safe() isolation + feature flags
# --------------------------------------------------------------------------- #
FLAGS = {
    "weighted_hits": True,
    "trophic": True,
    "reciprocity": True,
    "community_mixing": True,
    "participation_roles": True,
    "assortativity": True,
    "tail_stats": True,
    "rich_club": True,
    "partition_compare": True,
    "edge_turnover": True,
    "concentration": True,
}


def safe(flag: str | None = None):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if flag is not None and not FLAGS.get(flag, True):
                log.info("skipped (flag off): %s", fn.__name__)
                return None
            try:
                return fn(*args, **kwargs)
            except Exception:
                log.error("FAILED %s\n%s", fn.__name__, traceback.format_exc())
                return None
        return wrapped
    return deco


# --------------------------------------------------------------------------- #
# Graph assembly helpers
# --------------------------------------------------------------------------- #
@dataclass
class SparseGraph:
    """Directed weighted graph in CSR with a node index map."""
    A: object                 # csr_matrix, A[i, j] = w(i -> j)
    nodes: pd.Index           # position -> node id
    n: int = field(init=False)

    def __post_init__(self):
        self.n = self.A.shape[0]


def build_sparse(edges, weight_col="amount", log1p=False) -> SparseGraph:
    """Edge frame -> CSR adjacency. Uses host-side factorize for the index."""
    e = to_host(edges)
    all_ids = pd.Index(pd.unique(pd.concat([e["source"], e["dest"]], ignore_index=True)))
    src = all_ids.get_indexer(e["source"])
    dst = all_ids.get_indexer(e["dest"])
    w = e[weight_col].to_numpy(dtype="float64")
    if log1p:
        w = np.log1p(w)
    n = len(all_ids)
    A_host = __import__("scipy.sparse", fromlist=["sparse"]).csr_matrix(
        (w, (src, dst)), shape=(n, n)
    )
    A = xsp.csr_matrix(A_host) if GPU else A_host
    return SparseGraph(A=A, nodes=all_ids)


# --------------------------------------------------------------------------- #
# 6.1 Weighted HITS
# --------------------------------------------------------------------------- #
@safe("weighted_hits")
def weighted_hits(edges, log1p=True, max_iter=200, tol=1e-9) -> pd.DataFrame:
    """
    Weighted HITS via power iteration on W = adjacency of log1p(amount).
        a <- W^T h ; h <- W a ; L2-normalize each step.
    Returns DataFrame: node, hits_hub, hits_auth.
    """
    g = build_sparse(edges, log1p=log1p)
    W, WT = g.A, g.A.T.tocsr()
    h = xp.ones(g.n, dtype=xp.float64) / math.sqrt(g.n)
    a = xp.ones(g.n, dtype=xp.float64) / math.sqrt(g.n)
    for it in range(max_iter):
        a_new = WT @ h
        na = xp.linalg.norm(a_new)
        a_new = a_new / na if float(na) > 0 else a_new
        h_new = W @ a_new
        nh = xp.linalg.norm(h_new)
        h_new = h_new / nh if float(nh) > 0 else h_new
        delta = float(xp.abs(h_new - h).sum() + xp.abs(a_new - a).sum())
        h, a = h_new, a_new
        if delta < tol:
            log.info("weighted_hits converged @ iter %d (delta=%.2e)", it, delta)
            break
    return pd.DataFrame(
        {"node": g.nodes, "hits_hub": to_host_arr(h), "hits_auth": to_host_arr(a)}
    )


# --------------------------------------------------------------------------- #
# 6.2 Trophic levels & incoherence (MacKay-Johnson-Rogers 2020)
# --------------------------------------------------------------------------- #
@safe("trophic")
def trophic_levels(edges, weight_col="amount", tol=1e-8, max_iter=2000):
    """
    MJR 2020 trophic levels on a weighted digraph.
        u_i = s_in(i) + s_out(i)      (weighted total degree)
        v_i = s_in(i) - s_out(i)      (weighted imbalance)
        Lambda = diag(u) - W - W^T    (symmetric graph Laplacian form)
        Solve Lambda h = v  (singular; fix gauge by mean-centering h)
    Returns (node_df, F0):
        node_df: node, trophic_level, trophic_incoherence_local
        F0: global incoherence = sum_e w_e (h_j - h_i - 1)^2 / sum_e w_e
    """
    g = build_sparse(edges, weight_col=weight_col, log1p=False)
    W = g.A
    s_out = xp.asarray(W.sum(axis=1)).ravel()
    s_in = xp.asarray(W.sum(axis=0)).ravel()
    u = s_in + s_out
    v = s_in - s_out
    S = (W + W.T).tocsr()
    L = xsp.diags(u) - S
    # Regularize the singular system slightly for CG stability, then re-center.
    L_reg = L + xsp.identity(g.n, format="csr") * 1e-10
    h, info = _cg(L_reg, v, rtol=tol, maxiter=max_iter)
    if info != 0:
        log.warning("trophic CG did not fully converge (info=%s)", info)
    h = h - h.mean()
    h = h - h.min()  # shift so basal nodes sit near 0 (presentation convention)

    # Edge-level incoherence
    coo = W.tocoo()
    diff = h[coo.col] - h[coo.row] - 1.0
    w = coo.data
    F0 = float((w * diff**2).sum() / w.sum()) if float(w.sum()) > 0 else float("nan")

    # Node-local incoherence: weighted mean of (diff^2) over incident edges
    num = xp.zeros(g.n)
    den = xp.zeros(g.n)
    contrib = w * diff**2
    if GPU:
        import cupyx
        cupyx.scatter_add(num, coo.row, contrib)
        cupyx.scatter_add(num, coo.col, contrib)
        cupyx.scatter_add(den, coo.row, w)
        cupyx.scatter_add(den, coo.col, w)
    else:
        np.add.at(num, coo.row, contrib)
        np.add.at(num, coo.col, contrib)
        np.add.at(den, coo.row, w)
        np.add.at(den, coo.col, w)
    local = xp.where(den > 0, num / xp.maximum(den, 1e-300), xp.nan)

    node_df = pd.DataFrame(
        {
            "node": g.nodes,
            "trophic_level": to_host_arr(h),
            "trophic_incoherence_local": to_host_arr(local),
        }
    )
    return node_df, F0


# --------------------------------------------------------------------------- #
# 6.3 Weighted reciprocity (global + per node)
# --------------------------------------------------------------------------- #
@safe("reciprocity")
def weighted_reciprocity(edges):
    """
    Dyad-based weighted reciprocity.
      global:  sum_dyads min(w_uv, w_vu) / sum_dyads max(w_uv, w_vu)
      node u:  sum_v min(w_uv, w_vu) / sum_v max(w_uv, w_vu)
      also unweighted node reciprocity: fraction of neighbors with both edges.
    Returns (node_df, global_w, global_u).
    """
    e = to_host(edges)[["source", "dest", "amount"]].copy()
    fwd = e.rename(columns={"source": "u", "dest": "v", "amount": "w_uv"})
    rev = e.rename(columns={"source": "v", "dest": "u", "amount": "w_vu"})
    d = fwd.merge(rev, on=["u", "v"], how="outer")
    d[["w_uv", "w_vu"]] = d[["w_uv", "w_vu"]].fillna(0.0)
    d["mn"] = d[["w_uv", "w_vu"]].min(axis=1)
    d["mx"] = d[["w_uv", "w_vu"]].max(axis=1)
    d["both"] = ((d["w_uv"] > 0) & (d["w_vu"] > 0)).astype("float64")

    # Each unordered dyad appears twice in d (u,v) and (v,u): global sums halve out.
    global_w = float(d["mn"].sum() / d["mx"].sum()) if d["mx"].sum() > 0 else np.nan
    global_u = float(d["both"].sum() / len(d)) if len(d) else np.nan

    per = d.groupby("u", sort=False).agg(
        mn=("mn", "sum"), mx=("mx", "sum"), both=("both", "sum"), nn=("v", "count")
    )
    node_df = pd.DataFrame(
        {
            "node": per.index,
            "recip_w": (per["mn"] / per["mx"].replace(0, np.nan)).to_numpy(),
            "recip_u": (per["both"] / per["nn"]).to_numpy(),
        }
    ).reset_index(drop=True)
    return node_df, global_w, global_u


# --------------------------------------------------------------------------- #
# 6.4 Community mixing / conductance / internal reciprocity
# --------------------------------------------------------------------------- #
@safe("community_mixing")
def community_mixing(edges, membership: pd.DataFrame):
    """
    membership: DataFrame [node, community_id].
    Returns (community_df, node_df):
      community_df: community_id, n_nodes, n_internal_edges, internal_amount,
                    internal_volume, ext_out_amount, ext_in_amount,
                    mixing_ratio_w, mixing_ratio_u, density_directed,
                    weighted_density, conductance, internal_reciprocity_w
      node_df: node, intra_strength_ratio_w, intra_degree_ratio_u
    """
    e = to_host(edges).copy()
    m = to_host(membership).set_index("node")["community_id"]
    e["c_src"] = e["source"].map(m)
    e["c_dst"] = e["dest"].map(m)
    e = e.dropna(subset=["c_src", "c_dst"])
    e["internal"] = e["c_src"] == e["c_dst"]

    sizes = m.groupby(m).size().rename("n_nodes")

    internal = e[e["internal"]]
    ext = e[~e["internal"]]

    gi = internal.groupby("c_src").agg(
        n_internal_edges=("amount", "size"),
        internal_amount=("amount", "sum"),
        internal_volume=("volume", "sum") if "volume" in e else ("amount", "size"),
    )
    go = ext.groupby("c_src")["amount"].sum().rename("ext_out_amount")
    gin = ext.groupby("c_dst")["amount"].sum().rename("ext_in_amount")
    go_e = ext.groupby("c_src")["amount"].size().rename("ext_out_edges")
    gin_e = ext.groupby("c_dst")["amount"].size().rename("ext_in_edges")

    comm = (
        pd.DataFrame(sizes)
        .join([gi, go, gin, go_e, gin_e])
        .fillna(0.0)
        .rename_axis("community_id")
        .reset_index()
    )
    tot_amt = comm["internal_amount"] + comm["ext_out_amount"] + comm["ext_in_amount"]
    tot_edg = comm["n_internal_edges"] + comm["ext_out_edges"] + comm["ext_in_edges"]
    comm["mixing_ratio_w"] = np.where(tot_amt > 0, comm["internal_amount"] / tot_amt, np.nan)
    comm["mixing_ratio_u"] = np.where(tot_edg > 0, comm["n_internal_edges"] / tot_edg, np.nan)
    nn = comm["n_nodes"]
    denom = (nn * (nn - 1)).replace(0, np.nan) if hasattr(nn, "replace") else nn * (nn - 1)
    comm["density_directed"] = comm["n_internal_edges"] / denom
    comm["weighted_density"] = comm["internal_amount"] / denom

    # Conductance: cut / min(vol(C), vol(rest)); vol = incident amount.
    vol_c = 2 * comm["internal_amount"] + comm["ext_out_amount"] + comm["ext_in_amount"]
    total_vol = float(2 * e["amount"].sum())
    cut = comm["ext_out_amount"] + comm["ext_in_amount"]
    comm["conductance"] = cut / np.minimum(vol_c, total_vol - vol_c).clip(lower=1e-300)

    # Internal weighted reciprocity per community
    rec = []
    for cid, sub in internal.groupby("c_src", sort=False):
        r = weighted_reciprocity(sub[["source", "dest", "amount"]])
        rec.append((cid, r[1] if r is not None else np.nan))
    comm = comm.merge(
        pd.DataFrame(rec, columns=["community_id", "internal_reciprocity_w"]),
        on="community_id",
        how="left",
    )

    # Node embeddedness
    e["amt"] = e["amount"]
    out_int = e[e["internal"]].groupby("source")["amt"].sum()
    in_int = e[e["internal"]].groupby("dest")["amt"].sum()
    out_all = e.groupby("source")["amt"].sum()
    in_all = e.groupby("dest")["amt"].sum()
    intra_w = out_int.add(in_int, fill_value=0)
    all_w = out_all.add(in_all, fill_value=0)
    # unweighted: distinct neighbors
    pairs = e[["source", "dest", "internal"]].drop_duplicates()
    nb_int = (
        pd.concat(
            [
                pairs[pairs.internal].groupby("source")["dest"].nunique(),
                pairs[pairs.internal].groupby("dest")["source"].nunique(),
            ]
        )
        .groupby(level=0)
        .sum()
    )
    nb_all = (
        pd.concat(
            [pairs.groupby("source")["dest"].nunique(), pairs.groupby("dest")["source"].nunique()]
        )
        .groupby(level=0)
        .sum()
    )
    node_df = pd.DataFrame(
        {
            "node": all_w.index,
            "intra_strength_ratio_w": (intra_w.reindex(all_w.index).fillna(0) / all_w).to_numpy(),
            "intra_degree_ratio_u": (
                nb_int.reindex(all_w.index).fillna(0) / nb_all.reindex(all_w.index)
            ).to_numpy(),
        }
    ).reset_index(drop=True)
    return comm, node_df


# --------------------------------------------------------------------------- #
# 6.5 Participation coefficient, within-module z, Guimera-Amaral roles
# --------------------------------------------------------------------------- #
@safe("participation_roles")
def participation_z_roles(edges, membership: pd.DataFrame) -> pd.DataFrame:
    """
    Strength-based (amount) participation coefficient and within-module z.
    Roles per Guimera & Amaral (2005), z threshold 2.5:
      non-hubs: R1 ultra-peripheral P<=.05 | R2 peripheral <=.62
                | R3 connector <=.80 | R4 kinless >.80
      hubs:     R5 provincial P<=.30 | R6 connector <=.75 | R7 kinless >.75
    """
    e = to_host(edges)[["source", "dest", "amount"]].copy()
    m = to_host(membership).set_index("node")["community_id"]
    e["c_dst"] = e["dest"].map(m)
    e["c_src"] = e["source"].map(m)

    # strength of node u toward community c (out + in)
    out_c = e.groupby(["source", "c_dst"])["amount"].sum()
    in_c = e.groupby(["dest", "c_src"])["amount"].sum()
    out_c.index.names = in_c.index.names = ["node", "comm"]
    s_c = out_c.add(in_c, fill_value=0).rename("s_c").reset_index()

    s_tot = s_c.groupby("node")["s_c"].sum().rename("s_tot")
    s_c = s_c.merge(s_tot, on="node")
    part = 1 - s_c.assign(sq=lambda d: (d["s_c"] / d["s_tot"]) ** 2).groupby("node")["sq"].sum()

    own = pd.DataFrame({"node": m.index, "comm": m.values})
    intra = s_c.merge(own, on=["node", "comm"], how="inner")[["node", "comm", "s_c"]]
    intra = own.merge(intra, on=["node", "comm"], how="left").fillna({"s_c": 0.0})
    stats = intra.groupby("comm")["s_c"].agg(["mean", "std"])
    intra = intra.merge(stats, on="comm")
    intra["within_module_z"] = (intra["s_c"] - intra["mean"]) / intra["std"].replace(0, np.nan)

    df = intra[["node", "within_module_z"]].merge(
        part.rename("participation_coef"), on="node", how="left"
    )

    z = df["within_module_z"].fillna(0)
    P = df["participation_coef"].fillna(0)
    role = np.select(
        [
            (z <= 2.5) & (P <= 0.05),
            (z <= 2.5) & (P <= 0.62),
            (z <= 2.5) & (P <= 0.80),
            (z <= 2.5),
            (z > 2.5) & (P <= 0.30),
            (z > 2.5) & (P <= 0.75),
        ],
        ["R1", "R2", "R3", "R4", "R5", "R6"],
        default="R7",
    )
    df["ga_role"] = role
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6.6 Directed weighted assortativity (4 combos)
# --------------------------------------------------------------------------- #
@safe("assortativity")
def directed_assortativity4(edges) -> dict:
    """
    Amount-weighted Pearson correlation across edges between source and dest
    strengths, for the four (out|in)x(out|in) combinations.
    Returns dict: assort_out_in, assort_out_out, assort_in_in, assort_in_out.
    """
    e = to_host(edges)[["source", "dest", "amount"]].copy()
    s_out = e.groupby("source")["amount"].sum()
    s_in = e.groupby("dest")["amount"].sum()

    def col(series, key):
        return np.log1p(e[key].map(series).fillna(0).to_numpy())

    w = e["amount"].to_numpy(dtype="float64")

    def wcorr(x, y):
        wm = w / w.sum()
        mx, my = (wm * x).sum(), (wm * y).sum()
        cov = (wm * (x - mx) * (y - my)).sum()
        sx = math.sqrt((wm * (x - mx) ** 2).sum())
        sy = math.sqrt((wm * (y - my) ** 2).sum())
        return float(cov / (sx * sy)) if sx > 0 and sy > 0 else float("nan")

    return {
        "assort_out_in": wcorr(col(s_out, "source"), col(s_in, "dest")),
        "assort_out_out": wcorr(col(s_out, "source"), col(s_out, "dest")),
        "assort_in_in": wcorr(col(s_in, "source"), col(s_in, "dest")),
        "assort_in_out": wcorr(col(s_in, "source"), col(s_out, "dest")),
    }


# --------------------------------------------------------------------------- #
# 6.7 Tail statistics
# --------------------------------------------------------------------------- #
@safe("tail_stats")
def tail_stats(values, hill_k_frac=0.01) -> dict:
    """Gini, Hill tail index (top hill_k_frac), top-share stats for a 1-D array."""
    x = np.sort(np.asarray(to_host_arr(values), dtype="float64"))
    x = x[x > 0]
    n = len(x)
    if n < 10:
        return {"gini": np.nan, "hill_alpha": np.nan, "top1pct_share": np.nan,
                "top01pct_share": np.nan}
    cum = np.cumsum(x)
    gini = float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
    k = max(int(n * hill_k_frac), 10)
    tail = x[-k:]
    hill = float(1.0 / np.mean(np.log(tail / tail[0]))) if tail[0] > 0 else np.nan
    return {
        "gini": gini,
        "hill_alpha": hill,
        "top1pct_share": float(x[-max(n // 100, 1):].sum() / cum[-1]),
        "top01pct_share": float(x[-max(n // 1000, 1):].sum() / cum[-1]),
    }


# --------------------------------------------------------------------------- #
# 6.8 Weighted rich-club
# --------------------------------------------------------------------------- #
@safe("rich_club")
def weighted_rich_club(edges, fractions=(0.001, 0.005, 0.01, 0.05)) -> pd.DataFrame:
    """
    phi_w(f) = amount among the top-f strength nodes / total amount they touch.
    Reported raw across versions (null-model normalization optional, offline).
    """
    e = to_host(edges)[["source", "dest", "amount"]]
    s = (
        e.groupby("source")["amount"].sum()
        .add(e.groupby("dest")["amount"].sum(), fill_value=0)
        .sort_values(ascending=False)
    )
    rows = []
    for f in fractions:
        k = max(int(len(s) * f), 2)
        club = set(s.index[:k])
        src_in = e["source"].isin(club)
        dst_in = e["dest"].isin(club)
        internal = float(e.loc[src_in & dst_in, "amount"].sum())
        touched = float(e.loc[src_in | dst_in, "amount"].sum())
        rows.append(
            {"fraction": f, "club_size": k,
             "phi_w": internal / touched if touched > 0 else np.nan,
             "internal_amount": internal}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6.9 Partition comparison: NMI (arithmetic), ARI, Jaccard lifecycle matcher
# --------------------------------------------------------------------------- #
@safe("partition_compare")
def partition_compare(mA: pd.DataFrame, mB: pd.DataFrame) -> dict:
    """
    mA, mB: [node, community_id] for consecutive months.
    NMI uses arithmetic normalization (sklearn-consistent). Computed on the
    node intersection.
    """
    A = to_host(mA).set_index("node")["community_id"]
    B = to_host(mB).set_index("node")["community_id"]
    common = A.index.intersection(B.index)
    if len(common) == 0:
        return {"nmi": np.nan, "ari": np.nan, "n_common": 0}
    a = A.loc[common].astype("category").cat.codes.to_numpy()
    b = B.loc[common].astype("category").cat.codes.to_numpy()
    n = len(common)

    ct = pd.crosstab(a, b).to_numpy(dtype="float64")
    pij = ct / n
    pi = pij.sum(axis=1, keepdims=True)
    pj = pij.sum(axis=0, keepdims=True)
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / (pi @ pj)[nz])).sum())
    hi = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hj = float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())
    nmi = mi / ((hi + hj) / 2) if (hi + hj) > 0 else np.nan

    # ARI
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(ct).sum()
    ai = comb(ct.sum(axis=1)).sum()
    bj = comb(ct.sum(axis=0)).sum()
    exp = ai * bj / comb(n)
    mx = (ai + bj) / 2.0
    ari = float((sum_ij - exp) / (mx - exp)) if mx != exp else np.nan
    return {"nmi": float(nmi), "ari": ari, "n_common": int(n)}


@safe("partition_compare")
def match_partitions(mA, mB, match_thr=0.3, event_thr=0.15) -> pd.DataFrame:
    """
    Jaccard lifecycle matcher month A -> month B.
    Returns per-A-community: best B match, jaccard, event in
    {continued, split, merged, died}; B communities with no A parent = born
    (appended rows with event='born').
    """
    A = to_host(mA).rename(columns={"community_id": "cA"})
    B = to_host(mB).rename(columns={"community_id": "cB"})
    j = A.merge(B, on="node", how="outer")
    sizesA = j.groupby("cA")["node"].nunique()
    sizesB = j.groupby("cB")["node"].nunique()
    inter = j.dropna().groupby(["cA", "cB"])["node"].nunique().rename("inter").reset_index()
    inter["jaccard"] = inter.apply(
        lambda r: r["inter"] / (sizesA[r["cA"]] + sizesB[r["cB"]] - r["inter"]), axis=1
    )
    rows = []
    matched_B = set()
    for cA, grp in inter.groupby("cA"):
        grp = grp.sort_values("jaccard", ascending=False)
        best = grp.iloc[0]
        strong = grp[grp["jaccard"] >= event_thr]
        if best["jaccard"] >= match_thr:
            event = "split" if len(strong) > 1 else "continued"
            rows.append((cA, best["cB"], best["jaccard"], event))
            matched_B.add(best["cB"])
        elif len(strong):
            rows.append((cA, best["cB"], best["jaccard"], "merged"))
            matched_B.add(best["cB"])
        else:
            rows.append((cA, None, best["jaccard"], "died"))
    out = pd.DataFrame(rows, columns=["community_A", "community_B", "jaccard", "event"])
    born = [c for c in sizesB.index if c not in matched_B]
    if born:
        out = pd.concat(
            [out, pd.DataFrame({"community_A": None, "community_B": born,
                                "jaccard": np.nan, "event": "born"})],
            ignore_index=True,
        )
    return out


# --------------------------------------------------------------------------- #
# 6.10 Edge turnover
# --------------------------------------------------------------------------- #
@safe("edge_turnover")
def edge_turnover(edges_prev, edges_curr):
    """
    Graph-level and per-node Jaccard turnover of the distinct (source, dest)
    edge sets between consecutive snapshots.
    Returns (graph_jaccard_distance, node_df[node, edge_turnover_node]).
    """
    a = to_host(edges_prev)[["source", "dest"]].drop_duplicates()
    b = to_host(edges_curr)[["source", "dest"]].drop_duplicates()
    a["k"] = 1
    b["k"] = 2
    m = a.merge(b, on=["source", "dest"], how="outer", suffixes=("_a", "_b"))
    inter = m["k_a"].notna() & m["k_b"].notna()
    g_dist = 1 - inter.sum() / len(m)

    m["inter"] = inter.astype(int)
    per = pd.concat(
        [
            m.groupby("source").agg(total=("inter", "size"), inter=("inter", "sum")),
            m.groupby("dest").agg(total=("inter", "size"), inter=("inter", "sum")),
        ]
    ).groupby(level=0).sum()
    node_df = pd.DataFrame(
        {"node": per.index, "edge_turnover_node": (1 - per["inter"] / per["total"]).to_numpy()}
    ).reset_index(drop=True)
    return float(g_dist), node_df


# --------------------------------------------------------------------------- #
# 6.11 / 6.12 Threshold ladder + five-version builder
# --------------------------------------------------------------------------- #
@safe()
def threshold_ladder(edges, p_broad=99.9, p_strict=99.99) -> dict:
    """Data-driven thresholds from degree (raw edge count) and strength dists."""
    e = to_host(edges)
    deg = pd.concat([e.groupby("source").size(), e.groupby("dest").size()]) \
            .groupby(level=0).sum()
    strength = pd.concat([e.groupby("source")["amount"].sum(),
                          e.groupby("dest")["amount"].sum()]).groupby(level=0).sum()
    return {
        "T_deg_broad": float(np.percentile(deg, p_broad)),
        "T_deg_strict": float(np.percentile(deg, p_strict)),
        "T_str_broad": float(np.percentile(strength, p_broad)),
        "T_str_strict": float(np.percentile(strength, p_strict)),
        "_degree": deg,
        "_strength": strength,
    }


@safe()
def build_versions(edges, thresholds: dict) -> dict:
    """
    Returns {'V0': edges, 'V1': ..., 'V4': ...} plus '_retention' report frame.
    V1 deg>strict; V2 str>strict; V3 deg>strict OR str>strict;
    V4 deg>broad OR str>broad.
    """
    e = to_host(edges)
    deg, s = thresholds["_degree"], thresholds["_strength"]
    masks = {
        "V1": set(deg[deg > thresholds["T_deg_strict"]].index),
        "V2": set(s[s > thresholds["T_str_strict"]].index),
    }
    masks["V3"] = masks["V1"] | masks["V2"]
    masks["V4"] = set(deg[deg > thresholds["T_deg_broad"]].index) | \
                  set(s[s > thresholds["T_str_broad"]].index)
    out = {"V0": e}
    rows = [("V0", len(e), 0, float(e["amount"].sum()))]
    for v in ("V1", "V2", "V3", "V4"):
        drop = masks[v]
        ev = e[~e["source"].isin(drop) & ~e["dest"].isin(drop)]
        out[v] = ev
        rows.append((v, len(ev), len(drop), float(ev["amount"].sum())))
    out["_retention"] = pd.DataFrame(
        rows, columns=["version", "edges_kept", "nodes_removed", "amount_kept"]
    )
    return out


# --------------------------------------------------------------------------- #
# 6.13 Node concentration block
# --------------------------------------------------------------------------- #
@safe("concentration")
def node_concentration(edges) -> pd.DataFrame:
    """HHI in/out, top-1 shares, distinct degrees, strengths, flow indices."""
    e = to_host(edges)[["source", "dest", "amount"]]

    def side(key, other):
        g = e.groupby([key, other])["amount"].sum().reset_index()
        tot = g.groupby(key)["amount"].sum().rename("s")
        g = g.merge(tot, on=key)
        g["share2"] = (g["amount"] / g["s"]) ** 2
        agg = g.groupby(key).agg(
            hhi=("share2", "sum"),
            top1=("amount", "max"),
            s=("s", "first"),
            d=(other, "nunique"),
        )
        agg["top1_share"] = agg["top1"] / agg["s"]
        return agg

    o = side("source", "dest").rename(
        columns={"hhi": "hhi_out", "top1_share": "top1_share_out", "s": "s_out", "d": "d_out"}
    )[["hhi_out", "top1_share_out", "s_out", "d_out"]]
    i = side("dest", "source").rename(
        columns={"hhi": "hhi_in", "top1_share": "top1_share_in", "s": "s_in", "d": "d_in"}
    )[["hhi_in", "top1_share_in", "s_in", "d_in"]]
    df = o.join(i, how="outer").fillna(
        {"s_out": 0, "s_in": 0, "d_out": 0, "d_in": 0}
    ).rename_axis("node").reset_index()
    df["net_flow"] = df["s_in"] - df["s_out"]
    tot = df["s_in"] + df["s_out"]
    df["flow_ratio"] = np.where(tot > 0, df["s_in"] / tot, np.nan)
    df["throughflow"] = df[["s_in", "s_out"]].min(axis=1)
    mx = df[["s_in", "s_out"]].max(axis=1)
    df["pass_through_index"] = np.where(mx > 0, df["throughflow"] / mx, np.nan)
    dd = df["d_in"] + df["d_out"]
    df["in_out_asym"] = np.where(dd > 0, (df["d_in"] - df["d_out"]) / dd, np.nan)
    return df


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
def _smoke():
    rng = np.random.default_rng(7)
    n_nodes, n_edges = 400, 3000
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    keep = src != dst
    e = pd.DataFrame(
        {
            "source": [f"N{i}" for i in src[keep]],
            "dest": [f"N{i}" for i in dst[keep]],
            "amount": np.round(rng.lognormal(10, 1.5, keep.sum()), 2),
            "volume": rng.integers(1, 20, keep.sum()),
        }
    )
    e = e.groupby(["source", "dest"], as_index=False).agg(
        amount=("amount", "sum"), volume=("volume", "sum")
    )
    membership = pd.DataFrame(
        {"node": [f"N{i}" for i in range(n_nodes)],
         "community_id": [f"C{i % 8}" for i in range(n_nodes)]}
    )

    print("== weighted_hits ==");            print(weighted_hits(e).head(3))
    node_t, F0 = trophic_levels(e);          print(f"== trophic F0 = {F0:.3f} =="); print(node_t.head(3))
    nrec, gw, gu = weighted_reciprocity(e);  print(f"== reciprocity global w={gw:.4f} u={gu:.4f} ==")
    comm, nemb = community_mixing(e, membership); print("== community_mixing =="); print(comm.head(3))
    print("== participation_z_roles ==");    print(participation_z_roles(e, membership)["ga_role"].value_counts())
    print("== assortativity ==");            print(directed_assortativity4(e))
    print("== tail_stats ==");               print(tail_stats(e["amount"].to_numpy()))
    print("== rich_club ==");                print(weighted_rich_club(e))
    m2 = membership.copy(); m2.loc[m2.index[:40], "community_id"] = "C99"
    print("== partition_compare ==");        print(partition_compare(membership, m2))
    print("== match_partitions ==");         print(match_partitions(membership, m2)["event"].value_counts())
    e2 = e.sample(frac=0.8, random_state=1)
    gj, nt = edge_turnover(e, e2);           print(f"== edge_turnover graph = {gj:.3f} ==")
    thr = threshold_ladder(e);               vers = build_versions(e, thr)
    print("== versions ==");                 print(vers["_retention"])
    print("== node_concentration ==");       print(node_concentration(e).head(3))
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    _smoke()
