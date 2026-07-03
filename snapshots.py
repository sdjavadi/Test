"""
================================================================================
PKG Monthly Snapshot Graph Metrics Pipeline (GPU / RAPIDS)
================================================================================
Computes an extensive set of node-level, graph-level, month-over-month, and
longitudinal network metrics from monthly customer-to-customer payment
snapshots, using cuGraph / cuDF / CuPy end-to-end on GPU.

INPUT
-----
    ../data/cust_YYYY-MM.csv        columns: source,target,amount,volume
    (only `amount` is used as edge weight; `volume` is ignored)

OUTPUT (created under ../result/)
---------------------------------
    graph_metrics_monthly.csv               1 row / month, all graph-level scalars
    mom_summary_monthly.csv                 1 row / month-pair, MoM graph scalars
    nodes/node_metrics_YYYY-MM.parquet      per-node metrics for each month
    mom/mom_node_YYYY-MM.parquet            per-node MoM churn & role-drift
    edges/edge_metrics_YYYY-MM.parquet      per-edge metrics       [flag-gated]
    temporal/node_temporal_summary.parquet  burstiness / autocorr across months
    temporal/temporal_rings_YYYY-MM.parquet 3-month time-ordered rings [gated]
    hubs/hubs_YYYY-MM.csv                   excluded hub registry  [if enabled]
    run_log.txt                             per-step timing + failures

DESIGN NOTES
------------
* Weight = `amount` only. Parallel edges within a month are summed on load,
  so degree == distinct-neighbor count within each direction.
* Hub exclusion (HUB_DEGREE_THRESHOLD) removes mega-hubs BEFORE structural
  metrics, per the PKG hub-pollution mitigation. Hub-inclusive variants are
  kept where hubs ARE the point (rich-club, bow-tie) via RICHCLUB_ON_FULL.
* cuGraph native algos are wrapped in `safe()` so a RAPIDS version API
  mismatch degrades that one metric instead of killing the run. Tested
  against RAPIDS 23.x-25.x conventions; adjust the marked call sites if your
  cluster runs something older.
* Custom spectral metrics (weighted HITS, eigenvector, Katz, trophic levels,
  current-flow betweenness) are implemented with cupyx.scipy.sparse and stay
  on GPU.
================================================================================
"""

import os
import re
import glob
import time
import math
import traceback

import cudf
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpsl
import cugraph

# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_DIR = "../data"
OUTPUT_DIR = "../result"
FILE_GLOB = "cust_*.csv"

# --- hub exclusion -----------------------------------------------------------
HUB_DEGREE_THRESHOLD = 10_000   # distinct undirected neighbors; None = disabled
RICHCLUB_ON_FULL = True         # rich-club & bow-tie also on hub-inclusive graph

# --- cheap/medium metrics (on by default) -------------------------------------
RUN_BETWEENNESS = True
BETWEENNESS_K = 256             # sampled sources (unweighted shortest paths)
RUN_LOUVAIN = True
RUN_TRIANGLES = True            # undirected triangle count -> clustering coeff
RUN_EDGE_EMBEDDEDNESS = False   # per-edge Jaccard; needs WRITE_EDGE_METRICS
WRITE_EDGE_METRICS = False      # per-edge parquet output (large!)

# --- expensive / flag-gated metrics -------------------------------------------
RUN_CYCLIC_TRIANGLES = False    # A .* (A@A)^T  -- run only post-hub-exclusion
RUN_BURT_CONSTRAINT = False     # P + P@P      -- memory-hungry at 10M nodes
RUN_TROPHIC = False             # CG solve on (diag(u) - W - W^T)
RUN_CURRENT_FLOW_BTW = False    # landmark-pair CG approximation
CURRENT_FLOW_PAIRS = 64
RUN_TEMPORAL_RINGS = False      # 3-month time-ordered A->B->C->A motifs
TEMPORAL_RING_MIN_AMT_Q = 0.90  # only edges above this amount quantile

# --- iteration / solver parameters --------------------------------------------
POWER_ITERS = 100
POWER_TOL = 1e-8
KATZ_ALPHA_FRAC = 0.85          # alpha = frac / lambda_max
CG_TOL = 1e-6
CG_MAXITER = 500
RICH_CLUB_PCTS = [0.001, 0.01, 0.05]

INF_DIST = 2**31 - 2            # BFS unreachable sentinel guard

# ==============================================================================
# LOGGING / UTILITIES
# ==============================================================================
_LOG_PATH = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_PATH:
        with open(_LOG_PATH, "a") as f:
            f.write(line + "\n")


def safe(label, fn, default=None):
    """Run fn(); on any failure log traceback and return default."""
    t0 = time.time()
    try:
        out = fn()
        log(f"  ok   {label} ({time.time() - t0:.1f}s)")
        return out
    except Exception:
        log(f"  FAIL {label}\n{traceback.format_exc()}")
        return default


def ensure_dirs():
    for sub in ["", "nodes", "mom", "edges", "temporal", "hubs"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def month_key(path):
    m = re.search(r"cust_(\d{4}-\d{2})\.csv$", os.path.basename(path))
    return m.group(1) if m else None


# ==============================================================================
# LOADING & GRAPH CONSTRUCTION
# ==============================================================================
def load_month(path):
    """Read snapshot, aggregate parallel edges, drop self-loops & volume."""
    df = cudf.read_csv(path, usecols=["source", "target", "amount"])
    df = df.dropna()
    df["amount"] = df["amount"].astype("float64")
    df = df[df["amount"] > 0]
    df = df[df["source"] != df["target"]]
    df = (
        df.groupby(["source", "target"], as_index=False)
        .agg({"amount": "sum"})
    )
    return df


def renumber(edges):
    """Contiguous int32 vertex ids; returns (edges[src,dst,amount], nodes_df)."""
    nodes = cudf.concat([edges["source"], edges["target"]]).unique()
    nodes_df = cudf.DataFrame({"node": nodes})
    nodes_df["vid"] = cp.arange(len(nodes_df), dtype="int32")
    e = edges.merge(nodes_df, left_on="source", right_on="node", how="left")
    e = e.rename(columns={"vid": "src"}).drop(columns=["node"])
    e = e.merge(nodes_df, left_on="target", right_on="node", how="left")
    e = e.rename(columns={"vid": "dst"}).drop(columns=["node"])
    return e[["src", "dst", "amount", "source", "target"]], nodes_df


def build_cugraph(e, directed=True, weighted=True):
    G = cugraph.Graph(directed=directed)
    cols = {"source": "src", "destination": "dst"}
    if weighted:
        G.from_cudf_edgelist(e, edge_attr="amount", renumber=True, **cols)
    else:
        G.from_cudf_edgelist(e, renumber=True, **cols)
    return G


def to_csr(e, n, symmetric=False, binary=False):
    """CuPy CSR of the (weighted or binary) adjacency; rows = src."""
    src = e["src"].values.astype(cp.int32)
    dst = e["dst"].values.astype(cp.int32)
    w = cp.ones_like(src, dtype=cp.float64) if binary \
        else e["amount"].values.astype(cp.float64)
    W = cpsp.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()
    if symmetric:
        W = W + W.T
    return W


def symmetrized_edges(e):
    """Undirected weighted edge list: w(u,v) = w(u->v) + w(v->u), u < v."""
    a = e[["src", "dst", "amount"]]
    b = e[["dst", "src", "amount"]].rename(columns={"dst": "src", "src": "dst"})
    u = cudf.concat([a, b])
    lo = u[["src", "dst"]].min(axis=1)
    hi = u[["src", "dst"]].max(axis=1)
    u = cudf.DataFrame({"src": lo, "dst": hi, "amount": u["amount"]})
    u = u.groupby(["src", "dst"], as_index=False).agg({"amount": "sum"})
    return u


# ==============================================================================
# NODE-LEVEL: cuDF metrics (strengths, entropy, disparity, ...)
# ==============================================================================
def cudf_node_metrics(e, n):
    out_s = e.groupby("src").agg({"amount": "sum"}).rename(
        columns={"amount": "out_strength"})
    in_s = e.groupby("dst").agg({"amount": "sum"}).rename(
        columns={"amount": "in_strength"})
    out_d = e.groupby("src").size().rename("out_degree")
    in_d = e.groupby("dst").size().rename("in_degree")

    nm = cudf.DataFrame({"vid": cp.arange(n, dtype="int32")})
    nm = nm.merge(out_s, left_on="vid", right_index=True, how="left")
    nm = nm.merge(in_s, left_on="vid", right_index=True, how="left")
    nm = nm.merge(out_d.to_frame(), left_on="vid", right_index=True, how="left")
    nm = nm.merge(in_d.to_frame(), left_on="vid", right_index=True, how="left")
    for c in ["out_strength", "in_strength", "out_degree", "in_degree"]:
        nm[c] = nm[c].fillna(0)

    nm["net_flow"] = nm["in_strength"] - nm["out_strength"]
    nm["throughput"] = nm["in_strength"] + nm["out_strength"]
    tp = nm["throughput"]
    nm["passthrough_score"] = 1.0 - (nm["net_flow"].abs() / tp.where(tp > 0, 1))
    nm["total_degree"] = nm["out_degree"] + nm["in_degree"]

    # distinct undirected neighbors
    und = symmetrized_edges(e)
    pairs = cudf.concat([
        und[["src"]].rename(columns={"src": "vid"}),
        und[["dst"]].rename(columns={"dst": "vid"}),
    ])
    nbrs = pairs.groupby("vid").size().rename("distinct_neighbors")
    nm = nm.merge(nbrs.to_frame(), left_on="vid", right_index=True, how="left")
    nm["distinct_neighbors"] = nm["distinct_neighbors"].fillna(0)

    # per-direction share, disparity (Y-index) and Shannon entropy (nats)
    for grp, scol, tag in [("src", "out_strength", "out"),
                           ("dst", "in_strength", "in")]:
        tmp = e.merge(nm[["vid", scol]], left_on=grp, right_on="vid",
                      how="left")
        p = tmp["amount"] / tmp[scol]
        tmp["_y"] = p * p
        tmp["_h"] = -(p * p.log())
        agg = tmp.groupby(grp).agg({"_y": "sum", "_h": "sum"})
        agg.columns = [f"disparity_{tag}", f"entropy_{tag}"]
        nm = nm.merge(agg, left_on="vid", right_index=True, how="left")
        nm[f"disparity_{tag}"] = nm[f"disparity_{tag}"].fillna(0)
        nm[f"entropy_{tag}"] = nm[f"entropy_{tag}"].fillna(0)

    return nm, und


# ==============================================================================
# NODE-LEVEL: custom spectral metrics on CuPy sparse (all weighted by amount)
# ==============================================================================
def weighted_hits(W, iters=POWER_ITERS, tol=POWER_TOL):
    """Amount-weighted HITS. Returns (hub, authority) as CuPy arrays."""
    n = W.shape[0]
    h = cp.ones(n, dtype=cp.float64)
    a = cp.ones(n, dtype=cp.float64)
    for _ in range(iters):
        a_new = W.T @ h
        na = cp.linalg.norm(a_new)
        a_new = a_new / na if na > 0 else a_new
        h_new = W @ a_new
        nh = cp.linalg.norm(h_new)
        h_new = h_new / nh if nh > 0 else h_new
        if (cp.abs(h_new - h).max() < tol and
                cp.abs(a_new - a).max() < tol):
            h, a = h_new, a_new
            break
        h, a = h_new, a_new
    return h, a


def eigenvector_undirected(W_sym, iters=POWER_ITERS, tol=POWER_TOL):
    n = W_sym.shape[0]
    x = cp.ones(n, dtype=cp.float64) / math.sqrt(n)
    for _ in range(iters):
        x_new = W_sym @ x
        nrm = cp.linalg.norm(x_new)
        if nrm == 0:
            return x
        x_new /= nrm
        if cp.abs(x_new - x).max() < tol:
            return x_new
        x = x_new
    return x


def spectral_radius(W, iters=50):
    n = W.shape[0]
    x = cp.random.default_rng(7).random(n)
    x /= cp.linalg.norm(x)
    lam = 1.0
    for _ in range(iters):
        y = W.T @ x
        nrm = cp.linalg.norm(y)
        if nrm == 0:
            return 1.0
        lam = float(nrm)
        x = y / nrm
    return max(lam, 1e-12)


def katz_centrality(W, alpha_frac=KATZ_ALPHA_FRAC, iters=POWER_ITERS * 3,
                    tol=POWER_TOL):
    """x = alpha * W^T x + 1  (importance from weighted incoming paths)."""
    lam = spectral_radius(W)
    alpha = alpha_frac / lam
    n = W.shape[0]
    x = cp.ones(n, dtype=cp.float64)
    for _ in range(iters):
        x_new = alpha * (W.T @ x) + 1.0
        if cp.abs(x_new - x).max() < tol * cp.abs(x_new).max():
            return x_new
        x = x_new
    return x


def trophic_levels(e, n):
    """MacKay et al. trophic levels h and incoherence F0 (weighted)."""
    W = to_csr(e, n)
    w_in = cp.asarray(W.sum(axis=0)).ravel()
    w_out = cp.asarray(W.sum(axis=1)).ravel()
    u = w_in + w_out
    v = w_in - w_out
    Lam = cpsp.diags(u) - W - W.T
    jitter = 1e-8 * float(u.mean() if u.size else 1.0)
    Lam = Lam + cpsp.identity(n, format="csr") * jitter
    h, _ = cpsl.cg(Lam.tocsr(), v, tol=CG_TOL, maxiter=CG_MAXITER)
    h = h - h.min()
    # incoherence F0 = sqrt( sum w_ij (h_j - h_i - 1)^2 / sum w_ij )
    src = e["src"].values
    dst = e["dst"].values
    w = e["amount"].values.astype(cp.float64)
    diff = h[dst] - h[src] - 1.0
    f0 = float(cp.sqrt((w * diff * diff).sum() / w.sum()))
    return h, f0


def cyclic_triangles(e, n):
    """#directed 3-cycles through each node: rowsum(A .* (A@A)^T), binary A."""
    A = to_csr(e, n, binary=True)
    AA = A @ A
    D = A.multiply(AA.T)
    return cp.asarray(D.sum(axis=1)).ravel()


def burt_constraint(und, n):
    """Burt's constraint on the symmetrized weighted graph."""
    a = und[["src", "dst", "amount"]]
    b = und[["dst", "src", "amount"]].rename(
        columns={"dst": "src", "src": "dst"})
    full = cudf.concat([a, b])
    W = cpsp.coo_matrix(
        (full["amount"].values.astype(cp.float64),
         (full["src"].values.astype(cp.int32),
          full["dst"].values.astype(cp.int32))),
        shape=(n, n)).tocsr()
    rs = cp.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    P = cpsp.diags(1.0 / rs) @ W          # row-stochastic proportions
    M = P + P @ P                          # direct + one-step indirect
    B = W.copy()
    B.data = cp.ones_like(B.data)          # neighbor mask
    Mm = M.multiply(B)                     # restrict to direct neighbors
    Mm.data = Mm.data * Mm.data            # (p_ij + sum_q p_iq p_qj)^2
    return cp.asarray(Mm.sum(axis=1)).ravel()


def current_flow_betweenness(und, n, k_pairs=CURRENT_FLOW_PAIRS):
    """Landmark-pair approximation of random-walk (current-flow) betweenness."""
    W = cpsp.coo_matrix(
        (cp.concatenate([und["amount"].values.astype(cp.float64)] * 2),
         (cp.concatenate([und["src"].values.astype(cp.int32),
                          und["dst"].values.astype(cp.int32)]),
          cp.concatenate([und["dst"].values.astype(cp.int32),
                          und["src"].values.astype(cp.int32)]))),
        shape=(n, n)).tocsr()
    d = cp.asarray(W.sum(axis=1)).ravel()
    L = cpsp.diags(d) - W
    L = L + cpsp.identity(n, format="csr") * (1e-8 * float(d.mean() or 1.0))

    rng = cp.random.default_rng(11)
    prob = d / d.sum()
    src_e = und["src"].values
    dst_e = und["dst"].values
    w_e = und["amount"].values.astype(cp.float64)
    thr = cp.zeros(n, dtype=cp.float64)
    done = 0
    for _ in range(k_pairs):
        s, t = [int(x) for x in rng.choice(n, size=2, p=prob)]
        if s == t:
            continue
        b = cp.zeros(n, dtype=cp.float64)
        b[s], b[t] = 1.0, -1.0
        p, info = cpsl.cg(L, b, tol=CG_TOL, maxiter=CG_MAXITER)
        if info != 0:
            continue
        cur = cp.abs(w_e * (p[src_e] - p[dst_e]))
        contrib = cp.zeros(n, dtype=cp.float64)
        import cupyx
        cupyx.scatter_add(contrib, src_e, cur)
        cupyx.scatter_add(contrib, dst_e, cur)
        contrib *= 0.5
        contrib[s] = 0.0
        contrib[t] = 0.0
        thr += contrib
        done += 1
    return thr / max(done, 1)


# ==============================================================================
# GRAPH-LEVEL SCALARS
# ==============================================================================
def gini(x):
    x = cp.sort(x[x >= 0].astype(cp.float64))
    m = x.size
    if m == 0 or float(x.sum()) == 0:
        return 0.0
    idx = cp.arange(1, m + 1, dtype=cp.float64)
    return float((2.0 * (idx * x).sum()) / (m * x.sum()) - (m + 1.0) / m)


def hhi(x):
    tot = float(x.sum())
    if tot == 0:
        return 0.0
    p = x / tot
    return float((p * p).sum())


def reciprocity_metrics(e):
    rev = e[["src", "dst", "amount"]].rename(
        columns={"src": "dst", "dst": "src", "amount": "amount_rev"})
    m = e[["src", "dst", "amount"]].merge(rev, on=["src", "dst"], how="left")
    has_rev = m["amount_rev"].notna()
    edge_recip = float(has_rev.sum()) / max(len(m), 1)
    mn = m["amount"].where(
        m["amount"] < m["amount_rev"].fillna(0), m["amount_rev"].fillna(0))
    w_recip = float(mn.sum()) / max(float(m["amount"].sum()), 1e-12)
    return edge_recip, w_recip


def assortativity(e, nm):
    m = e.merge(nm[["vid", "out_degree"]], left_on="src", right_on="vid",
                how="left")
    m = m.merge(nm[["vid", "in_degree"]], left_on="dst", right_on="vid",
                how="left", suffixes=("", "_t"))
    x = m["out_degree"].values.astype(cp.float64)
    y = m["in_degree"].values.astype(cp.float64)
    if x.size < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    den = float(cp.sqrt((xm * xm).sum() * (ym * ym).sum()))
    return float((xm * ym).sum() / den) if den > 0 else float("nan")


def strength_degree_beta(nm):
    d = nm["total_degree"].values.astype(cp.float64)
    s = nm["throughput"].values.astype(cp.float64)
    mask = (d > 0) & (s > 0)
    if int(mask.sum()) < 10:
        return float("nan")
    lx, ly = cp.log(d[mask]), cp.log(s[mask])
    lxm = lx - lx.mean()
    den = float((lxm * lxm).sum())
    return float((lxm * (ly - ly.mean())).sum() / den) if den > 0 else float("nan")


def rich_club(e, nm, pcts=RICH_CLUB_PCTS):
    """Weighted rich-club at top-p% strength; normalized by strength share^2."""
    out = {}
    total_w = float(e["amount"].sum())
    tot_s = float(nm["throughput"].sum())
    for p in pcts:
        k = max(int(len(nm) * p), 1)
        rich = nm.nlargest(k, "throughput")[["vid"]]
        rich["is_rich"] = True
        m = e.merge(rich, left_on="src", right_on="vid", how="left")
        m = m.merge(rich, left_on="dst", right_on="vid", how="left",
                    suffixes=("_s", "_t"))
        both = m["is_rich_s"].fillna(False) & m["is_rich_t"].fillna(False)
        phi = float(m["amount"][both].sum()) / max(total_w, 1e-12)
        share = float(nm.nlargest(k, "throughput")["throughput"].sum()) \
            / max(tot_s, 1e-12) / 2.0  # /2: throughput double-counts edge ends
        expected = share * share
        tag = str(p).replace(".", "p")
        out[f"rich_club_raw_{tag}"] = phi
        out[f"rich_club_norm_{tag}"] = phi / expected if expected > 0 else float("nan")
    return out


def renyi2_vnge(und):
    """Rényi-2 von Neumann graph entropy H2 = -log(tr(Lhat^2))."""
    a = und["amount"].values.astype(cp.float64)
    pairs = cudf.concat([
        und[["src", "amount"]].rename(columns={"src": "vid"}),
        und[["dst", "amount"]].rename(columns={"dst": "vid"}),
    ])
    deg = pairs.groupby("vid").agg({"amount": "sum"})["amount"].values \
        .astype(cp.float64)
    trL = float(deg.sum())
    trL2 = float((deg * deg).sum() + 2.0 * (a * a).sum())
    if trL <= 0:
        return 0.0
    return float(-cp.log(cp.asarray(trL2 / (trL * trL))))


def flow_hierarchy_and_scc(e, G_dir):
    """SCC labels; flow hierarchy = frac of edges not inside any SCC (acyclic)."""
    scc = cugraph.strongly_connected_components(G_dir)
    scc = scc.rename(columns={"labels": "scc"})
    m = e.merge(scc, left_on="src", right_on="vertex", how="left")
    m = m.merge(scc, left_on="dst", right_on="vertex", how="left",
                suffixes=("_s", "_t"))
    acyclic = float((m["scc_s"] != m["scc_t"]).sum()) / max(len(m), 1)
    return scc, acyclic


def bowtie(e, scc, n):
    """CORE = largest SCC; IN reaches core; OUT reached from core."""
    sizes = scc.groupby("scc").size().rename("sz").reset_index()
    top = sizes.nlargest(1, "sz")
    core_label = int(top["scc"].iloc[0])
    core_sz = int(top["sz"].iloc[0])
    core = scc[scc["scc"] == core_label]["vertex"]

    virt = n  # virtual super-node id
    core_df = cudf.DataFrame({"dst": core.reset_index(drop=True)})
    core_df["src"] = virt

    def reach(edge_df):
        g = cugraph.Graph(directed=True)
        g.from_cudf_edgelist(edge_df, source="src", destination="dst",
                             renumber=True)
        b = cugraph.bfs(g, virt)
        b = b[(b["distance"] >= 0) & (b["distance"] < INF_DIST)]
        return b["vertex"]

    fwd = safe("bowtie fwd BFS", lambda: reach(
        cudf.concat([e[["src", "dst"]], core_df[["src", "dst"]]])))
    rev_edges = e[["dst", "src"]].rename(columns={"dst": "src", "src": "dst"})
    bwd = safe("bowtie bwd BFS", lambda: reach(
        cudf.concat([rev_edges, core_df[["src", "dst"]]])))

    # pure-CuPy label assembly: 0=CORE 1=IN 2=OUT 3=OTHER
    fwd_mask = cp.zeros(n, dtype=cp.bool_)
    bwd_mask = cp.zeros(n, dtype=cp.bool_)
    if fwd is not None:
        fv = fwd.values
        fwd_mask[fv[fv < n]] = True
    if bwd is not None:
        bv = bwd.values
        bwd_mask[bv[bv < n]] = True
    comp_arr = cp.full(n, 3, dtype=cp.int8)
    comp_arr[fwd_mask] = 2                       # reachable FROM core -> OUT
    comp_arr[bwd_mask & ~fwd_mask] = 1           # reaches core only -> IN
    comp_arr[fwd_mask & bwd_mask] = 0            # both directions -> CORE
    comp_arr[core.values] = 0                    # core itself
    comp = cudf.Series(comp_arr)
    shares = {
        "bowtie_core_share": float((comp_arr == 0).sum()) / n,
        "bowtie_in_share": float((comp_arr == 1).sum()) / n,
        "bowtie_out_share": float((comp_arr == 2).sum()) / n,
        "bowtie_other_share": float((comp_arr == 3).sum()) / n,
        "bowtie_core_size": core_sz,
    }
    return comp, shares


# ==============================================================================
# MONTH-OVER-MONTH
# ==============================================================================
def neighbor_sets(e):
    """Undirected (node, neighbor) pair table, deduplicated."""
    a = e[["src", "dst"]]
    b = e[["dst", "src"]].rename(columns={"dst": "src", "src": "dst"})
    p = cudf.concat([a, b]).drop_duplicates()
    return p.rename(columns={"src": "vid", "dst": "nbr"})


def jaccard_churn(prev_pairs, cur_pairs):
    common = prev_pairs.merge(cur_pairs, on=["vid", "nbr"], how="inner")
    c = common.groupby("vid").size().rename("common").to_frame()
    pa = prev_pairs.groupby("vid").size().rename("prev_n").to_frame()
    cb = cur_pairs.groupby("vid").size().rename("cur_n").to_frame()
    j = pa.join(cb, how="outer").join(c, how="outer").fillna(0).reset_index()
    union = j["prev_n"] + j["cur_n"] - j["common"]
    j["neighbor_jaccard"] = (j["common"] / union.where(union > 0, 1)).fillna(0)
    j["new_neighbors"] = j["cur_n"] - j["common"]
    j["lost_neighbors"] = j["prev_n"] - j["common"]
    return j[["vid", "neighbor_jaccard", "new_neighbors", "lost_neighbors",
              "prev_n", "cur_n"]]


def edge_persistence_and_new(prev_e, cur_e):
    m = cur_e[["source", "target", "amount"]].merge(
        prev_e[["source", "target"]].assign(existed=True),
        on=["source", "target"], how="left")
    m["is_new_edge"] = m["existed"].isna()
    p = prev_e[["source", "target"]].merge(
        cur_e[["source", "target"]].assign(still=True),
        on=["source", "target"], how="left")
    persistence = float(p["still"].notna().sum()) / max(len(p), 1)
    new_amt = m["amount"][m["is_new_edge"]]
    old_amt = m["amount"][~m["is_new_edge"]]
    stats = {
        "edge_persistence": persistence,
        "new_edge_share": float(m["is_new_edge"].sum()) / max(len(m), 1),
        "new_edge_mean_amount": float(new_amt.mean()) if len(new_amt) else 0.0,
        "new_edge_median_amount": float(new_amt.quantile(0.5)) if len(new_amt) else 0.0,
        "persisting_edge_mean_amount": float(old_amt.mean()) if len(old_amt) else 0.0,
    }
    return m[["source", "target", "is_new_edge"]], stats


def adjusted_rand(prev_lab, cur_lab):
    """ARI between two partitions given as DataFrames [node, community]."""
    m = prev_lab.merge(cur_lab, on="node", how="inner",
                       suffixes=("_p", "_c"))
    if len(m) < 2:
        return float("nan")
    n = len(m)

    def comb2_sum(s):
        v = s.values.astype(cp.float64)
        return float((v * (v - 1) / 2.0).sum())

    nij = comb2_sum(m.groupby(["community_p", "community_c"]).size())
    ai = comb2_sum(m.groupby("community_p").size())
    bj = comb2_sum(m.groupby("community_c").size())
    nc2 = n * (n - 1) / 2.0
    expected = ai * bj / nc2
    mx = 0.5 * (ai + bj)
    den = mx - expected
    return float((nij - expected) / den) if den != 0 else 1.0


def pct_rank(df, col):
    return df[col].rank(pct=True)


# ==============================================================================
# PER-MONTH PIPELINE
# ==============================================================================
def process_month(path, month):
    log(f"=== {month} : {os.path.basename(path)} ===")
    raw = load_month(path)
    e_full, nodes_full = renumber(raw)
    n_full = len(nodes_full)
    log(f"  loaded: {n_full:,} nodes, {len(e_full):,} aggregated edges")

    g_row = {"month": month, "n_nodes_raw": n_full, "n_edges_raw": len(e_full),
             "total_amount_raw": float(e_full["amount"].sum())}

    # ----- hub-inclusive metrics that WANT hubs -------------------------------
    if RICHCLUB_ON_FULL:
        nm_tmp, _ = cudf_node_metrics(e_full, n_full)
        g_row.update({f"full_{k}": v for k, v in
                      (safe("rich-club (full)", lambda: rich_club(e_full, nm_tmp)) or {}).items()})
        del nm_tmp

    # ----- hub exclusion -------------------------------------------------------
    e, nodes_df, n = e_full, nodes_full, n_full
    if HUB_DEGREE_THRESHOLD:
        pairs = neighbor_sets(e_full)
        deg = pairs.groupby("vid").size().rename("d").to_frame().reset_index()
        hub_vids = deg[deg["d"] > HUB_DEGREE_THRESHOLD]["vid"]
        if len(hub_vids):
            hubs = nodes_full.merge(hub_vids.to_frame(name="vid"), on="vid")
            hubs = hubs.merge(deg, on="vid").rename(
                columns={"d": "distinct_neighbors"})
            hubs[["node", "distinct_neighbors"]].to_csv(
                os.path.join(OUTPUT_DIR, "hubs", f"hubs_{month}.csv"),
                index=False)
            hub_set = hub_vids.to_frame(name="vid")
            hub_set["is_hub"] = True
            e = e_full.merge(hub_set, left_on="src", right_on="vid",
                             how="left").rename(columns={"is_hub": "hs"})
            if "vid" in e.columns:
                e = e.drop(columns=["vid"])
            e = e.merge(hub_set, left_on="dst", right_on="vid",
                        how="left").rename(columns={"is_hub": "ht"})
            e = e[e["hs"].isna() & e["ht"].isna()]
            e = e[["source", "target", "amount"]]
            e, nodes_df = renumber(e)
            n = len(nodes_df)
            log(f"  hub exclusion: removed {len(hub_vids):,} hubs -> "
                f"{n:,} nodes, {len(e):,} edges")
        g_row["n_hubs_excluded"] = int(len(hub_vids))

    g_row.update({"n_nodes": n, "n_edges": len(e),
                  "total_amount": float(e["amount"].sum())})

    # ----- cuDF node metrics ---------------------------------------------------
    nm, und = cudf_node_metrics(e, n)

    # ----- graph objects -------------------------------------------------------
    G_dir = build_cugraph(e, directed=True)
    G_und = build_cugraph(
        und.rename(columns={"src": "src", "dst": "dst"}), directed=False)
    W = to_csr(e, n)
    W_sym = to_csr(e, n, symmetric=True)

    # ----- cuGraph native ------------------------------------------------------
    pr = safe("pagerank (weighted)", lambda: cugraph.pagerank(G_dir))
    if pr is not None:
        nm = nm.merge(pr.rename(columns={"pagerank": "pagerank_w"}),
                      left_on="vid", right_on="vertex", how="left") \
               .drop(columns=["vertex"])

    core = safe("core number", lambda: cugraph.core_number(G_und))
    if core is not None:
        nm = nm.merge(core.rename(columns={"core_number": "k_core"}),
                      left_on="vid", right_on="vertex", how="left") \
               .drop(columns=["vertex"])

    if RUN_BETWEENNESS:
        btw = safe("betweenness (sampled, unweighted paths)",
                   lambda: cugraph.betweenness_centrality(
                       G_dir, k=min(BETWEENNESS_K, n)))
        if btw is not None:
            nm = nm.merge(
                btw.rename(columns={"betweenness_centrality": "betweenness"}),
                left_on="vid", right_on="vertex", how="left") \
                .drop(columns=["vertex"])

    if RUN_TRIANGLES:
        def _tri():
            t = cugraph.triangle_count(G_und)
            return t.rename(columns={"counts": "triangles"})
        tri = safe("triangle count", _tri)
        if tri is not None:
            nm = nm.merge(tri, left_on="vid", right_on="vertex", how="left") \
                   .drop(columns=["vertex"])
            k = nm["distinct_neighbors"]
            denom = (k * (k - 1) / 2.0).where(k > 1, 1)
            nm["clustering"] = (nm["triangles"].fillna(0) / denom) \
                .where(k > 1, 0.0)

    louv_labels, modularity = None, float("nan")
    if RUN_LOUVAIN:
        def _louv():
            parts, mod = cugraph.louvain(G_und)
            return parts.rename(columns={"partition": "community"}), mod
        r = safe("louvain", _louv)
        if r is not None:
            louv_labels, modularity = r
            nm = nm.merge(louv_labels, left_on="vid", right_on="vertex",
                          how="left").drop(columns=["vertex"])
            g_row["modularity"] = float(modularity)
            g_row["n_communities"] = int(nm["community"].nunique())

    # ----- custom spectral (CuPy) ----------------------------------------------
    r = safe("weighted HITS", lambda: weighted_hits(W))
    if r is not None:
        nm["hub_score"], nm["authority_score"] = \
            cudf.Series(r[0]), cudf.Series(r[1])

    ev = safe("eigenvector (undirected)", lambda: eigenvector_undirected(W_sym))
    if ev is not None:
        nm["eigenvector"] = cudf.Series(ev)

    kz = safe("katz (weighted)", lambda: katz_centrality(W))
    if kz is not None:
        nm["katz"] = cudf.Series(kz)

    if RUN_CYCLIC_TRIANGLES:
        ct = safe("cyclic triangles", lambda: cyclic_triangles(e, n))
        if ct is not None:
            nm["cyclic_triangles"] = cudf.Series(ct)

    if RUN_BURT_CONSTRAINT:
        bc = safe("burt constraint", lambda: burt_constraint(und, n))
        if bc is not None:
            nm["burt_constraint"] = cudf.Series(bc)

    if RUN_TROPHIC:
        tr = safe("trophic levels", lambda: trophic_levels(e, n))
        if tr is not None:
            nm["trophic_level"] = cudf.Series(tr[0])
            g_row["trophic_incoherence"] = tr[1]

    if RUN_CURRENT_FLOW_BTW:
        cf = safe("current-flow betweenness (approx)",
                  lambda: current_flow_betweenness(und, n))
        if cf is not None:
            nm["current_flow_btw"] = cudf.Series(cf)

    # bridging centrality = betweenness / (clustering + eps)
    if "betweenness" in nm.columns and "clustering" in nm.columns:
        nm["bridging_centrality"] = nm["betweenness"].fillna(0) / \
            (nm["clustering"].fillna(0) + 1e-6)

    # ----- graph-level scalars ---------------------------------------------------
    s = nm["throughput"].values
    g_row["density"] = len(e) / max(n * (n - 1), 1)
    g_row["avg_out_degree"] = float(nm["out_degree"].mean())
    g_row["gini_strength"] = safe("gini", lambda: gini(s), 0.0)
    g_row["hhi_strength"] = safe("hhi", lambda: hhi(s), 0.0)

    er, wr = safe("reciprocity", lambda: reciprocity_metrics(e), (0.0, 0.0))
    g_row["reciprocity_edges"], g_row["reciprocity_weighted"] = er, wr
    g_row["assortativity_out_in"] = safe(
        "assortativity", lambda: assortativity(e, nm), float("nan"))
    g_row["strength_degree_beta"] = safe(
        "s(k) beta", lambda: strength_degree_beta(nm), float("nan"))
    g_row.update(safe("rich-club", lambda: rich_club(e, nm)) or {})
    g_row["vnge_renyi2"] = safe("VNGE (Renyi-2)", lambda: renyi2_vnge(und), 0.0)

    wcc = safe("WCC", lambda: cugraph.weakly_connected_components(G_dir))
    if wcc is not None:
        g_row["giant_wcc_share"] = float(
            wcc.groupby("labels").size().max()) / n

    r = safe("SCC + flow hierarchy", lambda: flow_hierarchy_and_scc(e, G_dir))
    if r is not None:
        scc, hier = r
        g_row["flow_hierarchy"] = hier
        g_row["giant_scc_share"] = float(
            scc.groupby("scc").size().max()) / n
        bt = safe("bow-tie", lambda: bowtie(e, scc, n))
        if bt is not None:
            comp, shares = bt
            g_row.update(shares)
            nm["bowtie"] = comp.reset_index(drop=True)  # 0=CORE 1=IN 2=OUT 3=OTHER

    # ----- percentiles for role-drift tracking -----------------------------------
    for c in ["pagerank_w", "hub_score", "authority_score", "throughput"]:
        if c in nm.columns:
            nm[f"{c}_pct"] = pct_rank(nm, c)

    # ----- attach original node ids & write --------------------------------------
    nm = nm.merge(nodes_df, on="vid", how="left")
    nm["month"] = month
    out = os.path.join(OUTPUT_DIR, "nodes", f"node_metrics_{month}.parquet")
    nm.to_parquet(out)
    log(f"  wrote {out} ({len(nm):,} rows, {len(nm.columns)} cols)")

    louv_named = None
    if louv_labels is not None:
        louv_named = louv_labels.merge(nodes_df, left_on="vertex",
                                       right_on="vid",
                                       how="left")[["node", "community"]]

    return {"g_row": g_row, "edges": e[["source", "target", "amount"]],
            "pairs": neighbor_sets(e).merge(
                nodes_df, on="vid", how="left")
                .merge(nodes_df.rename(columns={"node": "nbr_node",
                                                "vid": "nbr"}),
                       on="nbr", how="left")[["node", "nbr_node"]]
                .rename(columns={"node": "vid", "nbr_node": "nbr"}),
            "node_pcts": nm[["node", "pagerank_w_pct", "hub_score_pct",
                             "authority_score_pct", "throughput_pct"]]
            if "pagerank_w_pct" in nm.columns else None,
            "louvain": louv_named,
            "node_strength": nm[["node", "throughput"]]}


# ==============================================================================
# CROSS-MONTH PASSES
# ==============================================================================
def mom_pass(prev, cur, month):
    log(f"--- MoM {month} vs previous ---")
    summary = {"month": month}

    j = safe("neighbor jaccard churn",
             lambda: jaccard_churn(prev["pairs"], cur["pairs"]))
    drift = None
    if prev["node_pcts"] is not None and cur["node_pcts"] is not None:
        drift = prev["node_pcts"].merge(cur["node_pcts"], on="node",
                                        how="inner", suffixes=("_prev", ""))
        for c in ["pagerank_w_pct", "hub_score_pct",
                  "authority_score_pct", "throughput_pct"]:
            drift[f"{c}_delta"] = drift[c] - drift[f"{c}_prev"]
        drift = drift[["node"] + [c for c in drift.columns if c.endswith("_delta")]]

    node_mom = None
    if j is not None:
        node_mom = j.rename(columns={"vid": "node"})
        if drift is not None:
            node_mom = node_mom.merge(drift, on="node", how="outer")
        node_mom["month"] = month
        p = os.path.join(OUTPUT_DIR, "mom", f"mom_node_{month}.parquet")
        node_mom.to_parquet(p)
        log(f"  wrote {p}")
        summary["median_neighbor_jaccard"] = float(
            node_mom["neighbor_jaccard"].median())

    r = safe("edge persistence / new-edge profile",
             lambda: edge_persistence_and_new(prev["edges"], cur["edges"]))
    if r is not None:
        edge_flags, stats = r
        summary.update(stats)
        if WRITE_EDGE_METRICS:
            p = os.path.join(OUTPUT_DIR, "edges", f"edge_metrics_{month}.parquet")
            edge_flags.to_parquet(p)

    if prev["louvain"] is not None and cur["louvain"] is not None:
        summary["community_ari"] = safe(
            "community ARI",
            lambda: adjusted_rand(prev["louvain"], cur["louvain"]),
            float("nan"))
    return summary


def temporal_rings(months, edge_cache):
    """A->B (m), B->C (m+1), C->A (m+2) rings over consecutive month triples."""
    for i in range(len(months) - 2):
        m1, m2, m3 = months[i], months[i + 1], months[i + 2]
        if not all(m in edge_cache for m in (m1, m2, m3)):
            continue

        def _rings():
            def top(df):
                thr = float(df["amount"].quantile(TEMPORAL_RING_MIN_AMT_Q))
                return df[df["amount"] >= thr]

            e1 = top(edge_cache[m1]).rename(
                columns={"source": "A", "target": "B", "amount": "amt1"})
            e2 = top(edge_cache[m2]).rename(
                columns={"source": "B", "target": "C", "amount": "amt2"})
            e3 = top(edge_cache[m3]).rename(
                columns={"source": "C", "target": "A", "amount": "amt3"})
            r = e1.merge(e2, on="B")
            r = r[r["A"] != r["C"]]
            r = r.merge(e3, on=["C", "A"])
            r["min_amount"] = r[["amt1", "amt2", "amt3"]].min(axis=1)
            return r

        rings = safe(f"temporal rings {m1}->{m3}", _rings)
        if rings is not None and len(rings):
            p = os.path.join(OUTPUT_DIR, "temporal",
                             f"temporal_rings_{m3}.parquet")
            rings.to_parquet(p)
            log(f"  wrote {p} ({len(rings):,} time-ordered rings)")


def longitudinal_pass(months):
    """Burstiness (CV) and lag-1 autocorrelation of node throughput."""
    log("--- longitudinal pass: burstiness & autocorrelation ---")
    frames = []
    for m in months:
        p = os.path.join(OUTPUT_DIR, "nodes", f"node_metrics_{m}.parquet")
        if os.path.exists(p):
            df = cudf.read_parquet(p, columns=["node", "throughput", "month"])
            frames.append(df)
    if len(frames) < 3:
        log("  <3 months of node metrics; skipping")
        return
    allm = cudf.concat(frames)
    g = allm.groupby("node").agg(
        {"throughput": ["mean", "std", "count"]})
    g.columns = ["mean_throughput", "std_throughput", "months_active"]
    g = g.reset_index()
    g["burstiness_cv"] = (g["std_throughput"] /
                          g["mean_throughput"].where(
                              g["mean_throughput"] > 0, 1)).fillna(0)

    # lag-1 autocorrelation of throughput via self-merge on consecutive months
    order_df = cudf.DataFrame({"month": months,
                               "t": cp.arange(len(months), dtype="int32")})
    allm = allm.merge(order_df, on="month", how="left")
    nxt = allm.rename(columns={"throughput": "throughput_next"})
    nxt["t"] = nxt["t"] - 1
    m2 = allm.merge(nxt[["node", "t", "throughput_next"]],
                    on=["node", "t"], how="inner")

    def _ac(df):
        x = df["throughput"].values.astype(cp.float64)
        y = df["throughput_next"].values.astype(cp.float64)
        xm, ym = x - x.mean(), y - y.mean()
        den = cp.sqrt((xm * xm).sum() * (ym * ym).sum())
        return float((xm * ym).sum() / den) if float(den) > 0 else float("nan")

    global_ac = safe("global lag-1 autocorr", lambda: _ac(m2), float("nan"))

    cov = m2.groupby("node").agg({"throughput": ["mean", "std"],
                                  "throughput_next": ["mean", "std"]})
    cov.columns = ["mx", "sx", "my", "sy"]
    m2 = m2.merge(cov.reset_index(), on="node", how="left")
    m2["_z"] = (m2["throughput"] - m2["mx"]) * (m2["throughput_next"] - m2["my"])
    num = m2.groupby("node").agg({"_z": "mean"}).rename(columns={"_z": "cov"})
    per = num.join(cov[["sx", "sy"]]).reset_index()
    per["autocorr_lag1"] = per["cov"] / (per["sx"] * per["sy"]).where(
        (per["sx"] * per["sy"]) > 0, 1)
    g = g.merge(per[["node", "autocorr_lag1"]], on="node", how="left")

    p = os.path.join(OUTPUT_DIR, "temporal", "node_temporal_summary.parquet")
    g.to_parquet(p)
    log(f"  wrote {p} ({len(g):,} nodes); global lag-1 autocorr={global_ac:.4f}")


# ==============================================================================
# OPTIONAL: Personalized PageRank helper (seed-based; not run by default)
# ==============================================================================
def personalized_pagerank(edges_named, seed_nodes, alpha=0.85):
    """PPR from a seed list (label diffusion / similarity targeting)."""
    e, nodes_df = renumber(edges_named)
    G = build_cugraph(e, directed=True)
    seeds = nodes_df.merge(
        cudf.DataFrame({"node": cudf.Series(seed_nodes)}), on="node")
    pers = cudf.DataFrame({"vertex": seeds["vid"],
                           "values": 1.0 / max(len(seeds), 1)})
    pr = cugraph.pagerank(G, alpha=alpha, personalization=pers)
    return pr.merge(nodes_df, left_on="vertex", right_on="vid",
                    how="left")[["node", "pagerank"]]


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    global _LOG_PATH
    ensure_dirs()
    _LOG_PATH = os.path.join(OUTPUT_DIR, "run_log.txt")

    files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
    months = [month_key(f) for f in files]
    files = [f for f, m in zip(files, months) if m]
    months = [m for m in months if m]
    if not files:
        log(f"No files matching {FILE_GLOB} in {INPUT_DIR}")
        return
    log(f"Found {len(files)} monthly snapshots: {months[0]} .. {months[-1]}")

    graph_rows, mom_rows = [], []
    prev, edge_cache = None, {}

    for path, month in zip(files, months):
        cur = process_month(path, month)
        graph_rows.append(cur["g_row"])
        if RUN_TEMPORAL_RINGS:
            edge_cache[month] = cur["edges"]

        if prev is not None:
            mom_rows.append(mom_pass(prev, cur, month))
        # keep only what MoM needs; free the rest
        prev = {k: cur[k] for k in
                ["edges", "pairs", "node_pcts", "louvain"]}

        cudf.DataFrame(graph_rows).to_csv(
            os.path.join(OUTPUT_DIR, "graph_metrics_monthly.csv"), index=False)
        if mom_rows:
            cudf.DataFrame(mom_rows).to_csv(
                os.path.join(OUTPUT_DIR, "mom_summary_monthly.csv"),
                index=False)

    if RUN_TEMPORAL_RINGS:
        temporal_rings(months, edge_cache)

    longitudinal_pass(months)
    log("DONE.")


if __name__ == "__main__":
    main()xs
