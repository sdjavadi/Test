"""
Payment Knowledge Graph (PKG) — Monthly Snapshot Metrics Pipeline (GPU / RAPIDS)
================================================================================
Computes node-level, edge-level, graph-level, and month-over-month (temporal)
network metrics from monthly customer-to-customer payment snapshots.

INPUT
    ../data/customers_YYYY-M-1.csv     columns: source, target, amount, volume
    (volume is ignored — all metrics are AMOUNT-weighted)

OUTPUT  (../result/)
    node_metrics_YYYY-MM.parquet       one row per customer per month
    edge_metrics_YYYY-MM.parquet       one row per (source,target) per month
    graph_metrics_monthly.csv          one row per month (global scalars)
    temporal_graph_metrics.csv         month-over-month graph scalars
    temporal_node_metrics.parquet      per-node MoM metrics (Jaccard churn, ...)
    node_burstiness.parquet            per-node cross-period volatility
    ordered_rings.parquet              temporally ordered 3-cycles (optional)
    run_log.txt                        processing log

REQUIREMENTS
    RAPIDS (cudf, cugraph, cupy) >= 23.x on a CUDA GPU server.

NOTES ON METHOD CHOICES (see METRICS_MANIFEST.md for full definitions)
    - Parallel edges are aggregated (sum of amount) per (source,target);
      therefore degree == distinct-neighbor count in these snapshots.
    - Spectral metrics (weighted PageRank, HITS, Katz, eigenvector) are
      implemented as explicit power iterations on cupy sparse matrices so
      that amount-weighting is guaranteed regardless of cuGraph version.
    - Betweenness is sampled (k sources) and UNWEIGHTED (shortest hop paths);
      current-flow betweenness is not available in cuGraph and is skipped.
    - Expensive sparse-matmul metrics (cycle triangles, Burt constraint) and
      the trophic-level linear solve are gated behind size guards / flags.
"""

from __future__ import annotations

import glob
import math
import os
import re
import sys
import time
import traceback

import cudf
import cugraph
import cupy as cp
from cupyx.scipy import sparse as cusp
from cupyx.scipy.sparse.linalg import cg

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_DIR = "../data"
RESULT_DIR = "../result"
FILE_GLOB = "customers_*.csv"

SRC_COL, DST_COL, AMT_COL = "source", "target", "amount"

# Spectral iteration params
PR_ALPHA = 0.85
PR_ITERS = 60
HITS_ITERS = 60
KATZ_ITERS = 60
KATZ_ALPHA_FRAC = 0.85          # alpha = frac / lambda_max (binary adjacency)
EIG_ITERS = 60
CONV_TOL = 1e-7

# Betweenness sampling (unweighted approximation)
BETWEENNESS_SAMPLES = 256        # set to None to disable
BETWEENNESS_MAX_NODES = 5_000_000

# Expensive sparse-matmul metrics (A@A): guard by nnz of adjacency
ENABLE_CYCLE_TRIANGLES = True
ENABLE_BURT_CONSTRAINT = True
MAX_NNZ_FOR_MATMUL = 60_000_000  # skip matmul metrics above this edge count

# Trophic levels (CG linear solve)
ENABLE_TROPHIC = True
TROPHIC_MAX_NODES = 10_000_000
TROPHIC_CG_TOL = 1e-5
TROPHIC_CG_MAXITER = 300

# Rich club
RICH_CLUB_TOP_FRAC = 0.01        # top 1% by strength

# Temporal ordered rings A->B (t), B->C (t+1), C->A (t+2)
ENABLE_ORDERED_RINGS = True
RING_AMOUNT_PERCENTILE = 0.90    # only edges above this amount percentile

# Suspicious new-edge flag: new edge, ~zero embeddedness, large amount
NEW_EDGE_LARGE_PERCENTILE = 0.99
EMBED_EPS = 1e-9

FLOAT = cp.float64
UNREACH = 2**31 - 1              # cuGraph BFS sentinel


# ----------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------
os.makedirs(RESULT_DIR, exist_ok=True)
_LOG_PATH = os.path.join(RESULT_DIR, "run_log.txt")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_LOG_PATH, "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------
# FILE DISCOVERY
# ----------------------------------------------------------------------------
def discover_files() -> list[tuple[str, str]]:
    """Return [(month_key 'YYYY-MM', path)] sorted chronologically."""
    pat = re.compile(r"customers_(\d{4})-(\d{1,2})-(\d{1,2})\.csv$")
    out = []
    for p in glob.glob(os.path.join(DATA_DIR, FILE_GLOB)):
        m = pat.search(os.path.basename(p))
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            out.append((f"{y}-{mo:02d}", p))
    out.sort(key=lambda t: t[0])
    return out


# ----------------------------------------------------------------------------
# LOADING & GRAPH CONSTRUCTION
# ----------------------------------------------------------------------------
def load_month(path: str) -> cudf.DataFrame:
    """Load a snapshot, drop volume, aggregate parallel edges by amount sum,
    remove self-loops (kept aside as a scalar in graph metrics)."""
    df = cudf.read_csv(path, usecols=[SRC_COL, DST_COL, AMT_COL])
    df = df.dropna(subset=[SRC_COL, DST_COL])
    df[AMT_COL] = df[AMT_COL].astype("float64")
    df = (
        df.groupby([SRC_COL, DST_COL], as_index=False)
        .agg({AMT_COL: "sum"})
    )
    return df


def build_ids(edges: cudf.DataFrame):
    """Map original node ids to contiguous ints [0, n). Returns
    (edf[sid, tid, amount], nodes cudf.DataFrame[node, id], n, self_loop_amt)."""
    nodes = cudf.concat([edges[SRC_COL], edges[DST_COL]]).unique()
    nodes = nodes.sort_values().reset_index(drop=True)
    nmap = cudf.DataFrame({"node": nodes, "id": cp.arange(len(nodes), dtype=cp.int32)})

    edf = edges.merge(nmap.rename(columns={"node": SRC_COL, "id": "sid"}), on=SRC_COL)
    edf = edf.merge(nmap.rename(columns={"node": DST_COL, "id": "tid"}), on=DST_COL)

    loop_mask = edf["sid"] == edf["tid"]
    self_loop_amt = float(edf.loc[loop_mask, AMT_COL].sum()) if loop_mask.any() else 0.0
    edf = edf[~loop_mask][["sid", "tid", AMT_COL]].reset_index(drop=True)
    return edf, nmap, len(nodes), self_loop_amt


def build_matrices(edf: cudf.DataFrame, n: int):
    """Build cupy sparse matrices:
       W  : directed amount-weighted CSR (rows = source)
       Wu : symmetric amount-weighted CSR (undirected projection, weights summed)
       A  : directed binary CSR
    """
    r = cp.asarray(edf["sid"].values, dtype=cp.int32)
    c = cp.asarray(edf["tid"].values, dtype=cp.int32)
    w = cp.asarray(edf[AMT_COL].values, dtype=FLOAT)

    W = cusp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    Wu = (W + W.T).tocsr()
    A = cusp.coo_matrix((cp.ones_like(w), (r, c)), shape=(n, n)).tocsr()
    A.data = cp.minimum(A.data, 1.0)
    return W, Wu, A


def build_cugraphs(edf: cudf.DataFrame):
    """Directed weighted G, undirected weighted Gu, reversed directed Gr."""
    g_df = edf.rename(columns={"sid": "src", "tid": "dst"})

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(g_df, source="src", destination="dst",
                         edge_attr=AMT_COL, renumber=False)

    und = cudf.concat([
        g_df[["src", "dst", AMT_COL]],
        g_df.rename(columns={"src": "dst", "dst": "src"})[["src", "dst", AMT_COL]],
    ])
    key = und["src"] <= und["dst"]
    und["a"] = und["src"].where(key, und["dst"])
    und["b"] = und["dst"].where(key, und["src"])
    und = und.groupby(["a", "b"], as_index=False).agg({AMT_COL: "sum"})
    Gu = cugraph.Graph(directed=False)
    Gu.from_cudf_edgelist(und, source="a", destination="b",
                          edge_attr=AMT_COL, renumber=False)

    Gr = cugraph.Graph(directed=True)
    Gr.from_cudf_edgelist(
        g_df.rename(columns={"src": "dst", "dst": "src"}),
        source="src", destination="dst", edge_attr=AMT_COL, renumber=False)
    return G, Gu, Gr, und


# ----------------------------------------------------------------------------
# NODE METRICS — basic flow structure (pure cudf/cupy)
# ----------------------------------------------------------------------------
def basic_node_metrics(edf: cudf.DataFrame, n: int) -> cudf.DataFrame:
    zeros = cp.zeros(n, dtype=FLOAT)

    def agg_to_array(df, key, col, how):
        g = df.groupby(key).agg({col: how}).reset_index()
        arr = zeros.copy()
        arr[cp.asarray(g[key].values)] = cp.asarray(g[col].values, dtype=FLOAT)
        return arr

    out_deg = agg_to_array(edf.assign(one=1.0), "sid", "one", "sum")
    in_deg = agg_to_array(edf.assign(one=1.0), "tid", "one", "sum")
    out_str = agg_to_array(edf, "sid", AMT_COL, "sum")
    in_str = agg_to_array(edf, "tid", AMT_COL, "sum")

    gross = in_str + out_str
    net = in_str - out_str
    mn = cp.minimum(in_str, out_str)
    mx = cp.maximum(in_str, out_str)
    pass_ratio = cp.where(mx > 0, mn / mx, 0.0)          # ~1 => pass-through
    net_norm = cp.where(gross > 0, net / gross, 0.0)     # -1 pure source .. +1 pure sink

    # Disparity Y = sum p^2 and normalized Shannon entropy, per direction
    def disparity_entropy(df, key):
        tot = df.groupby(key).agg({AMT_COL: "sum"}).reset_index() \
                .rename(columns={AMT_COL: "tot"})
        d = df.merge(tot, on=key)
        p = d[AMT_COL] / d["tot"]
        d = d.assign(p2=p * p, plogp=-(p * p.log()))
        g = d.groupby(key).agg({"p2": "sum", "plogp": "sum", AMT_COL: "count"}) \
             .reset_index().rename(columns={AMT_COL: "k"})
        idx = cp.asarray(g[key].values)
        Y = zeros.copy(); Y[idx] = cp.asarray(g["p2"].values, dtype=FLOAT)
        k = cp.asarray(g["k"].values, dtype=FLOAT)
        H = cp.asarray(g["plogp"].values, dtype=FLOAT)
        Hn = cp.where(k > 1, H / cp.log(k), 0.0)
        Hout = zeros.copy(); Hout[idx] = Hn
        return Y, Hout

    disp_out, ent_out = disparity_entropy(edf, "sid")
    disp_in, ent_in = disparity_entropy(edf, "tid")

    # Node-level weighted reciprocity: min(w_ij, w_ji) summed over partners / out_strength
    rev = edf.rename(columns={"sid": "tid", "tid": "sid", AMT_COL: "amt_rev"})
    both = edf.merge(rev, on=["sid", "tid"], how="inner")
    if len(both):
        both = both.assign(recip=both[[AMT_COL, "amt_rev"]].min(axis=1))
        g = both.groupby("sid").agg({"recip": "sum"}).reset_index()
        recip = zeros.copy()
        recip[cp.asarray(g["sid"].values)] = cp.asarray(g["recip"].values, dtype=FLOAT)
    else:
        recip = zeros.copy()
    recip_ratio = cp.where(out_str > 0, recip / out_str, 0.0)

    return cudf.DataFrame({
        "id": cp.arange(n, dtype=cp.int32),
        "in_degree": in_deg, "out_degree": out_deg, "degree": in_deg + out_deg,
        "in_strength": in_str, "out_strength": out_str, "strength": gross,
        "net_flow": net, "net_flow_norm": net_norm,
        "throughput": mn, "passthrough_ratio": pass_ratio,
        "disparity_in": disp_in, "disparity_out": disp_out,
        "entropy_in": ent_in, "entropy_out": ent_out,
        "reciprocity_w": recip_ratio,
    })


# ----------------------------------------------------------------------------
# SPECTRAL METRICS (cupy power iterations — guaranteed amount-weighted)
# ----------------------------------------------------------------------------
def _norm(v):
    s = cp.linalg.norm(v)
    return v / s if s > 0 else v


def weighted_pagerank(W, alpha=PR_ALPHA, iters=PR_ITERS):
    n = W.shape[0]
    out_str = cp.asarray(W.sum(axis=1)).ravel()
    inv = cp.where(out_str > 0, 1.0 / out_str, 0.0)
    P = cusp.diags(inv) @ W                       # row-stochastic on non-dangling
    dangling = (out_str == 0).astype(FLOAT)
    x = cp.full(n, 1.0 / n, dtype=FLOAT)
    for _ in range(iters):
        xd = float(dangling @ x)
        x_new = alpha * (P.T @ x) + (alpha * xd + (1 - alpha)) / n
        if float(cp.abs(x_new - x).sum()) < CONV_TOL:
            x = x_new; break
        x = x_new
    return x


def weighted_hits(W, iters=HITS_ITERS):
    n = W.shape[0]
    h = cp.full(n, 1.0 / math.sqrt(n), dtype=FLOAT)
    a = h.copy()
    for _ in range(iters):
        a_new = _norm(W.T @ h)                    # authorities: receive from good hubs
        h_new = _norm(W @ a_new)                  # hubs: pay good authorities
        if float(cp.abs(h_new - h).max()) < CONV_TOL:
            h, a = h_new, a_new; break
        h, a = h_new, a_new
    return h, a


def lambda_max(A, iters=50):
    n = A.shape[0]
    x = cp.random.default_rng(0).random(n, dtype=FLOAT)
    x = _norm(x)
    lam = 1.0
    for _ in range(iters):
        y = A.T @ x
        ny = cp.linalg.norm(y)
        if ny == 0:
            return 1.0
        lam = float(ny); x = y / ny
    return max(lam, 1.0)


def katz_in(A, iters=KATZ_ITERS):
    """Katz centrality over incoming BINARY paths (path-count influence)."""
    n = A.shape[0]
    alpha = KATZ_ALPHA_FRAC / lambda_max(A)
    beta = cp.ones(n, dtype=FLOAT)
    x = beta.copy()
    for _ in range(iters):
        x_new = alpha * (A.T @ x) + beta
        if float(cp.abs(x_new - x).max() / (cp.abs(x).max() + 1e-12)) < CONV_TOL:
            x = x_new; break
        x = x_new
    return x / cp.linalg.norm(x)


def eigenvector_undirected(Wu, iters=EIG_ITERS):
    n = Wu.shape[0]
    x = cp.full(n, 1.0 / math.sqrt(n), dtype=FLOAT)
    for _ in range(iters):
        x_new = _norm(Wu @ x)
        if float(cp.abs(x_new - x).max()) < CONV_TOL:
            x = x_new; break
        x = x_new
    return x


# ----------------------------------------------------------------------------
# cuGraph METRICS: betweenness, k-core, communities, triangles, embeddedness
# ----------------------------------------------------------------------------
def series_to_array(df, vcol, scol, n, dtype=FLOAT):
    arr = cp.zeros(n, dtype=dtype)
    arr[cp.asarray(df[vcol].values)] = cp.asarray(df[scol].values, dtype=dtype)
    return arr


def cugraph_node_metrics(G, Gu, n):
    res = {}

    # Sampled unweighted betweenness (hop-path approximation)
    if BETWEENNESS_SAMPLES and n <= BETWEENNESS_MAX_NODES:
        k = min(BETWEENNESS_SAMPLES, n)
        try:
            b = cugraph.betweenness_centrality(G, k=k, normalized=True)
            res["betweenness_approx"] = series_to_array(
                b, "vertex", "betweenness_centrality", n)
        except Exception as e:
            log(f"    betweenness skipped: {e}")

    # k-core (undirected)
    try:
        core = cugraph.core_number(Gu)
        cn = series_to_array(core, "vertex", "core_number", n)
        res["core_number"] = cn
        res["coreness_norm"] = cn / max(float(cn.max()), 1.0)
    except Exception as e:
        log(f"    core_number skipped: {e}")

    # Undirected local clustering via triangle counts: 2T / (d(d-1))
    try:
        tri = cugraph.triangle_count(Gu)
        ccol = "counts" if "counts" in tri.columns else "triangle_count"
        t = series_to_array(tri, "vertex", ccol, n)
        du = series_to_array(Gu.degree(), "vertex", "degree", n)
        denom = du * (du - 1.0)
        res["triangles_u"] = t
        res["clustering_u"] = cp.where(denom > 0, 2.0 * t / denom, 0.0)
    except Exception as e:
        log(f"    triangles/clustering skipped: {e}")

    # Communities: Leiden preferred, Louvain fallback
    mod, labels = float("nan"), None
    for fn, name in ((getattr(cugraph, "leiden", None), "leiden"),
                     (cugraph.louvain, "louvain")):
        if fn is None:
            continue
        try:
            r = fn(Gu)
            parts, mod = (r if isinstance(r, tuple) else (r, float("nan")))
            labels = series_to_array(parts, "vertex", "partition", n, dtype=cp.int64)
            log(f"    communities via {name}: modularity={mod:.4f}")
            break
        except Exception as e:
            log(f"    {name} failed: {e}")
    res["community"] = labels if labels is not None else cp.full(n, -1, dtype=cp.int64)
    return res, mod


def edge_embeddedness(Gu, und_edges: cudf.DataFrame) -> cudf.DataFrame:
    """Jaccard neighborhood overlap for every undirected edge."""
    pairs = und_edges[["a", "b"]].rename(columns={"a": "first", "b": "second"})
    try:
        j = cugraph.jaccard(Gu, vertex_pair=pairs)
        return j.rename(columns={"jaccard_coeff": "embeddedness"})
    except Exception as e:
        log(f"    edge embeddedness skipped: {e}")
        return None


# ----------------------------------------------------------------------------
# BOW-TIE, FLOW HIERARCHY, COMPONENTS
# ----------------------------------------------------------------------------
def components_and_bowtie(G, Gr, edf, n):
    out = {}
    # Weakly connected — giant share
    wcc = cugraph.weakly_connected_components(G)
    sizes = wcc.groupby("labels").size().reset_index().rename(columns={0: "sz"})
    szcol = [c for c in sizes.columns if c != "labels"][0]
    out["giant_wcc_share"] = float(sizes[szcol].max()) / n

    # Strongly connected — core of the bow-tie
    scc = cugraph.strongly_connected_components(G)
    scc_sizes = scc.groupby("labels").size().reset_index()
    szc = [c for c in scc_sizes.columns if c != "labels"][0]
    big_label = scc_sizes.sort_values(szc, ascending=False)["labels"].iloc[0]
    scc_arr = series_to_array(scc, "vertex", "labels", n, dtype=cp.int64)

    core_mask = scc_arr == int(big_label)
    core_nodes = cp.where(core_mask)[0]
    bowtie = cp.full(n, 3, dtype=cp.int8)         # 3 = TENDRIL/OTHER
    if len(core_nodes) > 1:
        seed = int(core_nodes[0])
        fwd = cugraph.bfs(G, start=seed)          # core ∪ OUT
        bwd = cugraph.bfs(Gr, start=seed)         # core ∪ IN
        f = series_to_array(fwd, "vertex", "distance", n, dtype=cp.int64)
        bkw = series_to_array(bwd, "vertex", "distance", n, dtype=cp.int64)
        # unreached vertices may be absent from bfs result -> 0 default; fix:
        f[f == 0] = cp.where(cp.arange(n) == seed, 0, f[f == 0])
        reach_f = (f > 0) | (cp.arange(n) == seed)
        reach_b = (bkw > 0) | (cp.arange(n) == seed)
        reach_f = reach_f & (f < UNREACH)
        reach_b = reach_b & (bkw < UNREACH)
        bowtie[reach_f & ~core_mask] = 2          # OUT
        bowtie[reach_b & ~core_mask] = 1          # IN
        bowtie[core_mask] = 0                     # CORE
    out["scc_core_share"] = float(core_mask.sum()) / n
    out["bowtie_in_share"] = float((bowtie == 1).sum()) / n
    out["bowtie_out_share"] = float((bowtie == 2).sum()) / n
    out["bowtie_other_share"] = float((bowtie == 3).sum()) / n

    # Flow hierarchy: 1 - fraction of edges inside a nontrivial SCC
    scc_size_map = cudf.DataFrame({"labels": scc_sizes["labels"],
                                   "scc_sz": scc_sizes[szc]})
    e = edf.copy()
    e["scc_s"] = cudf.Series(scc_arr[cp.asarray(e["sid"].values)])
    e["scc_t"] = cudf.Series(scc_arr[cp.asarray(e["tid"].values)])
    same = e["scc_s"] == e["scc_t"]
    e = e.merge(scc_size_map.rename(columns={"labels": "scc_s"}), on="scc_s", how="left")
    cyclic = same & (e["scc_sz"] > 1)
    n_edges = len(e)
    tot_amt = float(e[AMT_COL].sum())
    out["flow_hierarchy_cnt"] = 1.0 - float(cyclic.sum()) / max(n_edges, 1)
    out["flow_hierarchy_amt"] = 1.0 - float(e.loc[cyclic, AMT_COL].sum()) / max(tot_amt, 1e-12)
    return out, bowtie


# ----------------------------------------------------------------------------
# GRAPH-LEVEL SCALARS
# ----------------------------------------------------------------------------
def gini(x: cp.ndarray) -> float:
    x = cp.sort(x[x >= 0])
    nn = len(x)
    if nn == 0 or float(x.sum()) == 0:
        return 0.0
    i = cp.arange(1, nn + 1, dtype=FLOAT)
    return float((2 * (i * x).sum() / (nn * x.sum())) - (nn + 1) / nn)


def graph_scalars(edf, W, Wu, nd, n, self_loop_amt):
    m = len(edf)
    tot = float(edf[AMT_COL].sum())
    s = cp.asarray(nd["strength"].values, dtype=FLOAT)
    din = cp.asarray(nd["in_strength"].values, dtype=FLOAT)

    res = {
        "n_nodes": n, "n_edges": m, "total_amount": tot,
        "self_loop_amount": self_loop_amt,
        "density": m / (n * (n - 1)) if n > 1 else 0.0,
        "avg_degree": 2.0 * m / n,
        "gini_strength": gini(s),
        "hhi_in_strength": float(((din / max(tot, 1e-12)) ** 2).sum()),
    }

    # Weighted reciprocity: sum min(w_ij, w_ji) / sum w_ij
    rev = edf.rename(columns={"sid": "tid", "tid": "sid", AMT_COL: "amt_rev"})
    both = edf.merge(rev, on=["sid", "tid"], how="inner")
    rsum = float(both[[AMT_COL, "amt_rev"]].min(axis=1).sum()) if len(both) else 0.0
    res["reciprocity_w"] = rsum / max(tot, 1e-12)

    # Weighted strength assortativity: amount-weighted Pearson corr of endpoint strengths
    su = s[cp.asarray(edf["sid"].values)]
    sv = s[cp.asarray(edf["tid"].values)]
    w = cp.asarray(edf[AMT_COL].values, dtype=FLOAT)
    wn = w / w.sum()
    mu_u, mu_v = float(wn @ su), float(wn @ sv)
    cov = float(wn @ ((su - mu_u) * (sv - mu_v)))
    sd = math.sqrt(float(wn @ (su - mu_u) ** 2)) * math.sqrt(float(wn @ (sv - mu_v) ** 2))
    res["assortativity_w"] = cov / sd if sd > 0 else 0.0

    # s(k) scaling exponent beta: log strength ~ beta * log degree
    deg = cp.asarray(nd["degree"].values, dtype=FLOAT)
    mask = (deg > 0) & (s > 0)
    if int(mask.sum()) > 10:
        lx, ly = cp.log(deg[mask]), cp.log(s[mask])
        lxm, lym = lx.mean(), ly.mean()
        res["s_k_beta"] = float(((lx - lxm) * (ly - lym)).sum() / ((lx - lxm) ** 2).sum())
    else:
        res["s_k_beta"] = float("nan")

    # Rich club (weighted, unnormalized): amount share flowing among top-frac strength nodes
    k_top = max(int(n * RICH_CLUB_TOP_FRAC), 2)
    top_ids = cp.argsort(-s)[:k_top]
    top_mask = cp.zeros(n, dtype=bool); top_mask[top_ids] = True
    e_top = top_mask[cp.asarray(edf["sid"].values)] & top_mask[cp.asarray(edf["tid"].values)]
    amt_top = float(cp.asarray(edf[AMT_COL].values, dtype=FLOAT)[e_top].sum())
    res["rich_club_amount_share"] = amt_top / max(tot, 1e-12)
    res["rich_club_edge_density"] = float(e_top.sum()) / (k_top * (k_top - 1))

    # Von Neumann entropy — quadratic (Tsallis-2) approximation on undirected Laplacian
    d = cp.asarray(Wu.sum(axis=1)).ravel()
    trL = float(d.sum())
    trL2 = float((d ** 2).sum() + (Wu.data ** 2).sum())
    res["vn_entropy_q"] = 1.0 - trL2 / max(trL ** 2, 1e-12)

    # Degree-distribution entropy (normalized)
    dv, cnt = cp.unique(deg.astype(cp.int64), return_counts=True)
    p = cnt.astype(FLOAT) / n
    Hd = float(-(p * cp.log(p)).sum())
    res["degree_dist_entropy"] = Hd / math.log(max(len(dv), 2))
    return res


# ----------------------------------------------------------------------------
# EXPENSIVE MATMUL METRICS (guarded)
# ----------------------------------------------------------------------------
def cycle_triangles(A):
    """Directed cycle triangles through each node: diag(A^3) via
    rowsum((A@A) ∘ A^T). Guarded by MAX_NNZ_FOR_MATMUL."""
    AA = A @ A
    M = AA.multiply(A.T)
    return cp.asarray(M.sum(axis=1)).ravel()


def burt_constraint(Wu):
    """Burt's constraint c_i = Σ_{j∈N(i)} (p_ij + Σ_q p_iq p_qj)^2 on the
    undirected amount-weighted graph."""
    rs = cp.asarray(Wu.sum(axis=1)).ravel()
    inv = cp.where(rs > 0, 1.0 / rs, 0.0)
    P = cusp.diags(inv) @ Wu
    M = P + (P @ P)
    pattern = Wu.copy(); pattern.data = cp.ones_like(pattern.data)
    Mn = M.multiply(pattern)
    Mn.data = Mn.data ** 2
    return cp.asarray(Mn.sum(axis=1)).ravel()


def trophic_levels(W, n):
    """MacKay et al. trophic levels: solve (diag(u) - W - W^T) h = v, where
    u = in+out strength, v = in - out strength; returns (h, incoherence F0)."""
    win = cp.asarray(W.sum(axis=0)).ravel()
    wout = cp.asarray(W.sum(axis=1)).ravel()
    u = win + wout
    v = win - wout
    L = cusp.diags(u + 1e-8) - W - W.T
    h, info = cg(L.tocsr(), v, tol=TROPHIC_CG_TOL, maxiter=TROPHIC_CG_MAXITER)
    if info != 0:
        log(f"    trophic CG did not fully converge (info={info}); using best iterate")
    h = h - h.min()
    # incoherence F0 = Σ w_ij (h_j - h_i - 1)^2 / Σ w_ij
    coo = W.tocoo()
    diff = h[coo.col] - h[coo.row] - 1.0
    F0 = float((coo.data * diff ** 2).sum() / max(float(coo.data.sum()), 1e-12))
    return h, F0


# ----------------------------------------------------------------------------
# TEMPORAL (month-over-month) METRICS
# ----------------------------------------------------------------------------
def undirected_neighbor_pairs(edf: cudf.DataFrame) -> cudf.DataFrame:
    a = edf[["sid", "tid"]]
    b = a.rename(columns={"sid": "tid", "tid": "sid"})[["sid", "tid"]]
    p = cudf.concat([a, b]).drop_duplicates()
    return p.rename(columns={"sid": "node", "tid": "nbr"})


def temporal_pass(prev_edf, cur_edf, prev_pairs, cur_pairs, n_prev_amt_p99):
    """Returns (graph_row dict, node_df, cur_edge_flags df)."""
    g = {}

    # Edge persistence (directed pairs)
    inter = prev_edf.merge(cur_edf, on=["sid", "tid"], how="inner",
                           suffixes=("_p", "_c"))
    g["edge_persistence"] = len(inter) / max(len(prev_edf), 1)

    # New vs persisting edge amount profiles
    cur_flag = cur_edf.merge(prev_edf[["sid", "tid"]].assign(existed=1),
                             on=["sid", "tid"], how="left")
    cur_flag["is_new"] = cur_flag["existed"].isnull()
    new_e = cur_flag[cur_flag["is_new"]]
    old_e = cur_flag[~cur_flag["is_new"]]
    g["n_new_edges"] = len(new_e)
    g["new_edge_mean_amount"] = float(new_e[AMT_COL].mean()) if len(new_e) else 0.0
    g["persisting_edge_mean_amount"] = float(old_e[AMT_COL].mean()) if len(old_e) else 0.0
    g["new_edge_amount_share"] = float(new_e[AMT_COL].sum()) / max(float(cur_edf[AMT_COL].sum()), 1e-12)
    g["new_edge_large_count_p99"] = int((new_e[AMT_COL] > n_prev_amt_p99).sum()) if len(new_e) else 0

    # Per-node Jaccard churn of neighbor sets
    ip = prev_pairs.merge(cur_pairs, on=["node", "nbr"], how="inner") \
                   .groupby("node").size().reset_index().rename(columns={0: "inter"})
    icol = [c for c in ip.columns if c != "node"][0]
    dp = prev_pairs.groupby("node").size().reset_index()
    dpc = [c for c in dp.columns if c != "node"][0]
    dc = cur_pairs.groupby("node").size().reset_index()
    dcc = [c for c in dc.columns if c != "node"][0]
    jd = dp.rename(columns={dpc: "deg_prev"}) \
           .merge(dc.rename(columns={dcc: "deg_cur"}), on="node", how="outer") \
           .merge(ip.rename(columns={icol: "inter"}), on="node", how="left")
    jd = jd.fillna(0)
    union = jd["deg_prev"] + jd["deg_cur"] - jd["inter"]
    jd["neighbor_jaccard_prev"] = (jd["inter"] / union.where(union > 0, 1)).fillna(0)
    node_df = jd[["node", "deg_prev", "deg_cur", "neighbor_jaccard_prev"]]
    g["mean_neighbor_jaccard"] = float(jd["neighbor_jaccard_prev"].mean())
    return g, node_df, cur_flag[["sid", "tid", "is_new"]]


def strength_autocorr(prev_nd, cur_nd):
    a = prev_nd[["id", "strength"]].rename(columns={"strength": "s_prev"})
    b = cur_nd[["id", "strength"]].rename(columns={"strength": "s_cur"})
    m = a.merge(b, on="id", how="inner")
    if len(m) < 10:
        return float("nan")
    x = cp.asarray(m["s_prev"].values, dtype=FLOAT)
    y = cp.asarray(m["s_cur"].values, dtype=FLOAT)
    x, y = cp.log1p(x), cp.log1p(y)
    xm, ym = x - x.mean(), y - y.mean()
    denom = float(cp.linalg.norm(xm) * cp.linalg.norm(ym))
    return float((xm @ ym)) / denom if denom > 0 else float("nan")


def community_stability(prev_labels_df, cur_labels_df):
    """ARI/NMI on nodes present in both months (CPU/sklearn on label arrays)."""
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ImportError:
        return float("nan"), float("nan")
    m = prev_labels_df.merge(cur_labels_df, on="node", how="inner",
                             suffixes=("_p", "_c"))
    if len(m) < 10:
        return float("nan"), float("nan")
    lp = m["community_p"].to_numpy()
    lc = m["community_c"].to_numpy()
    return adjusted_rand_score(lp, lc), normalized_mutual_info_score(lp, lc)


def ordered_rings(e_t, e_t1, e_t2, months):
    """A->B in t, B->C in t+1, C->A in t+2 on top-amount edges only."""
    def top(e):
        thr = e[AMT_COL].quantile(RING_AMOUNT_PERCENTILE)
        return e[e[AMT_COL] >= thr][[SRC_COL, DST_COL, AMT_COL]]

    r1 = top(e_t).rename(columns={SRC_COL: "a", DST_COL: "b", AMT_COL: "amt1"})
    r2 = top(e_t1).rename(columns={SRC_COL: "b", DST_COL: "c", AMT_COL: "amt2"})
    r3 = top(e_t2).rename(columns={SRC_COL: "c", DST_COL: "a", AMT_COL: "amt3"})
    x = r1.merge(r2, on="b").merge(r3, on=["c", "a"])
    x = x[(x["a"] != x["b"]) & (x["b"] != x["c"]) & (x["a"] != x["c"])]
    if len(x):
        x["month_start"] = months[0]
        x["min_leg_amount"] = x[["amt1", "amt2", "amt3"]].min(axis=1)
    return x


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def process_month(month, path, prev_state):
    log(f"== {month} :: {os.path.basename(path)}")
    edges_orig = load_month(path)
    edf, nmap, n, self_loop_amt = build_ids(edges_orig)
    m = len(edf)
    log(f"    nodes={n:,} edges={m:,}")
    if n == 0 or m == 0:
        return prev_state

    W, Wu, A = build_matrices(edf, n)
    G, Gu, Gr, und_edges = build_cugraphs(edf)

    # ---- node metrics -------------------------------------------------------
    nd = basic_node_metrics(edf, n)
    nd["pagerank_w"] = weighted_pagerank(W)
    hub, auth = weighted_hits(W)
    nd["hits_hub"], nd["hits_authority"] = hub, auth
    nd["katz_in"] = katz_in(A)
    nd["eigenvector_u"] = eigenvector_undirected(Wu)

    cg_res, modularity = cugraph_node_metrics(G, Gu, n)
    for k_, v_ in cg_res.items():
        nd[k_] = v_

    if "betweenness_approx" in nd.columns and "clustering_u" in nd.columns:
        bc = cp.asarray(nd["betweenness_approx"].values, dtype=FLOAT)
        cl = cp.asarray(nd["clustering_u"].values, dtype=FLOAT)
        nd["bridging_centrality"] = bc / (cl + 0.01)

    if ENABLE_CYCLE_TRIANGLES and A.nnz <= MAX_NNZ_FOR_MATMUL:
        try:
            nd["cycle_triangles"] = cycle_triangles(A)
        except Exception as e:
            log(f"    cycle_triangles skipped: {e}")
    if ENABLE_BURT_CONSTRAINT and Wu.nnz <= MAX_NNZ_FOR_MATMUL:
        try:
            nd["burt_constraint"] = burt_constraint(Wu)
        except Exception as e:
            log(f"    burt_constraint skipped: {e}")

    troph_F0 = float("nan")
    if ENABLE_TROPHIC and n <= TROPHIC_MAX_NODES:
        try:
            h, troph_F0 = trophic_levels(W, n)
            nd["trophic_level"] = h
        except Exception as e:
            log(f"    trophic skipped: {e}")

    # ---- structure & graph scalars ------------------------------------------
    comp, bowtie = components_and_bowtie(G, Gr, edf, n)
    nd["bowtie_class"] = bowtie          # 0 CORE, 1 IN, 2 OUT, 3 OTHER
    gs = graph_scalars(edf, W, Wu, nd, n, self_loop_amt)
    gs.update(comp)
    gs["modularity"] = modularity
    gs["n_communities"] = int(nd["community"].nunique())
    gs["trophic_incoherence_F0"] = troph_F0
    gs["month"] = month

    # ---- edge metrics --------------------------------------------------------
    emb = edge_embeddedness(Gu, und_edges)
    e_out = edf.merge(nmap.rename(columns={"id": "sid", "node": SRC_COL}), on="sid") \
               .merge(nmap.rename(columns={"id": "tid", "node": DST_COL}), on="tid")
    if emb is not None:
        key = e_out["sid"] <= e_out["tid"]
        e_out["a"] = e_out["sid"].where(key, e_out["tid"])
        e_out["b"] = e_out["tid"].where(key, e_out["sid"])
        e_out = e_out.merge(emb.rename(columns={"first": "a", "second": "b"}),
                            on=["a", "b"], how="left")
        e_out = e_out.drop(columns=["a", "b"])

    # ---- temporal pass --------------------------------------------------------
    cur_pairs = undirected_neighbor_pairs(edf)
    # map ids -> original labels for cross-month joins (ids differ per month!)
    cur_pairs = cur_pairs.merge(nmap.rename(columns={"id": "node", "node": "node_lbl"}),
                                on="node") \
                         .merge(nmap.rename(columns={"id": "nbr", "node": "nbr_lbl"}),
                                on="nbr")[["node_lbl", "nbr_lbl"]] \
                         .rename(columns={"node_lbl": "node", "nbr_lbl": "nbr"})
    cur_edges_lbl = e_out[[SRC_COL, DST_COL, AMT_COL]] \
        .rename(columns={SRC_COL: "sid", DST_COL: "tid"})
    cur_nd_lbl = nd.merge(nmap, on="id")[["node", "strength"]] \
                   .rename(columns={"strength": "strength"}).assign(id=lambda d: d["node"])
    labels_lbl = nd.merge(nmap, on="id")[["node", "community"]]

    temporal_rows, temporal_node = None, None
    if prev_state is not None:
        p_edges, p_pairs, p_nd, p_labels, p_month, p_p99, p_raw = prev_state
        trow, tnode, edge_flags = temporal_pass(
            p_edges, cur_edges_lbl, p_pairs, cur_pairs, p_p99)
        trow["month"] = month
        trow["prev_month"] = p_month
        trow["strength_autocorr_log"] = strength_autocorr(p_nd, cur_nd_lbl)
        ari, nmi = community_stability(p_labels, labels_lbl)
        trow["community_ARI"], trow["community_NMI"] = ari, nmi
        temporal_rows, temporal_node = trow, tnode.assign(month=month)

        # is_new + suspicious flag onto edge table
        e_out = e_out.merge(
            edge_flags.rename(columns={"sid": SRC_COL, "tid": DST_COL}),
            on=[SRC_COL, DST_COL], how="left")
        e_out["is_new"] = e_out["is_new"].fillna(True)
        p99 = float(e_out[AMT_COL].quantile(NEW_EDGE_LARGE_PERCENTILE))
        if "embeddedness" in e_out.columns:
            e_out["suspicious_new_edge"] = (
                e_out["is_new"]
                & (e_out["embeddedness"].fillna(0) <= EMBED_EPS)
                & (e_out[AMT_COL] >= p99))

    # ---- write ----------------------------------------------------------------
    nd_out = nd.merge(nmap, on="id").drop(columns=["id"])
    cols = ["node"] + [c for c in nd_out.columns if c != "node"]
    nd_out[cols].to_parquet(os.path.join(RESULT_DIR, f"node_metrics_{month}.parquet"))
    e_out.drop(columns=["sid", "tid"], errors="ignore") \
         .to_parquet(os.path.join(RESULT_DIR, f"edge_metrics_{month}.parquet"))

    p99_amt = float(edf[AMT_COL].quantile(NEW_EDGE_LARGE_PERCENTILE))
    new_state = (cur_edges_lbl, cur_pairs, cur_nd_lbl, labels_lbl,
                 month, p99_amt, edges_orig)
    return gs, temporal_rows, temporal_node, new_state


def main():
    files = discover_files()
    if not files:
        log(f"No files matching {FILE_GLOB} in {DATA_DIR}"); sys.exit(1)
    log(f"Found {len(files)} monthly snapshots: {files[0][0]} .. {files[-1][0]}")

    graph_rows, temporal_graph_rows, temporal_node_frames = [], [], []
    strength_history = []            # (month, node, strength) for burstiness
    raw_history = {}                 # month -> original edges (for ordered rings)
    prev_state = None

    for month, path in files:
        try:
            gs, trow, tnode, prev_state = process_month(month, path, prev_state)
            graph_rows.append(gs)
            if trow is not None:
                temporal_graph_rows.append(trow)
            if tnode is not None:
                temporal_node_frames.append(tnode)
            cur_nd = prev_state[2]
            strength_history.append(
                cur_nd[["node", "strength"]].assign(month=month))
            if ENABLE_ORDERED_RINGS:
                raw_history[month] = prev_state[6]
        except Exception:
            log(f"!! {month} FAILED:\n{traceback.format_exc()}")
        cp.get_default_memory_pool().free_all_blocks()

    # ---- graph-level outputs ----
    import pandas as pd
    pd.DataFrame(graph_rows).set_index("month").to_csv(
        os.path.join(RESULT_DIR, "graph_metrics_monthly.csv"))
    if temporal_graph_rows:
        pd.DataFrame(temporal_graph_rows).set_index("month").to_csv(
            os.path.join(RESULT_DIR, "temporal_graph_metrics.csv"))
    if temporal_node_frames:
        cudf.concat(temporal_node_frames).to_parquet(
            os.path.join(RESULT_DIR, "temporal_node_metrics.parquet"))

    # ---- burstiness: CV of monthly strength per node ----
    if strength_history:
        sh = cudf.concat(strength_history)
        g = sh.groupby("node").agg({"strength": ["mean", "std", "count"]})
        g.columns = ["strength_mean", "strength_std", "months_active"]
        g = g.reset_index()
        g["strength_cv"] = (g["strength_std"] / g["strength_mean"]).fillna(0)
        g.to_parquet(os.path.join(RESULT_DIR, "node_burstiness.parquet"))

    # ---- temporally ordered 3-rings across consecutive month triples ----
    if ENABLE_ORDERED_RINGS and len(raw_history) >= 3:
        months = sorted(raw_history)
        rings = []
        for i in range(len(months) - 2):
            tri = ordered_rings(raw_history[months[i]],
                                raw_history[months[i + 1]],
                                raw_history[months[i + 2]],
                                months[i:i + 3])
            if len(tri):
                rings.append(tri)
            log(f"    ordered rings {months[i]}→{months[i+2]}: "
                f"{len(tri) if len(tri) else 0}")
        if rings:
            cudf.concat(rings).to_parquet(
                os.path.join(RESULT_DIR, "ordered_rings.parquet"))

    log("DONE.")


if __name__ == "__main__":
    main()
