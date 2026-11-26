#!/usr/bin/env python
"""
PKG network (F3), step 0c: community robustness test on the GPU server (v2 probe).  Run as a job; output = the log.

Question: can communities be made stable enough to use as features, and with which method?

Graphs with a known (planted) answer, three scenarios:
  A  planted, equal sizes (100) and degrees, mixing 0.33
  B  planted, equal sizes and degrees, mixing 0.50
  C  LFR-like: heavy-tailed community sizes (20 .. 2,000) and degrees (3 .. 1,000), mixing 0.50  <- closest to payments
Methods (each run three times: seed 1, seed 2, seed 3 on a graph with 1% of edges removed):
  GPU  cuGraph Leiden (one run)                      CPU  leidenalg CPM (two resolutions), leidenalg modularity,
  GPU  cuGraph ECG (ensemble of 16, fixed id types)        Infomap (two-level, undirected)
  GPU  Leiden consensus: R runs (new seed + new 1% edge removal each), co-assignment per edge, Leiden on the edges
       co-assigned in >= TAU of runs; two independent ensembles are compared. Node stability = mean co-assignment
       over a node's edges inside its consensus community.
Temporal test (scenarios A and C): slice 2 = slice 1 with 5% of edges replaced from the same model (true communities
unchanged). Ideal: identical memberships in both slices. Compared: independent fits (GPU and CPU), warm start
(slice 2 starts from slice 1's partition), multi-slice leidenalg (interslice weight 0.1 and 1).

Metrics: ARI against the truth, ARI between independent runs (or ensembles), ARI against the 1%-removed graph, number
of communities, share of nodes with stability >= 0.9 (consensus), ARI on that stable core, seconds per run.
Also: tool versions and the shared-folder / hdfs / sbatch check for the F3 data route.
"""
# ---- Settings ------------------------------------------------------------------------------------
OUT_DIR  = "./pkg_gpu_probe_v2_out"
N_NODES  = 300_000
SCENARIOS = {
    "A planted mix 0.33":    {"kind": "planted", "block": 100, "deg": 9, "mu": 1 / 3},
    "B planted mix 0.50":    {"kind": "planted", "block": 100, "deg": 8, "mu": 0.50},
    "C LFR-like mix 0.50":   {"kind": "lfr", "size_min": 20, "size_max": 2000, "size_alpha": 1.0,
                              "deg_min": 3, "deg_max": 1000, "deg_alpha": 1.2, "mu": 0.50},
}
DROP      = 0.01          # share of edges removed for the perturbation runs
ENSEMBLE  = 50            # runs per consensus ensemble (two ensembles per scenario)
TAU       = 0.5           # consensus graph keeps edges co-assigned in at least this share of runs
CPM_GAMMAS = [0.005, 0.02]
RUN_CPU   = True          # leidenalg / Infomap
TEMPORAL_SCENARIOS = ["A planted mix 0.33", "C LFR-like mix 0.50"]
TEMPORAL_CHANGE = 0.05
INTERSLICE = [0.1, 1.0]
SEED = 17

# ---- Imports and helpers -------------------------------------------------------------------------
import os, sys, time, json, shutil, socket, platform, subprocess, traceback, importlib
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()
RESULTS, TEMPORAL = [], []

def log(*a):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)
def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)
def show(df, title=None):
    if title:
        print(f"\n--- {title}", flush=True)
    print(df.to_string(index=False) if len(df) else "(empty)", flush=True)
def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return -1, "", str(e)
def section(fn):
    def w(*a, **k):
        t = time.time()
        try:
            return fn(*a, **k)
        except Exception as e:
            log(f"!! {fn.__name__} failed: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
            traceback.print_exc(limit=4)
        finally:
            log(f"{fn.__name__}: {(time.time() - t) / 60:.1f} min")
    return w
def to_host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)

HAVE = {}
for m in ["cugraph", "cudf", "cupy", "igraph", "leidenalg", "infomap", "sklearn", "lightgbm", "torch", "torch_geometric"]:
    try:
        HAVE[m] = importlib.import_module(m)
    except Exception:
        pass
from sklearn.metrics import adjusted_rand_score as ARI
GPU = all(k in HAVE for k in ("cugraph", "cudf", "cupy"))
CPU = RUN_CPU and all(k in HAVE for k in ("igraph", "leidenalg"))

# ---- 1. Environment and data route ---------------------------------------------------------------
@section
def environment():
    header("1. Versions and data route")
    rows = [("host", socket.gethostname()), ("python", f"{platform.python_version()} {sys.executable}"),
            ("home", os.path.expanduser("~")), ("this folder", os.path.abspath(".")),
            ("output folder", os.path.abspath(OUT_DIR))]
    for m in ["cugraph", "cudf", "cupy", "igraph", "leidenalg", "infomap", "sklearn", "lightgbm", "torch", "torch_geometric"]:
        rows.append((m, getattr(HAVE[m], "__version__", "ok") if m in HAVE else "not installed"))
    for b in ["hdfs", "hadoop", "sbatch", "squeue", "java", "kinit"]:
        rows.append((f"which {b}", shutil.which(b) or "not found"))
    c, o, e = run("df -h . | tail -1")
    rows.append(("filesystem of this folder", o))
    with open(os.path.join(OUT_DIR, "written_by_gpu_node.txt"), "w") as f:
        f.write(f"{socket.gethostname()} {dt.datetime.now().isoformat()}\n")
    rows.append(("marker file (check it from the notebook host)", os.path.abspath(os.path.join(OUT_DIR, "written_by_gpu_node.txt"))))
    show(pd.DataFrame(rows, columns=["item", "value"]))

# ---- 2. Graph generators -------------------------------------------------------------------------
def canonical(s, d):
    a, b = np.minimum(s, d), np.maximum(s, d)
    keep = a != b
    e = np.unique(np.stack([a[keep], b[keep]], axis=1), axis=0)
    return e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)

def make_truth(spec, n, rng):
    if spec["kind"] == "planted":
        truth = np.arange(n) // spec["block"]
        deg = np.full(n, spec["deg"])
    else:
        sizes = []
        while sum(sizes) < n:
            sizes.append(int(min(spec["size_max"], round(spec["size_min"] * (1 + rng.pareto(spec["size_alpha"]))))))
        truth = np.repeat(np.arange(len(sizes)), sizes)[:n]
        truth = truth[rng.permutation(n)]
        deg = np.minimum(spec["deg_max"], np.round(spec["deg_min"] * (1 + rng.pareto(spec["deg_alpha"], n)))).astype(int)
    return truth, deg

def sample_edges(truth, deg, mu, rng):
    n = len(truth)
    order = np.argsort(truth, kind="stable")
    sizes = np.bincount(truth)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    k_in = np.round((1 - mu) * deg).astype(int)
    k_out = deg - k_in
    s_in = np.repeat(np.arange(n), k_in)
    c = truth[s_in]
    d_in = order[starts[c] + np.floor(rng.random(s_in.size) * sizes[c]).astype(np.int64)]
    s_out = np.repeat(np.arange(n), k_out)
    d_out = rng.integers(0, n, size=s_out.size)
    return canonical(np.concatenate([s_in, s_out]), np.concatenate([d_in, d_out]))

def labels_full(vertex, part, n):
    """Partition of present vertices -> labels for all n nodes (absent nodes become singletons)."""
    out = np.full(n, -1, dtype=np.int64)
    out[np.asarray(vertex, dtype=np.int64)] = np.asarray(part, dtype=np.int64)
    miss = out < 0
    if miss.any():
        out[miss] = (out.max() + 1) + np.arange(miss.sum())
    return out

def drop(src, dst, seed, share=DROP):
    keep = np.random.default_rng(seed).random(src.size) >= share
    return src[keep], dst[keep]

# ---- 3. Methods ----------------------------------------------------------------------------------
def first(x):
    return x[0] if isinstance(x, tuple) else x

def gpu_graph(src, dst, w=None):
    cp, cudf, cg = HAVE["cupy"], HAVE["cudf"], HAVE["cugraph"]
    df = cudf.DataFrame({"src": cp.asarray(src, dtype=cp.int32), "dst": cp.asarray(dst, dtype=cp.int32),
                         "wt": cp.asarray(np.ones(len(src)) if w is None else w, dtype=cp.float32)})
    G = cg.Graph(directed=False)
    G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="wt")
    return G

def gpu_leiden(src, dst, n, seed, w=None):
    cg = HAVE["cugraph"]
    G = gpu_graph(src, dst, w)
    try:
        r = cg.leiden(G, random_state=int(seed))
    except TypeError:
        r = cg.leiden(G)
    p = first(r).to_pandas()
    return labels_full(p["vertex"].to_numpy(), p["partition"].to_numpy(), n)

def gpu_ecg(src, dst, n, seed):
    """ECG has no seed in this version: relabel nodes with a random permutation, run, map back."""
    cg = HAVE["cugraph"]
    perm = np.random.default_rng(seed).permutation(n)
    inv = np.empty(n, dtype=np.int64); inv[perm] = np.arange(n)
    G = gpu_graph(perm[src], perm[dst])
    p = first(cg.ecg(G)).to_pandas()
    return labels_full(inv[p["vertex"].to_numpy()], p["partition"].to_numpy(), n)

def gpu_consensus(src, dst, n, seed0):
    """R Leiden runs, each with its own seed and its own 1% edge removal; consensus = Leiden on co-assigned edges."""
    co = np.zeros(src.size, dtype=np.float64)
    for r in range(ENSEMBLE):
        k = np.random.default_rng(seed0 + r).random(src.size) >= DROP
        lab = gpu_leiden(src[k], dst[k], n, seed0 + r)
        co += (lab[src] == lab[dst])
    co /= ENSEMBLE
    m = co >= TAU
    cons = gpu_leiden(src[m], dst[m], n, seed0 + 99_999, w=co[m])
    same = cons[src] == cons[dst]
    num = np.bincount(src[same], weights=co[same], minlength=n) + np.bincount(dst[same], weights=co[same], minlength=n)
    den = np.bincount(src[same], minlength=n) + np.bincount(dst[same], minlength=n)
    stab = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    return cons, stab, float((co >= 0.9).mean()), float(((co > 0.1) & (co < 0.9)).mean())

def ig_graph(src, dst, n):
    ig = HAVE["igraph"]
    g = ig.Graph(n=n, edges=np.stack([src, dst], axis=1).tolist())
    g.vs["id"] = list(range(n))
    return g

def cpu_leiden(g, n, seed, gamma=None, initial=None):
    la = HAVE["leidenalg"]
    kw = {"seed": int(seed)}
    if initial is not None:
        kw["initial_membership"] = list(map(int, initial))
    if gamma is None:
        p = la.find_partition(g, la.ModularityVertexPartition, **kw)
    else:
        p = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=gamma, **kw)
    return np.asarray(p.membership, dtype=np.int64)

def cpu_infomap(src, dst, n, seed):
    im = HAVE["infomap"].Infomap(f"--two-level --silent --num-trials 1 --seed {int(seed)}")
    im.add_links(zip(src.tolist(), dst.tolist()))
    im.run()
    mods = im.get_modules()
    v = np.fromiter(mods.keys(), dtype=np.int64, count=len(mods))
    p = np.fromiter(mods.values(), dtype=np.int64, count=len(mods))
    return labels_full(v, p, n)

# ---- 4. Robustness per scenario ------------------------------------------------------------------
def record(scen, method, truth, a, b, c=None, secs=None, extra=None):
    r = {"scenario": scen, "method": method, "communities": int(len(np.unique(a))),
         "ari_truth": round(ARI(truth, a), 4), "ari_run_vs_run": round(ARI(a, b), 4),
         "ari_vs_1pct_removed": round(ARI(a, c), 4) if c is not None else None,
         "seconds_per_run": round(secs, 1) if secs is not None else None}
    r.update(extra or {})
    RESULTS.append(r)
    log(f"{scen} | {method}: " + ", ".join(f"{k} {v}" for k, v in r.items() if k not in ("scenario", "method")))

def three(fn, src, dst):
    t = time.time()
    a = fn(src, dst, 1)
    b = fn(src, dst, 2)
    s2, d2 = drop(src, dst, 3)
    c = fn(s2, d2, 3)
    return a, b, c, (time.time() - t) / 3

GRAPHS = {}
@section
def robustness():
    header("2. Robustness of communities on graphs with a known answer")
    for i, (scen, spec) in enumerate(SCENARIOS.items()):
        rng = np.random.default_rng(SEED + i)
        truth, deg = make_truth(spec, N_NODES, rng)
        src, dst = sample_edges(truth, deg, spec["mu"], rng)
        GRAPHS[scen] = (truth, deg, src, dst, spec)
        mix = float((truth[src] != truth[dst]).mean())
        dg = np.bincount(src, minlength=N_NODES) + np.bincount(dst, minlength=N_NODES)
        log(f"{scen}: {N_NODES:,} nodes, {src.size:,} edges, {len(np.unique(truth)):,} true communities, "
            f"measured mixing {mix:.2f}, degree p50 {np.median(dg):.0f} / p99 {np.percentile(dg, 99):.0f} / max {dg.max():,}")
        n = N_NODES
        if GPU:
            for name, fn in [("GPU Leiden (one run)", lambda s, d, k: gpu_leiden(s, d, n, k)),
                             ("GPU ECG (16)", lambda s, d, k: gpu_ecg(s, d, n, k))]:
                try:
                    a, b, c, secs = three(fn, src, dst); record(scen, name, truth, a, b, c, secs)
                except Exception as e:
                    log(f"{scen} | {name} failed: {type(e).__name__}: {str(e)[:200]}")
            try:
                t = time.time()
                ca, sa, hi_a, mid_a = gpu_consensus(src, dst, n, 1_000)
                cb, sb, _, _ = gpu_consensus(src, dst, n, 2_000)
                secs = (time.time() - t) / 2
                core = (sa >= 0.9) & (sb >= 0.9)
                record(scen, f"GPU Leiden consensus ({ENSEMBLE} runs, tau {TAU})", truth, ca, cb, None, secs,
                       {"edges_coassigned_ge_0.9": round(hi_a, 4), "edges_ambiguous_0.1_0.9": round(mid_a, 4),
                        "nodes_stable_ge_0.9": round(float((sa >= 0.9).mean()), 4),
                        "ari_run_vs_run_stable_core": round(ARI(ca[core], cb[core]), 4) if core.sum() > 1 else None,
                        "ari_truth_stable_core": round(ARI(truth[core], ca[core]), 4) if core.sum() > 1 else None})
                np.save(os.path.join(OUT_DIR, f"stability_{i}.npy"), sa)
            except Exception as e:
                log(f"{scen} | consensus failed: {type(e).__name__}: {str(e)[:200]}"); traceback.print_exc(limit=3)
        if CPU:
            try:
                t = time.time(); g = ig_graph(src, dst, n); log(f"{scen}: igraph built in {time.time() - t:.0f} s")
                g2s, g2d = drop(src, dst, 3); g2 = ig_graph(g2s, g2d, n)
                for gamma in CPM_GAMMAS + [None]:
                    name = f"CPU leidenalg CPM {gamma}" if gamma is not None else "CPU leidenalg modularity"
                    t = time.time()
                    a = cpu_leiden(g, n, 1, gamma); b = cpu_leiden(g, n, 2, gamma); c = cpu_leiden(g2, n, 3, gamma)
                    record(scen, name, truth, a, b, c, (time.time() - t) / 3)
                del g, g2
            except Exception as e:
                log(f"{scen} | leidenalg failed: {type(e).__name__}: {str(e)[:200]}"); traceback.print_exc(limit=3)
            if "infomap" in HAVE:
                try:
                    a, b, c, secs = three(lambda s, d, k: cpu_infomap(s, d, n, k), src, dst)
                    record(scen, "CPU Infomap (two-level)", truth, a, b, c, secs)
                except Exception as e:
                    log(f"{scen} | infomap failed: {type(e).__name__}: {str(e)[:200]}")
    R = pd.DataFrame(RESULTS); show(R); R.to_csv(os.path.join(OUT_DIR, "robustness.csv"), index=False)

# ---- 5. Temporal: same communities, 5% of edges replaced ------------------------------------------
@section
def temporal():
    header("3. Temporal test: slice 2 = slice 1 with 5% of edges replaced (true communities unchanged)")
    for scen in TEMPORAL_SCENARIOS:
        if scen not in GRAPHS:
            continue
        truth, deg, s1, d1, spec = GRAPHS[scen]
        n = N_NODES
        rng = np.random.default_rng(SEED + 500)
        keep = rng.random(s1.size) >= TEMPORAL_CHANGE
        ns, nd = sample_edges(truth, deg, spec["mu"], rng)
        pick = rng.random(ns.size) < (TEMPORAL_CHANGE * s1.size / max(ns.size, 1))
        s2, d2 = canonical(np.concatenate([s1[keep], ns[pick]]), np.concatenate([d1[keep], nd[pick]]))
        log(f"{scen}: slice 1 {s1.size:,} edges, slice 2 {s2.size:,} edges")
        def rec(method, m1, m2, secs):
            r = {"scenario": scen, "method": method, "ari_slice1_vs_slice2": round(ARI(m1, m2), 4),
                 "ari_truth_slice2": round(ARI(truth, m2), 4), "communities_slice2": int(len(np.unique(m2))),
                 "seconds": round(secs, 1)}
            TEMPORAL.append(r); log(f"{scen} | {method}: {r}")
        if GPU:
            t = time.time(); m1 = gpu_leiden(s1, d1, n, 1); m2 = gpu_leiden(s2, d2, n, 1); rec("GPU Leiden, independent, same seed", m1, m2, time.time() - t)
            t = time.time(); m2b = gpu_leiden(s2, d2, n, 2); rec("GPU Leiden, independent, other seed", m1, m2b, time.time() - t)
            try:
                t = time.time()
                c1, _, _, _ = gpu_consensus(s1, d1, n, 3_000); c2, _, _, _ = gpu_consensus(s2, d2, n, 4_000)
                rec(f"GPU Leiden consensus per slice ({ENSEMBLE} runs)", c1, c2, time.time() - t)
            except Exception as e:
                log(f"consensus temporal failed: {e}")
        if CPU:
            la = HAVE["leidenalg"]
            g1, g2 = ig_graph(s1, d1, n), ig_graph(s2, d2, n)
            t = time.time(); p1 = cpu_leiden(g1, n, 1); p2 = cpu_leiden(g2, n, 1); rec("CPU leidenalg, independent, same seed", p1, p2, time.time() - t)
            t = time.time(); p2w = cpu_leiden(g2, n, 2, initial=p1); rec("CPU leidenalg, warm start from slice 1", p1, p2w, time.time() - t)
            for w in INTERSLICE:
                try:
                    t = time.time()
                    mem, _ = la.find_partition_temporal([g1, g2], la.ModularityVertexPartition, interslice_weight=w,
                                                        vertex_id_attr="id", seed=1)
                    rec(f"CPU leidenalg multi-slice, interslice weight {w}", np.asarray(mem[0]), np.asarray(mem[1]), time.time() - t)
                except Exception as e:
                    log(f"multi-slice {w} failed: {type(e).__name__}: {str(e)[:200]}")
            del g1, g2
    T = pd.DataFrame(TEMPORAL); show(T); T.to_csv(os.path.join(OUT_DIR, "temporal.csv"), index=False)

if __name__ == "__main__":
    log(f"pkg_gpu_probe_v2 start on {socket.gethostname()} · GPU methods {'on' if GPU else 'off'} · CPU methods {'on' if CPU else 'off'}")
    environment()
    robustness()
    temporal()
    header(f"SUMMARY · total {(time.time() - T0) / 60:.1f} min · files in {os.path.abspath(OUT_DIR)}")
    if RESULTS:
        show(pd.DataFrame(RESULTS), "robustness")
    if TEMPORAL:
        show(pd.DataFrame(TEMPORAL), "temporal")
