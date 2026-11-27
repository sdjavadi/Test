#!/usr/bin/env python
"""
PKG network metrics (F3), GPU job.  Run with sbatch (pkg_network_gpu.sh); the log is the output.

Reads the windows that pkg_network_metrics.ipynb exports to NET_DIR/w=<month>/ (marker _EXPORTED) and, in month order,
writes NET_DIR/w=<month>/out/ (marker _GPU_DONE):
  node_metrics.parquet   nid, pr_int, pr_obs, hits_hub, hits_auth, clustering_int, trophic_level, circulation_hops
  fam_metrics.parquet    fid, family_id, wcc_size, core_number, comm, uid, comm_size, stability
  communities.parquet    uid, size, event (continue / split / merge / birth / small / death), parents
  gate.json              ensemble agreement, codelength savings vs a rewired graph, community_ok, trophic incoherence,
                         timings, workers, allocation
Waits for windows still being exported; stops when the notebook writes NET_DIR/_EXPORT_FINISHED and every window is
done, or shortly before the time budget. Resubmit to continue: finished windows are skipped.
Uses cuGraph / cupy when present, otherwise scipy / networkx (same definitions).
"""
# ---- Settings ------------------------------------------------------------------------------------
import os
NET_DIR = os.environ.get("PKG_NET_DIR") or os.path.abspath("../net_work")   # must be the folder the notebook prints in its first section
WEIGHT_VARIANT = "months"        # community weights: "months" | "logusd" | "both" (months × logusd)
WEIGHT_TEST = True               # compare the three variants on the first window this job processes
WEIGHT_TEST_RUNS = 5
MAIN_TRIALS = 3                  # Infomap trials for the main partition (best codelength kept)
ENSEMBLE = 10                    # perturbed runs (new seed + DROP of edges removed each)
DROP = 0.01
ARI_MIN = 0.80                   # community gate: median ARI(main, ensemble run)
SAVINGS_GAP = 0.05               # … and codelength savings above a rewired copy by at least this
STAB_MIN = 0.80                  # node stability below this: community fields not used downstream
MATCH_J = 0.30                   # month-to-month matching: Jaccard to keep a community uid
MIN_COMM = 3                     # communities smaller than this get no uid (event "small")
MAX_WORKERS = 10
MEM_FRACTION = 0.70              # of the Slurm memory limit, for parallel Infomap runs
PR_ALPHA = 0.85
TIME_BUDGET_H = 11.3             # stop starting new windows after this many hours (job limit 12 h)
POLL_S = 120
SEED = 17

# ---- Imports -------------------------------------------------------------------------------------
import sys, time, json, glob, socket, shutil, resource, traceback, warnings, multiprocessing as mp
import datetime as dt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
T0 = time.time()
try:
    import cudf, cupy as cp, cugraph
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cspl
    GPU = True
except Exception:
    GPU = False
import scipy.sparse as ssp
import scipy.sparse.linalg as sspl
from scipy.sparse.csgraph import connected_components
import infomap
from sklearn.metrics import adjusted_rand_score as ARI

def log(*a):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)
def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)

# ---- allocation ----------------------------------------------------------------------------------
def allocation():
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_mb:
        mem_gb = float(mem_mb) / 1024
    else:
        try:
            mem_gb = float([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1024 / 1024
        except Exception:
            mem_gb = 16.0
    return {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", "-"), "cores": cores, "mem_gb": round(mem_gb, 1),
            "gpu": GPU, "infomap": getattr(infomap, "__version__", "?")}
ALLOC = allocation()

def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

# ---- graph helpers -------------------------------------------------------------------------------
def index_edges(src_ids, dst_ids, idx):
    s = idx.get_indexer(src_ids); d = idx.get_indexer(dst_ids)
    ok = (s >= 0) & (d >= 0)
    return s[ok], d[ok], ok

def pagerank(s, d, w, n):
    out = np.full(n, np.nan)
    if len(s) == 0:
        return out
    if GPU:
        g = cudf.DataFrame({"src": s.astype(np.int32), "dst": d.astype(np.int32), "wt": w.astype(np.float32)})
        G = cugraph.Graph(directed=True); G.from_cudf_edgelist(g, "src", "dst", edge_attr="wt", store_transposed=True)
        r = cugraph.pagerank(G, alpha=PR_ALPHA).to_pandas()
        out[r["vertex"].to_numpy()] = r["pagerank"].to_numpy()
        return out
    present = np.unique(np.concatenate([s, d]))
    A = ssp.csr_matrix((w, (s, d)), shape=(n, n))
    outw = np.asarray(A.sum(1)).ravel()
    P = ssp.diags(np.where(outw > 0, 1 / np.maximum(outw, 1e-300), 0)) @ A
    m_ = len(present); x = np.zeros(n); x[present] = 1 / m_
    dang = np.zeros(n, bool); dang[present] = outw[present] == 0
    tele = np.zeros(n); tele[present] = 1 / m_
    for _ in range(200):
        xn = PR_ALPHA * (P.T @ x + x[dang].sum() * tele) + (1 - PR_ALPHA) * tele
        if np.abs(xn - x).sum() < 1e-10:
            x = xn; break
        x = xn
    out[present] = x[present]
    return out

def hits(s, d, n):
    hub = np.full(n, np.nan); auth = np.full(n, np.nan)
    if len(s) == 0:
        return hub, auth
    if GPU:
        g = cudf.DataFrame({"src": s.astype(np.int32), "dst": d.astype(np.int32)})
        G = cugraph.Graph(directed=True); G.from_cudf_edgelist(g, "src", "dst")
        r = cugraph.hits(G).to_pandas()
        hub[r["vertex"].to_numpy()] = r["hubs"].to_numpy(); auth[r["vertex"].to_numpy()] = r["authorities"].to_numpy()
        return hub, auth
    A = ssp.csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n)); A.data[:] = 1
    h = np.ones(n)
    for _ in range(100):
        a = A.T @ h; a /= max(a.sum(), 1e-300)
        hn = A @ a; hn /= max(hn.sum(), 1e-300)
        if np.abs(hn - h).sum() < 1e-10:
            h = hn; break
        h = hn
    present = np.unique(np.concatenate([s, d]))
    hub[present] = h[present]; auth[present] = a[present]
    return hub, auth

def undirected_pairs(s, d):
    a = np.minimum(s, d); b = np.maximum(s, d); k = a != b
    pr = np.unique(np.stack([a[k], b[k]], 1), axis=0)
    return pr[:, 0], pr[:, 1]

def clustering(s, d, n):
    a, b = undirected_pairs(s, d)
    deg = np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
    tri = np.zeros(n)
    if len(a):
        if GPU:
            g = cudf.DataFrame({"src": a.astype(np.int32), "dst": b.astype(np.int32)})
            G = cugraph.Graph(directed=False); G.from_cudf_edgelist(g, "src", "dst")
            r = cugraph.triangle_count(G).to_pandas()
            tri[r["vertex"].to_numpy()] = r["counts"].to_numpy()
        else:
            A = ssp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n)); A = A + A.T
            tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2
    return np.where(deg >= 2, 2 * tri / np.maximum(deg * (deg - 1), 1), np.nan)

def components(s, d, n):
    if len(s) == 0:
        return np.arange(n)
    if GPU:
        g = cudf.DataFrame({"src": s.astype(np.int32), "dst": d.astype(np.int32)})
        G = cugraph.Graph(directed=False); G.from_cudf_edgelist(g, "src", "dst")
        r = cugraph.weakly_connected_components(G)
        r = (r[0] if isinstance(r, tuple) else r).to_pandas()
        lab = np.arange(n) + n                 # isolated nodes: own component
        lab[r["vertex"].to_numpy()] = r["labels"].to_numpy()
        return pd.factorize(lab)[0]
    A = ssp.csr_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    return connected_components(A, directed=False)[1]

def core_number(s, d, n):
    a, b = undirected_pairs(s, d)
    out = np.zeros(n)
    if len(a) == 0:
        return out
    if GPU:
        g = cudf.DataFrame({"src": a.astype(np.int32), "dst": b.astype(np.int32)})
        G = cugraph.Graph(directed=False); G.from_cudf_edgelist(g, "src", "dst")
        r = cugraph.core_number(G).to_pandas()
        out[r["vertex"].to_numpy()] = r["core_number"].to_numpy()
        return out
    import networkx as nx
    G = nx.Graph(); G.add_edges_from(zip(a.tolist(), b.tolist()))
    for k, v in nx.core_number(G).items():
        out[k] = v
    return out

def trophic(s, d, w, n):
    """Improved trophic levels (MacKay, Johnson, Sansom 2020): L h = v, L = diag(u) − W − Wᵀ; min 0 per component."""
    h = np.full(n, np.nan)
    if len(s) == 0:
        return h, np.nan
    W = ssp.csr_matrix((w, (s, d)), shape=(n, n))
    sin = np.asarray(W.sum(0)).ravel(); sout = np.asarray(W.sum(1)).ravel()
    u = sin + sout; v = sin - sout
    L = (ssp.diags(u) - W - W.T).tocsr() + ssp.identity(n) * 1e-9
    Minv = 1 / np.maximum(u, 1e-9)
    if GPU:
        Lg = csp.csr_matrix(L); vg = cp.asarray(v); Mg = csp.diags(cp.asarray(Minv))
        try:
            x, info = cspl.cg(Lg, vg, tol=1e-8, maxiter=3000, M=Mg)
        except TypeError:
            x, info = cspl.cg(Lg, vg, atol=1e-8, maxiter=3000, M=Mg)
        x = cp.asnumpy(x)
    else:
        M = ssp.diags(Minv)
        try:
            x, info = sspl.cg(L, v, rtol=1e-8, maxiter=3000, M=M)
        except TypeError:
            x, info = sspl.cg(L, v, tol=1e-8, maxiter=3000, M=M)
    comp = components(s, d, n)
    x = pd.Series(x); x = (x - x.groupby(comp).transform("min")).to_numpy()
    present = u > 0
    h[present] = x[present]
    F0 = float(np.sum(w * (h[d] - h[s] - 1) ** 2) / np.sum(w)) if np.isfinite(h[s]).all() else np.nan
    return h, F0

def circulation(s, d, usd, out_total, n):
    """Expected on-us hops before a dollar leaves (proportional mixing): h = P (1 + h)."""
    h = np.full(n, np.nan)
    if len(s) == 0:
        return h
    share = usd / np.maximum(out_total[s], 1e-9)
    P = ssp.csr_matrix((np.minimum(share, 1.0), (s, d)), shape=(n, n))
    one = np.asarray(P.sum(1)).ravel()
    x = np.zeros(n)
    for _ in range(300):
        xn = one + P @ x
        if np.max(np.abs(xn - x)) < 1e-6:
            x = xn; break
        x = xn
    has = out_total > 0
    h[has] = np.minimum(x[has], 50)
    return h

# ---- Infomap ensemble ----------------------------------------------------------------------------
_G = {}
def infomap_run(args):
    """One Infomap run on the global graph _G (forked workers share it). drop > 0 removes that share of edges."""
    seed, trials, drop = args
    s, d, w, n = _G["s"], _G["d"], _G["w"], _G["n"]
    if drop > 0:
        k = np.random.default_rng(seed).random(len(s)) >= drop
        s, d, w = s[k], d[k], w[k]
    im = infomap.Infomap(f"--two-level --silent --num-trials {trials} --seed {seed}")
    im.add_links(zip(s.tolist(), d.tolist(), w.tolist()))
    im.run()
    lab = np.full(n, -1, dtype=np.int64)
    mods = im.get_modules()
    if mods:
        k_ = np.fromiter(mods.keys(), dtype=np.int64, count=len(mods)); v_ = np.fromiter(mods.values(), dtype=np.int64, count=len(mods))
        lab[k_] = v_
    miss = lab < 0
    lab[miss] = lab.max() + 1 + np.arange(miss.sum())
    cl = float(im.codelength); one = float(getattr(im, "one_level_codelength", np.nan))
    return lab, cl, one

def communities(s, d, w, n, ensemble, label=""):
    """Main partition (MAIN_TRIALS), ensemble of perturbed runs in parallel, co-assignment, node stability, gate."""
    _G.update(s=s, d=d, w=w, n=n)
    r0 = rss_gb(); t = time.time()
    main, cl, one = infomap_run((SEED, MAIN_TRIALS, 0.0))
    t_main = time.time() - t
    peak = max(rss_gb() - r0, len(s) * 400 / 1e9, 0.5)   # measured growth, or about 400 bytes per edge, whichever is larger
    workers = int(max(1, min(ALLOC["cores"], MAX_WORKERS, ensemble, (ALLOC["mem_gb"] * MEM_FRACTION) // peak)))
    t = time.time()
    args = [(SEED + 1000 + i, 1, DROP) for i in range(ensemble)]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(infomap_run, args)
    else:
        res = [infomap_run(a) for a in args]
    t_ens = time.time() - t
    aris = [ARI(main, r[0]) for r in res]
    co = np.mean([(r[0][s] == r[0][d]) for r in res], axis=0) if res else np.ones(len(s))
    same = main[s] == main[d]
    num = np.bincount(s[same], weights=co[same], minlength=n) + np.bincount(d[same], weights=co[same], minlength=n)
    den = np.bincount(s[same], minlength=n) + np.bincount(d[same], minlength=n)
    stab = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    # rewired copy: same degree sequence, weights shuffled
    rng = np.random.default_rng(SEED)
    stubs = np.concatenate([s, d]); rng.shuffle(stubs)
    rs, rd = stubs[: len(s)], stubs[len(s):]
    k = rs != rd
    _G.update(s=rs[k], d=rd[k], w=rng.permutation(w)[k], n=n)
    _, rcl, rone = infomap_run((SEED + 7, 1, 0.0))
    sav = 1 - cl / one if one and np.isfinite(one) else np.nan
    rsav = 1 - rcl / rone if rone and np.isfinite(rone) else np.nan
    g = {"label": label, "communities": int(len(np.unique(main))), "codelength": cl, "savings": sav, "savings_rewired": rsav,
         "ari_median": float(np.median(aris)) if aris else np.nan, "ari_min": float(np.min(aris)) if aris else np.nan,
         "nodes_stable_share": float(np.nanmean(stab >= STAB_MIN)) if np.isfinite(stab).any() else np.nan,
         "workers": workers, "peak_gb_per_run": round(peak, 2), "minutes_main": round(t_main / 60, 2), "minutes_ensemble": round(t_ens / 60, 2)}
    g["community_ok"] = bool(g["ari_median"] >= ARI_MIN and (g["savings"] - g["savings_rewired"]) >= SAVINGS_GAP)
    return main, stab, g

# ---- matching to last window ---------------------------------------------------------------------
def match(m, fam_ids, lab, prev):
    """prev: DataFrame family_id, uid (last window). Returns uid per family and the communities table."""
    cur = pd.DataFrame({"family_id": fam_ids, "comm": lab})
    size = cur.groupby("comm").size().rename("size")
    cur = cur.join(size, on="comm")
    uid = pd.Series([None] * len(cur), dtype=object)
    rows = []
    big = size[size >= MIN_COMM].index
    if prev is not None and len(prev):
        pv = prev.dropna(subset=["uid"])
        psize = pv.groupby("uid").size().rename("psize")
        inter = cur[cur.comm.isin(big)].merge(pv, on="family_id").groupby(["comm", "uid"]).size().rename("k").reset_index()
        inter = inter.join(size, on="comm").join(psize, on="uid")
        inter["J"] = inter.k / (inter["size"] + inter.psize - inter.k)
        best_c = inter.sort_values("k").drop_duplicates("comm", keep="last").set_index("comm")
        best_p = inter.sort_values("k").drop_duplicates("uid", keep="last").set_index("uid")
    else:
        inter = pd.DataFrame(columns=["comm", "uid", "k", "size", "psize", "J"]); best_c = best_p = inter.set_index("comm")
    cmap = {}
    for c in size.index:
        if c not in big:
            cmap[c] = (None, "small", ""); continue
        par = inter[(inter.comm == c) & (inter.k >= 0.3 * size[c])]["uid"].tolist() if len(inter) else []
        if c in best_c.index:
            u = best_c.loc[c, "uid"]
            mutual = u in best_p.index and best_p.loc[u, "comm"] == c
            if mutual and best_c.loc[c, "J"] >= MATCH_J:
                cmap[c] = (u, "continue", u); continue
            if len(par) >= 2:
                cmap[c] = (f"{m}:{c}", "merge", ",".join(map(str, par))); continue
            if not mutual and best_c.loc[c, "k"] >= 0.3 * best_c.loc[c, "psize"]:
                cmap[c] = (f"{m}:{c}", "split", str(u)); continue
        cmap[c] = (f"{m}:{c}", "birth", ",".join(map(str, par)))
    cur["uid"] = cur.comm.map(lambda c: cmap[c][0])
    comm_tab = pd.DataFrame([{"comm": c, "uid": v[0], "size": int(size[c]), "event": v[1], "parents": v[2]} for c, v in cmap.items()])
    if prev is not None and len(prev):
        alive = set(cur.uid.dropna()); used = set(",".join(comm_tab.parents).split(","))
        dead = [u for u in prev.uid.dropna().unique() if u not in alive and u not in used]
        if dead:
            comm_tab = pd.concat([comm_tab, pd.DataFrame({"comm": -1, "uid": dead, "size": 0, "event": "death", "parents": ""})])
    return cur, comm_tab

# ---- one window ----------------------------------------------------------------------------------
def rp(path):
    return pd.read_parquet(path)

def process(m, prev_dir, weight_test):
    wdir = os.path.join(NET_DIR, f"w={m}")
    odir = os.path.join(wdir, "out"); shutil.rmtree(odir, ignore_errors=True); os.makedirs(odir)
    tm = {}
    t = time.time()
    nodes = rp(f"{wdir}/nodes"); fams = rp(f"{wdir}/fams")
    ie = rp(f"{wdir}/int_edges"); oe = rp(f"{wdir}/obs_edges"); ce = rp(f"{wdir}/com_edges")
    n = len(nodes); idx = pd.Index(nodes.nid.to_numpy())
    nf = len(fams); fidx = pd.Index(fams.fid.to_numpy())
    tm["read"] = time.time() - t
    log(f"{m}: nodes {n:,}, families {nf:,}, internal {len(ie):,}, observed {len(oe):,}, community {len(ce):,} edges")
    t = time.time()
    s_i, d_i, ok_i = index_edges(ie.src.to_numpy(), ie.dst.to_numpy(), idx)
    s_o, d_o, ok_o = index_edges(oe.src.to_numpy(), oe.dst.to_numpy(), idx)
    w_i = ie.wm.to_numpy(float)[ok_i]; u_i = ie.usd.to_numpy(float)[ok_i]
    w_o = oe.wm.to_numpy(float)[ok_o]; u_o = oe.usd.to_numpy(float)[ok_o]
    pr_int = pagerank(s_i, d_i, w_i, n); pr_obs = pagerank(s_o, d_o, w_o, n)
    hub, auth = hits(s_i, d_i, n)
    clus = clustering(s_i, d_i, n)
    tro, F0 = trophic(s_o, d_o, np.log1p(u_o), n)
    circ = circulation(s_i, d_i, u_i, nodes.out_usd.to_numpy(float), n)
    tm["node_metrics"] = time.time() - t
    pd.DataFrame({"nid": nodes.nid.to_numpy(), "pr_int": pr_int, "pr_obs": pr_obs, "hits_hub": hub, "hits_auth": auth,
                  "clustering_int": clus, "trophic_level": tro, "circulation_hops": circ}).to_parquet(f"{odir}/node_metrics.parquet", index=False)
    log(f"{m}: node metrics in {tm['node_metrics'] / 60:.1f} min · trophic incoherence F0 {F0:.3f}")
    t = time.time()
    s_c, d_c, ok_c = index_edges(ce.src.to_numpy(), ce.dst.to_numpy(), fidx)
    wm_c = ce.w_months.to_numpy(float)[ok_c]; wl_c = ce.w_logusd.to_numpy(float)[ok_c]
    wv = {"months": wm_c, "logusd": wl_c, "both": wm_c * wl_c}
    comp = components(s_c, d_c, nf); wcc = pd.Series(comp).map(pd.Series(comp).value_counts()).to_numpy()
    core = core_number(s_c, d_c, nf)
    tests = []
    if weight_test and len(s_c):
        for v in ("months", "logusd", "both"):
            _, _, gv = communities(s_c, d_c, wv[v], nf, WEIGHT_TEST_RUNS, label=f"weight test: {v}")
            tests.append(gv); log(f"{m}: {gv}")
    if len(s_c):
        lab, stab, gate = communities(s_c, d_c, wv[WEIGHT_VARIANT], nf, ENSEMBLE, label=f"main: {WEIGHT_VARIANT}")
    else:
        lab, stab, gate = np.arange(nf), np.full(nf, np.nan), {"community_ok": False, "label": "no edges"}
    tm["communities"] = time.time() - t
    prev = pd.read_parquet(f"{prev_dir}/out/fam_metrics.parquet", columns=["family_id", "uid"]) if prev_dir else None
    cur, ctab = match(m, fams.family_id.to_numpy(), lab, prev)
    fm = pd.DataFrame({"fid": fams.fid.to_numpy(), "family_id": fams.family_id.to_numpy(), "wcc_size": wcc, "core_number": core,
                       "comm": lab, "uid": cur.uid.to_numpy(), "comm_size": cur["size"].to_numpy(), "stability": stab})
    fm.to_parquet(f"{odir}/fam_metrics.parquet", index=False)
    ctab.to_parquet(f"{odir}/communities.parquet", index=False)
    ev = ctab.event.value_counts().to_dict()
    gate.update({"window": m, "trophic_F0": F0, "events": ev, "weight_test": tests, "allocation": ALLOC,
                 "minutes": {k: round(v / 60, 2) for k, v in tm.items()}})
    json.dump(gate, open(f"{odir}/gate.json", "w"), default=float, indent=1)
    open(os.path.join(wdir, "_GPU_DONE"), "w").write(dt.datetime.now().isoformat(timespec="seconds"))
    log(f"{m}: communities {gate.get('communities')} · ARI median {gate.get('ari_median', float('nan')):.3f} · "
        f"savings {gate.get('savings', float('nan')):.3f} vs rewired {gate.get('savings_rewired', float('nan')):.3f} · "
        f"ok {gate['community_ok']} · events {ev} · {sum(tm.values()) / 60:.1f} min")
    return gate

# ---- main loop -----------------------------------------------------------------------------------
if __name__ == "__main__":
    header(f"PKG network GPU job · {ALLOC}")
    log(f"NET_DIR {NET_DIR}")
    rows, tested = [], False
    while True:
        wins = sorted(x.split("=", 1)[1] for x in os.listdir(NET_DIR) if x.startswith("w=")) if os.path.isdir(NET_DIR) else []
        todo = [w for w in wins if not os.path.exists(os.path.join(NET_DIR, f"w={w}", "_GPU_DONE"))]
        ready = [w for w in todo if os.path.exists(os.path.join(NET_DIR, f"w={w}", "_EXPORTED"))]
        finished = os.path.exists(os.path.join(NET_DIR, "_EXPORT_FINISHED"))
        if (time.time() - T0) / 3600 > TIME_BUDGET_H:
            log("time budget reached: resubmit to continue"); break
        if not todo and finished:
            log("every exported window is done"); break
        # process in month order: the next window must be exported before anything later
        if todo and todo[0] in ready:
            w = todo[0]
            prev = [x for x in wins if x < w]
            prev_dir = os.path.join(NET_DIR, f"w={prev[-1]}") if prev and os.path.exists(os.path.join(NET_DIR, f"w={prev[-1]}", "_GPU_DONE")) else None
            header(f"window {w}" + (f" (matched to {prev[-1]})" if prev_dir else " (no previous window)"))
            try:
                g = process(w, prev_dir, WEIGHT_TEST and not tested)
                tested = True
                rows.append({k: g.get(k) for k in ("window", "communities", "ari_median", "savings", "savings_rewired",
                                                   "nodes_stable_share", "community_ok", "trophic_F0")})
            except Exception as e:
                log(f"!! window {w} failed: {type(e).__name__}: {str(e)[:300]}"); traceback.print_exc(limit=5)
                break
            continue
        log(f"waiting for exports ({len(ready)} ready of {len(todo)} pending; export finished: {finished})")
        time.sleep(POLL_S)
    header(f"SUMMARY · {(time.time() - T0) / 60:.1f} min")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
