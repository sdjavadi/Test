#!/usr/bin/env python
"""
PKG network (F3), step 0b: GPU server probe.  Run as a job; everything goes to the log (stdout).

What it checks, in order (each part runs even if an earlier one fails):
  1. Kerberos ticket (kinit with the keytab, klist)
  2. Host: GPU (name, memory, driver), CPUs, RAM, disk
  3. Python packages (RAPIDS, torch / PyG, NetworKit, pyarrow ...) and which cuGraph functions exist
  4. Spark from this server (same settings as the notebooks): connect, read the edge table, write a sample to HDFS
  5. HDFS -> GPU plumbing: read that sample with pyarrow, with the hdfs CLI + cudf, cudf directly; write a result
     back to HDFS and read it with Spark (the round trip the F3 global step needs)
  6. cuGraph timing on random graphs (sizes of our views; runs last, largest graph last): PageRank, components, Louvain, Leiden, ECG, core number,
     triangles, sampled betweenness, Jaccard, random walks; seconds and peak GPU memory per algorithm
  7. Community stability on planted-partition graphs (two mixing levels): Louvain / Leiden / ECG across node
     orderings and with 1% of edges removed; ARI against the planted truth and between runs

Settings below. A summary CSV goes to OUT_DIR (local). Work files on HDFS are deleted at the end.
"""
# ---- Settings ------------------------------------------------------------------------------------
PRINCIPAL  = "pk36814@PNCBANK.COM"
KEYTAB     = "~/pk36814.keytab"
PYTHON_EXECUTORS = "/opt/anaconda/miniconda3/envs/default/bin/python"   # Python of the YARN executors
JAVA_HOME  = "/usr/java/latest"
SPARK_HOME = "/opt/cloudera/parcels/SPARK3/lib/spark3/"
SPARK_PY   = "/opt/cloudera/parcels/SPARK3/lib/spark3/python"
SPARK_PY4J = "/opt/cloudera/parcels/SPARK3/lib/spark3/python/lib/py4j-0.10.9.5-src.zip"
QUEUE      = "root.ua.CIB-AMG.spark"
DB         = "bdahd01p_dlcdi1_cdi_tm"
T_EDGE_M   = f"{DB}.pkg_edge_monthly"
OUT_DIR    = "./pkg_gpu_probe_out"

RUN_SPARK        = True
RUN_HDFS         = True           # needs Spark (it writes the sample)
RUN_CUGRAPH      = True
RUN_STABILITY    = True
SAMPLE_ROWS      = 2_000_000      # real edge rows written to HDFS for the transfer test
TIMING_SIZES     = [(1_000_000, 10_000_000), (3_000_000, 40_000_000), (10_000_000, 100_000_000)]  # nodes, edges
BETWEENNESS_K    = 1000           # sampled sources
STAB_NODES       = 1_000_000
STAB_BLOCK       = 100            # planted community size
STAB_MIXES       = [(6, 3), (4, 4)]   # (edges per node inside its block, edges per node to anywhere)
STAB_RUNS        = 3              # node orderings, and separately 1%-edge-removal draws
STAB_DROP        = 0.01
SEED             = 17

# ---- Imports and helpers -------------------------------------------------------------------------
import os, sys, time, json, glob, shutil, socket, platform, subprocess, threading, traceback, importlib, inspect
import datetime as dt
import numpy as np
import pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)
os.makedirs(OUT_DIR, exist_ok=True)
T0 = time.time()
SUMMARY = []                      # (section, item, value)

def log(*a):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)
def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)
def show(df, title=None):
    if title:
        print(f"\n--- {title}", flush=True)
    print(df.to_string(index=False) if len(df) else "(empty)", flush=True)
def note(section, item, value):
    SUMMARY.append({"section": section, "item": item, "value": str(value)})
def run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return -1, "", str(e)
def section(fn):
    """Run one part; never let a failure stop the others."""
    def w(*a, **k):
        t = time.time()
        try:
            return fn(*a, **k)
        except Exception as e:
            log(f"!! {fn.__name__} failed: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
            traceback.print_exc(limit=3)
            note(fn.__name__, "status", f"failed: {type(e).__name__}")
        finally:
            log(f"{fn.__name__}: {(time.time() - t) / 60:.1f} min")
    return w

def hdfs_path(p):
    """hdfs://nameservice/a/b -> /a/b (pyarrow wants the path without scheme and authority)."""
    return "/" + p.split("/", 3)[3] if p.startswith("hdfs://") else p

# GPU memory peak: poll nvidia-smi in the background
class GpuPeak:
    def __init__(self, every=0.5):
        self.every, self.peak, self._stop = every, 0.0, threading.Event()
    def _used(self):
        c, o, _ = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=10)
        return max((float(x) for x in o.split()), default=0.0) / 1024 if c == 0 and o else 0.0
    def __enter__(self):
        self.base = self._used(); self.peak = self.base
        def loop():
            while not self._stop.wait(self.every):
                self.peak = max(self.peak, self._used())
        self._t = threading.Thread(target=loop, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=5); self.peak = max(self.peak, self._used())

# ---- 1. Kerberos ---------------------------------------------------------------------------------
@section
def kerberos():
    header("1. Kerberos")
    c, o, e = run(["/bin/kinit", PRINCIPAL, "-k", "-t", os.path.expanduser(KEYTAB)])
    log(f"kinit exit {c}" + (f": {e[:200]}" if c else ""))
    c, o, e = run(["/bin/klist"])
    print(o or e, flush=True)
    note("kerberos", "kinit", "ok" if c == 0 else f"exit {c}")

# ---- 2. Host -------------------------------------------------------------------------------------
@section
def host():
    header("2. Host")
    rows = [("host", socket.gethostname()), ("python", f"{platform.python_version()} at {sys.executable}"),
            ("CPUs", os.cpu_count())]
    try:
        mi = {l.split(":")[0]: int(l.split(":")[1].split()[0]) / 1024 / 1024 for l in open("/proc/meminfo")}
        rows.append(("RAM total / available (GB)", f"{mi['MemTotal']:.1f} / {mi['MemAvailable']:.1f}"))
    except Exception as e:
        rows.append(("RAM", f"unreadable ({e})"))
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 2 ** 60:
                rows.append(("container memory limit (GB)", f"{int(v) / 1024 ** 3:.1f}"))
            break
        except Exception:
            pass
    for d in (".", "/tmp"):
        try:
            u = shutil.disk_usage(d); rows.append((f"disk free {d} (GB)", f"{u.free / 1e9:,.0f} of {u.total / 1e9:,.0f}"))
        except Exception:
            pass
    c, o, e = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
                   "--format=csv,noheader"])
    rows.append(("GPUs (index, name, memory total, used, driver, compute capability)", o.replace("\n", " | ") if c == 0 else f"nvidia-smi: {e[:120]}"))
    c, o, e = run("nvidia-smi | grep -i 'CUDA Version'")
    rows.append(("CUDA (driver)", o.strip() if o else "—"))
    for k in ("CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "HADOOP_CONF_DIR", "JAVA_HOME"):
        rows.append((f"env {k}", os.environ.get(k, "—")))
    df = pd.DataFrame(rows, columns=["item", "value"]); show(df)
    for r in rows:
        note("host", r[0], r[1])

# ---- 3. Packages and cuGraph functions -----------------------------------------------------------
HAVE = {}
@section
def packages():
    header("3. Packages")
    rows = []
    for m in ["cugraph", "pylibcugraph", "cudf", "cupy", "rmm", "dask", "dask_cudf", "dask_cuda", "cuml", "pynvml",
              "torch", "torch_geometric", "cugraph_pyg", "dgl", "networkit", "networkx", "igraph", "leidenalg",
              "pyarrow", "fsspec", "pandas", "numpy", "scipy", "sklearn", "lightgbm", "pyspark"]:
        try:
            mod = importlib.import_module(m); v = getattr(mod, "__version__", "ok"); HAVE[m] = mod
            rows.append((m, "ok", v))
        except Exception as e:
            rows.append((m, "no", str(e).splitlines()[0][:90] if str(e) else type(e).__name__))
    if "torch" in HAVE:
        t = HAVE["torch"]
        rows.append(("torch.cuda", "ok" if t.cuda.is_available() else "no", f"devices {t.cuda.device_count()}, CUDA {t.version.cuda}"))
    show(pd.DataFrame(rows, columns=["package", "status", "version / error"]))
    for r in rows:
        note("packages", r[0], f"{r[1]} {r[2]}")
    if "cugraph" in HAVE:
        cg = HAVE["cugraph"]
        fns = ["pagerank", "hits", "katz_centrality", "eigenvector_centrality", "betweenness_centrality",
               "edge_betweenness_centrality", "degree_centrality", "core_number", "k_core", "k_truss", "ktruss_subgraph",
               "triangle_count", "weakly_connected_components", "strongly_connected_components", "louvain", "leiden",
               "ecg", "spectralModularityMaximizationClustering", "spectralBalancedCutClustering", "analyzeClustering_modularity",
               "jaccard", "overlap", "sorensen", "node2vec", "random_walks", "uniform_random_walks", "biased_random_walks",
               "ego_graph", "bfs", "sssp", "induced_subgraph", "minimum_spanning_tree"]
        fr = []
        for f in fns:
            ok = hasattr(cg, f)
            sig = ""
            if ok:
                try:
                    sig = str(inspect.signature(getattr(cg, f)))[:140]
                except Exception:
                    sig = "?"
            fr.append((f, "yes" if ok else "no", sig))
        show(pd.DataFrame(fr, columns=["cugraph function", "available", "signature"]), "cuGraph functions")
        for r in fr:
            note("cugraph_functions", r[0], r[1])

# ---- 4. Spark from this server -------------------------------------------------------------------
spark = None
DB_LOC = None
WORK = None
@section
def spark_connect():
    global spark, DB_LOC, WORK
    header("4. Spark from this server")
    if not os.path.isdir(SPARK_HOME):
        log(f"SPARK_HOME {SPARK_HOME} not found on this server: Spark from here is not possible as configured")
        note("spark", "status", "SPARK_HOME missing"); return
    os.environ["PYSPARK_PYTHON"] = PYTHON_EXECUTORS
    os.environ["JAVA_HOME"] = JAVA_HOME
    os.environ["SPARK_HOME"] = SPARK_HOME
    sys.path.insert(0, SPARK_PY); sys.path.insert(0, SPARK_PY4J)
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
    config = {
        "spark.submit.deployMode": "client",
        "spark.yarn.am.memory": "4g", "spark.executor.memory": "4g", "spark.driver.memory": "4g",
        "spark.yarn.queue": QUEUE,
        "spark.yarn.am.memoryOverhead": "1g", "spark.executor.memoryOverhead": "1g", "spark.executor.cores": "1",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer", "spark.kryoserializer.buffer.max": "128m",
        "spark.files.overwrite": "true", "spark.files.maxPartitionBytes": "134217728",
        "spark.io.compression.codec": "snappy", "spark.io.compression.snappy.blockSize": "32k",
        "spark.dynamicAllocation.enabled": "true", "spark.dynamicAllocation.maxExecutors": "512",
        "spark.pyspark.driver.python": sys.executable,       # this server's Python (the GPU environment)
        "spark.pyspark.python": PYTHON_EXECUTORS,            # the executors' Python (unchanged)
        "spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT": "1", "spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT": "1",
        "spark.sql.execution.arrow.enabled": "false",
        "spark.hadoop.dfs.replication": "2", "spark.ui.showConsoleProgress": "false",
    }
    conf = SparkConf().setAll(list(config.items()))
    t = time.time()
    spark = SparkSession.builder.appName("pkg_gpu_probe").config(conf=conf).enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark {spark.version} session in {time.time() - t:.0f} s, application {spark.sparkContext.applicationId}")
    note("spark", "session", f"ok, Spark {spark.version}, {time.time() - t:.0f} s")
    DB_LOC = [r[1] for r in spark.sql(f"DESCRIBE DATABASE {DB}").collect() if r[0].lower() == "location"][0].rstrip("/")
    WORK = f"{DB_LOC}/_tmp_pkg_gpu_probe"
    log(f"database location {DB_LOC}")
    parts = sorted(r[0].split("=", 1)[1] for r in spark.sql(f"SHOW PARTITIONS {T_EDGE_M}").collect())
    last = parts[-1]
    t = time.time()
    s = (spark.table(T_EDGE_M).where(f"month = '{last}'").select("src_id", "dst_id", "src_kind", "dst_kind", "amount")
           .limit(SAMPLE_ROWS))
    s.write.mode("overwrite").parquet(f"{WORK}/sample")
    n = spark.read.parquet(f"{WORK}/sample").count()
    log(f"wrote {n:,} real edge rows of {last} to {WORK}/sample in {time.time() - t:.0f} s")
    note("spark", "write sample to HDFS", f"{n:,} rows, {time.time() - t:.0f} s")
    # the same rows with dense integer ids (what the F3 export will look like)
    from pyspark.sql import functions as F
    t = time.time()
    ids = (s.select(F.col("src_id").alias("id")).union(s.select(F.col("dst_id").alias("id"))).distinct()
             .withColumn("nid", F.monotonically_increasing_id()))
    e = (s.join(ids.withColumnRenamed("id", "src_id").withColumnRenamed("nid", "src"), "src_id")
          .join(ids.withColumnRenamed("id", "dst_id").withColumnRenamed("nid", "dst"), "dst_id")
          .select("src", "dst", F.col("amount").cast("double").alias("w")))
    e.write.mode("overwrite").parquet(f"{WORK}/sample_int")
    log(f"integer-id version written in {time.time() - t:.0f} s")

# ---- 5. HDFS <-> GPU plumbing --------------------------------------------------------------------
@section
def hdfs_plumbing():
    header("5. HDFS <-> GPU plumbing")
    if spark is None or WORK is None:
        log("no Spark session: skipped"); note("hdfs", "status", "skipped (no Spark)"); return
    rows = []
    path = f"{WORK}/sample_int"
    # a) hdfs CLI -> local disk -> cudf / pandas
    local = os.path.abspath(os.path.join(OUT_DIR, "sample_int"))
    shutil.rmtree(local, ignore_errors=True)
    t = time.time(); c, o, e = run(["hdfs", "dfs", "-get", path, local], timeout=900)
    if c == 0:
        mb = sum(os.path.getsize(f) for f in glob.glob(f"{local}/*.parquet")) / 1e6
        rows.append(("hdfs dfs -get to local disk", "ok", f"{mb:,.0f} MB in {time.time() - t:.1f} s"))
        if "cudf" in HAVE:
            t = time.time(); g = HAVE["cudf"].read_parquet(local)
            rows.append(("cudf.read_parquet(local)", "ok", f"{len(g):,} rows in {time.time() - t:.1f} s")); del g
        else:
            t = time.time(); g = pd.read_parquet(local)
            rows.append(("pandas.read_parquet(local)", "ok", f"{len(g):,} rows in {time.time() - t:.1f} s")); del g
    else:
        rows.append(("hdfs dfs -get", "failed", (e or o)[:150]))
    # b) pyarrow HadoopFileSystem (needs libhdfs and the Hadoop CLASSPATH)
    try:
        import pyarrow.fs as pafs, pyarrow.parquet as pq
        if "CLASSPATH" not in os.environ:
            c, o, e = run(["hadoop", "classpath", "--glob"])
            if c == 0:
                os.environ["CLASSPATH"] = o
        if "ARROW_LIBHDFS_DIR" not in os.environ:
            for cand in glob.glob("/opt/cloudera/parcels/CDH/lib64/libhdfs.so*") + glob.glob("/usr/lib/hadoop/lib/native/libhdfs.so*"):
                os.environ["ARROW_LIBHDFS_DIR"] = os.path.dirname(cand); break
        t = time.time(); fs = pafs.HadoopFileSystem("default")
        tb = pq.read_table(hdfs_path(path), filesystem=fs)
        rows.append(("pyarrow HadoopFileSystem read", "ok", f"{tb.num_rows:,} rows in {time.time() - t:.1f} s"))
        if "cudf" in HAVE:
            t = time.time(); g = HAVE["cudf"].DataFrame.from_arrow(tb)
            rows.append(("arrow table -> cudf", "ok", f"{len(g):,} rows in {time.time() - t:.1f} s")); del g
        # write a result back to HDFS with pyarrow
        out_tb = tb.slice(0, min(tb.num_rows, 500_000))
        dst = f"{WORK}/from_gpu_pyarrow"
        dst_p = hdfs_path(dst)
        fs.create_dir(dst_p)
        t = time.time(); pq.write_table(out_tb, f"{dst_p}/part-0.parquet", filesystem=fs)
        n = spark.read.parquet(dst).count()
        rows.append(("pyarrow write to HDFS, read by Spark", "ok", f"{n:,} rows, {time.time() - t:.1f} s"))
        del tb
    except Exception as ex:
        rows.append(("pyarrow HadoopFileSystem", "failed", f"{type(ex).__name__}: {str(ex).splitlines()[0][:150] if str(ex) else ''}"))
    # c) cudf straight from hdfs:// (fsspec)
    if "cudf" in HAVE:
        try:
            t = time.time(); g = HAVE["cudf"].read_parquet(path)
            rows.append(("cudf.read_parquet(hdfs://...)", "ok", f"{len(g):,} rows in {time.time() - t:.1f} s")); del g
        except Exception as ex:
            rows.append(("cudf.read_parquet(hdfs://...)", "failed", f"{type(ex).__name__}: {str(ex).splitlines()[0][:150] if str(ex) else ''}"))
    # d) local result -> hdfs CLI put -> Spark
    try:
        res = os.path.abspath(os.path.join(OUT_DIR, "result_local"))
        shutil.rmtree(res, ignore_errors=True); os.makedirs(res)
        pd.DataFrame({"node": np.arange(1_000_000), "score": np.random.default_rng(SEED).random(1_000_000)}).to_parquet(f"{res}/part-0.parquet")
        t = time.time(); c, o, e = run(["hdfs", "dfs", "-put", "-f", res, f"{WORK}/from_gpu_cli"], timeout=600)
        if c == 0:
            n = spark.read.parquet(f"{WORK}/from_gpu_cli").count()
            rows.append(("hdfs dfs -put result, read by Spark", "ok", f"{n:,} rows, {time.time() - t:.1f} s"))
        else:
            rows.append(("hdfs dfs -put", "failed", (e or o)[:150]))
    except Exception as ex:
        rows.append(("hdfs dfs -put", "failed", str(ex)[:150]))
    show(pd.DataFrame(rows, columns=["route", "status", "detail"]))
    for r in rows:
        note("hdfs", r[0], f"{r[1]}: {r[2]}")

# ---- 6. cuGraph timing ---------------------------------------------------------------------------
def gpu_edges(n, m, rng):
    """Random undirected edge list on the GPU: no self-loops, no duplicates, weight 1."""
    cp, cudf = HAVE["cupy"], HAVE["cudf"]
    s = cp.asarray(rng.integers(0, n, size=int(m * 1.05), dtype=np.int32))
    d = cp.asarray(rng.integers(0, n, size=int(m * 1.05), dtype=np.int32))
    a, b = cp.minimum(s, d), cp.maximum(s, d)
    df = cudf.DataFrame({"src": a, "dst": b})
    df = df[df.src != df.dst].drop_duplicates().head(m).reset_index(drop=True)
    df["wt"] = cp.float32(1.0)
    return df

def build(df, directed=False, weighted=True):
    cg = HAVE["cugraph"]
    G = cg.Graph(directed=directed)
    if weighted:
        G.from_cudf_edgelist(df, source="src", destination="dst", edge_attr="wt")
    else:
        G.from_cudf_edgelist(df[["src", "dst"]], source="src", destination="dst")
    return G

def timed(name, size, fn, rows):
    with GpuPeak() as gp:
        t = time.time()
        try:
            r = fn(); ok, info = "ok", r if isinstance(r, str) else ""
        except Exception as e:
            ok, info = "failed", f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}"
        sec_ = time.time() - t
    rows.append({"graph": size, "algorithm": name, "status": ok, "seconds": round(sec_, 1),
                 "gpu_peak_gb": round(gp.peak, 1), "gpu_base_gb": round(gp.base, 1), "info": info})
    log(f"{size} {name}: {ok} {sec_:.1f} s, peak {gp.peak:.1f} GB {info}")

def first(x):
    return x[0] if isinstance(x, tuple) else x

@section
def cugraph_timing():
    header("6. cuGraph timing on random graphs")
    if not all(k in HAVE for k in ("cugraph", "cudf", "cupy")):
        log("cugraph / cudf / cupy missing: skipped"); note("cugraph_timing", "status", "skipped"); return
    cg, cp = HAVE["cugraph"], HAVE["cupy"]
    rng = np.random.default_rng(SEED)
    rows = []
    for n, m in TIMING_SIZES:
        size = f"{n / 1e6:g}M nodes / {m / 1e6:g}M edges"
        try:
            with GpuPeak() as gp:
                t = time.time(); df = gpu_edges(n, m, rng); G = build(df); tb = time.time() - t
            rows.append({"graph": size, "algorithm": "build undirected graph", "status": "ok", "seconds": round(tb, 1),
                         "gpu_peak_gb": round(gp.peak, 1), "gpu_base_gb": round(gp.base, 1), "info": f"{len(df):,} edges"})
            log(f"{size} built in {tb:.1f} s")
        except Exception as e:
            rows.append({"graph": size, "algorithm": "build undirected graph", "status": "failed", "seconds": None,
                         "gpu_peak_gb": None, "gpu_base_gb": None, "info": str(e).splitlines()[0][:110]})
            log(f"{size}: build failed ({e}); larger sizes skipped"); break
        timed("pagerank", size, lambda: f"sum {float(cg.pagerank(G)['pagerank'].sum()):.3f}", rows)
        timed("weakly_connected_components", size, lambda: f"{int(first(cg.weakly_connected_components(G))['labels'].nunique()):,} components", rows)
        if hasattr(cg, "louvain"):
            timed("louvain", size, lambda: (lambda r: f"{int(r[0]['partition'].nunique()):,} communities, Q {r[1]:.3f}")(cg.louvain(G)), rows)
        if hasattr(cg, "leiden"):
            timed("leiden", size, lambda: (lambda r: f"{int(r[0]['partition'].nunique()):,} communities, Q {r[1]:.3f}")(cg.leiden(G)), rows)
        if hasattr(cg, "ecg") and m <= 40_000_000:
            def _ecg():
                try:
                    r = cg.ecg(G)
                except Exception:
                    r = cg.ecg(build(df, weighted=False))
                p = first(r)
                return f"{int(p['partition'].nunique()):,} communities"
            timed("ecg (ensemble of 16)", size, _ecg, rows)
        if hasattr(cg, "core_number"):
            timed("core_number", size, lambda: f"max core {int(cg.core_number(G)['core_number'].max())}", rows)
        if hasattr(cg, "triangle_count"):
            def _tri():
                r = cg.triangle_count(G)
                tot = int(r["counts"].sum()) // 3 if hasattr(r, "columns") else int(r)
                return f"{tot:,} triangles"
            timed("triangle_count", size, _tri, rows)
        if hasattr(cg, "betweenness_centrality") and m <= 40_000_000:
            timed(f"betweenness (k={BETWEENNESS_K})", size, lambda: f"max {float(cg.betweenness_centrality(G, k=BETWEENNESS_K)['betweenness_centrality'].max()):.2e}", rows)
        if hasattr(cg, "jaccard"):
            def _jac():
                vp = df[["src", "dst"]].head(100_000).rename(columns={"src": "first", "dst": "second"})
                try:
                    r = cg.jaccard(G, vertex_pair=vp)
                except TypeError:
                    r = cg.jaccard(G, vp)
                return f"mean {float(r['jaccard_coeff'].mean()):.4f} over {len(r):,} pairs"
            timed("jaccard (100K pairs)", size, _jac, rows)
        if hasattr(cg, "uniform_random_walks") or hasattr(cg, "random_walks"):
            def _rw():
                starts = HAVE["cudf"].Series(cp.arange(0, min(n, 100_000), dtype=np.int32))
                if hasattr(cg, "uniform_random_walks"):
                    r = cg.uniform_random_walks(G, start_vertices=starts, max_depth=10)
                else:
                    r = cg.random_walks(G, random_walks_type="uniform", start_vertices=starts, max_depth=10)
                return f"walk output {type(r).__name__}"
            timed("random walks (100K starts, depth 10)", size, _rw, rows)
        # directed PageRank (internal view is directed)
        def _dpr():
            Gd = build(df, directed=True)
            return f"sum {float(cg.pagerank(Gd)['pagerank'].sum()):.3f}"
        timed("pagerank (directed graph)", size, _dpr, rows)
        del G, df
        try:
            import gc; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    R = pd.DataFrame(rows); show(R); R.to_csv(f"{OUT_DIR}/cugraph_timing.csv", index=False)
    for r in rows:
        note("cugraph_timing", f"{r['graph']} {r['algorithm']}", f"{r['status']} {r['seconds']} s, peak {r['gpu_peak_gb']} GB")

# ---- 7. Community stability on planted partitions ------------------------------------------------
def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)

def nmi(a, b):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(a, b)

def planted(n, block, k_in, k_out, rng):
    """Planted partition: each node gets k_in edges inside its block and k_out to anywhere. Truth = block id."""
    src_in = np.repeat(np.arange(n, dtype=np.int64), k_in)
    dst_in = (src_in // block) * block + rng.integers(0, block, size=src_in.size)
    src_out = np.repeat(np.arange(n, dtype=np.int64), k_out)
    dst_out = rng.integers(0, n, size=src_out.size)
    s = np.concatenate([src_in, src_out]); d = np.concatenate([dst_in, dst_out])
    a, b = np.minimum(s, d), np.maximum(s, d)
    keep = a != b
    e = pd.DataFrame({"src": a[keep], "dst": b[keep]}).drop_duplicates().reset_index(drop=True)
    mu = float(((e.src // block) != (e.dst // block)).mean())
    return e, np.arange(n) // block, mu

def labels_from(parts, n, inv=None):
    """cuGraph partition frame -> numpy labels by original node id (inv maps permuted id -> original id)."""
    p = parts.to_pandas() if hasattr(parts, "to_pandas") else parts
    v = p["vertex"].to_numpy(); lab = p["partition"].to_numpy()
    if inv is not None:
        v = inv[v]
    out = np.full(n, -1, dtype=np.int64); out[v] = lab
    miss = out < 0
    if miss.any():                         # isolated nodes: own singleton labels
        out[miss] = lab.max() + 1 + np.arange(miss.sum())
    return out

@section
def stability():
    header("7. Community stability on planted partitions")
    if not all(k in HAVE for k in ("cugraph", "cudf", "cupy")):
        log("cugraph / cudf / cupy missing: skipped"); note("stability", "status", "skipped"); return
    try:
        import sklearn  # noqa
    except Exception:
        log("scikit-learn missing: skipped"); return
    cg, cudf = HAVE["cugraph"], HAVE["cudf"]
    algos = [(a, getattr(cg, a)) for a in ("louvain", "leiden", "ecg") if hasattr(cg, a)]
    rows, detail = [], []
    for k_in, k_out in STAB_MIXES:
        rng = np.random.default_rng(SEED + k_in)
        e, truth, mu = planted(STAB_NODES, STAB_BLOCK, k_in, k_out, rng)
        n = STAB_NODES
        scen = f"k_in {k_in}, k_out {k_out} (mixing {mu:.2f})"
        log(f"{scen}: {len(e):,} edges, {n // STAB_BLOCK:,} planted communities")
        for name, fn in algos:
            labs_order, labs_drop, qs, times = [], [], [], []
            def go(edf, inv=None):
                g = cudf.DataFrame({"src": edf.src.to_numpy(), "dst": edf.dst.to_numpy(), "wt": np.ones(len(edf), np.float32)})
                G = build(g)
                t = time.time()
                try:
                    r = fn(G)
                except Exception:
                    r = fn(build(g, weighted=False))
                times.append(time.time() - t)
                if isinstance(r, tuple) and len(r) > 1:
                    qs.append(float(r[1]))
                return labels_from(first(r), n, inv)
            try:
                for i in range(STAB_RUNS):                 # different node orderings
                    perm = np.random.default_rng(SEED + 100 + i).permutation(n)
                    inv = np.empty(n, dtype=np.int64); inv[perm] = np.arange(n)
                    pe = pd.DataFrame({"src": perm[e.src.to_numpy()], "dst": perm[e.dst.to_numpy()]})
                    labs_order.append(go(pe, inv))
                for i in range(STAB_RUNS):                 # 1% of edges removed
                    keep = np.random.default_rng(SEED + 200 + i).random(len(e)) >= STAB_DROP
                    labs_drop.append(go(e[keep]))
            except Exception as ex:
                log(f"{name} failed on {scen}: {ex}")
                rows.append({"scenario": scen, "algorithm": name, "status": f"failed: {str(ex)[:80]}"}); continue
            base = labs_order[0]
            pair_order = [ari(labs_order[i], labs_order[j]) for i in range(len(labs_order)) for j in range(i + 1, len(labs_order))]
            vs_drop = [ari(base, l) for l in labs_drop]
            # per-node stability: share of the other runs in which the node keeps at least half of its base-run partners
            allruns = labs_order[1:] + labs_drop
            node_stab = np.zeros(n)
            bdf = pd.DataFrame({"node": np.arange(n), "c": base})
            for l in allruns:
                t_ = pd.DataFrame({"c": base, "o": l})
                best = t_.groupby(["c", "o"]).size().reset_index(name="k").sort_values("k").drop_duplicates("c", keep="last")
                m_ = bdf.merge(best[["c", "o"]], on="c", how="left")
                node_stab += (l == m_["o"].to_numpy())
            node_stab /= len(allruns)
            rows.append({"scenario": scen, "algorithm": name, "status": "ok",
                         "communities_mean": int(np.mean([len(np.unique(l)) for l in labs_order])),
                         "modularity_mean": round(float(np.mean(qs)), 4) if qs else None,
                         "ari_vs_truth_mean": round(float(np.mean([ari(truth, l) for l in labs_order])), 4),
                         "nmi_vs_truth_mean": round(float(np.mean([nmi(truth, l) for l in labs_order[:1]])), 4),
                         "ari_between_orderings_min": round(float(min(pair_order)), 4) if pair_order else None,
                         "ari_vs_1pct_drop_min": round(float(min(vs_drop)), 4),
                         "nodes_stable_ge_0.9": round(float((node_stab >= 0.9).mean()), 4),
                         "nodes_stable_lt_0.5": round(float((node_stab < 0.5).mean()), 4),
                         "seconds_per_run": round(float(np.mean(times)), 1)})
            log(f"{scen} {name}: done")
    R = pd.DataFrame(rows); show(R); R.to_csv(f"{OUT_DIR}/stability.csv", index=False)
    for r in rows:
        note("stability", f"{r['scenario']} {r['algorithm']}", json.dumps({k: v for k, v in r.items() if k not in ("scenario", "algorithm")}))

# ---- Clean-up and summary ------------------------------------------------------------------------
@section
def cleanup():
    header("Clean-up")
    for p in (os.path.join(OUT_DIR, "sample_int"), os.path.join(OUT_DIR, "result_local")):
        shutil.rmtree(p, ignore_errors=True)
    if spark is not None and WORK:
        jp = spark._jvm.org.apache.hadoop.fs.Path(WORK)
        fs = jp.getFileSystem(spark._jsc.hadoopConfiguration())
        if fs.exists(jp):
            fs.delete(jp, True); log(f"deleted {WORK}")
        spark.stop()

if __name__ == "__main__":
    log(f"pkg_gpu_probe start on {socket.gethostname()}")
    kerberos()
    host()
    packages()
    if RUN_SPARK:
        spark_connect()
    if RUN_HDFS:
        hdfs_plumbing()
    if RUN_STABILITY:          # before the large timing graphs, so a GPU out-of-memory there cannot spoil it
        stability()
    if RUN_CUGRAPH:
        cugraph_timing()
    cleanup()
    S = pd.DataFrame(SUMMARY); S.to_csv(f"{OUT_DIR}/summary.csv", index=False)
    header(f"SUMMARY · total {(time.time() - T0) / 60:.1f} min · files in {os.path.abspath(OUT_DIR)}")
    show(S)
