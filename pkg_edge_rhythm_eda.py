# %% [markdown]
# # PKG Edge Rhythm — EDA & Calibration (spec v0.2)
#
# Retrospective full-window pass over 23 monthly edge snapshots
# (Jan 2024 = month 0 ... Nov 2025 = month 22).
#
# Purpose: calibrate the edge periodicity classifier and decide backbone viability.
# **Never ship these labels as historical facts** — the pipeline module will be
# as-of-month with a 6-month minimum lookback.
#
# Locked decisions (v0.2):
# - Modal amount band chosen EMPIRICALLY from the rel-dev distribution (Section 4).
#   `BAND = None` until inspected; band grid computed regardless for sensitivity.
# - `n_distinct_amt` uses nearest-dollar rounding.
# - `MIN_HISTORY = 6` months for classification eligibility.
# - Hub check reuses the existing 23-snapshot aggregate ablation node lists (P99 / P99.9).
# - Compute: cuDF on the GPU server.
#
# Known-pitfall guards baked in (see inline comments):
# - no iterrows anywhere; run-length encoding via global diff + boundary mask
# - no groupby.nlargest — global sort + drop_duplicates instead
# - np ufunc dispatch (np.log1p(series)) where logs are needed, never Series.log1p()

# %% Environment check
import json
from pathlib import Path

import numpy as np
import cupy as cp
import cudf

print("cudf", cudf.__version__)
free_b, total_b = cp.cuda.runtime.memGetInfo()
print(f"GPU mem: {free_b/1e9:.1f} GB free / {total_b/1e9:.1f} GB total")

# %% Parameters
DATA_DIR = Path("/path/to/monthly_edge_csvs")  # TODO: point at the 23 snapshot CSVs

# CRITICAL: chronological order must hold. If filenames sort chronologically
# (e.g. pkg_edges_202401.csv ... pkg_edges_202511.csv) the glob is enough;
# otherwise replace with an explicit ordered list.
MONTH_FILES = sorted(DATA_DIR.glob("*.csv"))
N_MONTHS = 23
LAST_MONTH = N_MONTHS - 1
assert len(MONTH_FILES) == N_MONTHS, f"expected {N_MONTHS} files, found {len(MONTH_FILES)}"
for i, f in enumerate(MONTH_FILES):
    print(f"month_idx {i:2d}  <-  {f.name}")  # eyeball this mapping before proceeding

OUT_DIR = Path("./edge_rhythm_eda")
OUT_DIR.mkdir(exist_ok=True)
STAGE0_PQ = OUT_DIR / "edge_month.parquet"
NODE_DIM_PQ = OUT_DIR / "node_dim.parquet"
META_JSON = OUT_DIR / "stage0_meta.json"
AGG_PQ = OUT_DIR / "edge_rhythm_agg.parquet"

MIN_HISTORY = 6           # locked (decision 3)
BAND = None               # set AFTER inspecting Section 4; fallback 0.05 if left None
BAND_GRID = [0.01, 0.025, 0.05, 0.10]
SUPPORT_GRID = [0.70, 0.80, 0.90]
RECURRING_T = 0.80        # provisional support cutoff for NAICS/backbone views; revisit after Section 7

# Existing ablation node lists (decision 4). Files need one column with the
# ORIGINAL node id (same id space as source/dest in the CSVs).
HUB_LISTS = {
    "P99":   Path("/path/to/ablation_P99_nodes.parquet"),    # TODO
    "P99_9": Path("/path/to/ablation_P99_9_nodes.parquet"),  # TODO
}
HUB_NODE_COL = "node"  # TODO: column name in those files

# %% Plot helpers
import plotly.express as px
import plotly.graph_objects as go

def show(fig, title=None):
    fig.update_layout(template="plotly_dark", font_family="IBM Plex Sans")
    if title:
        fig.update_layout(title=title)
    fig.show()

def flat(df):
    """Flatten cuDF MultiIndex agg columns: ('amount','sum') -> amount_sum."""
    df.columns = [
        "_".join([str(c) for c in col if c != ""]) if isinstance(col, tuple) else col
        for col in df.columns
    ]
    return df

# %% [markdown]
# ## Section 1 — Stage 0: stacked edge-history frame
# Build once, cache. Columns: edge_key int64, month_idx int8, amount float64, volume int32.
# edge_key = src_id * N_NODES + dst_id (recover src/dst by divmod).

# %% Stage 0 build (skipped if cached)
if STAGE0_PQ.exists():
    em = cudf.read_parquet(STAGE0_PQ)
    node_dim = cudf.read_parquet(NODE_DIM_PQ)
    N_NODES = json.loads(META_JSON.read_text())["n_nodes"]
    print(f"loaded cached stage0: {len(em):,} rows, {N_NODES:,} nodes")
else:
    parts, raw_rows = [], 0
    for m_idx, f in enumerate(MONTH_FILES):
        df = cudf.read_csv(
            f,
            usecols=["source", "dest", "amount", "volume"],
            dtype={"source": "str", "dest": "str", "amount": "float64", "volume": "int32"},
        )
        raw_rows += len(df)
        df["month_idx"] = np.int8(m_idx)
        parts.append(df)
        print(f"month {m_idx:2d}: {len(df):,} rows")
    em = cudf.concat(parts, ignore_index=True)
    del parts

    # --- factorize node ids early: never groupby/merge on strings downstream
    nodes = cudf.concat([em["source"], em["dest"]]).drop_duplicates().reset_index(drop=True)
    node_dim = cudf.DataFrame({"node": nodes})
    node_dim["node_id"] = cp.arange(len(node_dim), dtype="int64")
    N_NODES = len(node_dim)
    assert N_NODES < 3_000_000_000, "edge_key int64 packing would overflow"

    em = em.merge(node_dim.rename(columns={"node": "source", "node_id": "src_id"}),
                  on="source", how="left")
    em = em.merge(node_dim.rename(columns={"node": "dest", "node_id": "dst_id"}),
                  on="dest", how="left")
    assert em["src_id"].isnull().sum() == 0 and em["dst_id"].isnull().sum() == 0
    em = em.drop(columns=["source", "dest"])
    em["edge_key"] = em["src_id"] * np.int64(N_NODES) + em["dst_id"]
    em = em.drop(columns=["src_id", "dst_id"])

    # --- defensive dedup: if a (edge, month) pair appears twice, sum it and log
    before = len(em)
    em = em.groupby(["edge_key", "month_idx"], as_index=False).agg(
        {"amount": "sum", "volume": "sum"}
    )
    em["month_idx"] = em["month_idx"].astype("int8")
    em["volume"] = em["volume"].astype("int32")
    dups = before - len(em)
    print(f"dedup: {dups:,} duplicate (edge, month) rows collapsed" if dups else "dedup: clean")

    # --- sanity asserts
    assert (em["amount"] > 0).all(), "non-positive amounts present — investigate before proceeding"
    assert (em["volume"] >= 1).all()
    assert em["month_idx"].min() >= 0 and em["month_idx"].max() <= LAST_MONTH

    em.to_parquet(STAGE0_PQ)
    node_dim.to_parquet(NODE_DIM_PQ)
    META_JSON.write_text(json.dumps({"n_nodes": int(N_NODES), "raw_rows": int(raw_rows)}))
    print(f"stage0 cached: {len(em):,} rows, {N_NODES:,} nodes")

N_EDGES = em["edge_key"].nunique()
print(f"distinct edges across window: {N_EDGES:,}")

# %% [markdown]
# ## Section 2 — Stage 1 core aggregates (plain groupby pass)

# %%
em["is_v1"] = (em["volume"] == 1).astype("int8")
em["amt_r"] = em["amount"].round(0)  # nearest dollar (locked, decision 2)

core = em.groupby("edge_key").agg(
    {
        "month_idx": ["count", "min", "max"],
        "amount": ["sum", "mean", "median", "std", "min", "max"],
        "amt_r": ["nunique"],
        "volume": ["mean", "std"],
        "is_v1": ["mean"],
    }
)
core = flat(core.reset_index())
core = core.rename(
    columns={
        "month_idx_count": "months_present",
        "month_idx_min": "first_month",
        "month_idx_max": "last_month",
        "amount_sum": "amt_total",
        "amount_mean": "amt_mean",
        "amount_median": "amt_median",
        "amount_std": "amt_std",
        "amount_min": "amt_min",
        "amount_max": "amt_max",
        "amt_r_nunique": "n_distinct_amt",
        "volume_mean": "vol_mean",
        "volume_std": "vol_std",
        "is_v1_mean": "share_vol1",
    }
)
core["months_present"] = core["months_present"].astype("int16")
print(f"core agg: {len(core):,} edges")

# %% vol_mode via value-count + global sort (NOT groupby.nlargest — known stall)
vc = (
    em.groupby(["edge_key", "volume"])
    .agg({"month_idx": "count"})
    .reset_index()
    .rename(columns={"month_idx": "cnt"})
)
vc = vc.sort_values(["cnt", "volume"], ascending=[False, True])  # tie-break: smaller volume
vc = vc.drop_duplicates(subset="edge_key", keep="first")
vc = vc.rename(columns={"volume": "vol_mode", "cnt": "vol_mode_cnt"})
core = core.merge(vc[["edge_key", "vol_mode", "vol_mode_cnt"]], on="edge_key", how="left")
core["vol_mode_share"] = core["vol_mode_cnt"] / core["months_present"]
del vc

# %% [markdown]
# ## Section 3 — Window pass: gaps, streaks, phase concentration, halves CV
# Run-length encoding = global diff + edge-boundary mask + cumsum run ids.
# Same pattern as the community-lifecycle vectorization; zero iterrows.

# %%
em = em.sort_values(["edge_key", "month_idx"]).reset_index(drop=True)

ek_change = (em["edge_key"].diff().fillna(1) != 0)          # True at first row of each edge
em["d"] = em["month_idx"].astype("float32").diff().mask(ek_change)  # inter-arrival; null at edge start

new_run = ek_change | (em["d"] != 1).fillna(False)
em["run_id"] = new_run.astype("int64").cumsum()

# gap stats per edge (nulls at edge starts ignored by agg)
gaps = flat(em.groupby("edge_key").agg({"d": ["max", "mean", "std"]}).reset_index())
gaps = gaps.rename(
    columns={"d_max": "max_interarrival", "d_mean": "mean_interarrival", "d_std": "std_interarrival"}
)

# runs -> streaks
runs = (
    em.groupby(["edge_key", "run_id"])
    .agg({"month_idx": "count"})
    .reset_index()
    .rename(columns={"month_idx": "run_len"})
)
streaks = flat(runs.groupby("edge_key").agg({"run_len": ["max", "count"]}).reset_index())
streaks = streaks.rename(columns={"run_len_max": "longest_streak", "run_len_count": "n_runs"})
cur = runs.sort_values("run_id").drop_duplicates(subset="edge_key", keep="last")
cur = cur[["edge_key", "run_len"]].rename(columns={"run_len": "current_streak"})
del runs

# phase concentration k = 2, 3 (gated at interpretation time, Section 5)
phase_parts = []
for k in (2, 3):
    em[f"ph{k}"] = (em["month_idx"] % k).astype("int8")
    pc = (
        em.groupby(["edge_key", f"ph{k}"])
        .agg({"month_idx": "count"})
        .reset_index()
        .rename(columns={"month_idx": "cnt"})
    )
    pc = pc.sort_values("cnt", ascending=False).drop_duplicates(subset="edge_key", keep="first")
    pc = pc[["edge_key", "cnt"]].rename(columns={"cnt": f"max_phase_cnt_k{k}"})
    phase_parts.append(pc)
    em = em.drop(columns=[f"ph{k}"])

# halves CV (step-up fingerprint: rent escalators / raises)
em["rn"] = em.groupby("edge_key").cumcount()
mp = core[["edge_key", "months_present"]]
em = em.merge(mp, on="edge_key", how="left")
em["half"] = (em["rn"] * 2 >= em["months_present"]).astype("int8")
h = flat(em.groupby(["edge_key", "half"]).agg({"amount": ["mean", "std"]}).reset_index())
h["cv"] = h["amount_std"] / h["amount_mean"]
h0 = h[h["half"] == 0][["edge_key", "cv"]].rename(columns={"cv": "cv_h1"})
h1 = h[h["half"] == 1][["edge_key", "cv"]].rename(columns={"cv": "cv_h2"})
em = em.drop(columns=["rn", "half", "months_present"])
del h

# %% [markdown]
# ## Section 4 — Amount-deviation distribution → empirical band selection (decision 1)
# Merge each edge's median back; look at |amount − median| / median pooled across
# active months of classifiable edges (months_present ≥ MIN_HISTORY — shorter
# histories bias rel_dev toward 0 mechanically).
# Pick BAND at the valley between the near-zero spike (true fixed payments)
# and the diffuse bulk. Then set BAND in the Parameters cell and re-run from Section 5.

# %%
em = em.merge(core[["edge_key", "amt_median", "months_present"]], on="edge_key", how="left")
em["abs_dev"] = (em["amount"] - em["amt_median"]).abs()
em["rel_dev"] = em["abs_dev"] / em["amt_median"]

# MAD + robust CV while abs_dev is in hand
mad = em.groupby("edge_key").agg({"abs_dev": "median"}).reset_index()
mad = mad.rename(columns={"abs_dev": "amt_mad"})

# pooled distribution (gated)
dist = em[em["months_present"] >= MIN_HISTORY]
zero_share_cnt = float((dist["rel_dev"] <= 1e-9).mean())
zero_share_dlr = float(dist["amount"][dist["rel_dev"] <= 1e-9].sum() / dist["amount"].sum())
print(f"exact-median months (rel_dev ~ 0): {zero_share_cnt:.1%} of edge-months, "
      f"{zero_share_dlr:.1%} of dollars")

BIN_W = 0.005
dist = dist[["rel_dev", "amount"]].copy()
dist["rd_bin"] = (dist["rel_dev"].clip(upper=0.5) / BIN_W).astype("int16")
hist = dist.groupby("rd_bin").agg({"amount": ["sum", "count"]})
hist = flat(hist.reset_index()).to_pandas().sort_values("rd_bin")
hist["rel_dev"] = hist["rd_bin"] * BIN_W
hist["cnt_share"] = hist["amount_count"] / hist["amount_count"].sum()
hist["dlr_share"] = hist["amount_sum"] / hist["amount_sum"].sum()
del dist

fig = go.Figure()
fig.add_bar(x=hist["rel_dev"], y=hist["cnt_share"], name="edge-months (count)", opacity=0.65)
fig.add_bar(x=hist["rel_dev"], y=hist["dlr_share"], name="edge-months (dollar-weighted)", opacity=0.65)
fig.update_layout(barmode="overlay", xaxis_title="|amount − median| / median",
                  yaxis_title="share", yaxis_type="log")
show(fig, "Relative amount deviation — pooled, months_present ≥ 6 (pick BAND at the valley)")

# crude valley suggester on the smoothed count histogram — advisory only, decide by eye
sm = hist.set_index("rel_dev")["cnt_share"].rolling(5, center=True).mean().dropna()
interior = sm[(sm.index > 0.01) & (sm.index < 0.25)]
if len(interior):
    print(f"suggested valley (advisory): BAND ≈ {interior.idxmin():.3f}")

# modal share at the chosen band + the full grid (grid kept for sensitivity table)
band = BAND if BAND is not None else 0.05
print(f"using BAND = {band}" + ("  (FALLBACK — set BAND after inspecting the plot!)" if BAND is None else ""))
band_cols = {}
for b in [band] + [b for b in BAND_GRID if b != band]:
    col = f"in_band_{str(b).replace('.', 'p')}"
    em[col] = (em["rel_dev"] <= b).astype("int8")
    band_cols[b] = col
modal = em.groupby("edge_key").agg({c: "mean" for c in band_cols.values()}).reset_index()
modal = modal.rename(columns={band_cols[band]: "modal_amt_share"})
em = em.drop(columns=list(band_cols.values()) + ["abs_dev", "rel_dev", "amt_median", "months_present"])

# %% [markdown]
# ## Section 5 — Assemble `edge_rhythm_agg`, derived fields, provisional classes

# %%
agg = core
for part in [gaps, streaks, cur, mad, modal] + phase_parts:
    agg = agg.merge(part, on="edge_key", how="left")
del gaps, streaks, cur, mad, modal, phase_parts, core

agg["lifespan"] = (agg["last_month"] - agg["first_month"] + 1).astype("int16")
agg["support_lifespan"] = (agg["months_present"] / agg["lifespan"]).astype("float32")
agg["support_window"] = (agg["months_present"] / N_MONTHS).astype("float32")
agg["months_since_last"] = (LAST_MONTH - agg["last_month"]).astype("int16")
agg["amt_cv"] = agg["amt_std"] / agg["amt_mean"]
agg["amt_rcv"] = agg["amt_mad"] / agg["amt_median"]  # primary stability measure
agg["phase_share_k2"] = agg["max_phase_cnt_k2"] / agg["months_present"]
agg["phase_share_k3"] = agg["max_phase_cnt_k3"] / agg["months_present"]
agg = agg.merge(h0, on="edge_key", how="left").merge(h1, on="edge_key", how="left")
del h0, h1

# ended_flag — orthogonal to rhythm class by design (broken-recurrence signal)
agg["ended_flag"] = (agg["months_since_last"] >= 3)
agg["broken_established"] = agg["ended_flag"] & (agg["longest_streak"] >= 6)

# provisional rhythm classes — thresholds are grid points, NOT decisions.
# np.select-style priority via ascending-priority overwrites (lowest assigned first).
eligible = agg["months_present"] >= MIN_HISTORY
m_fixed = eligible & (agg["support_lifespan"] >= 0.9) & (agg["modal_amt_share"] >= 0.7)
m_var = eligible & (agg["support_lifespan"] >= 0.8)
m_per2 = eligible & (agg["phase_share_k2"] >= 0.9) & (agg["support_lifespan"] <= 0.75)
m_per3 = eligible & (agg["phase_share_k3"] >= 0.9) & (agg["support_lifespan"] <= 0.75)
m_oneshot = agg["months_present"] <= 2
m_tooyoung = agg["lifespan"] < MIN_HISTORY

agg["class_code"] = np.int8(6)                    # INTERMITTENT default
agg["class_code"] = agg["class_code"].mask(m_per3, np.int8(5))
agg["class_code"] = agg["class_code"].mask(m_per2, np.int8(4))
agg["class_code"] = agg["class_code"].mask(m_var, np.int8(3))
agg["class_code"] = agg["class_code"].mask(m_fixed, np.int8(2))
agg["class_code"] = agg["class_code"].mask(m_oneshot, np.int8(1))
agg["class_code"] = agg["class_code"].mask(m_tooyoung, np.int8(0))

CLASS_LABELS = {0: "TOO_YOUNG", 1: "ONE_SHOT", 2: "FIXED_RECURRING",
                3: "VARIABLE_RECURRING", 4: "PERIODIC_2", 5: "PERIODIC_3", 6: "INTERMITTENT"}

agg.to_parquet(AGG_PQ)
print(f"edge_rhythm_agg saved: {len(agg):,} edges x {len(agg.columns)} cols -> {AGG_PQ}")

# %% [markdown]
# ## Section 6 — Scale facts (first-slide numbers)

# %%
facts = {
    "distinct_edges": int(N_EDGES),
    "distinct_nodes": int(N_NODES),
    "edge_month_rows": int(len(em)),
    "total_dollars": float(agg["amt_total"].sum()),
    "eligible_edges_minhist": int((agg["months_present"] >= MIN_HISTORY).sum()),
    "class_counts": {CLASS_LABELS[k]: int(v) for k, v in
                     agg["class_code"].value_counts().to_pandas().items()},
    "class_dollars": {CLASS_LABELS[k]: float(v) for k, v in
                      agg.groupby("class_code")["amt_total"].sum().to_pandas().items()},
    "broken_established_edges": int(agg["broken_established"].sum()),
}
print(json.dumps(facts, indent=2))
(OUT_DIR / "scale_facts.json").write_text(json.dumps(facts, indent=2))

# %% [markdown]
# ## Section 7 — Survival curves: edges% and dollars% vs support threshold
# THE threshold/viability chart. Solid = eligible edges (months_present ≥ 6),
# dashed = all edges. Target claim to test: "~20% of edges, ~50%+ of dollars."

# %%
def survival(frame, support_col):
    t = frame[[support_col, "amt_total"]].copy()
    t["bin"] = (t[support_col].clip(0, 1) * 100).astype("int16")
    g = flat(t.groupby("bin").agg({"amt_total": ["sum", "count"]}).reset_index()).to_pandas()
    g = g.sort_values("bin", ascending=False)
    g["dollar_surv"] = g["amt_total_sum"].cumsum() / g["amt_total_sum"].sum()
    g["edge_surv"] = g["amt_total_count"].cumsum() / g["amt_total_count"].sum()
    g["threshold"] = g["bin"] / 100
    return g.sort_values("threshold")

for scol in ["support_lifespan", "support_window"]:
    fig = go.Figure()
    for label, frame, dash in [("eligible", agg[agg["months_present"] >= MIN_HISTORY], "solid"),
                               ("all", agg, "dash")]:
        s = survival(frame, scol)
        fig.add_scatter(x=s["threshold"], y=s["dollar_surv"], name=f"dollars ({label})",
                        line=dict(dash=dash))
        fig.add_scatter(x=s["threshold"], y=s["edge_surv"], name=f"edges ({label})",
                        line=dict(dash=dash))
    for t in SUPPORT_GRID:
        fig.add_vline(x=t, line_dash="dot", opacity=0.4)
    fig.update_layout(xaxis_title=f"{scol} ≥ t", yaxis_title="share surviving")
    show(fig, f"Survival curve — {scol}")

# %% [markdown]
# ## Section 8 — Joint distribution: support_lifespan × modal_amt_share
# Do FIXED and VARIABLE separate naturally, or smear? Count- and dollar-weighted.

# %%
j = agg[agg["months_present"] >= MIN_HISTORY][
    ["support_lifespan", "modal_amt_share", "amt_total"]].copy()
NB = 20
j["sb"] = (j["support_lifespan"].clip(0, 0.9999) * NB).astype("int16")
j["mb"] = (j["modal_amt_share"].clip(0, 0.9999) * NB).astype("int16")
jj = flat(j.groupby(["sb", "mb"]).agg({"amt_total": ["sum", "count"]}).reset_index()).to_pandas()
for val, name in [("amt_total_count", "count-weighted"), ("amt_total_sum", "dollar-weighted")]:
    piv = jj.pivot(index="mb", columns="sb", values=val).fillna(0)
    fig = px.imshow(np.log10(piv.values + 1), origin="lower",
                    labels=dict(x="support_lifespan bin", y="modal_amt_share bin",
                                color="log10"),
                    x=[f"{c/NB:.2f}" for c in piv.columns], y=[f"{i/NB:.2f}" for i in piv.index])
    show(fig, f"support × modal_amt_share ({name}, log10 scale)")
del j, jj

# %% [markdown]
# ## Section 9 — Hub check: survival under P99 / P99.9 ablation (existing lists)
# Decides whether the regular backbone is anything but processors/payroll.

# %%
agg["src_id"] = (agg["edge_key"] // np.int64(N_NODES)).astype("int64")
agg["dst_id"] = (agg["edge_key"] % np.int64(N_NODES)).astype("int64")

for level, path in HUB_LISTS.items():
    if not Path(path).exists():
        print(f"[skip] {level}: {path} not found — set HUB_LISTS paths")
        continue
    hub = cudf.read_parquet(path) if str(path).endswith("parquet") else cudf.read_csv(path)
    hub_ids = hub[[HUB_NODE_COL]].rename(columns={HUB_NODE_COL: "node"}).merge(
        node_dim, on="node", how="inner")["node_id"]
    excl = agg["src_id"].isin(hub_ids) | agg["dst_id"].isin(hub_ids)
    kept = agg[~excl]
    rec = kept[(kept["support_lifespan"] >= RECURRING_T) & (kept["months_present"] >= MIN_HISTORY)]
    rec_all = agg[(agg["support_lifespan"] >= RECURRING_T) & (agg["months_present"] >= MIN_HISTORY)]
    print(f"{level}: hub-touching edges = {float(excl.mean()):.1%}; "
          f"recurring dollars kept after ablation = "
          f"{float(rec['amt_total'].sum() / rec_all['amt_total'].sum()):.1%}")
    fig = go.Figure()
    for label, frame in [("all nodes", agg), (f"after {level} ablation", kept)]:
        s = survival(frame[frame["months_present"] >= MIN_HISTORY], "support_lifespan")
        fig.add_scatter(x=s["threshold"], y=s["dollar_surv"], name=f"dollars — {label}")
    fig.update_layout(xaxis_title="support_lifespan ≥ t", yaxis_title="dollar share surviving")
    show(fig, f"Survival with vs without {level} hubs")

# top-20 node concentration inside the high-support mass
rec = agg[(agg["support_lifespan"] >= RECURRING_T) & (agg["months_present"] >= MIN_HISTORY)]
node_mass = cudf.concat([
    rec[["src_id", "amt_total"]].rename(columns={"src_id": "nid"}),
    rec[["dst_id", "amt_total"]].rename(columns={"dst_id": "nid"}),
]).groupby("nid").agg({"amt_total": "sum"}).reset_index()
node_mass = node_mass.sort_values("amt_total", ascending=False)
top20 = float(node_mass.head(20)["amt_total"].sum() / rec["amt_total"].sum())
print(f"top-20 nodes touch {top20:.1%} of recurring dollar mass (double-counts both endpoints)")

# %% [markdown]
# ## Section 10 — NAICS conditioning: recurring dollar share by naics2
# Second lightweight pass over the CSVs for node→NAICS (most recent non-null).

# %%
naics_parts = []
for m_idx, f in enumerate(MONTH_FILES):
    ncols = ["source", "source_naics", "dest", "dest_naics"]
    df = cudf.read_csv(f, usecols=ncols, dtype={c: "str" for c in ncols})
    for side, ncol in [("source", "source_naics"), ("dest", "dest_naics")]:
        part = df[[side, ncol]].rename(columns={side: "node", ncol: "naics"}).dropna()
        part = part.drop_duplicates(subset="node")
        part["month_idx"] = np.int8(m_idx)
        naics_parts.append(part)
node_naics = cudf.concat(naics_parts, ignore_index=True)
del naics_parts
node_naics = node_naics.sort_values("month_idx").drop_duplicates(subset="node", keep="last")
node_naics["naics2"] = node_naics["naics"].str.slice(0, 2)
node_naics = node_naics.merge(node_dim, on="node", how="inner")[["node_id", "naics2"]]

nz = agg.merge(node_naics.rename(columns={"node_id": "src_id", "naics2": "src_naics2"}),
               on="src_id", how="left")
nz["recurring"] = (nz["support_lifespan"] >= RECURRING_T) & (nz["months_present"] >= MIN_HISTORY)
by_naics = flat(
    nz.groupby(["src_naics2", "recurring"]).agg({"amt_total": "sum"}).reset_index()
).to_pandas().pivot(index="src_naics2", columns="recurring", values="amt_total").fillna(0)
by_naics["rec_share"] = by_naics.get(True, 0) / by_naics.sum(axis=1)
by_naics = by_naics.sort_values("rec_share", ascending=False)
fig = px.bar(by_naics.reset_index(), x="src_naics2", y="rec_share")
show(fig, f"Recurring dollar share by payer naics2 (support ≥ {RECURRING_T})")
del nz

# %% [markdown]
# ## Section 11 — Volume fingerprints inside FIXED candidates
# vol_mode 1 ≈ rent/loan-like, 2 ≈ biweekly-like, 4 ≈ weekly-like.
# INFERRED, not proven — monthly granularity cannot resolve sub-monthly rhythm.

# %%
fx = agg[agg["class_code"] == 2]
vm = flat(fx.groupby("vol_mode").agg({"amt_total": ["count", "sum"]}).reset_index()).to_pandas()
vm = vm[vm["vol_mode"] <= 10].sort_values("vol_mode")
fig = go.Figure()
fig.add_bar(x=vm["vol_mode"], y=vm["amt_total_count"], name="edges")
fig.add_bar(x=vm["vol_mode"], y=vm["amt_total_sum"], name="dollars", yaxis="y2")
fig.update_layout(yaxis2=dict(overlaying="y", side="right"), xaxis_title="vol_mode")
show(fig, "Volume mode distribution — FIXED_RECURRING candidates")

# %% [markdown]
# ## Section 12 — Phase scan: is PERIODIC_2/3 worth keeping?

# %%
for code, name in [(4, "PERIODIC_2"), (5, "PERIODIC_3")]:
    sub = agg[agg["class_code"] == code]
    print(f"{name}: {len(sub):,} edges, ${float(sub['amt_total'].sum()):,.0f} "
          f"({float(sub['amt_total'].sum() / agg['amt_total'].sum()):.2%} of dollars)")

# %% [markdown]
# ## Section 13 — Threshold sensitivity table
# Class counts and dollar shares over support × band grid. Screenshot into the manifest.

# %%
import pandas as pd

# Section 4 merged per-band modal-share means into agg: the primary band is
# named modal_amt_share, the rest keep their in_band_* names. Map band -> column.
all_bands = BAND_GRID + ([band] if band not in BAND_GRID else [])
BAND_COL = {b: ("modal_amt_share" if b == band else f"in_band_{str(b).replace('.', 'p')}")
            for b in all_bands}

MODAL_CUT = 0.7  # share-of-months cutoff (fixed knob; band is the varying knob)
rows = []
total_dlr = float(agg["amt_total"].sum())
elig = agg["months_present"] >= MIN_HISTORY
for st in SUPPORT_GRID:
    m_rec = elig & (agg["support_lifespan"] >= st)
    for b, bc in BAND_COL.items():
        m_f = m_rec & (agg[bc] >= MODAL_CUT)
        rows.append({
            "support_t": st, "band": b,
            "recurring_edges": int(m_rec.sum()),
            "recurring_edge_share": float(m_rec.mean()),
            "recurring_dollar_share": float(agg["amt_total"][m_rec].sum() / total_dlr),
            "fixed_edges": int(m_f.sum()),
            "fixed_dollar_share": float(agg["amt_total"][m_f].sum() / total_dlr),
        })
sens = pd.DataFrame(rows)
print(sens.to_string(index=False))
sens.to_csv(OUT_DIR / "threshold_sensitivity.csv", index=False)

# %% [markdown]
# ## Done
# Decision checklist coming out of this notebook:
# 1. BAND (from Section 4 valley) — set in Parameters, re-run Sections 5–13.
# 2. Support threshold for FIXED / VARIABLE (from Sections 7–8).
# 3. Backbone viability verdict (Section 9): survives hub ablation or not.
# 4. Keep or drop PERIODIC_2/3 (Section 12).
# Screenshot Sections 4, 7, 8, 9 into the manifest as the calibration record.
