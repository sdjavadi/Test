"""
pkg_deposit_tier1.py — Tier 1: correlation geometry + lead–lag + matched event study.

Consumes the Tier 0 Hive tables and answers, in order of increasing strength:
  1. cross-sectional  — which metrics co-vary with deposit size/outcomes across customers?
  2. within-customer  — when THIS customer's network moves, do THEIR deposits move?
  3. lead–lag         — which side moves FIRST, and by how many months?
  4. event study      — do metric trajectories of decliners separate from matched
                        controls BEFORE the deposit event month t0?

Workflow: Spark pull (graph-visible customers only, ~190K rows) → pandas/scipy.
Outputs: CSVs + Plotly HTML figures under cfg.out_dir.

Method notes:
  * Spearman everywhere cross-sectional (heavy tails); pooled Pearson on
    within-customer demeaned deltas (rank-demeaning is not meaningful).
  * Metrics split into DELTA family (levels are stocks — strength, centrality;
    analyzed as Δ) and LEVEL family (turnover/contagion metrics are already
    rates of change — analyzed as levels). Deltas require consecutive graph
    months (month_idx diff == 1); gaps yield NaN, never a spanning difference.
  * Event study uses per-month cross-sectional PERCENTILE RANKS of each metric
    within the graph-visible sample — robust to tails and removes month/seasonal
    level shifts. Controls matched on (month, naics2, prior-balance decile),
    required event-free within ±guard months.
  * Lead–lag sign convention: lag k > 0 ⇒ metric at t−k vs deposit change at t,
    i.e. THE METRIC LEADS. k < 0 ⇒ deposits lead the metric.

Multiple testing: this is a SCREEN. Treat |rho| rankings and trajectory
separation as candidate-generators for Tier 2, not as final inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# ----------------------------------------------------------------------------- config

DELTA_METRICS = [  # stocks → analyze as Δlog / Δ
    "pkg_in_strength", "pkg_out_strength", "pkg_strength", "pkg_net_flow",
    "pkg_flow_ratio", "pkg_in_degree", "pkg_out_degree", "pkg_n_neighbors",
    "pkg_pagerank_logw", "pkg_core_number", "pkg_hhi_in", "pkg_top1_in_share",
    "pkg_hub_in_share", "pkg_hub_out_share", "pkg_clustering_coef",
    "pkg_avg_in_ticket", "pkg_in_volume",
]
LEVEL_METRICS = [  # already rates/flux → analyze as levels
    "pkg_payer_jaccard", "pkg_payee_jaccard", "pkg_lost_payer_amount_share",
    "pkg_new_payer_amount_share", "pkg_recurring_payer_amount_share",
    "pkg_top_payer_same", "pkg_top_payer_share_delta",
    "pkg_nbr_strength_trend", "pkg_inflow_from_shrinking_share",
    "pkg_activity_gap", "pkg_months_active",
]
EVENT_METRICS = [  # focused set for trajectory plots
    "pkg_in_strength", "pkg_net_flow", "pkg_in_degree",
    "pkg_payer_jaccard", "pkg_lost_payer_amount_share",
    "pkg_recurring_payer_amount_share", "pkg_nbr_strength_trend",
    "pkg_inflow_from_shrinking_share", "pkg_hub_in_share", "pkg_top_payer_same",
]


@dataclass
class Tier1Config:
    joint_table: str = "bdahd01p_dldsi1_dsi_lab.cust_deposit_pkg_metrics_joint_panel"
    graph_start: str = "2024-01"
    graph_end: str = "2025-11"

    delta_metrics: list[str] = field(default_factory=lambda: list(DELTA_METRICS))
    level_metrics: list[str] = field(default_factory=lambda: list(LEVEL_METRICS))
    event_metrics: list[str] = field(default_factory=lambda: list(EVENT_METRICS))

    # lead–lag
    lags: tuple = tuple(range(-6, 7))
    min_pairs_per_lag: int = 500

    # within-customer
    min_obs_per_customer: int = 8

    # event study
    event_pre: int = 6
    event_post: int = 3
    event_guard: int = 3          # controls must be event-free within ±guard of t0
    n_controls: int = 5
    min_event_traj_months: int = 4  # metric must be observed ≥ this many offsets
    size_deciles: int = 10
    random_state: int = 43

    out_dir: str = "./tier1_out"


# dark theme, house style
pio.templates["pkg_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(family="IBM Plex Sans, IBM Plex Mono, monospace", color="#e6e6e6"),
        colorway=["#4cc9f0", "#f72585", "#ffd166", "#06d6a0", "#b5179e",
                  "#f8961e", "#90be6d", "#577590"],
        xaxis=dict(gridcolor="#22252b"), yaxis=dict(gridcolor="#22252b"),
    )
)
pio.templates.default = "pkg_dark"


# ----------------------------------------------------------------------------- pull

def pull_panel(spark, cfg: Tier1Config) -> pd.DataFrame:
    """Graph-visible customers, all months through graph_end, → pandas."""
    keep_dep = [
        "cust_pwr_id", "month", "month_idx", "balance", "balance_status",
        "dep_recent_avg", "dep_prior_avg", "dep_pct_change", "dep_eligible",
        "dep_decline_flag", "dep_is_event_start", "dep_dlog_1m", "dep_dlog_3m",
        "dep_drawdown",
    ]
    metric_cols = sorted(set(cfg.delta_metrics + cfg.level_metrics + cfg.event_metrics))
    j = spark.table(cfg.joint_table)
    avail = [c for c in metric_cols if c in j.columns]
    missing = sorted(set(metric_cols) - set(avail))
    if missing:
        print(f"[pull][WARN] not in joint table, skipped: {missing}")
    extra = [c for c in ("pkg_naics2", "in_graph_month") if c in j.columns]

    from pyspark.sql import functions as F
    visible = (
        j.filter(F.col("in_graph_month") == 1)
        .select("cust_pwr_id").distinct()
    )
    pdf = (
        j.join(visible, "cust_pwr_id", "left_semi")
        .filter(F.col("month") <= cfg.graph_end)
        .select(*(keep_dep + extra + avail))
        .toPandas()
    )
    pdf = pdf.sort_values(["cust_pwr_id", "month_idx"]).reset_index(drop=True)
    print(f"[pull] rows={len(pdf):,} customers={pdf.cust_pwr_id.nunique():,} "
          f"graph rows={int(pdf.in_graph_month.sum()):,}")
    # trim config lists to what actually exists
    cfg.delta_metrics = [m for m in cfg.delta_metrics if m in pdf.columns]
    cfg.level_metrics = [m for m in cfg.level_metrics if m in pdf.columns]
    cfg.event_metrics = [m for m in cfg.event_metrics if m in pdf.columns]
    return pdf


# ----------------------------------------------------------------------------- prep

def _signed_dlog(s: pd.Series) -> pd.Series:
    """Δ signed-log for stocks that can be ≤0 (net_flow): sign(x)·log1p|x|."""
    return np.sign(s) * np.log1p(s.abs())


def prepare(pdf: pd.DataFrame, cfg: Tier1Config) -> pd.DataFrame:
    g = pdf.groupby("cust_pwr_id", sort=False)
    consec = g["month_idx"].diff().eq(1)

    for m in cfg.delta_metrics:
        base = _signed_dlog(pdf[m]) if pdf[m].min(skipna=True) < 0 else np.log1p(pdf[m])
        d = base.groupby(pdf["cust_pwr_id"]).diff()
        pdf[f"d_{m}"] = d.where(consec)

    # per-month cross-sectional percentile ranks (graph rows only)
    graph_mask = pdf["in_graph_month"] == 1
    for m in cfg.event_metrics:
        pdf[f"pr_{m}"] = (
            pdf.loc[graph_mask].groupby("month")[m].rank(pct=True)
        )

    # forward outcomes
    pdf["fwd_dlog_3m"] = g["dep_dlog_3m"].shift(-3)
    ev = pdf["dep_is_event_start"].fillna(0)
    pdf["fwd_event_3m"] = (
        ev.groupby(pdf["cust_pwr_id"])
        .transform(lambda s: s.shift(-1).fillna(0) + s.shift(-2).fillna(0)
                   + s.shift(-3).fillna(0))
        .clip(upper=1)
    )
    pdf["log_recent_avg"] = np.where(
        pdf["dep_recent_avg"] > 0, np.log(pdf["dep_recent_avg"]), np.nan
    )
    return pdf


# ----------------------------------------------------------------------------- 1. cross-sectional

def cross_sectional(pdf: pd.DataFrame, cfg: Tier1Config) -> pd.DataFrame:
    """Per-month Spearman of each metric vs (a) log balance level,
    (b) forward 3m Δlog balance, (c) forward event flag."""
    rows = []
    metrics = cfg.delta_metrics + cfg.level_metrics
    gm = pdf[pdf.in_graph_month == 1]
    for month, sub in gm.groupby("month"):
        for m in metrics:
            for target, tname in [
                ("log_recent_avg", "level_log_balance"),
                ("fwd_dlog_3m", "fwd_dlog_3m"),
                ("fwd_event_3m", "fwd_event_3m"),
            ]:
                s = sub[[m, target]].dropna()
                if len(s) < 100:
                    continue
                rho, p = stats.spearmanr(s[m], s[target])
                rows.append((month, m, tname, rho, p, len(s)))
    out = pd.DataFrame(rows, columns=["month", "metric", "target", "rho", "p", "n"])
    out.to_csv(f"{cfg.out_dir}/cross_sectional_corr.csv", index=False)

    for tname in out.target.unique():
        piv = out[out.target == tname].pivot(index="metric", columns="month", values="rho")
        piv = piv.reindex(piv.mean(axis=1).abs().sort_values(ascending=False).index)
        fig = px.imshow(
            piv, zmin=-0.5, zmax=0.5, color_continuous_scale="RdBu_r", aspect="auto",
            title=f"Cross-sectional Spearman vs {tname} (per month)",
        )
        fig.write_html(f"{cfg.out_dir}/heatmap_{tname}.html")
    return out


# ----------------------------------------------------------------------------- 2. within-customer

def within_customer(pdf: pd.DataFrame, cfg: Tier1Config) -> pd.DataFrame:
    """Pooled Pearson on within-customer-demeaned pairs: Δmetric (or level)
    vs same-month dep_dlog_1m. Answers co-movement net of who-the-customer-is."""
    rows = []
    feats = [f"d_{m}" for m in cfg.delta_metrics] + cfg.level_metrics
    for f in feats:
        s = pdf[["cust_pwr_id", f, "dep_dlog_1m"]].dropna()
        counts = s.groupby("cust_pwr_id").size()
        s = s[s.cust_pwr_id.isin(counts[counts >= cfg.min_obs_per_customer].index)]
        if len(s) < 1000:
            continue
        x = s[f] - s.groupby("cust_pwr_id")[f].transform("mean")
        y = s["dep_dlog_1m"] - s.groupby("cust_pwr_id")["dep_dlog_1m"].transform("mean")
        r, p = stats.pearsonr(x, y)
        rows.append((f, r, p, len(s), s.cust_pwr_id.nunique()))
    out = pd.DataFrame(rows, columns=["feature", "r_within", "p", "n_obs", "n_cust"])
    out = out.sort_values("r_within", key=np.abs, ascending=False)
    out.to_csv(f"{cfg.out_dir}/within_customer_corr.csv", index=False)
    print("\n[within-customer] top 10 by |r|:")
    print(out.head(10).to_string(index=False))
    return out


# ----------------------------------------------------------------------------- 3. lead–lag

def lead_lag(pdf: pd.DataFrame, cfg: Tier1Config) -> pd.DataFrame:
    """corr( feature(t−k), dep_dlog_1m(t) ) pooled on within-customer demeaned
    values. k>0 ⇒ metric leads deposits; k<0 ⇒ deposits lead the metric."""
    feats = [f"d_{m}" for m in cfg.delta_metrics] + cfg.level_metrics
    base = pdf[["cust_pwr_id", "month_idx", "dep_dlog_1m"] + feats].copy()
    rows = []
    g = base.groupby("cust_pwr_id", sort=False)
    for k in cfg.lags:
        shifted = g[feats].shift(k)          # feature at t−k aligned to row t
        idx_ok = g["month_idx"].shift(k).eq(base["month_idx"] - k)  # true k-month gap
        for f in feats:
            s = pd.DataFrame({
                "cust": base["cust_pwr_id"],
                "x": shifted[f].where(idx_ok),
                "y": base["dep_dlog_1m"],
            }).dropna()
            if len(s) < cfg.min_pairs_per_lag:
                continue
            x = s["x"] - s.groupby("cust")["x"].transform("mean")
            y = s["y"] - s.groupby("cust")["y"].transform("mean")
            r, p = stats.pearsonr(x, y)
            rows.append((f, k, r, p, len(s)))
    out = pd.DataFrame(rows, columns=["feature", "lag", "r", "p", "n"])
    out.to_csv(f"{cfg.out_dir}/lead_lag.csv", index=False)

    best = (
        out.loc[out.groupby("feature")["r"].transform(lambda s: s.abs() == s.abs().max())]
        .drop_duplicates("feature").sort_values("r", key=np.abs, ascending=False)
    )
    best.to_csv(f"{cfg.out_dir}/lead_lag_best.csv", index=False)
    print("\n[lead-lag] best lag per feature (k>0 = metric leads):")
    print(best.head(15).to_string(index=False))

    top = best.head(10).feature.tolist()
    fig = go.Figure()
    for f in top:
        sub = out[out.feature == f].sort_values("lag")
        fig.add_trace(go.Scatter(x=sub.lag, y=sub.r, mode="lines+markers", name=f))
    fig.update_layout(
        title="Lead–lag: corr(feature(t−k), Δlog deposit(t)) — k>0 ⇒ metric leads",
        xaxis_title="lag k (months)", yaxis_title="within-customer r",
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#888")
    fig.write_html(f"{cfg.out_dir}/lead_lag.html")
    return out


# ----------------------------------------------------------------------------- 4. event study

def event_study(pdf: pd.DataFrame, cfg: Tier1Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    lo = pd.Period(cfg.graph_start, "M").ordinal + 24 * 0 + cfg.event_pre  # need t−pre ≥ start
    hi = pd.Period(cfg.graph_end, "M").ordinal                             # t0 ≤ graph end
    # month_idx in the panel is months-since-1970, same basis as Period ordinal
    start_idx = pd.Period(cfg.graph_start, "M").ordinal

    ev_idx = pdf.groupby("cust_pwr_id")["month_idx"].apply(
        lambda s: set(s[pdf.loc[s.index, "dep_is_event_start"].fillna(0).eq(1)])
    ).to_dict()

    events = pdf[
        (pdf.dep_is_event_start == 1)
        & (pdf.month_idx >= start_idx + cfg.event_pre)
        & (pdf.month_idx <= hi)
    ][["cust_pwr_id", "month_idx", "month", "dep_prior_avg", "pkg_naics2"]].copy()
    print(f"\n[event] candidate events in window: {len(events):,} "
          f"({events.cust_pwr_id.nunique():,} customers)")

    # control pool: eligible, not in decline, event-free within ±guard
    pool = pdf[(pdf.dep_eligible == 1) & (pdf.dep_decline_flag == 0)][
        ["cust_pwr_id", "month_idx", "month", "dep_prior_avg", "pkg_naics2"]
    ].copy()

    def _event_free(cust, idx):
        evs = ev_idx.get(cust, set())
        return all(abs(idx - e) > cfg.event_guard for e in evs)

    pool = pool[[_event_free(c, i) for c, i in zip(pool.cust_pwr_id, pool.month_idx)]]

    # size deciles within month over events ∪ pool
    both = pd.concat([events.assign(_role="event"), pool.assign(_role="control")])
    both["size_decile"] = (
        both.groupby("month_idx")["dep_prior_avg"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), cfg.size_deciles,
                                     labels=False, duplicates="drop"))
    )
    events = both[both._role == "event"].copy()
    pool = both[both._role == "control"].copy()

    # match: same (month_idx, naics2, size_decile); fallback to (month_idx, size_decile)
    matched = []
    pool_g1 = {k: v for k, v in pool.groupby(["month_idx", "pkg_naics2", "size_decile"])}
    pool_g2 = {k: v for k, v in pool.groupby(["month_idx", "size_decile"])}
    for eid, e in enumerate(events.itertuples()):
        cand = pool_g1.get((e.month_idx, e.pkg_naics2, e.size_decile))
        if cand is None or len(cand) == 0:
            cand = pool_g2.get((e.month_idx, e.size_decile))
        if cand is None or len(cand) == 0:
            continue
        cand = cand[cand.cust_pwr_id != e.cust_pwr_id]
        take = cand.sample(min(cfg.n_controls, len(cand)), random_state=rng.integers(1 << 31))
        matched.append(pd.DataFrame({
            "event_id": eid, "role": "event", "cust_pwr_id": [e.cust_pwr_id],
            "t0_idx": [e.month_idx],
        }))
        matched.append(pd.DataFrame({
            "event_id": eid, "role": "control", "cust_pwr_id": take.cust_pwr_id.values,
            "t0_idx": e.month_idx,
        }))
    if not matched:
        raise RuntimeError("no matched events — check pkg_naics2 availability / pool size")
    cohort = pd.concat(matched, ignore_index=True)
    n_ev = (cohort.role == "event").sum()
    print(f"[event] matched events: {n_ev:,}; control rows: {len(cohort) - n_ev:,}")

    # explode offsets and join percentile-rank trajectories
    offsets = np.arange(-cfg.event_pre, cfg.event_post + 1)
    cohort = cohort.loc[cohort.index.repeat(len(offsets))].reset_index(drop=True)
    cohort["offset"] = np.tile(offsets, len(cohort) // len(offsets))
    cohort["month_idx"] = cohort["t0_idx"] + cohort["offset"]

    pr_cols = [f"pr_{m}" for m in cfg.event_metrics]
    panel = pdf[["cust_pwr_id", "month_idx"] + pr_cols]
    cohort = cohort.merge(panel, on=["cust_pwr_id", "month_idx"], how="left")

    # require minimum trajectory coverage on the pre-window (any metric)
    cov = cohort[cohort.offset <= 0].groupby(["event_id", "role", "cust_pwr_id"])[
        pr_cols
    ].count().max(axis=1)
    keep = cov[cov >= cfg.min_event_traj_months].reset_index()[
        ["event_id", "role", "cust_pwr_id"]
    ]
    cohort = cohort.merge(keep, on=["event_id", "role", "cust_pwr_id"], how="inner")

    long = cohort.melt(
        id_vars=["event_id", "role", "cust_pwr_id", "offset"],
        value_vars=pr_cols, var_name="metric", value_name="pct_rank",
    ).dropna()
    long["metric"] = long["metric"].str.replace("^pr_", "", regex=True)

    agg = (
        long.groupby(["metric", "role", "offset"])["pct_rank"]
        .agg(["mean", "sem", "count"]).reset_index()
    )
    agg.to_csv(f"{cfg.out_dir}/event_study.csv", index=False)

    # faceted figure
    fig = go.Figure()
    metrics = agg.metric.unique()
    from plotly.subplots import make_subplots
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(metrics),
                        shared_xaxes=True)
    for i, m in enumerate(metrics):
        r, c = i // ncols + 1, i % ncols + 1
        for role, color in [("event", "#f72585"), ("control", "#4cc9f0")]:
            sub = agg[(agg.metric == m) & (agg.role == role)].sort_values("offset")
            fig.add_trace(
                go.Scatter(
                    x=sub.offset, y=sub["mean"],
                    error_y=dict(array=1.96 * sub["sem"], thickness=1),
                    mode="lines+markers", name=role, legendgroup=role,
                    showlegend=(i == 0), line=dict(color=color),
                ),
                row=r, col=c,
            )
    fig.update_layout(
        height=280 * nrows, title="Event study — mean per-month percentile rank, "
        "decliners vs matched controls (t0 = decline-flag onset)",
    )
    fig.write_html(f"{cfg.out_dir}/event_study.html")
    return agg


# ----------------------------------------------------------------------------- main

def run(spark, cfg: Tier1Config | None = None):
    cfg = cfg or Tier1Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    pdf = pull_panel(spark, cfg)
    pdf = prepare(pdf, cfg)

    cs = cross_sectional(pdf, cfg)
    wc = within_customer(pdf, cfg)
    ll = lead_lag(pdf, cfg)
    es = event_study(pdf, cfg)

    print(f"\n[done] Tier 1 outputs in {cfg.out_dir}/ — "
          "csv: cross_sectional_corr, within_customer_corr, lead_lag(+best), "
          "event_study; html: heatmaps, lead_lag, event_study")
    return dict(panel=pdf, cross_sectional=cs, within=wc, lead_lag=ll, event=es)


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    run(SparkSession.builder.enableHiveSupport().getOrCreate())
