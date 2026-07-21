"""
pkg_deposit_tier2.py — Tier 2: decline-ONSET classifier with incremental-value design.

Predicts P(decline-flag onset within next `horizon` months) from rows that are
currently eligible and NOT in decline. Three feature sets on identical rows:

    A  deposit-only   — the baseline any deposit system could run today
    B  graph-only     — PKG-derived features alone
    C  combined       — A ∪ B

THE DELIVERABLE IS THE C−A GAP: does the payment network add predictive signal
beyond deposit history itself? Reported per rolling-origin fold and on the final
FREEZE TEST (train ≤ freeze_train_end; score feature months after it, whose
outcomes extend past the network-data horizon into 2026 — structurally
zero-leakage out-of-time evaluation).

Models: LogisticRegression (standardized, class-weighted — interpretable floor)
and HistGradientBoostingClassifier (native NaN handling — no imputation
distortion from intermittent graph presence).

Outputs (cfg.out_dir): fold_metrics.csv, freeze_metrics.csv,
freeze_predictions.csv, importances.csv, figures auc_by_fold.html,
lift_freeze.html, importance.html.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "pkg_dark" if "pkg_dark" in pio.templates else "plotly_dark"


# ----------------------------------------------------------------------------- config

STOCK_METRICS = [
    "pkg_in_strength", "pkg_out_strength", "pkg_strength", "pkg_net_flow",
    "pkg_flow_ratio", "pkg_in_degree", "pkg_out_degree", "pkg_n_neighbors",
    "pkg_pagerank_logw", "pkg_core_number", "pkg_hhi_in", "pkg_top1_in_share",
    "pkg_hub_in_share", "pkg_avg_in_ticket", "pkg_in_volume",
]
RATE_METRICS = [
    "pkg_payer_jaccard", "pkg_lost_payer_amount_share",
    "pkg_new_payer_amount_share", "pkg_recurring_payer_amount_share",
    "pkg_top_payer_same", "pkg_top_payer_share_delta",
    "pkg_nbr_strength_trend", "pkg_inflow_from_shrinking_share",
    "pkg_activity_gap", "pkg_months_active",
]


@dataclass
class Tier2Config:
    joint_table: str = "bdahd01p_dldsi1_dsi_lab.cust_deposit_pkg_metrics_joint_panel"
    graph_end: str = "2025-11"

    horizon: int = 3                      # onset window (months ahead)
    stock_metrics: list[str] = field(default_factory=lambda: list(STOCK_METRICS))
    rate_metrics: list[str] = field(default_factory=lambda: list(RATE_METRICS))

    # rolling-origin folds: (train ≤ cutoff, test = next 3 feature-months)
    fold_cutoffs: tuple = ("2024-12", "2025-03", "2025-06")
    freeze_train_end: str = "2025-08"     # freeze test: features 2025-09..2025-11

    n_permutation_repeats: int = 5
    random_state: int = 43
    out_dir: str = "./tier2_out"


def _idx(ym: str) -> int:
    y, m = map(int, ym.split("-"))
    return y * 12 + m - 1


# ----------------------------------------------------------------------------- pull

def pull_panel(spark, cfg: Tier2Config) -> pd.DataFrame:
    from pyspark.sql import functions as F
    j = spark.table(cfg.joint_table)
    metric_cols = [c for c in cfg.stock_metrics + cfg.rate_metrics if c in j.columns]
    missing = sorted(set(cfg.stock_metrics + cfg.rate_metrics) - set(metric_cols))
    if missing:
        print(f"[pull][WARN] skipped (absent): {missing}")
    cfg.stock_metrics = [m for m in cfg.stock_metrics if m in metric_cols]
    cfg.rate_metrics = [m for m in cfg.rate_metrics if m in metric_cols]

    keep = [
        "cust_pwr_id", "month", "month_idx", "balance", "balance_status",
        "dep_recent_avg", "dep_prior_avg", "dep_pct_change", "dep_eligible",
        "dep_decline_flag", "dep_is_event_start", "dep_dlog_1m", "dep_dlog_3m",
        "dep_drawdown", "in_graph_month",
    ] + metric_cols

    visible = j.filter(F.col("in_graph_month") == 1).select("cust_pwr_id").distinct()
    pdf = (
        j.join(visible, "cust_pwr_id", "left_semi")
        .filter(F.col("month") <= cfg.graph_end)
        .select(*keep)
        .toPandas()
        .sort_values(["cust_pwr_id", "month_idx"])
        .reset_index(drop=True)
    )
    print(f"[pull] rows={len(pdf):,} customers={pdf.cust_pwr_id.nunique():,}")
    return pdf


# ----------------------------------------------------------------------------- features

def build_features(pdf: pd.DataFrame, cfg: Tier2Config):
    g = lambda col: pdf.groupby("cust_pwr_id", sort=False)[col]
    consec = pdf.groupby("cust_pwr_id", sort=False)["month_idx"].diff().eq(1)

    dep_feats, pkg_feats = [], []

    # ---- A: deposit-only
    pdf["f_dep_dlog_1m"] = pdf["dep_dlog_1m"]
    pdf["f_dep_dlog_3m"] = pdf["dep_dlog_3m"]
    pdf["f_dep_drawdown"] = pdf["dep_drawdown"]
    pdf["f_dep_pct_change"] = pdf["dep_pct_change"]
    pdf["f_dep_log_bal"] = np.where(pdf.dep_recent_avg > 0,
                                    np.log(pdf.dep_recent_avg), np.nan)
    pdf["f_dep_vol_3m"] = g("dep_dlog_1m").transform(
        lambda s: s.rolling(3, min_periods=2).std()
    )
    pdf["f_dep_neg_streak"] = g("dep_dlog_1m").transform(
        lambda s: (s < 0).rolling(3, min_periods=3).sum()
    )
    dep_feats += ["f_dep_dlog_1m", "f_dep_dlog_3m", "f_dep_drawdown",
                  "f_dep_pct_change", "f_dep_log_bal", "f_dep_vol_3m",
                  "f_dep_neg_streak"]

    # ---- B: graph-only
    for m in cfg.stock_metrics:
        base = (np.sign(pdf[m]) * np.log1p(pdf[m].abs())
                if pdf[m].min(skipna=True) < 0 else np.log1p(pdf[m]))
        pdf[f"_lg_{m}"] = base
        d1 = base.groupby(pdf.cust_pwr_id).diff().where(consec)
        pdf[f"f_{m}_lvl"] = base
        pdf[f"f_{m}_d1"] = d1
        pdf[f"f_{m}_ch3"] = base - base.groupby(pdf.cust_pwr_id).shift(3)
        pdf[f"f_{m}_negstreak"] = (
            (d1 < 0).groupby(pdf.cust_pwr_id).transform(
                lambda s: s.rolling(3, min_periods=3).sum())
        )
        pkg_feats += [f"f_{m}_lvl", f"f_{m}_d1", f"f_{m}_ch3", f"f_{m}_negstreak"]

    for m in cfg.rate_metrics:
        pdf[f"f_{m}_lvl"] = pdf[m]
        pdf[f"f_{m}_m3"] = g(m).transform(lambda s: s.rolling(3, min_periods=2).mean())
        pkg_feats += [f"f_{m}_lvl", f"f_{m}_m3"]

    pdf["f_pkg_present"] = pdf["in_graph_month"].astype(float)
    pdf["f_pkg_present_3m"] = g("in_graph_month").transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    pkg_feats += ["f_pkg_present", "f_pkg_present_3m"]

    # ---- target: onset within next `horizon` months
    ev = pdf["dep_is_event_start"].fillna(0)
    fwd = np.zeros(len(pdf))
    for k in range(1, cfg.horizon + 1):
        fwd = fwd + ev.groupby(pdf.cust_pwr_id).shift(-k).fillna(0).values
    pdf["y_onset"] = (fwd > 0).astype(int)

    # at-risk rows only: eligible, not currently declining
    mask = (pdf.dep_eligible == 1) & (pdf.dep_decline_flag == 0)
    # outcome must be observable: need horizon months of flag ahead → deposit data
    # runs to 2026-06 so every feature month ≤ graph_end qualifies
    rows = pdf[mask].copy()
    print(f"[features] at-risk rows={len(rows):,}  onset rate="
          f"{rows.y_onset.mean():.3%}  dep={len(dep_feats)} pkg={len(pkg_feats)}")
    return rows, dep_feats, pkg_feats


# ----------------------------------------------------------------------------- models

def _fit_eval(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
              cfg: Tier2Config, label: str):
    Xtr, ytr = train[feats], train["y_onset"]
    Xte, yte = test[feats], test["y_onset"]
    out = []

    logit = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)),
    ])
    hgb = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=400,
        l2_regularization=1.0, random_state=cfg.random_state,
    )
    for name, model in [("logit", logit), ("hgb", hgb)]:
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        ap = average_precision_score(yte, p)
        # lift@top decile
        k = max(1, len(p) // 10)
        top = np.argsort(-p)[:k]
        lift = yte.iloc[top].mean() / max(yte.mean(), 1e-9)
        out.append(dict(set=label, model=name, auc=auc, pr_auc=ap,
                        lift_top10=lift, n_test=len(yte),
                        base_rate=yte.mean()))
    return out, dict(logit=logit, hgb=hgb), p  # p = last (hgb) test scores


def run(spark, cfg: Tier2Config | None = None):
    cfg = cfg or Tier2Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    pdf = pull_panel(spark, cfg)
    rows, dep_feats, pkg_feats = build_features(pdf, cfg)
    sets = {"A_deposit": dep_feats, "B_graph": pkg_feats,
            "C_combined": dep_feats + pkg_feats}

    # ---- rolling-origin folds
    fold_rows = []
    for cutoff in cfg.fold_cutoffs:
        ci = _idx(cutoff)
        train = rows[rows.month_idx <= ci]
        test = rows[(rows.month_idx > ci) & (rows.month_idx <= ci + 3)]
        if test.y_onset.sum() < 30:
            print(f"[fold {cutoff}] too few positives, skipped")
            continue
        for label, feats in sets.items():
            res, _, _ = _fit_eval(train, test, feats, cfg, label)
            for r in res:
                r["fold"] = cutoff
                fold_rows.append(r)
        a = [r for r in fold_rows if r["fold"] == cutoff]
        print(f"[fold {cutoff}] " + "  ".join(
            f"{r['set']}/{r['model']}: auc={r['auc']:.3f}" for r in a))
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(f"{cfg.out_dir}/fold_metrics.csv", index=False)

    # ---- freeze test: train ≤ freeze_train_end, score remaining feature months
    fi = _idx(cfg.freeze_train_end)
    train = rows[rows.month_idx <= fi]
    test = rows[rows.month_idx > fi]
    print(f"\n[freeze] train rows={len(train):,} test rows={len(test):,} "
          f"test onset rate={test.y_onset.mean():.3%} "
          f"(outcomes extend into Dec 2025 – Feb 2026)")
    freeze_rows, models_c, p_c = [], None, None
    for label, feats in sets.items():
        res, models, p = _fit_eval(train, test, feats, cfg, label)
        for r in res:
            r["fold"] = "freeze"
            freeze_rows.append(r)
        if label == "C_combined":
            models_c, p_c = models, p
    freeze = pd.DataFrame(freeze_rows)
    freeze.to_csv(f"{cfg.out_dir}/freeze_metrics.csv", index=False)
    print(freeze[["set", "model", "auc", "pr_auc", "lift_top10", "base_rate"]]
          .to_string(index=False))

    gap = (freeze.query("model=='hgb' and set=='C_combined'").auc.iloc[0]
           - freeze.query("model=='hgb' and set=='A_deposit'").auc.iloc[0])
    print(f"\n[freeze] *** C − A AUC gap (hgb) = {gap:+.4f} ***")

    # predictions for downstream review / stakeholder examples
    pred = test[["cust_pwr_id", "month", "y_onset"]].copy()
    pred["p_onset_combined"] = p_c
    pred.sort_values("p_onset_combined", ascending=False).to_csv(
        f"{cfg.out_dir}/freeze_predictions.csv", index=False)

    # ---- permutation importance (combined hgb, freeze test rows)
    samp = test.sample(min(len(test), 20_000), random_state=cfg.random_state)
    imp = permutation_importance(
        models_c["hgb"], samp[sets["C_combined"]], samp["y_onset"],
        n_repeats=cfg.n_permutation_repeats, random_state=cfg.random_state,
        scoring="roc_auc",
    )
    importances = (
        pd.DataFrame({"feature": sets["C_combined"],
                      "importance": imp.importances_mean,
                      "std": imp.importances_std})
        .sort_values("importance", ascending=False)
    )
    importances.to_csv(f"{cfg.out_dir}/importances.csv", index=False)
    print("\n[importance] top 15 (permutation, freeze test):")
    print(importances.head(15).to_string(index=False))

    # ---- figures
    fig = go.Figure()
    for label in sets:
        sub = folds[(folds.set == label) & (folds.model == "hgb")]
        fig.add_trace(go.Scatter(x=sub.fold, y=sub.auc,
                                 mode="lines+markers", name=label))
        fz = freeze[(freeze.set == label) & (freeze.model == "hgb")]
        fig.add_trace(go.Scatter(x=["freeze"], y=fz.auc, mode="markers",
                                 marker=dict(size=14, symbol="star"),
                                 name=f"{label} (freeze)", showlegend=False))
    fig.update_layout(title="ROC-AUC by fold — hgb (star = freeze test)",
                      yaxis_title="AUC")
    fig.write_html(f"{cfg.out_dir}/auc_by_fold.html")

    # lift curve on freeze test (combined)
    order = np.argsort(-p_c)
    y_sorted = test["y_onset"].to_numpy()[order]
    frac = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    cum_capture = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=frac, y=cum_capture, name="combined model"))
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="random",
                              line=dict(dash="dot", color="#888")))
    fig2.update_layout(title="Freeze test — cumulative capture of decline onsets",
                       xaxis_title="fraction of book flagged",
                       yaxis_title="fraction of onsets captured")
    fig2.write_html(f"{cfg.out_dir}/lift_freeze.html")

    top = importances.head(20).iloc[::-1]
    fig3 = go.Figure(go.Bar(x=top.importance, y=top.feature, orientation="h",
                            error_x=dict(array=top["std"])))
    fig3.update_layout(title="Permutation importance — combined model, freeze test",
                       height=600)
    fig3.write_html(f"{cfg.out_dir}/importance.html")

    print(f"\n[done] Tier 2 outputs in {cfg.out_dir}/")
    return dict(rows=rows, folds=folds, freeze=freeze, importances=importances)


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    run(SparkSession.builder.enableHiveSupport().getOrCreate())
