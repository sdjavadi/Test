"""
pkg_deposit_diagnostics.py — post-Tier-2 diagnostics on the freeze test.

Two questions, both about whether the null C−A result is a COVERAGE artifact
or a genuine absence of leading signal:

  D1 — VISIBILITY CUT: does the graph add lift where it actually sees the
       customer? Evaluates the C−A gap separately by graph-presence subset,
       two ways: (a) global models scored per subset; (b) models REFIT on
       fully-visible rows only — the sharper test, since a global model
       trained on majority-missing pkg features rationally learns to ignore
       them, which can mask subset-level signal.

  D2 — SURPRISE ATTRITION: among freeze rows the deposit model called
       low-risk, do graph scores separate the onsets that happened anyway?
       Average incremental AUC can be ~0 while the graph still catches a
       slice deposits structurally miss — that slice is the operational
       value case (the customer the RM gets no warning about today).

Reads the same joint panel as Tier 2; self-contained apart from importing
the Tier 2 feature builder (run from the same directory / notebook kernel).

Outputs: prints + {out_dir}/diagnostics_summary.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from pkg_deposit_tier2 import Tier2Config, pull_panel, build_features, _idx

RS = 43


def _hgb():
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=400,
        l2_regularization=1.0, random_state=RS,
    )


def _safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 50:
        return np.nan
    return roc_auc_score(y, p)


def _boot_gap(y, pa, pc, n_boot=200, seed=RS):
    """Bootstrap CI for AUC(C) − AUC(A) on the same rows."""
    rng = np.random.default_rng(seed)
    y, pa, pc = map(np.asarray, (y, pa, pc))
    gaps = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        gaps.append(roc_auc_score(y[i], pc[i]) - roc_auc_score(y[i], pa[i]))
    return (np.percentile(gaps, 2.5), np.percentile(gaps, 97.5)) if gaps else (np.nan, np.nan)


def run(spark, cfg: Tier2Config | None = None):
    cfg = cfg or Tier2Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    pdf = pull_panel(spark, cfg)
    rows, dep_feats, pkg_feats = build_features(pdf, cfg)
    all_feats = dep_feats + pkg_feats

    fi = _idx(cfg.freeze_train_end)
    train = rows[rows.month_idx <= fi]
    test = rows[rows.month_idx > fi].copy()
    print(f"[setup] train={len(train):,} test={len(test):,} "
          f"onset rate={test.y_onset.mean():.3%}")

    # global models (replicates Tier 2 freeze fit)
    mA, mB, mC = _hgb(), _hgb(), _hgb()
    mA.fit(train[dep_feats], train.y_onset)
    mB.fit(train[pkg_feats], train.y_onset)
    mC.fit(train[all_feats], train.y_onset)
    test["p_A"] = mA.predict_proba(test[dep_feats])[:, 1]
    test["p_B"] = mB.predict_proba(test[pkg_feats])[:, 1]
    test["p_C"] = mC.predict_proba(test[all_feats])[:, 1]

    out = []

    # ------------------------------------------------------------------ D1
    print("\n=== D1: visibility cut (global models) ===")
    subsets = {
        "full_vis (present_3m=1)": test.f_pkg_present_3m.eq(1.0),
        "partial (0<present_3m<1)": test.f_pkg_present_3m.gt(0) & test.f_pkg_present_3m.lt(1),
        "absent (present_3m=0)": test.f_pkg_present_3m.eq(0) | test.f_pkg_present_3m.isna(),
    }
    for name, m in subsets.items():
        s = test[m]
        aA, aB, aC = (_safe_auc(s.y_onset, s[c]) for c in ("p_A", "p_B", "p_C"))
        lo, hi = _boot_gap(s.y_onset, s.p_A, s.p_C) if len(s) > 200 else (np.nan, np.nan)
        print(f"  {name:28s} n={len(s):6,} base={s.y_onset.mean():6.2%} "
              f"AUC_A={aA:.3f} AUC_B={aB:.3f} AUC_C={aC:.3f} "
              f"gap={aC - aA:+.4f} [{lo:+.4f},{hi:+.4f}]")
        out.append(dict(diag="D1_global", subset=name, n=len(s),
                        base=s.y_onset.mean(), auc_A=aA, auc_B=aB, auc_C=aC,
                        gap=aC - aA, gap_lo=lo, gap_hi=hi))

    print("\n=== D1b: refit on fully-visible rows only ===")
    tr_v = train[train.f_pkg_present_3m.eq(1.0)]
    te_v = test[test.f_pkg_present_3m.eq(1.0)].copy()
    print(f"  visible train={len(tr_v):,} test={len(te_v):,} "
          f"base={te_v.y_onset.mean():.2%}")
    mA2, mC2 = _hgb(), _hgb()
    mA2.fit(tr_v[dep_feats], tr_v.y_onset)
    mC2.fit(tr_v[all_feats], tr_v.y_onset)
    pA2 = mA2.predict_proba(te_v[dep_feats])[:, 1]
    pC2 = mC2.predict_proba(te_v[all_feats])[:, 1]
    aA2, aC2 = _safe_auc(te_v.y_onset, pA2), _safe_auc(te_v.y_onset, pC2)
    lo, hi = _boot_gap(te_v.y_onset, pA2, pC2)
    print(f"  refit: AUC_A={aA2:.3f} AUC_C={aC2:.3f} gap={aC2 - aA2:+.4f} "
          f"[{lo:+.4f},{hi:+.4f}]")
    out.append(dict(diag="D1_refit_visible", subset="full_vis", n=len(te_v),
                    base=te_v.y_onset.mean(), auc_A=aA2, auc_B=np.nan,
                    auc_C=aC2, gap=aC2 - aA2, gap_lo=lo, gap_hi=hi))

    # ------------------------------------------------------------------ D2
    print("\n=== D2: surprise attrition (deposit-model blind spots) ===")
    for pool_name, q in [("bottom 50% by p_A", 0.50), ("bottom 80% by p_A", 0.80)]:
        thr = test.p_A.quantile(q)
        pool = test[test.p_A <= thr]
        n_onset = int(pool.y_onset.sum())
        aB = _safe_auc(pool.y_onset, pool.p_B)
        aC = _safe_auc(pool.y_onset, pool.p_C)
        aA = _safe_auc(pool.y_onset, pool.p_A)  # residual deposit ranking in-pool
        # lift of graph score inside the pool
        k = max(1, len(pool) // 10)
        top = pool.nlargest(k, "p_B")
        liftB = top.y_onset.mean() / max(pool.y_onset.mean(), 1e-9)
        print(f"  {pool_name:22s} n={len(pool):6,} onsets={n_onset:5,} "
              f"({pool.y_onset.mean():.2%})  in-pool AUC: A={aA:.3f} "
              f"B={aB:.3f} C={aC:.3f}  lift_top10(B)={liftB:.2f}")
        out.append(dict(diag="D2_pool", subset=pool_name, n=len(pool),
                        base=pool.y_onset.mean(), auc_A=aA, auc_B=aB,
                        auc_C=aC, gap=np.nan, gap_lo=np.nan, gap_hi=np.nan,
                        lift_B=liftB))

    # capture framing: onsets the deposit model misses at a top-decile alert budget
    thrA = test.p_A.quantile(0.90)
    missed = test[(test.p_A < thrA) & (test.y_onset == 1)]
    remaining = test[test.p_A < thrA]
    n_flag = max(1, len(remaining) // 10)
    graph_flags = remaining.nlargest(n_flag, "p_B")
    caught = int(graph_flags.y_onset.sum())
    expect = n_flag * remaining.y_onset.mean()
    print(f"\n  deposit top-decile alerting misses {len(missed):,} onsets; "
          f"a second-stage graph screen flagging 10% of the remainder "
          f"catches {caught:,} of them vs {expect:.0f} expected at random "
          f"({caught / max(expect, 1e-9):.2f}x)")
    out.append(dict(diag="D2_capture", subset="2nd_stage_graph_screen",
                    n=len(remaining), base=remaining.y_onset.mean(),
                    auc_A=np.nan, auc_B=np.nan, auc_C=np.nan, gap=np.nan,
                    gap_lo=np.nan, gap_hi=np.nan,
                    caught=caught, expected=expect))

    summary = pd.DataFrame(out)
    summary.to_csv(f"{cfg.out_dir}/diagnostics_summary.csv", index=False)
    print(f"\n[done] {cfg.out_dir}/diagnostics_summary.csv")
    return summary


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    run(SparkSession.builder.enableHiveSupport().getOrCreate())
