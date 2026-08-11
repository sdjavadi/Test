"""
fix_ladder_tiers.py
===================
One-off cleanup: remove a retired ablation tier from the artifacts already
written to ../metrics/, so the reuse path in pkg_pipeline.py stops
resurrecting it and completed month files stop carrying its rows.

Why this is needed
------------------
build_ladder()'s default is (99.0, 99.9) — the Python never asked for
P99_99. It came back because REUSE_LADDER=True and load_ladder() rebuilds
whatever tiers ladder_exclusions.csv happens to carry. The pipeline change
(LADDER_PCTS + reconcile) fixes the run; this fixes the files.

What it touches
---------------
  ladder_exclusions.csv   drop in_<TIER> column
  ladder_thresholds.csv   drop degree_<TIER>, strength_<TIER>
  node|graph|dist|community|roles/*.parquet|csv
                          drop rows where version == <TIER>
  state/trackers.pkl      drop the <TIER> tracker (dead weight; the run
                          loop only iterates the live versions, so this is
                          hygiene, not a correctness fix)

P99 and P99_9 are preserved bit-for-bit — nothing is recomputed. Month
files keep their completed rows, so the resume guard
(node_<month>.parquet exists -> skip) stays satisfied and no month is
recomputed either.

Usage
-----
    python fix_ladder_tiers.py                  # dry run, prints the plan
    python fix_ladder_tiers.py --apply
    python fix_ladder_tiers.py --tier P99_99 --out-dir ../metrics --apply
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

import pandas as pd

METRIC_DIRS = ("node", "graph", "dist", "community", "roles")
LADDER_EX = "ladder_exclusions.csv"
LADDER_TH = "ladder_thresholds.csv"
TRACKERS = os.path.join("state", "trackers.pkl")


def _backup(path: str, apply: bool) -> None:
    """Copy to <path>.bak once. Never overwrite an existing backup — the
    first one is the pre-cleanup state and is the one worth keeping."""
    bak = path + ".bak"
    if apply and not os.path.exists(bak):
        shutil.copy2(path, bak)


def clean_ladder_csvs(out_dir: str, tier: str, apply: bool) -> None:
    ex_path = os.path.join(out_dir, LADDER_EX)
    th_path = os.path.join(out_dir, LADDER_TH)

    if os.path.exists(ex_path):
        ex = pd.read_csv(ex_path, dtype={"node": str})
        drop = [c for c in ex.columns if c == f"in_{tier}"]
        if drop:
            print(f"  {LADDER_EX}: drop {drop} "
                  f"({len(ex):,} rows kept, node column = broadest tier)")
            if apply:
                _backup(ex_path, apply)
                ex.drop(columns=drop).to_csv(ex_path, index=False)
        else:
            print(f"  {LADDER_EX}: no in_{tier} column — already clean")
        remaining = [c[3:] for c in ex.columns if c.startswith("in_")
                     and c != f"in_{tier}"]
        print(f"    tiers after cleanup: broadest + {remaining}")
    else:
        print(f"  {LADDER_EX}: NOT FOUND in {out_dir}")

    if os.path.exists(th_path):
        th = pd.read_csv(th_path)
        drop = [c for c in th.columns
                if c in (f"degree_{tier}", f"strength_{tier}")]
        if drop:
            print(f"  {LADDER_TH}: drop {drop}")
            if apply:
                _backup(th_path, apply)
                th.drop(columns=drop).to_csv(th_path, index=False)
        else:
            print(f"  {LADDER_TH}: no {tier} thresholds — already clean")
    else:
        print(f"  {LADDER_TH}: NOT FOUND in {out_dir}")


def _read_any(path: str) -> pd.DataFrame:
    return (pd.read_parquet(path) if path.endswith(".parquet")
            else pd.read_csv(path))


def _write_any(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def clean_month_outputs(out_dir: str, tier: str, apply: bool) -> None:
    total_rows = 0
    total_files = 0
    for sub in METRIC_DIRS:
        d = os.path.join(out_dir, sub)
        if not os.path.isdir(d):
            continue
        hits = []
        for fn in sorted(os.listdir(d)):
            if not fn.endswith((".parquet", ".csv")):
                continue
            p = os.path.join(d, fn)
            try:
                df = _read_any(p)
            except Exception as exc:            # noqa: BLE001
                print(f"  {sub}/{fn}: unreadable ({exc}) — skipped")
                continue
            if "version" not in df.columns:
                continue
            n = int((df["version"].astype(str) == tier).sum())
            if not n:
                continue
            hits.append((fn, n, len(df)))
            total_rows += n
            total_files += 1
            if apply:
                _backup(p, apply)
                _write_any(df[df["version"].astype(str) != tier]
                           .reset_index(drop=True), p)
        if hits:
            print(f"  {sub}/: {len(hits)} file(s) carry {tier}")
            for fn, n, tot in hits:
                print(f"    {fn}: {n:,} of {tot:,} rows")
        else:
            print(f"  {sub}/: clean")
    print(f"  -> {total_rows:,} rows across {total_files} file(s)")


def clean_trackers(out_dir: str, tier: str, apply: bool) -> None:
    p = os.path.join(out_dir, TRACKERS)
    if not os.path.exists(p):
        print("  trackers.pkl: not found — nothing to do")
        return
    try:
        with open(p, "rb") as fh:
            st = pickle.load(fh)
    except Exception as exc:                    # noqa: BLE001
        # Expected when run standalone: the pickle holds TemporalTracker
        # instances, so unpickling needs that class importable under the
        # same module name it was pickled from. Harmless to skip — the
        # pipeline loads this file itself (where the class IS available)
        # and only iterates the live versions, so a leftover tracker is
        # dead weight, not a correctness problem.
        print(f"  trackers.pkl: not unpickleable here ({exc}) — skipped. "
              f"Harmless: the run loop only iterates live versions.")
        return
    tr = st.get("trackers", {})
    if tier not in tr:
        print(f"  trackers.pkl: no {tier} tracker — already clean")
        return
    print(f"  trackers.pkl: drop {tier} tracker "
          f"(keys now {sorted(k for k in tr if k != tier)})")
    if apply:
        _backup(p, apply)
        tr.pop(tier, None)
        st.get("prev_partition", {}).pop(tier, None)
        with open(p + ".tmp", "wb") as fh:
            pickle.dump(st, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(p + ".tmp", p)   # atomic, same policy as _save_trackers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="../metrics")
    ap.add_argument("--tier", default="P99_99",
                    help="tier label to remove (default: P99_99)")
    ap.add_argument("--apply", action="store_true",
                    help="write changes; without it this is a dry run")
    ap.add_argument("--skip-months", action="store_true",
                    help="only clean the ladder CSVs and trackers")
    a = ap.parse_args()

    if not os.path.isdir(a.out_dir):
        print(f"out-dir not found: {a.out_dir}")
        return 1

    mode = "APPLY" if a.apply else "DRY RUN (nothing written)"
    print(f"=== removing tier {a.tier} from {a.out_dir} — {mode} ===")
    print("\n[1/3] ladder artifacts")
    clean_ladder_csvs(a.out_dir, a.tier, a.apply)
    if a.skip_months:
        print("\n[2/3] month outputs: skipped (--skip-months)")
    else:
        print("\n[2/3] month outputs")
        clean_month_outputs(a.out_dir, a.tier, a.apply)
    print("\n[3/3] tracker checkpoint")
    clean_trackers(a.out_dir, a.tier, a.apply)

    if not a.apply:
        print("\nDry run only. Re-run with --apply to write. "
              "Originals are backed up to <file>.bak on first write.")
    else:
        print("\nDone. Next run should log: graph versions: "
              "['V0', 'P99', 'P99_9']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
