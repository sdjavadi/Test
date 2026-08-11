"""
pkg_runtime.py
==============
Timing instrumentation and session-budget tracking for the monthly run.

Purpose: the GPU session is capped (8h). You need to know by ~45 minutes in
whether the run will finish, not at hour 7. After the third month the
projector prints a total estimate and an ETA against the budget, and warns
loudly if the projection exceeds it.

Every component records one structured line:

    TIMING 2024-01 V0        trophic            152.3s   n=2068558

and the whole set is written to ../metrics/run_timings.csv at the end
(and after every month, so a killed session still leaves the profile).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

import pandas as pd

log = logging.getLogger("pkg_runtime")

_ROWS: list[dict] = []
_T0 = time.time()
_BUDGET_S = 8 * 3600.0
_OUT = None


def configure(budget_hours: float = 8.0, out_path: str | None = None,
              quiet_libs: bool = True):
    """Set the session budget and where the timing profile is written."""
    global _BUDGET_S, _OUT, _T0
    _BUDGET_S = float(budget_hours) * 3600.0
    _OUT = out_path
    _T0 = time.time()
    if quiet_libs:
        # cuGraph/numba/numexpr emit per-call INFO chatter that buries the
        # pipeline's own lines. Warnings and errors still come through.
        for name in ("numba", "numba.core", "numba.cuda", "cudf", "cugraph",
                     "rmm", "numexpr", "fsspec", "pyarrow"):
            logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def step(month: str, version: str, name: str, n: int | None = None):
    """Time one component. Logs on exit, including on exception."""
    t = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        dt = time.perf_counter() - t
        _ROWS.append({"time_key": month, "version": version, "component": name,
                      "secs": round(dt, 2), "n": n, "ok": ok})
        log.info("TIMING %-8s %-7s %-22s %8.1fs%s%s",
                 month, version, name, dt,
                 f"  n={n:,}" if n is not None else "",
                 "" if ok else "  [FAILED]")


def month_done(month: str, n_done: int, n_total: int):
    """Log a per-month summary and project the finish against the budget."""
    el = time.time() - _T0
    per = el / max(n_done, 1)
    proj = per * n_total
    remain = proj - el
    df = pd.DataFrame([r for r in _ROWS if r["time_key"] == month])
    top = ""
    if len(df):
        t = (df.groupby("component")["secs"].sum()
             .sort_values(ascending=False).head(3))
        top = " | slowest: " + ", ".join(f"{k} {v:.0f}s" for k, v in t.items())
    log.info("MONTH %s done (%d/%d) | elapsed %.2fh | %.1f min/month | "
             "projected total %.2fh | remaining %.2fh%s",
             month, n_done, n_total, el / 3600, per / 60,
             proj / 3600, max(remain, 0) / 3600, top)
    if proj > _BUDGET_S:
        log.warning("PROJECTION EXCEEDS BUDGET: %.2fh projected vs %.2fh "
                    "available. Months already written are checkpointed and "
                    "will be skipped on restart — consider killing now and "
                    "reducing scope (BETWEENNESS_K=0, or fewer versions).",
                    proj / 3600, _BUDGET_S / 3600)
    flush()


def flush():
    if _OUT and _ROWS:
        os.makedirs(os.path.dirname(_OUT) or ".", exist_ok=True)
        pd.DataFrame(_ROWS).to_csv(_OUT, index=False)


def summary() -> pd.DataFrame:
    """Component cost profile across the whole run."""
    if not _ROWS:
        return pd.DataFrame()
    df = pd.DataFrame(_ROWS)
    s = (df.groupby("component")["secs"]
         .agg(total="sum", mean="mean", n="size")
         .sort_values("total", ascending=False))
    s["pct"] = (100 * s["total"] / s["total"].sum()).round(1)
    return s.reset_index()


def log_summary():
    s = summary()
    if s.empty:
        return
    log.info("=== component cost profile (total %.2fh) ===",
             s["total"].sum() / 3600)
    for _, r in s.iterrows():
        log.info("  %-22s %8.1fs  %5.1f%%  (%d calls, %.1fs avg)",
                 r["component"], r["total"], r["pct"], r["n"], r["mean"])
