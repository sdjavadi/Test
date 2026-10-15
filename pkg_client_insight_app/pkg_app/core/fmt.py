"""Formatting helpers shared by every page."""
import re
import numpy as np
import pandas as pd


def money(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v): return "—"
    s = "−" if v < 0 else ""; v = abs(float(v))
    return (f"{s}${v / 1e9:,.{d}f}bn" if v >= 1e9 else f"{s}${v / 1e6:,.{d}f}m" if v >= 1e6
            else f"{s}${v / 1e3:,.0f}k" if v >= 1e3 else f"{s}${v:,.0f}")


def pct(v, d=0):
    return "—" if v is None or pd.isna(v) else f"{v:.{d}%}"


def num(v, d=0):
    return "—" if v is None or pd.isna(v) else f"{v:,.{d}f}"


def md2html(t):
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)


def norm_pwr(x):
    """Comparable power id: string, trimmed, leading zeros removed (deposit ids are zero-padded, metrics ids may not be)."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"): return None
    s = s.lstrip("0")
    return s or "0"


def cust_key(pwr=None, mdm_id=None):
    """One key per customer across modules: the power id when there is one, else the payment entity id."""
    p = norm_pwr(pwr)
    return p if p else (f"M:{mdm_id}" if mdm_id else None)


def display_name(name, key):
    if isinstance(name, str) and name.strip(): return name.strip()
    k = str(key or "")
    return f"Entity {k[2:]}" if k.startswith("M:") else f"Client {k}"


def safe_int(v, default=0):
    try: return int(v)
    except Exception: return default
