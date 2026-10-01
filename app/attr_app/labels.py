"""Plain-language names, display units and suggested conversations."""
import re
import numpy as np
import pandas as pd

BASE = {
    "active_days_in": "days with money coming in", "active_days_out": "days with money going out",
    "ntxn_all_in": "incoming payments", "ntxn_all_out": "outgoing payments",
    "ntxn_ext_in": "incoming payments from other banks' clients", "ntxn_ext_out": "payments to other banks' clients",
    "ntxn_int_in": "incoming payments from other PNC clients", "ntxn_int_out": "payments to other PNC clients",
    "amt_all_in": "dollars coming in", "amt_all_out": "dollars going out", "amt_ext_in": "dollars from other banks' clients",
    "amt_ext_out": "dollars to other banks' clients", "amt_int_in": "dollars from other PNC clients", "amt_int_out": "dollars to other PNC clients",
    "ncpty_all_in": "number of payers", "ncpty_all_out": "number of payees", "ncpty_ext_in": "payers at other banks",
    "ncpty_ext_out": "payees at other banks", "ncpty_int_in": "payers who bank at PNC", "ncpty_int_out": "payees who bank at PNC",
    "n_fi_out": "banks it pays into", "n_fi_in": "banks it receives from", "n_fi_ach_out": "banks it pays into by ACH",
    "n_fi_ach_in": "banks it receives from by ACH", "net_flow_all": "net flow (in minus out)", "net_flow_ext": "net flow with other banks' clients",
    "passthrough_ratio": "share of incoming money paid straight out", "selfpay_share_out": "share of outgoing money sent to its own accounts elsewhere",
    "selfpay_amt_out": "money sent to its own accounts at other banks", "avg_ticket_in": "average incoming payment",
    "avg_ticket_out": "average outgoing payment", "share_WIRE_in": "share received by wire", "share_CHECK_in": "share received by check",
    "share_ACH_in": "share received by ACH", "share_WIRE_out": "share paid by wire", "share_CHECK_out": "share paid by check",
    "share_ACH_out": "share paid by ACH", "rec_broken_in": "regular payers who skipped this month", "rec_broken_out": "regular payees skipped",
    "rec_broken_share_out": "share of regular payments that stopped", "rec_standing_out": "regular payees",
    "top1_share_in": "share of receipts from the largest payer", "hhi_in": "concentration of receipts",
    "new_cpty_ext_in": "new payers at other banks", "lost_cpty_ext_in": "payers lost", "reactivated_cpty_ext_in": "returning payers",
    "fi_rec_broken_out": "regular banking relationships that stopped", "pay_log_flow": "overall payment activity",
    "dep_log_bal": "balance", "dep_log_norm": "normal balance", "dep_bal_vs_norm": "balance vs normal",
    "dep_d1": "balance change vs last month", "dep_d3": "balance change vs the last 3 months", "dep_d6": "balance change vs 4–6 months earlier",
    "dep_d12": "balance change vs a year earlier", "dep_drawdown": "balance vs its 12-month peak", "dep_vol6": "balance volatility",
    "dep_range": "swing in daily balance within the month", "dep_eom_vs_avg": "month-end balance vs monthly average",
    "dep_siblings": "other entities in the relationship", "dep_sib_d6": "balances of other entities in the relationship",
    "dep_live_d6": "live accounts (change)", "dep_n_accts": "number of accounts", "dep_n_live": "live accounts", "dep_tenure": "time as a client",
    "dep_at_start": "long-standing client", "dep_earnings_credit_rate": "earnings credit rate", "dep_curr_int_rate": "interest rate paid",
    "dep_closed_3m": "accounts closed in the last 3 months", "dep_opened_3m": "accounts opened in the last 3 months",
}
SUFFIX = {"d6": "vs 4–6 months earlier", "d3": "vs the last 3 months", "peer": "vs similar-sized clients"}
LATE = {"dep_eom_vs_avg", "dep_d1"}


def plain(f, fallback=None):
    base, _, tr = str(f).partition("__")
    b = BASE.get(base) or re.sub(r"\s*\(.*?\)", "", str(fallback or base.replace("_", " "))).split(" — ")[0].strip()
    return b + (f" ({SUFFIX[tr]})" if tr in SUFFIX else "")


def is_ratio(base):
    return bool(re.search(r"(share|hhi|retention|entropy|coverage|ratio)", base)) and base != "dep_bal_vs_norm"


def to_display(var, v):
    """Model units → RM units: % change for log ratios, percentage points for shares."""
    if v is None or pd.isna(v): return np.nan
    base, _, tr = str(var).partition("__")
    if base.startswith("dep_live") or base.startswith("net_flow") or base in ("dep_siblings", "dep_n_accts", "dep_n_live", "dep_tenure", "dep_at_start"):
        return float(v)
    if is_ratio(base): return float(v) * 100
    if base.startswith("dep_log") or base == "pay_log_flow": return float(np.expm1(v))
    return float(np.expm1(v) * 100)


def fmt_value(var, v):
    x = to_display(var, v)
    if pd.isna(x): return "—"
    base = str(var).split("__")[0]
    if base.startswith("dep_log") or base == "pay_log_flow": return money(x)
    if base.startswith("net_flow") or base in ("dep_siblings", "dep_n_accts", "dep_n_live", "dep_tenure", "dep_at_start", "dep_live_d6"):
        return f"{x:+.2f}" if base.startswith(("net_flow", "dep_live")) else f"{x:.0f}"
    if is_ratio(base): return f"{x:+.0f} pts" if "__" in str(var) else f"{x:.0f}%"
    return f"{x:+.0f}%"


def money(v, d=1):
    if v is None or pd.isna(v): return "—"
    s = "−" if v < 0 else ""; v = abs(float(v))
    return (f"{s}${v / 1e9:,.{d}f}bn" if v >= 1e9 else f"{s}${v / 1e6:,.{d}f}m" if v >= 1e6
            else f"{s}${v / 1e3:,.0f}k" if v >= 1e3 else f"{s}${v:,.0f}")


PLAYS = {
    "relationship": ("Relationship risk", "Other entities in this relationship are already moving money. Talk to the relationship's decision maker, not only this entity."),
    "competitive": ("Competitive threat", "Money appears to be moving to another bank. Compare pricing, earnings credit and services with the likely competitor; prepare a retention offer."),
    "receipts": ("Receipts drying up", "Its own customers are paying it less, or paying into another bank. Ask about collections, lockbox and receivables arrangements."),
    "drawdown": ("Balance drawdown", "Deposits are falling. Ask about cash plans; offer sweep, money-market or liquidity options to keep the operating balance here."),
    "activity": ("Business slowing", "Fewer payments and trading partners. Check on the business; offer credit, working capital or payment products."),
}


def theme(var):
    v = str(var)
    if v.startswith("dep_sib"): return "relationship"
    if re.search(r"(n_fi|^fi_|selfpay|top_fi|share_WIRE|share_ACH|share_CHECK|n_rails)", v): return "competitive"
    if re.search(r"(_in$|_in__|active_days_in|ncpty_.*_in)", v): return "receipts"
    if v.startswith("dep_"): return "drawdown"
    return "activity"
