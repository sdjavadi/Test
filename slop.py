import numpy as np
import pandas as pd

# ----------------------------
# INPUTS / ASSUMPTIONS
# ----------------------------
# df columns (adjust names if needed):
#   'customer_name' : str (or 'customer_id')
#   'date'          : datetime64[ns]  (daily or monthly)
#   'total_amount'  : float
#   'total_number'  : float/int
#   'unique_counterparties' : float/int (optional)

METRICS = ["total_amount", "total_number", "unique_counterparties"]  # pick what you have
WINDOW_MONTHS = 6                      # last N months window for regression
MIN_POINTS = 4                         # minimum # data points to run regression
FILL_MISSING = True                    # if True, reindex months and fill missing with 0

# ----------------------------
# HELPER: monthly aggregate per customer (if your df is daily)
# ----------------------------
def to_monthly(df):
    out = df.copy()
    out["month"] = out["date"].values.astype("datetime64[M]")  # month start
    gb = out.groupby(["customer_name", "month"], as_index=False)
    # Sum metrics by month (change to .mean() if that’s what you plot)
    agg_dict = {m: "sum" for m in METRICS if m in out.columns}
    out = gb.agg(agg_dict)
    return out.rename(columns={"month": "date"})

# If your data is already monthly, comment the next line.
dfm = to_monthly(df)

# ----------------------------
# HELPER: fit OLS slope, intercept, R^2
# x is integers 0..n-1 (months), y is metric series
# ----------------------------
def ols_slope_r2(y: np.ndarray):
    """
    Returns slope, intercept, r2 for y ~ a + b*x with x = 0..n-1.
    NaN-safe: returns (np.nan, np.nan, np.nan) if not enough points or var(y)==0.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 or np.allclose(y, y[0]):  # not enough variation
        return np.nan, np.nan, np.nan

    x = np.arange(n, dtype=float)
    # closed-form OLS using polyfit for numerical stability
    b, a = np.polyfit(x, y, deg=1)  # y ~ a + b*x
    yhat = a + b*x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return b, a, r2

# ----------------------------
# CORE: compute slope/R² per customer & metric using last N months
# ----------------------------
def last_n_months_indexed(s, n, fill_missing):
    """Return a slice of the last N calendar months, optionally reindexed to complete months."""
    s = s.sort_values()
    if s.empty:
        return s

    # Build full month index between min and max appearance in the last N months span
    end = s.max()
    start = (end.to_period("M") - (n - 1)).to_timestamp()  # month start N-1 back
    if fill_missing:
        full_idx = pd.date_range(start, end, freq="MS")  # month starts
        return full_idx
    else:
        # Keep only observed months within window
        return s[(s >= start) & (s <= end)]

def regress_per_customer(group: pd.DataFrame) -> pd.Series:
    # restrict to last N months (by calendar months)
    months_idx = last_n_months_indexed(group["date"], WINDOW_MONTHS, FILL_MISSING)
    # align group to those months
    sub = group.set_index("date")
    if FILL_MISSING:
        sub = sub.reindex(months_idx, fill_value=0.0)
    else:
        sub = sub.loc[sub.index.isin(months_idx)]

    out = {}
    for col in METRICS:
        if col not in sub.columns:
            continue
        y = sub[col].astype(float).values
        if len(y) >= MIN_POINTS and not np.allclose(y, y[0]):
            slope, intercept, r2 = ols_slope_r2(y)
            delta = float(y[-1] - y[0])  # first–last sanity check
        else:
            slope, intercept, r2, delta = (np.nan, np.nan, np.nan, np.nan)

        base = col
        out[f"slope_{base}"] = slope
        out[f"r2_{base}"] = r2
        out[f"delta_{base}"] = delta
        out[f"last_{base}"] = float(y[-1]) if len(y) else np.nan

    # Optional: a simple composite for “steady growth”
    # (tune weights/thresholds later)
    if "slope_total_amount" in out and "r2_total_amount" in out:
        out["steady_growth_score_amount"] = (
            np.nan_to_num(out["slope_total_amount"], nan=0.0)
            * np.nan_to_num(out["r2_total_amount"], nan=0.0)
        )
    return pd.Series(out)

summary = (
    dfm
    .sort_values(["customer_name", "date"])
    .groupby("customer_name", group_keys=False)
    .apply(regress_per_customer)
    .reset_index()
)

# ----------------------------
# EXAMPLE: set a trigger
# ----------------------------
# You’ll probably choose segment-specific thresholds.
# Here is a simple illustrative rule for amount:
SLOPE_THRESH = summary["slope_total_amount"].quantile(0.9)   # top 10% slope
R2_MIN = 0.70

summary["flag_suspicious_amount"] = (
    (summary["slope_total_amount"] >= SLOPE_THRESH) &
    (summary["r2_total_amount"] >= R2_MIN)
)

# Sort by a composite to review
summary = summary.sort_values("steady_growth_score_amount", ascending=False)
summary.head(20)
