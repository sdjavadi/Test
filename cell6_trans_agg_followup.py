# ============================================================
# 6. Transaction aggregate follow-up: history, fit with PKG customers,
#    overlap with PKG payment dollars, RM history
# ============================================================
# Read-only. Works on the table in TABLE (cdm_tm_cor_trans_agg_v). Writes two small CSVs to OUT_DIR.
import os
from functools import reduce

ta = spark.table(TABLE)
_need = ["time_key", "acct_dim_key", "cust_pwr_id", "rm_empl_dim_key", "curnt_rm_empl_dim_key", "tmo_empl_dim_key"]
_miss = [c for c in _need if c not in ta.columns]
if _miss:
    raise RuntimeError(f"{TABLE} lacks {_miss}: this cell is written for cdm_tm_cor_trans_agg_v")

def _num(c):
    return F.coalesce(F.col(c).cast("double"), F.lit(0.0))

def _sum_of(cols):
    cols = [c for c in cols if c in ta.columns]
    return reduce(lambda a, b: a + b, [_num(c) for c in cols]) if cols else F.lit(0.0)

def _pid(c):
    # power id as text without leading zeros; '-1' / blank = missing (same rule as cell 5)
    s = F.regexp_replace(F.trim(F.col(c).cast("string")), r"^0+", "")
    return F.when(s.isNull() | (s == "") | (s == "-1"), F.lit(None)).otherwise(s)

def _pct(x):
    return "" if x is None or pd.isna(x) else f"{100 * float(x):.1f}%"

def _fmt(pdf, pct=(), ints=(), dec1=()):
    out = pdf.copy()
    for c in out.columns:
        if c in pct:
            out[c] = out[c].map(_pct)
        elif c in ints:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{int(round(v)):,}")
        elif c in dec1:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}")
    return out

# Direction from the customer's side (assumed from column names; ACH 'orig' = entries the customer sends:
# credits originated = money out, debits originated = collections in; e-check originated = converted checks collected)
RAILS = {
    "ACH":               (["ach_credits_recd_amt", "ach_debits_orig_amt"], ["ach_credits_orig_amt", "ach_debits_recd_amt"]),
    "Wire":              (["dom_wire_incoming_amt", "intl_wire_incoming_amt"], ["dom_wire_outgoing_amt", "intl_wire_outgoing_amt"]),
    "RTP":               (["realtime_pymt_credit_amt"], ["realtime_pymt_debit_amt"]),
    "Other electronic":  (["elctr_pymt_credit_amt", "elctr_check_orig_amt"], ["elctr_pymt_debit_amt", "elctr_check_rcvd_amt"]),
    "Merchant services": (["mrch_srvcs_credit_amt"], ["mrch_srvcs_debit_amt"]),
    "Checks":            (["check_dep_lockbox_amt", "check_dep_remote_amt"], ["check_disbursement_amt"]),
    "Branch deposits":   (["branch_dep_amt"], []),
    "Cash logistics":    (["cash_dep_logistics_amt", "cash_dep_smart_safe_amt"], ["cash_wdrw_logistics_amt"]),
}
IN_COLS  = [c for r in RAILS.values() for c in r[0]]
OUT_COLS = [c for r in RAILS.values() for c in r[1]]
CASH = [c for c in ["branch_dep_amt", "cash_dep_logistics_amt", "cash_dep_smart_safe_amt", "cash_wdrw_logistics_amt"] if c in ta.columns]
short = lambda c: c.replace("_amt", "").replace("cash_dep_", "cash_").replace("cash_wdrw_", "wdrw_")

parts = ["<h3>6. Transaction aggregate follow-up</h3>"]

# ---- A) one row per month over the whole history ---------------------------------------------
agg = [F.count("*").alias("accounts"),
       F.approx_count_distinct("cust_pwr_id", 0.02).alias("customers"),
       (F.sum(_sum_of(IN_COLS)) / 1e9).alias("in B"),
       (F.sum(_sum_of(OUT_COLS)) / 1e9).alias("out B")]
agg += [(F.sum(_num(c)) / 1e6).alias(short(c) + " M") for c in CASH]
agg += [F.sum((_num(c) != 0).cast("int")).alias(short(c) + " accts") for c in CASH]
if "bank_fee_debit_amt" in ta.columns:
    agg.append((F.sum(_num("bank_fee_debit_amt")) / 1e6).alias("fees M"))
agg += [F.avg((F.col("rm_empl_dim_key") > 0).cast("double")).alias("rm set"),
        F.avg((F.col("rm_empl_dim_key") != F.col("curnt_rm_empl_dim_key")).cast("double")).alias("rm vs curnt differ"),
        F.avg((F.col("tmo_empl_dim_key") > 0).cast("double")).alias("tmo set")]
hist = ta.groupBy("time_key").agg(*agg).orderBy("time_key").toPandas()
hist["month"] = pd.to_datetime(hist["time_key"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m")
hist = hist[["month"] + [c for c in hist.columns if c not in ("month", "time_key")] + ["time_key"]]
os.makedirs(OUT_DIR, exist_ok=True)
hist.to_csv(os.path.join(OUT_DIR, "history_by_month.csv"), index=False)

mp = pd.PeriodIndex(hist["month"], freq="M")
gaps = [str(p) for p in pd.period_range(mp.min(), mp.max(), freq="M") if p not in set(mp)]
parts.append(f"<p><b>History:</b> {len(hist)} months, {hist['month'].iloc[0]} to {hist['month'].iloc[-1]}; "
             f"missing months: {esc(', '.join(gaps)) if gaps else 'none'}. "
             f"B = billions, M = millions of dollars. in / out = all rails below, customer's side.</p>")

PCT = ("rm set", "rm vs curnt differ", "tmo set")
INTS = ["accounts", "customers"] + [c for c in hist.columns if c.endswith(" accts")]
DEC = [c for c in hist.columns if c.endswith(" B") or c.endswith(" M")]
yr = hist.groupby(hist["month"].str[:4]).tail(1)
parts += ["<p class='note'>One row per year (December, or the latest month of the year)</p>",
          tbl(_fmt(yr.drop(columns="time_key"), PCT, INTS, DEC)),
          "<p class='note'>Last 13 months</p>",
          tbl(_fmt(hist.tail(13).drop(columns="time_key"), PCT, INTS, DEC))]

# month-over-month jumps that may be definition or load changes
jumps = []
for c in ["accounts", "customers", "in B", "out B"] + [c for c in hist.columns if c.endswith(" M")]:
    v = hist[c].astype(float).values
    for i in range(1, len(v)):
        a, b = v[i - 1], v[i]
        if abs(a) > 0 and abs(b) > 0 and (b / a > 1.5 or b / a < 1 / 1.5):
            jumps.append((hist["month"].iloc[i], c, a, b, b / a))
        elif (a == 0) != (b == 0):
            jumps.append((hist["month"].iloc[i], c, a, b, None))
jp = pd.DataFrame(jumps, columns=["month", "measure", "previous", "this month", "ratio"])
parts.append(f"<p class='note'>Month-over-month jumps above 1.5x or below 0.67x, or to / from zero ({len(jp)}; first 60)</p>")
parts.append(tbl(jp.assign(previous=jp["previous"].map(lambda v: f"{v:,.2f}"),
                           **{"this month": jp["this month"].map(lambda v: f"{v:,.2f}"),
                              "ratio": jp["ratio"].map(lambda v: "" if v is None or pd.isna(v) else f"{v:.2f}")}),
                 max_rows=60))

# ---- B) latest month: fit with PKG customers ----------------------------------------------------
LATEST = int(hist["time_key"].max())
lat = ta.filter(F.col("time_key") == LATEST)
cust = (lat.select(_pid("cust_pwr_id").alias("pid"), _sum_of(IN_COLS).alias("v_in"), _sum_of(OUT_COLS).alias("v_out"),
                   _sum_of(CASH).alias("v_cash"))
           .filter("pid is not null").groupBy("pid")
           .agg(F.count("*").alias("accts"), F.sum("v_in").alias("v_in"), F.sum("v_out").alias("v_out"),
                F.sum("v_cash").alias("v_cash"))
           .persist(StorageLevel.MEMORY_AND_DISK))
dm = (spark.table(PKG_DIM).select(_pid("customer_pwr_id").alias("pid")).filter("pid is not null").distinct()
           .withColumn("in_pkg", F.lit(1)))
fit = (cust.join(dm, "pid", "left")
           .agg(F.count("*").alias("n"), F.sum(F.coalesce("in_pkg", F.lit(0))).alias("n_pkg"),
                F.sum(F.col("v_in") + F.col("v_out")).alias("usd"),
                F.sum(F.when(F.col("in_pkg") == 1, F.col("v_in") + F.col("v_out"))).alias("usd_pkg"),
                F.sum("v_cash").alias("cash"), F.sum(F.when(F.col("in_pkg") == 1, F.col("v_cash"))).alias("cash_pkg"),
                F.expr("percentile_approx(accts, 0.5)").alias("acc_med"),
                F.expr("percentile_approx(accts, 0.99)").alias("acc_p99"), F.max("accts").alias("acc_max"))
           .collect()[0])
n_dim = dm.count()
sh = lambda a, b: _pct(a / b) if b else ""
fit_tbl = pd.DataFrame([
    ("Customers (power ids) in the table", f"{fit['n']:,}", ""),
    ("... of which are PKG customers", f"{fit['n_pkg']:,}", sh(fit["n_pkg"], fit["n"])),
    ("PKG customers found in the table", f"{fit['n_pkg']:,} of {n_dim:,}", sh(fit["n_pkg"], n_dim)),
    ("Share of in + out dollars from PKG customers", "", sh(fit["usd_pkg"] or 0, fit["usd"] or 0)),
    ("Share of cash dollars (branch, logistics, safe) from PKG customers", "", sh(fit["cash_pkg"] or 0, fit["cash"] or 0)),
    ("Accounts per customer: median / p99 / max", f"{fit['acc_med']} / {fit['acc_p99']} / {fit['acc_max']}", ""),
], columns=["check", "count", "share"])
parts += [f"<p><b>Fit with PKG customers</b> ({PKG_DIM}, month {LATEST})</p>", tbl(fit_tbl)]

codes = (lat.select(F.when(F.col("rm_empl_dim_key") > 0, "real key").otherwise(F.col("rm_empl_dim_key").cast("string")).alias("rm"),
                    F.when(F.col("tmo_empl_dim_key") > 0, "real key").otherwise(F.col("tmo_empl_dim_key").cast("string")).alias("tmo"))
            .groupBy("rm", "tmo").count().orderBy(F.desc("count")).toPandas())
codes["share"] = (codes["count"] / codes["count"].sum()).map(_pct)
codes["count"] = codes["count"].map(lambda v: f"{v:,}")
parts += ["<p class='note'>RM / TMO key values in the latest month (negative = special codes, meaning set by the employee dimension)</p>",
          tbl(codes, max_rows=20)]

# ---- C) overlap with PKG payment dollars (amt_all_in / amt_all_out) ------------------------------
try:
    _pp = spark.sql(f"SHOW PARTITIONS {PKG_MET}").toPandas().iloc[:, 0].astype(str)
    pkg_months = sorted(v.split("month=", 1)[1].split("/")[0] for v in _pp if "month=" in v)
except Exception:
    pkg_months = sorted(r[0] for r in spark.table(PKG_MET).select("month").distinct().collect())
common = sorted(set(pkg_months) & set(hist["month"]))
if not common:
    parts.append(f"<p class='warn'>No month in both {esc(PKG_MET)} and {esc(TABLE)}: overlap check skipped.</p>")
else:
    cm = common[-1]
    tk = int(hist.loc[hist["month"] == cm, "time_key"].iloc[0])
    sel = [_pid("cust_pwr_id").alias("pid")]
    for r, (ins, outs) in RAILS.items():
        sel += [_sum_of(ins).alias(f"{r}|in"), _sum_of(outs).alias(f"{r}|out")]
    tv = (ta.filter(F.col("time_key") == tk).select(*sel).filter("pid is not null")
            .groupBy("pid").agg(*[F.sum(f"`{r}|{d}`").alias(f"{r}|{d}") for r in RAILS for d in ("in", "out")]))
    mt = (spark.table(PKG_MET).filter(F.col("month") == cm)
            .select(_pid("customer_pwr_id").alias("pid"),
                    F.coalesce(F.col("amt_all_in").cast("double"), F.lit(0.0)).alias("pkg_in"),
                    F.coalesce(F.col("amt_all_out").cast("double"), F.lit(0.0)).alias("pkg_out"))
            .filter("pid is not null").groupBy("pid").agg(F.sum("pkg_in").alias("pkg_in"), F.sum("pkg_out").alias("pkg_out")))
    both = mt.join(tv, "pid", "inner")
    elec_in = reduce(lambda a, b: a + b, [F.col(f"`{r}|in`") for r in ("ACH", "Wire", "RTP")])
    all_in = reduce(lambda a, b: a + b, [F.col(f"`{r}|in`") for r in RAILS])
    s = (both.withColumn("e_in", elec_in).withColumn("a_in", all_in)
             .agg(F.count("*").alias("n"), F.sum("pkg_in").alias("pkg_in"), F.sum("pkg_out").alias("pkg_out"),
                  *[F.sum(f"`{r}|{d}`").alias(f"{r}|{d}") for r in RAILS for d in ("in", "out")],
                  F.expr("percentile_approx(CASE WHEN pkg_in > 0 THEN e_in / pkg_in END, 0.5)").alias("med_elec"),
                  F.expr("percentile_approx(CASE WHEN pkg_in > 0 THEN a_in / pkg_in END, 0.5)").alias("med_all"))
             .collect()[0])
    f2 = lambda v: "n/a" if v is None else f"{v:.2f}"
    n_met = mt.count()
    rows = []
    for r in RAILS:
        rows.append((r, s[f"{r}|in"] / 1e9, s[f"{r}|out"] / 1e9,
                     s[f"{r}|in"] / s["pkg_in"] if s["pkg_in"] else None,
                     s[f"{r}|out"] / s["pkg_out"] if s["pkg_out"] else None))
    t_in, t_out = sum(x[1] for x in rows), sum(x[2] for x in rows)
    rows.append(("All rails in this table", t_in, t_out, t_in * 1e9 / s["pkg_in"] if s["pkg_in"] else None,
                 t_out * 1e9 / s["pkg_out"] if s["pkg_out"] else None))
    rows.append(("PKG metrics amt_all_in / amt_all_out", (s["pkg_in"] or 0) / 1e9, (s["pkg_out"] or 0) / 1e9, 1.0, 1.0))
    ov = pd.DataFrame(rows, columns=["rail", "in B", "out B", "in vs PKG in", "out vs PKG out"])
    parts += [f"<p><b>Overlap with PKG payment dollars</b> (month {cm}; {s['n']:,} customers in both, "
              f"of {n_met:,} PKG customers with a power id that month)</p>",
              tbl(_fmt(ov, ("in vs PKG in", "out vs PKG out"), (), ("in B", "out B"))),
              f"<p class='note'>Per customer, median of (ACH + wire + RTP in) / PKG in: {f2(s['med_elec'])}; "
              f"median of (all rails in) / PKG in: {f2(s['med_all'])}. "
              f"Near 1.0 = the same money as the PKG; well below 1 = the PKG sees more; above 1 = this table sees more.</p>"]

cust.unpersist()
parts.append(f"<p class='note'>Saved: {esc(os.path.abspath(os.path.join(OUT_DIR, 'history_by_month.csv')))}</p>")
show(parts)
