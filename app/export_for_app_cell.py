# =====================================================================
# EXPORT FOR THE APP — paste as a new cell at the end of pkg_attrition_v13_m4_model.ipynb
# (best placed just before §12, which unpersists the Spark frames; after §12 it still works, only slower)
# Writes: the model bundle (final model on all labelled months) and two tables for the app.
# =====================================================================
import joblib, datetime
from sklearn.isotonic import IsotonicRegression
t0 = time.time(); B = Block("Export for the app")
TGT_DB, APP_MONTHS, HIST_MONTHS = "bdahd01p_dlcdi1_cdi_tm", 6, 18
DB_LOC = [r[1] for r in spark.sql(f"DESCRIBE DATABASE {TGT_DB}").collect() if r[0].lower() == "location"][0].rstrip("/")

# ── 1 · final model: all labelled months, early stopping and calibration on the held-out clients ──
if GBM not in ("lightgbm", "xgboost"): raise RuntimeError("the app needs a lightgbm or xgboost model")
Gf = gbm_fit(Xall[~HOLD], yall[~HOLD], wall[~HOLD], Xall[HOLD], yall[HOLD], wall[HOLD], DEPTH)
mc = HOLD & (tall >= LAST_LABEL_T - CAL_RECENT + 1)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=P_CAP).fit(gbm_predict(Gf, Xall[mc]), yall[mc], sample_weight=wall[mc])
model_blob = Gf["m"].model_to_string() if GBM == "lightgbm" else bytes(Gf["m"].save_raw("json"))
_tim = TIM.dropna(subset=["onset_month"])
bundle = dict(engine=GBM, model=model_blob, n_iter=int(Gf["n"]), iso=iso, feats=list(FEATS),
              labels={f: LABEL.get(f, f) for f in FEATS}, famof={f: FAMOF.get(f, "") for f in FEATS},
              thr={f: (float(th), d) for f, (th, d) in THR.items()}, onset=dict(zip(_tim.feature, _tim.onset_month.astype(int))),
              late_feats=list(LATE_FEATS), k_main=K_MAIN, alpha=A_MAIN, p_cap=P_CAP, key_mode=KEY,
              trained_through=ym_of(LAST_LABEL_T), created=datetime.datetime.now().isoformat(timespec="minutes"), version="v13-m4")
joblib.dump(bundle, OUT / "attr_model_bundle.joblib")

# ── 2 · app features: money-still-here clients × last APP_MONTHS months ──
t_min = END_T - APP_MONTHS + 1
ym_expr = F.format_string("%04d-%02d", F.floor((F.col("t") + F.lit(M0)) / 12).cast("int"), ((F.col("t") + F.lit(M0)) % 12 + 1).cast("int"))
idn = (spark.table(MET_TABLE).filter(F.col("month") >= ym_of(t_min))
            .select(pk_expr("customer_pwr_id", KEY).alias("pwr"), "mdm_id", "customer_name", "naics_desc", "master_pwr_id")
            .filter("pwr is not null").groupBy("pwr")
            .agg(F.concat_ws(",", F.collect_set("mdm_id")).alias("mdm_ids"), F.first("customer_name", True).alias("customer_name"),
                 F.first("naics_desc", True).alias("naics_desc"), F.first("master_pwr_id", True).alias("master_pwr_id")))
APPF = (MF.filter(F.col("t") >= t_min)
          .select("cust_pwr_id", "pwr", "rltn_pwr_id", ym_expr.alias("month"), "bal", "bal_norm", "bal_eom", "capped",
                  *[F.col(f).cast("float").alias(f) for f in FEATS])
          .join(idn, "pwr", "left"))

def write_table(df, name, part=None):
    path = f"{DB_LOC}/{name}"
    w = (df.repartition(part) if part else df.repartition(8)).write.mode("overwrite").option("compression", "snappy")
    (w.partitionBy(part) if part else w).parquet(path)
    cols = ", ".join(f"`{f.name}` {f.dataType.simpleString()}" for f in df.schema.fields if f.name != part)
    spark.sql(f"DROP TABLE IF EXISTS {TGT_DB}.{name}")
    spark.sql(f"CREATE EXTERNAL TABLE {TGT_DB}.{name} ({cols}) " + (f"PARTITIONED BY (`{part}` STRING) " if part else "")
              + f"STORED AS PARQUET LOCATION '{path}'")
    if part: spark.sql(f"MSCK REPAIR TABLE {TGT_DB}.{name}")

APPF = APPF.select(*[c for c in APPF.columns if c != "month"], "month")         # partition column last
write_table(APPF, "pkg_attr_app_features", part="month")

# ── 3 · app history: 18 months of balances and raw payment metrics, listed clients and their relationships ──
cur = spark.table(f"{TGT_DB}.pkg_attr_app_features").filter(F.col("month") == ym_of(END_T))
rl = cur.select("rltn_pwr_id").filter("rltn_pwr_id is not null").distinct()
members = (X.filter(F.col("t") == END_T).join(rl, "rltn_pwr_id").select("cust_pwr_id")
             .union(cur.select("cust_pwr_id")).distinct())
RAW = [c for c in ["amt_all_in", "amt_all_out", "ntxn_all_in", "ntxn_all_out", "active_days_in", "ncpty_all_in", "ncpty_all_out",
                   "n_fi_out", "n_fi_in", "selfpay_amt_out", "share_WIRE_in", "share_CHECK_in", "net_flow_all"] if c in pm.columns]
HIST = (X.filter(F.col("t") >= END_T - HIST_MONTHS + 1).join(members, "cust_pwr_id")
          .withColumn("pwr", pk_expr("cust_pwr_id", KEY))
          .join(pm.select("pwr", "t", *RAW), ["pwr", "t"], "left")
          .join(idn.select("pwr", "customer_name"), "pwr", "left")
          .select("cust_pwr_id", "rltn_pwr_id", "customer_name", ym_expr.alias("month"), "bal", "bal_norm", "bal_eom", "bal_min",
                  "bal_max", "live", "n_live", "n_accts", *[F.col(c).cast("double").alias(c.lower()) for c in RAW]))
write_table(HIST, "pkg_attr_app_history")

n_f = spark.table(f"{TGT_DB}.pkg_attr_app_features").groupBy("month").count().orderBy("month").toPandas()
n_h = spark.table(f"{TGT_DB}.pkg_attr_app_history").count()
B.kv([("model bundle", str((OUT / "attr_model_bundle.joblib").resolve())), ("model", f"{GBM}, {Gf['n']} trees, trained through {ym_of(LAST_LABEL_T)}"),
      ("app features", f"{TGT_DB}.pkg_attr_app_features · {int(n_f['count'].sum()):,} rows over {len(n_f)} months"),
      ("app history", f"{TGT_DB}.pkg_attr_app_history · {n_h:,} rows"),
      ("Impala, once", "INVALIDATE METADATA bdahd01p_dlcdi1_cdi_tm.pkg_attr_app_features; "
                       "INVALIDATE METADATA bdahd01p_dlcdi1_cdi_tm.pkg_attr_app_history;")], "Written")
B.show(t0)
