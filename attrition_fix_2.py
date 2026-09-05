_p = spark.read.parquet(sp(PAIRS_DIR))
disp(_p.groupBy("kt", "flow").agg(F.count("*").alias("rows"),
       F.countDistinct("cust_pwr_id").alias("customers")).orderBy("kt", "flow"),
     title="2a-pre · pay_pairs vocabulary", n=20)
disp(_p.select("k").limit(5), title="2a-pre · sample entity keys", n=5)

PAIRS_KIND_COL   = "kt"
PAIRS_ENTITY_COL = "k"
PAIRS_FLOW_COL   = "flow"
PAIRS_FLOW_OUT   = "out"      # <- replace with the actual outbound value from 2a-pre
PAIRS_KIND_CPTY  = "cpty"     # <- and the two kt values
PAIRS_KIND_FIN   = "fin"



k_col  = PAIRS_KIND_COL   or resolve(pairs, ["kt", "kind", "entity_kind"], "kind col", required=True)
e_col  = PAIRS_ENTITY_COL or resolve(pairs, ["k", "entity", "entity_key"], "entity col", required=True)
fl_col = PAIRS_FLOW_COL   or resolve(pairs, ["flow", "direction", "dir"], "flow col")
id_col = resolve(pairs, ["cust_pwr_id"], "pay_pairs customer id", required=True)

pr = pairs.select(F.col(id_col).alias("cust_pwr_id"), "ym",
                  F.lower(F.trim(F.col(k_col).cast("string"))).alias("kind"),
                  F.col(e_col).cast("string").alias("entity"),
                  *([F.lower(F.trim(F.col(fl_col).cast("string"))).alias("flow")] if fl_col else []))

# cpty_new_out / fin_new_out are OUTBOUND. Without this the rebuilt feature is
# a different quantity from v2's and the 2b comparison is meaningless.
if fl_col:
    _before = pr.count()
    pr = pr.filter(F.col("flow") == F.lit(PAIRS_FLOW_OUT))
    _after = pr.count()
    print(f"flow filter '{PAIRS_FLOW_OUT}': {_before:,} -> {_after:,} pair-rows")
    assert _after > 0, (f"PAIRS_FLOW_OUT='{PAIRS_FLOW_OUT}' matched nothing - "
                        f"check the values printed in 2a-pre")
else:
    warn("no flow column - the rebuild mixes inbound and outbound pairs and is NOT "
         "comparable to v2's cpty_new_out")

pr = pr.withColumn("kind", F.when(F.col("kind") == F.lit(PAIRS_KIND_FIN), "fin")
                            .when(F.col("kind") == F.lit(PAIRS_KIND_CPTY), "cpty"))
_unk = pr.filter(F.col("kind").isNull()).count()
assert _unk == 0, f"{_unk:,} rows have a kt value that is neither " \
                  f"'{PAIRS_KIND_CPTY}' nor '{PAIRS_KIND_FIN}'"
pr = pr.drop("flow")
