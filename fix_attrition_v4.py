PANELS      = "hdfs://nameservice1/user/pk36814/attrition_v2"
PANELS_PREV = None


# Fail with the path, not with "unable to infer schema" - that message means
# Spark resolved the directory and found no parquet files, which is almost
# always a path that has not been built yet.
_jvm  = spark._jvm
_conf = spark._jsc.hadoopConfiguration()
_fs   = _jvm.org.apache.hadoop.fs.FileSystem.get(_jvm.java.net.URI(PANELS), _conf)
for _p in ("panel_account_month", "panel_customer_month", "panel_pay_features", PAIRS_DIR):
    _path = _jvm.org.apache.hadoop.fs.Path(sp(_p))
    if not _fs.exists(_path):
        raise FileNotFoundError(
            f"{sp(_p)} does not exist. PANELS is set to {PANELS} — point it at a build "
            f"that has been run (attrition_v2 for the 31-month panel).")
    if _fs.getContentSummary(_path).getFileCount() == 0:
        raise FileNotFoundError(f"{sp(_p)} exists but is empty.")
print(f"preflight OK — all panels present under {PANELS}")
