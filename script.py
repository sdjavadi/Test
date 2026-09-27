DSI = "/bdpp/dsi/01/str/pub/dsihd01p_dsi"
DB  = "bdahd01p_dlcdi1_cdi_tm"
loc = spark.catalog.getDatabase(DB).locationUri          # hdfs://nameservice1/...
TM  = loc.split("nameservice1", 1)[-1]
ups = [TM]
while ups[-1].count("/") > 2: ups.append(ups[-1].rsplit("/", 1)[0])
print(DB, "→", TM)
!hdfs groups
!hdfs dfs -ls -d {DSI} {TM}
!for d in {DSI} {TM}; do hdfs dfs -touchz $d/_wtest_pk36814 && hdfs dfs -rm -skipTrash -q $d/_wtest_pk36814 && echo "WRITE OK  $d" || echo "NO WRITE  $d"; done
!hdfs dfs -count -q -h -v {" ".join(ups)}




t = f"{DB}.pk36814_write_test"
spark.sql(f"CREATE EXTERNAL TABLE {t} (x INT) STORED AS PARQUET LOCATION '{loc}/pk36814_write_test'")
spark.sql(f"DROP TABLE {t}"); print("HIVE CREATE OK")
!hdfs dfs -rm -r -skipTrash -q {TM}/pk36814_write_test
