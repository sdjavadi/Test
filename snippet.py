l = spark.read.parquet(path("legs")).filter(F.col("month")=="2025-09")
l.groupBy("txn_id").count().groupBy("count").count().orderBy("count").show()
l.groupBy("topology","ego_dir").count().show()
