# WebHDFS probe, part 1 of 2 — run in a Spark notebook on the notebook host (after your three setup cells).
# Writes a small test file to your HDFS space and saves the namenode web addresses to the shared folder, where the
# GPU probe (pkg_webhdfs_probe.py, submitted with pkg_webhdfs_probe.sh) reads them.
import os, json, datetime as dt
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
DB = "bdahd01p_dlcdi1_cdi_tm"
SHARED = os.path.abspath("../net_work")               # the folder the GPU job already reads

jvm = spark._jvm
conf = spark._jsc.hadoopConfiguration()
keys = {}
it = conf.iterator()
while it.hasNext():
    e = it.next()
    k = e.getKey()
    if any(t in k for t in ("http-address", "https-address", "webhdfs", "httpfs", "nameservices", "ha.namenodes",
                            "defaultFS", "http.policy", "authentication")):
        keys[k] = conf.get(k)
db_loc = [r[1] for r in spark.sql(f"DESCRIBE DATABASE {DB}").collect() if r[0].lower() == "location"][0].rstrip("/")
test_dir = f"{db_loc}/_tmp_webhdfs_probe"
test_path = f"{test_dir}/hello.txt"
fs = jvm.org.apache.hadoop.fs.Path(test_path).getFileSystem(conf)
stream = fs.create(jvm.org.apache.hadoop.fs.Path(test_path), True)
stream.write(bytearray(b"hello from the notebook host " + dt.datetime.now().isoformat().encode()))
stream.close()
spark.range(1000).write.mode("overwrite").option("compression", "zstd").parquet(f"{test_dir}/small_parquet")
uri = jvm.java.net.URI(test_path)
info = {"hdfs_path": uri.getPath(), "parquet_dir": jvm.java.net.URI(f"{test_dir}/small_parquet").getPath(),
        "authority": uri.getAuthority(), "user": os.environ.get("USER"), "conf": keys,
        "written_at": dt.datetime.now().isoformat(timespec="seconds")}
os.makedirs(SHARED, exist_ok=True)
json.dump(info, open(os.path.join(SHARED, "_webhdfs_probe.json"), "w"), indent=1)
import pandas as pd
from IPython.display import display, HTML
display(HTML(f"<p>Test file: <code>{test_path}</code><br>Saved for the GPU probe: <code>{SHARED}/_webhdfs_probe.json</code></p>"
             + pd.DataFrame(sorted(keys.items()), columns=["hadoop setting", "value"]).to_html(index=False)))
