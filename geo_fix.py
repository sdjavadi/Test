import glob as _glob
from urllib.parse import urlparse

def _list_paths(pattern):
    """Resolve a glob on either the local filesystem or HDFS.

    A bare relative path is LOCAL. Hadoop's FileSystem.get() resolves against
    the default FS (HDFS), so it silently returns nothing for local paths —
    which shows up much later as an empty window rather than a missing file.
    """
    if urlparse(pattern).scheme in ("", "file"):
        hits = sorted(_glob.glob(pattern.replace("file://", "")))
        return ["file://" + os.path.abspath(h) for h in hits]
    fs = (spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem
          .get(spark.sparkContext._jsc.hadoopConfiguration()))
    Path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path
    return [s.getPath().toString() for s in fs.globStatus(Path(pattern))]

all_paths = _list_paths(EDGE_GLOB)
if not all_paths:
    raise FileNotFoundError(
        f"{EDGE_GLOB} matched nothing. cwd={os.getcwd()}")
print(f"{len(all_paths)} snapshot files found")
