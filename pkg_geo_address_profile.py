# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: PySpark
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PKG Geo — MDM Address Profiling & Extract
#
# **Source:** `dsihd01p_dsi.neo4j_address` (MDM party-level address table)
# **Target:** node-level geo CSV consumed by the PKG metric modules
#
# ---
#
# ## Read this before running
#
# **1. No `CREATE VIEW` needed anywhere.** Every step uses either the DataFrame API or
# `createOrReplaceTempView`, which is *session-scoped* and requires no DDL privilege on
# the metastore. Nothing in this notebook writes to a database.
#
# **2. This table is ALL customers — it is not the PKG universe.** That single fact
# changes how every number in here must be read. The MDM row count is dominated by
# retail parties who have never appeared in a C2C payment edge. So every statistic is
# reported against **three populations**, never one:
#
# | Population | What it is | What it's good for |
# |---|---|---|
# | `MDM_ALL` | every row in the address table | data-quality profiling only |
# | `PKG_NODE` | parties present as `source` or `dest` in the C2C metrics | the actual denominator for coverage |
# | `PKG_DOLLAR` | same, weighted by `amount` | the only number worth briefing |
#
# The Pittsburgh-concentration belief must be tested on `PKG_DOLLAR`, not `MDM_ALL`.
# The MDM-wide geography is the geography of PNC's retail book, which is a different
# question and will overstate local concentration.
#
# **3. `latitude_degrees` / `longitude_degrees` are STRING columns.** Empty strings,
# `'0'`, whitespace, the literal token `'null'`, and dropped negative signs all survive
# `IS NOT NULL`. Section C screens before casting. Never cast blind.
#
# **4. Impala functions do not exist in Spark.** `SPLIT_PART` and `FNV_HASH` from the
# earlier Impala drafts are replaced with `split()[i]` and `xxhash64` here.
#
# ---
#
# ## Sections
#
# | | |
# |---|---|
# | **0** | Config, schema discovery, snapshot selection |
# | **A** | Grain and history |
# | **B** | `addr_type` and `addr_loc_rec_type` domains |
# | **C** | Coordinate pathology (the string screen) |
# | **D** | Placeholder detection — F1 |
# | **E** | ZIP / state / country hygiene |
# | **F** | Multi-address parties — selection rule input |
# | **G** | **PKG join and three-population coverage — the blocker** |
# | **H** | Build and write the extract |
# | **I** | Report block to send back |

# %%
import json
import math
import re
from datetime import datetime

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.storagelevel import StorageLevel

# %% [markdown]
# ## 0. Config
#
# Everything environment-specific lives in this one cell. Nothing below it should need
# editing on the first run.

# %%
# ---- Tables -----------------------------------------------------------------
ADDR_TABLE = "dsihd01p_dsi.neo4j_address"
PKG_METRICS_TABLE = "bdahd01p_dlcdi1_cdi_tm.cust_c2c_metrics"
PKG_ROLES_TABLE = "bdahd01p_dlcdi1_cdi_tm.cust_c2c_roles"  # optional, section G6

# ---- Output -----------------------------------------------------------------
OUT_DIR_LOCAL = "../metrics/geo"          # driver-local, for the CSV handoff
OUT_DIR_HDFS = None                       # e.g. "/user/<you>/pkg/geo"; None = skip parquet

# ---- PKG time window --------------------------------------------------------
# Set to None to use every month in the metrics table. If the table is large,
# restrict to the months the geo module will actually score against.
PKG_TIME_COL = None       # auto-detected in 0.2 if left as None
PKG_TIME_MIN = None       # e.g. "202501"
PKG_TIME_MAX = None       # e.g. "202512"

# ---- Thresholds -------------------------------------------------------------
DUP_MIN_PARTIES = 25      # min distinct parties at one coordinate to flag as suspect
DUP_ROUND_DP = 5          # rounding for the coordinate dedup key (~1.1 m)
TOP_N = 60                # rows shown in ranked listings

# ---- Pittsburgh reference point (Point State Park) ---------------------------
PIT_LAT, PIT_LON = 40.4406, -79.9959

# ---- addr_type priority for the one-row-per-party selection rule -------------
# PLACEHOLDER. Section B1 prints the observed domain; come back and set this,
# then re-run section H. Lower number = preferred. Unmapped values get 99 and
# are reported loudly rather than silently ranked last.
ADDR_TYPE_PRIORITY = {
    "PHYSICAL": 0,
    "PRIMARY": 0,
    "LEGAL": 1,
    "BUSINESS": 1,
    "HOME": 2,
    "MAILING": 3,
    "BILLING": 4,
    "STATEMENT": 4,
}

REC_TYPE_PRIORITY = {"NORMAL": 0, "FIRM": 0, "HIGHRISE": 1, "RURALROUTE": 2}

# ---- Accumulator for the summary block in section I -------------------------
REPORT = {"generated_at": datetime.now().isoformat(timespec="seconds")}


def note(key, value):
    """Record a number for the section I summary and echo it."""
    REPORT[key] = value
    print(f"  {key}: {value}")


def show(df, n=TOP_N, truncate=False):
    df.show(n, truncate=truncate)


# %%
spark = (
    SparkSession.builder.appName("pkg_geo_address_profile")
    .config("spark.sql.shuffle.partitions", "400")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .enableHiveSupport()
    .getOrCreate()
)
print("Spark", spark.version)

# xxhash64 is Spark 3.0+. Fall back to a truncated sha2 on older CDH runtimes
# rather than murmur3 `hash()`, which is 32-bit and will collide at 10M distinct keys.
try:
    _ = F.xxhash64
    def stable_hash(*cols):
        return F.xxhash64(*cols)
except AttributeError:
    def stable_hash(*cols):
        return F.substring(F.sha2(F.concat_ws("|", *cols), 256), 1, 16)

# %% [markdown]
# ### 0.1 Full schema
#
# The `describe` output in the screenshot was cut off below `src_mdm_load_dt`, so the
# column list is not fully known. Print it, and specifically look for:
#
# - an **active / primary / current** flag
# - **effective start / end dates**
# - the real **snapshot column** (`load_dt` was used in the earlier query but was not
#   visible in the describe output)
# - any **geocode match-quality / precision code** from the address cleansing vendor

# %%
addr_raw = spark.table(ADDR_TABLE)
addr_cols = addr_raw.columns
print(f"{len(addr_cols)} columns\n")
addr_raw.printSchema()
note("addr_columns", addr_cols)

# %%
# Flag anything that looks like a flag, a date, or a quality code — these are the
# columns that change the selection rule and were not visible in the screenshot.
INTERESTING = r"(flag|ind|is_|active|current|primary|prim|status|eff|start|end|expir|" \
              r"valid|qual|match|precis|conf|score|geo|source|src|type|rank|seq)"
cand = [c for c in addr_cols if re.search(INTERESTING, c, re.I)]
print("Columns worth a second look:")
for c in cand:
    print("   ", c)

# %% [markdown]
# ### 0.2 Snapshot column detection and latest-snapshot selection
#
# This replaces the view. `createOrReplaceTempView` is session-local — no metastore
# write, no DDL privilege.

# %%
SNAP_CANDIDATES = ["load_dt", "src_mdm_load_dt", "etl_load_dt", "batch_dt", "as_of_dt", "dw_load_dt"]
SNAP_COL = next((c for c in SNAP_CANDIDATES if c in addr_cols), None)

if SNAP_COL is None:
    raise RuntimeError(
        f"No snapshot column found among {SNAP_CANDIDATES}. "
        f"Inspect the column list above and set SNAP_COL manually."
    )
print("snapshot column:", SNAP_COL)

snap_hist = (
    addr_raw.groupBy(SNAP_COL)
    .agg(F.count("*").alias("n_rows"), F.countDistinct("mdm_id").alias("n_parties"))
    .orderBy(SNAP_COL)
)
show(snap_hist, 50)

n_snaps = snap_hist.count()
note("n_snapshots_retained", n_snaps)

# %% [markdown]
# **Why this matters (D1, the registered-point change log).** If more than one snapshot
# is retained, the address change history can be reconstructed retroactively and the
# geo-dynamics module has a back-catalogue on day one. If only one is retained, the
# change log can only be built going forward — in which case archive a dated copy of
# the extract every month starting now, or that signal is lost permanently.

# %%
MAX_SNAP = addr_raw.agg(F.max(SNAP_COL)).collect()[0][0]
print("latest snapshot:", MAX_SNAP)

addr = addr_raw.filter(F.col(SNAP_COL) == F.lit(MAX_SNAP))
addr = addr.persist(StorageLevel.MEMORY_AND_DISK)
addr.createOrReplaceTempView("addr_latest")   # session-scoped; no CREATE privilege needed

note("snapshot_used", str(MAX_SNAP))

# %% [markdown]
# ## A. Grain and history

# %%
grain = addr.agg(
    F.count("*").alias("n_rows"),
    F.countDistinct("mdm_address_id").alias("n_addr_ids"),
    F.countDistinct("mdm_id").alias("n_parties"),
).collect()[0]

note("A_n_rows", grain["n_rows"])
note("A_n_addr_ids", grain["n_addr_ids"])
note("A_n_parties", grain["n_parties"])
note("A_addr_per_party", round(grain["n_rows"] / max(grain["n_parties"], 1), 4))

if grain["n_rows"] != grain["n_addr_ids"]:
    print("  !! mdm_address_id is NOT unique in this snapshot — the grain is not one "
          "row per address. Investigate before section H.")

# %% [markdown]
# ## B. `addr_type` and `addr_loc_rec_type`
#
# `addr_type` is the mailing-vs-physical determinant flagged earlier as the blocker on
# the representativeness test. Without it, the registered-address problem is
# unmeasurable. This is the first thing to read.

# %%
# Reusable numeric-coordinate screen. NOTE: applied here only as a coverage indicator;
# section C does the full pathology breakdown.
NUM_RE = r"^\s*-?\d+(\.\d+)?\s*$"

lat_num = F.when(F.col("latitude_degrees").rlike(NUM_RE), F.trim(F.col("latitude_degrees")).cast("double"))
lon_num = F.when(F.col("longitude_degrees").rlike(NUM_RE), F.trim(F.col("longitude_degrees")).cast("double"))

has_geo = (
    lat_num.isNotNull() & lon_num.isNotNull()
    & ~((lat_num == 0) & (lon_num == 0))
    & lat_num.between(-90, 90) & lon_num.between(-180, 180)
).cast("int")

addr_g = addr.withColumn("_lat", lat_num).withColumn("_lon", lon_num).withColumn("_has_geo", has_geo)
addr_g = addr_g.persist(StorageLevel.MEMORY_AND_DISK)

# %%
# B1 — addr_type domain and geocode coverage per type
b1 = (
    addr_g.groupBy(F.upper(F.trim("addr_type")).alias("addr_type"))
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("mdm_id").alias("n_parties"),
        F.sum("_has_geo").alias("n_geocoded"),
        F.round(100 * F.avg("_has_geo"), 2).alias("pct_geocoded"),
    )
    .orderBy(F.desc("n_rows"))
)
show(b1)
REPORT["B1_addr_type"] = [r.asDict() for r in b1.collect()]

observed_types = {r["addr_type"] for r in REPORT["B1_addr_type"] if r["addr_type"]}
unmapped = sorted(observed_types - set(ADDR_TYPE_PRIORITY))
if unmapped:
    print(f"\n  !! addr_type values not in ADDR_TYPE_PRIORITY: {unmapped}")
    print("     Set their priority in cell 0 and re-run section H.")

# %%
# B2 — addr_type x rec_type. Confirms whether PO Boxes concentrate in mailing types
# (expected, benign) or contaminate physical/legal types (a real problem).
b2 = (
    addr_g.groupBy(
        F.upper(F.trim("addr_type")).alias("addr_type"),
        F.upper(F.trim("addr_loc_rec_type")).alias("rec_type"),
    )
    .agg(F.count("*").alias("n_rows"), F.countDistinct("mdm_id").alias("n_parties"))
    .orderBy("addr_type", F.desc("n_rows"))
)
show(b2, 100)
REPORT["B2_type_x_rec"] = [r.asDict() for r in b2.collect()]

# %%
# B3 — full rec_type domain. Firm, RuralRoute, GeneralDelivery and military types
# exist in the USPS AIS standard and each needs its own geo_status row.
b3 = (
    addr_g.groupBy(F.upper(F.trim("addr_loc_rec_type")).alias("rec_type"))
    .agg(
        F.count("*").alias("n_rows"),
        F.sum("_has_geo").alias("n_geocoded"),
        F.round(100 * F.avg("_has_geo"), 2).alias("pct_geocoded"),
    )
    .withColumn("pct_of_rows", F.round(100 * F.col("n_rows") / F.sum("n_rows").over(Window.partitionBy()), 3))
    .orderBy(F.desc("n_rows"))
)
show(b3)
REPORT["B3_rec_type"] = [r.asDict() for r in b3.collect()]

# %% [markdown]
# ## C. Coordinate pathology
#
# Every bucket below `1_null` survives a plain `IS NOT NULL` check. Bucket
# `7_positive_lon` is the one that silently relocates US customers to Central Asia —
# it's a dropped negative sign, fixable rather than discardable.

# %%
lat_s = F.trim(F.coalesce(F.col("latitude_degrees"), F.lit("")))
lon_s = F.trim(F.coalesce(F.col("longitude_degrees"), F.lit("")))
NULL_TOKENS = ["null", "none", "na", "n/a", "nan", "unknown", "-", "."]

coord_state = (
    F.when(F.col("latitude_degrees").isNull() | F.col("longitude_degrees").isNull(), "1_null")
    .when((lat_s == "") | (lon_s == ""), "2_empty")
    .when(F.lower(lat_s).isin(NULL_TOKENS) | F.lower(lon_s).isin(NULL_TOKENS), "3_null_token")
    .when(~lat_s.rlike(NUM_RE) | ~lon_s.rlike(NUM_RE), "4_non_numeric")
    .when((lat_s.cast("double") == 0) & (lon_s.cast("double") == 0), "5_null_island")
    .when(~lat_s.cast("double").between(-90, 90) | ~lon_s.cast("double").between(-180, 180), "6_out_of_range")
    .when(lon_s.cast("double") > 0, "7_positive_lon_dropped_sign")
    .when(
        lat_s.cast("double").between(24, 50) & lon_s.cast("double").between(-125, -66),
        "8_ok_conus",
    )
    .otherwise("9_ok_non_conus")
)

c1 = (
    addr.withColumn("coord_state", coord_state)
    .groupBy("coord_state")
    .agg(F.count("*").alias("n_rows"), F.countDistinct("mdm_id").alias("n_parties"))
    .withColumn("pct", F.round(100 * F.col("n_rows") / F.sum("n_rows").over(Window.partitionBy()), 3))
    .orderBy("coord_state")
)
show(c1)
REPORT["C1_coord_state"] = [r.asDict() for r in c1.collect()]

# %%
# C1b — sample the pathological buckets. Seeing the actual strings is worth more
# than the counts for deciding what is salvageable.
bad = addr.withColumn("coord_state", coord_state).filter(
    F.col("coord_state").rlike("^[2-7]_")
)
show(
    bad.select("coord_state", "latitude_degrees", "longitude_degrees", "city",
               "state_or_province", "zip_cd", "addr_loc_rec_type").limit(40),
    40,
)

# %%
# C2 — decimal places as a geocode-quality proxy.
# <=2 decimals is ~1 km resolution: that is a centroid, not a rooftop, regardless of
# what rec_type claims.
dp = lambda c: F.coalesce(F.length(F.split(F.trim(F.col(c)), r"\.").getItem(1)), F.lit(0))

c2 = (
    addr_g.filter(F.col("_has_geo") == 1)
    .withColumn("lat_dp", dp("latitude_degrees"))
    .withColumn("lon_dp", dp("longitude_degrees"))
    .withColumn("coord_decimals", F.least("lat_dp", "lon_dp"))
    .groupBy("coord_decimals", F.upper(F.trim("addr_loc_rec_type")).alias("rec_type"))
    .agg(F.count("*").alias("n_rows"))
    .orderBy("coord_decimals", F.desc("n_rows"))
)
show(c2, 60)
REPORT["C2_coord_decimals"] = [r.asDict() for r in c2.collect()]

# %% [markdown]
# ## D. Placeholder detection (F1)
#
# The duplicate-coordinate screen **must be conditioned on `rec_type`**. Duplicates
# inside `HighRise` are expected — that's the building, not a defect. Duplicates inside
# `Normal` are the signal: registered-agent offices, CPA firms, shared service
# addresses, and PNC's own branch addresses.

# %%
coord_key = F.concat_ws(
    "_",
    F.round(F.col("_lat"), DUP_ROUND_DP).cast("string"),
    F.round(F.col("_lon"), DUP_ROUND_DP).cast("string"),
)

d1 = (
    addr_g.filter(F.col("_has_geo") == 1)
    .withColumn("rec_type", F.upper(F.trim("addr_loc_rec_type")))
    .withColumn("coord_key", coord_key)
    .groupBy("rec_type", "coord_key")
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("mdm_id").alias("n_parties"),
        F.min("city").alias("sample_city"),
        F.min("state_or_province").alias("sample_state"),
        F.min("addr_line_1").alias("sample_line1"),
    )
    .filter(F.col("n_parties") >= DUP_MIN_PARTIES)
    .orderBy(F.desc("n_parties"))
)
d1 = d1.persist()
show(d1, 100, truncate=60)
note("D1_suspect_coord_clusters", d1.count())
REPORT["D1_top_clusters"] = [r.asDict() for r in d1.limit(40).collect()]

# %%
# D1b — how much of the book sits on a suspect coordinate?
sus_keys = d1.select("rec_type", "coord_key").distinct()
flagged = (
    addr_g.filter(F.col("_has_geo") == 1)
    .withColumn("rec_type", F.upper(F.trim("addr_loc_rec_type")))
    .withColumn("coord_key", coord_key)
    .join(F.broadcast(sus_keys), ["rec_type", "coord_key"], "left_semi")
)
note("D1b_parties_on_suspect_coord", flagged.select("mdm_id").distinct().count())

# %%
# D2 — shared service addresses, independent of geocode quality.
# Catches the same phenomenon for rows whose coordinates differ by rounding noise.
norm_line1 = F.regexp_replace(F.upper(F.trim(F.coalesce(F.col("addr_line_1"), F.lit("")))), "[^A-Z0-9 ]", "")
zip5 = F.split(F.trim(F.coalesce(F.col("zip_cd"), F.lit(""))), "-").getItem(0)

d2 = (
    addr.filter(F.trim(F.coalesce(F.col("addr_line_1"), F.lit(""))) != "")
    .withColumn("norm_addr", norm_line1)
    .withColumn("zip5", zip5)
    .groupBy("norm_addr", "zip5")
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("mdm_id").alias("n_parties"),
        F.min("city").alias("sample_city"),
        F.min("state_or_province").alias("sample_state"),
    )
    .filter(F.col("n_parties") >= DUP_MIN_PARTIES)
    .orderBy(F.desc("n_parties"))
)
show(d2, 80, truncate=50)
REPORT["D2_top_shared_addresses"] = [r.asDict() for r in d2.limit(40).collect()]

# %%
# D3 — token flags on the address lines
u1 = F.upper(F.trim(F.coalesce(F.col("addr_line_1"), F.lit(""))))
u2 = F.upper(F.trim(F.coalesce(F.col("addr_line_2"), F.lit(""))))
u3 = F.upper(F.trim(F.coalesce(F.col("addr_line_3"), F.lit(""))))
u4 = F.upper(F.trim(F.coalesce(F.col("addr_line_4"), F.lit(""))))

REG_AGENT_RE = (
    r"(REGISTERED AGENT|CORPORATION SERVICE|CT CORPORATION|NATIONAL REGISTERED|"
    r"INCORP SERVICES|LEGALZOOM|COGENCY|VCORP|NORTHWEST REGISTERED|"
    r"UNITED STATES CORPORATION AGENT|CSC )"
)

flags = {
    "po_box": u1.rlike(r"(^| )P ?\.? ?O ?\.? ?BOX"),
    "care_of": u1.rlike(r"(^| )(C/O|C O |ATTN)"),
    "pmb": u1.rlike(r"(^| )PMB"),
    "rural_route": u1.rlike(r"(^| )(RR |RURAL ROUTE|HC )"),
    "general_delivery": u1.rlike(r"GENERAL DELIVERY"),
    "reg_agent": u1.rlike(REG_AGENT_RE),
    "pnc_own": u1.rlike(r"(^| )PNC( |$)"),
    "unit_residential": u2.rlike(r"^(APT|UNIT|#|TRLR|LOT|SPC)"),
    "unit_commercial": u2.rlike(r"^(STE|SUITE|FL |FLOOR|RM |DEPT|BLDG)"),
    "line3_used": u3 != "",
    "line4_used": u4 != "",
}

d3 = addr.agg(
    F.count("*").alias("n_rows"),
    *[F.sum(F.when(cond, 1).otherwise(0)).alias(name) for name, cond in flags.items()],
)
show(d3, 1)
REPORT["D3_token_flags"] = d3.collect()[0].asDict()

# %% [markdown]
# ## E. ZIP, state, country hygiene
#
# ZIP5 is the join key to the HUD USPS crosswalk → ZCTA → county FIPS → CBSA, and CBSA
# is the real unit of analysis for corridors and metro exposure. It also survives when
# lat/lon does not, which makes it the backbone of section G's coverage test.

# %%
zip_t = F.trim(F.coalesce(F.col("zip_cd"), F.lit("")))
zip_shape = (
    F.when(zip_t == "", "empty")
    .when(zip_t.rlike(r"^\d{5}-\d{4}$"), "zip9_dash")
    .when(zip_t.rlike(r"^\d{9}$"), "zip9_flat")
    .when(zip_t.rlike(r"^\d{5}$"), "zip5")
    .when(zip_t.rlike(r"^\d{4}$"), "zip4_LEADING_ZERO_ALREADY_LOST")
    .when(zip_t.rlike(r"^[A-Z]\d[A-Z]"), "canada_fsa")
    .otherwise("other")
)

e1 = (
    addr.withColumn("zip_shape", zip_shape)
    .groupBy("zip_shape")
    .agg(F.count("*").alias("n_rows"))
    .orderBy(F.desc("n_rows"))
)
show(e1)
REPORT["E1_zip_shape"] = [r.asDict() for r in e1.collect()]

# %%
e2 = (
    addr.groupBy(F.upper(F.trim("addr_country")).alias("country"))
    .agg(F.count("*").alias("n_rows"), F.countDistinct("mdm_id").alias("n_parties"))
    .orderBy(F.desc("n_rows"))
)
show(e2, 40)
REPORT["E2_country"] = [r.asDict() for r in e2.limit(40).collect()]

e3 = (
    addr_g.groupBy(F.upper(F.trim("state_or_province")).alias("state"))
    .agg(
        F.count("*").alias("n_rows"),
        F.countDistinct("mdm_id").alias("n_parties"),
        F.round(100 * F.avg("_has_geo"), 2).alias("pct_geocoded"),
    )
    .orderBy(F.desc("n_parties"))
)
show(e3, 80)
REPORT["E3_state"] = [r.asDict() for r in e3.limit(80).collect()]

# %% [markdown]
# ## F. Multi-address parties
#
# Feeds two things: the `addr_rank = 1` selection rule in section H, and
# `n_addresses_per_mdm_id` as a free multi-site indicator for the locality
# classification (F3).

# %%
per_party = (
    addr_g.groupBy("mdm_id")
    .agg(
        F.count("*").alias("n_addr"),
        F.sum("_has_geo").alias("n_geo"),
        F.countDistinct(F.when(F.col("_has_geo") == 1, coord_key)).alias("n_points"),
        F.avg(F.when(F.col("_has_geo") == 1, F.col("_lat"))).alias("lat_c"),
        F.avg(F.when(F.col("_has_geo") == 1, F.col("_lon"))).alias("lon_c"),
    )
    .persist()
)

f1 = (
    per_party.groupBy("n_addr", "n_points")
    .agg(F.count("*").alias("n_parties"))
    .orderBy(F.desc("n_parties"))
)
show(f1, 40)
REPORT["F1_addr_point_counts"] = [r.asDict() for r in f1.limit(40).collect()]

# %%
# F2 — for multi-point parties, how far apart are the points?
# Max distance from the party's own centroid; proper haversine, not a degree proxy.
def haversine_km(lat1, lon1, lat2, lon2):
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.pow(F.sin(dlat / 2), 2) + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(dlon / 2), 2)
    return F.lit(6371.0088) * 2 * F.asin(F.sqrt(F.least(a, F.lit(1.0))))


spread = (
    addr_g.filter(F.col("_has_geo") == 1)
    .join(per_party.select("mdm_id", "lat_c", "lon_c", "n_points"), "mdm_id")
    .filter(F.col("n_points") > 1)
    .withColumn("km_from_centroid", haversine_km(F.col("_lat"), F.col("_lon"), F.col("lat_c"), F.col("lon_c")))
    .groupBy("mdm_id")
    .agg(F.max("km_from_centroid").alias("max_km"))
)

f2 = (
    spread.withColumn(
        "km_bucket",
        F.when(F.col("max_km") < 1, "0_under_1km")
        .when(F.col("max_km") < 10, "1_1_10km")
        .when(F.col("max_km") < 50, "2_10_50km")
        .when(F.col("max_km") < 250, "3_50_250km")
        .when(F.col("max_km") < 1000, "4_250_1000km")
        .otherwise("5_over_1000km"),
    )
    .groupBy("km_bucket")
    .agg(F.count("*").alias("n_parties"))
    .orderBy("km_bucket")
)
show(f2)
REPORT["F2_multipoint_spread"] = [r.asDict() for r in f2.collect()]

# %% [markdown]
# **Reading F2.** Parties in the `4_` and `5_` buckets have addresses that cannot both
# be "where the customer is". Those are the multi-site or HQ-vs-operations cases, and
# `n_addresses_per_mdm_id` plus this spread is the cheapest available multi-site flag —
# it feeds directly into F3 without needing any payment data.

# %% [markdown]
# ## G. PKG join and three-population coverage
#
# **This is the blocker section.** Everything above is data quality; this decides
# whether the geo module is buildable at all.
#
# The MDM table is all customers. The PKG node set is the parties with at least one
# C2C edge. These are different populations and the second is the only denominator
# that matters for the geo modules.

# %%
pkg_raw = spark.table(PKG_METRICS_TABLE)
print(f"{len(pkg_raw.columns)} columns")
pkg_raw.printSchema()

# %%
# G0 — resolve the time column and apply the window
if PKG_TIME_COL is None:
    TIME_CANDIDATES = ["time_key", "month_key", "yyyymm", "period", "as_of_month", "rpt_month", "load_dt"]
    PKG_TIME_COL = next((c for c in TIME_CANDIDATES if c in pkg_raw.columns), None)
print("PKG time column:", PKG_TIME_COL)

pkg = pkg_raw
if PKG_TIME_COL and PKG_TIME_MIN:
    pkg = pkg.filter(F.col(PKG_TIME_COL) >= F.lit(PKG_TIME_MIN))
if PKG_TIME_COL and PKG_TIME_MAX:
    pkg = pkg.filter(F.col(PKG_TIME_COL) <= F.lit(PKG_TIME_MAX))

if PKG_TIME_COL:
    show(pkg.groupBy(PKG_TIME_COL).agg(F.count("*").alias("n_edges")).orderBy(PKG_TIME_COL), 40)

# %%
# G1 — build the PKG node universe: union of the source and dest sides, with the
# dollar and volume weights that make the coverage number briefable.
NAICS_PLACEHOLDER_RE = r"^\s*(\*+|-1|0+|UNK|UNKNOWN|N/?A)?\s*$"


def naics_status(col):
    code = F.trim(F.split(F.coalesce(col, F.lit("")), r"\|").getItem(0))
    desc = F.upper(F.trim(F.split(F.coalesce(col, F.lit("")), r"\|").getItem(1)))
    return (
        F.when(F.trim(F.coalesce(col, F.lit(""))) == "", "missing")
        .when(code.rlike(NAICS_PLACEHOLDER_RE) | desc.isin("UNKNOWN", "UNK", ""), "placeholder")
        .otherwise("valid")
    )


src = pkg.select(
    F.trim(F.col("source").cast("string")).alias("node_id"),
    F.col("source_name").alias("node_name"),
    naics_status(F.col("source_naics")).alias("naics_status"),
    F.col("amount").cast("double").alias("amt"),
    F.col("volume").cast("double").alias("vol"),
    F.lit(1).alias("as_src"),
    F.lit(0).alias("as_dst"),
)
dst = pkg.select(
    F.trim(F.col("dest").cast("string")).alias("node_id"),
    F.col("dest_name").alias("node_name"),
    naics_status(F.col("dest_naics")).alias("naics_status"),
    F.col("amount").cast("double").alias("amt"),
    F.col("volume").cast("double").alias("vol"),
    F.lit(0).alias("as_src"),
    F.lit(1).alias("as_dst"),
)

pkg_nodes = (
    src.unionByName(dst)
    .groupBy("node_id")
    .agg(
        F.sum("amt").alias("amt_total"),
        F.sum("vol").alias("vol_total"),
        F.max("as_src").alias("is_src"),
        F.max("as_dst").alias("is_dst"),
        F.first("node_name", ignorenulls=True).alias("node_name"),
        F.min("naics_status").alias("naics_status"),  # 'missing' < 'placeholder' < 'valid'
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

note("G1_pkg_nodes", pkg_nodes.count())
note("G1_pkg_dollars", float(pkg_nodes.agg(F.sum("amt_total")).collect()[0][0] or 0))

# %% [markdown]
# ### G2 — identifier reconciliation
#
# Before any join: are these the same identifier space at all? `mdm_id` is
# `varchar(50)`. If PKG `source` is an account-level or hashed identifier rather than
# a party-level MDM id, the match rate will be near zero and no amount of normalisation
# will fix it — that becomes a data-engineering ask, not an analytics problem.

# %%
def id_profile(df, col, label):
    c = F.trim(F.col(col).cast("string"))
    return (
        df.select(
            F.lit(label).alias("side"),
            F.length(c).alias("id_len"),
            F.when(c.rlike(r"^\d+$"), "numeric")
            .when(c.rlike(r"^[A-Za-z0-9]+$"), "alnum")
            .otherwise("other").alias("charset"),
            F.when(c.rlike(r"^0"), 1).otherwise(0).alias("leading_zero"),
        )
        .groupBy("side", "id_len", "charset")
        .agg(F.count("*").alias("n"), F.max("leading_zero").alias("any_leading_zero"))
    )


idp = id_profile(addr, "mdm_id", "mdm_id").unionByName(
    id_profile(pkg_nodes, "node_id", "pkg_node")
).orderBy("side", F.desc("n"))
show(idp, 40)
REPORT["G2_id_profile"] = [r.asDict() for r in idp.collect()]

# %%
# G2b — test several join normalisations and report the best. This is the cell that
# actually unblocks the module.
mdm_ids = addr.select(F.trim(F.col("mdm_id").cast("string")).alias("raw")).distinct().persist()
pkg_ids = pkg_nodes.select(F.col("node_id").alias("raw")).distinct().persist()

NORMS = {
    "raw":            lambda c: c,
    "upper":          lambda c: F.upper(c),
    "strip_zeros":    lambda c: F.regexp_replace(c, r"^0+", ""),
    "digits_only":    lambda c: F.regexp_replace(c, r"[^0-9]", ""),
    "upper_alnum":    lambda c: F.regexp_replace(F.upper(c), r"[^A-Z0-9]", ""),
    "zfill18":        lambda c: F.lpad(F.regexp_replace(c, r"^0+", ""), 18, "0"),
}

n_pkg_ids = pkg_ids.count()
rows = []
for name, fn in NORMS.items():
    a = mdm_ids.select(fn(F.col("raw")).alias("k")).distinct()
    b = pkg_ids.select(fn(F.col("raw")).alias("k")).distinct()
    m = b.join(a, "k", "left_semi").count()
    rows.append({"norm": name, "pkg_ids_matched": m, "pct": round(100 * m / max(n_pkg_ids, 1), 3)})
    print(f"  {name:14s} {m:>12,}  {rows[-1]['pct']:>7.3f}%")

REPORT["G2b_join_normalisations"] = rows
BEST_NORM = max(rows, key=lambda r: r["pkg_ids_matched"])["norm"]
note("G2b_best_norm", BEST_NORM)
note("G2b_best_match_pct", max(r["pct"] for r in rows))

if max(r["pct"] for r in rows) < 50:
    print("\n  !! Fewer than half of PKG nodes resolve to an MDM party under any "
          "normalisation. The two sides are probably different identifier spaces "
          "(account vs party, or one side hashed). Stop and confirm the intended "
          "join path before building on this.")

# %%
norm_fn = NORMS[BEST_NORM]

geo_party = (
    addr_g.groupBy(norm_fn(F.trim(F.col("mdm_id").cast("string"))).alias("join_key"))
    .agg(
        F.max("_has_geo").alias("has_geo"),
        F.count("*").alias("n_addr"),
        F.max(F.upper(F.trim("state_or_province"))).alias("state_any"),
        F.max(F.substring(F.split(F.trim(F.coalesce(F.col("zip_cd"), F.lit(""))), "-").getItem(0), 1, 3)).alias("zip3_any"),
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

pkg_j = pkg_nodes.withColumn("join_key", norm_fn(F.col("node_id")))
joined = pkg_j.join(geo_party, "join_key", "left").persist(StorageLevel.MEMORY_AND_DISK)

# %%
# G3 — the three-population coverage table. This is the headline number.
cov = joined.agg(
    F.count("*").alias("pkg_nodes"),
    F.sum(F.when(F.col("n_addr").isNotNull(), 1).otherwise(0)).alias("matched_to_mdm"),
    F.sum(F.coalesce(F.col("has_geo"), F.lit(0))).alias("with_coords"),
    F.sum("amt_total").alias("dollars_total"),
    F.sum(F.when(F.col("has_geo") == 1, F.col("amt_total")).otherwise(0.0)).alias("dollars_geocoded"),
    F.sum("vol_total").alias("volume_total"),
    F.sum(F.when(F.col("has_geo") == 1, F.col("vol_total")).otherwise(0.0)).alias("volume_geocoded"),
).collect()[0].asDict()

cov["pct_nodes_matched"] = round(100 * cov["matched_to_mdm"] / max(cov["pkg_nodes"], 1), 2)
cov["pct_nodes_geocoded"] = round(100 * cov["with_coords"] / max(cov["pkg_nodes"], 1), 2)
cov["pct_dollars_geocoded"] = round(100 * cov["dollars_geocoded"] / max(cov["dollars_total"], 1), 2)
cov["pct_volume_geocoded"] = round(100 * cov["volume_geocoded"] / max(cov["volume_total"], 1), 2)

print(json.dumps(cov, indent=2, default=str))
REPORT["G3_coverage"] = cov

# also: how much of the MDM book never appears in the graph at all
mdm_only = geo_party.join(pkg_j.select("join_key").distinct(), "join_key", "left_anti").count()
note("G3_mdm_parties_not_in_pkg", mdm_only)

# %% [markdown]
# **Reading G3.**
#
# - `pct_dollars_geocoded` **above** `pct_nodes_geocoded` → coverage is concentrated in
#   large customers, and the geo modules are more trustworthy than the raw node rate
#   suggests. Brief the dollar number.
# - `pct_dollars_geocoded` **below** `pct_nodes_geocoded` → geocoding is best on small
#   nodes, the modules are weaker than they look, and F4 becomes urgent.
# - `pct_nodes_matched` well above `pct_nodes_geocoded` → the parties are in MDM, they
#   just have no usable coordinate. That's an imputation problem (M3), which is
#   tractable.
# - `pct_nodes_matched` low → an identifier or entity-scope problem, which is not.

# %%
# G4 — is geo missingness informative? (M2)
# Compare graph-side structure across geo-present vs geo-missing PKG nodes.
m2 = (
    joined.withColumn(
        "geo_pop",
        F.when(F.col("n_addr").isNull(), "not_in_mdm")
        .when(F.col("has_geo") == 1, "geo_present")
        .otherwise("geo_missing"),
    )
    .groupBy("geo_pop")
    .agg(
        F.count("*").alias("n_nodes"),
        F.round(F.sum("amt_total") / 1e6, 1).alias("amt_musd"),
        F.round(F.avg("amt_total"), 1).alias("mean_amt"),
        F.round(F.expr("percentile_approx(amt_total, 0.5)"), 1).alias("median_amt"),
        F.round(F.avg("vol_total"), 2).alias("mean_vol"),
        F.round(F.avg("is_src"), 3).alias("frac_as_payer"),
        F.round(F.avg("is_dst"), 3).alias("frac_as_payee"),
    )
    .orderBy("geo_pop")
)
show(m2)
REPORT["G4_missingness_informative"] = [r.asDict() for r in m2.collect()]

# %%
# G5 — geo coverage x NAICS status. Two missingness phenomena, or one?
m2b = (
    joined.withColumn(
        "geo_pop",
        F.when(F.col("n_addr").isNull(), "not_in_mdm")
        .when(F.col("has_geo") == 1, "geo_present")
        .otherwise("geo_missing"),
    )
    .groupBy("naics_status", "geo_pop")
    .agg(F.count("*").alias("n_nodes"), F.round(F.sum("amt_total") / 1e6, 1).alias("amt_musd"))
    .orderBy("naics_status", "geo_pop")
)
show(m2b, 30)
REPORT["G5_geo_x_naics"] = [r.asDict() for r in m2b.collect()]

# %% [markdown]
# **Why G5 matters.** If geo-missing and NAICS-placeholder are the *same* nodes, then
# there is one enrichment gap, not two, and one fix closes both. If they're
# independent, the imputation ladder (M3) can lean on NAICS as a conditioning prior —
# which it cannot do if the two are collinear.

# %% [markdown]
# ### G6 — F4: the Pittsburgh test, done honestly
#
# Coverage-vs-distance cannot be computed from coordinates, because the nodes of
# interest are exactly the ones without coordinates. Use **ZIP3 as the location proxy**
# — it survives when lat/lon does not — and place each ZIP3 using the median coordinate
# of the geocoded rows that share it. Self-referential, but valid: it locates the
# non-geocoded rows without using their own (missing) geocode.

# %%
zip3_centroid = (
    addr_g.filter(F.col("_has_geo") == 1)
    .withColumn("zip3", F.substring(F.split(F.trim(F.coalesce(F.col("zip_cd"), F.lit(""))), "-").getItem(0), 1, 3))
    .filter(F.col("zip3").rlike(r"^\d{3}$"))
    .groupBy("zip3")
    .agg(
        F.expr("percentile_approx(_lat, 0.5)").alias("z_lat"),
        F.expr("percentile_approx(_lon, 0.5)").alias("z_lon"),
        F.count("*").alias("n_ref"),
    )
    .filter(F.col("n_ref") >= 20)
    .withColumn("km_from_pit", haversine_km(F.col("z_lat"), F.col("z_lon"), F.lit(PIT_LAT), F.lit(PIT_LON)))
    .persist()
)
note("G6_zip3_reference_points", zip3_centroid.count())

# %%
ring = (
    F.when(F.col("km_from_pit").isNull(), "9_no_zip3")
    .when(F.col("km_from_pit") < 50, "0_under_50km")
    .when(F.col("km_from_pit") < 150, "1_50_150km")
    .when(F.col("km_from_pit") < 400, "2_150_400km")
    .when(F.col("km_from_pit") < 1000, "3_400_1000km")
    .otherwise("4_over_1000km")
)

zip3_ref = zip3_centroid.select(F.col("zip3").alias("zip3_any"), "km_from_pit")

g6 = (
    joined.join(zip3_ref, "zip3_any", "left")
    .withColumn("ring", ring)
    .groupBy("ring")
    .agg(
        F.count("*").alias("pkg_nodes"),
        F.round(100 * F.avg(F.coalesce(F.col("has_geo"), F.lit(0))), 2).alias("pct_geocoded"),
        F.sum("amt_total").alias("_amt"),
    )
    .withColumn("amt_musd", F.round(F.col("_amt") / 1e6, 1))
    .withColumn(
        "pct_of_dollars",
        F.round(100 * F.col("_amt") / F.sum("_amt").over(Window.partitionBy()), 2),
    )
    .drop("_amt")
    .orderBy("ring")
)
show(g6)
REPORT["G6_pittsburgh_rings"] = [r.asDict() for r in g6.collect()]

# %% [markdown]
# **Reading G6.** Two numbers, and they answer different questions.
#
# `pct_of_dollars` in ring `0_under_50km` is the actual test of the
# Pittsburgh-concentration belief — on the graph-visible, dollar-weighted population,
# which is the only version of the claim worth putting in a deck.
#
# `pct_geocoded` declining with distance is the coverage artifact. If local
# relationships are older and branch-originated, their geocodes are cleaner, and any
# node-count map will exaggerate local concentration by construction. If that gradient
# is present, every downstream geographic statistic needs a coverage-adjusted twin.

# %%
# G7 — geo coverage by role (optional; skip if the roles table is unavailable)
try:
    roles = spark.table(PKG_ROLES_TABLE)
    roles.printSchema()
    role_col = next((c for c in roles.columns if "role" in c.lower()), None)
    id_col = next((c for c in roles.columns if c.lower() in ("source", "node_id", "customer_id", "mdm_id")), None)
    if role_col and id_col:
        rj = (
            joined.join(
                roles.select(
                    norm_fn(F.trim(F.col(id_col).cast("string"))).alias("join_key"),
                    F.col(role_col).alias("role"),
                ).distinct(),
                "join_key",
                "left",
            )
            .groupBy("role")
            .agg(
                F.count("*").alias("n_nodes"),
                F.round(100 * F.avg(F.coalesce(F.col("has_geo"), F.lit(0))), 2).alias("pct_geocoded"),
                F.round(F.sum("amt_total") / 1e6, 1).alias("amt_musd"),
            )
            .orderBy(F.desc("n_nodes"))
        )
        show(rj, 40)
        REPORT["G7_geo_x_role"] = [r.asDict() for r in rj.collect()]
except Exception as e:
    print("roles join skipped:", e)

# %% [markdown]
# ## H. Build the extract
#
# Two files:
#
# 1. **`pkg_geo_address_all`** — every address row, with derived flags. Keeps the audit
#    trail and is what the D1 change log will diff against next month.
# 2. **`pkg_geo_node`** — one row per party (`addr_rank = 1`), joined to the PKG node
#    set. This is the metrics-ready file.
#
# Raw `addr_line_*` text is deliberately **not** carried into either file — the derived
# flags and `addr_norm_hash` preserve everything the placeholder screen needs at a much
# smaller PII surface. If a one-time entity-resolution pass needs the raw strings, make
# that a separate restricted extract rather than widening this one.

# %%
addr_type_u = F.upper(F.trim(F.coalesce(F.col("addr_type"), F.lit(""))))
rec_type_u = F.upper(F.trim(F.coalesce(F.col("addr_loc_rec_type"), F.lit(""))))

addr_type_rank = F.coalesce(
    *[F.when(addr_type_u == k, F.lit(v)) for k, v in ADDR_TYPE_PRIORITY.items()],
    F.lit(99),
)
rec_type_rank = F.coalesce(
    *[F.when(rec_type_u == k, F.lit(v)) for k, v in REC_TYPE_PRIORITY.items()],
    F.lit(9),
)

geo_status = (
    F.when(F.col("_has_geo") == 0, "missing")
    .when(rec_type_u.isin("POSTOFFICEBOX", "GENERALDELIVERY"), "placeholder")
    .when(u1.rlike(r"(^| )P ?\.? ?O ?\.? ?BOX"), "placeholder")
    .when(F.col("_coord_decimals") <= 2, "placeholder")   # ~1 km: a centroid, not a point
    .otherwise("valid")
)

ext = (
    addr_g.withColumn("_coord_decimals", F.least(dp("latitude_degrees"), dp("longitude_degrees")))
    .withColumn("geo_status", geo_status)
    .select(
        F.trim(F.col("mdm_id").cast("string")).alias("mdm_id"),
        F.trim(F.col("mdm_address_id").cast("string")).alias("mdm_address_id"),
        norm_fn(F.trim(F.col("mdm_id").cast("string"))).alias("join_key"),
        addr_type_u.alias("addr_type"),
        rec_type_u.alias("addr_loc_rec_type"),
        F.col("_lat").alias("lat"),
        F.col("_lon").alias("lon"),
        F.col("geo_status"),
        F.when(rec_type_u == "HIGHRISE", 1).otherwise(0).alias("shared_structure"),
        F.col("_coord_decimals").alias("coord_decimals"),
        F.when(F.col("_has_geo") == 1, coord_key).alias("coord_key"),
        F.split(F.trim(F.coalesce(F.col("zip_cd"), F.lit(""))), "-").getItem(0).alias("zip5"),
        F.when(zip_t.rlike(r"^\d{5}-\d{4}$"), F.split(zip_t, "-").getItem(1)).alias("zip4"),
        F.upper(F.trim(F.coalesce(F.col("state_or_province"), F.lit("")))).alias("state"),
        F.upper(F.trim(F.coalesce(F.col("city"), F.lit("")))).alias("city"),
        F.upper(F.trim(F.coalesce(F.col("addr_country"), F.lit("")))).alias("country"),
        stable_hash(F.concat_ws("|", F.regexp_replace(u1, "[^A-Z0-9]", ""),
                                F.split(F.trim(F.coalesce(F.col("zip_cd"), F.lit(""))), "-").getItem(0))
                    ).cast("string").alias("addr_norm_hash"),
        F.when(u2.rlike(r"^(APT|UNIT|#|TRLR|LOT|SPC)"), "residential")
         .when(u2.rlike(r"^(STE|SUITE|FL |FLOOR|RM |DEPT|BLDG)"), "commercial")
         .otherwise(F.lit(None).cast("string")).alias("addr_unit_type"),
        F.when(u1.rlike(REG_AGENT_RE), 1).otherwise(0).alias("flag_reg_agent"),
        F.when(u1.rlike(r"(^| )(C/O|C O |ATTN|PMB)"), 1).otherwise(0).alias("flag_care_of"),
        F.when(u1.rlike(r"(^| )PNC( |$)"), 1).otherwise(0).alias("flag_pnc_address"),
        F.col("addr_cleansed_date").cast("string").alias("addr_cleansed_date"),
        F.lit(str(MAX_SNAP)).alias("snapshot"),
        addr_type_rank.alias("_at_rank"),
        rec_type_rank.alias("_rt_rank"),
    )
    .withColumn("n_addresses_per_mdm_id", F.count("*").over(Window.partitionBy("mdm_id")))
    .withColumn(
        "addr_rank",
        F.row_number().over(
            Window.partitionBy("mdm_id").orderBy(
                F.when(F.col("geo_status") == "valid", 0)
                 .when(F.col("geo_status") == "placeholder", 1).otherwise(2),
                F.col("_at_rank"),
                F.col("_rt_rank"),
                F.col("addr_cleansed_date").desc_nulls_last(),
                F.col("mdm_address_id"),
            )
        ),
    )
    .drop("_at_rank", "_rt_rank")
    .persist(StorageLevel.MEMORY_AND_DISK)
)

show(ext.limit(15), 15, truncate=28)
note("H_extract_rows", ext.count())

# %%
# H1 — flag any addr_type that fell through to rank 99 and actually influenced a pick
fell_through = (
    ext.filter((F.col("addr_rank") == 1) & (F.col("n_addresses_per_mdm_id") > 1))
    .groupBy("addr_type")
    .agg(F.count("*").alias("n_selected"))
    .orderBy(F.desc("n_selected"))
)
show(fell_through, 30)
if unmapped:
    print(f"  !! ADDR_TYPE_PRIORITY is still missing {unmapped} — the selection above "
          f"is provisional.")

# %%
# H2 — the node-level file: one row per party, restricted to PKG-visible nodes,
# carrying the graph-side weights so the geo module never has to re-derive them.
node_geo = (
    ext.filter(F.col("addr_rank") == 1)
    .drop("addr_rank")
    .join(
        pkg_j.select(
            "join_key",
            F.col("node_id").alias("pkg_node_id"),
            "amt_total", "vol_total", "is_src", "is_dst", "naics_status",
        ),
        "join_key",
        "inner",
    )
    .drop("join_key")
    .persist(StorageLevel.MEMORY_AND_DISK)
)

n_node = node_geo.count()
n_dupe = node_geo.groupBy("pkg_node_id").count().filter(F.col("count") > 1).count()
note("H2_node_rows", n_node)
note("H2_duplicate_pkg_nodes", n_dupe)
assert n_dupe == 0, "pkg_node_id is not unique in node_geo — fix the selection rule before writing."

show(
    node_geo.groupBy("geo_status").agg(
        F.count("*").alias("n_nodes"),
        F.round(F.sum("amt_total") / 1e6, 1).alias("amt_musd"),
    ).orderBy("geo_status")
)

# %%
# H3 — write. Parquet is strictly better here (cuDF reads it natively and it keeps
# dtypes), but CSV is written too since that is the PKG convention. The single-CSV
# path pulls to the driver, so it is gated on row count.
import os

os.makedirs(OUT_DIR_LOCAL, exist_ok=True)

if OUT_DIR_HDFS:
    ext.drop("join_key").write.mode("overwrite").parquet(f"{OUT_DIR_HDFS}/pkg_geo_address_all")
    node_geo.write.mode("overwrite").parquet(f"{OUT_DIR_HDFS}/pkg_geo_node")
    print("parquet written to", OUT_DIR_HDFS)

MAX_DRIVER_ROWS = 15_000_000
if n_node <= MAX_DRIVER_ROWS:
    pdf = node_geo.toPandas()
    # Force string dtypes: mdm_id is 18+ digits and will lose precision as float64;
    # zip5 will lose its leading zero (08861 -> 8861).
    for c in ["mdm_id", "mdm_address_id", "pkg_node_id", "zip5", "zip4", "addr_norm_hash"]:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("string")
    out_csv = f"{OUT_DIR_LOCAL}/pkg_geo_node_{MAX_SNAP}.csv"
    pdf.to_csv(out_csv, index=False)
    print(f"wrote {len(pdf):,} rows -> {out_csv}")
    print(pdf.dtypes)
elif OUT_DIR_HDFS:
    print(f"{n_node:,} rows exceeds MAX_DRIVER_ROWS; writing partitioned CSV instead.")
    node_geo.write.mode("overwrite").option("header", True).csv(f"{OUT_DIR_HDFS}/pkg_geo_node_csv")
else:
    raise RuntimeError(
        f"{n_node:,} rows is too large to pull to the driver and OUT_DIR_HDFS is not set. "
        f"Set OUT_DIR_HDFS in cell 0, or raise MAX_DRIVER_ROWS if the driver can take it."
    )

# %% [markdown]
# ## I. Report block
#
# Run this last and paste the printed JSON back. It's the whole profile in one block —
# enough to fix the `addr_type` priority rule, finalise the `geo_status` derivation,
# and decide whether the imputation ladder (M3) is worth building now or whether the
# identifier join needs solving first.

# %%
print(json.dumps(REPORT, indent=2, default=str))

# %%
with open(f"{OUT_DIR_LOCAL}/geo_profile_report_{MAX_SNAP}.json", "w") as fh:
    json.dump(REPORT, fh, indent=2, default=str)
print("report written")

# %% [markdown]
# ---
#
# ## What I need back, in priority order
#
# 1. **`G2b_join_normalisations` and `G3_coverage`** — everything else is contingent on
#    these. If the match rate is low, the next conversation is about identifiers, not
#    geography.
# 2. **`B1_addr_type`** — the observed domain, so `ADDR_TYPE_PRIORITY` can be set
#    properly instead of guessed.
# 3. **`C1_coord_state`** — specifically the size of `4_non_numeric`,
#    `5_null_island`, and `7_positive_lon_dropped_sign`.
# 4. **`G6_pittsburgh_rings`** — both columns. `pct_of_dollars` answers the business
#    question; `pct_geocoded` tells us whether the answer is trustworthy.
# 5. **`D1_top_clusters` and `D2_top_shared_addresses`** — the top 20 of each. The
#    identity of the biggest clusters usually explains itself at a glance.
# 6. **The full column list from 0.1** — particularly whether an active/primary flag or
#    effective dates exist, since those change the selection rule.
#
# Two things worth deciding regardless of what the numbers say:
#
# - If `n_snapshots_retained` is 1, start archiving a dated copy of
#   `pkg_geo_address_all` every month now. The registered-point change log (D1) cannot
#   be reconstructed later.
# - `geo_status = placeholder` currently absorbs PO boxes, general delivery, and
#   anything geocoded to ≤2 decimals. That last rule is a judgement call — if C2 shows
#   a large low-precision population, it may deserve its own status value rather than
#   being pooled with PO boxes, since the two fail in different ways.
