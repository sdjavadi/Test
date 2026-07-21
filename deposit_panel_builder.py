"""
deposit_panel_builder.py — Tier 0 panel assembly for the deposit ↔ PKG joint analysis.

Builds four tables from (a) the liquidity-attrition lab table (wide balances) and
(b) the PKG monthly node metrics (one version of the ablation ladder):

  1. {prefix}_deposit_panel   — (cust_pwr_id, month) long deposit panel with recomputed
                                rolling 3v6 decline flags, eligibility, event markers,
                                and deposit-side features (dep_* columns)
  2. {prefix}_joint_panel     — deposit panel LEFT JOIN pkg node metrics (pkg_* columns);
                                deposit months beyond the graph window are retained with
                                null metrics (needed for the Nov-2025 freeze test later)
  3. {prefix}_customer_dim    — one row per customer: activity bounds, first event month,
                                C2C visibility index + tier, account-closure ratio
  4. {prefix}_coverage        — per-month join coverage in both directions

Join-key model (revised): the metrics Hive table already carries a `cust_pwr_id`
column (attached at Hive-write time). That column — NOT `node` — is the join key.
Most metric rows have NULL cust_pwr_id (unmapped nodes: counterparty-like, internal,
or simply without deposit relationships); they are filtered out before joining.
Expectation: <100K mapped nodes ⇒ the joint panel is graph-sparse by construction
and the `invisible` visibility tier will dominate. That sparsity is a finding
(the counterparty-gap story), not a bug.

Design decisions encoded here (see chat discussion):
  * decline flags recomputed at EVERY month from the wide bal_* columns — the table's
    prior_avg / recent_avg / pct_change (anchored to latest_month) are ignored
  * flag requires FULL windows (3 non-null recent, 6 non-null prior) and a prior_avg
    dollar floor to keep tiny-balance noise out of the event set
  * zero-vs-null semantics unresolved until EDA — both surfaced in diagnostics;
    nulls treated as missing, zeros as real values inside the active span
  * metrics grain check on (cust_pwr_id, month): several nodes may map to one power id;
    default policy is fail-fast, optional keep-max-strength representative
  * join guard runs graph→deposit: mapped nodes carry a power id by construction, so
    they should overwhelmingly appear in the deposit table — a near-zero rate means
    a key format/dtype mismatch (2026-07 typing-join incident convention), not coverage
  * all network metric columns prefixed pkg_, all derived deposit features prefixed dep_

OPEN ITEMS (confirm before production run):
  * cfg.node_metrics_table — actual Hive table name for the node metrics
  * cfg.min_prior_avg floor (default $10K — placeholder)
  * time_key format in the metrics table (normalizer handles YYYY-MM / YYYYMM / date)
  * deposit-table grain: assumed one row per cust_pwr_id; job fails fast if not
  * metrics mapping grain: is (cust_pwr_id, month) unique after the null filter?
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


# ----------------------------------------------------------------------------- config

@dataclass
class PanelConfig:
    # -- sources
    deposit_table: str = "bdahd01p_dldsi1_dsi_lab.pl43379_dsi_liq_attrition_base"
    node_metrics_source: str = "hive"            # "hive" | "parquet"
    node_metrics_table: str = "bdahd01p_dldsi1_dsi_lab.pl43379_pkg_node_metrics"  # TODO confirm
    node_metrics_parquet: str = "/data/pkg/metrics/node/*.parquet"                # if source=parquet
    version: str = "P99_9"                       # primary analytical ladder tier

    # -- windows
    deposit_start: str = "2024-01"
    graph_start: str = "2024-01"
    graph_end: str = "2025-11"                   # last PKG snapshot

    # -- decline definition (mirrors upstream 3v6 @ 30%, recomputed monthly)
    recent_window: int = 3
    prior_window: int = 6
    decline_threshold: float = 0.30
    min_prior_avg: float = 10_000.0              # eligibility floor; TODO calibrate

    # -- metrics mapping grain policy when multiple nodes share (cust_pwr_id, month)
    on_duplicate_mapping: str = "fail"           # "fail" | "keep_max_strength"

    # -- join guard (direction: mapped graph nodes → deposit table)
    join_guard_sample: int = 50_000
    join_guard_min_rate: float = 0.05            # below ⇒ key/dtype mismatch, raise
    join_guard_warn_rate: float = 0.70           # mapped nodes *should* be deposit customers

    # -- outputs
    out_db: str = "bdahd01p_dldsi1_dsi_lab"
    out_prefix: str = "pl43379_dsi_liq_pkg"
    write_mode: str = "overwrite"

    # metric columns to carry into the joint panel; empty = all
    metric_cols: list[str] = field(default_factory=list)


BAL_RE = re.compile(r"^bal_(\d{4})_(\d{2})$")


# ----------------------------------------------------------------------------- session

def get_spark(app_name: str = "deposit_panel_builder") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .enableHiveSupport()
        .config("spark.sql.shuffle.partitions", "400")
        .getOrCreate()
    )
    # known Cloudera hazard: driver/worker python mismatch — surface it early
    print(f"[env] driver python: {sys.version.split()[0]}")
    for k in ("spark.pyspark.python", "spark.yarn.appMasterEnv.PYSPARK_PYTHON"):
        v = spark.conf.get(k, None)
        if v:
            print(f"[env] {k} = {v}")
    return spark


# ----------------------------------------------------------------------------- helpers

def _month_idx(col):
    """Months since 1970-01 — dense integer index safe for rowsBetween windows."""
    d = F.to_date(F.concat(col, F.lit("-01")))
    return (F.year(d) * 12 + F.month(d) - 1).cast("int")


def _normalize_time_key(col):
    """Accept 'YYYY-MM', 'YYYYMM', or date-like; emit 'YYYY-MM'."""
    s = F.col(col).cast("string")
    return (
        F.when(s.rlike(r"^\d{4}-\d{2}$"), s)
        .when(s.rlike(r"^\d{6}$"), F.concat(s.substr(1, 4), F.lit("-"), s.substr(5, 2)))
        .otherwise(F.date_format(F.to_date(s), "yyyy-MM"))
    )


def assert_unique_grain(df: DataFrame, key: str, what: str) -> None:
    n, k = df.count(), df.select(key).distinct().count()
    print(f"[grain] {what}: rows={n:,} distinct {key}={k:,}")
    if n != k:
        (
            df.groupBy(key).count().filter("count > 1")
            .orderBy(F.desc("count")).limit(20).show(truncate=False)
        )
        raise ValueError(
            f"{what} is not unique on {key} ({n - k:,} extra rows). "
            "Resolve grain (rltn vs cust rollup?) before building the panel."
        )


def qa_describe(df: DataFrame, cols: list[str], label: str) -> None:
    """Manifest QA convention: eyeball .describe() on every new metric family."""
    print(f"\n[qa] describe — {label}")
    df.select(*cols).describe().show(truncate=False)


# ----------------------------------------------------------------------------- stage 1: melt

def melt_deposits(df: DataFrame) -> DataFrame:
    pairs = sorted(
        (m.group(1) + "-" + m.group(2), c)
        for c in df.columns
        for m in [BAL_RE.match(c)]
        if m
    )
    if not pairs:
        raise ValueError("no bal_YYYY_MM columns found in deposit table")
    print(f"[melt] {len(pairs)} balance columns: {pairs[0][0]} … {pairs[-1][0]}")

    stack_args = ", ".join(f"'{mo}', cast(`{col}` as double)" for mo, col in pairs)
    long = (
        df.select(
            F.trim(F.col("cust_pwr_id").cast("string")).alias("cust_pwr_id"),
            F.expr(f"stack({len(pairs)}, {stack_args}) as (month, balance)"),
        )
        .withColumn("month_idx", _month_idx(F.col("month")))
    )
    return long  # dense by construction: one row per customer per bal column


# ----------------------------------------------------------------------------- stage 2: activity bounds + diagnostics

def add_activity_bounds(long: DataFrame) -> DataFrame:
    active = F.col("balance").isNotNull() & (F.col("balance") != 0)
    w_cust = Window.partitionBy("cust_pwr_id")

    return (
        long.withColumn("_active_idx", F.when(active, F.col("month_idx")))
        .withColumn("first_active_idx", F.min("_active_idx").over(w_cust))
        .withColumn("last_active_idx", F.max("_active_idx").over(w_cust))
        .drop("_active_idx")
        .withColumn(
            "balance_status",
            F.when(F.col("first_active_idx").isNull(), F.lit("never_active"))
            .when(F.col("month_idx") < F.col("first_active_idx"), F.lit("pre_first_activity"))
            .when(F.col("month_idx") > F.col("last_active_idx"), F.lit("post_last_activity"))
            .otherwise(F.lit("active_span")),
        )
    )


def zero_null_diagnostics(long: DataFrame) -> None:
    """Zero-vs-null semantics are unresolved — print the evidence, decide in EDA."""
    print("\n[diag] balance null/zero/negative shares by month")
    (
        long.groupBy("month")
        .agg(
            F.count("*").alias("n"),
            F.avg(F.col("balance").isNull().cast("int")).alias("null_share"),
            F.avg((F.col("balance") == 0).cast("int")).alias("zero_share"),
            F.avg((F.col("balance") < 0).cast("int")).alias("neg_share"),
        )
        .orderBy("month")
        .show(200, truncate=False)
    )


# ----------------------------------------------------------------------------- stage 3: rolling features + decline flags

def add_deposit_features(long: DataFrame, cfg: PanelConfig) -> DataFrame:
    rw, pw = cfg.recent_window, cfg.prior_window
    w = Window.partitionBy("cust_pwr_id").orderBy("month_idx")
    w_recent = w.rowsBetween(-(rw - 1), 0)
    w_prior = w.rowsBetween(-(rw + pw - 1), -rw)
    w_cum = w.rowsBetween(Window.unboundedPreceding, 0)

    log_bal = F.when(F.col("balance") > 0, F.log(F.col("balance")))

    df = (
        long
        .withColumn("dep_recent_avg", F.avg("balance").over(w_recent))
        .withColumn("dep_recent_n", F.count("balance").over(w_recent))
        .withColumn("dep_prior_avg", F.avg("balance").over(w_prior))
        .withColumn("dep_prior_n", F.count("balance").over(w_prior))
        .withColumn("_log_bal", log_bal)
        .withColumn("dep_dlog_1m", F.col("_log_bal") - F.lag("_log_bal", 1).over(w))
        .withColumn("dep_dlog_3m", F.col("_log_bal") - F.lag("_log_bal", 3).over(w))
        .withColumn("_run_max", F.max("balance").over(w_cum))
        .withColumn(
            "dep_drawdown",
            F.when(F.col("_run_max") > 0, F.col("balance") / F.col("_run_max") - 1.0),
        )
        .drop("_log_bal", "_run_max")
    )

    eligible = (
        (F.col("dep_recent_n") == rw)
        & (F.col("dep_prior_n") == pw)
        & (F.col("dep_prior_avg") >= cfg.min_prior_avg)
    )
    pct = (F.col("dep_recent_avg") - F.col("dep_prior_avg")) / F.col("dep_prior_avg")

    df = (
        df.withColumn("dep_eligible", eligible.cast("int"))
        .withColumn("dep_pct_change", F.when(eligible, pct))
        .withColumn(
            "dep_decline_flag",
            F.when(eligible, (pct <= -cfg.decline_threshold).cast("int")),
        )
    )

    # event start = flag turns on (nulls treated as 0 for the transition only)
    flag0 = F.coalesce(F.col("dep_decline_flag"), F.lit(0))
    df = df.withColumn(
        "dep_is_event_start",
        ((flag0 == 1) & (F.coalesce(F.lag(flag0, 1).over(w), F.lit(0)) == 0)).cast("int"),
    )
    w_cust = Window.partitionBy("cust_pwr_id")
    return df.withColumn(
        "dep_first_event_month",
        F.min(F.when(F.col("dep_is_event_start") == 1, F.col("month"))).over(w_cust),
    )


# ----------------------------------------------------------------------------- stage 4: PKG metrics

def load_node_metrics(spark: SparkSession, cfg: PanelConfig) -> DataFrame:
    if cfg.node_metrics_source == "hive":
        m = spark.table(cfg.node_metrics_table)
    else:
        m = spark.read.parquet(cfg.node_metrics_parquet)

    m = m.filter(F.col("version") == cfg.version).withColumn(
        "month", _normalize_time_key("time_key")
    )

    # cust_pwr_id was attached at Hive-write time and is the ONLY join key.
    # NULL / empty ⇒ unmapped node (no deposit relationship) ⇒ excluded here.
    n_total = m.count()
    m = m.withColumn(
        "cust_pwr_id", F.trim(F.col("cust_pwr_id").cast("string"))
    ).filter(F.col("cust_pwr_id").isNotNull() & (F.col("cust_pwr_id") != ""))
    n_mapped_rows = m.count()
    n_mapped_ids = m.select("cust_pwr_id").distinct().count()
    print(
        f"[metrics] {cfg.version}: rows total={n_total:,}, "
        f"mapped rows={n_mapped_rows:,} ({n_mapped_rows / max(n_total, 1):.1%}), "
        f"distinct mapped cust_pwr_id={n_mapped_ids:,}"
    )

    m = resolve_mapping_grain(m, cfg)

    m = m.drop("time_key", "version")
    if cfg.metric_cols:
        keep = ["cust_pwr_id", "month", "node"] + cfg.metric_cols
        m = m.select(*[c for c in keep if c in m.columns])

    # prefix every metric column (node id kept as pkg_node for traceability)
    for c in m.columns:
        if c not in ("cust_pwr_id", "month"):
            m = m.withColumnRenamed(c, f"pkg_{c}")
    return m


def resolve_mapping_grain(m: DataFrame, cfg: PanelConfig) -> DataFrame:
    """Several graph nodes may share one power id. Default: fail fast and decide.
    Optional: keep the max-strength node per (cust_pwr_id, month) as representative.
    NOTE: a sum/aggregate rollup is deliberately NOT offered — strengths would add
    but spectral/percentile metrics (pagerank, peer ranks) cannot be summed sensibly.
    If multi-node customers matter, that becomes its own design discussion."""
    dupes = m.groupBy("cust_pwr_id", "month").count().filter("count > 1")
    n_dup = dupes.count()
    if n_dup == 0:
        print("[grain] metrics mapping is unique on (cust_pwr_id, month)")
        return m
    print(f"[grain][WARN] {n_dup:,} (cust_pwr_id, month) pairs map to multiple nodes")
    dupes.orderBy(F.desc("count")).limit(20).show(truncate=False)
    if cfg.on_duplicate_mapping == "keep_max_strength":
        w = Window.partitionBy("cust_pwr_id", "month").orderBy(
            F.desc_nulls_last("strength"), F.asc("node")
        )
        out = m.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")
        print("[grain] resolved: keeping max-strength node per (cust_pwr_id, month)")
        return out
    raise ValueError(
        "metrics mapping not unique on (cust_pwr_id, month); set "
        "on_duplicate_mapping='keep_max_strength' or resolve upstream."
    )


def join_guard(deposit_long: DataFrame, metrics: DataFrame, cfg: PanelConfig) -> None:
    """Direction: mapped graph nodes → deposit table. A mapped node carries a
    power id by construction, so the overwhelming majority should exist in the
    deposit table. Near-zero ⇒ key format/dtype mismatch ⇒ raise (2026-07
    incident convention). Moderately low ⇒ warn (deposit table scope narrower
    than the mapping — worth understanding, not fatal)."""
    mapped = metrics.select("cust_pwr_id").distinct().limit(cfg.join_guard_sample)
    n = mapped.count()
    matched = mapped.join(
        deposit_long.select("cust_pwr_id").distinct(), "cust_pwr_id", "left_semi"
    ).count()
    rate = matched / max(n, 1)
    print(f"[guard] mapped-node→deposit match rate on {n:,} sampled ids: {rate:.1%}")
    if rate < cfg.join_guard_min_rate:
        raise ValueError(
            f"match rate {rate:.1%} < {cfg.join_guard_min_rate:.0%} — almost certainly "
            "an ID format/dtype mismatch between the mapping and the deposit table."
        )
    if rate < cfg.join_guard_warn_rate:
        print(
            f"[guard][WARN] match rate {rate:.1%} below {cfg.join_guard_warn_rate:.0%} — "
            "many mapped nodes lack rows in the attrition base; check table scope "
            "(account types? relationship filter?) before trusting coverage numbers."
        )


# ----------------------------------------------------------------------------- stage 5: visibility index

def build_customer_dim(
    deposit_panel: DataFrame, metrics: DataFrame, wide: DataFrame, cfg: PanelConfig
) -> DataFrame:
    overlap = deposit_panel.filter(
        (F.col("month") >= cfg.graph_start) & (F.col("month") <= cfg.graph_end)
    )
    j = overlap.join(
        metrics.select("cust_pwr_id", "month", "pkg_strength"),
        ["cust_pwr_id", "month"],
        "left",
    )

    per_cust = j.groupBy("cust_pwr_id").agg(
        F.sum((F.col("balance_status") == "active_span").cast("int")).alias(
            "months_dep_active_overlap"
        ),
        F.sum(F.col("pkg_strength").isNotNull().cast("int")).alias("months_in_graph"),
        F.expr(
            "percentile_approx(case when dep_recent_avg > 0 "
            "then pkg_strength / dep_recent_avg end, 0.5)"
        ).alias("median_c2c_intensity"),
        F.max("dep_first_event_month").alias("dep_first_event_month"),
        F.min("first_active_idx").alias("first_active_idx"),
        F.max("last_active_idx").alias("last_active_idx"),
    )

    per_cust = per_cust.withColumn(
        "graph_presence_rate",
        F.when(
            F.col("months_dep_active_overlap") > 0,
            F.col("months_in_graph") / F.col("months_dep_active_overlap"),
        ),
    )

    # composite visibility: mean pct_rank of presence + intensity, terciled among
    # graph-seen customers; never-seen customers form the dominant `invisible` tier
    # (expected: <100K mapped nodes vs. the full deposit population)
    seen = per_cust.filter(F.col("months_in_graph") > 0)
    w_pres = Window.orderBy("graph_presence_rate")
    w_int = Window.orderBy("median_c2c_intensity")
    seen = (
        seen.withColumn("_pr", F.percent_rank().over(w_pres))
        .withColumn("_ir", F.percent_rank().over(w_int))
        .withColumn("_vis_score", (F.col("_pr") + F.col("_ir")) / 2.0)
        .withColumn("_tier", F.ntile(3).over(Window.orderBy("_vis_score")))
        .withColumn(
            "visibility_tier",
            F.element_at(
                F.array(F.lit("low"), F.lit("mid"), F.lit("high")), F.col("_tier")
            ),
        )
        .drop("_pr", "_ir", "_tier")
    )
    unseen = per_cust.filter(
        (F.col("months_in_graph") == 0) | F.col("months_in_graph").isNull()
    ).withColumn("_vis_score", F.lit(None).cast("double")).withColumn(
        "visibility_tier", F.lit("invisible")
    )
    dim = seen.unionByName(unseen).withColumnRenamed("_vis_score", "visibility_score")

    # customer-level attributes from the wide table (account counts kept out of the panel)
    attrs = wide.select(
        F.trim(F.col("cust_pwr_id").cast("string")).alias("cust_pwr_id"),
        F.col("rltn_pwr_id").cast("string").alias("rltn_pwr_id"),
        "latest_month",
        F.col("latest_num_accounts").alias("num_accounts"),
        F.col("latest_num_closed_accounts").alias("num_closed_accounts"),
        "peak_balance",
    ).withColumn(
        "closed_account_ratio",
        F.when(
            F.col("num_accounts") > 0,
            F.col("num_closed_accounts") / F.col("num_accounts"),
        ),
    )
    return dim.join(attrs, "cust_pwr_id", "left")


# ----------------------------------------------------------------------------- stage 6: coverage report

def build_coverage(
    deposit_panel: DataFrame, metrics: DataFrame, joint: DataFrame, cfg: PanelConfig
) -> DataFrame:
    dep = (
        deposit_panel.filter(F.col("balance_status") == "active_span")
        .groupBy("month")
        .agg(F.countDistinct("cust_pwr_id").alias("n_dep_active"))
    )
    gr = metrics.groupBy("month").agg(
        F.countDistinct("cust_pwr_id").alias("n_graph_mapped")
    )
    jt = (
        joint.filter(
            (F.col("balance_status") == "active_span")
            & F.col("pkg_strength").isNotNull()
        )
        .groupBy("month")
        .agg(F.countDistinct("cust_pwr_id").alias("n_joined"))
    )
    return (
        dep.join(gr, "month", "full")
        .join(jt, "month", "full")
        .withColumn("dep_coverage", F.col("n_joined") / F.col("n_dep_active"))
        .withColumn("mapped_coverage", F.col("n_joined") / F.col("n_graph_mapped"))
        .orderBy("month")
    )


# ----------------------------------------------------------------------------- main

def run(cfg: PanelConfig | None = None) -> None:
    cfg = cfg or PanelConfig()
    spark = get_spark()

    wide = spark.table(cfg.deposit_table)
    assert_unique_grain(wide, "cust_pwr_id", "deposit lab table")

    long = melt_deposits(wide)
    long = add_activity_bounds(long)
    zero_null_diagnostics(long)

    deposit_panel = add_deposit_features(long, cfg)
    qa_describe(
        deposit_panel,
        ["balance", "dep_recent_avg", "dep_prior_avg", "dep_pct_change",
         "dep_decline_flag", "dep_dlog_1m", "dep_drawdown"],
        "deposit features",
    )

    metrics = load_node_metrics(spark, cfg)
    join_guard(deposit_panel, metrics, cfg)

    joint = deposit_panel.join(
        metrics, ["cust_pwr_id", "month"], "left"
    ).withColumn("in_graph_month", F.col("pkg_strength").isNotNull().cast("int"))

    customer_dim = build_customer_dim(deposit_panel, metrics, wide, cfg)
    qa_describe(
        customer_dim,
        ["graph_presence_rate", "median_c2c_intensity", "visibility_score",
         "closed_account_ratio"],
        "visibility index",
    )
    coverage = build_coverage(deposit_panel, metrics, joint, cfg)
    coverage.show(40, truncate=False)

    t = lambda name: f"{cfg.out_db}.{cfg.out_prefix}_{name}"
    for df, name in [
        (deposit_panel, "deposit_panel"),
        (joint, "joint_panel"),
        (customer_dim, "customer_dim"),
        (coverage, "coverage"),
    ]:
        df.write.mode(cfg.write_mode).format("parquet").saveAsTable(t(name))
        print(f"[write] {t(name)}")

    print("\n[done] Tier 0 panel build complete.")


if __name__ == "__main__":
    run()
