# 09 · Engineering notes and traps

*Every entry below cost a run or a debugging session. They are listed so the next person does not
pay for them again.*

---

## 1. Environment

| | |
|---|---|
| Spark | 3.3.2 (Cloudera parcel) |
| Python | 3.9 |
| numpy | 2.2.6 |
| Cluster | PySpark on YARN, JupyterHub at `paejup.pncint.net` |

```python
spark = (SparkSession.builder.appName("pkg_attrition_eda_vN")
         .config("spark.sql.shuffle.partitions", "800")   # 400 spills on the pair panel
         .config("spark.sql.execution.arrow.pyspark.enabled", "false")
         .enableHiveSupport().getOrCreate())
```

### Arrow must stay OFF
PySpark 3.3.2's arrow conversion path references `np.object0` and `np.bool8`, **both removed in
numpy 2.0**. This is not only the `decimal(15,0)` case — the whole arrow path is unsafe on this
pairing. Everything collects through the row path with decimals pre-cast to double.

### numpy 2 removals this code routes around
| Removed | Use instead |
|---|---|
| `np.trapz` | Rank-based Mann-Whitney AUC (exact, no trapezoid) |
| `np.float_`, `np.NaN`, `np.Inf`, `np.object0`, `np.bool8` | Never referenced |
| `np.in1d` | `np.isin` |
| `np.array(x, copy=False)` now **raises** | `np.asarray` |

---

## 2. Traps that cost a run

| Trap | Symptom | Fix |
|---|---|---|
| **`m_idx` is ABSOLUTE, not 0-based** | `range(18, M_MAX-H+1)` built **24,296 empty folds**; the collect loop ran until interrupted | Every month constant is an OFFSET from `M_MIN`. Assert `1 <= len(ORIGINS) <= MAX_ORIGINS` |
| **`MIN_REF > 1.0` on a bounded ratio** | Five feature families silently null; blocks reported `n_feat` identical to the baseline and their 0.0000 deltas were read as evidence | Additive dd for ratios (`04` §1). **Watch `n_feat` in every ablation** |
| **Pivot naming depends on agg count** | `{pivotValue}_{aggAlias}` only with **2+** aggregations. With one agg the column is the bare pivot value | Assert the expected columns exist right after the pivot |
| **Duplicate `ym`: partition path AND data column** | Read-time conflict on partition discovery | The partition path carries it. Do not also `withColumn("ym", ...)` |
| **Ambiguous column after a self-derived join** | `Reference 'n_payers' is ambiguous` — `GEN` was a filter of `payers` and already had the column | Aggregate the derived frame directly; do not join the parent back |
| **`F.sum()` of a boolean** | `sum requires numeric, not boolean` | `F.sum(F.when(cond, 1).otherwise(0))` |
| **Unbroadcast range join** | A 31-row month list against the pair panel plans as sort-merge and does not return | `F.broadcast(months)` |
| **`jaro_winkler_similarity` is not a Spark builtin** | `Undefined function` — it is Impala/Snowflake | Exact-after-normalise or shared 8-char prefix. Deliberately conservative |
| **HDFS URI in `pathlib.Path`** | `Permission denied … inode="/"` — `Path` collapses `hdfs://host/p` → `hdfs:/host/p` | Local `Path` and HDFS **string** are separate config vars, never mixed |
| **Arrow + `decimal(15,0)`** | `module 'numpy' has no attribute 'object0'` | Arrow off; cast decimals to double before collecting |
| **Fanned-out account→customer map** | 241,499 rows for 229,363 accounts; ~5% of customers' payment volume silently inflated | One row per account, latest link wins, **assert 1:1** |
| **Single unfiltered payments job** | SparkContext killed | One Spark job **per month**, partitioned, skip-if-exists. ~100–115M sides/month, 60–130s each |
| **Duplicate keys in a `kv()` dict** | Silently prints only the last value; a table showed `3` where `12` and `7` belonged | `kv()` takes ordered **pairs** and **raises** on a repeated label |
| **`rowsBetween` on months** | Windows shift where a month is missing | `rangeBetween` on an integer `m_idx` |
| **`countDistinct` ignores NULL** | A column null on one row and populated on another reads as identical | `coalesce(col, "<NULL>")` first |
| **int64/string id mismatch** | 2026-07 incident: typed **every** counterparty `unknown` and produced entirely plausible output | Every id is a string end to end. Match-rate assert before anything else |
| **`element_at` with a Column index** | Version-fragile | `F.expr("element_at(arr, cast(size(arr)/2 as int)+1)")` |
| **`np.concatenate(...) or default`** | `truth value of an array is ambiguous` | Filter empty arrays explicitly |
| **Spark pivot naming** | `Column 'fin_out_new_out' does not exist` | Output is `{pivotValue}_{aggAlias}` → `fin_new_out` |
| **`F.avg(boolean)`** | `avg requires numeric` | `.cast("double")` |
| **Column names that shadow DataFrame methods** | `df.cov` returns the *method*, not the column | Rename the column (`cov` → `coverage`) rather than switching to bracket access |
| **Referencing a column a prior notebook never persisted** | `AnalysisException: m_C_p30` | v3 dropped `C_p30`; v5 rebuilds it |

---

## 3. Code conventions

```python
def disp(obj, title=None, n=60, save=None):
    """Never .show(). Collect to pandas, optionally write a CSV, render with a title."""

def kv(pairs, title=None, save=None):
    """ORDERED PAIRS, not a dict literal. Raises on a repeated label."""

def note(qid, question, answer, detail=""):
    """Append to FINDINGS_v*.csv — the machine-readable findings ledger."""

def collect_pd(sdf, label, max_rows=3_000_000):
    """Print the row count BEFORE the wait. Hard ceiling. Arrow is off, so budget
       ~1 min per 200–300k rows at ~50 numeric columns."""
```

- **Resumable everything.** Per-month loops check `exists(path)` and skip.
- **Stage flags** at the top of each notebook (`RUN_AUDIT`, `RUN_BUILD`, `RUN_ABLATION`) so the
  cheap audit runs before the hours-long build.
- **Fail-fast guards.** The typing join samples 100k edge nodes and raises if it misses >50%.
- **Nothing globally suppressed.** A blanket `filterwarnings("ignore")` would have hidden a real
  divide bug in the graph work.

---

## 4. Performance rules

- Per-group `nlargest` is banned — global sort + `groupby.head`/`cumsum`.
- No `groupby.quantile` for weighted percentiles — sort by value, take the weighted-cumulative
  crossing.
- No `groupby.apply` for entropy — two groupbys.
- Cap the counterparty tail: `PAIR_TOP_N = 200` per client-month by amount. Past that it is one-off
  payments and it is the join cost, not the signal.
- One Spark job per month for anything touching the payment table.

---

## 5. Runtime budgets (measured)

| Step | Wall |
|---|---|
| v7 block 1 — panels, dd, labels | 144 s |
| v7 block 3 — risk set + collects | 78 s |
| v8 §2 inventory | 99 s |
| v8 §5 key QA | 274 s |
| v8 per-month enriched build | ~5 s/month × 31 |
| v8 §7 feature assembly | 5.5 min |
| v8b §1 ratio repair | 189 s |
| Ablation (19 specs × 7 folds) | ~2 min |

Collects: TRAIN 312,881 × 275 in 29 s; each TEST month ~75–80k × 275 in 7 s.

*Internal — PNC Treasury Management, Data Science*
