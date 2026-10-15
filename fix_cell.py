# =====================================================================
# §8 · DIAGNOSTICS — paste as a new cell at the end of pkg_fraud_signals_v1.ipynb and run it (≈ 1–3 min).
# Reads the tables §7 just wrote (no recompute). Answers: what drives the volume, why almost everything is High,
# and how many flags each candidate tightening would remove. Nothing is written.
# =====================================================================
t0 = time.time(); B = Block("§8 · Diagnostics: what drives the volume")
SIG = spark.table(f"{TGT_DB}.{OUT_PREFIX}signals")
HIS = spark.table(f"{TGT_DB}.{OUT_PREFIX}history")
LAST = SIG.agg(F.max("month")).first()[0]
S = topd(SIG)
L = S[(S.month == LAST) & (S.score > 0)].copy()
TYPS = ["passthrough", "funnel", "split", "spike", "roundtrip"]
for k in TYPS: L[f"is_{k}"] = L[f"score_{k}"].fillna(0) > 0

# ── 1 · volume and score saturation per typology ──
scr = pd.read_csv(OUT / "f1_5a_counts.csv") if (OUT / "f1_5a_counts.csv").exists() else None
n_scr = int(scr.loc[scr.month == LAST, "screened"].iloc[0]) if scr is not None and (scr.month == LAST).any() else np.nan
rows = []
for k in TYPS:
    s = L.loc[L[f"is_{k}"], f"score_{k}"]
    rows.append(dict(typology=k, flagged=len(s), pct_of_screened=len(s) / n_scr if n_scr else np.nan,
                     p10=s.quantile(.1), p50=s.median(), p90=s.quantile(.9), share_ge_70=(s >= 70).mean(),
                     usd_involved=L.loc[L[f"is_{k}"], "usd_involved"].sum()))
B.table(pd.DataFrame(rows), f"8a · {LAST}: flags and score spread per typology (screened = {n_scr:,})",
        pct=("pct_of_screened", "share_ge_70"), money=("usd_involved",))

# ── 2 · who is flagged: party type, size band, top industries ──
B.table(L.groupby("size_band").agg(entities=("mdm_id", "size"), usd=("usd_involved", "sum")).reset_index(),
        "8b · Flagged by size band (1 = smallest fifth of screened flow)", money=("usd",))
top_ind = (L.assign(ind=L.naics_desc.fillna("unknown")).groupby(["primary", "ind"]).size().rename("entities").reset_index()
            .sort_values("entities", ascending=False).groupby("primary").head(4))
B.table(top_ind, "8c · Top industries per primary typology")
if "party_type" in L: B.table(L.groupby(["party_type", "primary"]).size().rename("entities").reset_index(), "8d · Party type × primary typology")

# ── 3 · pass-through: zero-balance / sweep accounts and money that stays inside the corporate family ──
pt = L[L.is_passthrough].copy()
pt["days_bucket"] = pd.cut(pt.days_held, [-0.01, 0.1, 0.5, 1, 2, 5], labels=["<0.1 d", "0.1–0.5 d", "0.5–1 d", "1–2 d", "2–5 d"])
pt["bal_bucket"] = pd.cut(pt.bal_3m, [-1, 1_000, 25_000, 250_000, np.inf], labels=["< $1k", "$1k–25k", "$25k–250k", "> $250k"])
B.table(pd.crosstab(pt.bal_bucket, pt.days_bucket).reset_index(), "8e · Pass-through: average balance × days of inflow held")
fam = None
if PAIR_OK and len(pt):
    CDm = spark.table(CUST_DIM).select("mdm_id", F.col("master_pwr_id").cast("string").alias("own_master"))
    ids = spark.createDataFrame(pt[["mdm_id"]].drop_duplicates())
    win = M_HIST[-3:] if LAST == END else [ym_add(LAST, -2), ym_add(LAST, -1), LAST]
    P = (spark.table(PAIR_TABLE).filter(F.col("month").isin(win) & (F.col("dir") == "out"))
              .join(ids.withColumnRenamed("mdm_id", "cust"), "cust")
              .join(CDm.withColumnRenamed("mdm_id", "cust"), "cust", "left")
              .join(CDm.withColumnRenamed("mdm_id", "other").withColumnRenamed("own_master", "o_master"), "other", "left"))
    fam = topd(P.groupBy("cust").agg(
        F.sum("amount").alias("out_all"),
        F.sum(F.when((F.col("scope") == "int") & (F.col("o_master") == F.col("own_master")), F.col("amount")).otherwise(0)).alias("out_family"),
        F.sum(F.when(F.col("scope") == "int", F.col("amount")).otherwise(0)).alias("out_pnc"),
        F.sum(F.when(F.col("scope") == "self", F.col("amount")).otherwise(0)).alias("out_self"))
        .withColumnRenamed("cust", "mdm_id"))
    pt = pt.merge(fam, on="mdm_id", how="left")
    pt["family_share"] = pt.out_family / pt.out_all.where(pt.out_all > 0)
    pt["pnc_share"] = pt.out_pnc / pt.out_all.where(pt.out_all > 0)
    pt["fam_bucket"] = pd.cut(pt.family_share.fillna(0), [-0.01, 0.1, 0.5, 0.9, 1.0], labels=["< 10%", "10–50%", "50–90%", "≥ 90%"])
    B.table(pt.groupby("fam_bucket", observed=False).agg(entities=("mdm_id", "size"), usd=("usd_involved", "sum"),
                                                          pnc_share=("pnc_share", "median")).reset_index(),
            "8f · Pass-through: share of money out that goes to its own corporate family at PNC (same master power id)",
            pct=("pnc_share",), money=("usd",))

# ── 4 · spike: how big was "normal", how long the entity has been active, which month ──
sk = S[(S.score_spike.fillna(0) > 0)].copy()
B.table(sk.groupby("month").agg(entities=("mdm_id", "nunique")).reset_index(), "8g · Spike flags per scoring month (a jump in one month only hints at a data change)")
skl = sk[sk.month == LAST].copy()
if len(skl):
    H = topd(HIS.filter(F.col("mdm_id").isin(list(skl.mdm_id.unique()))).select("mdm_id", "month", "flow_m"))
    prior = H[(H.month < LAST) & (H.month >= ym_add(LAST, -6))]
    act = prior.assign(on=prior.flow_m > 0).groupby("mdm_id").on.sum().rename("active_prior6")
    skl = skl.merge(act, on="mdm_id", how="left").fillna({"active_prior6": 0})
    skl["base_bucket"] = pd.cut(skl.base.fillna(0), [-1, 10_000, 50_000, 250_000, np.inf], labels=["< $10k", "$10k–50k", "$50k–250k", "> $250k"])
    B.table(pd.crosstab(skl.base_bucket, skl.active_prior6.astype(int)).reset_index(),
            f"8h · Spikes in {LAST}: normal monthly flow × months active in the prior 6")
    B.table(skl.spike_x.describe(percentiles=[.25, .5, .75, .9]).rename("spike_x").reset_index(), "8i · Spike multiple")

# ── 5 · largest dollar cases (sanity check on usd_involved) ──
B.table(L.sort_values("usd_involved", ascending=False).head(15)[["customer_name", "naics_desc", "primary", "typologies", "tier", "score",
                                                                 "flow_3m", "usd_involved", "days_held", "payees_avg"]],
        "8j · Top 15 by $ involved", money=("flow_3m", "usd_involved"))

# ── 6 · what candidate tightenings would leave (cumulative, latest month) ──
keep = pd.Series(True, index=L.index)
def step(name, drop):
    global keep
    keep = keep & ~drop.reindex(L.index).fillna(False).astype(bool)
    return dict(step=name, entities_left=int(keep.sum()),
                **{k: int((keep & L[f"is_{k}"]).sum()) for k in TYPS})
fs = L.mdm_id.map(pt.set_index("mdm_id").family_share) if fam is not None else pd.Series(np.nan, index=L.index)
only_pt = L.is_passthrough & (L.n_typ == 1)
sk_act = L.mdm_id.map(skl.set_index("mdm_id").active_prior6) if len(skl) else pd.Series(np.nan, index=L.index)
W = [dict(step="as is", entities_left=len(L), **{k: int(L[f"is_{k}"].sum()) for k in TYPS}),
     step("pass-through: drop ≥ 50% of money out to own corporate family", only_pt & (fs >= .5)),
     step("pass-through: drop balance < $1k (zero-balance / swept accounts)", only_pt & (L.bal_3m < 1_000)),
     step("spike: drop normal month < $25k", L.is_spike & (L.n_typ == 1) & (L.base.fillna(0) < 25_000)),
     step("spike: drop < 4 of prior 6 months active (new or dormant)", L.is_spike & (L.n_typ == 1) & (sk_act < 4)),
     step("spike: drop spike multiple < 20×", L.is_spike & (L.n_typ == 1) & (L.spike_x < 20))]
B.table(pd.DataFrame(W), "8k · Candidate tightenings, applied cumulatively (only entities whose only pattern is the one tightened)")
B.note("Read 8k with 8e–8i: a step is worth keeping when it removes a large block that has an ordinary explanation "
       "(sweeps, intra-family cash concentration, new relationships) rather than just shaving the tail.")
B.show(t0)
