"""
subro_driver.py — notebook cells for insurance subrogation extraction.

Paste block by block into Jupyter (each `# %%` is a cell), or run as a script.
Assumes pkn_subro.py sits next to the notebook and snapshots live in ../data/.
"""

# %% ------------------------------------------------------------------ setup
import warnings

import numpy as np
import pandas as pd

import pkn_subro as ps

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "../data"
MONTHS = ps.month_range("2025-01", "2025-11")
print(len(MONTHS), "snapshots:", MONTHS[0], "->", MONTHS[-1])


# %% -------------------------------------------- 1. aggregate the 11 months
# Folds one month at a time; peak memory ~ |aggregate| + |one month|.
# Expect a few minutes for ~5M rows/month.
agg = ps.build_aggregate(MONTHS, data_dir=DATA_DIR, verbose=True)

edges = agg.edges          # s, d, amount, volume, n_rels, amt_cv, avg_ticket, ...
nodes = agg.nodes          # node, node_id, name, naics_code/desc, naics2..6
N = len(nodes)

print(f"\n{len(nodes):,} nodes / {len(edges):,} aggregated edges")
print(edges[["amount", "volume", "n_rels", "avg_ticket", "amt_cv"]].describe())


# %% ------------------------------------------------- 2. sanity-check NAICS
print("NAICS coverage:", nodes["naics_code"].notna().mean().round(3))
print(nodes["naics_code"].str.len().value_counts(dropna=False))
print(nodes["naics_sector"].value_counts().head(10))

# The insurance corner of the customer base
ins = nodes[nodes["naics_code"].fillna("").str.startswith(("5241", "5242"))]
print("\ninsurance-coded nodes:", len(ins))
print(ins.groupby("naics_code")["node"].size().sort_values(ascending=False).head(15))


# %% ---------------------------------------------------- 3. select the seeds
# NAICS union name-regex. Keep seed_reason — 'name'-only hits are exactly the
# population where the NAICS field is missing or wrong.
seeds_df = ps.select_seeds(
    nodes,
    naics_prefixes=("5241", "5242"),
    name_regex=ps.INSURANCE_NAME_STRONG,
    exclude_regex=ps.INSURANCE_NAME_EXCLUDE,
)
seeds = seeds_df["node"].to_numpy()
carriers = seeds_df.loc[seeds_df["is_carrier"], "node"].to_numpy()

print(seeds_df["seed_reason"].value_counts())
print("seeds:", len(seeds), "| carriers (5241* or name-confirmed):", len(carriers))

# ALWAYS eyeball the name-only hits before trusting them
print(seeds_df.query("seed_reason == 'name'")[["node_id", "name", "naics_desc"]].head(40))

# Optional: widen with the weak pattern and review manually
# wide = ps.select_seeds(nodes, name_regex=ps.INSURANCE_NAME_STRONG + "|" + ps.INSURANCE_NAME_WEAK)


# %% ------------------------------------------ 4. first-order ego subgraph
ego, keep_nodes = ps.ego_subgraph(edges, seeds, k=1, n_nodes=N)
ps.describe_ego(ego, keep_nodes, seeds, agg)

ego_nodes = nodes[nodes["node"].isin(keep_nodes)].copy()
stats = ps.node_stats(ego, n_nodes=N)
ego_nodes = ego_nodes.merge(stats, on="node", how="left")


# %% ------------------------------- 5. carrier-to-carrier core + scoring
core = ps.carrier_core(ego, carriers, N)         # carriers only, not 5242*
pairs = ps.bilateral_pairs(core)
scored = ps.score_subrogation(pairs, n_months=agg.n_months)
scored = ps.flag_affiliates(scored, nodes)

cols = ["name_a", "name_b", "gross", "net", "reciprocity", "rels_max",
        "vol_total", "ticket", "cv", "subro_score", "likely_affiliate"]
subro = scored.query("~likely_affiliate")
print(subro[cols].head(30).to_string(index=False))

# What the score is pushing DOWN — check these are reinsurance / commissions
print("\nlow scorers:")
print(scored[cols].tail(15).to_string(index=False))


# %% ------------------------- 6. intermediaries: carrier -> X -> carrier
# Recovery vendors, arbitration forums, netting agents. Balanced throughput
# (in ~= out) with many carriers on both sides is the netting fingerprint.
inter = ps.pass_through_intermediaries(ego, carriers, nodes, N,
                                       min_carriers_each_side=2, top=50)
print(inter[["name", "naics_code", "naics_desc", "in_carriers", "out_carriers",
             "in_amt", "out_amt", "balance", "throughput"]].to_string(index=False))


# %% ------------------------------------ 7. payer / payee side of the ego
print("PAYEES (claim disbursement footprint)")
print(ps.counterparty_naics_profile(ego, carriers, nodes, N,
                                    side="out", level="naics4", top=25)
        .to_string(index=False))

print("\nPAYERS (premium + recovery inflow)")
print(ps.counterparty_naics_profile(ego, carriers, nodes, N,
                                    side="in", level="naics4", top=25)
        .to_string(index=False))


# %% -------------------------- 8. shared vendors = subrogation anchors
# A shop / clinic / firm paid by many carriers is where two carriers touched the
# same loss. Tight clusters here are also the staged-accident-ring signal.
shared = ps.shared_counterparties(ego, carriers, nodes, N,
                                  side="out", min_seeds=3, top=200)
print(shared[["name", "naics_desc", "n_seeds", "amount", "volume",
              "avg_ticket"]].head(30).to_string(index=False))

# Concentrate on the ones that are small businesses with many carriers —
# a two-bay body shop paid by nine carriers is worth a look.
print(shared.query("n_seeds >= 5 and volume < 500")
            .sort_values("n_seeds", ascending=False)
            .head(20)[["name", "naics_desc", "n_seeds", "amount", "volume"]]
            .to_string(index=False))


# %% ------------------------------------ 9. monthly panel (pass 2) + lags
panel = ps.monthly_panel(MONTHS, keep_nodes, agg.index, data_dir=DATA_DIR)
name = nodes.set_index("node")["name"]

# monthly series for the top candidate pair
top = subro.iloc[0]
a, b = int(top["a"]), int(top["b"])
series = (panel[((panel.s == a) & (panel.d == b)) | ((panel.s == b) & (panel.d == a))]
          .pivot_table(index="mi", columns="s", values="amount", aggfunc="sum")
          .reindex(range(len(MONTHS))).fillna(0))
series.columns = [name.get(c, c) for c in series.columns]
print(series)

# recovery-lag motif: A pays a shared vendor, B reimburses A k months later
vendors = shared["cp"].to_numpy()[:50]
x = (panel[(panel.s == a) & (panel.d.isin(vendors))]
     .groupby("mi")["amount"].sum().reindex(range(len(MONTHS)), fill_value=0))
y = (panel[(panel.s == b) & (panel.d == a)]
     .groupby("mi")["amount"].sum().reindex(range(len(MONTHS)), fill_value=0))
print(ps.lagged_xcorr(x, y, max_lag=6))


# %% ---------------------------------------------- 10. networkx for viz/QA
G = ps.to_networkx(ego, nodes)
print(G)

import networkx as nx
core_ids = set(carriers.tolist())
H = G.subgraph([n for n in G.nodes if n in core_ids]).copy()
print("carrier core:", H, "| reciprocity:", round(nx.reciprocity(H) or 0, 3))
print("density:", round(nx.density(H), 4))

# export for Neo4j / PyViz / the Streamlit explorer
out = ego.merge(nodes[["node", "node_id", "name", "naics_code", "naics_desc"]]
                .add_prefix("src_"), left_on="s", right_on="src_node", how="left")
out = out.merge(nodes[["node", "node_id", "name", "naics_code", "naics_desc"]]
                .add_prefix("dst_"), left_on="d", right_on="dst_node", how="left")
out.drop(columns=["src_node", "dst_node"]).to_parquet("subro_ego_edges.parquet")
subro.to_parquet("subro_pair_candidates.parquet")
inter.to_parquet("subro_intermediaries.parquet")
shared.to_parquet("subro_shared_vendors.parquet")
print("written.")
