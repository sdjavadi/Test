# %% [markdown]
# # PKG Notebook — Aggregated Network (Jan–Nov 2025) & Subrogation Candidate Extraction
#
# Loads monthly snapshot CSVs, builds an aggregated payment graph over the
# chosen window, parses NAICS code/description + multi-digit rollups, and
# sets up tooling to identify and inspect candidate insurance-subrogation
# sub-networks (NAICS + name filtering, first-order ego graph, payer/payee
# and reciprocal-flow analysis).

# %%
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# %% [markdown]
# ## 0. Config

# %%
DATA_DIR = Path("../data")
FILE_PREFIX = "cust_"
START_YM = "2025-01"
END_YM = "2025-11"

RAW_DTYPES = {
    "source": "string",
    "source_name": "string",
    "source_naics": "string",
    "amount": "float64",
    "volume": "float64",   # transaction count per edge per month; float tolerates NaN
    "dest": "string",
    "dest_name": "string",
    "dest_naics": "string",
}

def month_range(start_ym: str, end_ym: str) -> List[str]:
    start = pd.Timestamp(start_ym + "-01")
    end = pd.Timestamp(end_ym + "-01")
    return pd.date_range(start, end, freq="MS").strftime("%Y-%m").tolist()

MONTHS = month_range(START_YM, END_YM)
print(f"Months to load ({len(MONTHS)}): {MONTHS[0]} .. {MONTHS[-1]}")

# %% [markdown]
# ## 1. Locate files & load, tagging each row with its source month

# %%
def find_missing_files(months, data_dir=DATA_DIR, prefix=FILE_PREFIX):
    missing = []
    for ym in months:
        p = data_dir / f"{prefix}{ym}.csv"
        if not p.exists():
            missing.append(str(p))
    return missing

missing = find_missing_files(MONTHS)
if missing:
    print(f"WARNING: {len(missing)} file(s) not found, will be skipped:")
    for m in missing:
        print(f"  - {m}")

# %%
def load_month(ym: str, data_dir=DATA_DIR, prefix=FILE_PREFIX) -> pd.DataFrame:
    path = data_dir / f"{prefix}{ym}.csv"
    df = pd.read_csv(path, dtype=RAW_DTYPES)
    df["year_month"] = ym
    return df

frames = []
t0 = time.time()
for ym in MONTHS:
    path = DATA_DIR / f"{FILE_PREFIX}{ym}.csv"
    if not path.exists():
        continue
    t1 = time.time()
    df = load_month(ym)
    frames.append(df)
    print(f"  {ym}: {len(df):>10,} rows  ({time.time() - t1:5.1f}s)")

if not frames:
    raise FileNotFoundError(
        f"No snapshot files found in {DATA_DIR.resolve()} for {MONTHS[0]}..{MONTHS[-1]}. "
        f"Check DATA_DIR and FILE_PREFIX."
    )

raw = pd.concat(frames, ignore_index=True)
del frames
print(f"\nLoaded {len(raw):,} total rows from {len(MONTHS)} months in {time.time() - t0:.1f}s")
print(f"Memory footprint: {raw.memory_usage(deep=True).sum() / 1e9:.3f} GB")

# %% [markdown]
# ## 2. NAICS parsing — split "CODE|DESCRIPTION" and derive 2-6 digit rollups
#
# Known dirty values from the source system (missing, '-1', 'UNKNOWN',
# '******') are treated as invalid at every digit level rather than
# silently truncated into a fake code.

# %%
NAICS_SENTINELS = {"-1", "", "UNKNOWN", "******"}

def split_naics(raw_series: pd.Series) -> pd.DataFrame:
    s = raw_series.fillna("").astype(str)
    split = s.str.split("|", n=1, expand=True)
    code = split[0].str.strip()
    if split.shape[1] > 1:
        desc = split[1].str.strip()
    else:
        desc = pd.Series("", index=s.index)

    is_valid = code.str.fullmatch(r"\d{2,6}").fillna(False) & ~code.isin(NAICS_SENTINELS)

    out = pd.DataFrame({
        "naics_code": code.where(is_valid),
        "naics_desc": desc.where(is_valid),
        "naics_valid": is_valid,
    })
    for n in (2, 3, 4, 5, 6):
        long_enough = out["naics_code"].str.len() >= n
        out[f"naics{n}"] = out["naics_code"].str.slice(0, n).where(long_enough.fillna(False))
    return out

# %%
src_naics = split_naics(raw["source_naics"]).add_prefix("source_")
dst_naics = split_naics(raw["dest_naics"]).add_prefix("dest_")

raw = pd.concat([raw.drop(columns=["source_naics", "dest_naics"]), src_naics, dst_naics], axis=1)

print("Source NAICS validity: {:.1%}".format(raw["source_naics_valid"].mean()))
print("Dest   NAICS validity: {:.1%}".format(raw["dest_naics_valid"].mean()))

# %% [markdown]
# ## 3. Node attribute table (name + NAICS)
#
# A node can appear as `source` in some rows and `dest` in others, and its
# name/NAICS can drift slightly month to month (reclassification, minor
# name changes). We union both sides and, per node, take the most recent
# *non-null* value for each attribute (pandas groupby `.last()` skips NaNs
# by default, so an early good value isn't lost to a later blank).

# %%
def build_node_table(raw: pd.DataFrame) -> pd.DataFrame:
    # year_month is a row-level column (which month this transaction came
    # from); the naics/name columns are the ones split by source_/dest_.
    naics_cols = ["naics_code", "naics_desc", "naics_valid",
                  "naics2", "naics3", "naics4", "naics5", "naics6"]

    src = raw[["source", "source_name", "year_month"] + [f"source_{c}" for c in naics_cols]].copy()
    src.columns = ["node_id", "name", "year_month"] + naics_cols

    dst = raw[["dest", "dest_name", "year_month"] + [f"dest_{c}" for c in naics_cols]].copy()
    dst.columns = ["node_id", "name", "year_month"] + naics_cols

    both = pd.concat([src, dst], ignore_index=True).sort_values("year_month")
    nodes = both.groupby("node_id", sort=False).last().drop(columns="year_month").reset_index()
    return nodes

nodes = build_node_table(raw)
print(f"Unique nodes: {len(nodes):,}")

# %% [markdown]
# ## 4. Aggregated edge table over the window
#
# - amount_total : sum of monthly `amount` across the window
# - volume_total : sum of monthly `volume` (txn count) across the window
# - n_rels       : number of distinct months this (source, dest) pair
#                  appeared as an edge at all (ranges 1..len(MONTHS)) --
#                  a persistence/recurrence signal, distinct from volume
#
# If this OOMs on the full history the way the monthly pipeline's
# aggregate graph did, aggregate incrementally per month (running-sum a
# dict keyed by (source,dest), discard each month's raw rows) instead of
# concatenating everything first -- same principle as the streaming
# design already used in the production pipeline.

# %%
edges = (
    raw.groupby(["source", "dest"], sort=False)
       .agg(amount_total=("amount", "sum"),
            volume_total=("volume", "sum"),
            n_rels=("amount", "size"))
       .reset_index()
)
edges["avg_amount_per_txn"] = edges["amount_total"] / edges["volume_total"].replace(0, np.nan)

n_touched = pd.unique(pd.concat([edges["source"], edges["dest"]], ignore_index=True))
print(f"Unique directed edges (source, dest pairs): {len(edges):,}")
print(f"Unique nodes touched by an edge:            {len(n_touched):,}")

# %% [markdown]
# ## 5. Pick a graph backend based on actual scale
#
# NetworkX is pure Python: great API, but construction and anything beyond
# basic degree/neighbor queries gets uncomfortable somewhere in the low
# hundreds of thousands of edges and painful well before a few million.
# NetworKit is a C++/OpenMP backend built for exactly this scale, at the
# cost of a clunkier API (integer node ids, fewer convenience methods).
# Single-month snapshots already run 3-5M edges, so don't assume -- check
# the actual aggregated count and branch on it.

# %%
N_EDGE_THRESHOLD = 750_000
N_NODE_THRESHOLD = 750_000

n_edges = len(edges)
n_nodes = len(nodes)
use_networkit = (n_edges > N_EDGE_THRESHOLD) or (n_nodes > N_NODE_THRESHOLD)

print(f"nodes={n_nodes:,}  edges={n_edges:,}  -> backend = "
      f"{'networkit' if use_networkit else 'networkx'}")

# %%
idx_of: Optional[Dict[str, int]] = None
id_of: Optional[Dict[int, str]] = None

if use_networkit:
    try:
        import networkit as nk
    except ImportError:
        raise ImportError(
            "This graph needs networkit at this scale. Install with: "
            "pip install networkit --break-system-packages"
        )

    node_ids = pd.unique(pd.concat([edges["source"], edges["dest"]], ignore_index=True))
    idx_of = {nid: i for i, nid in enumerate(node_ids)}
    id_of = {i: nid for nid, i in idx_of.items()}

    G = nk.Graph(n=len(node_ids), weighted=True, directed=True)
    for row in edges.itertuples(index=False):
        G.addEdge(idx_of[row.source], idx_of[row.dest], row.amount_total)

    edge_attr = edges.set_index(["source", "dest"])

else:
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes["node_id"])
    G.add_edges_from(
        (r.source, r.dest, {"amount": r.amount_total,
                             "volume": r.volume_total,
                             "n_rels": r.n_rels,
                             "avg_amount_per_txn": r.avg_amount_per_txn})
        for r in edges.itertuples(index=False)
    )
    node_attr_cols = ["name", "naics_code", "naics_desc", "naics2", "naics3", "naics4", "naics5", "naics6"]
    nx.set_node_attributes(G, nodes.set_index("node_id")[node_attr_cols].to_dict("index"))

n_g_nodes = G.numberOfNodes() if use_networkit else G.number_of_nodes()
n_g_edges = G.numberOfEdges() if use_networkit else G.number_of_edges()
print(f"Graph built: {n_g_nodes:,} nodes, {n_g_edges:,} edges")

# %% [markdown]
# ## 6. EDA -- surface candidate NAICS codes for "insurance / subrogation"
#
# Rather than hardcoding NAICS codes from memory, pull the actual distinct
# (code, description) pairs in *your* data whose description mentions
# insurance/claims/subrogation-adjacent terms, so codes are picked off
# what's really present.

# %%
KEYWORD_PATTERN = r"insur|casualt|reinsur|subrogat|claim|underwrit|assurance"

candidate_naics = (
    nodes.loc[nodes["naics_desc"].str.contains(KEYWORD_PATTERN, case=False, na=False),
              ["naics_code", "naics_desc"]]
    .drop_duplicates()
    .sort_values("naics_code")
)
print(candidate_naics.to_string(index=False))

# %% [markdown]
# ## 7. Node selection: NAICS prefix list + optional name regex

# %%
def select_nodes(nodes: pd.DataFrame,
                  naics_prefixes: Optional[List[str]] = None,
                  naics_level: int = 3,
                  name_regex: Optional[str] = None) -> pd.DataFrame:
    """
    naics_prefixes : e.g. ['524'] to match the naics3 column, or specific
                      6-digit codes with naics_level=6.
    name_regex     : optional, applied to `name` (case-insensitive), OR'd
                      with the NAICS match if both are given.
    """
    mask = pd.Series(False, index=nodes.index)
    if naics_prefixes:
        col = f"naics{naics_level}"
        mask = mask | nodes[col].isin(naics_prefixes)
    if name_regex:
        mask = mask | nodes["name"].str.contains(name_regex, case=False, na=False, regex=True)
    return nodes.loc[mask].copy()

# Adjust this once you've looked at `candidate_naics` above.
# 524 = Insurance Carriers & Related Activities (carriers, TPAs, claims adjusters, brokers)
seed_nodes = select_nodes(
    nodes,
    naics_prefixes=["524"],
    naics_level=3,
    name_regex=r"subrogat|arbitration forum",
)
print(f"\nSeed candidates: {len(seed_nodes):,}")
print(seed_nodes.head(20).to_string(index=False))

# %% [markdown]
# ## 8. First-order ego graph around the seed set (both payers and payees)
#
# This is the union of in- and out-neighbors across the whole seed set,
# then the induced subgraph on {seeds} u neighbors -- not a single-center
# nx.ego_graph, since we want edges between the seeds' counterparties too
# (e.g. two suspected subrogation partners that also share a counterparty).

# %%
def ego_subgraph(G, seed_ids, use_networkit=False, idx_of=None, id_of=None):
    if use_networkit:
        seed_idx = [idx_of[s] for s in seed_ids if s in idx_of]
        keep = set(seed_idx)
        for s in seed_idx:
            keep.update(G.iterNeighbors(s))
            keep.update(G.iterInNeighbors(s))
        sub = nk.graphtools.subgraphFromNodes(G, keep)
        kept_ids = [id_of[i] for i in keep]
        return sub, kept_ids
    else:
        keep = set(seed_ids)
        for s in seed_ids:
            if G.has_node(s):
                keep.update(G.successors(s))
                keep.update(G.predecessors(s))
        sub = G.subgraph(keep).copy()
        return sub, list(keep)

seed_ids = seed_nodes["node_id"].tolist()
if not seed_ids:
    raise ValueError(
        "No seed nodes matched -- widen naics_prefixes/name_regex in select_nodes(), "
        "or check candidate_naics above for the right codes."
    )

if use_networkit:
    sub, kept_ids = ego_subgraph(G, seed_ids, use_networkit=True, idx_of=idx_of, id_of=id_of)
    print(f"Ego subgraph: {sub.numberOfNodes():,} nodes, {sub.numberOfEdges():,} edges")
else:
    sub, kept_ids = ego_subgraph(G, seed_ids, use_networkit=False)
    print(f"Ego subgraph: {sub.number_of_nodes():,} nodes, {sub.number_of_edges():,} edges")

# %% [markdown]
# ## 9. Payer / payee profile + reciprocal-flow check
#
# The subrogation "signature" in an aggregated window is a pair of nodes
# with edges running BOTH ways: money moving in both directions between
# the same two institutions, unusual for a typical vendor/customer
# relationship but exactly what you'd expect from two carriers that each
# subrogate against the other across a portfolio of claims. This is
# checked on the *aggregated* graph deliberately -- subrogation recovery
# lags the original claim payment by weeks to months, so same-month
# reciprocity would badly undercount real relationships.

# %%
def payer_payee_profile(edges: pd.DataFrame, node_id: str) -> dict:
    incoming = edges.loc[edges["dest"] == node_id]
    outgoing = edges.loc[edges["source"] == node_id]
    return {
        "node_id": node_id,
        "n_payers": incoming["source"].nunique(),
        "n_payees": outgoing["dest"].nunique(),
        "amount_in": incoming["amount_total"].sum(),
        "amount_out": outgoing["amount_total"].sum(),
        "volume_in": incoming["volume_total"].sum(),
        "volume_out": outgoing["volume_total"].sum(),
    }

profiles = pd.DataFrame([payer_payee_profile(edges, n) for n in seed_ids])
profiles = profiles.merge(nodes[["node_id", "name"]], on="node_id", how="left")
print(profiles.sort_values("amount_in", ascending=False).head(20).to_string(index=False))

# %%
def reciprocal_pairs(edges: pd.DataFrame, node_subset: Set[str]) -> pd.DataFrame:
    e = edges.loc[edges["source"].isin(node_subset) & edges["dest"].isin(node_subset),
                  ["source", "dest", "amount_total", "volume_total", "n_rels"]]
    merged = e.merge(e, left_on=["source", "dest"], right_on=["dest", "source"],
                      suffixes=("_fwd", "_rev"))
    merged = merged.loc[merged["source_fwd"] < merged["dest_fwd"]]
    merged["balance_ratio"] = (
        merged[["amount_total_fwd", "amount_total_rev"]].min(axis=1) /
        merged[["amount_total_fwd", "amount_total_rev"]].max(axis=1)
    )
    return merged.sort_values("balance_ratio", ascending=False)

recip = reciprocal_pairs(edges, set(seed_ids))
name_lookup = nodes[["node_id", "name"]]
recip = recip.merge(name_lookup.rename(columns={"node_id": "source_fwd", "name": "source_name"}),
                     on="source_fwd", how="left")
recip = recip.merge(name_lookup.rename(columns={"node_id": "dest_fwd", "name": "dest_name"}),
                     on="dest_fwd", how="left")

print(f"\nReciprocal pairs among seed nodes: {len(recip):,}")
cols = ["source_name", "dest_name", "amount_total_fwd", "amount_total_rev",
        "balance_ratio", "n_rels_fwd", "n_rels_rev"]
print(recip[cols].head(20).to_string(index=False))
