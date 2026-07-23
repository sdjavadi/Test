"""
pkg_roles.py
============
Customer role taxonomies + NAICS peer ranking, built ON TOP of the node
panel produced by pkg_pipeline (../metrics/node/node_{YYYY-MM}.parquet).

Streams months in chronological order (same memory discipline as the
pipeline), keeps only small per-version state (previous roles, previous
strength), and writes per month:

    ../metrics/roles/roles_{YYYY-MM}.parquet
        one row per (version, node):
        - 5 role taxonomies (raw + hysteresis-stabilized + months_in_role)
        - strength_mom (log MoM strength ratio)
        - NAICS peer features: peer_level, peer_size, industry percentiles
          for key metrics, gap-to-peer-median strength

    ../metrics/roles/transitions.csv   (at the end)
        long-format transition counts (version, taxonomy, from, to, n)
        on STABILIZED roles — the behavioral-drift engine.

Design notes
------------
- All thresholds are PERCENTILE-CALIBRATED within (month, version), so
  they survive the ablation ladder and dollar inflation. Fixed cutoffs
  (0.5 flow_ratio etc.) are used only where the quantity is already a
  bounded ratio.
- Hysteresis: the stabilized role switches only after the same new raw
  role is observed 2 consecutive months; prevents threshold flicker.
- Peer percentiles use ADAPTIVE granularity: the finest NAICS level
  (4 -> 3 -> 2 digits) whose peer group has >= MIN_PEERS members that
  month. `peer_level` records which level was used; customers with
  unknown NAICS get NaN percentiles (never percentile vs 'unknown').
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_roles")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

NODE_DIR = "../metrics/node"
OUT_DIR = "../metrics/roles"
MIN_PEERS = 30
THIN_DEGREE = 3        # < 3 counterparts = thin file: concentration and
                       # turnover rules are arithmetic, not behavior

# metrics ranked within the industry peer group
PEER_METRICS = ["strength", "in_strength", "degree", "net_flow",
                "hhi_in", "pagerank_logw", "payer_jaccard", "strength_mom"]

TAXONOMIES = ["flow_role", "hierarchy_role", "dependence_role",
              "embeddedness_role", "dynamics_role"]


# ---------------------------------------------------------------------------
# role assignment (vectorized, percentile-calibrated per month x version)
# ---------------------------------------------------------------------------

def _q(s: pd.Series, p: float) -> float:
    v = s.dropna()
    return float(v.quantile(p)) if len(v) else np.nan


def assign_flow_role(df: pd.DataFrame) -> np.ndarray:
    thr_through = _q(df["throughflow"], 0.90)
    med_recip = _q(df.get("reciprocity_amount_share",
                          df.get("reciprocity_node_w",
                                 pd.Series(dtype=float))), 0.5)
    fr = df["flow_ratio"].to_numpy(float)
    tf = df["throughflow"].to_numpy(float)
    ins = df["in_strength"].to_numpy(float)
    outs = df["out_strength"].to_numpy(float)
    rec_s = df.get("reciprocity_amount_share",
                   df.get("reciprocity_node_w",
                          pd.Series(np.nan, index=df.index)))
    rec = rec_s.to_numpy(float)
    return np.select(
        [
            (ins > 0) & (outs <= 0),
            (outs > 0) & (ins <= 0),
            (np.abs(fr) <= 0.15) & (tf >= thr_through),
            fr > 0.5,
            fr < -0.5,
            (np.abs(fr) <= 0.15) & (rec >= med_recip),
        ],
        ["one_sided", "one_sided", "conduit",
         "collector", "distributor", "trader"],
        default="mixed")


def assign_hierarchy_role(df: pd.DataFrame) -> np.ndarray:
    t = df.get("trophic_level", pd.Series(np.nan, index=df.index))
    lo, hi = _q(t, 1 / 3), _q(t, 2 / 3)
    tv = t.to_numpy(float)
    return np.select(
        [np.isnan(tv), tv <= lo, tv >= hi],
        ["unknown", "upstream_supplier", "downstream_buyer"],
        default="midstream")


def assign_dependence_role(df: pd.DataFrame) -> np.ndarray:
    eligible = df["degree"] >= THIN_DEGREE
    med_hin = _q(df.loc[eligible, "hhi_in"], 0.5)
    med_hout = _q(df.loc[eligible, "hhi_out"], 0.5)
    hub_in = df.get("hub_in_share",
                    pd.Series(0.0, index=df.index)).fillna(0).to_numpy(float)
    t1i = df["top1_in_share"].fillna(0).to_numpy(float)
    t1o = df["top1_out_share"].fillna(0).to_numpy(float)
    hin = df["hhi_in"].to_numpy(float)
    hout = df["hhi_out"].to_numpy(float)
    return np.select(
        [
            ~eligible.to_numpy(),
            hub_in > 0.7,
            t1i > 0.7,
            t1o > 0.7,
            (hin <= med_hin) & (hout <= med_hout),
        ],
        ["single_relationship", "infra_dependent", "concentrated",
         "concentrated", "balanced"],
        default="balanced")


def assign_embeddedness_role(df: pd.DataFrame) -> np.ndarray:
    eligible = df["degree"] >= THIN_DEGREE
    core_s = df.get("core_number", pd.Series(dtype=float))
    thr_core = _q(core_s[eligible.reindex(core_s.index, fill_value=False)]
                  if len(core_s) else core_s, 0.90)
    lo_core = _q(core_s[eligible.reindex(core_s.index, fill_value=False)]
                 if len(core_s) else core_s, 0.25)
    part = df.get("participation_coef",
                  pd.Series(np.nan, index=df.index)).to_numpy(float)
    core = df.get("core_number",
                  pd.Series(np.nan, index=df.index)).to_numpy(float)
    intra = df.get("frac_intra_edges_w",
                   pd.Series(np.nan, index=df.index)).to_numpy(float)
    deg = df["degree"].to_numpy(float)
    # peripheral is checked FIRST: a degree-1 node has frac_intra = 1.0
    # trivially and previously flooded embedded_local (74% of the book).
    # local_anchor uses STRICT > p90 to prevent integer-tie inflation.
    return np.select(
        [
            (deg < THIN_DEGREE) | (core <= lo_core),
            part >= 0.62,
            (core > thr_core) & (intra >= 0.7),
            intra >= 0.7,
        ],
        ["peripheral", "connector", "embedded", "embedded"],
        default="intermediate")


def assign_dynamics_role(df: pd.DataFrame,
                         prev_indeg: pd.Series | None = None) -> np.ndarray:
    pj = df.get("payer_jaccard", pd.Series(np.nan, index=df.index))
    p25, p50 = _q(pj, 0.25), _q(pj, 0.5)
    mom = df["strength_mom"].to_numpy(float)
    lost = df.get("lost_payer_amount_share",
                  pd.Series(np.nan, index=df.index)).fillna(0).to_numpy(float)
    new = df.get("new_payer_amount_share",
                 pd.Series(np.nan, index=df.index)).fillna(0).to_numpy(float)
    tenure = df.get("months_since_first_seen",
                    pd.Series(np.nan, index=df.index)).to_numpy(float)
    gap = df.get("activity_gap",
                 pd.Series(np.nan, index=df.index)).to_numpy(float)
    pjv = pj.to_numpy(float)
    # thin-file gate: with 1-2 payers, losing one payer IS 50-100% of the
    # base — that is arithmetic, not bleeding. Require >= THIN_DEGREE
    # payers in the PRIOR month for bleeding, current month for expanding.
    if prev_indeg is not None:
        pin = prev_indeg.reindex(df["node"]).to_numpy(float)
    else:
        pin = np.full(len(df), np.nan)
    fat_prev = pin >= THIN_DEGREE
    fat_now = df["in_degree"].to_numpy(float) >= THIN_DEGREE
    return np.select(
        [
            tenure <= 2,
            gap > 1,
            fat_prev & ((lost >= 0.4)
                        | ((mom <= -0.3) & (pjv <= p25))),
            fat_now & (new >= 0.4) & (mom > 0.1),
            (pjv >= p50) & (np.abs(mom) < 0.35),
        ],
        ["newcomer", "intermittent", "bleeding", "expanding", "steady"],
        default="variable")


# ---------------------------------------------------------------------------
# NAICS peer percentiles (adaptive granularity)
# ---------------------------------------------------------------------------

def peer_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """For each customer: rank percentile of PEER_METRICS within the finest
    NAICS peer group (naics4 -> naics3 -> naics2) with >= MIN_PEERS members
    this month. Adds peer_level, peer_size, strength_vs_peer_median."""
    out = pd.DataFrame(index=df.index)
    levels = [("naics4", 4), ("naics3", 3), ("naics2", 2)]
    # group sizes at each level
    size = {}
    for col, _ in levels:
        if col in df.columns:
            size[col] = df.groupby(col, dropna=True)[col].transform("size")
        else:
            size[col] = pd.Series(np.nan, index=df.index)
    # pick finest adequate level per row
    level_choice = np.select(
        [df.get("naics4").notna() & (size["naics4"] >= MIN_PEERS),
         df.get("naics3").notna() & (size["naics3"] >= MIN_PEERS),
         df.get("naics2").notna() & (size["naics2"] >= MIN_PEERS)],
        [4, 3, 2], default=0)
    out["peer_level"] = level_choice
    out["peer_size"] = np.select(
        [level_choice == 4, level_choice == 3, level_choice == 2],
        [size["naics4"], size["naics3"], size["naics2"]], default=np.nan)
    for m in PEER_METRICS:
        out[f"{m}_pctl_naics"] = np.nan
    out["strength_vs_peer_median"] = np.nan

    for col, lv in levels:
        mask = level_choice == lv
        if not mask.any() or col not in df.columns:
            continue
        sub = df.loc[mask]
        g = sub.groupby(col)
        for m in PEER_METRICS:
            if m not in df.columns:
                continue
            out.loc[mask, f"{m}_pctl_naics"] = g[m].rank(pct=True)
        med = g["strength"].transform("median")
        out.loc[mask, "strength_vs_peer_median"] = np.log1p(
            sub["strength"]) - np.log1p(med)
    return out


# ---------------------------------------------------------------------------
# streaming driver with hysteresis + transition accumulation
# ---------------------------------------------------------------------------

def run(node_dir: str = NODE_DIR, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(node_dir, "node_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no node_*.parquet under {node_dir}")

    # per (version, taxonomy): previous raw + stabilized roles per node
    prev_raw: dict = {}
    prev_stable: dict = {}
    months_in: dict = {}
    prev_strength: dict = {}
    prev_indeg: dict = {}
    transitions: Counter = Counter()

    for path in paths:
        time_key = re.search(r"node_(\d{4}-\d{2})\.parquet", path).group(1)
        panel = pd.read_parquet(path)
        month_out = []

        for version, df in panel.groupby("version", observed=True):
            df = df.reset_index(drop=True)
            key = version

            # strength momentum vs previous month
            ps = prev_strength.get(key)
            if ps is not None:
                prev = ps.reindex(df["node"]).to_numpy(float)
                df["strength_mom"] = np.log(
                    (df["strength"].to_numpy(float) + 1.0) / (prev + 1.0))
                df.loc[pd.isna(prev), "strength_mom"] = np.nan
            else:
                df["strength_mom"] = np.nan
            prev_strength[key] = df.set_index("node")["strength"]

            roles = pd.DataFrame({"node": df["node"]})
            roles["flow_role"] = assign_flow_role(df)
            roles["hierarchy_role"] = assign_hierarchy_role(df)
            roles["dependence_role"] = assign_dependence_role(df)
            roles["embeddedness_role"] = assign_embeddedness_role(df)
            roles["dynamics_role"] = assign_dynamics_role(
                df, prev_indeg.get(key))
            prev_indeg[key] = df.set_index("node")["in_degree"]

            # hysteresis + months_in_role + transitions per taxonomy
            for tax in TAXONOMIES:
                raw = roles.set_index("node")[tax]
                pr = prev_raw.get((key, tax))
                pstab = prev_stable.get((key, tax))
                if pstab is None:
                    stable = raw.copy()
                    m_in = pd.Series(1, index=raw.index)
                else:
                    pstab_a = pstab.reindex(raw.index)
                    pr_a = pr.reindex(raw.index)
                    confirmed = raw.eq(pr_a)          # same new role 2x
                    stable = pstab_a.where(~confirmed, raw)
                    stable = stable.fillna(raw)       # nodes new this month
                    prev_m = months_in.get((key, tax),
                                           pd.Series(dtype=float)
                                           ).reindex(raw.index)
                    m_in = np.where(stable.eq(pstab_a) & pstab_a.notna(),
                                    prev_m.fillna(0) + 1, 1)
                    m_in = pd.Series(m_in, index=raw.index)
                    # transitions on stabilized roles
                    ch = stable[pstab_a.notna() & stable.ne(pstab_a)]
                    frm = pstab_a.loc[ch.index]
                    for f, t_ in zip(frm, ch):
                        transitions[(key, tax, f, t_)] += 1
                roles[f"{tax}_stable"] = stable.reindex(
                    roles["node"]).to_numpy()
                roles[f"{tax}_months_in"] = m_in.reindex(
                    roles["node"]).to_numpy()
                prev_raw[(key, tax)] = raw
                prev_stable[(key, tax)] = stable
                months_in[(key, tax)] = m_in

            peers = peer_percentiles(df)

            # -- role stability -------------------------------------------
            mi_cols = [f"{tax}_months_in" for tax in TAXONOMIES]
            mi = roles[mi_cols].astype(float)
            roles["role_stability_score"] = mi.mean(axis=1)
            # normalized by months observed (a 2-month customer can't have
            # a 10-month stable role); clipped to [0, 1]
            obs = df.set_index("node")["months_active"].reindex(
                roles["node"]).to_numpy(float) \
                if "months_active" in df.columns else np.nan
            roles["role_stability_norm"] = np.clip(
                roles["role_stability_score"] / np.maximum(obs, 1.0), 0, 1)
            # how many taxonomies switched stable role THIS month
            # (months_in == 1 signals a fresh role; masked for nodes we
            # have never seen before, where all five are trivially 1)
            seen_before = roles["node"].isin(
                ps.index if ps is not None else [])
            n_switch = (mi == 1).sum(axis=1).astype(float)
            roles["n_role_switches"] = np.where(seen_before, n_switch,
                                                np.nan)

            id_cols = [c for c in ("cust_name", "naics_desc")
                       if c in df.columns]
            block = pd.concat(
                [roles.reset_index(drop=True),
                 df[["strength_mom", "naics2", "naics3", "naics4"]
                    + id_cols].reset_index(drop=True),
                 peers.reset_index(drop=True)], axis=1)
            block.insert(0, "version", version)
            block.insert(0, "time_key", time_key)
            month_out.append(block)

        out = pd.concat(month_out, ignore_index=True)
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype("string")
            elif out[c].dtype == np.float64:
                out[c] = out[c].astype(np.float32)
        p = os.path.join(out_dir, f"roles_{time_key}.parquet")
        out.to_parquet(p)
        log.info("wrote %s (%d rows, %d cols)", p, len(out), out.shape[1])
        del panel, month_out, out
        gc.collect()

    tr = pd.DataFrame(
        [(k[0], k[1], k[2], k[3], n) for k, n in transitions.items()],
        columns=["version", "taxonomy", "from_role", "to_role", "n"]
    ).sort_values(["version", "taxonomy", "n"], ascending=[1, 1, 0])
    tr.to_csv(os.path.join(out_dir, "transitions.csv"), index=False)
    log.info("wrote transitions.csv (%d distinct transitions)", len(tr))


if __name__ == "__main__":
    run()
