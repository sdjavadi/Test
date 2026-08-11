"""
pkg_geo_metrics.py
==================
Per-node geographic metrics, computed from a version's edge list plus the
customer coordinate table.

Computed PER VERSION, not once on RAW. Hub removal changes who a node's
counterparties are, and the V0 -> P99_9 comparison reversed six geographic
findings; a single RAW-graph geo block would carry that distortion into
every version.

Two families, deliberately separated because they have different
requirements and different failure modes:

  SPREAD  — properties of the counterparty cloud. Needs only the
            counterparties' coordinates, so it is defined for a node with no
            location of its own. This is the family that transfers to
            counterparties later.

  REACH   — distance from the node's OWN point to its counterparties. Needs
            the node's own coordinates. NaN otherwise, never zero.

Vectorised throughout: no iterrows, no per-node apply. Weighted percentiles
and the 80%-coverage count use the global-sort + groupby-cumsum pattern
rather than groupby.quantile / groupby.nlargest.

All distances are amount-weighted. Great-circle, WGS84 mean radius.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_geo_metrics")

R_EARTH_KM = 6371.0088

# locality bands (km of counterparty spread)
LOCALITY_BANDS = ((50.0, "LOCAL"), (250.0, "REGIONAL"),
                  (1000.0, "MULTI_MARKET"))
LOCALITY_TOP = "NATIONAL"

MIN_CP_FOR_SPREAD = 2      # below this, spread is undefined not zero
MIN_CP_FOR_ENTROPY = 5     # Shannon entropy is biased low at small n


def _haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dla, dlo = la2 - la1, lo2 - lo1
    a = np.sin(dla / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def _weighted_centroid(df: pd.DataFrame) -> pd.DataFrame:
    """Amount-weighted centroid via 3D unit vectors.

    Averaging lat/lon directly is wrong across the antimeridian and biased
    for wide spreads; the vector mean is correct everywhere and yields the
    resultant length R (0 = maximally dispersed, 1 = all at one point) as a
    by-product, which is a spread measure that needs no distance pass.
    """
    la, lo = np.radians(df["cp_lat"].to_numpy(float)), \
        np.radians(df["cp_lon"].to_numpy(float))
    w = df["w"].to_numpy(float)
    t = pd.DataFrame({
        "node": df["node"].to_numpy(),
        "x": w * np.cos(la) * np.cos(lo),
        "y": w * np.cos(la) * np.sin(lo),
        "z": w * np.sin(la),
        "w": w,
    })
    g = t.groupby("node", sort=False)[["x", "y", "z", "w"]].sum()
    norm = np.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)
    out = pd.DataFrame(index=g.index)
    out["geo_centroid_lat"] = np.degrees(np.arctan2(g["z"], np.hypot(g["x"], g["y"])))
    out["geo_centroid_lon"] = np.degrees(np.arctan2(g["y"], g["x"]))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["geo_R"] = np.divide(norm.to_numpy(), g["w"].to_numpy(),
                                 out=np.full(len(g), np.nan),
                                 where=g["w"].to_numpy() > 0)
    return out


def _weighted_pctiles(node, dist, w, qs=(0.5, 0.9), prefix="reach"):
    """Amount-weighted percentiles of `dist` per node.

    Global sort + groupby cumsum; no groupby.quantile (which ignores weights)
    and no per-group apply.
    """
    d = pd.DataFrame({"node": node, "d": dist, "w": w})
    d = d[np.isfinite(d["d"]) & (d["w"] > 0)]
    if d.empty:
        return pd.DataFrame(columns=[f"geo_{prefix}_p{int(q*100)}_km" for q in qs])
    d = d.sort_values(["node", "d"], kind="stable")
    tot = d.groupby("node", sort=False)["w"].transform("sum")
    frac = d.groupby("node", sort=False)["w"].cumsum() / tot
    out = None
    for q in qs:
        hit = (d.loc[frac >= q].groupby("node", sort=False)["d"].first()
               .rename(f"geo_{prefix}_p{int(q*100)}_km"))
        out = hit.to_frame() if out is None else out.join(hit, how="outer")
    return out


def _coverage_count(node, share_key, w, target=0.80, name="geo_n_zip3_80"):
    """Distinct `share_key` values covering `target` of amount, per node."""
    d = pd.DataFrame({"node": node, "k": share_key, "w": w}).dropna(subset=["k"])
    if d.empty:
        return pd.Series(dtype="float32", name=name)
    g = d.groupby(["node", "k"], sort=False)["w"].sum().reset_index()
    g = g.sort_values(["node", "w"], ascending=[True, False], kind="stable")
    tot = g.groupby("node", sort=False)["w"].transform("sum")
    cum = g.groupby("node", sort=False)["w"].cumsum()
    prev = cum - g["w"]
    # count every key whose cumulative share BEFORE it is still under target
    return (g.loc[prev < target * tot]
            .groupby("node", sort=False).size().rename(name).astype("float32"))


def _entropy(node, share_key, w, name="geo_zip3_entropy"):
    d = pd.DataFrame({"node": node, "k": share_key, "w": w}).dropna(subset=["k"])
    if d.empty:
        return pd.Series(dtype="float32", name=name)
    g = d.groupby(["node", "k"], sort=False)["w"].sum()
    tot = g.groupby(level=0).sum()
    p = g / tot.reindex(g.index.get_level_values(0)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return (pd.Series(terms, index=g.index).groupby(level=0).sum()
            .rename(name).astype("float32"))


def _side(e: pd.DataFrame, coords: pd.DataFrame, node_col: str, cp_col: str,
          tag: str) -> pd.DataFrame:
    """Geo metrics for one direction ('in' = payers, 'out' = payees)."""
    cm = coords.set_index("node")
    df = pd.DataFrame({
        "node": e[node_col].to_numpy(),
        "cp": e[cp_col].to_numpy(),
        "w": e["amount"].to_numpy(float),
    })
    df["cp_lat"] = cm["lat"].reindex(df["cp"]).to_numpy(float)
    df["cp_lon"] = cm["lon"].reindex(df["cp"]).to_numpy(float)
    df["cp_zip3"] = cm["zip3"].reindex(df["cp"]).to_numpy()
    df["cp_state"] = cm["state"].reindex(df["cp"]).to_numpy()

    tot_w = df.groupby("node", sort=False)["w"].sum()
    loc = df[df["cp_lat"].notna() & df["cp_lon"].notna()]

    out = pd.DataFrame(index=tot_w.index)
    # coverage: what share of this side's dollars has a located counterparty
    cov = loc.groupby("node", sort=False)["w"].sum()
    out[f"geo_cov_amt_{tag}"] = (cov / tot_w).astype("float32")
    out[f"geo_n_cp_located_{tag}"] = (
        loc.drop_duplicates(["node", "cp"]).groupby("node", sort=False).size()
        .astype("float32"))

    if loc.empty:
        return out

    cen = _weighted_centroid(loc)
    d_cen = _haversine(loc["cp_lat"].to_numpy(float), loc["cp_lon"].to_numpy(float),
                       cen["geo_centroid_lat"].reindex(loc["node"]).to_numpy(float),
                       cen["geo_centroid_lon"].reindex(loc["node"]).to_numpy(float))
    sq = pd.DataFrame({"node": loc["node"].to_numpy(),
                       "wd2": loc["w"].to_numpy(float) * d_cen ** 2,
                       "w": loc["w"].to_numpy(float)})
    gsq = sq.groupby("node", sort=False)[["wd2", "w"]].sum()
    n_cp = loc.drop_duplicates(["node", "cp"]).groupby("node", sort=False).size()
    spread = np.sqrt(np.divide(gsq["wd2"].to_numpy(), gsq["w"].to_numpy(),
                               out=np.full(len(gsq), np.nan),
                               where=gsq["w"].to_numpy() > 0))
    spread = pd.Series(spread, index=gsq.index)
    out[f"geo_spread_{tag}_km"] = spread.where(
        n_cp.reindex(spread.index) >= MIN_CP_FOR_SPREAD).astype("float32")

    # home-market concentration.
    # Distinct NA sentinels on each side so that unknown == unknown is False:
    # pandas 'string' dtype propagates pd.NA through ==, and the resulting
    # nullable boolean is ambiguous under & / .to_numpy(bool).
    # Denominator is LOCATED dollars, not total: dividing by total would
    # confound home concentration with geocoding coverage, which is already
    # reported separately as geo_cov_amt_{tag}.
    loc_w = loc.groupby("node", sort=False)["w"].sum()
    for key, col in (("zip3", f"geo_home_zip3_share_{tag}"),
                     ("state", f"geo_home_state_share_{tag}")):
        own = pd.Series(cm[key].reindex(loc["node"]).to_numpy(),
                        index=loc.index).astype("string").fillna("\x00OWN")
        cp = pd.Series(loc[f"cp_{key}"].to_numpy(),
                       index=loc.index).astype("string").fillna("\x00CP")
        flag = (own == cp).to_numpy(dtype=bool)
        num = loc.loc[flag].groupby("node", sort=False)["w"].sum()
        val = (num / loc_w.reindex(num.index)).reindex(out.index)
        # A node with located counterparties and a known own key, none of
        # which match, has a home share of exactly 0 — not "unknown". NaN is
        # reserved for no located counterparties, or no own key to compare to.
        own_known = pd.Series(cm[key].reindex(out.index).notna().to_numpy(),
                              index=out.index)
        has_loc = pd.Series(out.index.isin(loc_w.index), index=out.index)
        val = val.mask(val.isna() & own_known & has_loc, 0.0)
        out[col] = val.astype("float32")

    return out


def geo_node_metrics(edges: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Per-node geographic block for one (month, version).

    edges  : source, dest, amount  (self-edges are dropped)
    coords : node, lat, lon, zip3, zip5, state   — geo_status == 'valid' only

    Returns one row per node appearing in `edges`. Nodes with no located
    counterparty get NaN spread/reach, never zero.
    """
    if coords is None or coords.empty:
        return pd.DataFrame(columns=["node"])

    e = edges.loc[edges["source"] != edges["dest"], ["source", "dest", "amount"]]
    if e.empty:
        return pd.DataFrame(columns=["node"])

    cm = coords.set_index("node")
    ins = _side(e, coords, "dest", "source", "in")
    outs = _side(e, coords, "source", "dest", "out")
    res = ins.join(outs, how="outer")
    res.index.name = "node"

    # ---- combined (both directions pooled) --------------------------------
    both = pd.concat([
        pd.DataFrame({"node": e["dest"].to_numpy(), "cp": e["source"].to_numpy(),
                      "w": e["amount"].to_numpy(float)}),
        pd.DataFrame({"node": e["source"].to_numpy(), "cp": e["dest"].to_numpy(),
                      "w": e["amount"].to_numpy(float)}),
    ], ignore_index=True)
    both["cp_lat"] = cm["lat"].reindex(both["cp"]).to_numpy(float)
    both["cp_lon"] = cm["lon"].reindex(both["cp"]).to_numpy(float)
    both["cp_zip3"] = cm["zip3"].reindex(both["cp"]).to_numpy()
    loc = both[both["cp_lat"].notna() & both["cp_lon"].notna()]

    if not loc.empty:
        cen = _weighted_centroid(loc)
        res = res.join(cen.astype("float32"), how="outer")

        d_cen = _haversine(
            loc["cp_lat"].to_numpy(float), loc["cp_lon"].to_numpy(float),
            cen["geo_centroid_lat"].reindex(loc["node"]).to_numpy(float),
            cen["geo_centroid_lon"].reindex(loc["node"]).to_numpy(float))
        sq = pd.DataFrame({"node": loc["node"].to_numpy(),
                           "wd2": loc["w"].to_numpy(float) * d_cen ** 2,
                           "w": loc["w"].to_numpy(float)})
        gsq = sq.groupby("node", sort=False)[["wd2", "w"]].sum()
        n_cp = loc.drop_duplicates(["node", "cp"]).groupby("node", sort=False).size()
        spread = pd.Series(
            np.sqrt(np.divide(gsq["wd2"].to_numpy(), gsq["w"].to_numpy(),
                              out=np.full(len(gsq), np.nan),
                              where=gsq["w"].to_numpy() > 0)),
            index=gsq.index)
        res["geo_spread_km"] = spread.where(
            n_cp.reindex(spread.index) >= MIN_CP_FOR_SPREAD).astype("float32")

        # market breadth
        res = res.join(_coverage_count(loc["node"].to_numpy(),
                                       loc["cp_zip3"].to_numpy(),
                                       loc["w"].to_numpy(float)), how="left")
        ent = _entropy(loc["node"].to_numpy(), loc["cp_zip3"].to_numpy(),
                       loc["w"].to_numpy(float))
        res = res.join(ent.where(n_cp.reindex(ent.index) >= MIN_CP_FOR_ENTROPY),
                       how="left")

        # ---- REACH: needs the node's own point ---------------------------
        own_lat = cm["lat"].reindex(loc["node"]).to_numpy(float)
        own_lon = cm["lon"].reindex(loc["node"]).to_numpy(float)
        has_own = np.isfinite(own_lat) & np.isfinite(own_lon)
        if has_own.any():
            dist = np.where(
                has_own,
                _haversine(own_lat, own_lon,
                           loc["cp_lat"].to_numpy(float),
                           loc["cp_lon"].to_numpy(float)),
                np.nan)
            wsum = pd.DataFrame({"node": loc["node"].to_numpy(),
                                 "wd": loc["w"].to_numpy(float) * dist,
                                 "w": np.where(np.isfinite(dist),
                                               loc["w"].to_numpy(float), 0.0)})
            gw = wsum.groupby("node", sort=False)[["wd", "w"]].sum()
            res["geo_reach_mean_km"] = pd.Series(
                np.divide(gw["wd"].to_numpy(), gw["w"].to_numpy(),
                          out=np.full(len(gw), np.nan),
                          where=gw["w"].to_numpy() > 0),
                index=gw.index).astype("float32")
            res = res.join(
                _weighted_pctiles(loc["node"].to_numpy(), dist,
                                  loc["w"].to_numpy(float)).astype("float32"),
                how="left")

            # representativeness: registered point vs flow-weighted centroid
            idx = res.index
            rl = cm["lat"].reindex(idx).to_numpy(float)
            rlon = cm["lon"].reindex(idx).to_numpy(float)
            cl = res.get("geo_centroid_lat", pd.Series(np.nan, index=idx)) \
                    .to_numpy(float)
            clon = res.get("geo_centroid_lon", pd.Series(np.nan, index=idx)) \
                      .to_numpy(float)
            ok = np.isfinite(rl) & np.isfinite(cl)
            rv = np.full(len(idx), np.nan)
            rv[ok] = _haversine(rl[ok], rlon[ok], cl[ok], clon[ok])
            res["geo_registered_vs_flow_km"] = rv.astype("float32")

    # ---- locality class ---------------------------------------------------
    if "geo_spread_km" in res.columns:
        s = res["geo_spread_km"]
        cls = np.full(len(res), None, dtype=object)
        assigned = ~np.isfinite(s.to_numpy(float))
        for cut, label in LOCALITY_BANDS:
            m = (~assigned) & (s.to_numpy(float) < cut)
            cls[m] = label
            assigned |= m
        cls[(~assigned) & np.isfinite(s.to_numpy(float))] = LOCALITY_TOP
        res["geo_locality_class"] = pd.Series(cls, index=res.index).astype("string")

    return res.reset_index()
