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

# Canonical output schema. Every column is emitted on every call, NaN-filled
# when the month/version cannot support it — a month where (say) no node has
# its own coordinates must not silently drop `geo_reach_*`, or the parquet
# schema drifts and the cross-month concat breaks. Mirrors the
# schema-stability rule in PKG_MONTHLY_METRICS_MANIFEST §0.
GEO_FLOAT_COLUMNS = [
    "geo_cov_amt_in", "geo_n_cp_located_in", "geo_spread_in_km",
    "geo_home_zip3_share_in", "geo_home_state_share_in",
    "geo_cov_amt_out", "geo_n_cp_located_out", "geo_spread_out_km",
    "geo_home_zip3_share_out", "geo_home_state_share_out",
    "geo_centroid_lat", "geo_centroid_lon", "geo_R",
    "geo_spread_km", "geo_n_zip3_80", "geo_zip3_entropy",
    "geo_reach_mean_km", "geo_reach_p50_km", "geo_reach_p90_km",
    "geo_registered_vs_flow_km",
]
GEO_STR_COLUMNS = ["geo_locality_class"]
GEO_COLUMNS = GEO_FLOAT_COLUMNS + GEO_STR_COLUMNS


# ---------------------------------------------------------------------------
# prepared per-month lookup
# ---------------------------------------------------------------------------

class GeoIndex:
    """Coordinate lookup prepared ONCE per month, keyed by integer codes.

    The naive form — `coords.set_index("node")` then `.reindex(edge_ids)` —
    is what made geo the slowest component in the first production run
    (405s on V0, but still 314s on P99 with 4.4x fewer edges: an ~287s fixed
    cost that had nothing to do with edge count). Two causes:

      * the customer table is ~20M rows while a month touches ~2M nodes, so
        every lookup built and probed a hash table 10x larger than needed;
      * `set_index` + `reindex` ran 12 times per month (3 calls x 4 versions)
        on an *object*-dtype string index, which hashes per element.

    Fix: restrict to the month's nodes, factorize ids to int32 codes once,
    and hold plain numpy arrays. Lookups become O(1) array indexing.
    """

    __slots__ = ("codes", "ids", "lat", "lon", "zip3", "state", "has_geo", "n")

    def __init__(self, coords: pd.DataFrame, nodes=None):
        c = coords.drop_duplicates("node")
        if nodes is not None:
            # Every node in the month gets a code, located or not: an
            # unlocated node still has SPREAD over its located counterparties
            # (that separation is the point of the SPREAD/REACH split), so
            # dropping it here would silently lose those rows.
            # get_indexer, not isin — isin on a 20M-row arrow-string index
            # cost ~9s per call in profiling, and the positions are needed
            # anyway.
            want = pd.Index(pd.unique(np.asarray(nodes)))
            pos = pd.Index(c["node"].to_numpy()).get_indexer(want)
            hit = pos >= 0
            c = pd.concat([
                c.iloc[pos[hit]],
                pd.DataFrame({"node": want[~hit].to_numpy()}).reindex(
                    columns=c.columns),
            ], ignore_index=True)
        # -1 is the "unknown node" code; every array carries a sentinel row 0
        # so a miss indexes safely instead of needing a mask at every call.
        self.ids = np.asarray(c["node"].to_numpy(), dtype=object)
        self.codes = pd.Index(self.ids)
        self.n = len(self.codes)
        # sentinel id for the "unknown node" row
        self.ids = np.append(self.ids, None)
        self.lat = np.append(c["lat"].to_numpy(np.float64), np.nan)
        self.lon = np.append(c["lon"].to_numpy(np.float64), np.nan)
        # zip3 / state as integer codes: string equality on 5M rows is the
        # second-largest cost after the reindex it replaces
        z, zu = pd.factorize(c["zip3"].to_numpy(), use_na_sentinel=True)
        s, su = pd.factorize(c["state"].to_numpy(), use_na_sentinel=True)
        self.zip3 = np.append(z.astype(np.int32), -1)
        self.state = np.append(s.astype(np.int32), -1)
        self.has_geo = np.append(np.isfinite(c["lat"].to_numpy(np.float64))
                                 & np.isfinite(c["lon"].to_numpy(np.float64)),
                                 False)

    def __len__(self) -> int:
        return self.n

    def code(self, ids) -> np.ndarray:
        """Map node ids -> row positions; misses -> the sentinel row."""
        pos = self.codes.get_indexer(pd.Index(np.asarray(ids)))
        return np.where(pos < 0, self.n, pos)


def _prepare(coords, edges) -> GeoIndex:
    if isinstance(coords, GeoIndex):
        return coords
    nodes = np.concatenate([edges["source"].to_numpy(),
                            edges["dest"].to_numpy()])
    return GeoIndex(coords, nodes)


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


def _side(e: pd.DataFrame, gi: "GeoIndex", node_col: str, cp_col: str,
          tag: str, node_code=None, cp_code=None) -> pd.DataFrame:
    """Geo metrics for one direction ('in' = payers, 'out' = payees).

    node_code / cp_code are precomputed row positions; passing them lets the
    caller factorize each endpoint once instead of once per direction.
    """
    nc = gi.code(e[node_col]) if node_code is None else node_code
    cc = gi.code(e[cp_col]) if cp_code is None else cp_code
    # GROUP BY INTEGER CODES, NEVER STRINGS. Every pandas groupby re-factorizes
    # its key; on an arrow-backed string column that was ~0.19s per groupby and
    # this function performs dozens of them. Codes are int32 -> the factorize
    # is a no-op. Ids are restored once, at the end of geo_node_metrics.
    df = pd.DataFrame({
        "node": nc,
        "cp": cc,
        "w": e["amount"].to_numpy(float),
        "cp_lat": gi.lat[cc],
        "cp_lon": gi.lon[cc],
        "cp_zip3": gi.zip3[cc],
        "cp_state": gi.state[cc],
        "_own_zip3": gi.zip3[nc],
        "_own_state": gi.state[nc],
    })

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
        # integer codes: -1 is "unknown", and -1 == -1 must NOT count as a
        # match, so require both sides known. (The old string form used
        # distinct NA sentinels to get the same effect.)
        own = loc[f"_own_{key}"].to_numpy()
        cpv = loc[f"cp_{key}"].to_numpy()
        flag = (own >= 0) & (cpv >= 0) & (own == cpv)
        num = loc.loc[flag].groupby("node", sort=False)["w"].sum()
        val = (num / loc_w.reindex(num.index)).reindex(out.index)
        # A node with located counterparties and a known own key, none of
        # which match, has a home share of exactly 0 — not "unknown". NaN is
        # reserved for no located counterparties, or no own key to compare to.
        arr = gi.zip3 if key == "zip3" else gi.state
        own_known = pd.Series(arr[out.index.to_numpy()] >= 0, index=out.index)
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
    if coords is None or (hasattr(coords, "empty") and coords.empty):
        return pd.DataFrame(columns=["node"] + GEO_COLUMNS)

    e = edges.loc[edges["source"] != edges["dest"], ["source", "dest", "amount"]]
    if e.empty:
        return pd.DataFrame(columns=["node"] + GEO_COLUMNS)

    # one factorization pass for the whole call: each endpoint is coded once
    # and both directions reuse it
    gi = _prepare(coords, e)
    src_c = gi.code(e["source"])
    dst_c = gi.code(e["dest"])

    ins = _side(e, gi, "dest", "source", "in", node_code=dst_c, cp_code=src_c)
    outs = _side(e, gi, "source", "dest", "out", node_code=src_c, cp_code=dst_c)
    res = ins.join(outs, how="outer")
    res.index.name = "node"

    # ---- combined (both directions pooled) --------------------------------
    both = pd.concat([
        pd.DataFrame({"node": dst_c, "cp": src_c,
                      "w": e["amount"].to_numpy(float)}),
        pd.DataFrame({"node": src_c, "cp": dst_c,
                      "w": e["amount"].to_numpy(float)}),
    ], ignore_index=True)
    cpc = np.concatenate([src_c, dst_c])
    both["cp_lat"] = gi.lat[cpc]
    both["cp_lon"] = gi.lon[cpc]
    both["cp_zip3"] = gi.zip3[cpc]
    loc = both[np.isfinite(both["cp_lat"]) & np.isfinite(both["cp_lon"])]

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
        own_c = loc["node"].to_numpy()
        own_lat = gi.lat[own_c]
        own_lon = gi.lon[own_c]
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
            idx_c = idx.to_numpy()
            rl = gi.lat[idx_c]
            rlon = gi.lon[idx_c]
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

    # schema stability: fill any column this month/version could not produce
    for c in GEO_FLOAT_COLUMNS:
        if c not in res.columns:
            res[c] = np.float32(np.nan)
    for c in GEO_STR_COLUMNS:
        if c not in res.columns:
            res[c] = pd.Series(pd.NA, index=res.index, dtype="string")
    res = res[GEO_COLUMNS]
    res = res.reset_index()
    # codes -> ids, once, at the boundary
    res["node"] = gi.ids[res["node"].to_numpy()]
    return res
