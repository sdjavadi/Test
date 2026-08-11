"""
pkg_customers.py
================
Customer dimension loader. Replaces the two full passes over the monthly
snapshots that previously derived node identity, NAICS and entity typing.

Input: customers.csv
    mdm_id, customer_name, naics_cd, naics_desc, addr_loc_rec_type,
    longitude_degree, latitude_degree, zip_cd, state_or_province, city

Design notes
------------
- **Current-state, not point-in-time.** One customer row is applied to every
  historical snapshot. Location / name / NAICS rarely change and a correct
  current value is a better estimate of last March than a stale snapshot one.
  Consequence: re-running after a customer refresh CHANGES historical metrics.
  That is deliberate; bump ATTR_VERSION when it happens.
- **NAICS code and description arrive as separate columns.** No '|' splitting.
- **Absence is empty, not a placeholder token.** naics_status therefore
  collapses to valid|missing in practice; `placeholder` is retained in the
  enum with a minimal residual rule so that if a sentinel ever reappears we
  see it in the counts rather than silently classifying it valid.
- **IDs are strings everywhere.** mdm_id is read as str and never coerced.

Public API
----------
    cust = load_customers("../data/customers.csv")
    cust.attrs    # node -> identity + NAICS + geo passthrough columns
    cust.typing   # node -> entity_type, naics_status, node_type, naics_clean
    cust.coords   # node -> lat, lon, zip3, state, geo_status  (valid only)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_customers")

ATTR_VERSION = "cust_v1"

# Residual sentinels. Expected to match ~0 rows now that absence is empty;
# kept so a regression is visible in the logged counts instead of silent.
PLACEHOLDER_NAICS = {"-1", "0", "00", "000000", "999999", "******",
                     "UNKNOWN", "N/A", "NA", "NULL", "NONE"}

# USPS AIS record types that do not denote the customer's own location.
PLACEHOLDER_REC_TYPES = {"POSTOFFICEBOX", "GENERALDELIVERY"}
SHARED_STRUCTURE_REC_TYPES = {"HIGHRISE"}

_BUSINESS_TOKENS = {
    "LLC", "INC", "CORP", "CORPORATION", "LTD", "LP", "LLP", "LLLP", "PLLC",
    "PC", "CO", "COMPANY", "TRUST", "DBA", "ASSOC", "ASSOCIATES",
    "ASSOCIATION", "GROUP", "HOLDINGS", "ENTERPRISES", "ENTERPRISE",
    "FOUNDATION", "CHURCH", "MINISTRIES", "SCHOOL", "ACADEMY", "UNIVERSITY",
    "COLLEGE", "CITY", "COUNTY", "BANK", "PARTNERS", "PARTNERSHIP",
    "SERVICES", "SOLUTIONS", "SYSTEMS", "CONSTRUCTION", "CONTRACTING",
    "PROPERTIES", "REALTY", "MEDICAL", "DENTAL", "CLINIC", "HOSPITAL",
    "LAW", "FARMS", "MOTORS", "AUTO", "RESTAURANT", "MARKET", "SHOP",
    "SALON", "AGENCY", "INDUSTRIES", "CENTER", "CENTRE", "&",
}

# canonical -> accepted aliases in the source file
_COLMAP = {
    "mdm_id":            ("mdm_id", "node", "cust_pwr_id"),
    "customer_name":     ("customer_name", "cust_name", "name"),
    "naics_cd":          ("naics_cd", "naics_code", "naics"),
    "naics_desc":        ("naics_desc", "naics_description"),
    "addr_loc_rec_type": ("addr_loc_rec_type", "rec_type"),
    "longitude_degree":  ("longitude_degree", "longitude_degrees", "lon",
                          "longitude"),
    "latitude_degree":   ("latitude_degree", "latitude_degrees", "lat",
                          "latitude"),
    "zip_cd":            ("zip_cd", "zip", "postal_cd"),
    "state_or_province": ("state_or_province", "state"),
    "city":              ("city",),
}

_NUM_RE = r"^\s*-?\d+(\.\d+)?\s*$"


@dataclass
class Customers:
    attrs: pd.DataFrame     # node + all passthrough columns
    typing: pd.DataFrame    # node, entity_type, naics_status, node_type, naics_clean
    coords: pd.DataFrame    # node, lat, lon, zip3, state, geo_status (valid rows only)
    n_rows: int = 0

    def coverage(self, nodes) -> dict:
        """Attribute coverage over an arbitrary node universe."""
        nodes = pd.Index(pd.unique(pd.Series(nodes, dtype="object").astype(str)))
        a = self.attrs.set_index("node")
        c = self.coords.set_index("node")
        n = max(len(nodes), 1)
        return {
            "n_nodes": len(nodes),
            "pct_in_customers": round(100 * nodes.isin(a.index).sum() / n, 2),
            "pct_naics_valid": round(
                100 * nodes.isin(self.typing.loc[
                    self.typing["naics_status"] == "valid", "node"]).sum() / n, 2),
            "pct_geo_valid": round(100 * nodes.isin(c.index).sum() / n, 2),
        }


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise header whitespace/case and map aliases to canonical names."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    out = {}
    for canon, aliases in _COLMAP.items():
        for a in aliases:
            if a in df.columns:
                out[canon] = df[a]
                break
        else:
            out[canon] = pd.Series(pd.NA, index=df.index, dtype="string")
            log.warning("customers.csv: column '%s' not found — filled empty",
                        canon)
    return pd.DataFrame(out)


def _num(series: pd.Series) -> pd.Series:
    """String -> float only where the string is actually numeric.

    latitude/longitude arrive as text; empty strings, whitespace and 'null'
    tokens all survive a plain isna() check, so screen before casting.
    """
    s = series.astype("string").str.strip()
    ok = s.str.match(_NUM_RE).fillna(False)
    return pd.to_numeric(s.where(ok), errors="coerce")


def _tokenize(names: pd.Series) -> pd.Series:
    up = names.fillna("").astype(str).str.upper()
    punct = re.compile(r"[^A-Z0-9& ]+")
    return up.map(lambda s: punct.sub(" ", s).split())


def load_customers(path: str) -> Customers:
    """Read customers.csv and derive typing + geo status. One pass."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"customers file not found: {path}")

    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df = _resolve_columns(raw)
    n_raw = len(df)

    df["node"] = df["mdm_id"].astype("string").str.strip()
    df = df[df["node"].notna() & (df["node"] != "")]
    dup = int(df["node"].duplicated().sum())
    if dup:
        log.warning("customers: %d duplicate mdm_id rows — keeping first", dup)
        df = df.drop_duplicates("node", keep="first")

    for c in ("customer_name", "naics_cd", "naics_desc",
              "addr_loc_rec_type", "zip_cd", "state_or_province", "city"):
        df[c] = df[c].astype("string").str.strip()

    # ---- NAICS ------------------------------------------------------------
    code = df["naics_cd"].fillna("")
    is_missing = (code == "").to_numpy(dtype=bool)
    is_ph = code.str.upper().isin(PLACEHOLDER_NAICS).fillna(False).to_numpy(bool)
    looks_code = code.str.match(r"^\d{2}").fillna(False).to_numpy(bool)
    naics_status = np.select(
        [is_missing, is_ph, looks_code],
        ["missing", "placeholder", "valid"], default="placeholder")
    naics_clean = code.str.extract(r"^(\d{2,6})")[0].where(
        pd.Series(naics_status == "valid", index=df.index))

    # ---- entity type ------------------------------------------------------
    toks = _tokenize(df["customer_name"])
    has_biz = toks.map(lambda t: any(w in _BUSINESS_TOKENS for w in t)) \
                  .to_numpy(dtype=bool)
    n_alpha = toks.map(lambda t: sum(w.isalpha() for w in t)).to_numpy(int)
    n_tok = toks.map(len).to_numpy(int)
    person_shaped = (~has_biz) & (n_alpha == n_tok) & (n_tok >= 2) & (n_tok <= 3)
    entity_type = np.select([has_biz, person_shaped],
                            ["business", "individual"], default="unknown")

    # ---- node_type: valid NAICS wins; otherwise the name decides ----------
    is_valid = naics_status == "valid"
    is_biz = entity_type == "business"
    node_type = np.select(
        [is_valid,
         is_biz & (naics_status == "placeholder"),
         is_biz & (naics_status == "missing"),
         entity_type == "individual"],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual"],
        default="unknown")

    # ---- geography --------------------------------------------------------
    lat = _num(df["latitude_degree"])
    lon = _num(df["longitude_degree"])
    rec = df["addr_loc_rec_type"].fillna("").str.upper()
    zip_digits = df["zip_cd"].fillna("").str.replace(r"[^0-9]", "", regex=True)
    zip5 = zip_digits.where(zip_digits.str.len() >= 5).str[:5]
    zip3 = zip_digits.where(zip_digits.str.len() >= 3).str[:3]

    has_geo = (lat.notna() & lon.notna()
               & ~((lat == 0) & (lon == 0))
               & lat.between(-90, 90) & lon.between(-180, 180)).to_numpy(bool)
    # customers.csv carries no address lines, so addr_loc_rec_type is the
    # only placeholder signal available. (Name-based PO-box matching would
    # flag businesses literally named 'PO Box ...'.)
    is_ph_geo = rec.isin(PLACEHOLDER_REC_TYPES).to_numpy(bool)
    geo_status = np.select([~has_geo, is_ph_geo],
                           ["missing", "placeholder"], default="valid")

    attrs = pd.DataFrame({
        "node": df["node"].astype("string"),
        "cust_name": df["customer_name"],
        "naics_cd": df["naics_cd"],
        "naics_desc": df["naics_desc"],
        "addr_loc_rec_type": df["addr_loc_rec_type"],
        "lat": lat.astype("float32"),
        "lon": lon.astype("float32"),
        "zip5": zip5.astype("string"),
        "zip3": zip3.astype("string"),
        "state": df["state_or_province"],
        "city": df["city"],
        "geo_status": pd.Series(geo_status, index=df.index).astype("string"),
        "shared_structure": rec.isin(SHARED_STRUCTURE_REC_TYPES)
                               .astype("int8").to_numpy(),
    }).reset_index(drop=True)

    # naics hierarchy, derived once here rather than per snapshot
    nc = naics_clean.reset_index(drop=True)
    for k in range(2, 7):
        attrs[f"naics{k}"] = nc.str[:k].where(nc.str.len() >= k).astype("string")
    attrs["naics_known"] = attrs["naics2"].notna().astype("float32")

    typing = pd.DataFrame({
        "node": df["node"].astype("string"),
        "entity_type": entity_type,
        "naics_status": naics_status,
        "node_type": node_type,
        "naics_clean": nc.to_numpy(),
    }).reset_index(drop=True)

    coords = attrs.loc[attrs["geo_status"] == "valid",
                       ["node", "lat", "lon", "zip3", "zip5", "state"]] \
                  .reset_index(drop=True)

    log.info("customers: %d rows -> %d unique nodes (%s)",
             n_raw, len(attrs), ATTR_VERSION)
    log.info("customers: naics_status=%s",
             pd.Series(naics_status).value_counts().to_dict())
    log.info("customers: node_type=%s",
             pd.Series(node_type).value_counts().to_dict())
    log.info("customers: geo_status=%s",
             pd.Series(geo_status).value_counts().to_dict())
    if (naics_status == "placeholder").sum():
        log.warning("customers: %d placeholder NAICS values survived — "
                    "expected 0 now that absence is empty; check "
                    "PLACEHOLDER_NAICS",
                    int((naics_status == "placeholder").sum()))

    return Customers(attrs=attrs, typing=typing, coords=coords, n_rows=len(attrs))


def naics_partition(attrs: pd.DataFrame) -> pd.DataFrame:
    """NAICS2 as a pseudo-partition, for the naics_participation metric.
    Unknown sector -> its own bucket so those nodes are not collapsed."""
    n2 = attrs["naics2"].fillna("NA")
    return pd.DataFrame({"node": attrs["node"].to_numpy(),
                         "community_id": n2.astype(str).to_numpy()})
