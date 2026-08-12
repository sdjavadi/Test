"""
pkg_customers.py  (v6)
======================
Customer dimension loader.

v5 -> v6 — three changes, all driven by what the source table actually is.

1. **`party_type` is now the entity-type source of truth.** The MDM party
   model carries P (person) / O (organisation) / null. That is a declared
   attribute; the v5 token typer on `customer_name` was an inference. Both
   are kept, in separate columns, forever:

       entity_type_observed   <- party_type          (declared)
       entity_type_inferred   <- customer_name       (heuristic)
       entity_type            <- resolved            (observed wins)
       entity_type_source     <- party_type|name|none

   Same discipline as naics_observed / naics_imputed: an inference never
   overwrites an observation in place, because you cannot audit what you
   cannot see.

2. **NAICS applicability is separated from NAICS quality.** `naics_status`
   still describes the *field value* (valid|placeholder|missing) and nothing
   else. Whether a NAICS was ever *expected* is a different axis, now
   carried in `naics_applicable` (1 for organisations). Empty NAICS on a
   P row is not a data gap — the field does not apply. Folding the two into
   one enum is the exact error the three-way taxonomy was created to avoid,
   so they stay orthogonal. Enrichment coverage = valid / applicable.

3. **The source is an OUTER JOIN of the address table and the customer
   table**, so a row can carry geography without identity, identity without
   geography, and one mdm_id can appear on several rows (one per address).
   v5's `drop_duplicates(keep='first')` was arbitrary under that shape and
   could discard the row holding the only valid coordinate. v6 resolves
   duplicates deterministically:
       - identity fields are coalesced across the group (first non-null),
       - the geo BLOCK is taken whole from the best-ranked single row,
         never mixed field-by-field across rows,
       - conflicting identity values are counted and logged.
   `attr_profile` records which side of the join each node came from.

Deliberately NOT ingested
-------------------------
`date_of_birth_inc` is present in the source. It is not read. Age is a
prohibited basis under ECOA/Reg B, and putting it in the dimension that
feeds every node metric creates a fair-lending surface for no analytical
need we currently have. If a use case ever requires it, ingest it in a
separate, access-controlled table with a documented purpose — not here.

Design notes carried forward from v5
------------------------------------
- **Current-state, not point-in-time.** One customer row is applied to every
  historical snapshot. Re-running after a customer refresh CHANGES historical
  metrics. That is deliberate; bump ATTR_VERSION when it happens.
- **NAICS code and description arrive as separate columns.** No '|' split.
- **IDs are strings everywhere.** mdm_id is read as str and never coerced.

Public API
----------
    cust = load_customers("../data/customers.csv")
    cust.attrs        # node -> identity + NAICS + geo passthrough
    cust.typing       # node -> entity typing (observed/inferred/resolved)
    cust.coords       # node -> lat, lon, zip3, state  (geo_status valid only)
    cust.disagreement # observed x inferred confusion matrix (QA artefact)
    cust.coverage(nodes)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_customers")

ATTR_VERSION = "cust_v2_partytype"

# ---------------------------------------------------------------------------
# Typing precedence switch
# ---------------------------------------------------------------------------
# True  : party_type decides business vs individual; a valid NAICS on a P row
#         does NOT promote it to business (it is flagged as a sole-proprietor
#         candidate in `entity_class` instead).
# False : v5 behaviour — a valid NAICS outranks everything.
#
# Default True: party_type is a declared legal attribute and NAICS is an
# enrichment artefact. Flip only with the disagreement table in hand — this
# switch moves `node_type`, which moves every share_* composition column and
# the headline individual-share figure.
PARTY_TYPE_WINS = True

# party_type domain. Confirmed P / O / null on the source table (2026-08).
# Anything else is typed unknown and logged loudly rather than guessed at.
_PARTY_MAP = {"P": "individual", "O": "business"}

# NAICS sentinels. Unlike v5, these are EXPECTED to be non-zero: '******'
# with description 'UNKNOWN' is present in the source table, it is not only
# a Cypher-injected artefact. Counted, not warned on, unless the share is
# large enough to distort the taxonomy.
PLACEHOLDER_NAICS = {"-1", "0", "00", "000000", "999999", "******",
                     "UNKNOWN", "N/A", "NA", "NULL", "NONE"}
PLACEHOLDER_WARN_SHARE = 0.05

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
    "party_type":        ("party_type", "prty_type", "party_typ",
                          "party_type_cd"),
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
    # optional passthroughs — absent columns are filled empty with a warning
    "household_id":      ("household_id", "hh_id"),
    "customer_start_dt": ("customer_start_dt", "cust_start_dt"),
}

# Columns that describe WHO the customer is. Coalesced across duplicate rows.
_IDENTITY_FIELDS = ("customer_name", "party_type", "naics_cd", "naics_desc",
                    "household_id", "customer_start_dt")
# Columns that describe WHERE they are. Taken as an indivisible block from
# one row — mixing lat from one address with zip from another invents a
# location that does not exist.
_GEO_FIELDS = ("addr_loc_rec_type", "longitude_degree", "latitude_degree",
               "zip_cd", "state_or_province", "city")

_NUM_RE = r"^\s*-?\d+(\.\d+)?\s*$"


@dataclass
class Customers:
    attrs: pd.DataFrame          # node + all passthrough columns
    typing: pd.DataFrame         # node + entity typing + naics status
    coords: pd.DataFrame         # node, lat, lon, zip3, zip5, state (valid)
    disagreement: pd.DataFrame = field(default_factory=pd.DataFrame)
    join_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0

    def coverage(self, nodes) -> dict:
        """Attribute coverage over an arbitrary node universe.

        `pct_naics_valid_of_business` is the honest enrichment number: NAICS
        is not expected on a person, so persons do not belong in the
        denominator.
        """
        nodes = pd.Index(pd.unique(pd.Series(nodes, dtype="object").astype(str)))
        a = self.attrs.set_index("node")
        c = self.coords.set_index("node")
        t = self.typing.set_index("node")
        n = max(len(nodes), 1)
        tn = t.reindex(nodes)
        biz = (tn["entity_type"] == "business")
        n_biz = int(biz.sum())
        return {
            "n_nodes": len(nodes),
            "pct_in_customers": round(100 * nodes.isin(a.index).sum() / n, 2),
            "pct_party_type_observed": round(
                100 * (tn["entity_type_source"] == "party_type").sum() / n, 2),
            "pct_business": round(100 * n_biz / n, 2),
            "pct_individual": round(
                100 * int((tn["entity_type"] == "individual").sum()) / n, 2),
            "pct_naics_valid": round(
                100 * int((tn["naics_status"] == "valid").sum()) / n, 2),
            "pct_naics_valid_of_business": round(
                100 * int(((tn["naics_status"] == "valid") & biz).sum())
                / max(n_biz, 1), 2),
            "pct_geo_valid": round(100 * nodes.isin(c.index).sum() / n, 2),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _geo_rank(df: pd.DataFrame) -> np.ndarray:
    """Rank candidate address rows for one customer. Higher wins.

        3  usable coordinate, own-premises record type
        2  usable coordinate, PO box / general delivery
        1  no coordinate but a zip that could still key a CBSA
        0  nothing locational
    """
    lat = _num(df["latitude_degree"])
    lon = _num(df["longitude_degree"])
    has_xy = (lat.notna() & lon.notna()
              & ~((lat == 0) & (lon == 0))
              & lat.between(-90, 90) & lon.between(-180, 180)).to_numpy(bool)
    rec = df["addr_loc_rec_type"].fillna("").astype(str).str.upper()
    is_ph = rec.isin(PLACEHOLDER_REC_TYPES).to_numpy(bool)
    zipd = (df["zip_cd"].fillna("").astype(str)
            .str.replace(r"[^0-9]", "", regex=True).str.len() >= 5).to_numpy(bool)
    return np.select([has_xy & ~is_ph, has_xy, zipd], [3, 2, 1], default=0)


def _resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse an outer-join fanout to one row per node.

    Identity is coalesced across the group; geography is taken whole from
    the single best-ranked row. Returns the collapsed frame and a stats dict.
    """
    n_rows_in = len(df)
    dup_mask = df["node"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    stats = {"rows_in": n_rows_in, "dup_rows": n_dup_rows,
             "conflict_party_type": 0, "conflict_naics_cd": 0,
             "conflict_customer_name": 0}
    if n_dup_rows == 0:
        return df.reset_index(drop=True), stats

    # conflicting declared values within one mdm_id — should be ~0; a
    # non-zero count means the join key is not what we think it is.
    d = df.loc[dup_mask]
    for c in ("party_type", "naics_cd", "customer_name"):
        nun = d.groupby("node")[c].nunique(dropna=True)
        stats[f"conflict_{c}"] = int((nun > 1).sum())

    # identity: first non-null per node (groupby.first skips NA)
    ident = (df.groupby("node", sort=False)[list(_IDENTITY_FIELDS)]
             .first().reset_index())
    # geography: one whole row, chosen by rank then by original order
    g = df[["node", *_GEO_FIELDS]].copy()
    g["_rank"] = _geo_rank(df)
    g["_ord"] = np.arange(len(g))
    g = (g.sort_values(["node", "_rank", "_ord"],
                       ascending=[True, False, True])
         .drop_duplicates("node", keep="first")
         .drop(columns=["_rank", "_ord"]))
    out = ident.merge(g, on="node", how="left")
    log.warning("customers: outer-join fanout — %d rows collapsed to %d "
                "nodes (%d duplicate rows); identity coalesced, geo taken "
                "from best-ranked address row",
                n_rows_in, len(out), n_dup_rows)
    for c in ("party_type", "naics_cd", "customer_name"):
        if stats[f"conflict_{c}"]:
            log.warning("customers: %d nodes carry CONFLICTING %s across "
                        "duplicate rows — first non-null kept; investigate "
                        "the join key", stats[f"conflict_{c}"], c)
    return out.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------------

def load_customers(path: str, party_type_wins: bool | None = None
                   ) -> Customers:
    """Read customers.csv and derive typing + geo status. One pass."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"customers file not found: {path}")
    ptw = PARTY_TYPE_WINS if party_type_wins is None else bool(party_type_wins)

    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df = _resolve_columns(raw)
    n_raw = len(df)

    df["node"] = df["mdm_id"].astype("string").str.strip()
    n_no_id = int((df["node"].isna() | (df["node"] == "")).sum())
    if n_no_id:
        # address-side rows that matched no customer at all: no mdm_id, so
        # they can never join the graph. Dropped, but counted.
        log.warning("customers: %d rows dropped — no mdm_id (address-side "
                    "rows of the outer join with no customer match)", n_no_id)
    df = df[df["node"].notna() & (df["node"] != "")]

    for c in ("customer_name", "party_type", "naics_cd", "naics_desc",
              "addr_loc_rec_type", "zip_cd", "state_or_province", "city",
              "household_id", "customer_start_dt"):
        df[c] = df[c].astype("string").str.strip()

    df, join_stats = _resolve_duplicates(df)

    # ---- entity type: OBSERVED (party_type) -------------------------------
    pt = df["party_type"].fillna("").str.upper().str[:1]
    unexpected = df.loc[~pt.isin(list(_PARTY_MAP) + [""]), "party_type"]
    if len(unexpected):
        log.warning("customers: %d rows carry party_type outside {P,O,null}: "
                    "%s — typed unknown", len(unexpected),
                    unexpected.value_counts().head(5).to_dict())
    entity_type_observed = pt.map(_PARTY_MAP).fillna("unknown").to_numpy(object)

    # ---- entity type: INFERRED (name tokens, v5 logic, retained) ----------
    toks = _tokenize(df["customer_name"])
    has_biz = toks.map(lambda t: any(w in _BUSINESS_TOKENS for w in t)) \
                  .to_numpy(dtype=bool)
    n_alpha = toks.map(lambda t: sum(w.isalpha() for w in t)).to_numpy(int)
    n_tok = toks.map(len).to_numpy(int)
    person_shaped = (~has_biz) & (n_alpha == n_tok) & (n_tok >= 2) & (n_tok <= 3)
    entity_type_inferred = np.select([has_biz, person_shaped],
                                     ["business", "individual"],
                                     default="unknown")

    # ---- resolve: observed wins, inference is the fallback only -----------
    obs_known = entity_type_observed != "unknown"
    inf_known = entity_type_inferred != "unknown"
    entity_type = np.where(obs_known, entity_type_observed,
                           entity_type_inferred)
    entity_type_source = np.select([obs_known, inf_known],
                                   ["party_type", "name"], default="none")

    # ---- NAICS: field quality only (applicability is a separate axis) -----
    code = df["naics_cd"].fillna("")
    is_missing = (code == "").to_numpy(dtype=bool)
    is_ph = code.str.upper().isin(PLACEHOLDER_NAICS).fillna(False).to_numpy(bool)
    looks_code = code.str.match(r"^\d{2}").fillna(False).to_numpy(bool)
    naics_status = np.select(
        [is_missing, is_ph, looks_code],
        ["missing", "placeholder", "valid"], default="placeholder")
    naics_clean = code.str.extract(r"^(\d{2,6})")[0].where(
        pd.Series(naics_status == "valid", index=df.index))

    # applicability: NAICS is expected of an organisation, not of a person.
    # Empty NAICS on a P row is NOT a data-quality gap.
    naics_applicable = (entity_type == "business").astype("int8")
    # Precedence: an OBSERVED valid code is reported as valid even on a
    # person (sole proprietor) — never mask an observation as inapplicable.
    # 'not_applicable' means "no code, and none was expected".
    naics_coverage_class = np.select(
        [naics_status == "valid", naics_applicable == 0,
         naics_status == "placeholder"],
        ["valid", "not_applicable", "placeholder"], default="missing")

    # ---- node_type: the 5-category composition key (schema-stable) --------
    is_valid = naics_status == "valid"
    is_biz = entity_type == "business"
    is_ind = entity_type == "individual"
    if ptw:
        node_type = np.select(
            [is_biz & is_valid,
             is_biz & (naics_status == "placeholder"),
             is_biz & (naics_status == "missing"),
             is_ind],
            ["business_naics_valid", "business_naics_placeholder",
             "business_naics_missing", "individual"],
            default="unknown")
    else:  # v5 precedence, for the A/B comparison
        node_type = np.select(
            [is_valid,
             is_biz & (naics_status == "placeholder"),
             is_biz & (naics_status == "missing"),
             is_ind],
            ["business_naics_valid", "business_naics_placeholder",
             "business_naics_missing", "individual"],
            default="unknown")

    # what v5 WOULD have produced, on the same rows — this is the number that
    # tells you whether the 94.3%-individual headline moves.
    node_type_v5 = np.select(
        [is_valid,
         (entity_type_inferred == "business") & (naics_status == "placeholder"),
         (entity_type_inferred == "business") & (naics_status == "missing"),
         entity_type_inferred == "individual"],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual"],
        default="unknown")

    # ---- entity_class: the fine-grained label (NOT a composition key) -----
    # Exists so sole-proprietor candidates and NAICS-on-person cases stay
    # visible without adding categories to node_type (which would silently
    # break the share_* columns' sum-to-one property).
    entity_class = np.select(
        [is_biz & is_valid,
         is_biz & (naics_status == "placeholder"),
         is_biz & (naics_status == "missing"),
         is_ind & is_valid,
         is_ind,
         is_valid],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual_naics_valid", "individual",
         "unknown_naics_valid"],
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

    # ---- which side of the outer join did this node come from? ------------
    has_identity = (df["customer_name"].notna()
                    | df["party_type"].notna()
                    | df["naics_cd"].notna()).to_numpy(bool)
    attr_profile = np.select(
        [has_identity & has_geo, has_identity, has_geo],
        ["identity+geo", "identity_only", "geo_only"], default="neither")

    attrs = pd.DataFrame({
        "node": df["node"].astype("string"),
        "cust_name": df["customer_name"],
        "party_type": df["party_type"],
        "naics_cd": df["naics_cd"],
        "naics_desc": df["naics_desc"],
        "household_id": df["household_id"],
        "customer_start_dt": df["customer_start_dt"],
        "addr_loc_rec_type": df["addr_loc_rec_type"],
        "lat": lat.astype("float32"),
        "lon": lon.astype("float32"),
        "zip5": zip5.astype("string"),
        "zip3": zip3.astype("string"),
        "state": df["state_or_province"],
        "city": df["city"],
        "geo_status": pd.Series(geo_status, index=df.index).astype("string"),
        "attr_profile": pd.Series(attr_profile, index=df.index).astype("string"),
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
        "entity_type_observed": entity_type_observed,
        "entity_type_inferred": entity_type_inferred,
        "entity_type_source": entity_type_source,
        "entity_class": entity_class,
        "naics_status": naics_status,
        "naics_applicable": naics_applicable,
        "naics_coverage_class": naics_coverage_class,
        "node_type": node_type,
        "node_type_v5": node_type_v5,
        "naics_clean": nc.to_numpy(),
    }).reset_index(drop=True)

    coords = attrs.loc[attrs["geo_status"] == "valid",
                       ["node", "lat", "lon", "zip3", "zip5", "state"]] \
                  .reset_index(drop=True)

    dis = typing_disagreement(typing)
    join_profile = pd.DataFrame([{
        **join_stats,
        "nodes_out": len(attrs),
        "rows_no_mdm_id": n_no_id,
        **attrs["attr_profile"].value_counts().to_dict(),
    }])

    _log_load(attrs, typing, dis, ptw, n_raw)
    return Customers(attrs=attrs, typing=typing, coords=coords,
                     disagreement=dis, join_profile=join_profile,
                     n_rows=len(attrs))


# ---------------------------------------------------------------------------
# QA: observed vs inferred
# ---------------------------------------------------------------------------

def typing_disagreement(typing: pd.DataFrame) -> pd.DataFrame:
    """Long-form confusion between the declared and inferred entity type,
    plus the node_type churn v5 -> v6.

    Read this before trusting any composition share. The cell that matters
    most is (observed=business, inferred=individual): person-named
    organisations — trusts, estates, sole proprietorships, single-member
    LLCs. v5 typed them as households.
    """
    if typing.empty:
        return pd.DataFrame()
    x = (typing.groupby(["entity_type_observed", "entity_type_inferred"])
         .size().rename("n").reset_index())
    x["pct"] = (100 * x["n"] / x["n"].sum()).round(3)
    x["agree"] = np.where(
        x["entity_type_observed"] == "unknown", "no_observation",
        np.where(x["entity_type_inferred"] == "unknown", "no_inference",
                 np.where(x["entity_type_observed"]
                          == x["entity_type_inferred"], "agree", "CONFLICT")))
    y = (typing.groupby(["node_type_v5", "node_type"]).size()
         .rename("n").reset_index())
    y["pct"] = (100 * y["n"] / y["n"].sum()).round(3)
    y["agree"] = np.where(y["node_type_v5"] == y["node_type"],
                          "agree", "CHANGED")
    x.insert(0, "axis", "entity_type")
    y.insert(0, "axis", "node_type")
    y = y.rename(columns={"node_type_v5": "entity_type_observed",
                          "node_type": "entity_type_inferred"})
    return pd.concat([x, y], ignore_index=True)


def _log_load(attrs, typing, dis, ptw, n_raw):
    log.info("customers: %d source rows -> %d unique nodes (%s, "
             "PARTY_TYPE_WINS=%s)", n_raw, len(attrs), ATTR_VERSION, ptw)
    for col in ("attr_profile",):
        log.info("customers: %s=%s", col,
                 attrs[col].value_counts().to_dict())
    for col in ("entity_type_source", "entity_type", "entity_class",
                "naics_status", "naics_coverage_class", "node_type"):
        log.info("customers: %s=%s", col,
                 typing[col].value_counts().to_dict())
    log.info("customers: geo_status=%s",
             attrs["geo_status"].value_counts().to_dict())

    n = max(len(typing), 1)
    ph = int((typing["naics_status"] == "placeholder").sum())
    if ph / n > PLACEHOLDER_WARN_SHARE:
        log.warning("customers: %d (%.1f%%) placeholder NAICS values — above "
                    "the %.0f%% review threshold; check PLACEHOLDER_NAICS "
                    "against the source distribution",
                    ph, 100 * ph / n, 100 * PLACEHOLDER_WARN_SHARE)
    else:
        log.info("customers: %d placeholder NAICS values (%.2f%%) — expected "
                 "non-zero; '******' is present in the source table", ph,
                 100 * ph / n)

    # the two numbers worth reading out loud
    if not dis.empty:
        d = dis[dis["axis"] == "entity_type"]
        conflict = float(d.loc[d["agree"] == "CONFLICT", "pct"].sum())
        biz_person = float(d.loc[
            (d["entity_type_observed"] == "business")
            & (d["entity_type_inferred"] == "individual"), "pct"].sum())
        churn = float(dis.loc[(dis["axis"] == "node_type")
                              & (dis["agree"] == "CHANGED"), "pct"].sum())
        log.info("TYPING DISAGREEMENT: observed vs inferred conflict %.2f%% "
                 "| person-named organisations %.2f%% | node_type churn "
                 "v5->v6 %.2f%% of nodes", conflict, biz_person, churn)
        if churn > 5:
            log.warning("node_type churn %.2f%% — every share_* composition "
                        "column and the headline individual-share figure "
                        "move by this much. Re-run the geographic population "
                        "split before citing v5 numbers.", churn)


def write_qa(cust: Customers, out_dir: str) -> None:
    """Persist the QA artefacts next to the metrics outputs."""
    qa = os.path.join(out_dir, "qa")
    os.makedirs(qa, exist_ok=True)
    cust.disagreement.to_csv(
        os.path.join(qa, "customers_typing_disagreement.csv"), index=False)
    cust.join_profile.to_csv(
        os.path.join(qa, "customers_join_profile.csv"), index=False)
    (cust.typing.groupby(["entity_type", "naics_coverage_class"])
     .size().rename("n").reset_index()
     .to_csv(os.path.join(qa, "customers_naics_coverage.csv"), index=False))
    log.info("customers: QA artefacts written to %s", qa)


def naics_partition(attrs: pd.DataFrame) -> pd.DataFrame:
    """NAICS2 as a pseudo-partition, for the naics_participation metric.
    Unknown sector -> its own bucket so those nodes are not collapsed."""
    n2 = attrs["naics2"].fillna("NA")
    return pd.DataFrame({"node": attrs["node"].to_numpy(),
                         "community_id": n2.astype(str).to_numpy()})
