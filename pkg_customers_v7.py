"""
pkg_customers.py  (v7)
======================
Customer dimension loader. Source is now **parquet**.

v6 -> v7
--------
1. **Parquet is the source of truth.** The schema travels with the file, so
   the loader no longer has to defend against text ambiguity: a string is a
   string, a double is a double, and NULL is distinct from empty string.
   Only the columns the pipeline needs are read off disk.
2. **Parquet moves the failure modes, it does not remove them.** Two are
   real and both are handled explicitly:
     - `mdm_id` stored as BIGINT/DOUBLE. A double cannot hold an 18-digit id
       exactly (>2^53) and stringifies as '1.00030190886e+11' or '...0'.
       Loud warning + integral check + Int64 route.
     - `zip_cd` stored as an integer. **Leading zeros are gone** — 02108
       becomes 2108, which silently relocates every New England customer to
       a 4-digit ZIP that keys nothing. Detected, zero-padded, warned.
   Neither can happen from a CSV read with `dtype=str`; both are new here.
3. **`household_id`, `customer_start_dt`, `date_of_birth_inc` are not read.**
   They are not part of the customer extract.

Retained from v6
----------------
- `party_type` is the observed entity type; the name typer is retained as a
  separate, permanently-preserved column (§ entity typing below).
- NAICS field quality (`naics_status`) and NAICS applicability
  (`naics_applicable`) are orthogonal axes, never folded together.
- The extract is an OUTER JOIN of the address and customer tables:
  identity coalesced across duplicate rows, geography taken whole from the
  best-ranked row, `attr_profile` recording which side each node came from.
- **Current-state, not point-in-time.** One customer row applies to every
  historical snapshot. Re-running after a refresh CHANGES historical
  metrics — deliberate; bump ATTR_VERSION when it happens.
- **IDs are strings everywhere.**

Public API
----------
    cust = load_customers("../data/customers.parquet")
    cust.attrs        # node -> identity + NAICS + geo passthrough
    cust.typing       # node -> entity typing (observed/inferred/resolved)
    cust.coords       # node -> lat, lon, zip3, zip5, state (valid only)
    cust.disagreement # observed x inferred confusion (QA artefact)
    cust.join_profile # outer-join shape, fanout, identity conflicts
    cust.coverage(nodes)
    pkg_customers.write_qa(cust, out_dir)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_customers")

ATTR_VERSION = "cust_v3_parquet"

# ---------------------------------------------------------------------------
# Typing precedence switch
# ---------------------------------------------------------------------------
# True  : party_type decides business vs individual; a valid NAICS on a P row
#         does NOT promote it to business (flagged `individual_naics_valid`
#         in entity_class instead).
# False : v5 behaviour — a valid NAICS outranks everything.
#
# Default True: party_type is a declared legal attribute, NAICS is an
# enrichment artefact. Flip only with the disagreement table in hand — this
# switch moves `node_type`, which moves every share_* composition column and
# the headline individual-share figure.
PARTY_TYPE_WINS = True

# party_type domain. Confirmed P / O / null on the source table (2026-08).
_PARTY_MAP = {"P": "individual", "O": "business"}

# NAICS sentinels. EXPECTED to be non-zero: '******' / 'UNKNOWN' is present
# in the source table, it is not only a Cypher-injected artefact.
# NOTE: if naics_cd arrives as an INTEGER column, '******' cannot survive the
# type and will already have become NULL upstream — the placeholder count
# then drops to ~0 not because quality improved but because the sentinel was
# destroyed by the cast. _dtype_report() flags a numeric naics_cd for this
# reason.
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

# canonical -> accepted aliases in the source file (matched case-insensitively)
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
}

# WHO the customer is — coalesced across duplicate rows.
_IDENTITY_FIELDS = ("customer_name", "party_type", "naics_cd", "naics_desc")
# WHERE they are — taken as an indivisible block from one row. Mixing a
# latitude from one address with a ZIP from another invents a location.
_GEO_FIELDS = ("addr_loc_rec_type", "longitude_degree", "latitude_degree",
               "zip_cd", "state_or_province", "city")

# Columns that MUST be text. A numeric dtype here means information was
# destroyed by the writer, not by us.
_MUST_BE_TEXT = ("mdm_id", "zip_cd", "naics_cd", "party_type")

_NUM_RE = r"^\s*-?\d+(\.\d+)?\s*$"
_MAX_EXACT_INT = 2 ** 53


@dataclass
class Customers:
    attrs: pd.DataFrame          # node + all passthrough columns
    typing: pd.DataFrame         # node + entity typing + naics status
    coords: pd.DataFrame         # node, lat, lon, zip3, zip5, state (valid)
    disagreement: pd.DataFrame = field(default_factory=pd.DataFrame)
    join_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    dtype_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0

    def coverage(self, nodes) -> dict:
        """Attribute coverage over an arbitrary node universe.

        `pct_naics_valid_of_business` is the honest enrichment number: NAICS
        is not expected of a person, so persons do not belong in the
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
# dtype-aware readers
# ---------------------------------------------------------------------------

def _read_source(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the mapped columns. Returns (raw frame, dtype report).

    Parquet carries its schema, so column selection happens before the read
    rather than after — on a 30M-row extract that is the difference between
    loading eleven columns and loading whatever the extract happens to have.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq", ".parq"):
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(path).schema_arrow
        available = list(schema.names)
        lower = {c.lower(): c for c in available}
        wanted, resolved = [], {}
        for canon, aliases in _COLMAP.items():
            for a in aliases:
                if a in lower:
                    wanted.append(lower[a])
                    resolved[canon] = lower[a]
                    break
        df = pd.read_parquet(path, columns=wanted)
        rep = pd.DataFrame([
            {"canonical": c, "source_column": resolved.get(c),
             "arrow_type": (str(schema.field(resolved[c]).type)
                            if c in resolved else None),
             "present": c in resolved}
            for c in _COLMAP])
        log.info("customers: parquet read — %d of %d mapped columns present",
                 len(wanted), len(_COLMAP))
    elif ext == ".csv":
        log.warning("customers: reading CSV (%s). Parquet is the supported "
                    "source from v7; CSV loses NULL-vs-empty and forces "
                    "text-parsing defences.", path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         na_values=[""])
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        rep = pd.DataFrame([{"canonical": c, "source_column": None,
                             "arrow_type": "string(csv)", "present": None}
                            for c in _COLMAP])
    else:
        raise ValueError(f"unsupported customers file extension: {ext!r} "
                         f"(expected .parquet or .csv)")
    return df, rep


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map source columns (case-insensitively) to canonical names."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    out = {}
    for canon, aliases in _COLMAP.items():
        for a in aliases:
            if a in lower:
                out[canon] = df[lower[a]]
                break
        else:
            out[canon] = pd.Series(pd.NA, index=df.index, dtype="string")
            log.warning("customers: column '%s' not found — filled empty",
                        canon)
    return pd.DataFrame(out)


def _dtype_report(df: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    """Flag columns whose parquet type destroys information.

    These cannot happen on a CSV read with dtype=str; they are specific to a
    typed source and are silent unless looked for.
    """
    notes = []
    for c in _MUST_BE_TEXT:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            if c == "zip_cd":
                msg = ("ZIP stored numerically — LEADING ZEROS LOST; "
                       "zero-padding to 5, but 0-prefixed ZIPs cannot be "
                       "distinguished from genuinely short values")
            elif c == "mdm_id":
                msg = ("id stored numerically — cast to string; a float id "
                       "above 2^53 is not exactly representable")
            elif c == "naics_cd":
                msg = ("NAICS stored numerically — the '******' sentinel "
                       "cannot survive this type and became NULL upstream; "
                       "placeholder counts are NOT comparable to v6")
            else:
                msg = "expected text, stored numerically"
            notes.append({"column": c, "dtype": str(s.dtype), "issue": msg})
    out = pd.DataFrame(notes)
    for _, r in out.iterrows():
        log.warning("customers: DTYPE — %s is %s: %s",
                    r["column"], r["dtype"], r["issue"])
    if not rep.empty:
        out = pd.concat([rep.assign(issue=None), out], ignore_index=True)
    return out


def _id_to_str(s: pd.Series, name: str = "mdm_id") -> pd.Series:
    """Node ids to string without float contamination.

    A BIGINT id read as int64 stringifies fine. A DOUBLE id does not: it
    renders as '100030190886.0', and above 2^53 it is not exactly the value
    that was written. Both routes go through Int64.
    """
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        non_integral = int((v != v.round()).sum())
        too_big = int((v.abs() >= _MAX_EXACT_INT).sum())
        log.warning("customers: %s is float64 — converting via Int64. "
                    "%d non-integral values, %d beyond 2^53 (NOT exactly "
                    "representable; fix the extract to CAST AS STRING)",
                    name, non_integral, too_big)
        return s.round().astype("Int64").astype("string")
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64").astype("string")
    return s.astype("string").str.strip()


def _text(s: pd.Series) -> pd.Series:
    """Any dtype -> nullable string, stripped, empty normalised to NA.

    Parquet preserves the NULL / '' distinction that CSV collapses. Every
    downstream test is 'is this absent', so the two are normalised to one
    representation here, deliberately and in one place, rather than being
    handled inconsistently by whichever `.fillna("")` runs first.
    """
    if pd.api.types.is_float_dtype(s):
        out = s.round().astype("Int64").astype("string")
    elif pd.api.types.is_integer_dtype(s):
        out = s.astype("Int64").astype("string")
    else:
        out = s.astype("string")
    out = out.str.strip()
    return out.mask(out == "")


def _num(s: pd.Series) -> pd.Series:
    """Coordinates to float, whatever the source dtype.

    Already numeric -> straight through. Text -> screened with a numeric
    regex first, because empty strings, whitespace and 'null' tokens all
    survive a plain isna() check and would cast to NaN silently mixed with
    genuine nulls.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    t = s.astype("string").str.strip()
    ok = t.str.match(_NUM_RE).fillna(False)
    return pd.to_numeric(t.where(ok), errors="coerce")


def _zip_digits(s: pd.Series) -> pd.Series:
    """ZIP to a digit string, restoring leading zeros where the source type
    destroyed them."""
    if pd.api.types.is_numeric_dtype(s):
        d = s.round().astype("Int64").astype("string")
        short = int((d.str.len() < 5).sum())
        d = d.str.zfill(5)
        if short:
            log.warning("customers: %d numeric ZIPs shorter than 5 digits "
                        "zero-padded (02108 arrives as 2108)", short)
        return d.fillna("")
    return s.astype("string").fillna("").str.replace(r"[^0-9]", "",
                                                     regex=True)


def _tokenize(names: pd.Series) -> pd.Series:
    up = names.fillna("").astype(str).str.upper()
    punct = re.compile(r"[^A-Z0-9& ]+")
    return up.map(lambda s: punct.sub(" ", s).split())


def _geo_rank(df: pd.DataFrame) -> np.ndarray:
    """Rank candidate address rows for one customer. Higher wins.

        3  usable coordinate, own-premises record type
        2  usable coordinate, PO box / general delivery
        1  no coordinate but a ZIP that could still key a CBSA
        0  nothing locational
    """
    lat = _num(df["latitude_degree"])
    lon = _num(df["longitude_degree"])
    has_xy = (lat.notna() & lon.notna()
              & ~((lat == 0) & (lon == 0))
              & lat.between(-90, 90) & lon.between(-180, 180)).to_numpy(bool)
    rec = df["addr_loc_rec_type"].astype("string").fillna("").str.upper()
    is_ph = rec.isin(PLACEHOLDER_REC_TYPES).to_numpy(bool)
    zipd = (_zip_digits(df["zip_cd"]).str.len() >= 5).to_numpy(bool)
    return np.select([has_xy & ~is_ph, has_xy, zipd], [3, 2, 1], default=0)


def _resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse an outer-join fanout to one row per node.

    Identity is coalesced across the group; geography is taken whole from
    the single best-ranked row.
    """
    n_rows_in = len(df)
    dup_mask = df["node"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    stats = {"rows_in": n_rows_in, "dup_rows": n_dup_rows,
             "conflict_party_type": 0, "conflict_naics_cd": 0,
             "conflict_customer_name": 0}
    if n_dup_rows == 0:
        return df.reset_index(drop=True), stats

    d = df.loc[dup_mask]
    for c in ("party_type", "naics_cd", "customer_name"):
        nun = d.groupby("node")[c].nunique(dropna=True)
        stats[f"conflict_{c}"] = int((nun > 1).sum())

    ident = (df.groupby("node", sort=False)[list(_IDENTITY_FIELDS)]
             .first().reset_index())
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
    """Read customers.parquet and derive typing + geo status. One pass."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"customers file not found: {path}")
    ptw = PARTY_TYPE_WINS if party_type_wins is None else bool(party_type_wins)

    raw, schema_rep = _read_source(path)
    df = _resolve_columns(raw)
    n_raw = len(df)
    dtypes = _dtype_report(df, schema_rep)

    df["node"] = _id_to_str(df["mdm_id"])
    n_no_id = int(df["node"].isna().sum() + (df["node"] == "").sum())
    if n_no_id:
        # address-side rows that matched no customer: no mdm_id, so they can
        # never join the graph. Dropped, but counted.
        log.warning("customers: %d rows dropped — no mdm_id (address-side "
                    "rows of the outer join with no customer match)", n_no_id)
    df = df[df["node"].notna() & (df["node"] != "")]

    for c in ("customer_name", "party_type", "naics_cd", "naics_desc",
              "addr_loc_rec_type", "state_or_province", "city"):
        df[c] = _text(df[c])

    df, join_stats = _resolve_duplicates(df)

    # ---- entity type: OBSERVED (party_type) -------------------------------
    pt = df["party_type"].fillna("").str.upper().str[:1]
    unexpected = df.loc[~pt.isin(list(_PARTY_MAP) + [""]), "party_type"]
    if len(unexpected):
        log.warning("customers: %d rows carry party_type outside {P,O,null}: "
                    "%s — typed unknown", len(unexpected),
                    unexpected.value_counts().head(5).to_dict())
    entity_type_observed = pt.map(_PARTY_MAP).fillna("unknown").to_numpy(object)

    # ---- entity type: INFERRED (name tokens, retained) --------------------
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
    # Precedence: an OBSERVED valid code reports valid even on a person
    # (sole proprietor) — never mask an observation as inapplicable.
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
    else:  # legacy precedence, for the A/B comparison
        node_type = np.select(
            [is_valid,
             is_biz & (naics_status == "placeholder"),
             is_biz & (naics_status == "missing"),
             is_ind],
            ["business_naics_valid", "business_naics_placeholder",
             "business_naics_missing", "individual"],
            default="unknown")

    # what the pre-party_type rule WOULD have produced on the same rows —
    # the number that tells you whether the 94.3%-individual headline moves.
    node_type_v5 = np.select(
        [is_valid,
         (entity_type_inferred == "business") & (naics_status == "placeholder"),
         (entity_type_inferred == "business") & (naics_status == "missing"),
         entity_type_inferred == "individual"],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual"],
        default="unknown")

    # ---- entity_class: fine-grained label (NOT a composition key) ---------
    # Keeps sole-proprietor candidates visible without adding categories to
    # node_type, which would silently break the share_* sum-to-one property.
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
    zip_digits = _zip_digits(df["zip_cd"])
    zip5 = zip_digits.where(zip_digits.str.len() >= 5).str[:5]
    zip3 = zip_digits.where(zip_digits.str.len() >= 3).str[:3]

    has_geo = (lat.notna() & lon.notna()
               & ~((lat == 0) & (lon == 0))
               & lat.between(-90, 90) & lon.between(-180, 180)).to_numpy(bool)
    # the extract carries no address lines, so addr_loc_rec_type is the only
    # placeholder signal available. (Name-based PO-box matching would flag
    # businesses literally named 'PO Box ...'.)
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

    _log_load(attrs, typing, dis, ptw, n_raw, path)
    return Customers(attrs=attrs, typing=typing, coords=coords,
                     disagreement=dis, join_profile=join_profile,
                     dtype_report=dtypes, n_rows=len(attrs))


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def typing_disagreement(typing: pd.DataFrame) -> pd.DataFrame:
    """Long-form confusion between the declared and inferred entity type,
    plus the node_type churn against the pre-party_type rule.

    Read this before trusting any composition share. The cell that matters
    most is (observed=business, inferred=individual): person-named
    organisations — trusts, estates, sole proprietorships, single-member
    LLCs. The name typer counted them as households.
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


def _log_load(attrs, typing, dis, ptw, n_raw, path):
    log.info("customers: %s — %d source rows -> %d unique nodes (%s, "
             "PARTY_TYPE_WINS=%s)", os.path.basename(path), n_raw,
             len(attrs), ATTR_VERSION, ptw)
    log.info("customers: attr_profile=%s",
             attrs["attr_profile"].value_counts().to_dict())
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
                 "non-zero when naics_cd is a text column", ph, 100 * ph / n)

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
                 "%.2f%% of nodes", conflict, biz_person, churn)
        if churn > 5:
            log.warning("node_type churn %.2f%% — every share_* composition "
                        "column and the headline individual-share figure "
                        "move by this much. Re-run the geographic population "
                        "split before citing the earlier numbers.", churn)


def write_qa(cust: Customers, out_dir: str) -> None:
    """Persist the QA artefacts next to the metrics outputs."""
    qa = os.path.join(out_dir, "qa")
    os.makedirs(qa, exist_ok=True)
    cust.disagreement.to_csv(
        os.path.join(qa, "customers_typing_disagreement.csv"), index=False)
    cust.join_profile.to_csv(
        os.path.join(qa, "customers_join_profile.csv"), index=False)
    cust.dtype_report.to_csv(
        os.path.join(qa, "customers_source_schema.csv"), index=False)
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
