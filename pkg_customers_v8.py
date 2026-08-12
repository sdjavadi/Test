"""
pkg_customers.py  (v8)
======================
Customer dimension loader. Parquet source. **Memory-lean.**

v7 -> v8 — this release exists because v7 OOM-killed the batch step on the
full extract. Three causes, all in v7, all mine:

1. **`np.select` with string choices returns a fixed-width unicode array.**
   `<U26` is **104 bytes per element** — 3.1 GB per column at 30M rows.
   v7 built nine of them (entity_type, entity_type_source, entity_class,
   naics_status, naics_coverage_class, node_type, node_type_v5, geo_status,
   attr_profile) ≈ **28 GB of transient string arrays** before a single
   DataFrame was assembled. v5 built four; v6/v7 more than doubled it while
   adding the typing columns.
   → Fixed: `_select_cat()` selects **integer codes** (int8) and builds a
   Categorical from them. 104 bytes/row becomes 1 byte/row. Same values,
   same semantics, ~30 MB per column instead of 3.1 GB.

2. **The name typer materialised one Python list per row.**
   `up.map(lambda s: punct.sub(" ", s).split())` allocates 30M list objects
   (~3.6 GB in list headers alone, before the str objects inside), held
   alive across three subsequent `.map()` passes. Measured 2.85 s/M — ~85 s
   at 30M, most of it spent allocating.
   → Fixed: `_name_typer()` is fully vectorised on `.str` accessors.
   Verified **1.000 agreement** with the v7 logic on both outputs.

3. **The dimension was materialised three times.** `attrs` (30M) + `typing`
   (30M) + the pipeline's `attrs.merge(typing)` (30M) all alive at once.
   → Fixed: one frame. `typing` is a narrow projection, and the pipeline no
   longer merges.

Also: low-cardinality passthrough columns (`party_type`, `state`, `city`,
`zip3`/`zip5`, `naics_desc`, `naics2`–`naics6`) are Categorical. `cust_name`
stays a string — it is ~unique, so a dictionary would only add overhead.

Peak RSS is logged after every stage. If this OOMs again, the log says where.

Retained from v7
----------------
- Parquet source with column pushdown; dtype hazards detected and reported
  (`mdm_id` as float, `zip_cd` as int destroying leading zeros, `naics_cd`
  as int destroying the '******' sentinel).
- `party_type` is the observed entity type; the name typer is retained as a
  separate, permanently-preserved column.
- NAICS field quality and NAICS applicability are orthogonal axes.
- Outer-join resolution: identity coalesced, geography taken whole from the
  best-ranked row, `attr_profile` recording the join side.
- Current-state, not point-in-time. IDs are strings everywhere.

Public API
----------
    cust = load_customers("../data/customers.parquet")
    cust.attrs        # THE frame: identity + geo + NAICS hierarchy + typing
    cust.typing       # narrow projection of the same frame (no second copy)
    cust.coords       # node, lat, lon, zip3, zip5, state (valid only)
    cust.disagreement / cust.join_profile / cust.dtype_report
    cust.coverage(nodes)
    pkg_customers.write_qa(cust, out_dir)
    pkg_customers.node_universe(snapshot_paths)   # optional dimension filter
"""

from __future__ import annotations

import gc
import glob
import logging
import os
import resource
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger("pkg_customers")

ATTR_VERSION = "cust_v4_lean"

# ---------------------------------------------------------------------------
# Typing precedence switch
# ---------------------------------------------------------------------------
# True  : party_type decides business vs individual; a valid NAICS on a P row
#         does NOT promote it to business (flagged `individual_naics_valid`
#         in entity_class instead).
# False : pre-party_type behaviour — a valid NAICS outranks everything.
PARTY_TYPE_WINS = True

_PARTY_MAP = {"P": "individual", "O": "business"}

# ---------------------------------------------------------------------------
# Fixed category sets — schema stability AND the memory fix
# ---------------------------------------------------------------------------
# Declaring categories up front does two things: it pins the node-table schema
# (a category cannot appear that was not specified), and it lets every derived
# column be stored as int8 codes. The pipeline pads absent nodes with
# 'not_in_customers', so that value MUST be a declared category or the pad
# silently becomes NaN on concat.
ENTITY_TYPE_CATS = ("business", "individual", "unknown")
SOURCE_CATS = ("party_type", "name", "none")
NAICS_STATUS_CATS = ("valid", "placeholder", "missing")
COVERAGE_CLASS_CATS = ("valid", "placeholder", "missing", "not_applicable")
NODE_TYPE_CATS = ("business_naics_valid", "business_naics_placeholder",
                  "business_naics_missing", "individual", "unknown")
ENTITY_CLASS_CATS = NODE_TYPE_CATS + ("individual_naics_valid",
                                      "unknown_naics_valid")
GEO_STATUS_CATS = ("valid", "placeholder", "missing", "not_in_customers")
ATTR_PROFILE_CATS = ("identity+geo", "identity_only", "geo_only", "neither",
                     "not_in_customers")

PLACEHOLDER_NAICS = {"-1", "0", "00", "000000", "999999", "******",
                     "UNKNOWN", "N/A", "NA", "NULL", "NONE"}
PLACEHOLDER_WARN_SHARE = 0.05

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
# One alternation, longest-first so CORPORATION is not eaten by CORP.
# Bounded by start/space rather than \b because '&' is not a word character.
_BIZ_RE = (r"(?:^| )(?:"
           + "|".join(sorted((__import__("re").escape(t)
                              for t in _BUSINESS_TOKENS), key=len,
                             reverse=True))
           + r")(?: |$)")
# 2 or 3 all-alphabetic tokens == the v7 rule (n_alpha == n_tok, 2 <= n <= 3)
_PERSON_RE = r"[A-Z]+ [A-Z]+( [A-Z]+)?"

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

_IDENTITY_FIELDS = ("customer_name", "party_type", "naics_cd", "naics_desc")
_GEO_FIELDS = ("addr_loc_rec_type", "longitude_degree", "latitude_degree",
               "zip_cd", "state_or_province", "city")
_MUST_BE_TEXT = ("mdm_id", "zip_cd", "naics_cd", "party_type")

TYPING_COLS = ["node", "entity_type", "entity_type_observed",
               "entity_type_inferred", "entity_type_source", "entity_class",
               "naics_status", "naics_applicable", "naics_coverage_class",
               "node_type", "node_type_v5", "naics_clean"]

_NUM_RE = r"^\s*-?\d+(\.\d+)?\s*$"
_MAX_EXACT_INT = 2 ** 53


# ---------------------------------------------------------------------------
# memory instrumentation
# ---------------------------------------------------------------------------

def _rss_gb() -> float:
    """Peak RSS of this process, GB. Linux ru_maxrss is in KB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_048_576


def _mem(stage: str, n: int | None = None):
    log.info("MEM %-26s peak RSS %6.2f GB%s", stage, _rss_gb(),
             f"  rows={n:,}" if n is not None else "")


def _select_cat(conds, labels, cats, default) -> pd.Categorical:
    """np.select over CATEGORY CODES, not strings.

    `np.select(conds, ["business_naics_valid", ...])` returns a `<U26` array
    — 104 bytes per element, 3.1 GB at 30M rows, per column. Selecting int
    codes and building the Categorical from them costs 1 byte per element.
    This is the single change that made the full extract fit in memory.
    """
    cats = list(cats)
    codes = np.select(conds, [cats.index(l) for l in labels],
                      default=cats.index(default)).astype(np.int8)
    return pd.Categorical.from_codes(codes, categories=cats)


@dataclass
class Customers:
    """`attrs` is the single materialised frame. `typing` is a projection of
    it, not a second copy — v7 held both plus the pipeline's merge of the
    two, three 30M-row frames for one dimension."""
    attrs: pd.DataFrame
    coords: pd.DataFrame
    disagreement: pd.DataFrame = field(default_factory=pd.DataFrame)
    join_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    dtype_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_rows: int = 0

    @property
    def typing(self) -> pd.DataFrame:
        return self.attrs[[c for c in TYPING_COLS if c in self.attrs.columns]]

    def coverage(self, nodes) -> dict:
        nodes = pd.Index(pd.unique(pd.Series(nodes, dtype="object").astype(str)))
        a = self.attrs.set_index("node")
        n = max(len(nodes), 1)
        an = a.reindex(nodes)
        biz = (an["entity_type"] == "business")
        n_biz = int(biz.sum())
        return {
            "n_nodes": len(nodes),
            "pct_in_customers": round(100 * nodes.isin(a.index).sum() / n, 2),
            "pct_party_type_observed": round(
                100 * int((an["entity_type_source"] == "party_type").sum()) / n, 2),
            "pct_business": round(100 * n_biz / n, 2),
            "pct_individual": round(
                100 * int((an["entity_type"] == "individual").sum()) / n, 2),
            "pct_naics_valid": round(
                100 * int((an["naics_status"] == "valid").sum()) / n, 2),
            "pct_naics_valid_of_business": round(
                100 * int(((an["naics_status"] == "valid") & biz).sum())
                / max(n_biz, 1), 2),
            "pct_geo_valid": round(
                100 * int((an["geo_status"] == "valid").sum()) / n, 2),
        }


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _read_source(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the mapped columns."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq", ".parq"):
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(path).schema_arrow
        lower = {c.lower(): c for c in schema.names}
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
             "present": c in resolved} for c in _COLMAP])
        log.info("customers: parquet read — %d of %d mapped columns present, "
                 "%d rows", len(wanted), len(_COLMAP), len(df))
    elif ext == ".csv":
        log.warning("customers: reading CSV (%s). Parquet is the supported "
                    "source; CSV loses NULL-vs-empty.", path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        rep = pd.DataFrame([{"canonical": c, "source_column": None,
                             "arrow_type": "string(csv)", "present": None}
                            for c in _COLMAP])
    else:
        raise ValueError(f"unsupported customers file extension: {ext!r}")
    return df, rep


def node_universe(snapshot_glob: str) -> pd.Index:
    """Union of node ids across the monthly snapshots, two columns at a time.

    Optional lever, not the default. The dimension covers ~30.8M MDM parties
    while the graph touches ~5.3M; holding the other 25M for the whole run
    buys only the ability to report coverage against MDM_ALL. Passing the
    result to `load_customers(node_filter=...)` cuts the rows entering the
    typing stage by ~80%.

    Returns an Index, not a set: `isin` against an Index is hash-joined by
    pandas, while a 5M-element Python set costs ~600 MB on its own. Note the
    filter is applied AFTER the read, not pushed into pyarrow — a `filters=`
    predicate with 5M values would be evaluated per row group and is far
    slower than reading and masking. Costs one streaming pass (~1-2 min).
    """
    import pyarrow.parquet as pq
    import pyarrow.csv as pv
    paths = sorted(glob.glob(snapshot_glob))
    if not paths:
        raise FileNotFoundError(f"no snapshots matched {snapshot_glob}")
    seen: set = set()
    for p in paths:
        if p.endswith(".csv"):
            t = pv.read_csv(p, convert_options=pv.ConvertOptions(
                include_columns=["source", "dest"]))
        else:
            t = pq.read_table(p, columns=["source", "dest"])
        for c in ("source", "dest"):
            seen.update(t.column(c).cast("string").to_pylist())
        del t
    idx = pd.Index(pd.Series(sorted(seen), dtype="string"))
    del seen
    gc.collect()
    log.info("node_universe: %d distinct nodes across %d snapshots",
             len(idx), len(paths))
    return idx


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).strip().lower(): c for c in df.columns}
    out = {}
    for canon, aliases in _COLMAP.items():
        for a in aliases:
            if a in lower:
                out[canon] = df[lower[a]]
                break
        else:
            out[canon] = pd.Series(pd.NA, index=df.index, dtype="string")
            log.warning("customers: column '%s' not found — filled empty", canon)
    return pd.DataFrame(out)


def _dtype_report(df: pd.DataFrame, rep: pd.DataFrame) -> pd.DataFrame:
    notes = []
    for c in _MUST_BE_TEXT:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        msg = {
            "zip_cd": ("ZIP stored numerically — LEADING ZEROS LOST; "
                       "zero-padding to 5, but 0-prefixed ZIPs cannot be "
                       "distinguished from genuinely short values"),
            "mdm_id": ("id stored numerically — cast to string; a float id "
                       "above 2^53 is not exactly representable"),
            "naics_cd": ("NAICS stored numerically — the '******' sentinel "
                         "cannot survive this type and became NULL upstream; "
                         "placeholder counts are NOT comparable across a "
                         "dtype change"),
        }.get(c, "expected text, stored numerically")
        notes.append({"column": c, "dtype": str(df[c].dtype), "issue": msg})
    out = pd.DataFrame(notes)
    for _, r in out.iterrows():
        log.warning("customers: DTYPE — %s is %s: %s",
                    r["column"], r["dtype"], r["issue"])
    if not rep.empty:
        out = pd.concat([rep.assign(issue=None), out], ignore_index=True)
    return out


def _id_to_str(s: pd.Series, name: str = "mdm_id") -> pd.Series:
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        log.warning("customers: %s is float64 — converting via Int64. "
                    "%d non-integral, %d beyond 2^53 (NOT exactly "
                    "representable; fix the extract to CAST AS STRING)",
                    name, int((v != v.round()).sum()),
                    int((v.abs() >= _MAX_EXACT_INT).sum()))
        return s.round().astype("Int64").astype("string")
    if pd.api.types.is_integer_dtype(s):
        return s.astype("Int64").astype("string")
    return s.astype("string").str.strip()


def _text(s: pd.Series) -> pd.Series:
    """Any dtype -> nullable string, stripped, empty normalised to NA.
    Parquet preserves the NULL / '' distinction CSV collapses; every
    downstream test asks only 'is this absent', so the two are normalised
    here, once, rather than by whichever .fillna('') runs first."""
    if pd.api.types.is_float_dtype(s):
        out = s.round().astype("Int64").astype("string")
    elif pd.api.types.is_integer_dtype(s):
        out = s.astype("Int64").astype("string")
    else:
        out = s.astype("string")
    out = out.str.strip()
    return out.mask(out == "")


def _num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    t = s.astype("string").str.strip()
    ok = t.str.match(_NUM_RE).fillna(False)
    return pd.to_numeric(t.where(ok), errors="coerce")


def _zip_digits(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        d = s.round().astype("Int64").astype("string")
        short = int((d.str.len() < 5).sum())
        d = d.str.zfill(5)
        if short:
            log.warning("customers: %d numeric ZIPs shorter than 5 digits "
                        "zero-padded (02108 arrives as 2108)", short)
        return d.fillna("")
    return s.astype("string").fillna("").str.replace(r"[^0-9]", "", regex=True)


def _name_typer(names: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised entity-type inference from the customer name.

    Replaces v7's `up.map(lambda s: punct.sub(' ', s).split())`, which
    allocated one Python list per row — ~3.6 GB of list headers at 30M rows
    (before the str objects inside), held across three further `.map()`
    passes. Verified to agree 1.000 with the v7 logic on both outputs.

    person_shaped in v7 = no business token, and n_alpha == n_tok, and
    2 <= n_tok <= 3. "every token alphabetic, 2 or 3 of them" is exactly
    the fullmatch below, so no token list is needed.
    """
    norm = (names.fillna("").astype("string").str.upper()
            .str.replace(r"[^A-Z0-9& ]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    has_biz = norm.str.contains(_BIZ_RE, regex=True, na=False).to_numpy(bool)
    person = (~has_biz) & norm.str.fullmatch(_PERSON_RE,
                                             na=False).to_numpy(bool)
    del norm
    return has_biz, person


def _geo_rank(df: pd.DataFrame) -> np.ndarray:
    lat = _num(df["latitude_degree"])
    lon = _num(df["longitude_degree"])
    has_xy = (lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))
              & lat.between(-90, 90) & lon.between(-180, 180)).to_numpy(bool)
    rec = df["addr_loc_rec_type"].astype("string").fillna("").str.upper()
    is_ph = rec.isin(PLACEHOLDER_REC_TYPES).to_numpy(bool)
    zipd = (_zip_digits(df["zip_cd"]).str.len() >= 5).to_numpy(bool)
    return np.select([has_xy & ~is_ph, has_xy, zipd], [3, 2, 1],
                     default=0).astype(np.int8)


def _resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse an outer-join fanout to one row per node. Identity coalesced
    across the group; geography taken whole from the best-ranked row."""
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
        stats[f"conflict_{c}"] = int((d.groupby("node")[c].nunique(dropna=True)
                                      > 1).sum())
    del d

    ident = (df.groupby("node", sort=False)[list(_IDENTITY_FIELDS)]
             .first().reset_index())
    # idxmax on the rank, not a full sort: sorting 30M string keys is the
    # expensive way to answer "which row wins per group". idxmax breaks ties
    # on first occurrence, which is the stable order we want anyway.
    rank = pd.Series(_geo_rank(df), index=df.index)
    keep = rank.groupby(df["node"], sort=False).idxmax()
    del rank
    g = df.loc[keep.to_numpy(), ["node", *_GEO_FIELDS]]
    out = ident.merge(g, on="node", how="left")
    del ident, g, keep
    gc.collect()
    log.warning("customers: outer-join fanout — %d rows collapsed to %d "
                "nodes (%d duplicate rows); identity coalesced, geo taken "
                "from best-ranked address row", n_rows_in, len(out), n_dup_rows)
    for c in ("party_type", "naics_cd", "customer_name"):
        if stats[f"conflict_{c}"]:
            log.warning("customers: %d nodes carry CONFLICTING %s across "
                        "duplicate rows — first non-null kept; investigate "
                        "the join key", stats[f"conflict_{c}"], c)
    return out.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------------

def load_customers(path: str, party_type_wins: bool | None = None,
                   node_filter=None) -> Customers:
    """Read the customer extract and derive typing + geo status. One pass.

    node_filter: optional iterable of node ids. Restricts the dimension to
    the graph node universe — see `node_universe()`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"customers file not found: {path}")
    ptw = PARTY_TYPE_WINS if party_type_wins is None else bool(party_type_wins)
    _mem("start")

    raw, schema_rep = _read_source(path)
    n_raw = len(raw)
    _mem("after parquet read", n_raw)

    df = _resolve_columns(raw)
    del raw
    gc.collect()
    dtypes = _dtype_report(df, schema_rep)

    df["node"] = _id_to_str(df["mdm_id"])
    df.drop(columns=["mdm_id"], inplace=True)
    n_no_id = int(df["node"].isna().sum())
    if n_no_id:
        log.warning("customers: %d rows dropped — no mdm_id (address-side "
                    "rows of the outer join with no customer match)", n_no_id)
    df = df[df["node"].notna()]

    if node_filter is not None:
        # Restrict to the graph node universe BEFORE the typing stage — that
        # is where the per-row cost lives, so filtering here is what the
        # filter is for. Coverage against MDM_ALL is no longer derivable
        # from this object once applied.
        before = len(df)
        df = df[df["node"].isin(pd.Index(node_filter))]
        log.info("customers: node_filter applied — %d of %d rows retained "
                 "(%.1f%%)", len(df), before, 100 * len(df) / max(before, 1))
        gc.collect()

    for c in ("customer_name", "party_type", "naics_cd", "naics_desc",
              "addr_loc_rec_type", "state_or_province", "city"):
        df[c] = _text(df[c])
    _mem("after text normalisation", len(df))

    df, join_stats = _resolve_duplicates(df)
    n = len(df)
    _mem("after duplicate resolution", n)

    # ---- entity type: OBSERVED (party_type) -------------------------------
    pt = df["party_type"].fillna("").str.upper().str[:1]
    bad = ~pt.isin(list(_PARTY_MAP) + [""])
    if bad.any():
        log.warning("customers: %d rows carry party_type outside {P,O,null}: "
                    "%s — typed unknown", int(bad.sum()),
                    df.loc[bad, "party_type"].value_counts().head(5).to_dict())
    is_p = (pt == "P").to_numpy(bool)
    is_o = (pt == "O").to_numpy(bool)
    del pt, bad
    entity_type_observed = _select_cat([is_o, is_p],
                                       ["business", "individual"],
                                       ENTITY_TYPE_CATS, "unknown")

    # ---- entity type: INFERRED (vectorised name typer) --------------------
    has_biz, person_shaped = _name_typer(df["customer_name"])
    entity_type_inferred = _select_cat([has_biz, person_shaped],
                                       ["business", "individual"],
                                       ENTITY_TYPE_CATS, "unknown")
    _mem("after name typing", n)

    # ---- resolve: observed wins, inference is the fallback only -----------
    obs_known = is_o | is_p
    inf_known = has_biz | person_shaped
    is_biz = np.where(obs_known, is_o, has_biz)
    is_ind = np.where(obs_known, is_p, person_shaped)
    entity_type = _select_cat([is_biz, is_ind], ["business", "individual"],
                              ENTITY_TYPE_CATS, "unknown")
    entity_type_source = _select_cat([obs_known, inf_known],
                                     ["party_type", "name"],
                                     SOURCE_CATS, "none")
    del obs_known, inf_known, is_o, is_p

    # ---- NAICS: field quality only (applicability is a separate axis) -----
    code = df["naics_cd"].fillna("")
    is_missing = (code == "").to_numpy(bool)
    is_ph = code.str.upper().isin(PLACEHOLDER_NAICS).fillna(False).to_numpy(bool)
    looks_code = code.str.match(r"^\d{2}").fillna(False).to_numpy(bool)
    is_valid = (~is_missing) & (~is_ph) & looks_code
    naics_status = _select_cat([is_missing, is_ph, looks_code],
                               ["missing", "placeholder", "valid"],
                               NAICS_STATUS_CATS, "placeholder")
    naics_clean = code.str.extract(r"^(\d{2,6})")[0].where(
        pd.Series(is_valid, index=df.index)).astype("category")
    del code, looks_code

    naics_applicable = is_biz.astype(np.int8)
    # An OBSERVED valid code reports valid even on a person (sole proprietor)
    # — never mask an observation as inapplicable. 'not_applicable' means
    # "no code, and none was expected".
    naics_coverage_class = _select_cat(
        [is_valid, ~is_biz, is_ph], ["valid", "not_applicable", "placeholder"],
        COVERAGE_CLASS_CATS, "missing")

    # ---- node_type: the 5-category composition key (schema-stable) --------
    if ptw:
        node_type = _select_cat(
            [is_biz & is_valid, is_biz & is_ph, is_biz & is_missing, is_ind],
            ["business_naics_valid", "business_naics_placeholder",
             "business_naics_missing", "individual"],
            NODE_TYPE_CATS, "unknown")
    else:
        node_type = _select_cat(
            [is_valid, is_biz & is_ph, is_biz & is_missing, is_ind],
            ["business_naics_valid", "business_naics_placeholder",
             "business_naics_missing", "individual"],
            NODE_TYPE_CATS, "unknown")

    # what the pre-party_type rule WOULD have produced on the same rows
    node_type_v5 = _select_cat(
        [is_valid, has_biz & is_ph, has_biz & is_missing, person_shaped],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual"],
        NODE_TYPE_CATS, "unknown")

    # ---- entity_class: fine-grained label (NOT a composition key) ---------
    entity_class = _select_cat(
        [is_biz & is_valid, is_biz & is_ph, is_biz & is_missing,
         is_ind & is_valid, is_ind, is_valid],
        ["business_naics_valid", "business_naics_placeholder",
         "business_naics_missing", "individual_naics_valid", "individual",
         "unknown_naics_valid"],
        ENTITY_CLASS_CATS, "unknown")
    del has_biz, person_shaped, is_ph, is_missing
    _mem("after typing", n)

    # ---- geography --------------------------------------------------------
    lat = _num(df["latitude_degree"]).astype("float32")
    lon = _num(df["longitude_degree"]).astype("float32")
    rec = df["addr_loc_rec_type"].fillna("").str.upper()
    zipd = _zip_digits(df["zip_cd"])
    zip5 = zipd.where(zipd.str.len() >= 5).str[:5].astype("category")
    zip3 = zipd.where(zipd.str.len() >= 3).str[:3].astype("category")
    del zipd

    has_geo = (lat.notna() & lon.notna() & ~((lat == 0) & (lon == 0))
               & lat.between(-90, 90) & lon.between(-180, 180)).to_numpy(bool)
    is_ph_geo = rec.isin(PLACEHOLDER_REC_TYPES).to_numpy(bool)
    geo_status = _select_cat([~has_geo, is_ph_geo], ["missing", "placeholder"],
                             GEO_STATUS_CATS, "valid")
    has_identity = (df["customer_name"].notna() | df["party_type"].notna()
                    | df["naics_cd"].notna()).to_numpy(bool)
    attr_profile = _select_cat(
        [has_identity & has_geo, has_identity, has_geo],
        ["identity+geo", "identity_only", "geo_only"],
        ATTR_PROFILE_CATS, "neither")
    shared = rec.isin(SHARED_STRUCTURE_REC_TYPES).astype("int8").to_numpy()
    del rec, has_identity, has_geo, is_ph_geo

    # ---- assemble ONE frame ------------------------------------------------
    attrs = pd.DataFrame({
        "node": df["node"].astype("string"),
        "cust_name": df["customer_name"],          # ~unique: stays a string
        "party_type": df["party_type"].astype("category"),
        "naics_cd": df["naics_cd"].astype("category"),
        "naics_desc": df["naics_desc"].astype("category"),
        "addr_loc_rec_type": df["addr_loc_rec_type"].astype("category"),
        "lat": lat, "lon": lon, "zip5": zip5, "zip3": zip3,
        "state": df["state_or_province"].astype("category"),
        "city": df["city"].astype("category"),
        "geo_status": geo_status,
        "attr_profile": attr_profile,
        "shared_structure": shared,
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
        "naics_clean": naics_clean.reset_index(drop=True),
    })
    del df, lat, lon, zip5, zip3, geo_status, attr_profile, shared
    del entity_type, entity_type_observed, entity_type_inferred
    del entity_type_source, entity_class, naics_status, naics_coverage_class
    del node_type, node_type_v5, is_biz, is_ind, is_valid
    gc.collect()

    nc = attrs["naics_clean"].astype("string")
    for k in range(2, 7):
        attrs[f"naics{k}"] = (nc.str[:k].where(nc.str.len() >= k)
                              .astype("category"))
    attrs["naics_known"] = attrs["naics2"].notna().astype("float32")
    del nc
    gc.collect()
    _mem("after assembly", len(attrs))

    coords = attrs.loc[attrs["geo_status"] == "valid",
                       ["node", "lat", "lon", "zip3", "zip5", "state"]] \
                  .reset_index(drop=True)
    dis = typing_disagreement(attrs)
    join_profile = pd.DataFrame([{
        **join_stats, "nodes_out": len(attrs), "rows_no_mdm_id": n_no_id,
        **attrs["attr_profile"].value_counts().to_dict()}])

    _log_load(attrs, dis, ptw, n_raw, path)
    _mem("done", len(attrs))
    return Customers(attrs=attrs, coords=coords, disagreement=dis,
                     join_profile=join_profile, dtype_report=dtypes,
                     n_rows=len(attrs))


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def typing_disagreement(t: pd.DataFrame) -> pd.DataFrame:
    """Declared vs inferred entity type, plus node_type churn against the
    pre-party_type rule. The cell that matters most is
    (observed=business, inferred=individual): person-named organisations."""
    if t.empty:
        return pd.DataFrame()
    x = (t.groupby(["entity_type_observed", "entity_type_inferred"],
                   observed=True).size().rename("n").reset_index())
    x["pct"] = (100 * x["n"] / x["n"].sum()).round(3)
    x["agree"] = np.where(
        x["entity_type_observed"] == "unknown", "no_observation",
        np.where(x["entity_type_inferred"] == "unknown", "no_inference",
                 np.where(x["entity_type_observed"]
                          == x["entity_type_inferred"], "agree", "CONFLICT")))
    y = (t.groupby(["node_type_v5", "node_type"], observed=True)
         .size().rename("n").reset_index())
    y["pct"] = (100 * y["n"] / y["n"].sum()).round(3)
    y["agree"] = np.where(y["node_type_v5"] == y["node_type"],
                          "agree", "CHANGED")
    x.insert(0, "axis", "entity_type")
    y.insert(0, "axis", "node_type")
    y = y.rename(columns={"node_type_v5": "entity_type_observed",
                          "node_type": "entity_type_inferred"})
    return pd.concat([x, y], ignore_index=True)


def _log_load(attrs, dis, ptw, n_raw, path):
    log.info("customers: %s — %d source rows -> %d unique nodes (%s, "
             "PARTY_TYPE_WINS=%s)", os.path.basename(path), n_raw,
             len(attrs), ATTR_VERSION, ptw)
    for col in ("attr_profile", "entity_type_source", "entity_type",
                "entity_class", "naics_status", "naics_coverage_class",
                "node_type", "geo_status"):
        log.info("customers: %s=%s", col,
                 attrs[col].value_counts().to_dict())
    n = max(len(attrs), 1)
    ph = int((attrs["naics_status"] == "placeholder").sum())
    if ph / n > PLACEHOLDER_WARN_SHARE:
        log.warning("customers: %d (%.1f%%) placeholder NAICS — above the "
                    "%.0f%% review threshold", ph, 100 * ph / n,
                    100 * PLACEHOLDER_WARN_SHARE)
    else:
        log.info("customers: %d placeholder NAICS (%.2f%%) — expected "
                 "non-zero when naics_cd is a text column", ph, 100 * ph / n)
    if not dis.empty:
        d = dis[dis["axis"] == "entity_type"]
        conflict = float(d.loc[d["agree"] == "CONFLICT", "pct"].sum())
        bp = float(d.loc[(d["entity_type_observed"] == "business")
                         & (d["entity_type_inferred"] == "individual"),
                         "pct"].sum())
        churn = float(dis.loc[(dis["axis"] == "node_type")
                              & (dis["agree"] == "CHANGED"), "pct"].sum())
        log.info("TYPING DISAGREEMENT: conflict %.2f%% | person-named "
                 "organisations %.2f%% | node_type churn %.2f%%",
                 conflict, bp, churn)
        if churn > 5:
            log.warning("node_type churn %.2f%% — every share_* composition "
                        "column and the headline individual-share figure "
                        "move by this much.", churn)


def write_qa(cust: Customers, out_dir: str) -> None:
    qa = os.path.join(out_dir, "qa")
    os.makedirs(qa, exist_ok=True)
    cust.disagreement.to_csv(
        os.path.join(qa, "customers_typing_disagreement.csv"), index=False)
    cust.join_profile.to_csv(
        os.path.join(qa, "customers_join_profile.csv"), index=False)
    cust.dtype_report.to_csv(
        os.path.join(qa, "customers_source_schema.csv"), index=False)
    (cust.attrs.groupby(["entity_type", "naics_coverage_class"],
                        observed=True).size().rename("n").reset_index()
     .to_csv(os.path.join(qa, "customers_naics_coverage.csv"), index=False))
    log.info("customers: QA artefacts written to %s", qa)


def naics_partition(attrs: pd.DataFrame) -> pd.DataFrame:
    n2 = attrs["naics2"].astype("string").fillna("NA")
    return pd.DataFrame({"node": attrs["node"].to_numpy(),
                         "community_id": n2.astype(str).to_numpy()})
