"""
pkn_subro.py
============
Insurance **subrogation network** extraction from PKN monthly customer-payment
snapshots.

Design notes
------------
*Never* materialise the full aggregate graph in a Python graph library.
The pipeline is:

    monthly CSVs  --pandas-->  aggregated edge table (int32 node codes)
                  --mask-->    seed set (NAICS / name regex)
                  --mask-->    1st-order ego induced subgraph  (small)
                  --networkx-> analysis on the small subgraph only

Rationale: ~5M edges/month x 11 months de-duplicated is O(10-30M) unique edges.
NetworkX would need ~15-25 GB for that (dict-of-dict per node + per edge attr
dict). Pandas holds the same table in ~1 GB. The insurance ego subgraph is
typically 1e4-1e6 edges, which NetworkX handles comfortably and with far nicer
attribute ergonomics than networkit. Use `to_networkit()` only if you later want
global metrics over the whole aggregate without touching cuGraph.

Two-pass design:
    Pass 1  build_aggregate()   -> memory-lean whole-graph aggregate
    Pass 2  monthly_panel()     -> full monthly detail, restricted to ego nodes

Author: prepared for PKN / Treasury Management Data Science
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "NodeIndex",
    "month_range",
    "split_naics",
    "naics_levels",
    "SECTOR_MAP",
    "load_month",
    "build_aggregate",
    "AggregateResult",
    "node_stats",
    "INSURANCE_NAICS",
    "INSURANCE_NAICS_PREFIXES",
    "INSURANCE_NAME_STRONG",
    "INSURANCE_NAME_WEAK",
    "INSURANCE_NAME_EXCLUDE",
    "CARRIER_NAME_HINT",
    "select_seeds",
    "k_hop_nodes",
    "ego_subgraph",
    "carrier_core",
    "bilateral_pairs",
    "score_subrogation",
    "flag_affiliates",
    "counterparty_naics_profile",
    "shared_counterparties",
    "pass_through_intermediaries",
    "monthly_panel",
    "lagged_xcorr",
    "to_networkx",
    "to_networkit",
    "describe_ego",
]

RAW_COLUMNS = [
    "source", "source_name", "source_naics",
    "amount", "volume",
    "dest", "dest_name", "dest_naics",
]


# --------------------------------------------------------------------------- #
# 1. Node identifier index (string id -> int32 code)
# --------------------------------------------------------------------------- #
class NodeIndex:
    """Incremental string -> int32 encoder, stable across months."""

    def __init__(self) -> None:
        self._map: dict[str, int] = {}
        self.ids: list[str] = []

    def __len__(self) -> int:
        return len(self.ids)

    def encode(self, values: pd.Series) -> pd.Series:
        codes = values.map(self._map)
        missing = codes.isna()
        if missing.any():
            new_vals = pd.unique(values[missing])
            start = len(self.ids)
            self._map.update({v: start + i for i, v in enumerate(new_vals)})
            self.ids.extend(new_vals.tolist())
            codes = values.map(self._map)
        return codes.astype("int32")

    def decode(self, codes: Iterable[int]) -> list[str]:
        ids = self.ids
        return [ids[int(c)] for c in codes]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"node": np.arange(len(self.ids), dtype="int32"),
                             "node_id": self.ids})


# --------------------------------------------------------------------------- #
# 2. Month helpers
# --------------------------------------------------------------------------- #
def month_range(start: str = "2025-01", end: str = "2025-11") -> list[str]:
    """['2025-01', ..., '2025-11'] inclusive."""
    return [str(p) for p in pd.period_range(start=start, end=end, freq="M")]


def month_path(data_dir: str, month: str, prefix: str = "cust_") -> str:
    return os.path.join(data_dir, f"{prefix}{month}.csv")


# --------------------------------------------------------------------------- #
# 3. NAICS parsing
# --------------------------------------------------------------------------- #
# Values seen in the wild: "524126|Direct Property and Casualty Insurance
# Carriers", "-1|UNKNOWN", "******", "", NaN.
_NAICS_VALID = re.compile(r"^\d{2,6}$")

SECTOR_MAP = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, and Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade", "45": "Retail Trade",
    "48": "Transportation and Warehousing", "49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental and Leasing",
    "54": "Professional, Scientific, and Technical Services",
    "55": "Management of Companies and Enterprises",
    "56": "Administrative and Support and Waste Management",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services (except Public Administration)",
    "92": "Public Administration",
}


def split_naics(s: pd.Series) -> pd.DataFrame:
    """Split 'code|description' into cleaned code / description columns.

    Invalid or unknown codes ('-1', '******', '', NaN) become <NA>; the
    description is kept whenever present so nothing is silently lost.
    """
    txt = s.fillna("").astype("string").str.strip()
    parts = txt.str.split("|", n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = pd.NA

    code = parts[0].str.strip().str.replace(r"\.0$", "", regex=True)
    desc = parts[1].str.strip()

    valid = code.str.match(_NAICS_VALID).fillna(False)
    code = code.where(valid, other=pd.NA)

    desc = desc.replace({"": pd.NA, "UNKNOWN": pd.NA, "unknown": pd.NA})
    return pd.DataFrame({"naics_code": code, "naics_desc": desc}, index=s.index)


def naics_levels(code: pd.Series,
                 levels: Sequence[int] = (2, 3, 4, 5, 6),
                 add_sector: bool = True) -> pd.DataFrame:
    """Prefix-slice a NAICS code into naics2..naics6.

    A level is <NA> when the source code is shorter than that level, so a
    4-digit code yields naics2/3/4 and null naics5/6 (rather than a fake pad).
    """
    code = code.astype("string")
    ln = code.str.len()
    out = {}
    for L in levels:
        out[f"naics{L}"] = code.str.slice(0, L).where(ln >= L)
    df = pd.DataFrame(out, index=code.index)
    if add_sector and "naics2" in df:
        df["naics_sector"] = df["naics2"].map(SECTOR_MAP).astype("string")
    return df


# --------------------------------------------------------------------------- #
# 4. Snapshot loading
# --------------------------------------------------------------------------- #
def load_month(path: str, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """Read one monthly snapshot CSV with tight dtypes."""
    dtypes = {
        "source": "string", "source_name": "string", "source_naics": "string",
        "dest": "string", "dest_name": "string", "dest_naics": "string",
        "amount": "float64", "volume": "float64",
    }
    df = pd.read_csv(path, dtype=dtypes, usecols=usecols)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for c in ("source", "dest"):
        df[c] = df[c].str.strip()
    df = df[df["source"].notna() & df["dest"].notna()]
    df["amount"] = df["amount"].fillna(0.0)
    df["volume"] = df["volume"].fillna(0.0)
    return df


@dataclass
class AggregateResult:
    edges: pd.DataFrame          # one row per (source, dest) pair, whole period
    nodes: pd.DataFrame          # one row per node, with parsed NAICS
    index: NodeIndex
    months: list[str] = field(default_factory=list)

    @property
    def n_months(self) -> int:
        return len(self.months)


def build_aggregate(months: Sequence[str],
                    data_dir: str = "../data",
                    prefix: str = "cust_",
                    drop_self_loops: bool = True,
                    verbose: bool = True) -> AggregateResult:
    """Fold monthly snapshots into a single weighted, aggregated edge table.

    Edge weights
    ------------
    amount    : total $ over the period
    volume    : total transaction count over the period
    n_rels    : number of *snapshots* in which the edge exists  (1 per month)
    amount_sq : sum of squared monthly amounts -> monthly mean/std/CV, no lists
    first_i / last_i : first and last month index in which the edge appears

    Memory: the accumulator is folded one month at a time, so peak usage is
    roughly |aggregate so far| + |current month| rather than sum of all months.
    """
    idx = NodeIndex()
    acc: pd.DataFrame | None = None
    node_attrs: dict[int, tuple] = {}          # code -> (name, naics_raw)
    node_first: dict[int, int] = {}
    node_last: dict[int, int] = {}
    node_months: dict[int, int] = {}

    for i, m in enumerate(months):
        path = month_path(data_dir, m, prefix)
        df = load_month(path)
        df["s"] = idx.encode(df["source"])
        df["d"] = idx.encode(df["dest"])
        if drop_self_loops:
            df = df[df["s"] != df["d"]]

        # --- node attributes: last month wins ---------------------------- #
        side_s = df[["s", "source_name", "source_naics"]].rename(
            columns={"s": "node", "source_name": "name", "source_naics": "naics_raw"})
        side_d = df[["d", "dest_name", "dest_naics"]].rename(
            columns={"d": "node", "dest_name": "name", "dest_naics": "naics_raw"})
        nn = pd.concat([side_s, side_d], ignore_index=True)
        nn = nn.drop_duplicates(subset="node", keep="last")
        for node, name, naics in nn.itertuples(index=False):
            node_attrs[node] = (name, naics)
            node_last[node] = i
            node_first.setdefault(node, i)
            node_months[node] = node_months.get(node, 0) + 1

        # --- within-month edge dedupe ------------------------------------ #
        g = (df.groupby(["s", "d"], sort=False, as_index=False)
               .agg(amount=("amount", "sum"), volume=("volume", "sum")))
        g["n_rels"] = np.int32(1)
        g["amount_sq"] = g["amount"] ** 2
        g["first_i"] = np.int16(i)
        g["last_i"] = np.int16(i)

        # --- fold --------------------------------------------------------- #
        if acc is None:
            acc = g
        else:
            acc = (pd.concat([acc, g], ignore_index=True)
                     .groupby(["s", "d"], sort=False, as_index=False)
                     .agg(amount=("amount", "sum"),
                          volume=("volume", "sum"),
                          n_rels=("n_rels", "sum"),
                          amount_sq=("amount_sq", "sum"),
                          first_i=("first_i", "min"),
                          last_i=("last_i", "max")))
        if verbose:
            print(f"  [{i + 1}/{len(months)}] {m}: "
                  f"{len(df):,} rows -> aggregate {len(acc):,} edges, "
                  f"{len(idx):,} nodes")
        del df, g

    assert acc is not None, "no months loaded"
    edges = acc

    # --- derived edge features -------------------------------------------- #
    n = edges["n_rels"].astype("float64")
    edges["amt_mean_month"] = edges["amount"] / n
    var = (edges["amount_sq"] / n) - edges["amt_mean_month"] ** 2
    edges["amt_std_month"] = np.sqrt(var.clip(lower=0))
    edges["amt_cv"] = (edges["amt_std_month"] /
                       edges["amt_mean_month"].replace(0, np.nan))
    edges["avg_ticket"] = edges["amount"] / edges["volume"].replace(0, np.nan)
    edges["span"] = (edges["last_i"] - edges["first_i"] + 1).astype("int16")
    edges["density"] = n / edges["span"]
    edges = edges.drop(columns=["amount_sq"])

    # --- node table -------------------------------------------------------- #
    codes = np.arange(len(idx), dtype="int32")
    names = pd.Series([node_attrs.get(c, (pd.NA, pd.NA))[0] for c in codes],
                      dtype="string")
    naics_raw = pd.Series([node_attrs.get(c, (pd.NA, pd.NA))[1] for c in codes],
                          dtype="string")
    nodes = pd.DataFrame({
        "node": codes,
        "node_id": idx.ids,
        "name": names,
        "naics_raw": naics_raw,
        "first_i": [node_first.get(c, -1) for c in codes],
        "last_i": [node_last.get(c, -1) for c in codes],
        "months_seen": [node_months.get(c, 0) for c in codes],
    })
    nodes = pd.concat([nodes, split_naics(nodes["naics_raw"])], axis=1)
    nodes = pd.concat([nodes, naics_levels(nodes["naics_code"])], axis=1)
    nodes["name_clean"] = (nodes["name"].str.upper()
                           .str.replace(r"[^A-Z0-9 &]", " ", regex=True)
                           .str.replace(r"\s+", " ", regex=True).str.strip())

    return AggregateResult(edges=edges, nodes=nodes, index=idx,
                           months=list(months))


def node_stats(edges: pd.DataFrame, n_nodes: int | None = None) -> pd.DataFrame:
    """In/out degree, strength and volume per node from the aggregate table."""
    out = (edges.groupby("s", sort=False)
                .agg(out_deg=("d", "size"), out_amt=("amount", "sum"),
                     out_vol=("volume", "sum"), out_rels=("n_rels", "sum")))
    inn = (edges.groupby("d", sort=False)
                .agg(in_deg=("s", "size"), in_amt=("amount", "sum"),
                     in_vol=("volume", "sum"), in_rels=("n_rels", "sum")))
    idx = pd.Index(range(n_nodes), name="node") if n_nodes else None
    st = out.join(inn, how="outer")
    if idx is not None:
        st = st.reindex(idx)
    st = st.fillna(0)
    st.index.name = "node"
    st["deg"] = st["out_deg"] + st["in_deg"]
    st["strength"] = st["out_amt"] + st["in_amt"]
    st["net_amt"] = st["in_amt"] - st["out_amt"]
    return st.reset_index()


# --------------------------------------------------------------------------- #
# 5. Insurance seed selection
# --------------------------------------------------------------------------- #
INSURANCE_NAICS = {
    # --- carriers (the subrogation core lives here) --------------------- #
    "524113": "Direct Life Insurance Carriers",
    "524114": "Direct Health and Medical Insurance Carriers",
    "524126": "Direct Property and Casualty Insurance Carriers",
    "524127": "Direct Title Insurance Carriers",
    "524128": "Other Direct Insurance (except Life, Health, Medical) Carriers",
    "524130": "Reinsurance Carriers",
    # --- the service ring around the carriers ---------------------------- #
    "524210": "Insurance Agencies and Brokerages",
    "524291": "Claims Adjusting",
    "524292": "Third Party Administration of Insurance and Pension Funds",
    "524298": "All Other Insurance Related Activities",
}

INSURANCE_NAICS_PREFIXES = ("5241", "5242")

# Downstream NAICS that show up on the *payee* side of a claim and therefore
# anchor the subrogation motif (carrier pays vendor -> other carrier reimburses).
CLAIM_VENDOR_NAICS = {
    "8111": "Automotive Repair and Maintenance",
    "6211": "Offices of Physicians",
    "6213": "Offices of Other Health Practitioners",
    "6214": "Outpatient Care Centers",
    "6216": "Home Health Care Services",
    "6221": "General Medical and Surgical Hospitals",
    "5411": "Legal Services",
    "5416": "Management, Scientific, and Technical Consulting",
    "5324": "Commercial and Industrial Machinery Rental (incl. auto rental)",
    "2382": "Building Equipment Contractors (restoration)",
    "5617": "Services to Buildings and Dwellings (mitigation/restoration)",
    "4231": "Motor Vehicle and Parts Merchant Wholesalers (salvage)",
}

# Strong signal: almost never a false positive.
INSURANCE_NAME_STRONG = (
    r"(?i)(\binsurance\b|\binsurers?\b|\bassurance\b|\bcasualty\b|\bindemnity\b"
    r"|\bunderwriters?\b|\breciprocal\b|\brisk retention\b|\bRRG\b"
    r"|\breinsur\w*|\bsubrogation\b|\bsubro\b|\barbitration forums?\b"
    r"|\bins\s+co\b|\bins\.\s|\bassur\b)"
)

# Weak signal: use with a NAICS confirmation or eyeball the hits.
INSURANCE_NAME_WEAK = (
    r"(?i)(\bmutual\b|\bclaims?\b|\badjust\w+|\bTPA\b|\brecovery\b|\bsalvage\b"
    r"|\bcaptive\b|\bsurety\b|\bwarranty\b|\bmedicare secondary\b|\bMSP\b"
    r"|\bself[- ]?insur\w*)"
)

# Things that match on 'mutual'/'assurance' but are not carriers.
INSURANCE_NAME_EXCLUDE = (
    r"(?i)(\bmutual fund\b|\bcredit union\b|\bsavings\b|\bmutual of america "
    r"retirement\b|\bagency services llc\b|\btitle agency\b)"
)


# Narrow carrier-only pattern: used as a fallback for is_carrier when the NAICS
# code is missing. Deliberately excludes agency / adjusting / brokerage words.
CARRIER_NAME_HINT = (
    r"(?i)(\binsurance\b|\binsurers?\b|\bassurance\b|\bcasualty\b"
    r"|\bindemnity\b|\bunderwriters?\b|\breciprocal\b|\bRRG\b"
    r"|\breinsur\w*|\bins\s+co\b)"
)


def select_seeds(nodes: pd.DataFrame,
                 naics_prefixes: Sequence[str] = INSURANCE_NAICS_PREFIXES,
                 naics_exact: Iterable[str] | None = None,
                 name_regex: str | None = INSURANCE_NAME_STRONG,
                 exclude_regex: str | None = INSURANCE_NAME_EXCLUDE,
                 require_both: bool = False) -> pd.DataFrame:
    """Return the seed node table with a `seed_reason` column.

    seed_reason in {'naics', 'name', 'both'} — keep it, it is the single most
    useful column when you eyeball precision later. Nodes matched only by name
    are exactly the population where NAICS is missing or wrong.
    """
    naics = nodes["naics_code"].fillna("")
    m_naics = pd.Series(False, index=nodes.index)
    for p in naics_prefixes or []:
        m_naics |= naics.str.startswith(p)
    if naics_exact:
        m_naics |= naics.isin(set(naics_exact))

    if name_regex:
        m_name = nodes["name"].fillna("").str.contains(name_regex, regex=True)
    else:
        m_name = pd.Series(False, index=nodes.index)

    if exclude_regex:
        m_excl = nodes["name"].fillna("").str.contains(exclude_regex, regex=True)
    else:
        m_excl = pd.Series(False, index=nodes.index)

    keep = (m_naics & m_name) if require_both else (m_naics | m_name)
    keep &= ~(m_excl & ~m_naics)          # a NAICS match survives the exclusion

    reason = np.where(m_naics & m_name, "both",
                      np.where(m_naics, "naics", "name"))
    out = nodes.loc[keep].copy()
    out["seed_reason"] = reason[keep.values]
    # A missing NAICS must not silently demote a real carrier out of the core:
    # fall back to a narrow carrier-name pattern only when the code is absent.
    code = out["naics_code"].fillna("")
    hint = out["name"].fillna("").str.contains(CARRIER_NAME_HINT, regex=True)
    out["is_carrier"] = code.str.startswith("5241") | ((code == "") & hint)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Ego / induced subgraph extraction
# --------------------------------------------------------------------------- #
def k_hop_nodes(edges: pd.DataFrame,
                seeds: Sequence[int],
                k: int = 1,
                n_nodes: int | None = None,
                direction: str = "both") -> np.ndarray:
    """Node ids within k undirected (or directed) hops of the seed set."""
    s = edges["s"].to_numpy()
    d = edges["d"].to_numpy()
    n = n_nodes or int(max(s.max(), d.max())) + 1
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(seeds, dtype=np.int64)] = True

    for _ in range(k):
        if direction == "out":
            sel = mask[s]
            new = d[sel]
        elif direction == "in":
            sel = mask[d]
            new = s[sel]
        else:
            sel_o, sel_i = mask[s], mask[d]
            new = np.concatenate([d[sel_o], s[sel_i]])
        mask[new] = True
    return np.flatnonzero(mask).astype("int32")


def ego_subgraph(edges: pd.DataFrame,
                 seeds: Sequence[int],
                 k: int = 1,
                 n_nodes: int | None = None,
                 direction: str = "both") -> tuple[pd.DataFrame, np.ndarray]:
    """Induced subgraph on the k-hop neighbourhood of `seeds`.

    'Induced' matters: it keeps neighbour-to-neighbour edges (e.g. body shop ->
    salvage yard), which is where a lot of the interesting structure lives.
    """
    keep_nodes = k_hop_nodes(edges, seeds, k=k, n_nodes=n_nodes,
                             direction=direction)
    n = n_nodes or int(max(edges["s"].max(), edges["d"].max())) + 1
    mask = np.zeros(n, dtype=bool)
    mask[keep_nodes] = True
    sub = edges[mask[edges["s"].to_numpy()] & mask[edges["d"].to_numpy()]].copy()
    return sub.reset_index(drop=True), keep_nodes


# --------------------------------------------------------------------------- #
# 7. Subrogation analytics
# --------------------------------------------------------------------------- #
def _seed_mask(seeds: Sequence[int], n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[np.asarray(seeds, dtype=np.int64)] = True
    return m


def carrier_core(edges: pd.DataFrame,
                 seeds: Sequence[int],
                 n_nodes: int) -> pd.DataFrame:
    """Edges where *both* endpoints are insurance seeds.

    Premium flows and claim payments leave the insurance set; subrogation
    settles inside it. This is the candidate pool.
    """
    m = _seed_mask(seeds, n_nodes)
    return edges[m[edges["s"].to_numpy()] & m[edges["d"].to_numpy()]].copy()


def bilateral_pairs(core: pd.DataFrame) -> pd.DataFrame:
    """Collapse directed carrier-to-carrier edges into unordered pairs.

    Subrogation is structurally two-way: over a year, each carrier is sometimes
    the recovering party and sometimes the paying party. One-way flows between
    carriers are more likely reinsurance, fronting, or commissions.
    """
    a = np.minimum(core["s"].to_numpy(), core["d"].to_numpy())
    b = np.maximum(core["s"].to_numpy(), core["d"].to_numpy())
    df = core.assign(a=a, b=b, fwd=(core["s"].to_numpy() == a))

    def _agg(sub: pd.DataFrame, tag: str) -> pd.DataFrame:
        return (sub.groupby(["a", "b"], sort=False)
                   .agg(**{f"amt_{tag}": ("amount", "sum"),
                           f"vol_{tag}": ("volume", "sum"),
                           f"rels_{tag}": ("n_rels", "sum"),
                           f"cv_{tag}": ("amt_cv", "mean"),
                           f"ticket_{tag}": ("avg_ticket", "mean")}))

    fwd = _agg(df[df["fwd"]], "ab")
    bwd = _agg(df[~df["fwd"]], "ba")
    out = fwd.join(bwd, how="outer").fillna(
        {"amt_ab": 0, "amt_ba": 0, "vol_ab": 0, "vol_ba": 0,
         "rels_ab": 0, "rels_ba": 0}).reset_index()

    out["gross"] = out["amt_ab"] + out["amt_ba"]
    out["net"] = (out["amt_ab"] - out["amt_ba"]).abs()
    lo = np.minimum(out["amt_ab"], out["amt_ba"])
    hi = np.maximum(out["amt_ab"], out["amt_ba"]).replace(0, np.nan)
    out["reciprocity"] = (lo / hi).fillna(0.0)
    out["netting_ratio"] = 1 - (out["net"] / out["gross"].replace(0, np.nan))
    out["rels_max"] = out[["rels_ab", "rels_ba"]].max(axis=1)
    out["vol_total"] = out["vol_ab"] + out["vol_ba"]
    out["ticket"] = out["gross"] / out["vol_total"].replace(0, np.nan)
    out["cv"] = out[["cv_ab", "cv_ba"]].mean(axis=1)
    out["bidirectional"] = (out["amt_ab"] > 0) & (out["amt_ba"] > 0)
    return out


def score_subrogation(pairs: pd.DataFrame,
                      n_months: int,
                      ticket_lo: float = 1_000.0,
                      ticket_hi: float = 100_000.0,
                      weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Heuristic 0-1 subrogation likelihood for a carrier-carrier pair.

    Components (all 0-1):
      persistence  months present / months observed  — subro is a standing,
                   near-monthly settlement relationship, not a one-off deal
      reciprocity  min(A->B, B->A) / max(...)        — both carriers recover
      steadiness   1 / (1 + CV of monthly amount)    — smooth run-rate
      ticket_fit   log-band membership in [lo, hi]   — subro demands sit well
                   below reinsurance treaties and well above claim payments
      volume_fit   log-scaled transaction count      — many items, not one wire

    These weights are *priors*. Calibrate them the moment you can get even 20
    confirmed subrogation relationships from the insurance TM relationship team.
    """
    w = {"persistence": 0.30, "reciprocity": 0.30, "steadiness": 0.10,
         "ticket_fit": 0.20, "volume_fit": 0.10}
    if weights:
        w.update(weights)

    p = pairs.copy()
    p["persistence"] = (p["rels_max"] / float(n_months)).clip(0, 1)
    p["steadiness"] = 1.0 / (1.0 + p["cv"].fillna(1.0))

    t = p["ticket"].replace(0, np.nan)
    lt, llo, lhi = np.log10(t), np.log10(ticket_lo), np.log10(ticket_hi)
    centre, halfwidth = (llo + lhi) / 2, (lhi - llo) / 2
    p["ticket_fit"] = np.exp(-((lt - centre) / halfwidth) ** 2).fillna(0.0)

    p["volume_fit"] = (np.log10(p["vol_total"].clip(lower=1)) / 3.0).clip(0, 1)

    p["subro_score"] = sum(w[k] * p[k] for k in w)
    p.loc[~p["bidirectional"], "subro_score"] *= 0.5   # one-way = weak evidence
    return p.sort_values("subro_score", ascending=False)


def flag_affiliates(pairs: pd.DataFrame,
                    nodes: pd.DataFrame,
                    threshold: float = 0.85) -> pd.DataFrame:
    """Flag pairs that look like the *same insurance group*.

    Intra-group transfers ('XYZ Insurance Co' <-> 'XYZ Indemnity Co') are
    bidirectional, persistent and steady — they will otherwise sit at the top of
    the subrogation ranking. Reuses the first-token blocking + Jaro-Winkler
    approach from the self-payment pipeline; falls back to difflib.
    """
    try:
        from rapidfuzz.distance import JaroWinkler
        sim = JaroWinkler.similarity
    except ImportError:                                     # pragma: no cover
        try:
            from jellyfish import jaro_winkler_similarity as sim
        except ImportError:
            from difflib import SequenceMatcher
            def sim(x, y):  # noqa: E306
                return SequenceMatcher(None, x, y).ratio()
            warnings.warn("rapidfuzz/jellyfish not found — using difflib")

    lut = nodes.set_index("node")["name_clean"]
    na = pairs["a"].map(lut).fillna("")
    nb = pairs["b"].map(lut).fillna("")
    out = pairs.copy()
    out["name_a"], out["name_b"] = na.values, nb.values
    out["name_sim"] = [sim(x, y) for x, y in zip(na, nb)]
    first_a = na.str.split().str[0].fillna("")
    first_b = nb.str.split().str[0].fillna("")
    out["same_first_token"] = (first_a == first_b) & (first_a != "")
    out["likely_affiliate"] = (out["name_sim"] >= threshold) | out["same_first_token"]
    return out


def counterparty_naics_profile(ego: pd.DataFrame,
                               seeds: Sequence[int],
                               nodes: pd.DataFrame,
                               n_nodes: int,
                               side: str = "out",
                               level: str = "naics4",
                               top: int = 25) -> pd.DataFrame:
    """Who the insurance nodes pay ('out') or get paid by ('in'), by NAICS.

    The payee profile is the claim-disbursement footprint (body shops, medical,
    legal, restoration). The payer profile is premium + recovery inflow.
    """
    m = _seed_mask(seeds, n_nodes)
    s, d = ego["s"].to_numpy(), ego["d"].to_numpy()
    if side == "out":
        sel = m[s] & ~m[d]
        cp = d[sel]
    else:
        sel = ~m[s] & m[d]
        cp = s[sel]

    sub = ego[sel].copy()
    lut = nodes.set_index("node")
    sub["cp"] = cp
    sub[level] = sub["cp"].map(lut[level])
    sub["desc"] = sub["cp"].map(lut["naics_desc"])

    prof = (sub.groupby(level, dropna=False)
               .agg(counterparties=("cp", "nunique"),
                    edges=("cp", "size"),
                    amount=("amount", "sum"),
                    volume=("volume", "sum"),
                    avg_ticket=("avg_ticket", "median"),
                    example=("desc", "first"))
               .sort_values("amount", ascending=False))
    prof["amount_share"] = prof["amount"] / prof["amount"].sum()
    return prof.head(top).reset_index()


def shared_counterparties(ego: pd.DataFrame,
                          seeds: Sequence[int],
                          nodes: pd.DataFrame,
                          n_nodes: int,
                          side: str = "out",
                          min_seeds: int = 2,
                          top: int = 100) -> pd.DataFrame:
    """Bipartite projection: counterparties touched by >= min_seeds insurers.

    A repair shop, clinic or law firm paid by many carriers is the *anchor* of a
    subrogation triangle (both carriers touched the same loss event). It is also
    the classic staged-accident-ring signal when the same small cluster of
    providers keeps reappearing across carriers.
    """
    m = _seed_mask(seeds, n_nodes)
    s, d = ego["s"].to_numpy(), ego["d"].to_numpy()
    sel = (m[s] & ~m[d]) if side == "out" else (~m[s] & m[d])
    sub = ego[sel].copy()
    sub["cp"] = d[sel] if side == "out" else s[sel]
    sub["seed"] = s[sel] if side == "out" else d[sel]

    agg = (sub.groupby("cp")
              .agg(n_seeds=("seed", "nunique"),
                   amount=("amount", "sum"),
                   volume=("volume", "sum"),
                   rels=("n_rels", "sum"),
                   avg_ticket=("avg_ticket", "median"))
              .query("n_seeds >= @min_seeds")
              .sort_values(["n_seeds", "amount"], ascending=False))
    lut = nodes.set_index("node")
    for c in ("node_id", "name", "naics_code", "naics_desc", "naics4"):
        agg[c] = agg.index.map(lut[c])
    agg["concentration"] = agg["amount"] / agg["n_seeds"]
    return agg.head(top).reset_index()


def pass_through_intermediaries(ego: pd.DataFrame,
                                carriers: Sequence[int],
                                nodes: pd.DataFrame,
                                n_nodes: int,
                                min_carriers_each_side: int = 2,
                                exclude_nodes: Sequence[int] | None = None,
                                top: int = 100) -> pd.DataFrame:
    """Nodes that sit *between* carriers: carrier -> X -> carrier.

    Pass the **carrier subset (5241*) only** as `carriers`, not the full seed
    set — otherwise the 5242* service entities you are hunting for (arbitration
    forums, recovery vendors, TPAs) sit inside the anchor set and can never be
    found as intermediaries.

    This is the recovery-vendor / arbitration-forum / clearinghouse motif. When
    subrogation is outsourced or netted, the bilateral carrier-carrier edge
    disappears and reappears as two edges through X — so this catches exactly
    the population the carrier_core() view is blind to.
    """
    m = _seed_mask(carriers, n_nodes)
    if exclude_nodes is not None and len(exclude_nodes):
        ego = ego[~ego["s"].isin(exclude_nodes) & ~ego["d"].isin(exclude_nodes)]
    s, d = ego["s"].to_numpy(), ego["d"].to_numpy()

    inbound = ego[m[s] & ~m[d]].copy()          # carrier -> X
    inbound["x"] = d[m[s] & ~m[d]]
    outbound = ego[~m[s] & m[d]].copy()         # X -> carrier
    outbound["x"] = s[~m[s] & m[d]]

    a = inbound.groupby("x").agg(in_carriers=("s", "nunique"),
                                 in_amt=("amount", "sum"),
                                 in_vol=("volume", "sum"))
    b = outbound.groupby("x").agg(out_carriers=("d", "nunique"),
                                  out_amt=("amount", "sum"),
                                  out_vol=("volume", "sum"))
    j = a.join(b, how="inner")
    j = j[(j["in_carriers"] >= min_carriers_each_side) &
          (j["out_carriers"] >= min_carriers_each_side)]

    j["throughput"] = np.minimum(j["in_amt"], j["out_amt"])
    j["balance"] = j["throughput"] / np.maximum(j["in_amt"], j["out_amt"])
    j["carriers_touched"] = j["in_carriers"] + j["out_carriers"]
    lut = nodes.set_index("node")
    for c in ("node_id", "name", "naics_code", "naics_desc"):
        j[c] = j.index.map(lut[c])
    return (j.sort_values(["balance", "throughput"], ascending=False)
             .head(top).reset_index())


# --------------------------------------------------------------------------- #
# 8. Pass 2 — monthly panel for the ego subgraph only
# --------------------------------------------------------------------------- #
def monthly_panel(months: Sequence[str],
                  keep_nodes: Sequence[int],
                  index: NodeIndex,
                  data_dir: str = "../data",
                  prefix: str = "cust_",
                  verbose: bool = True) -> pd.DataFrame:
    """Re-read the snapshots keeping only edges inside the ego node set.

    Cheap (the ego set is small) and it gives you the time axis that the
    aggregate throws away — needed for lag motifs, drift and seasonality.
    """
    n = len(index)
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(keep_nodes, dtype=np.int64)] = True
    frames = []
    for i, m in enumerate(months):
        df = load_month(month_path(data_dir, m, prefix))
        df["s"] = df["source"].map(index._map)
        df["d"] = df["dest"].map(index._map)
        df = df[df["s"].notna() & df["d"].notna()]
        s = df["s"].astype("int32").to_numpy()
        d = df["d"].astype("int32").to_numpy()
        df = df[mask[s] & mask[d]]
        keep = df[["s", "d", "amount", "volume"]].copy()
        keep["s"] = keep["s"].astype("int32")
        keep["d"] = keep["d"].astype("int32")
        keep["month"] = m
        keep["mi"] = np.int16(i)
        frames.append(keep)
        if verbose:
            print(f"  [{i + 1}/{len(months)}] {m}: {len(keep):,} ego edges")
    return pd.concat(frames, ignore_index=True)


def lagged_xcorr(x: Sequence[float], y: Sequence[float],
                 max_lag: int = 6) -> pd.DataFrame:
    """Correlation of y against x shifted forward by 0..max_lag months.

    Use it for the recovery-lag motif: x = carrier A's payments to a shared
    vendor, y = carrier B -> carrier A flow. A peak at lag 1-6 is the
    fingerprint of a demand-and-recover cycle rather than a coincidental
    correlation.
    """
    x = pd.Series(np.asarray(x, dtype="float64"))
    y = pd.Series(np.asarray(y, dtype="float64"))
    rows = []
    for L in range(max_lag + 1):
        xi, yi = x.iloc[:len(x) - L], y.iloc[L:]
        if len(xi) >= 3 and xi.std() > 0 and yi.std() > 0:
            r = float(np.corrcoef(xi.to_numpy(), yi.to_numpy())[0, 1])
        else:
            r = np.nan
        rows.append({"lag": L, "corr": r, "n": len(xi)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 9. Graph construction (small subgraphs only)
# --------------------------------------------------------------------------- #
def to_networkx(edges: pd.DataFrame,
                nodes: pd.DataFrame | None = None,
                edge_attrs: Sequence[str] = ("amount", "volume", "n_rels",
                                             "avg_ticket", "amt_cv"),
                node_attrs: Sequence[str] = ("node_id", "name", "naics_code",
                                             "naics_desc", "naics2", "naics4",
                                             "naics_sector"),
                max_edges: int = 3_000_000):
    """Build a DiGraph. Guardrail: refuses very large edge tables on purpose."""
    import networkx as nx
    if len(edges) > max_edges:
        raise MemoryError(
            f"{len(edges):,} edges > max_edges={max_edges:,}. Filter first "
            f"(ego subgraph / percentile ablation) or use to_networkit().")
    attrs = [c for c in edge_attrs if c in edges.columns]
    G = nx.from_pandas_edgelist(edges, source="s", target="d",
                                edge_attr=attrs, create_using=nx.DiGraph)
    if nodes is not None:
        cols = [c for c in node_attrs if c in nodes.columns]
        sub = nodes[nodes["node"].isin(G.nodes)].set_index("node")[cols]
        import networkx as _nx
        _nx.set_node_attributes(G, sub.to_dict("index"))
    return G


def to_networkit(edges: pd.DataFrame, weight: str = "amount"):
    """Compact CSR graph for whole-aggregate metrics. Returns (G, id_map).

    Only reach for this if you want global centralities on the full aggregate
    without going to cuGraph — networkit stores ~10x lighter than networkx.
    """
    import networkit as nk
    codes = pd.unique(pd.concat([edges["s"], edges["d"]], ignore_index=True))
    id_map = {int(c): i for i, c in enumerate(codes)}
    G = nk.graph.Graph(len(id_map), weighted=True, directed=True)
    for s, d, w in zip(edges["s"], edges["d"], edges[weight]):
        G.addEdge(id_map[int(s)], id_map[int(d)], float(w))
    return G, id_map


# --------------------------------------------------------------------------- #
# 10. Convenience summary
# --------------------------------------------------------------------------- #
def describe_ego(ego: pd.DataFrame, keep_nodes: np.ndarray,
                 seeds: Sequence[int], agg: AggregateResult) -> None:
    n_seed = len(seeds)
    print(f"aggregate      : {len(agg.nodes):,} nodes / {len(agg.edges):,} edges "
          f"over {agg.n_months} months")
    print(f"seeds          : {n_seed:,} insurance nodes")
    print(f"ego (1-hop)    : {len(keep_nodes):,} nodes / {len(ego):,} edges "
          f"({len(ego) / max(len(agg.edges), 1):.2%} of aggregate)")
    core = carrier_core(ego, seeds, len(agg.nodes))
    print(f"insurer-insurer: {len(core):,} directed edges, "
          f"${core['amount'].sum():,.0f}")
