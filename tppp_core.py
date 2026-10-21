"""
TPPP Prospect Finder: which outside companies move money like third-party payment and payroll processors.
No Streamlit inside, so the dev team can wrap it in FastAPI as is.

  1. Universe: counterparties (ent_l2) with eligible metrics in the period, their behaviour averaged over the months
     they were active (tppp_universe_sql), then a screen: organisations on at least one side, no government / card
     network / insurer / generic descriptor names.
  2. Examples (label 1): the registered TPPPs' accounts at other banks (name match to vector_customer TPPPs), known
     processor names (seed list), and counterparties reviewers marked "processor". Everything else is label 0;
     reviewer "not a processor" never becomes an example.
  3. Model: logistic regression on percentile ranks of the behaviour features. Every counterparty is scored by a
     model that did not see it (out-of-fold, examples of one brand kept in the same fold), so an example's score is
     an honest test: the share of examples landing in the top 1% is the reliability figure shown in the app.
  4. Explanations: each feature's contribution (weight x centred percentile) gives "why"; the nearest example in
     the weighted feature space gives "most similar example".
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import prospect_core as pc

# ================================================================== configuration
T_VECTOR = "dsihd01p_dsi.vector_customer"
TPPP_TYPE = "Third-Party Payment/Payroll Processor"
MIN_CLIENTS = 5             # PNC clients per active month, either way, to be scored
MIN_MONTHS = 2              # active months in the period
MIN_ORG_SHARE = 0.2         # at least 20% of its money with organisations on one side (processors serve businesses)
N_FOLDS = 5
C_REG = 0.5                 # logistic regression regularisation (smaller = smoother weights)
SEED = 7
VERDICT_PATH = os.environ.get("TPPP_VERDICTS", "data/tppp_verdicts.csv")
VERDICTS = ["processor", "not a processor", "unsure"]
SOURCES = {"registered": "Registered TPPP's account at another bank",
           "known": "Known processor name",
           "confirmed": "Marked 'processor' by a reviewer"}

KNOWN_PROC_RE = (r"\b(ADP|PAYCHEX|PAYCOM|PAYLOCITY|DAYFORCE|CERIDIAN|GUSTO|PAYCOR|ISOLVED|TRINET|JUSTWORKS|INSPERITY|"
                 r"RIPPLING|STRIPE|PAYPAL|BRAINTREE|WORLDPAY|FISERV|FIRST DATA|GLOBAL PAYMENTS|HEARTLAND PAYMENT|TSYS|"
                 r"ELAVON|PAYMENTECH|AUTHORIZE NET|SHIFT4|NUVEI|ADYEN|PAYSAFE|REPAY|DWOLLA|BILL COM|AVIDXCHANGE|"
                 r"CLICKPAY|APPFOLIO|YARDI|PAYLEASE|ZEGO|REALPAGE)\b")     # seed list; edit freely
EXCL_RE = (r"\b(?:VISA|MASTERCARD|AMERICAN EXPRESS|AMEX|DISCOVER|IRS|TREASURY|TREAS|SSA|STATE OF|DEPT OF|"
           r"DEPARTMENT OF|COUNTY|CITY OF|COMMONWEALTH|INSURANCE|ASSURANCE|MUTUAL|FANNIE MAE|FREDDIE MAC|"
           r"FEDERAL HOME LOAN|CITICARD|CITI CARDS?|COMENITY|SYNCHRONY|CAPITAL ONE|CHASE CARD|CARD SERVICES|"
           r"HONDA|HYUNDAI|TOYOTA|NISSAN|KIA MOTORS|FORD CREDIT|GM FINANCIAL|STATE FARM|GEICO|PROGRESSIVE|"
           r"ALLSTATE|NATIONWIDE|LIBERTY MUTUAL|USAA|PNC)\b"
           r"|SALESTAX|SALES TAX|USETAX|USE TAX|TAXPYMT|DEPT REV|\bCAS\b|CASUALTY|\bINS\b|INS CO\b")
LEGAL = ["INCORPORATED", "CORPORATION", "COMPANY", "LIMITED", "INC", "LLC", "LLP", "LP", "LTD", "CORP", "CO", "PC",
         "PLLC", "PA", "PLC", "NA", "THE", "DBA"]
BRAND_EXCLUDE = {"TRUIST", "SUMITOMO"}      # registered TPPPs whose first word is not a processor brand
BRAND_BANK_RE = r"\b(?:BANK|BANKING|BANCORP|BANCSHARES|FINANCIAL CORP|SAVINGS|CREDIT UNION|FSB)\b"
COMMON_FIRST = {"FIRST", "AMERICAN", "NATIONAL", "UNITED", "GLOBAL", "GENERAL", "INTERNATIONAL", "CENTRAL", "NORTH",
                "SOUTH", "EAST", "WEST", "CITY", "COUNTY", "STATE", "PREMIER", "PACIFIC", "ATLANTIC", "SUMMIT"}

# (alias in the universe query, source column, plain label, kind). size = scale of activity, shape = how it behaves.
FEATURES = [
    ("amt_in", "amt_in", "money from PNC clients", "size"),
    ("amt_out", "amt_out", "money to PNC clients", "size"),
    ("clients_in", "n_cust_in", "PNC clients paying it", "size"),
    ("clients_out", "n_cust_out", "PNC clients it pays", "size"),
    ("tm_in", "n_cust_tm_in", "TM clients paying it", "size"),
    ("tm_out", "n_cust_tm_out", "TM clients it pays", "size"),
    ("f_rec_in", "rec_standing_in", "recurring payers", "size"),
    ("f_rec_out", "rec_standing_out", "recurring payees", "size"),
    ("f_org_in", "org_share_in", "share of money in from organisations", "shape"),
    ("f_org_out", "org_share_out", "share of money out to organisations", "shape"),
    ("f_tmsh_in", "tm_share_in", "share of money in from TM clients", "shape"),
    ("f_tmsh_out", "tm_share_out", "share of money out to TM clients", "shape"),
    ("f_top1_in", "top1_share_in", "largest payer's share", "shape"),
    ("f_top1_out", "top1_share_out", "largest payee's share", "shape"),
    ("f_dep_in", "dep_wavg_in", "its share of its payers' outside payments", "shape"),
    ("f_dep_out", "dep_wavg_out", "its share of its payees' outside receipts", "shape"),
    ("f_two_way", "two_way_ratio", "two-way balance of money", "shape"),
    ("f_days_in", "active_days_in", "days with money in", "shape"),
    ("f_days_out", "active_days_out", "days with money out", "shape"),
    ("f_ticket_in", "avg_ticket_in", "average payment in", "shape"),
    ("f_ticket_out", "avg_ticket_out", "average payment out", "shape"),
    ("f_ret_in", "retention_in", "payer retention", "shape"),
    ("f_ret_out", "retention_out", "payee retention", "shape"),
    ("f_newamt_in", "new_amt_share_in", "money in from new payers", "shape"),
    ("f_newamt_out", "new_amt_share_out", "money out to new payees", "shape"),
    ("f_ach_in", "share_ach_in", "ACH share of money in", "shape"),
    ("f_ach_out", "share_ach_out", "ACH share of money out", "shape"),
    ("f_wire_in", "share_wire_in", "wire share of money in", "shape"),
    ("f_wire_out", "share_wire_out", "wire share of money out", "shape"),
    ("f_rails_in", "n_rails_in", "payment rails in", "shape"),
    ("f_rails_out", "n_rails_out", "payment rails out", "shape"),
    ("f_nfi", "n_fi", "banks holding its accounts", "shape"),
    ("f_topfi", "top_fi_share", "share at its main bank", "shape"),
    ("f_nids", "n_ids_active", "account ids active", "shape"),
]
DERIVED = [("ach_orig", "months as an ACH originator", "shape"),
           ("role_flip", "organisations in, individuals out", "shape")]
LABEL = {a: lab for a, _, lab, _ in FEATURES} | {a: lab for a, lab, _ in DERIVED}
KIND = {a: k for a, _, _, k in FEATURES} | {a: k for a, _, k in DERIVED}


# ================================================================== names
def name_norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(s or "").upper())).strip()


_LEGAL_RE = re.compile(r"\b(" + "|".join(LEGAL) + r")\b")


def name_key(s: str) -> str | None:
    """Same idea as the entity map's key: upper case, legal words removed, single letters joined (L L C -> LLC)."""
    k = name_norm(s)
    k = re.sub(r"\b([A-Z]) (?=[A-Z]\b)", r"\1", k)
    k = re.sub(r"\s+", " ", _LEGAL_RE.sub(" ", k)).strip()
    return k if len(k) >= 5 else None


def norm_pwr(x) -> str | None:
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    d = re.sub(r"\D", "", s)
    return d.zfill(10) if d else None


# ================================================================== queries
def tppp_pwr_sql() -> str:
    return (f"SELECT DISTINCT CAST(cust_pwr_id AS STRING) AS pwr FROM {T_VECTOR} "
            f"WHERE customer_type = '{TPPP_TYPE}' AND load_date = (SELECT MAX(load_date) FROM {T_VECTOR})")


def tppp_customers_sql(pwrs) -> str:
    """Registered TPPP customers; power ids tried with and without leading zeros."""
    forms = sorted({p for x in pwrs for p in (x, x.lstrip("0")) if p})
    return (f"SELECT mdm_id, CAST(customer_pwr_id AS STRING) AS customer_pwr_id, "
            f"CAST(master_pwr_id AS STRING) AS master_pwr_id, customer_name FROM {pc.T_CDIM} "
            f"WHERE CAST(customer_pwr_id AS STRING) IN ({pc._lit(forms)})")


def tppp_universe_sql(per, cols: set[str], dim_cols: set[str] | None) -> str:
    """One row per counterparty: behaviour averaged over the months it was active in the period (n_cust_all > 0)."""
    need = {"ent_id", "month", "ent_name", "name_class", "fi_dom", "eligible", "amt_in", "amt_out", "n_cust_in",
            "n_cust_out", "n_cust_all"}
    miss = need - cols
    if miss:
        raise ValueError(f"counterparty metrics lack {sorted(miss)}")
    M, p0, end = pc.Q["month"], per[0], per[-1]
    act = "n_cust_all > 0"
    avg = lambda expr, a: f"AVG(CASE WHEN {act} THEN {expr} END) AS {a}"
    aggs = [f"SUM(CASE WHEN {act} THEN 1 ELSE 0 END) AS months_active",
            f"MAX(CASE WHEN {act} THEN {M} END) AS last_month",
            avg("CAST(n_cust_all AS DOUBLE)", "clients")]
    aggs += [avg(f"CAST({c} AS DOUBLE)", a) for a, c, _, _ in FEATURES if c in cols]
    if "cpty_type" in cols:
        aggs.append(avg("CASE WHEN cpty_type = 'NON_PNC_ACH_ORIGINATOR' THEN 1.0 ELSE 0.0 END", "ach_orig"))
    last = ["t.ent_name", "t.fi_dom"] + (["CAST(t.n_fi AS DOUBLE) AS n_fi"] if "n_fi" in cols else [])
    dim = [f"d.{c}" for c in ("pnc_match_n",) if dim_cols and c in dim_cols]
    a_out = [x.rsplit(" AS ", 1)[1] for x in aggs if not x.endswith(" last_month")]
    sel = ", ".join(["a.ent_id"] + [f"a.{c}" for c in a_out]
                    + [f"l.{c.rsplit(' AS ', 1)[-1].replace('t.', '')}" for c in last] + dim)
    return f"""
WITH a AS (
  SELECT ent_id, {', '.join(aggs)}
  FROM {pc.T_CPTY_MET}
  WHERE {M} BETWEEN '{p0}' AND '{end}' AND eligible AND name_class IN ({pc._lit(pc.CLASSES)})
  GROUP BY ent_id
  HAVING SUM(CASE WHEN {act} THEN 1 ELSE 0 END) >= {MIN_MONTHS}
     AND AVG(CASE WHEN {act} THEN CAST(n_cust_all AS DOUBLE) END) >= {float(MIN_CLIENTS)}
),
l AS (
  SELECT t.ent_id, {', '.join(last)}
  FROM {pc.T_CPTY_MET} t JOIN a ON t.ent_id = a.ent_id AND t.{M} = a.last_month
  WHERE t.{M} BETWEEN '{p0}' AND '{end}'
)
SELECT {sel}
FROM a JOIN l ON a.ent_id = l.ent_id
{f"LEFT JOIN {pc.T_ENT_DIM} d ON a.ent_id = d.ent_id" if dim else ""}
ORDER BY a.clients DESC
LIMIT {int(pc.MAX_ROWS)}"""


# ================================================================== universe and examples
def prepare_universe(df: pd.DataFrame, tppp: pd.DataFrame) -> pd.DataFrame:
    """Types, flags, the screen, and the two automatic example sources (registered TPPPs' accounts, known names)."""
    df = df.drop_duplicates("ent_id").reset_index(drop=True)
    feats = [a for a, _, _, _ in FEATURES] + ["ach_orig", "clients", "months_active", "n_fi", "pnc_match_n"]
    df = pc._num(df, feats)
    df["ent_name"] = df.ent_name.fillna("").astype(str).str.upper().str.strip()
    norm = df.ent_name.map(name_norm)
    df["amt"] = df.amt_in.fillna(0) + df.amt_out.fillna(0)
    share_in = df.amt_in.fillna(0) / df.amt.where(df.amt > 0)
    df["flow"] = np.select([share_in >= 0.8, share_in <= 0.2], pc.FLOWS[:2], pc.FLOWS[2])
    df["tm_all"] = df.get("tm_in", 0) + df.get("tm_out", 0) if "tm_in" in df else np.nan
    df["pnc_account"] = (np.where(df.pnc_match_n.fillna(0) > 0, "Yes", "No") if "pnc_match_n" in df else "Unknown")
    if {"f_org_in", "f_org_out"} <= set(df.columns):
        df["role_flip"] = df.f_org_in.fillna(0) - df.f_org_out.fillna(0)
    # flags
    brand = norm.str.extract(KNOWN_PROC_RE, expand=False)
    df["known_proc"] = brand.notna()
    df["known_brand"] = brand
    df["excluded"] = norm.str.contains(EXCL_RE, regex=True)
    df["generic"] = df.ent_name.map(pc.is_generic)
    org_max = df[[c for c in ("f_org_in", "f_org_out") if c in df]].max(axis=1) if "f_org_in" in df else 1.0
    df["screened"] = (org_max >= MIN_ORG_SHARE) & ~df.excluded & ~df.generic
    # registered TPPPs' accounts at other banks: the entity's key equals a TPPP key or starts with a brand key
    df["tppp_match"] = None
    if tppp is not None and len(tppp):
        keys = {}
        for nm in tppp.customer_name.dropna():
            k = name_key(nm)
            if not k or re.search(BRAND_BANK_RE, name_norm(nm)):
                continue
            first = k.split()[0]
            brand_ok = (first not in BRAND_EXCLUDE and first not in COMMON_FIRST and len(first) >= 4)
            keys[k] = (nm, brand_ok)
        ek = df.ent_name.map(name_key).fillna("")
        hit = []
        for k in ek:
            m = keys.get(k)
            if m is None and k:
                w = k.split()
                for j in range(min(3, len(w) - 1), 0, -1):          # longest brand prefix first
                    cand = keys.get(" ".join(w[:j]))
                    if cand and cand[1]:
                        m = cand
                        break
            hit.append(m[0] if m else None)
        df["tppp_match"] = hit
    df["registered"] = df.tppp_match.notna()
    return df


def load_verdicts(path: str = VERDICT_PATH) -> pd.DataFrame:
    cols = ["at", "ent_id", "ent_name", "verdict", "note", "reviewer"]
    try:
        v = pd.read_csv(path, dtype=str).fillna("")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)
    return v[[c for c in cols if c in v.columns]]


def latest_verdicts(v: pd.DataFrame) -> pd.DataFrame:
    return v.sort_values("at").drop_duplicates("ent_id", keep="last") if len(v) else v


def save_verdict(ent_id: str, ent_name: str, verdict: str, note: str, reviewer: str,
                 path: str = VERDICT_PATH) -> None:
    if verdict not in VERDICTS + ["clear"]:
        raise ValueError("unknown verdict")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["at", "ent_id", "ent_name", "verdict", "note", "reviewer"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), ent_id, ent_name, verdict,
                    note.replace("\n", " ").strip(), reviewer])


# ================================================================== model
def model_features(df: pd.DataFrame, use_size: bool) -> list[str]:
    cand = [a for a, _, _, _ in FEATURES] + [a for a, _, _ in DERIVED]
    return [f for f in cand if f in df and df[f].notna().mean() > 0.05 and (use_size or KIND[f] != "size")]


def _folds(y: np.ndarray, groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Examples of one brand share a fold (so a brand is never in its own training set); others random."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, k, len(y))
    g = pd.unique(groups[y == 1])
    rng.shuffle(g)
    gf = {x: i % k for i, x in enumerate(g)}
    fold[y == 1] = [gf[x] for x in groups[y == 1]]
    return fold


def score(S: pd.DataFrame, feats: list[str], y: np.ndarray, groups: np.ndarray) -> dict:
    """Out-of-fold scores (0-100 percentile within each fold), full-data weights, reliability and explanations."""
    X = S[feats].rank(pct=True).fillna(0.5).to_numpy() - 0.5         # centred percentiles
    n = len(S)
    fold = _folds(y, groups, N_FOLDS, SEED)
    pctl = np.zeros(n)
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if y[tr].sum() < 3 or te.sum() == 0:
            continue
        m = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(X[tr], y[tr])
        d = m.decision_function(X[te])
        pctl[te] = pd.Series(d).rank(pct=True).to_numpy()
    full = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(X, y)
    coef = full.coef_[0]
    contrib = X * coef
    order = np.argsort(-contrib, axis=1)[:, :3]
    pos_rank = X + 0.5
    why = []
    for i in range(n):
        parts = []
        for j in order[i]:
            if contrib[i, j] <= 0.05:
                continue
            p = pos_rank[i, j]
            parts.append(f"{LABEL[feats[j]]} top {max(1, round(100 * (1 - p)))}%" if coef[j] > 0
                         else f"low {LABEL[feats[j]]} (bottom {max(1, round(100 * p))}%)")
        why.append(" · ".join(parts))
    # nearest example in the weighted space (|weights| as feature importance)
    w = np.abs(coef) / (np.abs(coef).sum() or 1)
    Xw = X * np.sqrt(w)
    E = Xw[y == 1]
    e_idx = np.flatnonzero(y == 1)
    near, dist = np.empty(n, dtype=int), np.empty(n)
    for a in range(0, n, 20000):
        blk = Xw[a:a + 20000]
        d2 = (blk ** 2).sum(1)[:, None] - 2 * blk @ E.T + (E ** 2).sum(1)[None, :]
        is_self = np.isin(np.arange(a, a + len(blk)), e_idx)
        if is_self.any():                                            # an example's nearest is another example
            rows = np.flatnonzero(is_self)
            cols = np.searchsorted(e_idx, np.arange(a, a + len(blk))[rows])
            d2[rows, cols] = np.inf
        j = d2.argmin(1)
        near[a:a + len(blk)] = e_idx[j]
        dist[a:a + len(blk)] = np.sqrt(np.maximum(d2[np.arange(len(blk)), j], 0))
    weights = pd.DataFrame({"feature": feats, "label": [LABEL[f] for f in feats], "kind": [KIND[f] for f in feats],
                            "weight": coef}).sort_values("weight", ascending=False)
    pos = pctl[y == 1]
    rel = pd.DataFrame([{"top": f"top {t:g}%", "share of hidden examples found": float((pos >= 1 - t / 100).mean()),
                         "a random list would find": t / 100,
                         "counterparties to review": int(np.ceil(n * t / 100))} for t in (0.5, 1, 2, 5)])
    return {"pctl": 100 * pctl, "why": why, "near": near, "near_dist": dist, "weights": weights, "reliability": rel,
            "X": X, "coef": coef, "feats": feats}


def run_model(U: pd.DataFrame, verdicts: pd.DataFrame, sources: tuple, use_size: bool) -> dict:
    """Examples from the chosen sources, out-of-fold scores, everything the app shows."""
    S = U[U.screened].reset_index(drop=True).copy()
    V = latest_verdicts(verdicts)
    vmap = dict(zip(V.ent_id, V.verdict)) if len(V) else {}
    S["verdict"] = S.ent_id.map(vmap).fillna("")
    S.loc[S.verdict == "clear", "verdict"] = ""
    src = np.full(len(S), "", dtype=object)
    grp = S.ent_id.astype(str).to_numpy().astype(object)
    if "confirmed" in sources:
        m = (S.verdict == "processor").to_numpy()
        src[m] = "confirmed"
    if "known" in sources:
        m = S.known_proc.to_numpy() & (src == "")
        src[m] = "known"
        grp[m] = ("K:" + S.known_brand.fillna("")).to_numpy()[m]
    if "registered" in sources:
        m = S.registered.to_numpy() & (src == "")
        src[m] = "registered"
        grp[m] = ("T:" + S.tppp_match.fillna("").map(lambda s: ((name_key(s) or s or "?").split() or ["?"])[0])).to_numpy()[m]
    src[(S.verdict == "not a processor").to_numpy()] = ""            # reviewers' rejections never teach
    S["example"] = src
    y = (src != "").astype(int)
    if y.sum() < 10:
        raise ValueError(f"only {y.sum()} examples among scored counterparties: switch more example sources on")
    feats = model_features(S, use_size)
    R = score(S, feats, y, grp)
    S["score"] = R["pctl"]
    S["why"] = R["why"]
    S["similar"] = S.ent_name.to_numpy()[R["near"]]
    S["similar_src"] = S.example.to_numpy()[R["near"]]
    order = S.sort_values(["score", "amt"], ascending=False).index                # ties broken by money
    S.loc[order, "rank"] = np.arange(1, len(S) + 1)
    S["rank"] = S["rank"].astype(int)
    return {"S": S, "weights": R["weights"], "reliability": R["reliability"], "n_examples": int(y.sum()),
            "X": R["X"], "coef": R["coef"], "feats": feats,
            "by_source": pd.Series(src[src != ""]).map(SOURCES).value_counts()}


def similar_examples(res: dict, i: int, k: int = 3) -> pd.DataFrame:
    """The k examples nearest to row i of res['S'] in the weighted feature space, with the closest features."""
    S, X, coef, feats = res["S"], res["X"], res["coef"], res["feats"]
    w = np.abs(coef) / (np.abs(coef).sum() or 1)
    ex = np.flatnonzero((S.example != "").to_numpy())
    ex = ex[ex != i]
    d = np.sqrt((((X[ex] - X[i]) ** 2) * w).sum(1))
    top = ex[np.argsort(d)[:k]]
    out = []
    for j in top:
        gap = np.abs(X[j] - X[i]) * (w > np.median(w))
        close = [LABEL[feats[c]] for c in np.argsort(gap) if w[c] > np.median(w)][:3]
        out.append({"example": S.ent_name.iat[j], "why it is an example": SOURCES.get(S.example.iat[j], ""),
                    "closest on": ", ".join(close)})
    return pd.DataFrame(out)


# ================================================================== exports
def enrichment_files(T: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Names only, behind a random-looking row id, for outside research; the key stays inside PNC."""
    rid = T.ent_id.map(lambda x: hashlib.sha256(f"{salt}|{x}".encode()).hexdigest()[:12])
    return (pd.DataFrame({"row_id": rid, "name": T.ent_name}),
            pd.DataFrame({"row_id": rid, "ent_id": T.ent_id, "name": T.ent_name}))
