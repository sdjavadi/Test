"""
TPPP Prospect Finder: which outside companies move money like third-party payment and payroll processors.
No Streamlit inside, so the dev team can wrap it in FastAPI as is.

  1. Universe: counterparties (ent_l2) with eligible metrics in the period, their behaviour averaged over the months
     they were active (tppp_universe_sql), then a screen: organisations on at least one side, no government / card
     network / insurer / generic descriptor names. Optional evidence measures (timing, client overlap) come from
     pkg_tppp_cpty_evidence_monthly when that table exists (built by pkg_tppp_cpty_evidence.ipynb).
  2. Examples (label 1): the registered TPPPs' accounts at other banks (name match to vector_customer TPPPs), known
     processor names (seed list), and counterparties reviewers marked "processor". Everything else is label 0;
     reviewer "not a processor" stays label 0 with extra weight (a firm negative) and never becomes an example.
  3. Model: logistic regression on percentile ranks of the behaviour measures. Every counterparty is scored by a
     model that did not see it (out-of-fold, examples of one brand kept in the same fold), so an example's score is
     an honest test: the share of examples landing in the top 1% is the reliability figure shown in the app.
  4. Explanations: each measure's contribution (weight x centred percentile) gives "why"; the nearest examples in
     the weighted measure space give "most similar example" and the processor type it "looks like".
  5. Value: how much business it does with PNC clients (money and TM clients), as a percentile, kept separate from
     the TPPP score so the two can be read together.
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
T_EVIDENCE = f"{pc.DB}.pkg_tppp_cpty_evidence_monthly"     # optional; built by pkg_tppp_cpty_evidence.ipynb
TPPP_TYPE = "Third-Party Payment/Payroll Processor"
MIN_CLIENTS = 5             # PNC clients per active month, either way, to be scored
MIN_MONTHS = 2              # active months in the period
MIN_ORG_SHARE = 0.2         # at least 20% of its money with organisations on one side (processors serve businesses)
N_FOLDS = 5
C_REG = 0.5                 # logistic regression regularisation (smaller = smoother weights)
NEG_WEIGHT = 5.0            # weight of a reviewer's "not a processor" (a firm negative) vs an ordinary counterparty
K_NEAR = 5                  # nearest examples used for "looks like"
SEED = 7
VERDICT_PATH = os.environ.get("TPPP_VERDICTS", "data/tppp_verdicts.csv")
RESEARCH_PATH = os.environ.get("TPPP_RESEARCH", "data/tppp_research.csv")
RESEARCH_SALT = "pkg-tppp-research"     # row ids of the names file; fixed so results join back in any period
VERDICTS = ["processor", "not a processor", "unsure"]
SOURCES = {"registered": "Registered TPPP's account at another bank",
           "known": "Known processor name",
           "confirmed": "Marked 'processor' by a reviewer"}

KNOWN_PROC_RE = (r"\b(ADP|PAYCHEX|PAYCOM|PAYLOCITY|DAYFORCE|CERIDIAN|GUSTO|PAYCOR|ISOLVED|TRINET|JUSTWORKS|INSPERITY|"
                 r"RIPPLING|STRIPE|PAYPAL|BRAINTREE|WORLDPAY|FISERV|FIRST DATA|GLOBAL PAYMENTS|HEARTLAND PAYMENT|TSYS|"
                 r"ELAVON|PAYMENTECH|AUTHORIZE NET|SHIFT4|NUVEI|ADYEN|PAYSAFE|REPAY|DWOLLA|BILL COM|AVIDXCHANGE|"
                 r"CLICKPAY|APPFOLIO|YARDI|PAYLEASE|ZEGO|REALPAGE)\b")     # seed list; edit freely (and BRAND_TYPE)
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

# ---- processor types: what kind of processor an example is, and so what a prospect "looks like"
PAYROLL, CARD, BILLPAY, PROPERTY, OTHER = ("Payroll and HR", "Card and merchant payments", "Bill pay and payables",
                                           "Property payments", "Other processor")
TYPES = [PAYROLL, CARD, BILLPAY, PROPERTY, OTHER]
BRAND_TYPE = {**{b: PAYROLL for b in ["ADP", "PAYCHEX", "PAYCOM", "PAYLOCITY", "DAYFORCE", "CERIDIAN", "GUSTO",
                                      "PAYCOR", "ISOLVED", "TRINET", "JUSTWORKS", "INSPERITY", "RIPPLING"]},
              **{b: CARD for b in ["STRIPE", "PAYPAL", "BRAINTREE", "WORLDPAY", "FISERV", "FIRST DATA",
                                   "GLOBAL PAYMENTS", "HEARTLAND PAYMENT", "TSYS", "ELAVON", "PAYMENTECH",
                                   "AUTHORIZE NET", "SHIFT4", "NUVEI", "ADYEN", "PAYSAFE", "REPAY"]},
              **{b: BILLPAY for b in ["DWOLLA", "BILL COM", "AVIDXCHANGE"]},
              **{b: PROPERTY for b in ["CLICKPAY", "APPFOLIO", "YARDI", "PAYLEASE", "ZEGO", "REALPAGE"]}}
TYPE_WORDS = [(PAYROLL, r"\b(?:PAYROLL|PEO|HCM|HR|STAFFING|EMPLOYER|WORKFORCE|PAYCARD)\b"),
              (PROPERTY, r"\b(?:RENT|RENTAL|RENTS|PROPERTY|PROPERTIES|RESIDENT|LEASING|HOA)\b"),
              (BILLPAY, r"\b(?:BILL|BILLPAY|BILLING|INVOICE|PAYABLES?|REMIT\w*)\b"),
              (CARD, r"\b(?:MERCHANT|MERCH|CARD|ACQUIR\w*|GATEWAY|PROCESSING|PAYMENTS?)\b")]
TYPE_NAICS = {"541214": PAYROLL, "561330": PAYROLL, "522320": CARD}

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
# Evidence from the daily edges and client overlap (optional table). (alias, label, what it means)
EVIDENCE = [
    ("e_pass2", "money out right after money in",
     "Share of its money out to PNC clients that had at least as much money in from them in the 2 days before: "
     "pass-through, as when an employer funds a payroll that goes out a day or two later."),
    ("e_payday", "money out on its 4 biggest days",
     "Share of a month's money out sent on its 4 biggest days: payroll pays on a few paydays, a merchant processor "
     "settles every day."),
    ("e_weekday", "money out on its main weekday",
     "Share of money out sent on its most common weekday: weekly or biweekly pay runs."),
    ("e_coclient", "paying organisations that also pay a registered TPPP",
     "Share of the organisations paying it that also pay a registered TPPP at PNC in the same month."),
    ("e_switch", "new paying organisations that just left a registered TPPP",
     "Share of its new paying organisations that stopped paying a registered TPPP in the 3 months before: it does the "
     "job the TPPP did for them."),
]
LABEL = ({a: lab for a, _, lab, _ in FEATURES} | {a: lab for a, lab, _ in DERIVED}
         | {a: lab for a, lab, _ in EVIDENCE})
KIND = {a: k for a, _, _, k in FEATURES} | {a: k for a, _, k in DERIVED} | {a: "evidence" for a, _, _ in EVIDENCE}
EVIDENCE_HELP = {a: h for a, _, h in EVIDENCE}
# Switching is shown to reviewers but kept out of the model: clients leaving a registered TPPP for a counterparty is
# something the examples (mostly the TPPPs' own accounts and the big brands) rarely show, so the model would learn it
# as a sign against being a processor, the opposite of what it means.
MODEL_EVIDENCE = {"e_pass2", "e_payday", "e_weekday", "e_coclient"}


# ================================================================== names and types
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


def type_of(name: str, naics=None, brand=None) -> str:
    """Processor type of an example: the known brand, else words in the name, else the industry code."""
    if isinstance(brand, str) and brand in BRAND_TYPE:
        return BRAND_TYPE[brand]
    n = name_norm(name)
    for t, pat in TYPE_WORDS[:3]:
        if re.search(pat, n):
            return t
    code = re.sub(r"\D", "", str(naics or ""))[:6]
    if code in TYPE_NAICS:
        return TYPE_NAICS[code]
    if re.search(TYPE_WORDS[3][1], n):
        return CARD
    return OTHER


# ================================================================== queries
def tppp_pwr_sql() -> str:
    return (f"SELECT DISTINCT CAST(cust_pwr_id AS STRING) AS pwr FROM {T_VECTOR} "
            f"WHERE customer_type = '{TPPP_TYPE}' AND load_date = (SELECT MAX(load_date) FROM {T_VECTOR})")


def tppp_customers_sql(pwrs) -> str:
    """Registered TPPP customers; power ids tried with and without leading zeros."""
    forms = sorted({p for x in pwrs for p in (x, x.lstrip("0")) if p})
    return (f"SELECT mdm_id, CAST(customer_pwr_id AS STRING) AS customer_pwr_id, "
            f"CAST(master_pwr_id AS STRING) AS master_pwr_id, customer_name, CAST(naics_cd AS STRING) AS naics_cd "
            f"FROM {pc.T_CDIM} WHERE CAST(customer_pwr_id AS STRING) IN ({pc._lit(forms)})")


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


def evidence_sql(per, cols: set[str]) -> str | None:
    """Evidence measures over the period as ratios of monthly sums (None if the table lacks the columns)."""
    M = pc.Q["month"]
    d = lambda c: f"CAST({c} AS DOUBLE)"
    parts = []
    if {"pt2_out_amt", "out_amt"} <= cols:
        parts.append(f"SUM({d('pt2_out_amt')}) / NULLIF(SUM({d('out_amt')}), 0) AS e_pass2")
    if {"top4_out_amt", "out_amt"} <= cols:
        parts.append(f"SUM({d('top4_out_amt')}) / NULLIF(SUM({d('out_amt')}), 0) AS e_payday")
    if {"wkday_max_out_amt", "out_amt"} <= cols:
        parts.append(f"SUM({d('wkday_max_out_amt')}) / NULLIF(SUM({d('out_amt')}), 0) AS e_weekday")
    if {"payer_orgs_tppp", "payer_orgs"} <= cols:
        parts.append(f"SUM({d('payer_orgs_tppp')}) / NULLIF(SUM({d('payer_orgs')}), 0) AS e_coclient")
    if {"switch_from_tppp", "new_payer_orgs"} <= cols:
        parts.append(f"SUM({d('switch_from_tppp')}) / NULLIF(SUM({d('new_payer_orgs')}), 0) AS e_switch")
        parts.append(f"SUM({d('switch_from_tppp')}) AS switch_n")
    if not parts or "ent_id" not in cols:
        return None
    return (f"SELECT ent_id, {', '.join(parts)} FROM {T_EVIDENCE} "
            f"WHERE {M} BETWEEN '{per[0]}' AND '{per[-1]}' GROUP BY ent_id")


def previous_period(per) -> tuple | None:
    """The period of the same length just before this one (clipped at the start of history), or None."""
    end = pc.add_months(per[0], -1)
    if end < pc.FIRST_MONTH:
        return None
    prev = pc.period(max(pc.add_months(per[0], -len(per)), pc.FIRST_MONTH), end)
    return prev if len(prev) >= MIN_MONTHS else None


# ================================================================== universe and examples
def prepare_universe(df: pd.DataFrame, tppp: pd.DataFrame, evidence: pd.DataFrame | None = None) -> pd.DataFrame:
    """Types, flags, the screen, and the two automatic example sources (registered TPPPs' accounts, known names)."""
    df = df.drop_duplicates("ent_id").reset_index(drop=True)
    if evidence is not None and len(evidence):
        df = df.merge(evidence.drop_duplicates("ent_id"), on="ent_id", how="left")
    feats = ([a for a, _, _, _ in FEATURES] + [a for a, _, _ in EVIDENCE]
             + ["ach_orig", "clients", "months_active", "n_fi", "pnc_match_n", "switch_n"])
    df = pc._num(df, feats)
    df["ent_name"] = df.ent_name.fillna("").astype(str).str.upper().str.strip()
    norm = df.ent_name.map(name_norm)
    df["amt"] = df.amt_in.fillna(0) + df.amt_out.fillna(0)
    share_in = df.amt_in.fillna(0) / df.amt.where(df.amt > 0)
    df["flow"] = np.select([share_in >= 0.8, share_in <= 0.2], pc.FLOWS[:2], pc.FLOWS[2])
    df["tm_all"] = df.tm_in.fillna(0) + df.tm_out.fillna(0) if "tm_in" in df else np.nan
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
    df["tppp_match"], df["tppp_naics"] = None, None
    if tppp is not None and len(tppp):
        keys = {}
        naics = tppp.naics_cd if "naics_cd" in tppp else pd.Series(None, index=tppp.index)
        for nm, nc in zip(tppp.customer_name, naics):
            if not isinstance(nm, str):
                continue
            k = name_key(nm)
            if not k or re.search(BRAND_BANK_RE, name_norm(nm)):
                continue
            first = k.split()[0]
            brand_ok = (first not in BRAND_EXCLUDE and first not in COMMON_FIRST and len(first) >= 4)
            keys[k] = (nm, brand_ok, nc)
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
            hit.append(m)
        df["tppp_match"] = [h[0] if h else None for h in hit]
        df["tppp_naics"] = [h[2] if h else None for h in hit]
    df["registered"] = df.tppp_match.notna()
    return df


# ================================================================== reviews and outside research (files)
def _read_csv(path: str, cols: list[str]) -> pd.DataFrame:
    try:
        v = pd.read_csv(path, dtype=str).fillna("")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=cols)
    return v[[c for c in cols if c in v.columns]]


def _append_csv(path: str, cols: list[str], rows: list[list]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        w.writerows(rows)


V_COLS = ["at", "ent_id", "ent_name", "verdict", "note", "reviewer"]
R_COLS = ["at", "ent_id", "ent_name", "answer", "confidence", "ptype", "evidence", "source"]


def load_verdicts(path: str = VERDICT_PATH) -> pd.DataFrame:
    return _read_csv(path, V_COLS)


def latest_verdicts(v: pd.DataFrame) -> pd.DataFrame:
    return v.sort_values("at").drop_duplicates("ent_id", keep="last") if len(v) else v


def save_verdict(ent_id: str, ent_name: str, verdict: str, note: str, reviewer: str,
                 path: str = VERDICT_PATH) -> None:
    if verdict not in VERDICTS + ["clear"]:
        raise ValueError("unknown verdict")
    _append_csv(path, V_COLS, [[dt.datetime.now().isoformat(timespec="seconds"), ent_id, ent_name, verdict,
                                note.replace("\n", " ").strip(), reviewer]])


def row_id(ent_id: str) -> str:
    return hashlib.sha256(f"{RESEARCH_SALT}|{ent_id}".encode()).hexdigest()[:12]


def enrichment_files(T: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Names only, behind a random-looking row id, for outside research; the key stays inside PNC."""
    rid = T.ent_id.map(row_id)
    return (pd.DataFrame({"row_id": rid, "name": T.ent_name}),
            pd.DataFrame({"row_id": rid, "ent_id": T.ent_id, "name": T.ent_name}))


def _pick(cols: dict, *cands):
    return next((cols[c] for c in cands if c in cols), None)


def parse_research(raw: pd.DataFrame, U: pd.DataFrame, source: str) -> tuple[pd.DataFrame, str]:
    """Outside research results -> one row per counterparty. Needs a row_id column (from the names file) and an
    answer column (yes / no / unsure, true / false, processor / not ...); confidence and evidence are optional."""
    cols = {re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c for c in raw.columns}
    c_id = _pick(cols, "row_id", "rowid", "id")
    c_ans = _pick(cols, "is_tppp", "tppp", "is_processor", "processor", "answer", "verdict", "result", "label")
    if c_id is None or c_ans is None:
        return pd.DataFrame(columns=R_COLS), ("The file needs a row_id column and an answer column (is_tppp, answer, "
                                              "verdict …). Columns found: " + ", ".join(map(str, raw.columns)))
    c_conf = _pick(cols, "confidence", "probability", "prob", "conf", "score")
    c_ev = _pick(cols, "evidence", "rationale", "reason", "reasons", "explanation", "notes", "note", "sources", "url")
    c_type = _pick(cols, "type", "processor_type", "category", "kind")
    lut = dict(zip(U.ent_id.map(row_id), U.ent_id))
    names = dict(zip(U.ent_id, U.ent_name))

    def answer(x):
        s = str(x).strip().lower()
        if re.fullmatch(r"(yes|y|true|1|tppp|processor|is a processor|likely( tppp| processor)?)", s):
            return "processor"
        if re.fullmatch(r"(no|n|false|0|not( a)? (tppp|processor)|unlikely)", s):
            return "not a processor"
        return "unsure"

    def conf(x):
        try:
            v = float(str(x).strip().rstrip("%"))
        except ValueError:
            return ""
        return f"{100 * v:.0f}" if v <= 1 else f"{min(v, 100):.0f}"

    ids = raw[c_id].astype(str).str.strip()
    ent = ids.map(lut)
    ok = ent.notna()
    now = dt.datetime.now().isoformat(timespec="seconds")
    out = pd.DataFrame({"at": now, "ent_id": ent[ok], "ent_name": ent[ok].map(names),
                        "answer": raw.loc[ok, c_ans].map(answer),
                        "confidence": raw.loc[ok, c_conf].map(conf) if c_conf else "",
                        "ptype": raw.loc[ok, c_type].astype(str) if c_type else "",
                        "evidence": raw.loc[ok, c_ev].astype(str).str.slice(0, 500) if c_ev else "",
                        "source": source})
    msg = f"{int(ok.sum()):,} of {len(raw):,} rows matched a counterparty in this period"
    if (~ok).any():
        msg += f"; {int((~ok).sum()):,} row ids not found (a different period, or not from this app's names file)"
    return out[R_COLS], msg


def save_research(rows: pd.DataFrame, path: str = RESEARCH_PATH) -> None:
    _append_csv(path, R_COLS, rows[R_COLS].astype(str).values.tolist())


def load_research(path: str = RESEARCH_PATH) -> pd.DataFrame:
    r = _read_csv(path, R_COLS)
    return r.sort_values("at").drop_duplicates("ent_id", keep="last") if len(r) else r


# ================================================================== model
def model_features(df: pd.DataFrame, use_size: bool) -> list[str]:
    cand = [a for a, _, _, _ in FEATURES] + [a for a, _, _ in DERIVED] + [a for a, _, _ in EVIDENCE
                                                                       if a in MODEL_EVIDENCE]
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


def _nearest(Xw: np.ndarray, ex: np.ndarray, k: int):
    """For every row, the k nearest example rows (never itself) and their distances; ex = sorted example indices."""
    n, k = len(Xw), min(k, max(len(ex) - 1, 1))
    E = Xw[ex]
    e2 = (E ** 2).sum(1)
    idx, dist = np.empty((n, k), dtype=int), np.empty((n, k))
    for a in range(0, n, 20000):
        blk = Xw[a:a + 20000]
        d2 = (blk ** 2).sum(1)[:, None] - 2 * blk @ E.T + e2[None, :]
        rows = np.arange(a, a + len(blk))
        self_ = np.isin(rows, ex)
        if self_.any():
            r = np.flatnonzero(self_)
            d2[r, np.searchsorted(ex, rows[r])] = np.inf
        part = np.argpartition(d2, kth=min(k, d2.shape[1] - 1) - 1, axis=1)[:, :k]
        pd2 = np.take_along_axis(d2, part, 1)
        o = np.argsort(pd2, axis=1)
        idx[a:a + len(blk)] = ex[np.take_along_axis(part, o, 1)]
        dist[a:a + len(blk)] = np.sqrt(np.maximum(np.take_along_axis(pd2, o, 1), 0))
    return idx, dist


def score(S: pd.DataFrame, feats: list[str], y: np.ndarray, groups: np.ndarray, weight: np.ndarray) -> dict:
    """Out-of-fold scores (0-100 percentile within each fold), full-data weights, reliability and explanations."""
    X = S[feats].rank(pct=True).fillna(0.5).to_numpy() - 0.5         # centred percentiles
    n = len(S)
    fold = _folds(y, groups, N_FOLDS, SEED)
    pctl = np.zeros(n)
    fit = lambda m: LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(
        X[m], y[m], sample_weight=weight[m])
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        if y[tr].sum() < 3 or te.sum() == 0:
            continue
        pctl[te] = pd.Series(fit(tr).decision_function(X[te])).rank(pct=True).to_numpy()
    coef = fit(np.ones(n, bool)).coef_[0]
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
    w = np.abs(coef) / (np.abs(coef).sum() or 1)
    near, dist = _nearest(X * np.sqrt(w), np.flatnonzero(y == 1), K_NEAR)
    weights = pd.DataFrame({"feature": feats, "label": [LABEL[f] for f in feats], "kind": [KIND[f] for f in feats],
                            "weight": coef}).sort_values("weight", ascending=False)
    pos = pctl[y == 1]
    rel = pd.DataFrame([{"top": f"top {t:g}%", "share of hidden examples found": float((pos >= 1 - t / 100).mean()),
                         "a random list would find": t / 100,
                         "counterparties to review": int(np.ceil(n * t / 100))} for t in (0.5, 1, 2, 5)])
    return {"pctl": 100 * pctl, "why": why, "near": near, "near_dist": dist, "weights": weights, "reliability": rel,
            "X": X, "coef": coef, "feats": feats}


def run_model(U: pd.DataFrame, verdicts: pd.DataFrame, sources: tuple, use_size: bool) -> dict:
    """Examples from the chosen sources, out-of-fold scores, types, value and everything the app shows."""
    S = U[U.screened].reset_index(drop=True).copy()
    if len(S) < 50:
        raise ValueError(f"only {len(S)} counterparties pass the screen in this period: choose a longer or later period")
    V = latest_verdicts(verdicts)
    vmap = dict(zip(V.ent_id, V.verdict)) if len(V) else {}
    S["verdict"] = S.ent_id.map(vmap).fillna("")
    S.loc[S.verdict == "clear", "verdict"] = ""
    src = np.full(len(S), "", dtype=object)
    grp = S.ent_id.astype(str).to_numpy().astype(object)
    if "confirmed" in sources:
        src[(S.verdict == "processor").to_numpy()] = "confirmed"
    if "known" in sources:
        m = S.known_proc.to_numpy() & (src == "")
        src[m] = "known"
        grp[m] = ("K:" + S.known_brand.fillna("")).to_numpy()[m]
    if "registered" in sources:
        m = S.registered.to_numpy() & (src == "")
        src[m] = "registered"
        tm = S.tppp_match.astype(object).where(S.tppp_match.notna(), "")
        grp[m] = ("T:" + tm.map(lambda s: ((name_key(s) or s or "?").split() or ["?"])[0])).to_numpy()[m]
    rejected = (S.verdict == "not a processor").to_numpy()
    src[rejected] = ""                                               # reviewers' rejections never teach as examples
    S["example"] = src
    y = (src != "").astype(int)
    if y.sum() < 10:
        raise ValueError(f"only {y.sum()} examples among scored counterparties: switch more example sources on")
    weight = np.where(rejected, NEG_WEIGHT, 1.0)                     # ... but they count as firm negatives
    feats = model_features(S, use_size)
    R = score(S, feats, y, grp, weight)
    S["score"] = R["pctl"]
    S["why"] = R["why"]
    S["similar"] = S.ent_name.to_numpy()[R["near"][:, 0]]
    # processor type: of each example from its brand / registered TPPP / name; of everyone by the nearest examples
    ex_type = np.full(len(S), "", dtype=object)
    for i in np.flatnonzero(y == 1):
        r = S.iloc[i]
        ex_type[i] = (type_of(r.ent_name, brand=r.known_brand) if src[i] == "known" else
                      type_of(r.tppp_match if isinstance(r.tppp_match, str) else r.ent_name, r.tppp_naics)
                      if src[i] == "registered" else
                      type_of(r.ent_name))
    votes = ex_type[R["near"]]                                        # n x K types of the nearest examples
    ptype, pshare = [], []
    for row in votes:
        vals, cnt = np.unique(row, return_counts=True)
        best = vals[cnt == cnt.max()]
        t = next(x for x in row if x in best)                        # ties: the nearer example's type
        ptype.append(t)
        pshare.append(cnt.max() / len(row))
    S["ptype"], S["ptype_share"], S["example_type"] = ptype, pshare, ex_type
    # value: business with PNC clients, separate from the TPPP score
    reach = S.tm_all if "tm_all" in S and S.tm_all.notna().any() else S.clients
    S["value"] = 100 * (S.amt.rank(pct=True) + reach.rank(pct=True)) / 2
    order = S.sort_values(["score", "amt"], ascending=False).index                # ties broken by money
    S.loc[order, "rank"] = np.arange(1, len(S) + 1)
    S["rank"] = S["rank"].astype(int)
    return {"S": S, "weights": R["weights"], "reliability": R["reliability"], "n_examples": int(y.sum()),
            "n_negatives": int(rejected.sum()), "X": R["X"], "coef": R["coef"], "feats": feats,
            "by_source": pd.Series(src[src != ""]).map(SOURCES).value_counts(),
            "by_type": pd.Series(ex_type[y == 1]).value_counts(),
            "evidence_used": [f for f in feats if KIND[f] == "evidence"]}


def similar_examples(res: dict, i: int, k: int = 3) -> pd.DataFrame:
    """The k examples nearest to row i of res['S'] in the weighted measure space, with the closest measures."""
    S, X, coef, feats = res["S"], res["X"], res["coef"], res["feats"]
    w = np.abs(coef) / (np.abs(coef).sum() or 1)
    ex = np.flatnonzero((S.example != "").to_numpy())
    ex = ex[ex != i]
    d = np.sqrt((((X[ex] - X[i]) ** 2) * w).sum(1))
    top = ex[np.argsort(d)[:k]]
    big = w > np.median(w)
    out = []
    for j in top:
        gap = np.where(big, np.abs(X[j] - X[i]), np.inf)
        close = [LABEL[feats[c]] for c in np.argsort(gap)[:3] if big[c]]
        out.append({"example": S.ent_name.iat[j], "type": S.example_type.iat[j],
                    "why it is an example": SOURCES.get(S.example.iat[j], ""), "closest on": ", ".join(close)})
    return pd.DataFrame(out)
