"""Data access. One source object per mode with the same methods: ImpalaSource (tables built by the notebooks, read through
the in-house dbi helper) and DemoSource (synthetic, in core/demo.py). These methods are the natural contract for the future
FastAPI layer: each one maps to an endpoint."""
import re
import numpy as np
import pandas as pd

ID_RE = re.compile(r"^[A-Za-z0-9_\-:]+$")
YM_RE = re.compile(r"^\d{4}-\d{2}$")


def ids_sql(values):
    """Quoted id list for SQL; only validated ids ever reach a query."""
    vals = [str(v) for v in values if v is not None and str(v) not in ("nan", "None", "")]
    bad = [v for v in vals if not ID_RE.match(v)]
    if bad: raise ValueError(f"unexpected id format: {bad[:3]}")
    return ", ".join(f"'{v}'" for v in dict.fromkeys(vals)) or "''"


def ym(v):
    if not YM_RE.match(str(v)): raise ValueError(f"bad month {v}")
    return str(v)


def ym_add(m, k):
    y, mo = map(int, m.split("-")); i = y * 12 + mo - 1 + k
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


def numeric(df, keep=()):
    for c in df.columns:
        if c not in keep and df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= df[c].notna().sum() * 0.99 and df[c].notna().any():
                df[c] = conv
    return df


class ImpalaSource:
    """Reads through the team's dbi helper; connection and authentication are handled there."""
    def __init__(self, cfg):
        import importlib
        self.cfg, self.db = cfg, cfg["database"]
        self.dbi = importlib.import_module(cfg.get("dbi_module", "dbi"))

    def q(self, sql):
        df = self.dbi.db_get_query(sql, dsn=self.cfg["dbi_dsn"], pool=self.cfg["dbi_pool"],
                                   conn_options=dict(self.cfg.get("dbi_conn_options", {})))
        df = pd.DataFrame(df)
        df.columns = [str(c).split(".")[-1] for c in df.columns]      # "table.col" → "col"
        return df

    def t(self, key):
        return f"{self.db}.{self.cfg[key]}"

    # ── attrition ────────────────────────────────────────────────────
    def months(self):
        m = self.q(f"SELECT DISTINCT month FROM {self.t('features_table')}")
        return sorted(m.month.astype(str), reverse=True)

    def features(self, month):
        return self.q(f"SELECT * FROM {self.t('features_table')} WHERE month = '{ym(month)}'")

    def customer_features(self, cust):
        return self.q(f"SELECT * FROM {self.t('features_table')} WHERE cust_pwr_id = {ids_sql([cust])} ORDER BY month")

    def history(self, custs):
        return self.q(f"SELECT * FROM {self.t('history_table')} WHERE cust_pwr_id IN ({ids_sql(custs)}) ORDER BY cust_pwr_id, month")

    def relationship(self, rltn):
        return self.q(f"SELECT cust_pwr_id, customer_name, month, bal, bal_norm FROM {self.t('history_table')} "
                      f"WHERE rltn_pwr_id = {ids_sql([rltn])} ORDER BY cust_pwr_id, month")

    def mdm_ids(self, cust):
        r = self.q(f"SELECT mdm_ids FROM {self.t('features_table')} WHERE cust_pwr_id = {ids_sql([cust])} ORDER BY month DESC LIMIT 1")
        return [x for x in str(r.mdm_ids.iloc[0] if len(r) else "").split(",") if x]

    # ── suspicious activity ──────────────────────────────────────────
    def fraud_signals(self):
        return numeric(self.q(f"SELECT * FROM {self.t('fraud_signals_table')}"),
                       keep=("mdm_id", "customer_pwr_id", "pwr", "master_pwr_id", "naics_cd", "month", "rt_top_other"))

    def fraud_history(self, mdm_ids):
        return numeric(self.q(f"SELECT * FROM {self.t('fraud_history_table')} WHERE mdm_id IN ({ids_sql(mdm_ids)}) ORDER BY mdm_id, month"),
                       keep=("mdm_id", "month"))

    def fraud_evidence(self, mdm_ids):
        return numeric(self.q(f"SELECT * FROM {self.t('fraud_evidence_table')} WHERE mdm_id IN ({ids_sql(mdm_ids)})"),
                       keep=("mdm_id", "month", "txn_date", "cpty_id", "cpty_name", "bank", "direction", "typology"))

    # ── shared: customers, payment profile and network ───────────────
    def latest_month(self):
        return str(self.q(f"SELECT MAX(month) AS m FROM {self.t('metrics_table')}").m.iloc[0])

    def search_customers(self, text, limit=50):
        """Any client by name (contains) or by power id / payment entity id (exact)."""
        t = re.sub(r"[^A-Za-z0-9 &.,'\-]", "", str(text)).strip().lower()[:60]
        if len(t) < 2: return pd.DataFrame(columns=["mdm_id", "customer_pwr_id", "customer_name", "naics_desc"])
        like = t.replace("'", "''")
        idq = f" OR regexp_replace(CAST(customer_pwr_id AS STRING), '^0+', '') = '{t.lstrip('0')}' OR lower(mdm_id) = '{like}'" \
            if ID_RE.match(t) else ""
        return self.q(f"SELECT mdm_id, customer_pwr_id, customer_name, naics_desc FROM {self.t('cust_dim_table')} "
                      f"WHERE lower(customer_name) LIKE '%{like}%'{idq} LIMIT {int(limit)}")

    def lookup(self, key):
        """Name, power id and payment entities for one customer key (normalized power id, or M:<mdm_id>)."""
        key = str(key)
        if key.startswith("M:"):
            where = f"mdm_id = {ids_sql([key[2:]])}"
        else:
            if not ID_RE.match(key): return None
            where = f"regexp_replace(CAST(customer_pwr_id AS STRING), '^0+', '') = '{key}'"
        r = self.q(f"SELECT mdm_id, customer_pwr_id, master_pwr_id, customer_name, naics_desc FROM {self.t('cust_dim_table')} WHERE {where}")
        if not len(r): return None
        nm = r.customer_name.dropna()
        return dict(key=key, name=nm.iloc[0] if len(nm) else None, pwr=str(r.customer_pwr_id.dropna().iloc[0]) if r.customer_pwr_id.notna().any() else None,
                    mdm_ids=sorted(set(r.mdm_id.astype(str))), naics_desc=r.naics_desc.dropna().iloc[0] if r.naics_desc.notna().any() else None,
                    master_pwr_id=r.master_pwr_id.dropna().iloc[0] if r.master_pwr_id.notna().any() else None)

    # ── shared: payment profile and network ──────────────────────────
    def metrics_history(self, mdm_ids, start_month):
        """Monthly payment metrics for any entity (used when a module has no stored history for it)."""
        return numeric(self.q(f"SELECT * FROM {self.t('metrics_table')} WHERE mdm_id IN ({ids_sql(mdm_ids)}) "
                              f"AND month >= '{ym(start_month)}' ORDER BY mdm_id, month"),
                       keep=("mdm_id", "customer_pwr_id", "master_pwr_id", "customer_name", "naics_cd", "naics_desc", "month"))

    def _names(self, ids):
        ids = [c for c in dict.fromkeys(ids) if ID_RE.match(str(c))][:5000]
        if not ids: return pd.DataFrame(columns=["cpty_id", "name", "bank"])
        ext = self.q(f"SELECT cpty_id, {self.cfg['cpty_name_col']} AS name, {self.cfg['cpty_fi_col']} AS bank "
                     f"FROM {self.t('cpty_table')} WHERE cpty_id IN ({ids_sql(ids)})")
        cust = self.q(f"SELECT mdm_id AS cpty_id, customer_name AS name, customer_pwr_id, 'PNC' AS bank FROM {self.t('cust_dim_table')} "
                      f"WHERE mdm_id IN ({ids_sql(ids)})")
        return pd.concat([cust, ext], ignore_index=True).drop_duplicates("cpty_id")

    def counterparties(self, mdm_ids, end_month, n_months=6):
        """Payers (money in) and payees (money out) by month, with names, banks and, for PNC clients, their power id."""
        ids = ids_sql(mdm_ids); m0 = ym_add(ym(end_month), -(n_months - 1)); e = self.t("edge_table")
        inn = self.q(f"SELECT month, src_id AS cpty_id, src_kind AS kind, SUM(amount) AS amount, SUM(n_txn) AS n_txn FROM {e} "
                     f"WHERE month BETWEEN '{m0}' AND '{end_month}' AND dst_id IN ({ids}) AND NOT is_self "
                     f"GROUP BY month, src_id, src_kind")
        out = self.q(f"SELECT month, dst_id AS cpty_id, dst_kind AS kind, is_self, SUM(amount) AS amount, SUM(n_txn) AS n_txn FROM {e} "
                     f"WHERE month BETWEEN '{m0}' AND '{end_month}' AND src_id IN ({ids}) "
                     f"GROUP BY month, dst_id, dst_kind, is_self")
        names = self._names(list(inn.cpty_id) + list(out.cpty_id))
        for d in (inn, out):
            d["amount"] = pd.to_numeric(d.amount, errors="coerce").astype(float)
            d["n_txn"] = pd.to_numeric(d.n_txn, errors="coerce")
        return inn.merge(names, on="cpty_id", how="left"), out.merge(names, on="cpty_id", how="left")

    def internal_ties(self, mdm_ids, end_month, n_months=6):
        """Payments between the given PNC clients (the ego network's alter–alter ties)."""
        if len(mdm_ids) < 2: return pd.DataFrame(columns=["src_id", "dst_id", "amount", "n_txn"])
        ids = ids_sql(mdm_ids); m0 = ym_add(ym(end_month), -(n_months - 1))
        d = self.q(f"SELECT src_id, dst_id, SUM(amount) AS amount, SUM(n_txn) AS n_txn FROM {self.t('edge_table')} "
                   f"WHERE month BETWEEN '{m0}' AND '{end_month}' AND src_kind = 'CUST' AND dst_kind = 'CUST' "
                   f"AND src_id IN ({ids}) AND dst_id IN ({ids}) AND src_id <> dst_id GROUP BY src_id, dst_id")
        d["amount"] = pd.to_numeric(d.amount, errors="coerce").astype(float)
        return d
