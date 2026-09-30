"""Data sources with one interface: ImpalaSource (tables written by the model notebook) and DemoSource (synthetic)."""
import re
import numpy as np
import pandas as pd

ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
HIST_METRICS = ["amt_all_in", "amt_all_out", "ntxn_all_in", "ntxn_all_out", "active_days_in", "ncpty_all_in", "ncpty_all_out",
                "n_fi_out", "n_fi_in", "selfpay_amt_out", "share_wire_in", "share_check_in", "net_flow_all"]


def _ids(values):
    vals = [str(v) for v in values if v is not None and str(v) != "nan"]
    bad = [v for v in vals if not ID_RE.match(v)]
    if bad: raise ValueError(f"unexpected id format: {bad[:3]}")
    return ", ".join(f"'{v}'" for v in vals) or "''"


def ym_add(ym, k):
    y, m = map(int, ym.split("-")); i = y * 12 + m - 1 + k
    return f"{i // 12:04d}-{i % 12 + 1:02d}"


# ─────────────────────────────────────────────────────────────────────
class ImpalaSource:
    def __init__(self, cfg):
        self.cfg, self.db = cfg, cfg["database"]
        self._conn = None

    def conn(self):
        if self._conn is None:
            from impala.dbapi import connect
            p = dict(self.cfg["impala"])
            self._conn = connect(host=p["host"], port=int(p.get("port", 21050)),
                                 auth_mechanism=p.get("auth_mechanism", "GSSAPI"), use_ssl=bool(p.get("use_ssl", True)),
                                 kerberos_service_name=p.get("kerberos_service_name", "impala"),
                                 user=p.get("user"), password=p.get("password"), timeout=p.get("timeout", 600))
        return self._conn

    def q(self, sql):
        cur = self.conn().cursor()
        try:
            cur.execute(sql)
            cols = [d[0].split(".")[-1] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)
        finally:
            cur.close()

    def t(self, key):
        return f"{self.db}.{self.cfg[key]}"

    def months(self):
        m = self.q(f"SELECT DISTINCT month FROM {self.t('features_table')}")
        return sorted(m.month.astype(str), reverse=True)

    def features(self, month):
        if not re.match(r"^\d{4}-\d{2}$", month): raise ValueError(month)
        return self.q(f"SELECT * FROM {self.t('features_table')} WHERE month = '{month}'")

    def customer_features(self, cust):
        return self.q(f"SELECT * FROM {self.t('features_table')} WHERE cust_pwr_id = {_ids([cust])} ORDER BY month")

    def history(self, custs):
        return self.q(f"SELECT * FROM {self.t('history_table')} WHERE cust_pwr_id IN ({_ids(custs)}) ORDER BY cust_pwr_id, month")

    def relationship(self, rltn):
        return self.q(f"SELECT cust_pwr_id, customer_name, month, bal, bal_norm FROM {self.t('history_table')} "
                      f"WHERE rltn_pwr_id = {_ids([rltn])} ORDER BY cust_pwr_id, month")

    def counterparties(self, mdm_ids, end_month, n_months=6):
        """Payers (money in) and payees (money out) by month, with names and banks."""
        ids = _ids(mdm_ids); m0 = ym_add(end_month, -(n_months - 1)); e = self.t("edge_table")
        inn = self.q(f"SELECT month, src_id AS cpty_id, src_kind AS kind, SUM(amount) AS amount, SUM(n_txn) AS n_txn FROM {e} "
                     f"WHERE month BETWEEN '{m0}' AND '{end_month}' AND dst_id IN ({ids}) AND NOT is_self "
                     f"GROUP BY month, src_id, src_kind")
        out = self.q(f"SELECT month, dst_id AS cpty_id, dst_kind AS kind, is_self, SUM(amount) AS amount, SUM(n_txn) AS n_txn FROM {e} "
                     f"WHERE month BETWEEN '{m0}' AND '{end_month}' AND src_id IN ({ids}) "
                     f"GROUP BY month, dst_id, dst_kind, is_self")
        cp = set(inn.cpty_id) | set(out.cpty_id)
        names = pd.DataFrame(columns=["cpty_id", "name", "bank"])
        c_ids = [c for c in cp if ID_RE.match(str(c))]
        if c_ids:
            names = self.q(f"SELECT cpty_id, {self.cfg['cpty_name_col']} AS name, {self.cfg['cpty_fi_col']} AS bank "
                           f"FROM {self.t('cpty_table')} WHERE cpty_id IN ({_ids(c_ids[:5000])})")
            cust = self.q(f"SELECT mdm_id AS cpty_id, customer_name AS name, 'PNC' AS bank FROM {self.t('cust_dim_table')} "
                          f"WHERE mdm_id IN ({_ids(c_ids[:5000])})")
            names = pd.concat([names, cust], ignore_index=True).drop_duplicates("cpty_id")
        for d in (inn, out):
            d["amount"] = pd.to_numeric(d.amount, errors="coerce").astype(float)
        return inn.merge(names, on="cpty_id", how="left"), out.merge(names, on="cpty_id", how="left")


# ─────────────────────────────────────────────────────────────────────
class DemoSource:
    """Synthetic clients in the same shape as the real tables, so the app can be tried without data access."""
    WORDS_A = ["Allegheny", "Summit", "Riverbend", "Keystone", "Harbor", "Liberty", "Cedar", "Ironworks", "Northgate", "Three Rivers",
               "Blue Ridge", "Monongahela", "Lakeshore", "Pioneer", "Granite", "Beacon", "Maple", "Sterling", "Crescent", "Frontier"]
    WORDS_B = ["Logistics", "Foods", "Medical Group", "Builders", "Manufacturing", "Dental Partners", "Supply Co", "Holdings",
               "Auto Group", "Services", "Distribution", "Energy", "Properties", "Packaging", "Staffing", "Labs"]
    NAICS = ["Wholesale trade", "Manufacturing", "Health care", "Construction", "Retail trade", "Transportation", "Professional services",
             "Real estate", "Accommodation and food", "Finance and insurance"]
    BANKS = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Citizens", "Huntington", "Fifth Third", "KeyBank", "Truist", "M&T"]

    def __init__(self, n=900, n_months=20, end="2026-08", seed=7):
        rng = np.random.default_rng(seed); self.rng = rng
        self.all_months = [ym_add(end, -k) for k in range(n_months - 1, -1, -1)]
        M = n_months; T = np.arange(M)
        ids = [f"{100000 + i:018d}" for i in range(n)]
        names = [f"{rng.choice(self.WORDS_A)} {rng.choice(self.WORDS_B)}" for _ in range(n)]
        rl = [f"R{i // 2 if rng.random() < .35 else 10000 + i:06d}" for i in range(n)]
        norm = np.exp(rng.normal(13.5, 1.5, n)); leaver = rng.random(n) < 0.22
        exit_m = np.where(leaver, rng.integers(M - 9, M + 5, n), 10 ** 6)
        kind = rng.choice(["competitive", "receipts", "drawdown"], n, p=[.4, .4, .2])
        rows, hist = [], []
        for i in range(n):
            e = exit_m[i]; base_in = np.exp(rng.normal(np.log(norm[i]) + 0.4, .4)); base_n = max(3, rng.poisson(40))
            k = np.clip((T - (e - 6)) / 6, 0, 1) if leaver[i] else np.zeros(M)          # 0 → 1 over the 6 months before exit
            wob = rng.normal(0, .06, M)
            amt_in = base_in * (1 - .7 * k * (kind[i] != "drawdown")) * np.exp(wob)
            n_in = np.maximum(0, rng.poisson(base_n * (1 - .6 * k * (kind[i] != "drawdown"))))
            days_in = np.minimum(22, np.maximum(0, np.round(n_in * .4 + rng.normal(0, 1, M))))
            payers = np.maximum(0, np.round(base_n * .3 * (1 - .5 * k) + rng.normal(0, 1, M)))
            amt_out = base_in * np.exp(rng.normal(0, .06, M)) * (1 + 2.5 * (T == e - 1))
            n_fi = np.maximum(1, np.round(3 + 2 * k * (kind[i] == "competitive") + rng.normal(0, .5, M)))
            selfpay = amt_out * np.clip(.02 + .5 * k * (kind[i] == "competitive"), 0, 1)
            bal = norm[i] * np.exp(rng.normal(0, .04, M)) * np.where(T >= e, 0.002, np.where(T == e - 1, .45, 1 - .25 * k * (kind[i] == "drawdown")))
            eom = bal * np.where(T == e - 1, .5, 1.0) * np.exp(rng.normal(0, .02, M))
            for t in range(M):
                hist.append(dict(cust_pwr_id=ids[i], rltn_pwr_id=rl[i], customer_name=names[i], month=self.all_months[t], bal=bal[t],
                                 bal_norm=norm[i], bal_eom=eom[t], bal_min=bal[t] * .8, bal_max=bal[t] * 1.2, live=int(t < e),
                                 amt_all_in=amt_in[t], amt_all_out=amt_out[t], ntxn_all_in=n_in[t], ntxn_all_out=int(n_in[t] * .8),
                                 active_days_in=days_in[t], ncpty_all_in=payers[t], ncpty_all_out=payers[t] * .7, n_fi_out=n_fi[t], n_fi_in=3,
                                 selfpay_amt_out=selfpay[t], share_wire_in=.2 + .1 * k[t], share_check_in=.3 - .1 * k[t],
                                 net_flow_all=amt_in[t] - amt_out[t], y=int(leaver[i] and t < e <= t + 6), exit_t=e, t=t))
        H = pd.DataFrame(hist)
        H["naics_desc"] = H.cust_pwr_id.map(dict(zip(ids, rng.choice(self.NAICS, n))))
        H["segment_desc"] = H.cust_pwr_id.map(dict(zip(ids, rng.choice(["Middle Market", "Business Banking", "Corporate"], n, p=[.5, .35, .15]))))
        self.H = H.sort_values(["cust_pwr_id", "t"]).reset_index(drop=True)
        self.F = self._features(self.H)
        self.mdm = {c: [f"M{c[-6:]}0"] for c in ids}
        self.kind = dict(zip(ids, kind)); self.leaver = dict(zip(ids, leaver)); self.exit = dict(zip(ids, exit_m))

    @staticmethod
    def _features(H):
        g = H.groupby("cust_pwr_id", sort=False)
        L = lambda c: np.log1p(H[c].clip(lower=0))
        def chg(s, a, b):
            base = s.groupby(H.cust_pwr_id).transform(lambda x: x.shift(a).rolling(b - a + 1, min_periods=2).mean())
            return s - base
        F = H[["cust_pwr_id", "rltn_pwr_id", "customer_name", "naics_desc", "segment_desc", "month", "bal", "bal_norm", "bal_eom", "t", "y", "exit_t"]].copy()
        F["capped"] = np.minimum(F.bal, F.bal_norm)
        lb = np.log1p(H.bal)
        F["dep_log_norm"] = np.log1p(H.bal_norm); F["dep_log_bal"] = lb
        F["dep_bal_vs_norm"] = lb - np.log1p(H.bal_norm)
        F["dep_d1"] = lb - lb.groupby(H.cust_pwr_id).shift(1)
        F["dep_d6"] = chg(lb, 4, 6); F["dep_d3"] = chg(lb, 1, 3)
        F["dep_eom_vs_avg"] = np.log1p(H.bal_eom) - lb
        F["dep_range"] = np.log1p(H.bal_max) - np.log1p(H.bal_min)
        rl_tot = H.groupby(["rltn_pwr_id", "t"]).bal.transform("sum") - H.bal
        rl_n = H.groupby(["rltn_pwr_id", "t"]).cust_pwr_id.transform("count") - 1
        F["dep_siblings"] = np.log1p(rl_n)
        F["dep_sib_d6"] = np.where(rl_n > 0, chg(np.log1p(rl_tot), 4, 6), np.nan)
        F["pay_log_flow"] = np.log1p(H.amt_all_in + H.amt_all_out)
        for c in ["amt_all_in", "active_days_in", "ntxn_all_in", "ncpty_all_in", "amt_all_out", "n_fi_out"]:
            F[f"{c}__d6"] = chg(L(c), 4, 6); F[f"{c}__d3"] = chg(L(c), 1, 3)
        F["selfpay_share_out__d3"] = chg(H.selfpay_amt_out / H.amt_all_out.clip(lower=1), 1, 3)
        F["share_WIRE_in__d3"] = chg(H.share_wire_in, 1, 3)
        dec = F.groupby("month").bal_norm.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
        for c in ["amt_all_in", "ntxn_all_in", "ncpty_all_in"]:
            lv = L(c); F[f"{c}__peer"] = lv - lv.groupby([F.month, dec]).transform("median")
        live_msh = (H.live == 1) & (H.bal >= .5 * H.bal_norm)
        return F[live_msh.values].reset_index(drop=True)

    def months(self):
        return sorted(self.F.month.unique(), reverse=True)

    def features(self, month):
        return self.F[self.F.month == month].drop(columns=["y", "exit_t", "t"])

    def customer_features(self, cust):
        return self.F[self.F.cust_pwr_id == cust].drop(columns=["y", "exit_t", "t"])

    def history(self, custs):
        return self.H[self.H.cust_pwr_id.isin(list(custs))].drop(columns=["y", "exit_t", "t"])

    def relationship(self, rltn):
        return self.H[self.H.rltn_pwr_id == rltn][["cust_pwr_id", "customer_name", "month", "bal", "bal_norm"]]

    def mdm_ids(self, cust):
        return self.mdm.get(cust, [])

    def counterparties(self, mdm_ids, end_month, n_months=6):
        c = next((k for k, v in self.mdm.items() if v == list(mdm_ids)), None)
        r = np.random.default_rng(abs(hash(c)) % 2 ** 32); ms = [ym_add(end_month, -k) for k in range(n_months - 1, -1, -1)]
        lv, kd = self.leaver.get(c, False), self.kind.get(c)
        inn, out = [], []
        for j in range(12):
            nm = f"{r.choice(self.WORDS_A)} {r.choice(self.WORDS_B)}"; bank = r.choice(self.BANKS + ["PNC"])
            base = float(np.exp(r.normal(11, 1))); stop = lv and kd != "drawdown" and j < 4
            for i, m in enumerate(ms):
                if stop and i >= n_months - 3 + (j % 2): continue
                inn.append(dict(month=m, cpty_id=f"C{j}", kind="CPTY", amount=base * np.exp(r.normal(0, .15)), n_txn=int(r.poisson(6)), name=nm, bank=bank))
        for j in range(8):
            nm = f"{r.choice(self.WORDS_A)} {r.choice(self.WORDS_B)}"; bank = r.choice(self.BANKS); base = float(np.exp(r.normal(11, 1)))
            for i, m in enumerate(ms):
                out.append(dict(month=m, cpty_id=f"P{j}", kind="CPTY", is_self=False, amount=base * np.exp(r.normal(0, .15)), n_txn=5, name=nm, bank=bank))
        if lv and kd == "competitive":
            for i, m in enumerate(ms[-3:]):
                out.append(dict(month=m, cpty_id="SELF1", kind="CPTY", is_self=True, amount=float(np.exp(r.normal(14, .3))) * (i + 1),
                                n_txn=2 + i, name="(own account)", bank="JPMorgan Chase"))
        return pd.DataFrame(inn), pd.DataFrame(out)


def demo_bundle(src):
    """Train a small LightGBM on the synthetic clients so the demo runs the same scoring path as production."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    F = src.F; M = len(src.all_months)
    feats = [c for c in F.columns if c.startswith(("dep_", "pay_")) or "__" in c]
    lab = F[F.t <= M - 7]
    hold = lab.cust_pwr_id.map(lambda c: int(c[-3:]) % 5 == 0).to_numpy()
    X = lab[feats].to_numpy(np.float32); y = lab.y.to_numpy()
    params = dict(objective="binary", learning_rate=.05, num_leaves=15, min_data_in_leaf=40, feature_fraction=.8, verbose=-1, seed=7)
    m = lgb.train(params, lgb.Dataset(X[~hold], y[~hold]), 300, valid_sets=[lgb.Dataset(X[hold], y[hold])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=.001, y_max=.95).fit(m.predict(X[hold], num_iteration=m.best_iteration), y[hold])
    st_ = F[F.y == 0]
    thr = {}
    for f, d in [("amt_all_in__d6", "falls"), ("active_days_in__d6", "falls"), ("ntxn_all_in__d6", "falls"), ("ncpty_all_in__d6", "falls"),
                 ("n_fi_out__d6", "rises"), ("selfpay_share_out__d3", "rises"), ("dep_sib_d6", "falls"), ("dep_eom_vs_avg", "falls"), ("dep_d1", "falls")]:
        thr[f] = (float(st_[f].quantile(.10 if d == "falls" else .90)), d)
    onset = {"amt_all_in__d6": -5, "active_days_in__d6": -4, "ntxn_all_in__d6": -3, "ncpty_all_in__d6": -3, "n_fi_out__d6": -3,
             "selfpay_share_out__d3": -2, "dep_sib_d6": -3, "dep_eom_vs_avg": -1, "dep_d1": -1}
    return dict(engine="lightgbm", model=m.model_to_string(), n_iter=m.best_iteration, iso=iso, feats=feats, labels={},
                famof={}, thr=thr, onset=onset, late_feats=["dep_eom_vs_avg", "dep_d1"], k_main=1000, alpha=.5, p_cap=.95,
                trained_through=src.all_months[M - 7], created="demo", version="demo")
