"""Synthetic data in the shape of the real tables, so the whole app can be shown without data access (mode = "demo").
One world of clients feeds every module — deposit attrition, suspicious-activity patterns and the payment network — so the
same client can appear in several views, as in production."""
import numpy as np
import pandas as pd

from .data import ym_add
from .fmt import norm_pwr

WORDS_A = ["Allegheny", "Summit", "Riverbend", "Keystone", "Harbor", "Liberty", "Cedar", "Ironworks", "Northgate", "Three Rivers",
           "Blue Ridge", "Monongahela", "Lakeshore", "Pioneer", "Granite", "Beacon", "Maple", "Sterling", "Crescent", "Frontier",
           "Ohio Valley", "Bridgeview", "Fox Chapel", "Laurel", "Juniata", "Susquehanna", "Delaware", "Chesapeake"]
WORDS_B = ["Logistics", "Foods", "Medical Group", "Builders", "Manufacturing", "Dental Partners", "Supply Co", "Holdings",
           "Auto Group", "Services", "Distribution", "Energy", "Properties", "Packaging", "Staffing", "Labs"]
NAICS = [("423", "Wholesale trade"), ("332", "Manufacturing"), ("621", "Health care"), ("236", "Construction"),
         ("452", "Retail trade"), ("484", "Transportation"), ("541", "Professional services"), ("531", "Real estate"),
         ("722", "Accommodation and food"), ("523", "Finance and insurance")]
BANKS = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Citizens", "Huntington", "Fifth Third", "KeyBank", "Truist", "M&T",
         "First National", "Dollar Bank", "TD Bank"]
SHELL_A = ["Apex", "Global", "Prime", "Vertex", "Orion", "Atlas", "Nova", "Zenith", "Pinnacle", "Meridian", "Sapphire", "Crown",
           "Harborline", "Silverleaf", "Northstar", "Redwood", "Bluewater", "Goldcrest", "Ironbridge", "Westbrook"]
SHELL_B = ["Trading LLC", "Consulting LLC", "Ventures Inc", "Imports LLC", "Capital LLC", "Solutions LLC", "Holdings LLC",
           "Group LLC", "Partners LLC", "Exports Inc", "Logistics LLC", "Enterprises LLC"]

# fraud-signal settings mirrored from pkg_fraud_signals_v1.ipynb §0
MIN_FLOW_3M, PT_MIN, PT_MONTH_MIN, PT_MONTHS_REQ, PT_MAX_DAYS_HELD = 300_000, .90, .85, 2, 5.0
FUN_MIN_PAYERS, FUN_MIN_FAN, FUN_MIN_TOP1_OUT, FUN_MIN_PT = 20, 10.0, .50, .70
SPIKE_MIN_FLOW_M, SPIKE_MIN_X, SPIKE_MIN_NEW, SPIKE_BASE_FLOOR = 250_000, 10.0, .50, 1_000
RT_MIN_AMT, RT_MIN_SHARE, PROCESSOR_MIN_CPTY = 250_000, .25, 2_000
PASS_THROUGH_NAICS = ("5223", "541214", "541191", "5313", "488510", "5613", "5242", "5231")
TYP = ["passthrough", "funnel", "split", "spike", "roundtrip"]


class DemoSource:
    def __init__(self, n=900, n_months=20, end="2026-08", seed=7):
        rng = np.random.default_rng(seed); self.rng = rng
        self.all_months = [ym_add(end, -k) for k in range(n_months - 1, -1, -1)]
        M = n_months; T = np.arange(M); self.M = M
        ids = [f"{100000 + i:018d}" for i in range(n)]; self.ids = ids
        names = [f"{rng.choice(WORDS_A)} {rng.choice(WORDS_B)}" for _ in range(n)]
        roman = ["", "II", "III", "IV", "V", "VI", "VII", "VIII"]
        names = [f"{nm} {roman[min(names[:i].count(nm), 7)]}".strip() for i, nm in enumerate(names)]
        rl = [f"R{i // 2 if rng.random() < .35 else 10000 + i:06d}" for i in range(n)]
        naics = [NAICS[j] for j in rng.integers(0, len(NAICS), n)]
        norm = np.exp(rng.normal(13.5, 1.5, n)); leaver = rng.random(n) < 0.22
        exit_m = np.where(leaver, rng.integers(M - 9, M + 5, n), 10 ** 6)
        kind = rng.choice(["competitive", "receipts", "drawdown"], n, p=[.4, .4, .2])
        self.mdm = {c: [f"M{c[-6:]}0"] for c in ids}
        self.cust_of = {f"M{c[-6:]}0": c for c in ids}
        hist = []
        for i in range(n):
            e = exit_m[i]; base_in = np.exp(rng.normal(np.log(norm[i]) + 0.4, .4)); base_n = max(3, rng.poisson(40))
            k = np.clip((T - (e - 6)) / 6, 0, 1) if leaver[i] else np.zeros(M)          # 0 → 1 over the 6 months before exit
            wob = rng.normal(0, .12, M)
            amt_in = base_in * (1 - .7 * k * (kind[i] != "drawdown")) * np.exp(wob)
            n_in = np.maximum(0, rng.poisson(base_n * (1 - .6 * k * (kind[i] != "drawdown"))))
            days_in = np.minimum(22, np.maximum(0, np.round(n_in * .4 + rng.normal(0, 1, M))))
            payers = np.maximum(0, np.round(base_n * .3 * (1 - .5 * k) + rng.normal(0, 1, M)))
            amt_out = base_in * np.exp(rng.normal(0, .06, M)) * (1 + 2.5 * (T == e - 1))
            gone = np.where(T >= e, .01, 1.0)                                          # after the exit the account goes quiet
            amt_in, amt_out, n_in = amt_in * gone, amt_out * gone, np.round(n_in * gone)
            n_fi = np.maximum(1, np.round(3 + 2 * k * (kind[i] == "competitive") + rng.normal(0, .8, M)))
            selfpay = amt_out * np.clip(.02 + .5 * k * (kind[i] == "competitive"), 0, 1)
            bal = norm[i] * np.exp(rng.normal(0, .08, M)) * np.where(T >= e, 0.002, np.where(T == e - 1, .45, 1 - .25 * k * (kind[i] == "drawdown")))
            eom = bal * np.where(T == e - 1, .5, 1.0) * np.exp(rng.normal(0, .02, M))
            top1 = rng.uniform(.12, .35); wire = rng.uniform(.05, .4)
            for t in range(M):
                hist.append(dict(cust_pwr_id=ids[i], rltn_pwr_id=rl[i], customer_name=names[i], month=self.all_months[t], bal=bal[t],
                                 bal_norm=norm[i], bal_eom=eom[t], bal_min=bal[t] * .8, bal_max=bal[t] * 1.2, live=int(t < e),
                                 amt_all_in=amt_in[t], amt_all_out=amt_out[t], ntxn_all_in=n_in[t], ntxn_all_out=int(n_in[t] * .8),
                                 active_days_in=days_in[t], ncpty_all_in=payers[t], ncpty_all_out=max(1.0, round(payers[t] * .7)),
                                 n_fi_out=n_fi[t], n_fi_in=3, selfpay_amt_out=selfpay[t], share_wire_in=float(np.clip(.2 + .1 * k[t] + rng.normal(0, .08), 0, 1)),
                                 share_check_in=float(np.clip(.3 - .1 * k[t] + rng.normal(0, .08), 0, 1)), net_flow_all=amt_in[t] - amt_out[t],
                                 top1_share_out=float(np.clip(top1 * np.exp(rng.normal(0, .1)), 0, 1)),
                                 new_amt_share_in=float(rng.uniform(0, .08)), new_amt_share_out=float(rng.uniform(0, .08)),
                                 share_WIRE_out=float(np.clip(wire + rng.normal(0, .03), 0, 1)),
                                 y=int(leaver[i] and t < e <= t + 6), exit_t=e, t=t))
        H = pd.DataFrame(hist)
        H["naics_cd"] = H.cust_pwr_id.map(dict(zip(ids, [c for c, _ in naics])))
        H["naics_desc"] = H.cust_pwr_id.map(dict(zip(ids, [d for _, d in naics])))
        H["segment_desc"] = H.cust_pwr_id.map(dict(zip(ids, rng.choice(["Middle Market", "Business Banking", "Corporate"], n, p=[.5, .35, .15]))))
        H["mdm_id"] = H.cust_pwr_id.map(lambda c: self.mdm[c][0])
        self.kind = dict(zip(ids, kind)); self.leaver = dict(zip(ids, leaver)); self.exit = dict(zip(ids, exit_m))
        self.H = H.sort_values(["cust_pwr_id", "t"]).reset_index(drop=True)
        self.fkind, self.fparam = {}, {}
        self._plant_fraud(rng)
        self.F = self._features(self.H)
        self._links(rng)
        self.FS, self.FH, self.FE = self._fraud_tables()

    # ═════════════════════════════════════════════════════════════════
    # suspicious-activity patterns planted in the same clients
    # ═════════════════════════════════════════════════════════════════
    def _plant_fraud(self, rng):
        M, ids = self.M, self.ids
        lv = [c for c in ids if self.leaver[c] and self.exit[c] > M]          # leaving, money still here: overlaps attrition
        st = [c for c in ids if not self.leaver[c]]
        rng.shuffle(lv); rng.shuffle(st)
        plan = [("split", 3, lv), ("roundtrip", 2, lv), ("spike", 1, lv),
                ("passthrough", 12, st), ("funnel", 8, st), ("pt_funnel", 3, st), ("spike", 5, st), ("split", 6, st),
                ("roundtrip", 5, st)]
        used = set()
        for k, cnt, pool in plan:
            for c in [x for x in pool if x not in used][:cnt]:
                self.fkind[c] = k; used.add(c)
        H = self.H.set_index(["cust_pwr_id", "t"])
        T = np.arange(M)
        for c, k in self.fkind.items():
            p = dict(kind=k); r = H.loc[c]
            if k in ("passthrough", "payroll", "funnel", "pt_funnel"):
                t0 = int(rng.integers(M - 10, M - 5)) if k != "payroll" else 0; p["t0"] = t0
                L = float(np.exp(rng.normal(14.6 if k != "payroll" else 15.5, .5)))
                on = T >= t0
                if k in ("funnel", "pt_funnel"):
                    payers = np.where(on, rng.integers(50, 120, M), np.maximum(3, r.ncpty_all_in.to_numpy()))
                    ticket = rng.uniform(1500, 4500); ntx_in = np.round(payers * rng.uniform(1, 1.5, M))
                    amt_in = np.where(on, ntx_in * ticket * np.exp(rng.normal(0, .08, M)), L * .06 * np.exp(rng.normal(0, .1, M)))
                    payees = np.where(on, rng.integers(1, 4, M), np.maximum(2, r.ncpty_all_out.to_numpy()))
                    ntx_out = np.where(on, rng.integers(2, 7, M), r.ntxn_all_out.to_numpy())
                    lo, hi = (.95, .99) if k == "pt_funnel" else (.88, .97)
                    out = amt_in * np.where(on, rng.uniform(lo, hi, M), rng.uniform(.6, .9, M))
                    top1 = np.where(on, rng.uniform(.65, .9, M), rng.uniform(.15, .35, M)); wire = np.where(on, rng.uniform(.88, .99, M), .2)
                    held = rng.uniform(1, 2.5) if k == "pt_funnel" else rng.uniform(3, 8)
                    new_in = np.where(on, rng.uniform(.3, .5, M), .03)
                else:
                    payroll = k == "payroll"
                    amt_in = np.where(on, L * np.exp(rng.normal(0, .08, M)), L * .07 * np.exp(rng.normal(0, .1, M)))
                    payers = np.where(on, rng.integers(5, 9, M) if payroll else rng.integers(8, 16, M), np.maximum(3, r.ncpty_all_in.to_numpy()))
                    ntx_in = np.where(on, rng.integers(10, 25, M), r.ntxn_all_in.to_numpy())
                    payees = np.where(on, rng.integers(2500, 6000, M) if payroll else rng.integers(5, 11, M), np.maximum(2, r.ncpty_all_out.to_numpy()))
                    ntx_out = np.where(on, payees * (1 if payroll else 2), r.ntxn_all_out.to_numpy())
                    out = amt_in * np.where(on, rng.uniform(.96, .995, M), rng.uniform(.6, .9, M))
                    top1 = np.where(on, rng.uniform(.002, .01, M) if payroll else rng.uniform(.22, .45, M), rng.uniform(.15, .35, M))
                    wire = np.where(on, .02 if payroll else rng.uniform(.6, .95, M), .2)
                    held = rng.uniform(1, 3)
                    new_in = np.where((T >= t0) & (T <= t0 + 1), .8, .05)
                bal = np.where(on, amt_in / 30 * held, amt_in / 30 * 20) * np.exp(rng.normal(0, .05, M))
                upd = dict(amt_all_in=amt_in, amt_all_out=out, ntxn_all_in=ntx_in, ntxn_all_out=ntx_out, ncpty_all_in=payers,
                           ncpty_all_out=payees, active_days_in=np.minimum(22, np.maximum(1, ntx_in // 2)), selfpay_amt_out=0.0,
                           top1_share_out=top1, share_WIRE_out=wire, new_amt_share_in=new_in, bal=bal, bal_eom=bal * .9,
                           bal_min=bal * .3, bal_max=bal * 2.5, bal_norm=float(np.median(bal[-12:])), net_flow_all=amt_in - out)
                if k == "payroll":
                    for c_ in ("naics_cd", "naics_desc"):
                        H.loc[c, c_] = {"naics_cd": "541214", "naics_desc": "Payroll services"}[c_]
            elif k == "spike":
                ts = [M - 1] if rng.random() < .6 else [M - 2]; p["t_spike"] = ts[0]
                base = float(rng.uniform(30_000, 60_000)); big = float(rng.uniform(1.5e6, 4e6))
                amt_in = base * np.exp(rng.normal(0, .15, M)); out = amt_in * rng.uniform(.8, 1.0, M)
                amt_in[ts] = big; out[ts] = big * rng.uniform(.85, .97)
                new_in = np.full(M, .03); new_in[ts] = rng.uniform(.85, .95); new_out = np.full(M, .03); new_out[ts] = rng.uniform(.88, .97)
                payers = np.maximum(2, r.ncpty_all_in.to_numpy() // 3); payers[ts] += 12
                payees = np.maximum(2, r.ncpty_all_out.to_numpy() // 3); payees[ts] += 4
                bal = np.maximum(r.bal.to_numpy() * .05, base * .8)
                upd = dict(amt_all_in=amt_in, amt_all_out=out, ncpty_all_in=payers, ncpty_all_out=payees, new_amt_share_in=new_in,
                           new_amt_share_out=new_out, net_flow_all=amt_in - out, selfpay_amt_out=0.0, bal=bal, bal_norm=float(np.median(bal)),
                           bal_eom=bal, bal_min=bal * .7, bal_max=np.where(np.isin(T, ts), big * .6, bal * 1.3))
                if not self.leaver[c]: upd.update(bal_norm=float(np.median(bal)))
            else:                                                   # split / roundtrip: ordinary flows, the pattern sits on top
                L = float(np.exp(rng.normal(13.9 if k == "roundtrip" else 13.3, .3)))
                amt_in = L * np.exp(rng.normal(0, .08, M)); out = amt_in * rng.uniform(.75, .95, M)
                upd = dict(amt_all_in=amt_in, amt_all_out=out, net_flow_all=amt_in - out, selfpay_amt_out=out * .01)
                if k == "split":
                    p.update(n_below=int(rng.integers(30, 110)), n_above=int(rng.integers(0, 6)), split_days=int(rng.integers(5, 20)))
                    p["amt_below"] = p["n_below"] * float(rng.uniform(9300, 9900))
                else:
                    p.update(n_rt=int(rng.integers(1, 4)), rt_frac=float(rng.uniform(.4, .8)))
            for col, v in upd.items():
                H.loc[c, col] = v
            self.fparam[c] = p
        self.H = H.reset_index()

    @staticmethod
    def _features(H):
        g = H.groupby("cust_pwr_id", sort=False)
        L = lambda c: np.log1p(H[c].clip(lower=0))
        def chg(s, a, b):
            base = s.groupby(H.cust_pwr_id).transform(lambda x: x.shift(a).rolling(b - a + 1, min_periods=2).mean())
            return s - base
        F = H[["cust_pwr_id", "rltn_pwr_id", "customer_name", "naics_desc", "segment_desc", "month", "bal", "bal_norm", "bal_eom", "t", "y", "exit_t"]].copy()
        F["mdm_ids"] = H.mdm_id
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

    # ── counterparties: one link list per client, amounts follow the client's monthly totals ──
    def _links(self, rng):
        M = self.M; H = self.H
        nm = dict(zip(H.cust_pwr_id, H.customer_name))
        self.links = {c: {"in": [], "out": []} for c in self.ids}
        cp_names = {}; used_names = {}
        def ext(prefix, c, j, shell=False):
            cid = f"X{prefix}{c[-6:]}{j:03d}"
            if cid not in cp_names:
                used = used_names.setdefault(c, {nm[c]})
                for _ in range(40):                                     # names unique within one client's network
                    nm_ = f"{rng.choice(SHELL_A)} {rng.choice(SHELL_B)}" if shell else f"{rng.choice(WORDS_A)} {rng.choice(WORDS_B)}"
                    if nm_ not in used: break
                used.add(nm_); cp_names[cid] = (nm_, str(rng.choice(BANKS)))
            return cid
        def add(c, d, cid, kind, w, months=None, n=1.0):
            self.links[c][d].append(dict(cpty_id=cid, kind=kind, w=float(w), months=months, n=n))
        def pnc(a, b, w_out, w_in, months=None, n=1.0):               # a pays b, both PNC clients
            add(a, "out", self.mdm[b][0], "CUST", w_out, months, n); add(b, "in", self.mdm[a][0], "CUST", w_in, months, n)
        others = [c for c in self.ids if c not in self.fkind]
        for c in self.ids:
            k = self.fkind.get(c); p = self.fparam.get(c, {})
            if k in ("funnel", "pt_funnel"):
                for j in range(70): add(c, "in", ext("F", c, j), "CPTY", rng.lognormal(0, .4), months=set(range(p["t0"], M)), n=1.3)
                for j in range(4): add(c, "in", ext("P", c, j), "CPTY", 4 * rng.lognormal(0, .3), months=set(range(0, p["t0"])), n=4)
                add(c, "out", ext("W", c, 0, shell=True), "CPTY", 8, months=set(range(p["t0"], M)), n=2)
                add(c, "out", ext("W", c, 1, shell=True), "CPTY", 2, months=set(range(p["t0"], M)), n=1)
                for j in range(3): add(c, "out", ext("Q", c, j), "CPTY", 1, months=set(range(0, p["t0"])), n=4)
            elif k in ("passthrough", "payroll"):
                n_in, n_out = (6, 60) if k == "payroll" else (12, 7)
                for j in range(n_in): add(c, "in", ext("P", c, j, shell=(k == "passthrough" and j < 3)), "CPTY", rng.lognormal(0, .5), n=2)
                for j in range(n_out): add(c, "out", ext("E" if k == "payroll" else "Q", c, j, shell=(k == "passthrough")), "CPTY",
                                           rng.lognormal(0, .3 if k == "payroll" else .6), n=40 if k == "payroll" else 2)
            elif k == "spike":
                ts = {p["t_spike"]}; base = set(range(M)) - ts
                for j in range(4): add(c, "in", ext("P", c, j), "CPTY", 1, months=base, n=3)
                for j in range(3): add(c, "out", ext("Q", c, j), "CPTY", 1, months=base, n=3)
                for j in range(12): add(c, "in", ext("S", c, j, shell=j < 6), "CPTY", rng.lognormal(0, .4), months=ts, n=1)
                for j in range(4): add(c, "out", ext("T", c, j, shell=True), "CPTY", rng.lognormal(0, .4), months=ts, n=1)
            else:
                n_in = int(np.clip(rng.poisson(10), 3, 20)); n_out = int(np.clip(rng.poisson(8), 3, 16))
                for j in range(n_in): add(c, "in", ext("P", c, j), "CPTY", rng.lognormal(0, .9), n=rng.integers(1, 6))
                for j in range(n_out): add(c, "out", ext("Q", c, j), "CPTY", rng.lognormal(0, .9), n=rng.integers(1, 6))
                if k == "split":
                    for j in range(2): add(c, "out", ext("Z", c, j), "CPTY", 1.2, months=set(range(M - 4, M)), n=p["n_below"] / 6)
                if k == "roundtrip":
                    p["rt"] = []
                    for j in range(p["n_rt"]):
                        cid = ext("R", c, j, shell=True); w = 6.0 / (j + 1)
                        add(c, "out", cid, "CPTY", w, months=set(range(M - 5, M)), n=2); add(c, "in", cid, "CPTY", w * .9, months=set(range(M - 5, M)), n=2)
                        p["rt"].append(cid)
                if k is None and rng.random() < .25:                  # ordinary trade between PNC clients
                    for _ in range(int(rng.integers(1, 3))):
                        b = others[int(rng.integers(0, len(others)))]
                        if b != c: pnc(c, b, rng.uniform(.5, 2), rng.uniform(.3, 1), n=3)
        # chains the network view makes visible: pass-through → pass-through → pass-through, and a funnel feeding one
        pts = [c for c, k in self.fkind.items() if k == "passthrough"]; fns = [c for c, k in self.fkind.items() if k == "funnel"]
        for a, b in zip(pts[:2], pts[1:3]): pnc(a, b, 6, 5)
        if fns and len(pts) > 3: pnc(fns[0], pts[3], 6, 4)
        if len(fns) > 1 and len(pts) > 4: pnc(fns[1], pts[4], 5, 3); pnc(pts[4], pts[0], 4, 3)
        self.cp_names = cp_names
        for c in self.ids:                                          # a handful of normal PNC payers for the flagged clients too
            if c in self.fkind and rng.random() < .5:
                b = others[int(rng.integers(0, len(others)))]; pnc(b, c, rng.uniform(.3, 1), rng.uniform(.3, 1), n=2)

    def _flows(self, c, months):
        """Counterparty rows for one client and a list of month indexes (amounts follow its monthly totals)."""
        h = self.H[self.H.cust_pwr_id == c].set_index("t"); r = np.random.default_rng(int(c[-6:]))
        rows = {"in": [], "out": []}
        for d, col in (("in", "amt_all_in"), ("out", "amt_all_out")):
            L = self.links[c][d]
            for t in months:
                act = [x for x in L if x["months"] is None or t in x["months"]]
                tot = sum(x["w"] for x in act)
                if not act or tot <= 0: continue
                amt = float(h.loc[t, col]) * (.97 if d == "out" else 1.0)
                for x in act:
                    a = amt * x["w"] / tot * float(np.exp(r.normal(0, .08)))
                    rows[d].append(dict(month=self.all_months[t], cpty_id=x["cpty_id"], kind=x["kind"], amount=a,
                                        n_txn=max(1, int(round(x["n"] * np.exp(r.normal(0, .2))))), is_self=False))
        return pd.DataFrame(rows["in"]), pd.DataFrame(rows["out"])

    def _names(self, ids):
        rows = []
        for i in dict.fromkeys(ids):
            if i in self.cust_of:
                c = self.cust_of[i]; rows.append(dict(cpty_id=i, name=self.H.customer_name[self.H.cust_pwr_id == c].iloc[0], bank="PNC",
                                                    customer_pwr_id=c.lstrip("0")))
            elif i in self.cp_names:
                rows.append(dict(cpty_id=i, name=self.cp_names[i][0], bank=self.cp_names[i][1], customer_pwr_id=None))
        return pd.DataFrame(rows, columns=["cpty_id", "name", "bank", "customer_pwr_id"])

    # ── fraud tables: the notebook's §2 and §5 in pandas, on the same synthetic clients ──
    def _fraud_tables(self):
        H = self.H.sort_values(["cust_pwr_id", "t"]).copy(); M = self.M; SC_T = [M - 3, M - 2, M - 1]
        H["out_adj"] = (H.amt_all_out - H.selfpay_amt_out.fillna(0)).clip(lower=0)
        H["pt_m"] = np.minimum(H.amt_all_in, H.out_adj) / np.maximum(np.maximum(H.amt_all_in, H.out_adj), 1e-9)
        H["flow_m"] = H.amt_all_in + H.amt_all_out
        g = H.groupby("cust_pwr_id", sort=False)
        r3 = lambda s: s.groupby(H.cust_pwr_id).transform(lambda x: x.rolling(3, min_periods=1).sum())
        a3 = lambda s: s.groupby(H.cust_pwr_id).transform(lambda x: x.rolling(3, min_periods=1).mean())
        X = H[["cust_pwr_id", "mdm_id", "rltn_pwr_id", "customer_name", "naics_cd", "naics_desc", "month", "t", "flow_m", "amt_all_in", "amt_all_out"]].copy()
        X["in_3m"] = r3(H.amt_all_in); X["out_3m"] = r3(H.out_adj); X["out_raw_3m"] = r3(H.amt_all_out)
        X["flow_3m"] = X.in_3m + X.out_raw_3m
        X["pt_3m"] = np.minimum(X.in_3m, X.out_3m) / np.maximum(np.maximum(X.in_3m, X.out_3m), 1e-9)
        X["pt_months"] = r3((H.pt_m >= PT_MONTH_MIN).astype(int))
        X["bal_3m"] = a3(H.bal); X["days_held"] = X.bal_3m / (X.in_3m / 91.0); X["turnover"] = X.in_3m / X.bal_3m
        X["payers_avg"] = a3(H.ncpty_all_in); X["payees_avg"] = a3(H.ncpty_all_out)
        X["fan"] = X.payers_avg / X.payees_avg.clip(lower=1); X["top1_out"] = a3(H.top1_share_out)
        X["ticket_in"] = X.in_3m / r3(H.ntxn_all_in).clip(lower=1); X["ticket_out"] = X.out_raw_3m / r3(H.ntxn_all_out).clip(lower=1)
        X["ticket_ratio"] = X.ticket_out / X.ticket_in
        X["wire_out"] = r3(H.amt_all_out * H.share_WIRE_out) / X.out_raw_3m
        X["new_payer_share"] = r3(H.amt_all_in * H.new_amt_share_in) / X.in_3m
        X["ext_in_share"] = 1.0
        X["selfpay_share_3m"] = r3(H.selfpay_amt_out) / X.out_raw_3m
        X["base"] = H.groupby("cust_pwr_id").flow_m.transform(lambda x: x.shift(2).rolling(6, min_periods=1).median())
        X["new_share_m"] = (H.amt_all_in * H.new_amt_share_in + H.amt_all_out * H.new_amt_share_out) / H.flow_m.clip(lower=1)
        X["spike_x"] = X.flow_m / X.base.fillna(0).clip(lower=SPIKE_BASE_FLOOR)
        X["existed_before"] = X.base.fillna(0) >= SPIKE_BASE_FLOOR
        X["base_active"] = H.groupby("cust_pwr_id").flow_m.transform(lambda x: (x > 0).astype(int).shift(2).rolling(6, min_periods=1).sum())
        X["family_share_3m"] = 0.0
        X = X[X.t.isin(SC_T) & (X.flow_3m >= MIN_FLOW_3M)].copy()
        # planted split / round-trip measurements
        for c in ["split_thr", "n_below", "n_above", "amt_below", "split_days", "heap", "split_z", "rt_amt", "n_rt", "rt_top_amt"]: X[c] = np.nan
        X["rt_top_other"] = None; X["rt_top_name"] = None
        for c, p in self.fparam.items():
            m = X.cust_pwr_id == c
            if p["kind"] == "split":
                f = np.where(X.loc[m, "t"] == M - 1, 1.0, np.where(X.loc[m, "t"] == M - 2, .8, .55))
                nb = np.round(p["n_below"] * f); na = np.round(p["n_above"] * f)
                X.loc[m, ["split_thr", "n_below", "n_above", "amt_below", "split_days"]] = np.c_[np.full(m.sum(), 1e4), nb, na, p["amt_below"] * f, np.round(p["split_days"] * f)]
                X.loc[m, "heap"] = (nb + 1) / (na + 1); X.loc[m, "split_z"] = (nb - na) / np.sqrt(np.maximum(nb + na, 1))
            if p["kind"] == "roundtrip":
                X.loc[m, "rt_amt"] = X.loc[m, "out_raw_3m"] * p["rt_frac"] * np.where(X.loc[m, "t"] == M - 3, .7, 1.0)
                X.loc[m, "n_rt"] = p["n_rt"]; X.loc[m, "rt_top_amt"] = X.loc[m, "rt_amt"] * (.6 if p["n_rt"] > 1 else 1.0)
                X.loc[m, "rt_top_other"] = p["rt"][0]; X.loc[m, "rt_top_name"] = self.cp_names[p["rt"][0]][0]
        X["rt_share"] = X.rt_amt.fillna(0) / X.out_raw_3m
        # peers: five size bands; within a band, the same industry when it has enough clients, else the whole band
        X["size_band"] = X.groupby("month").flow_3m.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
        n_ind = X.groupby(["month", "naics_desc", "size_band"]).cust_pwr_id.transform("count")
        by_ind = n_ind >= 20
        X["peer_group"] = np.where(by_ind, X.naics_desc, "all industries")
        X["peer_n"] = np.where(by_ind, n_ind, X.groupby(["month", "size_band"]).cust_pwr_id.transform("count"))
        for c in ["pt_3m", "turnover", "flow_3m", "fan", "top1_out", "ticket_ratio", "rt_share"]:
            ind = X.groupby(["month", "naics_desc", "size_band"])[c].rank(pct=True, method="average")
            band = X.groupby(["month", "size_band"])[c].rank(pct=True, method="average")
            X[f"pr_{c}"] = np.where(by_ind, ind, band)
        X["processor_like"] = (X.payers_avg >= PROCESSOR_MIN_CPTY) | (X.payees_avg >= PROCESSOR_MIN_CPTY)
        X["industry_pt"] = X.naics_cd.astype(str).str.startswith(PASS_THROUGH_NAICS)
        X["treasury_sweep"] = X.selfpay_share_3m.fillna(0) >= .5
        guard = np.where(X.processor_like, .5, 1.0) * np.where(X.industry_pt, .6, 1.0)
        z = lambda c: X[c].fillna(0)
        gate = {"passthrough": (X.pt_3m >= PT_MIN) & (X.pt_months >= PT_MONTHS_REQ) & X.days_held.notna() & (X.days_held <= PT_MAX_DAYS_HELD),
                "funnel": (X.payers_avg >= FUN_MIN_PAYERS) & (X.fan >= FUN_MIN_FAN) & (X.top1_out >= FUN_MIN_TOP1_OUT) & (X.pt_3m >= FUN_MIN_PT),
                "split": (z("n_below") >= 10) & (z("heap") >= 3) & (z("split_z") >= 4) & (z("split_days") >= 3),
                "spike": (X.flow_m >= SPIKE_MIN_FLOW_M) & (X.spike_x >= SPIKE_MIN_X) & (z("new_share_m") >= SPIKE_MIN_NEW) & (z("base") >= 25_000) & (z("base_active") >= 5),
                "roundtrip": (z("rt_amt") >= RT_MIN_AMT) & (z("rt_share") >= RT_MIN_SHARE)}
        sc = {"passthrough": 100 * guard * (.4 * z("pr_pt_3m") + .4 * X.pr_turnover.fillna(z("pr_flow_3m")) + .2 * z("pr_flow_3m")),
              "funnel": 100 * guard * (.35 * z("pr_fan") + .35 * z("pr_top1_out") + .3 * X.pr_ticket_ratio.fillna(.5)),
              "split": 100 * np.minimum(1, .5 * np.minimum(z("heap") / 10, 1) + .3 * np.minimum(z("n_below") / 30, 1) + .2 * np.minimum(z("split_days") / 5, 1)),
              "spike": 100 * np.minimum(1, .5 * np.minimum(np.log10(X.spike_x.clip(lower=1)) / 2, 1) + .5 * z("new_share_m")),
              "roundtrip": 100 * (.5 + .3 * np.minimum((z("rt_share") - RT_MIN_SHARE) / .5, 1) + .2 * z("pr_flow_3m"))}
        for k in TYP: X[f"score_{k}"] = np.where(gate[k], sc[k], 0.0)
        S_ = X[[f"score_{k}" for k in TYP]]
        X["n_typ"] = (S_ > 0).sum(axis=1); top = S_.max(axis=1)
        X["score"] = np.where(X.n_typ > 0, np.minimum(100, top + 5 * (X.n_typ - 1)), 0.0)
        X["primary"] = np.where(X.n_typ > 0, S_.idxmax(axis=1).str.replace("score_", ""), None)
        X["typologies"] = [",".join(k for k in TYP if r[f"score_{k}"] > 0) for _, r in X.iterrows()]
        usd = {"passthrough": np.minimum(X.in_3m, X.out_3m), "funnel": X.in_3m, "split": z("amt_below"), "spike": X.flow_m, "roundtrip": z("rt_amt")}
        X["usd_involved"] = np.max(np.c_[[np.where(X[f"score_{k}"] > 0, usd[k], 0.0) for k in TYP]].T, axis=1)
        # tiers by review capacity (as in the notebook; demo capacity scaled to the small synthetic world)
        best = np.zeros(len(X), dtype=int)
        for k in TYP:
            rk = X.groupby("month")[f"score_{k}"].rank(ascending=False, method="first")
            sk = X[f"score_{k}"].to_numpy()
            lvl = np.where((rk <= 8) & (sk >= 70), 3, np.where((rk <= 20) & (sk >= 50), 2, np.where(sk > 0, 1, 0)))
            best = np.maximum(best, lvl)
        X["tier"] = np.select([best == 3, best == 2, best == 1], ["High", "Elevated", "Watch"], default=None)
        flagged = X.loc[X.score > 0, "mdm_id"].unique()
        S = X[X.mdm_id.isin(flagged)].copy()
        S["customer_pwr_id"] = S.cust_pwr_id.str.lstrip("0"); S["pwr"] = S.customer_pwr_id; S["master_pwr_id"] = S.rltn_pwr_id
        S["party_type"] = "O"; S["bal_3m"] = S.bal_3m
        S = S.drop(columns=["cust_pwr_id", "rltn_pwr_id", "t", "amt_all_in", "amt_all_out"]).reset_index(drop=True)
        # history: 12 months per flagged entity
        Hh = H[H.mdm_id.isin(flagged) & (H.t >= M - 12)]
        FH = Hh[["mdm_id", "month", "amt_all_in", "amt_all_out", "out_adj", "ntxn_all_in", "ntxn_all_out", "ncpty_all_in", "ncpty_all_out",
                 "top1_share_out", "selfpay_amt_out", "new_amt_share_in", "new_amt_share_out", "share_WIRE_out", "n_fi_out", "active_days_in",
                 "bal", "pt_m", "flow_m"]].rename(columns={"bal": "bal_avg"}).reset_index(drop=True)
        # evidence: near-threshold payment days (split) and both directions of round-trip pairs
        ev = []; r = np.random.default_rng(11)
        for c, p in self.fparam.items():
            mdm = self.mdm[c][0]
            if mdm not in flagged: continue
            if p["kind"] == "split":
                zs = [x["cpty_id"] for x in self.links[c]["out"] if x["cpty_id"].startswith("XZ")]
                for t in range(M - 3, M):
                    f = {M - 1: 1.0, M - 2: .8, M - 3: .55}[t]; nb = int(round(p["n_below"] * f / 3)); nd = int(round(p["split_days"] * f / 3))
                    days = sorted(r.choice(np.arange(1, 29), size=min(28, nb), replace=False))
                    for j, d in enumerate(days):
                        is_split = j < nd; n = int(r.integers(2, 4)) if is_split else 1
                        cid = zs[j % len(zs)] if zs else f"XZ{c[-6:]}000"
                        ev.append(dict(mdm_id=mdm, typology="split", month=self.all_months[t], txn_date=f"{self.all_months[t]}-{d:02d}",
                                       direction="out", cpty_id=cid, cpty_name=self.cp_names.get(cid, ("", ""))[0],
                                       bank=self.cp_names.get(cid, ("", "unknown"))[1], n_txn=float(n),
                                       amount=float(sum(r.uniform(9200, 9990, n))), amount_back=np.nan, threshold=1e4, split_day=bool(is_split)))
            if p["kind"] == "roundtrip":
                tot = float(S.loc[(S.mdm_id == mdm) & (S.month == self.all_months[-1]), "rt_amt"].fillna(0).sum()) or 3e5
                ws = [6.0 / (j + 1) for j in range(p["n_rt"])]
                for cid, w in zip(p["rt"], ws):
                    sent = tot * w / sum(ws) / .9
                    ev.append(dict(mdm_id=mdm, typology="roundtrip", month=self.all_months[-1], txn_date=None, direction="ext", cpty_id=cid,
                                   cpty_name=self.cp_names[cid][0], bank=self.cp_names[cid][1], n_txn=6.0, amount=sent, amount_back=sent * .9,
                                   threshold=np.nan, split_day=None))
        FE = pd.DataFrame(ev, columns=["mdm_id", "typology", "month", "txn_date", "direction", "cpty_id", "cpty_name", "bank", "n_txn",
                                       "amount", "amount_back", "threshold", "split_day"])
        return S, FH, FE

    # ═════════════════════════════════════════════════════════════════
    # the same methods as ImpalaSource
    # ═════════════════════════════════════════════════════════════════
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

    def fraud_signals(self):
        return self.FS.copy()

    def fraud_history(self, mdm_ids):
        return self.FH[self.FH.mdm_id.isin(list(mdm_ids))].copy()

    def fraud_evidence(self, mdm_ids):
        return self.FE[self.FE.mdm_id.isin(list(mdm_ids))].copy()

    def latest_month(self):
        return self.all_months[-1]

    def metrics_history(self, mdm_ids, start_month):
        h = self.H[self.H.mdm_id.isin(list(mdm_ids)) & (self.H.month >= start_month)]
        return h.drop(columns=["y", "exit_t", "t", "bal", "bal_norm", "bal_eom", "bal_min", "bal_max", "live"]).assign(
            customer_pwr_id=lambda d: d.cust_pwr_id.str.lstrip("0"))

    def counterparties(self, mdm_ids, end_month, n_months=6):
        e = self.all_months.index(end_month) if end_month in self.all_months else self.M - 1
        ts = list(range(max(0, e - n_months + 1), e + 1)); inn, out = [], []
        for m in mdm_ids:
            c = self.cust_of.get(m)
            if c is None: continue
            i, o = self._flows(c, ts); inn.append(i); out.append(o)
        cols = ["month", "cpty_id", "kind", "amount", "n_txn", "is_self"]
        inn = pd.concat(inn, ignore_index=True) if inn else pd.DataFrame(columns=cols)
        out = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
        names = self._names(list(inn.get("cpty_id", [])) + list(out.get("cpty_id", [])))
        return inn.drop(columns=["is_self"], errors="ignore").merge(names, on="cpty_id", how="left"), out.merge(names, on="cpty_id", how="left")

    def internal_ties(self, mdm_ids, end_month, n_months=6):
        ids = set(mdm_ids); rows = []
        for m in ids:
            c = self.cust_of.get(m)
            if c is None: continue
            for x in self.links[c]["out"]:
                if x["kind"] == "CUST" and x["cpty_id"] in ids and x["cpty_id"] != m:
                    _, o = self._flows(c, list(range(self.M - n_months, self.M)))
                    a = float(o.loc[o.cpty_id == x["cpty_id"], "amount"].sum()) if len(o) else 0.0
                    rows.append(dict(src_id=m, dst_id=x["cpty_id"], amount=a, n_txn=float(n_months)))
        return pd.DataFrame(rows, columns=["src_id", "dst_id", "amount", "n_txn"])

    def search_customers(self, text, limit=50):
        t = str(text).strip().lower()
        if not t: return pd.DataFrame(columns=["mdm_id", "customer_pwr_id", "customer_name", "naics_desc"])
        h = self.H[self.H.t == self.M - 1]
        hit = h[h.customer_name.str.lower().str.contains(t, regex=False) | (h.cust_pwr_id.str.lstrip("0") == t.lstrip("0"))
                | (h.mdm_id.str.lower() == t)]
        return hit.assign(customer_pwr_id=hit.cust_pwr_id.str.lstrip("0"))[["mdm_id", "customer_pwr_id", "customer_name", "naics_desc"]].head(limit)

    def lookup(self, key):
        h = self.H[self.H.t == self.M - 1]
        r = h[h.mdm_id == key[2:]] if str(key).startswith("M:") else h[h.cust_pwr_id.map(norm_pwr) == key]
        if not len(r): return None
        r = r.iloc[0]
        return dict(key=key, name=r.customer_name, pwr=r.cust_pwr_id.lstrip("0"), mdm_ids=[r.mdm_id], naics_desc=r.naics_desc,
                    master_pwr_id=r.rltn_pwr_id)


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
