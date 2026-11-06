"""
paygraph_synth.py: synthetic business payment network for the graph ML workshop.

Everything here is invented: names, amounts and relationships. Any resemblance
to a real company is coincidental. The network is built so that every lab has a
known answer to check against.

Planted structure
-----------------
* 12 ecosystems (communities) of businesses, each a mix of sectors in a region
* 6 hubs that almost everyone pays or is paid by (payroll processor, tax agency,
  utility, card acquirer, insurer, fuel card): they turn the picture into a hairball
* individuals (employees, tenants, bill payers, a little person-to-person)
* local-structure cases: funnel, pass-through, round-trip
* legitimate look-alikes: property manager (looks like a funnel), two-way trader
  (looks like a round-tripper)
* vendor bank-account changes for the payee-consensus check
* churn with contagion: businesses whose counterparties left are more likely to leave
* second half of the year: ecosystem E02 shrinks, E05 and E12 merge, E11 is born

Usage
-----
    python paygraph_synth.py --out data --seed 7

Outputs (CSV)
-------------
    nodes.csv                 one row per account holder (what the bank knows)
    edges_monthly.csv         month, src, dst, amount, n_txn (payer -> payee)
    transactions_2026-08.csv  daily transactions for the last month
    vendor_payments.csv       payer -> vendor payments with the vendor account used
    truth.csv                 planted answers: ecosystem, archetype, churn label
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

MONTHS = [str(p) for p in pd.period_range("2025-09", "2026-08", freq="M")]
N_MONTHS = len(MONTHS)          # 12
H2_START = 6                    # 2026-03: second half starts
CUTOFF = 8                      # 2026-05: last month a churn model may see
FOCUS = 11                      # 2026-08: month with daily transactions

SECTORS = {
    "11": "Agriculture", "22": "Utilities", "23": "Construction",
    "31": "Food manufacturing", "32": "Chemical and pharma manufacturing",
    "33": "Machinery and parts manufacturing", "42": "Wholesale trade",
    "44": "Retail trade", "48": "Transportation", "52": "Finance and insurance",
    "53": "Real estate", "54": "Professional services", "55": "Holding companies",
    "56": "Admin and support services", "62": "Health care", "72": "Food services",
    "81": "Other services", "92": "Public administration",
}

# id, name, region, sector mix, size
ECOS = [
    ("E01", "Pittsburgh health system", "PA-West", {"62": .45, "42": .20, "54": .15, "56": .10, "53": .10}, 230),
    ("E02", "Ohio auto-parts supply chain", "OH", {"33": .50, "42": .20, "48": .20, "54": .10}, 220),
    ("E03", "Philadelphia construction", "PA-East", {"23": .50, "42": .20, "53": .15, "54": .15}, 210),
    ("E04", "New Jersey pharma corridor", "NJ", {"32": .45, "54": .30, "42": .15, "56": .10}, 180),
    ("E05", "Chicago food distribution", "IL", {"42": .40, "72": .35, "48": .15, "31": .10}, 200),
    ("E06", "Mid-Atlantic real estate", "MD-DC", {"53": .50, "23": .20, "56": .20, "54": .10}, 170),
    ("E07", "Kentucky logistics hub", "KY", {"48": .55, "42": .20, "81": .15, "33": .10}, 170),
    ("E08", "Michigan machinery", "MI", {"33": .55, "42": .25, "54": .20}, 160),
    ("E09", "Professional services network", "Multi-state", {"54": .50, "52": .25, "55": .15, "56": .10}, 170),
    ("E10", "Retail chain ecosystem", "Multi-state", {"44": .50, "42": .30, "48": .20}, 170),
    ("E11", "Data-center build-out", "VA-OH", {"23": .40, "33": .25, "22": .15, "42": .20}, 110),
    ("E12", "Midwest agri-food processing", "IN-IA", {"11": .35, "31": .40, "42": .25}, 160),
]
NEIGH = {
    "E01": ["E09", "E04"], "E02": ["E08", "E07"], "E03": ["E06", "E11"],
    "E04": ["E09", "E01"], "E05": ["E12", "E10"], "E06": ["E03", "E09"],
    "E07": ["E02", "E10"], "E08": ["E02", "E11"], "E09": ["E01", "E04", "E06"],
    "E10": ["E07", "E05"], "E11": ["E03", "E08", "E02"], "E12": ["E05", "E07"],
}

PREFIX = ("Allegheny Riverbend Keystone Summit Lakeside Ironbridge Bluegrass Harbor Prairie Northgate "
          "Copperline Maplewood Granite Beacon Liberty Meridian Cedar Oakmont Westfield Pioneer Redstone "
          "Silverlake Fairview Clearwater Highpoint Brookside Stonegate Lantern Ashford Wexford Juniper "
          "Sterling Foxhall Elmhurst Crestview Bridgeway Kingsley Hollow Creek Tristate Midland Union "
          "Heritage Frontier Eastgate Larkspur Monarch Sycamore Halcyon Ridgeline Vantage").split()
WORDS = {
    "11": ["Farms", "Growers", "Grain Co-op", "Orchards"],
    "22": ["Electric", "Energy Services", "Power Systems"],
    "23": ["Builders", "Concrete", "Electrical Contractors", "Mechanical", "Construction"],
    "31": ["Foods", "Dairy", "Packing", "Bakery Supply"],
    "32": ["Pharma Labs", "Biologics", "Chemical", "Compounding"],
    "33": ["Tooling", "Auto Components", "Precision Parts", "Castings", "Machine Works"],
    "42": ["Supply", "Distributors", "Wholesale", "Trading"],
    "44": ["Outfitters", "Home Store", "Market", "Retail Group"],
    "48": ["Freight", "Logistics", "Trucking", "Transport"],
    "52": ["Capital", "Insurance Agency", "Financial"],
    "53": ["Realty", "Property Management", "Properties", "Real Estate"],
    "54": ["Consulting", "Analytics", "Engineering", "Accounting", "Advisors"],
    "55": ["Holdings", "Group"],
    "56": ["Staffing", "Facility Services", "Janitorial", "Security Services"],
    "62": ["Medical Group", "Surgical Center", "Clinic", "Health Partners", "Imaging"],
    "72": ["Kitchen", "Catering", "Grill", "Restaurant Group"],
    "81": ["Services", "Repair", "Laundry"],
    "92": ["Authority"],
}
SUFFIX = ["LLC", "Inc", "Co", "Group", "Partners", "Corp"]


class _Names:
    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def make(self, sector):
        for _ in range(200):
            n = f"{self.rng.choice(PREFIX)} {self.rng.choice(WORDS[sector])} {self.rng.choice(SUFFIX)}"
            if n not in self.used:
                self.used.add(n)
                return n
        n = f"{self.rng.choice(PREFIX)} {self.rng.choice(WORDS[sector])} {len(self.used)}"
        self.used.add(n)
        return n


def generate(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = _Names(rng)

    # ------------------------------------------------------------------ nodes
    nodes = []          # dicts
    counter = {"O": 0, "P": 0}
    def add_node(kind, eco, sector, region, size, archetype="normal", name=None):
        counter[kind] += 1
        nid = f"{'B' if kind == 'O' else 'P'}{counter[kind]:04d}"
        nodes.append(dict(node_id=nid, party_type=kind, eco=eco, sector_code=sector, region=region,
                          size=float(size), archetype=archetype,
                          name=name if name else (names.make(sector) if kind == "O" else f"Individual {nid}")))
        return nid

    for eco, _, region, mix, n in ECOS:
        codes, w = list(mix), np.array(list(mix.values()))
        for s in rng.choice(codes, size=n, p=w / w.sum()):
            add_node("O", eco, s, region, rng.lognormal(np.log(40_000), 0.9))

    hubs = [
        ("Keystone Payroll Services", "54", "payroll"), ("Commonwealth Tax Authority", "92", "tax"),
        ("Tri-State Power and Light", "22", "utility"), ("Summit Card Acquiring", "52", "card"),
        ("Allied Mutual Insurance", "52", "insurer"), ("FleetLine Fuel Card", "52", "fuel"),
    ]
    hub_id = {}
    for nm, sec, key in hubs:
        hub_id[key] = add_node("O", "HUB", sec, "Multi-state", 5e6, archetype="hub", name=nm)

    regions = [e[2] for e in ECOS]
    for _ in range(1500):
        add_node("P", "IND", "", rng.choice(regions), rng.lognormal(np.log(4000), 0.4), archetype="individual")

    N = pd.DataFrame(nodes)
    biz = N[(N.party_type == "O") & (N.archetype == "normal")]
    eco_members = {e: biz[biz.eco == e].node_id.to_numpy() for e, *_ in ECOS}
    attract = {nid: a for nid, a in zip(biz.node_id, rng.pareto(1.6, len(biz)) + 1)}
    size = dict(zip(N.node_id, N["size"]))
    sector_of = dict(zip(N.node_id, N.sector_code))
    eco_of = dict(zip(N.node_id, N.eco))

    # ------------------------------------------------------------------ edges
    E = []   # (src, dst, amt, p, start, end, ntx, tag)
    def edge(s, d, amt, p=0.9, start=0, end=N_MONTHS - 1, ntx=2.0, tag="normal"):
        if s != d:
            E.append((s, d, float(amt), float(p), int(start), int(end), float(ntx), tag))

    def pick(eco, k, exclude):
        pool = eco_members[eco]
        w = np.array([attract[x] for x in pool])
        w = w / w.sum()
        out = []                                   # a list, not a set: keeps runs reproducible
        for x in rng.choice(pool, size=min(len(pool), k * 3), p=w, replace=True):
            if x not in exclude and x not in out:
                out.append(x)
            if len(out) >= k:
                break
        return out

    all_ecos = [e for e, *_ in ECOS]
    for nid, eco in zip(biz.node_id, biz.eco):
        k = int(np.clip(round(rng.lognormal(np.log(4.5), 0.55)), 1, 30))
        p_same = 0.65 if eco == "E11" else 0.82
        targets = []
        for _ in range(k):
            u = rng.random()
            if u < p_same:
                targets.append(eco)
            elif u < 0.96:
                targets.append(rng.choice(NEIGH[eco]))
            else:
                targets.append(rng.choice(all_ecos))
        extra = list(rng.choice(NEIGH["E11"], 2)) if eco == "E11" else []
        chosen = set()
        for d in [pick(t, 1, {nid}) for t in extra]:
            if d:                                   # E11 side work in H1 that winds down in H2
                chosen.add(d[0])
                edge(nid, d[0], size[nid] / k * rng.lognormal(0, 0.6), p=.85, end=H2_START - 1, ntx=2)
        for t in sorted(set(targets)):
            for d in pick(t, targets.count(t), {nid} | chosen):
                chosen.add(d)
                start = H2_START if (eco == "E11" and eco_of[d] == "E11") else 0
                amt = size[nid] / k * rng.lognormal(0, 0.6)
                edge(nid, d, amt, p=rng.beta(6, 1.5), start=start, ntx=1 + rng.poisson(1.5))
                if rng.random() < 0.04:
                    edge(d, nid, amt * rng.uniform(.1, .5), p=rng.beta(4, 2), start=start, ntx=1.5)

    # E11 is born in H2: build-out contracts wire its members to each other
    e11 = eco_members["E11"]
    for nid in e11:
        for d in rng.choice(e11, 6, replace=False):
            edge(nid, d, size[nid] * rng.uniform(.2, .6), p=.92, start=H2_START, ntx=2, tag="birth")

    # E05 <-> E12 merge: a new supply contract wires the two together in H2
    for _ in range(240):
        a, b = rng.choice(eco_members["E05"]), rng.choice(eco_members["E12"])
        if rng.random() < .7:
            edge(a, b, size[a] * rng.uniform(.05, .25), p=.9, start=H2_START, tag="merge")
        else:
            edge(b, a, size[b] * rng.uniform(.05, .2), p=.9, start=H2_START, tag="merge")

    # hubs
    bids = biz.node_id.to_numpy()
    for nid in bids:
        s, sec = size[nid], sector_of[nid]
        if rng.random() < .75: edge(nid, hub_id["payroll"], .35 * s, p=.97, ntx=2)
        if rng.random() < .85: edge(nid, hub_id["tax"], .08 * s, p=.6, ntx=1)
        if rng.random() < .70: edge(nid, hub_id["utility"], .03 * s, p=.95, ntx=1)
        if sec in ("44", "72", "62", "81") or rng.random() < .08: edge(hub_id["card"], nid, .5 * s, p=.98, ntx=4)
        if rng.random() < .50: edge(nid, hub_id["insurer"], .04 * s, p=.9, ntx=1)
        if rng.random() < .05: edge(hub_id["insurer"], nid, .3 * s, p=.2, ntx=1)
        if sec in ("48", "23"): edge(nid, hub_id["fuel"], .06 * s, p=.95, ntx=3)

    inds = N[N.archetype == "individual"].node_id.to_numpy()
    for pid in inds:
        if rng.random() < .8: edge(hub_id["payroll"], pid, rng.uniform(2000, 6000), p=.98, ntx=2)
        if rng.random() < .6: edge(pid, hub_id["utility"], rng.uniform(80, 300), p=.95, ntx=1)
        if rng.random() < .3:
            edge(pid, rng.choice(inds), rng.uniform(20, 400), p=.4, ntx=1, tag="p2p")

    # ------------------------------------------------------- planted cases
    def new_biz(eco, sector, sz, arch):
        region = dict((e[0], e[2]) for e in ECOS)[eco]
        nid = add_node("O", eco, sector, region, sz, archetype=arch)
        size[nid], sector_of[nid], eco_of[nid] = sz, sector, eco
        return nid

    def normal_cover(nid, eco, k=2):
        for d in pick(eco, k, {nid}):
            edge(nid, d, size[nid] * rng.uniform(.1, .3), p=.8)
        edge(nid, hub_id["utility"], size[nid] * .03, p=.95, ntx=1)
        edge(nid, hub_id["tax"], size[nid] * .05, p=.6, ntx=1)

    tenants = rng.permutation(inds)
    used_tenants = 0
    arche = {"funnel": [], "pass_through": [], "round_trip": [], "rt_partner": [], "property_manager": [], "two_way_trader": []}

    for i in range(6):                                 # funnels: new behaviour from 2026-04
        eco = rng.choice(["E01", "E03", "E05", "E07", "E09", "E10"][i:i + 1] or all_ecos)
        f = new_biz(eco, rng.choice(["81", "44"]), 9000, "funnel")
        arche["funnel"].append(f)
        normal_cover(f, eco)
        for pid in rng.choice(inds, rng.integers(35, 61), replace=False):
            edge(pid, f, rng.uniform(300, 2500), p=.85, start=7, ntx=1 + rng.integers(0, 3), tag="funnel_in")
        far = [e for e in all_ecos if e != eco and e not in NEIGH[eco]]
        dest = rng.choice(eco_members[rng.choice(far)])
        edge(f, dest, 1.0, p=1.0, start=7, ntx=4, tag="funnel_out")      # amount set from inflows later

    for i in range(6):                                 # pass-through: from 2026-03
        eco = all_ecos[(2 * i + 1) % 12]
        t = new_biz(eco, rng.choice(["54", "42"]), 20000, "pass_through")
        arche["pass_through"].append(t)
        normal_cover(t, eco)
        payers = [rng.choice(eco_members[e]) for e in rng.choice(all_ecos, rng.integers(6, 13))]
        for s in sorted(set(payers)):
            edge(s, t, rng.uniform(20000, 120000), p=.9, start=6, ntx=2, tag="pt_in")
        payees = [rng.choice(eco_members[e]) for e in rng.choice(all_ecos, rng.integers(6, 13))]
        for d in sorted(set(payees) - set(payers)):
            edge(t, d, 1.0, p=1.0, start=6, ntx=2, tag="pt_out")          # amount set from inflows later

    for i in range(5):                                 # round-trippers: from 2026-02
        eco = all_ecos[(3 * i + 2) % 12]
        a = new_biz(eco, rng.choice(["42", "54"]), 60000, "round_trip")
        arche["round_trip"].append(a)
        normal_cover(a, eco, k=4)
        partners = []
        for _ in range(4):
            b = new_biz(eco, rng.choice(["54", "56"]), 15000, "rt_partner")
            edge(b, hub_id["utility"], 400, p=.95, ntx=1)
            partners.append(b)
            arche["rt_partner"].append(b)
            edge(a, b, rng.uniform(40000, 150000), p=.9, start=5, ntx=1, tag="rt_out")
            edge(b, a, 1.0, p=1.0, start=5, ntx=1, tag="rt_back")         # amount set from outflow later
        edge(partners[0], partners[1], rng.uniform(30000, 80000), p=.9, start=5, ntx=1, tag="rt_tri")

    for i, eco in enumerate(["E06", "E03", "E01", "E09"]):   # property managers (legit)
        pm = new_biz(eco, "53", 50000, "property_manager")
        arche["property_manager"].append(pm)
        n_t = int(rng.integers(40, 71))
        for pid in tenants[used_tenants:used_tenants + n_t]:
            edge(pid, pm, rng.uniform(1200, 2500), p=.98, ntx=1, tag="rent_in")
        used_tenants += n_t
        owners = list(pick(eco, 3, {pm})) + list(pick("E03" if eco != "E03" else "E06", 2, {pm}))
        for d in owners:
            edge(pm, d, 1.0, p=1.0, ntx=1, tag="pm_out")                  # amount set from rents later
        edge(pm, hub_id["utility"], 2500, p=.95, ntx=1)
        edge(pm, hub_id["tax"], 4000, p=.6, ntx=1)

    for i in range(8):                                 # two-way traders (legit)
        eco = all_ecos[(5 * i + 3) % 12]
        tw = new_biz(eco, "42", 80000, "two_way_trader")
        arche["two_way_trader"].append(tw)
        for c in pick(eco, int(rng.integers(6, 10)), {tw}):
            amt = rng.uniform(10000, 60000)
            edge(tw, c, amt, p=.92, ntx=3, tag="tw")
            edge(c, tw, amt * rng.uniform(.3, .7), p=.9, ntx=3, tag="tw")
        edge(tw, hub_id["payroll"], 25000, p=.97)
        edge(tw, hub_id["tax"], 6000, p=.6)

    N = pd.DataFrame(nodes)
    ed = pd.DataFrame(E, columns=["src", "dst", "amt", "p", "start", "end", "ntx", "tag"])
    ed = ed.drop_duplicates(["src", "dst"]).reset_index(drop=True)
    # ordinary relationship turnover: some partnerships end, some begin during the year
    nb = (ed.tag == "normal") & ed.src.str.startswith("B") & ed.dst.str.startswith("B") \
        & ~ed.src.isin(hub_id.values()) & ~ed.dst.isin(hub_id.values())
    ends = nb & (rng.random(len(ed)) < .12)
    ed.loc[ends, "end"] = rng.integers(3, 11, ends.sum())
    starts = nb & (ed.start == 0) & ~ends & (rng.random(len(ed)) < .12)
    ed.loc[starts, "start"] = rng.integers(1, 9, starts.sum())

    # --------------------------------------------------- churn and shrink
    normal_ids = N[(N.party_type == "O") & (N.archetype == "normal")].node_id.to_numpy()
    silent_from = {}
    for nid in normal_ids:
        if eco_of[nid] == "E02" and rng.random() < .30:
            silent_from[nid] = int(rng.integers(6, 9))           # E02 shrink in H2
        elif rng.random() < .04:
            silent_from[nid] = int(rng.integers(5, 8))           # early leavers elsewhere
    early = set(silent_from)

    b2b = ed[ed.src.str.startswith("B") & ed.dst.str.startswith("B")]
    w = b2b.amt * b2b.p
    tot = pd.concat([pd.Series(w.values, b2b.src.values), pd.Series(w.values, b2b.dst.values)]).groupby(level=0).sum()
    lost_w = pd.concat([pd.Series(np.where(b2b.dst.isin(early), w, 0), b2b.src.values),
                        pd.Series(np.where(b2b.src.isin(early), w, 0), b2b.dst.values)]).groupby(level=0).sum()
    expo = (lost_w / tot).fillna(0)

    z = dict(zip(normal_ids, rng.normal(size=len(normal_ids))))
    churners = set()
    for nid in normal_ids:
        if nid in early:
            continue
        logit = -3.8 + 0.9 * z[nid] + 8.0 * expo.get(nid, 0) + 1.6 * (eco_of[nid] == "E02")
        if rng.random() < 1 / (1 + np.exp(-logit)):
            churners.add(nid)
            silent_from[nid] = int(rng.integers(9, 12))
    # second wave: trading partners of leavers tend to leave together (churn clusters).
    # They leave when their partner does, so they show no warning of their own.
    first = set(churners)
    wave2 = set()
    lost_w2 = pd.concat([pd.Series(np.where(b2b.dst.isin(first), w, 0), b2b.src.values),
                         pd.Series(np.where(b2b.src.isin(first), w, 0), b2b.dst.values)]).groupby(level=0).sum()
    expo2 = (lost_w2 / tot).fillna(0)
    for nid in normal_ids:
        if nid in silent_from:
            continue
        if rng.random() < 1 / (1 + np.exp(-(-6.0 + 16.0 * expo2.get(nid, 0)))):
            churners.add(nid)
            wave2.add(nid)
            silent_from[nid] = int(rng.integers(9, 12))
    dippers = set(rng.choice([x for x in normal_ids if x not in silent_from], 260, replace=False))

    # planted cases keep living counterparties (a separate generator keeps the main stream unchanged)
    rng2 = np.random.default_rng(seed + 1)
    alive_pool = {e: [x for x in eco_members[e] if x not in silent_from] for e in eco_members}
    for col, tags_ in (("dst", ("funnel_out", "pt_out", "pm_out")), ("src", ("pt_in",))):
        m = ed.tag.isin(tags_) & ed[col].isin(list(silent_from))
        for idx in ed.index[m]:
            eco = eco_of[ed.at[idx, col]]
            ed.at[idx, col] = rng2.choice(alive_pool[eco])
    ed = ed.drop_duplicates(["src", "dst"]).reset_index(drop=True)

    # ------------------------------------------------------ monthly expansion
    M = np.arange(N_MONTHS)
    n_e = len(ed)
    p = np.repeat(ed.p.to_numpy()[:, None], N_MONTHS, 1)
    amt = ed.amt.to_numpy()[:, None] * rng.lognormal(0, 0.25, (n_e, N_MONTHS))
    in_window = (M[None, :] >= ed.start.to_numpy()[:, None]) & (M[None, :] <= ed.end.to_numpy()[:, None])

    src, dst = ed.src.to_numpy(), ed.dst.to_numpy()
    sil_s = np.array([silent_from.get(x, 99) for x in src])[:, None]
    sil_d = np.array([silent_from.get(x, 99) for x in dst])[:, None]
    alive = (M[None, :] < sil_s) & (M[None, :] < sil_d)

    pre = (M >= 6) & (M <= CUTOFF)
    is_ch_d = np.isin(dst, list(churners))[:, None]
    p = np.where(is_ch_d & pre, p * .85, p)             # some payers of a churner pull back first
    zs = np.array([z.get(x, 0.0) for x in src])[:, None]
    amt = np.where(pre[None, :] & np.isin(src, normal_ids)[:, None], amt * np.exp(-.06 * zs), amt)
    is_dip = np.isin(src, list(dippers))[:, None]
    amt = np.where(is_dip & pre, amt * rng.uniform(.55, .85, (n_e, 1)), amt)   # decoys: dip, no partner loss
    e02_int = (np.array([eco_of.get(x) for x in src]) == "E02") & (np.array([eco_of.get(x) for x in dst]) == "E02")
    amt = np.where(e02_int[:, None] & (M >= H2_START), amt * .85, amt)

    active = in_window & alive & (rng.random((n_e, N_MONTHS)) < p)

    # churners stop paying some suppliers before they leave, but keep total spend roughly level
    # (fewer counterparties, not smaller payments): the account view barely moves
    b2b_out = np.char.startswith(dst.astype(str), "B") & ~np.isin(dst, list(hub_id.values()))
    pre_idx = np.where(pre)[0]
    for c in sorted(churners):
        if c in wave2 or rng.random() > .75:
            continue                                     # a quarter leave with no warning
        mine = np.where(src == c)[0]
        drop_pool = mine[b2b_out[mine]]
        if len(drop_pool) == 0:
            continue
        k = max(1, int(round(len(drop_pool) * rng.uniform(.2, .55))))
        drop = rng.choice(drop_pool, k, replace=False)
        before = (amt[mine][:, pre_idx] * active[mine][:, pre_idx]).sum(0)
        active[np.ix_(drop, pre_idx)] = False
        keep = np.setdiff1d(mine, drop)
        after = (amt[keep][:, pre_idx] * active[keep][:, pre_idx]).sum(0)
        scale = np.where(after > 0, before * rng.uniform(.85, 1.05) / np.maximum(after, 1e-9), 1.0)
        amt[np.ix_(keep, pre_idx)] *= scale[None, :]

    ntx = np.maximum(1, rng.poisson(np.repeat(ed.ntx.to_numpy()[:, None], N_MONTHS, 1)))

    # flows that follow inflows (funnel, pass-through, round-trip, property manager)
    def inflow(node):
        m = (dst == node)
        return (amt[m] * active[m]).sum(0)

    def set_out(tag, node, total_by_month):
        m = np.where((src == node) & (ed.tag.to_numpy() == tag))[0]
        if len(m) == 0:
            return
        share = rng.dirichlet(np.ones(len(m)) * 2)
        for j, idx in enumerate(m):
            amt[idx] = total_by_month * share[j]
            active[idx] = in_window[idx] & alive[idx] & (total_by_month > 0)

    for f in arche["funnel"]:
        fin = (amt[(dst == f) & (ed.tag.to_numpy() == "funnel_in")] * active[(dst == f) & (ed.tag.to_numpy() == "funnel_in")]).sum(0)
        set_out("funnel_out", f, fin * .95)
    for t in arche["pass_through"]:
        m = (dst == t) & (ed.tag.to_numpy() == "pt_in")
        set_out("pt_out", t, (amt[m] * active[m]).sum(0) * .98)
    for pm in arche["property_manager"]:
        m = (dst == pm) & (ed.tag.to_numpy() == "rent_in")
        set_out("pm_out", pm, (amt[m] * active[m]).sum(0) * .75)
    tags = ed.tag.to_numpy()
    for idx in np.where(tags == "rt_back")[0]:
        back = np.where((src == dst[idx]) & (dst == src[idx]) & (tags == "rt_out"))[0][0]
        amt[idx] = amt[back] * rng.uniform(.93, .99)
        active[idx] = active[back]

    ii, mm = np.nonzero(active)
    em = pd.DataFrame({"month": np.array(MONTHS)[mm], "src": src[ii], "dst": dst[ii],
                       "amount": np.round(amt[ii, mm], 2), "n_txn": ntx[ii, mm], "_e": ii})

    # ------------------------------------------------------ daily transactions
    days = 31
    last = em[em.month == MONTHS[FOCUS]].copy()
    rows = []
    arch_of = dict(zip(N.node_id, N.archetype))

    def split(total, k):
        return np.round(total * rng.dirichlet(np.ones(k) * 3), 2)

    funnel_set, pt_set = set(arche["funnel"]), set(arche["pass_through"])
    pm_set, rt_set = set(arche["property_manager"]), set(arche["round_trip"])
    handled = set()

    for f in arche["funnel"]:                 # inflows any day, sweeps every 5 days
        ins = last[(last.dst == f) & last.src.str.startswith("P")]
        inflows = []
        for _, r in ins.iterrows():
            for a, d in zip(split(r.amount, int(r.n_txn)), rng.integers(1, 29, int(r.n_txn))):
                inflows.append((d, a)); rows.append((d, r.src, f, a))
        outs = last[(last.src == f) & (last._e.isin(np.where(tags == "funnel_out")[0]))]
        handled |= set(ins.index) | set(outs.index)
        if len(outs):
            dest = outs.iloc[0].dst
            sweeps = [4, 9, 14, 19, 24, 29]
            prev = 0
            for sday in sweeps:
                tot_in = sum(a for d, a in inflows if prev < d <= sday)
                if tot_in > 0:
                    rows.append((min(days, sday + rng.integers(0, 2)), f, dest, round(tot_in * .95, 2)))
                prev = sday
    for t in arche["pass_through"]:           # each inflow leaves within 0-2 days
        ins = last[(last.dst == t) & (last._e.isin(np.where(tags == "pt_in")[0]))]
        outs = last[(last.src == t) & (last._e.isin(np.where(tags == "pt_out")[0]))]
        handled |= set(ins.index) | set(outs.index)
        payees = outs.dst.tolist()
        for _, r in ins.iterrows():
            for a, d in zip(split(r.amount, int(r.n_txn)), rng.integers(1, 28, int(r.n_txn))):
                rows.append((d, r.src, t, a))
                if payees:
                    k = int(rng.integers(1, 3))
                    for a2, dd in zip(split(a * .98, k), rng.choice(payees, k)):
                        rows.append((min(days, d + int(rng.integers(0, 3))), t, dd, a2))
    for a in arche["round_trip"]:             # money goes out and comes back 1-4 days later
        outs = last[(last.src == a) & (last._e.isin(np.where(tags == "rt_out")[0]))]
        backs = last[(last.dst == a) & (last._e.isin(np.where(tags == "rt_back")[0]))]
        handled |= set(outs.index) | set(backs.index)
        back_amt = dict(zip(backs.src, backs.amount))
        for _, r in outs.iterrows():
            d = int(rng.integers(1, 26))
            rows.append((d, a, r.dst, r.amount))
            if r.dst in back_amt:
                rows.append((min(days, d + int(rng.integers(1, 5))), r.dst, a, back_amt[r.dst]))
    for pm in arche["property_manager"]:      # rent on days 1-5, payouts days 12-20
        ins = last[last.dst == pm]
        outs = last[last.src == pm]
        handled |= set(ins.index) | set(outs.index)
        for _, r in ins.iterrows():
            rows.append((int(rng.integers(1, 6)), r.src, pm, r.amount))
        for _, r in outs.iterrows():
            rows.append((int(rng.integers(12, 21)), pm, r.dst, r.amount))

    rest = last.drop(index=list(handled))
    rep = rest.loc[rest.index.repeat(rest.n_txn)]
    k = rep.groupby(level=0).cumcount()
    rep = rep.assign(_w=rng.gamma(3, 1, len(rep)))
    rep["_w"] = rep._w / rep.groupby(level=0)._w.transform("sum")
    rows += list(zip(rng.integers(1, days + 1, len(rep)), rep.src, rep.dst, np.round(rep.amount * rep._w, 2)))

    tx = pd.DataFrame(rows, columns=["day", "src", "dst", "amount"])
    tx["date"] = pd.to_datetime(MONTHS[FOCUS] + "-01") + pd.to_timedelta(tx.day.astype(int) - 1, unit="D")
    tx = tx.sort_values(["date", "src", "dst"]).reset_index(drop=True)
    tx.insert(0, "txn_id", [f"T{i:06d}" for i in range(1, len(tx) + 1)])
    tx = tx[["txn_id", "date", "src", "dst", "amount"]]
    tx["date"] = tx.date.dt.strftime("%Y-%m-%d")

    # monthly table for the focus month must equal the daily transactions
    agg = tx.groupby(["src", "dst"]).agg(amount=("amount", "sum"), n_txn=("amount", "size")).reset_index()
    agg.insert(0, "month", MONTHS[FOCUS])
    em = pd.concat([em[em.month != MONTHS[FOCUS]].drop(columns="_e"), agg], ignore_index=True)
    em["amount"] = em.amount.round(2)
    em = em.sort_values(["month", "src", "dst"]).reset_index(drop=True)

    # ------------------------------------------------------ vendor payments
    last4 = em[em.month.isin(MONTHS[8:])]
    b2b_last = last4[last4.src.str.startswith("B") & last4.dst.isin(normal_ids) & last4.src.isin(normal_ids)]
    payer_counts = b2b_last.groupby("dst").src.nunique().sort_values(ascending=False)
    vendors = payer_counts.index[:4].tolist()
    vp = b2b_last[b2b_last.dst.isin(vendors)].rename(columns={"src": "payer", "dst": "vendor"})[["month", "payer", "vendor", "amount"]].copy()
    old = {v: f"Bank {'ABCD'[i]} acct ..{rng.integers(1000, 9999)}" for i, v in enumerate(vendors)}
    vp["vendor_account"] = vp.vendor.map(old)
    scen = {vendors[0]: "fraud_one_payer", vendors[1]: "real_move_all_payers",
            vendors[2]: "fraud_one_payer_two_months", vendors[3]: "no_change"}
    v0 = vp[(vp.vendor == vendors[0]) & (vp.month == MONTHS[11])]
    if len(v0):
        vp.loc[v0.sample(1, random_state=seed).index, "vendor_account"] = "Bank Q acct ..8841"
    vp.loc[(vp.vendor == vendors[1]) & vp.month.isin(MONTHS[10:]), "vendor_account"] = "Bank C acct ..5510"
    v2 = vp[(vp.vendor == vendors[2]) & vp.month.isin(MONTHS[10:])]
    if len(v2):
        victim = v2.payer.sample(1, random_state=seed).iloc[0]
        vp.loc[(vp.vendor == vendors[2]) & (vp.payer == victim) & vp.month.isin(MONTHS[10:]), "vendor_account"] = "Bank R acct ..3302"
    vp = vp.sort_values(["vendor", "month", "payer"]).reset_index(drop=True)

    # ------------------------------------------------------ tables
    eco_name = {e[0]: e[1] for e in ECOS}
    N["sector"] = N.sector_code.map(SECTORS).fillna("")
    N["size_band"] = pd.cut(N["size"], [0, 15000, 60000, 1e12], labels=["small", "mid", "large"]).astype(str)
    N.loc[N.party_type == "P", "size_band"] = ""
    nodes_out = N[["node_id", "name", "party_type", "sector_code", "sector", "region", "size_band"]].copy()

    truth = N[["node_id", "eco", "archetype"]].rename(columns={"eco": "planted_ecosystem"}).copy()
    truth["ecosystem_name"] = truth.planted_ecosystem.map(eco_name).fillna(truth.planted_ecosystem.map({"HUB": "Hub", "IND": "Individuals"}))
    truth["is_hub"] = truth.archetype.eq("hub")
    truth["silent_from_month"] = truth.node_id.map(lambda x: MONTHS[silent_from[x]] if x in silent_from else "")
    truth["churn_label"] = truth.node_id.map(lambda x: int(x in silent_from and CUTOFF < silent_from[x]))
    truth["early_leaver"] = truth.node_id.map(lambda x: int(x in silent_from and silent_from[x] <= CUTOFF))
    truth["vendor_scenario"] = truth.node_id.map(scen).fillna("")
    truth = truth[["node_id", "planted_ecosystem", "ecosystem_name", "archetype", "is_hub",
                   "churn_label", "early_leaver", "silent_from_month", "vendor_scenario"]]

    # shuffle business ids so planted cases are not given away by their numbers
    biz_ids = N.loc[N.party_type == "O", "node_id"].tolist()
    perm = rng.permutation(len(biz_ids))
    remap = {old: f"B{perm[i] + 1:04d}" for i, old in enumerate(biz_ids)}
    rm = lambda s: s.map(lambda x: remap.get(x, x))
    nodes_out["node_id"] = rm(nodes_out.node_id)
    truth["node_id"] = rm(truth.node_id)
    em["src"], em["dst"] = rm(em.src), rm(em.dst)
    tx["src"], tx["dst"] = rm(tx.src), rm(tx.dst)
    vp["payer"], vp["vendor"] = rm(vp.payer), rm(vp.vendor)
    nodes_out = nodes_out.sort_values("node_id").reset_index(drop=True)
    truth = truth.sort_values("node_id").reset_index(drop=True)
    em = em.sort_values(["month", "src", "dst"]).reset_index(drop=True)
    tx = tx.sort_values(["date", "src", "dst"]).reset_index(drop=True)
    tx["txn_id"] = [f"T{i:06d}" for i in range(1, len(tx) + 1)]
    vp = vp.sort_values(["vendor", "month", "payer"]).reset_index(drop=True)

    return {"nodes": nodes_out, "edges_monthly": em, f"transactions_{MONTHS[FOCUS]}": tx,
            "vendor_payments": vp, "truth": truth}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, df in generate(a.seed).items():
        path = os.path.join(a.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"{path:45s} {len(df):>9,d} rows")


if __name__ == "__main__":
    main()
