"""The five patterns in plain words: what we see, why it matters, what else it could be, the evidence lines on a card and
the rule table on the customer page. Thresholds mirror §0 of pkg_fraud_signals_v1.ipynb (keep them in step)."""
import pandas as pd

from ..core.fmt import money


def _n(v):
    return v is not None and not pd.isna(v)


def _p(v, d=0):
    return f"{v:.{d}%}" if _n(v) else "—"


def _f(v, d=0):
    return f"{v:,.{d}f}" if _n(v) else "—"


TYPES = {
    "passthrough": dict(
        name="Pass-through", icon="🔁", color="red",
        what="Money comes in and goes straight back out; very little of it stays in the account.",
        why="Accounts used only to move money on for someone else (layering) look like this: high turnover and almost no balance "
            "left behind, month after month.",
        innocent=["payroll and payment processors", "title, escrow and trust accounts", "collection accounts swept to another entity",
                  "brokers and agents paying out client funds"]),
    "funnel": dict(
        name="Funnel account", icon="🔻", color="violet",
        what="Many payers send money in; it leaves to one or two recipients, often by wire and in much larger amounts.",
        why="A known way to gather money from many sources — sometimes many places — and move it on in bulk.",
        innocent=["a business collecting from customers and paying its main supplier", "franchise or dealer collections",
                  "insurance premium or tuition collection"]),
    "split": dict(
        name="Split payments", icon="✂️", color="orange",
        what="Payments cluster just under a reporting threshold, with several on the same day to the same counterparty.",
        why="Keeping each payment under a threshold (structuring) is a red flag in itself, whatever the source of the money.",
        innocent=["fixed invoices or instalments that happen to sit just under $10,000", "a payment-system limit on single payments"]),
    "spike": dict(
        name="Sudden spike", icon="⚡", color="blue",
        what="Activity jumps many times above the client's own normal, mostly with counterparties never seen before.",
        why="Quiet accounts that suddenly move large sums with new parties can point to account takeover, bust-out or mule activity.",
        innocent=["a one-off event: asset sale, refinancing, acquisition", "a seasonal business at its peak",
                  "operating accounts moving to PNC from another bank"]),
    "roundtrip": dict(
        name="Round-trip", icon="🔄", color="green",
        what="Money sent to a counterparty comes back from that same counterparty.",
        why="Circular flows can hide where money came from or make activity look bigger than it is.",
        innocent=["intercompany loans and repayments", "refunds and returned payments",
                  "cash pooling with a related company outside the recorded corporate family"]),
}
ORDER = list(TYPES)
PEER_LABEL = {"pr_pt_3m": "money in ≈ money out", "pr_turnover": "turnover (flow ÷ balance)", "pr_flow_3m": "size of flows",
              "pr_fan": "payers per payee", "pr_top1_out": "share sent to the largest recipient",
              "pr_ticket_ratio": "outgoing vs incoming payment size", "pr_rt_share": "money that came back"}
PEER_FOR = {"passthrough": ["pr_pt_3m", "pr_turnover", "pr_flow_3m"], "funnel": ["pr_fan", "pr_top1_out", "pr_ticket_ratio"],
            "split": [], "spike": ["pr_flow_3m"], "roundtrip": ["pr_rt_share", "pr_flow_3m"]}


def name(k):
    return TYPES.get(k, {}).get("name", k or "—")


def icon(k):
    return TYPES.get(k, {}).get("icon", "•")


def listed(r):
    return [k for k in str(getattr(r, "typologies", "") or "").split(",") if k in TYPES]


def key_stat(k, r):
    """(label, value) for the card's third metric."""
    g = lambda c: getattr(r, c, None)
    return {"passthrough": ("Paid straight out", _p(g("pt_3m"))),
            "funnel": ("Payers → payees", f"{_f(g('payers_avg'))} → {_f(g('payees_avg'))}"),
            "split": ("Just under / over", f"{_f(g('n_below'))} / {_f(g('n_above'))}"),
            "spike": ("vs its normal", f"{_f(g('spike_x'))}×"),
            "roundtrip": ("Came back", money(g("rt_amt")))}.get(k, ("Score", _f(g("score"))))


def evidence(k, r):
    """Plain sentences with the numbers that triggered the pattern (markdown)."""
    g = lambda c: getattr(r, c, None)
    peer = f"similar clients ({g('peer_group')})" if g("peer_group") else "similar clients"
    if k == "passthrough":
        out = [f"Received **{money(g('in_3m'))}** and paid out **{money(g('out_3m'))}** in 3 months — **{_p(g('pt_3m'))}** matched.",
               f"In ≈ out in **{_f(g('pt_months'))} of 3** months."]
        if _n(g("days_held")): out.append(f"The average balance holds only **{_f(g('days_held'), 1)} days** of incoming money.")
        if _n(g("pr_turnover")):
            out.append(f"The highest turnover among {peer}." if g("pr_turnover") >= .99 else f"Turnover higher than **{_p(g('pr_turnover'))}** of {peer}.")
        return out
    if k == "funnel":
        return [f"**{_f(g('payers_avg'))}** payers a month → **{_f(g('payees_avg'))}** payees (**{_f(g('fan'))}×** fan-in).",
                f"The largest recipient takes **{_p(g('top1_out'))}** of outgoing money; **{_p(g('wire_out'))}** leaves by wire.",
                f"Average payment in **{money(g('ticket_in'))}**, out **{money(g('ticket_out'))}** (**{_f(g('ticket_ratio'))}×** larger)."]
    if k == "split":
        thr = money(g("split_thr") or 10_000)
        return [f"**{_f(g('n_below'))}** payments just under {thr} (within 10%) against **{_f(g('n_above'))}** just over, in 3 months.",
                f"**{_f(g('split_days'))}** days when several under-threshold payments to one counterparty added up to more than {thr}.",
                f"**{money(g('amt_below'))}** moved this way."]
    if k == "spike":
        return [f"**{money(g('flow_m'))}** moved in {g('month')} — **{_f(g('spike_x'))}×** its normal month ({money(g('base'))}).",
                f"**{_p(g('new_share_m'))}** of it with counterparties never seen before."]
    if k == "roundtrip":
        out = [f"**{money(g('rt_amt'))}** went out and came back from the same counterparties — **{_p(g('rt_share'))}** of outgoing money."]
        if g("rt_top_name") or g("rt_top_other"):
            out.append(f"Largest: **{g('rt_top_name') or g('rt_top_other')}** ({money(g('rt_top_amt'))}); "
                       f"**{_f(g('n_rt'))}** counterparties involved.")
        return out
    return []


def rules(k, r):
    """(measure, this client, rule, met) rows — why the pattern fired."""
    g = lambda c: getattr(r, c, None)
    ge = lambda v, t: _n(v) and v >= t
    if k == "passthrough":
        return [("Money in ≈ money out, 3 months", _p(g("pt_3m")), "≥ 90%", ge(g("pt_3m"), .90)),
                ("Months with in ≈ out (≥ 85%)", f"{_f(g('pt_months'))} of 3", "≥ 2", ge(g("pt_months"), 2)),
                ("Days of incoming money kept as balance", _f(g("days_held"), 1), "≤ 5", _n(g("days_held")) and g("days_held") <= 5),
                ("Average balance (below: swept account)", money(g("bal_3m")), "≥ $1k", ge(g("bal_3m"), 1_000))]
    if k == "funnel":
        return [("Payers per month", _f(g("payers_avg")), "≥ 20", ge(g("payers_avg"), 20)),
                ("Payers per payee", _f(g("fan"), 1), "≥ 10", ge(g("fan"), 10)),
                ("Share sent to the largest recipient", _p(g("top1_out")), "≥ 50%", ge(g("top1_out"), .5)),
                ("Money in ≈ money out", _p(g("pt_3m")), "≥ 70%", ge(g("pt_3m"), .7))]
    if k == "split":
        return [("Payments within 10% under the threshold", _f(g("n_below")), "≥ 10", ge(g("n_below"), 10)),
                ("Under vs over", f"{_f(g('heap'), 1)}×", "≥ 3×", ge(g("heap"), 3)),
                ("How unusual the excess under is (z)", _f(g("split_z"), 1), "≥ 4", ge(g("split_z"), 4)),
                ("Days with split payments", _f(g("split_days")), "≥ 3", ge(g("split_days"), 3))]
    if k == "spike":
        return [("Money moved this month", money(g("flow_m")), "≥ $250k", ge(g("flow_m"), 250_000)),
                ("Times its normal month", f"{_f(g('spike_x'))}×", "≥ 10×", ge(g("spike_x"), 10)),
                ("Share with new counterparties", _p(g("new_share_m")), "≥ 50%", ge(g("new_share_m"), .5)),
                ("Its normal month (established history)", money(g("base")), "≥ $25k", ge(g("base"), 25_000)),
                ("Active months in the 6 before", f"{_f(g('base_active'))} of 6", "≥ 5", ge(g("base_active"), 5))]
    if k == "roundtrip":
        return [("Money that came back, 3 months", money(g("rt_amt")), "≥ $250k", ge(g("rt_amt"), 250_000)),
                ("Share of outgoing money", _p(g("rt_share")), "≥ 25%", ge(g("rt_share"), .25)),
                ("Counterparties two-way with many PNC clients", "left out", "hubs ignored", True)]
    return []


def guards(r):
    """Ordinary explanations the model already accounts for (each lowers or qualifies the score)."""
    out = []
    if bool(getattr(r, "processor_like", False)):
        out.append("Thousands of payers or payees — looks like a payment processor (pass-through and funnel scores × 0.5).")
    if bool(getattr(r, "industry_pt", False)):
        out.append(f"Industry where pass-through is normal ({getattr(r, 'naics_desc', '') or 'payroll, escrow, title, brokers'}) — pass-through and funnel scores × 0.6.")
    fs = getattr(r, "family_share_3m", None)
    if _n(fs) and fs >= .05:
        out.append(f"{_p(fs)} of its money moves within its own corporate family at PNC; those transfers are left out of pass-through and funnel.")
    if str(getattr(r, "naics_cd", "") or "").startswith(("5411", "5259", "5311")) and float(getattr(r, "score_spike", 0) or 0) > 0:
        out.append("Law firm, trust or landlord — large one-off movements are normal (spike score × 0.6).")
    if bool(getattr(r, "treasury_sweep", False)):
        out.append("Half or more of outgoing money goes to its own accounts at other banks (treasury sweep); those transfers are left out.")
    return out
