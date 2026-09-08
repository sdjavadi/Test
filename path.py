# =====================================================================
# 5x · UNIT FIX + CAPACITY-MATCHED INCUMBENT
# =====================================================================
INC_K   = int(round(len(inc)/len(ORIGINS)))
ALL_ATT = int(S.loc[S.y == 1, "cust_pwr_id"].nunique())

def qm(fl, label, per_month=None):
    raw, dis, con = fatigue(fl)
    tp_rows = int(fl.y.sum())
    tp_cust = int(fl.loc[fl.y == 1, "cust_pwr_id"].nunique())
    ld = lead_profile(fl[fl.y == 1])
    return dict(queue=label,
                alerts_per_month=per_month or round(raw/len(ORIGINS)),
                alerts_raw=raw, conversations=con,
                tp_rows=tp_rows, tp_customers=tp_cust,
                precision=tp_rows/max(raw, 1),
                # customers per customer — the cost figure that was wrong
                conversations_per_tp_customer=con/max(tp_cust, 1),
                recall_customers=tp_cust/max(ALL_ATT, 1),
                median_lead_m=float(ld.median()) if len(ld) else np.nan,
                p90_lead_m=float(ld.quantile(.9)) if len(ld) else np.nan)

MATCH = pd.DataFrame(
    [qm(flagged(S, BEST_MODEL, K), f"{BEST_MODEL} @ K={K}") for K in [250, 1000, 2500, INC_K]]
    + [qm(inc, "incumbent 30% rule", per_month=INC_K)])
disp(MATCH.round(3), title=f"5e &middot; Capacity-matched. The incumbent runs {INC_K:,} alerts a "
     "month; until now nothing was compared to it at that size. tp and recall are DISTINCT "
     "CUSTOMERS — 5a mixed customer-months with customers and understated the cost per alert",
     save="v7_capacity_matched")
