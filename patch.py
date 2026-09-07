# GEN is a FILTER of payers, so it already carries n_payers. Joining payers
# back puts two identically-named columns in scope and F.sum("n_payers")
# cannot resolve them. Aggregate GEN directly.
_g_names, _g_links = GEN.count(), (GEN.agg(F.sum("n_payers")).collect()[0][0] or 0)
_a_names, _a_links = payers.count(), (payers.agg(F.sum("n_payers")).collect()[0][0] or 0)
kv([("median payers per counterparty name", med),
    ("generic threshold (median x %d, floored)" % GENERIC_MEDIAN_MULT, thr),
    ("distinct counterparty names", _a_names),
    ("names flagged generic", _g_names),
    ("share of names flagged", round(pct(_g_names, _a_names), 5)),
    ("share of customer-counterparty links they carry", round(pct(_g_links, _a_links), 4)),
    ("QA wall (s)", round(time.time()-t0))],
   title="5d &middot; Generic-name registry...",
   save="v8_generic_registry")
