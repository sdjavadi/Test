# TPPP Prospect Finder (draft 2)

Outside companies (counterparties) that move money like third-party payment and payroll processors. They are ranked
by a model that learns from examples, and reviewers' verdicts feed the next ranking.

```
streamlit run app.py
```

Reads Impala through `dbi`; it never writes to Impala. Needs `streamlit` ≥ 1.35 (to click rows), `scikit-learn`,
`pandas`, `numpy` and `plotly`.

## Files

| File | What it holds |
|---|---|
| `tppp_core.py` | TPPP logic, with no Streamlit code: features, universe query, examples, model, reliability, explanations, review file, exports |
| `app.py` | the page: List, Map, Explore, Examples, How the score works, Reviews, Export and research; the counterparty page with its TPPP view, review form and review queue |
| `cpty_view.py` | the counterparty page shared with the Prospect Finder: network, PNC clients, accounts/banks/names, month by month |
| `prospect_core.py` | the same module as in the Prospect Finder: queries and helpers for one counterparty |
| `pkg_tppp_cpty_evidence.ipynb` (separate, Spark) | builds the optional evidence table `pkg_tppp_cpty_evidence_monthly` |

## How it works

1. **Counterparties scored:** those with eligible counterparty metrics, at least 5 PNC clients a month and 2 active
   months in the period. They also pass a screen: at least 20% of their money is with organisations on one side, and
   the name is not government, a card network, an insurer, PNC, or a generic descriptor.
2. **Examples:**
   - registered TPPPs' accounts at other banks, from the latest `vector_customer` snapshot, with names matched by
     cleaned key or brand prefix;
   - known processor names (the seed list in `tppp_core.KNOWN_PROC_RE`);
   - counterparties reviewers marked *processor*.

   *Not a processor* never becomes an example. Registered TPPPs' own accounts are never shown as prospects, because
   they are existing clients.
3. **Model:** logistic regression on percentiles of 36 behaviour measures. Every counterparty is scored by a model
   that did not see it, and examples of one brand are held out together. The score is that out-of-fold percentile,
   so an example's score is an honest test. "Hidden examples found in the top 1%" is the reliability figure.
4. **Evidence table (optional):** monthly measures built in Spark from the daily edges and the client pairs. The
   app adds them to the model when the table exists:
   - money out right after money in;
   - money out on its 4 biggest days;
   - money out on its main weekday;
   - paying organisations that also pay a registered TPPP.

   Organisations that switched to a counterparty from a registered TPPP are shown, and can be filtered on, but are
   kept out of the score. The examples rarely show switching, so the model would learn it the wrong way round.
5. **Explanations:**
   - *why*: the measures adding most to the score;
   - *most similar example*: the nearest example on the measures the model weighs most.

## What the main page offers

- **Two scores.** The *TPPP score* says how processor-like a counterparty is. *Value* says how much business it does
  with PNC clients: the average of its percentiles for money with PNC clients a month and TM clients a month.
- **Looks like.** The processor type of its 5 nearest examples: payroll and HR, card and merchant payments, bill pay
  and payables, property payments, or other. Shown for TPPP scores of 80 and above.
- **Map.** TPPP score against value for the filtered list. **Explore** plots any two measures, with examples
  highlighted. Click a point to open the counterparty.
- **Review loop.** *Next to review* opens the highest-ranked unreviewed counterparty in the current filters, and
  *Save and open the next* records a review and moves on. *Not a processor* counts 5 times as a negative.
- **Outside research.** Export the names file, have it researched, and import the results (`row_id` plus an
  answer, optional confidence, evidence and type). The results show on the list and the page but don't change the
  score until a reviewer confirms them.
- **Previous period** (Settings). Scores the period of the same length just before, and flags counterparties as
  *newly processor-like* (95+ now, below 80 before) or *consistently high*.

## Check on the first run

- **`vector_customer` columns:** `cust_pwr_id`, `customer_type`, `load_date`, as in the TPPP notebooks.
- **Power ids:** the match to `pkg_cust_dim` tries each id with and without leading zeros.
- **Research file:** imported research goes to `data/tppp_research.csv`, or to the path in `TPPP_RESEARCH`.
- **Review file:** reviews are appended to `data/tppp_verdicts.csv`, relative to where the app runs. Set the
  environment variable `TPPP_VERDICTS` to a path on shared storage so every user and restart sees the same reviews.
- **Feature columns:** each measure is used only if its column exists in `pkg_cpty_metrics_monthly`. The "How the
  score works" tab says how many were used.
