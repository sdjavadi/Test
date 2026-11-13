# Data Center Network Explorer (draft 5)

Upload the data center workbook, pick data centers or second-order parties (or find any PNC customer), and see the
payment network around them: who pays whom, which flows are regular and whether they are on schedule, what the
workbook says, and the credit snapshot of the PNC customers involved.

```
streamlit run app.py
```

Reads Impala through `dbi`; it never writes to Impala. Needs `streamlit` >= 1.35 (to click nodes and rows), `pandas`,
`numpy`, `plotly`, `openpyxl`.

## Files

| File | What it holds |
|---|---|
| `dc_core.py` | No Streamlit code: workbook reader, matching to the payment data, queries, graph building |
| `dc_ui.py` | Style, number formats, cached reads, the network drawing, small charts |
| `app.py` | Network page (upload, selection, options, network, tables) and the page router |
| `dc_pages.py` | Customer page and counterparty page |
| `.streamlit/config.toml` | Light theme, 50 MB upload limit |

## Using the network page

- **Nothing is read from the database until you press *Update the network*.** Choose tabs, data centers, parties and
  options freely; a note says when the picture no longer matches the controls. "Show its network" on a node redraws
  at once, because it is itself a request to draw.
- **One entry per company name:** a company listed under several workbook ids (a Power ID in one tab, a made-up id in
  another) appears once in the pickers, by name only.
- **Export:** *Download as HTML* saves the interactive picture as one self-contained file. For a PNG, use the camera
  button at the top right of the picture (twice the screen resolution).

## A company seen two ways: never merged

A workbook company can have PNC customer records and also accounts at other banks that the payment data knows only by
name. The two are always separate nodes:

- **PNC side** (navy diamond, label "(PNC)"): its PNC customers, grouped by corporate family when *Group by family*
  is on. Opens the customer page: payments through PNC accounts, credit, deposits.
- **Outside side** (navy square, label "(outside)"): its accounts at other banks, matched by name; several
  counterparty groups found for one company are combined. Opens the counterparty page, with a tab listing the groups
  so each name match can be checked. Only their payments with PNC customers are visible.
- Each page has a button to the other side. Payments between the two sides (the company moving money between its PNC
  and outside accounts) show as a line between the two nodes.
- A company found neither way is a hollow diamond, kept so its workbook relationships still show.

Many data center borrowers have PNC credit but keep their operating accounts elsewhere: their PNC side then has DHE and
PD but no payments, and their flows are all on the outside side.

## Risk view (first draft, to try)

Under the network, for the selection as drawn. It uses only tables that already exist. Nothing in it is calibrated:
it is a screen to find where to look, and the contagion study should replace its assumptions with measured ones.

- **Who depends on the selection:** every PNC customer the selection pays, with the share of its money in that comes
  from the selection, its DHE, PD now and 12 months ago, and whether its regular stream from the selection is on
  schedule, late or stopped. Needs one extra read (customer totals), so it runs on a button.
- **If the selection stops paying:** the selection stops 25-100% of its payments. First round: each customer loses that
  share of what it received. Second round: a customer that lost x% of its money in is assumed to pay x% less to
  everyone it pays. Shows the customers hit, their DHE, and the money in lost. No buffers, no substitution, no
  third round; the second round follows the 300 largest first-round customers.
- **Concentration of the selection:** share of its money in (out) from its largest and five largest payers (payees),
  and the split by relationship type in the workbook.
- **Warning signs:** regular streams with the selection that are late, overdue or stopped; PNC neighbours in default
  or past due.

Limits: customer level (not family); outside counterparties cannot be dependents (their total money in is unknown);
a selection seen only through outside accounts shows only its payments with PNC customers.

## Chart conventions

- Money in and out over time: bars above zero in green (in) and below zero in red (out).
- Deposits: bars for the month-end balance, lines for the other measures.

## How to read the network

- **Nodes:** navy diamond = selected; blue circle = PNC customer (or family); slate square = outside counterparty;
  hollow circle = the neighbours not drawn, summed; hollow violet diamond = listed in the workbook, no payments seen.
- **Lines:** arrows point the way the money moved; width grows with the dollars (square root).
  - solid green = regular stream, on schedule;
  - solid orange with a triangle = regular, late or overdue;
  - solid red with a cross = regular, stopped;
  - dashed grey = not regular;
  - dotted violet = relationship listed in the workbook (with or without payments).
- **Hover:** a regular line shows the total and the median payment; a casual line shows the amount only.
- **Click a node:** show its network, add it to the selection, or open its page.

## How a workbook company is found in the payment data

1. **By Power ID:** a 10-digit workbook id is looked up in `pkg_cust_dim` as a customer and as a parent. With
   *Group by family* on, the whole corporate family stands behind the node.
2. **By name:** every workbook name (and the PKG names of the same Power ID) is cleaned into a key (upper case, no
   brackets, punctuation, legal words or spaces) and matched exactly to outside counterparties in the last 12 months
   of `pkg_cpty_metrics_monthly`. A company with a Power ID can also be paid through an account at another bank, so
   both routes are used together.

The name search is the slow step. Its result is saved under `cache/` (set `DC_APP_CACHE` to move it), so a workbook is
searched once per month of data. **The cache holds client names: keep it out of git.**

## Definitions

- **Period:** 3, 6, 12 or 24 months ending at the latest month of `pkg_cust_pair_month_v4`.
- **Regular stream:** from `pkg_cust_reg_stream` (at least 4 payment days, a known cadence, 70% of gaps on time). The
  latest row of each stream in the last `REG_LOOKBACK` (12) months gives its state. A stream marked "not regular", or
  one that left the table without a stop, counts as not regular.
- **A line with several streams** (a family, or several accounts): its state and median payment are those of its
  largest regular stream; the hover says how many of its streams are regular.
- **Hubs:** outside counterparties at hub tier 2 or above. Hidden by default, except companies listed in the workbook.
- **DHE, PD:** latest month of `pkg_tm_cust_month`. For a family: DHE summed over its Power IDs, PD the worst.

## Check on the first run (not testable outside the cluster)

- **Name search speed** at upload (`NAME_MONTHS = 12` in `dc_core.py`; lower it if it is too slow).
- **The name-key expression in Impala** (`sql_name_key`): it is written without look-ahead so Impala's regular
  expressions accept it, and it gives the same keys as the notebook on test names.
- **Reading `pkg_cust_reg_stream`** for one selection (`REG_LOOKBACK = 12`; lower it if it is slow).
- **`pkg_dep_cust_month`:** the Deposits tab shows only if the table has `customer_pwr_id`, `month` and `dep_bal_eom`.
- Optional columns are used only when `DESCRIBE` shows them; a missing optional table hides its part of the app.

## Not in this draft

- Rail filter on the network (the pair table has no rail; it needs the account-level edges).
- Search for an outside counterparty by name (open one by clicking it, or from a customer's Counterparties tab).
- Merging several name-matched accounts of one company into one node.
