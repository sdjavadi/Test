# Graph ML workshop kit

Four lab notebooks and a synthetic business payment network for a three-hour introduction to graph ML on payment data. Everything in `data/` is invented, with known answers planted in it so every lab can check itself.

## Contents

| Path | What it is |
|---|---|
| `labs/Lab1_graph_basics_paths_centrality.ipynb` | Payment table as a graph, hubs, shortest paths, degree / betweenness / PageRank (PageRank by hand) |
| `labs/Lab2_local_structure.ipynb` | Ego networks, overlap coefficient, funnels, pass-through, round trips, payee consensus |
| `labs/Lab3_communities.ipynb` | Karate club, modularity, Louvain with a resolution slider, ecosystems, communities over time |
| `labs/Lab4_features_links_embeddings_gnn.ipynb` | Link prediction, graph features in a churn model, FastRP embeddings, a GraphSAGE forward pass, SGC |
| `labs/labkit.py` | Shared loading and plotting helpers (the graph ideas live in the notebooks) |
| `data/` | The synthetic network (CSV) |
| `paygraph_synth.py` | The generator: `python paygraph_synth.py --out my_data --seed 7` builds a fresh network with the same planted structure |

Keep the shipped `data/` for the labs: their example accounts and the numbers in the slides refer to it. A fresh run (any seed) plants the same patterns under different ids and gives slightly different numbers; the same seed always gives the same files.

## Running

Open the notebooks from the `labs/` folder and run top to bottom. Each lab runs in under a minute on a laptop.

Libraries: pandas, numpy, scipy, networkx (2.8 or later), plotly, scikit-learn, matplotlib, and lightgbm for Lab 4 (it falls back to scikit-learn if LightGBM is missing). Tested on Python 3.10 with pandas 2.3.3, numpy 2.2.6, networkx 3.4 and plotly 5.24.

To point the labs at a copy of the data elsewhere, set the environment variable `PAYGRAPH_DATA` to that folder.

## The data

| File | Rows | One row is |
|---|---|---|
| `nodes.csv` | 3,705 | an account holder: `node_id`, `name`, `party_type` (O business, P person), `sector_code`, `sector`, `region`, `size_band` |
| `edges_monthly.csv` | about 200,000 | payer → payee totals for one month: `month`, `src`, `dst`, `amount`, `n_txn` (2025-09 to 2026-08) |
| `transactions_2026-08.csv` | about 33,000 | daily payments in the last month: `txn_id`, `date`, `src`, `dst`, `amount` |
| `vendor_payments.csv` | about 1,200 | payer → vendor payments with the vendor account used (payee consensus) |
| `truth.csv` | 3,705 | the planted answers: ecosystem, archetype, churn label, vendor scenario |

What is planted:

- 12 ecosystems of businesses (a region plus a supply chain) and 6 hubs that turn every picture into a hairball
- 6 funnels, 6 pass-through accounts and 5 round-trippers, next to legitimate look-alikes: 4 property managers and 8 two-way traders
- 4 vendors for payee consensus: two one-payer frauds, one real bank move, one unchanged
- churn with contagion: some businesses drop suppliers before leaving, some leave when their main partner does, and one ecosystem shrinks
- second half-year: two ecosystems merge, one is born, one shrinks

`truth.csv` is only read in the reveal cells. Names are invented; any resemblance to a real company is coincidental.
