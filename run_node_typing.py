"""
run_node_typing.py
===================
Standalone runner for JUST the node-typing step (pkg_custom_metrics.
build_node_typing). This is the cheap, one-time part of the composition
metrics — a single pass over all snapshots to classify every node by
entity_type / naics_status / node_type.

Run this BEFORE the full pkg_pipeline.py run so you can sanity-check the
classification (and extend PLACEHOLDER_NAICS from real EDA) without paying
for the full graph-metrics computation.

Usage:
    python run_node_typing.py
    python run_node_typing.py --data-dir ../data --out ../metrics/node_typing.csv

Output: node_typing.csv (node, entity_type, naics_status, node_type,
naics_clean) plus a console summary of the distributions to eyeball.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import pandas as pd

import pkg_custom_metrics as cm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("run_node_typing")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="../data",
                    help="directory containing cust_YYYY-MM.csv snapshots")
    ap.add_argument("--out", default="../metrics/node_typing.csv",
                    help="output path for the typing table")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "cust_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no cust_*.csv under {args.data_dir}")
    log.info("building node typing from %d snapshots", len(paths))

    typing = cm.build_node_typing(paths)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    typing.to_csv(args.out, index=False)
    log.info("wrote %s (%d nodes)", args.out, len(typing))

    # ---- sanity-check summary -------------------------------------------
    print("\n=== node_type distribution ===")
    print(typing["node_type"].value_counts(normalize=True)
          .mul(100).round(1).astype(str).add("%").to_string())

    print("\n=== entity_type distribution ===")
    print(typing["entity_type"].value_counts(normalize=True)
          .mul(100).round(1).astype(str).add("%").to_string())

    print("\n=== naics_status distribution ===")
    print(typing["naics_status"].value_counts(normalize=True)
          .mul(100).round(1).astype(str).add("%").to_string())

    unknown_share = (typing["entity_type"] == "unknown").mean()
    print(f"\nunknown entity_type share: {unknown_share:.1%}")
    if unknown_share > 0.15:
        print("  -> HIGH. Inspect a sample of 'unknown' names below and "
              "consider extending _BUSINESS_TOKENS in pkg_custom_metrics.py.")

    ph_values = (pd.concat([pd.read_csv(p, usecols=["source_naics"])
                            .rename(columns={"source_naics": "naics"})
                            for p in paths[:1]])["naics"]
                 .dropna().value_counts().head(20))
    print("\n=== most common NAICS values in snapshot 1 (verify against "
          "PLACEHOLDER_NAICS) ===")
    print(ph_values.to_string())

    print("\n=== sample of 'unknown' entity_type names (spot-check) ===")
    unk_nodes = typing.loc[typing["entity_type"] == "unknown", "node"].head(20)
    if len(unk_nodes):
        names = (pd.read_csv(paths[0], usecols=["source", "source_name"])
                 .drop_duplicates("source")
                 .set_index("source")["source_name"])
        for n in unk_nodes:
            if n in names.index:
                print(f"  {n}: {names.loc[n]}")


if __name__ == "__main__":
    main()
