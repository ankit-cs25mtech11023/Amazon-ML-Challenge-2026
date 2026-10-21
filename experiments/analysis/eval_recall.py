"""Recall of a candidate file against train ground truth, by rank cutoff and country."""
import sys

import polars as pl

cand = pl.read_parquet(sys.argv[1] if len(sys.argv) > 1 else "work/cand/train_tok.parquet")
pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
country = pl.concat([pl.read_parquet(f"work/norm/train_source{k}.parquet", columns=["entity_id", "country"]) for k in (2, 3)]) \
    .rename({"entity_id": "o_id"})
T = pairs.height
hit = pairs.join(cand, on=["o_id", "s1_id"], how="left").join(country, on="o_id")
print(f"true pairs {T}, candidate rows {cand.height} ({cand.height / cand['o_id'].n_unique():.1f} per O record)")
for k in (1, 2, 3, 5, 10, 20, 30, 50):
    print(f"  recall@{k:<3d} {(hit['rank'] < k).sum() / T:.4f}")
print(hit.group_by("country").agg((pl.col("rank").is_not_null()).mean().alias("recall_any")).sort("country"))
