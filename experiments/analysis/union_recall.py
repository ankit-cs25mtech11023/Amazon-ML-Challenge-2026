"""Recall of token vs embedding vs union candidate sets on train."""
import polars as pl

e = pl.read_parquet("work/cand/train_emb.parquet")
t = pl.read_parquet("work/cand/train_tok.parquet")
p = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
T = p.height
h = p.join(t.select("o_id", "s1_id", "rank"), on=["o_id", "s1_id"], how="left") \
     .join(e.select("o_id", "s1_id", "erank"), on=["o_id", "s1_id"], how="left")
for k in (1, 5, 10, 20):
    print(f"emb recall@{k}: {(h['erank'] < k).sum() / T:.4f}")
for kt, ke in [(10, 0), (10, 5), (10, 10), (20, 10), (20, 20), (30, 20)]:
    hit = ((h["rank"] < kt).fill_null(False) | (h["erank"] < ke).fill_null(False)).sum() / T
    n = pl.concat([t.filter(pl.col("rank") < kt).select("o_id", "s1_id"),
                   e.filter(pl.col("erank") < ke).select("o_id", "s1_id")]).unique().height
    print(f"union tok@{kt} + emb@{ke}: recall {hit:.4f}  rows {n / 1e6:.0f}M")
