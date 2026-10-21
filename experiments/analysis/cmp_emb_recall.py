"""Recall of base vs fine-tuned embedding retrieval on train entities NOT used for fine-tuning."""
import polars as pl

p = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
p = p.filter(pl.col("s1_id").hash(11) % 4 != 0)
T = p.height
for tag in ("emb", "embft"):
    e = pl.read_parquet(f"work/cand/train_{tag}.parquet")
    h = p.join(e.select("o_id", "s1_id", "erank"), on=["o_id", "s1_id"], how="left")
    print(tag, "  ".join(f"@{k}={(h['erank'] < k).sum() / T:.4f}" for k in (1, 3, 5, 10, 20)))
