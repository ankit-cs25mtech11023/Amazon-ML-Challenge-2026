"""Candidate-set size vs recall on train, for a tighter final blocking stage.

Blocking stage 2 (rule, no learned model): keep a pair if it is in the record's top-KT token hits or top-KE
fine-tuned-embedding hits; then cap each Source 1 entity at its top-CAP records by embedding cosine.
Reports candidates per S1 entity and recall of true pairs, overall and for the pairs v9.1-style blending accepts.
"""
import polars as pl

u = pl.read_parquet("work/cand/train_union4.parquet", columns=["o_id", "s1_id", "rank", "erank", "escore", "score"])
pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"}) \
          .with_columns(pl.lit(1).alias("y"))
n_s1 = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"]).height
T = pairs.height
u = u.join(pairs, on=["o_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
print(f"full union: {u.height/n_s1:.1f} per S1, recall {u['y'].sum()/T:.4f}", flush=True)
for kt, ke in [(10, 10), (5, 5), (3, 5), (3, 3), (2, 3), (2, 2), (1, 2)]:
    v = u.filter((pl.col("rank") < kt) | (pl.col("erank") < ke))
    v = v.with_columns(pl.col("escore").rank("ordinal", descending=True).over("s1_id").alias("srank"))
    line = f"tok<{kt:2d} | emb<{ke:2d}: {v.height/n_s1:6.1f}/S1 recall {v['y'].sum()/T:.4f}"
    for cap in (20, 12, 8):
        w = v.filter(pl.col("srank") <= cap)
        line += f" | cap{cap}: {w.height/n_s1:5.1f}/S1 rec {w['y'].sum()/T:.4f}"
    print(line, flush=True)
