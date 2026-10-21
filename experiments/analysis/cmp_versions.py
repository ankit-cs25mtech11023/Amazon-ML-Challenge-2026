"""Compare submissions per country: avg matches per S1, empty share, and pairwise agreement."""
import polars as pl

s = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
pairs = {}
for v in ("v2", "v3", "v4", "v5"):
    m = pl.read_csv(f"output/{v}/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0) \
          .with_columns(pl.col("matched_entity_ids").fill_null(""))
    p = m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids") \
         .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1_id", "matched_entity_ids": "o_id"})
    pairs[v] = p
    e = s.join(p.group_by("s1_id").len(), on="s1_id", how="left").fill_null(0)
    print(v, e.group_by("country").agg(pl.col("len").mean().round(3).alias("avg"), (pl.col("len") == 0).mean().round(4).alias("empty")).sort("country").to_dicts())
for a, b in (("v3", "v5"), ("v4", "v5")):
    only_b = pairs[b].join(pairs[a], on=["s1_id", "o_id"], how="anti").join(s, on="s1_id")
    only_a = pairs[a].join(pairs[b], on=["s1_id", "o_id"], how="anti").join(s, on="s1_id")
    print(f"{b} not in {a}:", only_b.group_by("country").len().sort("country").to_dicts(),
          f"| {a} not in {b}:", only_a.group_by("country").len().sort("country").to_dicts())
