"""Label-free diagnostics: compare France vs US/India on test (and train) score/decision statistics."""
import glob

import lightgbm as lgb
import polars as pl

pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
m = lgb.Booster(model_file="work/model/lgb.txt")
cols = m.feature_name()
country = pl.concat([pl.read_parquet(f"work/norm/test_source{k}.parquet", columns=["entity_id", "country"]) for k in (2, 3)]).rename({"entity_id": "o_id"})
parts = []
for f in sorted(glob.glob("work/feat/test/part_*.parquet")):
    d = pl.read_parquet(f)
    d = d.with_columns(pl.Series("p", m.predict(d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy(), num_threads=64)))
    parts.append(d)
d = pl.concat(parts).join(country, on="o_id")
d.select("o_id", "s1_id", "p").write_parquet("work/model/test_pred_v1.parquet")
top = d.filter(pl.col("rank") == 0)
best = d.sort("p", descending=True).unique("o_id", keep="first")
print("== per country: token-retrieval top score, gap, model best p, assigned share (tau .65)")
print(top.group_by("country").agg(pl.col("score").median().alias("top_score_med"), pl.col("gap_second").median().alias("gap2_med"),
      pl.col("atok_jac").median().alias("addr_jac_med"), pl.col("n_tset").median().alias("name_tset_med")).sort("country"))
print(best.group_by("country").agg(pl.col("p").median().alias("p_med"), (pl.col("p") >= 0.65).mean().alias("assigned"),
      ((pl.col("p") > 0.2) & (pl.col("p") < 0.8)).mean().alias("uncertain")).sort("country"))
# train reference: expected assigned share = matched fraction ~0.74
print("train reference: ~74% of S2/S3 records have a true match")
s1 = pl.read_csv("student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)
o = pl.concat([pl.read_csv(f"student_resource/dataset/test/test_source{k}.tsv", separator="\t", quote_char=None, infer_schema_length=0) for k in (2, 3)])
unc = best.filter((pl.col("country") == "France") & (pl.col("p") > 0.3) & (pl.col("p") < 0.7)).sample(15, seed=1)
x = unc.join(o.select(pl.col("entity_id").alias("o_id"), pl.col("business_name").alias("o_name"), pl.col("business_address").alias("o_addr")), on="o_id") \
       .join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s_name"), pl.col("business_address").alias("s_addr")), on="s1_id")
print("== uncertain France examples"); print(x.select("p", "o_name", "s_name", "o_addr", "s_addr"))
lo = best.filter((pl.col("country") == "France") & (pl.col("p") < 0.3)).sample(10, seed=2)
x = lo.join(o.select(pl.col("entity_id").alias("o_id"), pl.col("business_name").alias("o_name"), pl.col("business_address").alias("o_addr")), on="o_id") \
      .join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("s_name"), pl.col("business_address").alias("s_addr")), on="s1_id")
print("== low-p France best candidates (rejected)"); print(x.select("p", "o_name", "s_name", "o_addr", "s_addr"))
