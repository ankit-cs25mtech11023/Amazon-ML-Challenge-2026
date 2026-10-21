import polars as pl
D="work/data/"
gt=pl.read_parquet(D+"train_ground_truth.parquet")
s1=pl.read_parquet(D+"train_source1.parquet"); s2=pl.read_parquet(D+"train_source2.parquet"); s3=pl.read_parquet(D+"train_source3.parquet")
gt=gt.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element()!="")).alias("m"))
gt=gt.with_columns(pl.col("m").list.len().alias("n"),
  pl.col("m").list.eval(pl.element().str.starts_with("S2-")).list.sum().alias("n2"),
  pl.col("m").list.eval(pl.element().str.starts_with("S3-")).list.sum().alias("n3"))
print("singleton frac", (gt["n"]==0).mean())
print(gt["n"].value_counts().sort("n").head(15))
print("n2 dist",gt["n2"].value_counts().sort("n2").head(10))
print("n3 dist",gt["n3"].value_counts().sort("n3").head(10))
pairs=gt.select("source1_entity_id","m").explode("m").drop_nulls()
print("total pairs",pairs.height)
print("dup matched ids (matched to >1 S1):",pairs.height-pairs["m"].n_unique())
m2=pairs.filter(pl.col("m").str.starts_with("S2-"))["m"].n_unique(); m3=pairs.filter(pl.col("m").str.starts_with("S3-"))["m"].n_unique()
print(f"S2 matched frac {m2/s2.height:.3f}  S3 matched frac {m3/s3.height:.3f}")
for n,df in [("s1",s1),("s2",s2),("s3",s3)]:
    print(n, df["country"].value_counts().sort("count",descending=True).head(5).to_dicts(),
      "empty name",(df["business_name"]=="").mean() if df["business_name"].dtype==pl.Utf8 else None,
      "null addr", df["business_address"].is_null().mean(), "empty addr",(df["business_address"]=="").mean())
    print(" dup ids", df.height-df["entity_id"].n_unique(), " dup name+addr", df.height-df.select("business_name","business_address").unique().height)
for n in ["source1","source2","source3"]:
    t=pl.read_parquet(D+f"test_{n}.parquet"); print("test",n,t["country"].value_counts().sort("count",descending=True).to_dicts())
gt.select("source1_entity_id","n").write_parquet(D+"gt_n.parquet")
pairs.write_parquet(D+"train_pairs.parquet")
