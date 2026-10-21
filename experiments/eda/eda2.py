import polars as pl, random
pl.Config.set_fmt_str_lengths(90); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
D="work/data/"
s1=pl.read_parquet(D+"train_source1.parquet"); s2=pl.read_parquet(D+"train_source2.parquet"); s3=pl.read_parquet(D+"train_source3.parquet")
other=pl.concat([s2,s3])
pairs=pl.read_parquet(D+"train_pairs.parquet")
j=pairs.join(s1,left_on="source1_entity_id",right_on="entity_id").join(other,left_on="m",right_on="entity_id",suffix="_o")
print("country mismatch frac",(j["country"]!=j["country_o"]).mean())
print("exact name eq",(j["business_name"]==j["business_name_o"]).mean(),
      " lower-eq",(j["business_name"].str.to_lowercase()==j["business_name_o"].str.to_lowercase()).mean())
print("exact addr eq",(j["business_address"]==j["business_address_o"]).mean())
# id leakage checks
num=lambda c: pl.col(c).str.slice(3).cast(pl.Int64)
k=j.select(num("source1_entity_id").alias("a"),num("m").alias("b")).sample(200000,seed=0)
import numpy as np
print("corr ids",np.corrcoef(k["a"].to_numpy(),k["b"].to_numpy())[0,1])
# row-order leakage
s1i=s1.with_row_index("r1").select("entity_id","r1"); oi=other.with_row_index("r2").select("entity_id","r2")
kk=pairs.join(s1i,left_on="source1_entity_id",right_on="entity_id").join(oi,left_on="m",right_on="entity_id").sample(200000,seed=0)
print("corr rows",np.corrcoef(kk["r1"].to_numpy(),kk["r2"].to_numpy())[0,1])
for c in ["US","India"]:
    ids=j.filter(pl.col("country")==c)["source1_entity_id"].unique().sample(4,seed=3)
    print(f"\n==== {c} examples")
    print(j.filter(pl.col("source1_entity_id").is_in(ids.implode())).sort("source1_entity_id").select("source1_entity_id","business_name","business_address","m","business_name_o","business_address_o"))
# non-latin scripts
for n,df in [("s1",s1),("s2",s2),("s3",s3)]:
    print(n,"devanagari name frac",df["business_name"].str.contains(r"[ऀ-ॿ]").mean(), "non-ascii",df["business_name"].str.contains(r"[^\x00-\x7F]").mean())
