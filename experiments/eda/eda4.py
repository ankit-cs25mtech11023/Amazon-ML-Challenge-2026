import polars as pl
pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_rows(30)
D="work/data/"
for n in ["source1","source2","source3"]:
    t=pl.read_parquet(D+f"test_{n}.parquet").filter(pl.col("country")=="France").sample(5,seed=4)
    print(n); print(t.select("entity_id","business_name","business_address"))
o=pl.concat([pl.read_parquet(D+"train_source2.parquet"),pl.read_parquet(D+"train_source3.parquet")])
pairs=pl.read_parquet(D+"train_pairs.parquet"); s1=pl.read_parquet(D+"train_source1.parquet")
nl=o.filter(pl.col("business_name").str.contains(r"[ऀ-෿]"))
print("non-latin (Indic) names frac of all S2/3:", nl.height/o.height, " of India S2/3:", nl.height/o.filter(pl.col('country')=='India').height)
j=pairs.join(nl,left_on="m",right_on="entity_id").join(s1,left_on="source1_entity_id",right_on="entity_id",suffix="_s1").sample(6,seed=2)
print(j.select("business_name_s1","business_name","business_address"))
print("matched frac of indic-name records:", nl["entity_id"].is_in(pairs["m"].implode()).mean())
# address-empty or name-only records matched?
e=o.filter(pl.col("business_address")=="")
print("empty-addr matched frac", e["entity_id"].is_in(pairs["m"].implode()).mean())
