import polars as pl, time
D="work/data/"
def norm(c): return pl.col(c).str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}","").str.replace_all(r"[^a-z0-9 ]"," ")
AT=norm("business_address").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1).str.strip_chars_start("0")).list.unique()
t=time.time()
S1=pl.read_parquet(D+"train_source1.parquet")
s1=S1.sample(100000,seed=0).with_columns(AT.alias("t"))
o=pl.concat([pl.read_parquet(D+"train_source2.parquet"),pl.read_parquet(D+"train_source3.parquet")]).with_columns(AT.alias("t"))
dfq=pl.concat([S1.with_columns(AT.alias("t")).select("country","t"),o.select("country","t")]).explode("t").drop_nulls().group_by("country","t").len().rename({"len":"df"})
pairs=pl.read_parquet(D+"train_pairs.parquet").rename({"source1_entity_id":"a","m":"b"}).filter(pl.col("a").is_in(s1["entity_id"].implode()))
T=pairs.height
r1=s1.select(pl.col("entity_id").alias("a"),"country","t").explode("t").drop_nulls().join(dfq,on=["country","t"]).sort("df").with_columns(pl.int_range(pl.len()).over("a").alias("rk"))
oe=o.select(pl.col("entity_id").alias("b"),"country","t").explode("t").drop_nulls().join(dfq,on=["country","t"])
print(f"prep {time.time()-t:.0f}s",flush=True)
for k,cap,minshare in [(3,2000,1),(5,2000,1),(5,2000,2),(8,5000,2),(8,20000,2),(12,20000,2),(12,20000,3)]:
    rr=r1.filter((pl.col("rk")<k)&(pl.col("df")<cap)).select("a","country","t")
    c=rr.join(oe.filter(pl.col("df")<cap).select("b","country","t"),on=["country","t"]).group_by("a","b").len().filter(pl.col("len")>=minshare)
    hit=c.join(pairs,on=["a","b"]).height
    # keep top-N by shared count
    top=c.sort("len",descending=True).group_by("a").head(50); hit50=top.join(pairs,on=["a","b"]).height
    print(f"S1 {k:2d} rarest addr tok df<{cap:5d} share>={minshare}: per-S1={c.height/s1.height:8.1f} recall={hit/T:.4f} | top50 recall={hit50/T:.4f}",flush=True)
print(f"{time.time()-t:.0f}s")
