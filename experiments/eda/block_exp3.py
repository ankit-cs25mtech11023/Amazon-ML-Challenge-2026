import polars as pl, time
D="work/data/"
SUF=r"\b(the|inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|incorporated|plc|llp|sarl|sas|sa|group)\b"
def norm(c): return pl.col(c).str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}","").str.replace_all(r"[^a-z0-9 ]"," ")
NT=norm("business_name").str.replace_all(SUF," ").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique()
NUM=norm("business_address").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0")).list.unique()
t=time.time()
s1=pl.read_parquet(D+"train_source1.parquet").sample(300000,seed=0).with_columns(NT.alias("nt"),NUM.alias("num"))
o=pl.concat([pl.read_parquet(D+"train_source2.parquet"),pl.read_parquet(D+"train_source3.parquet")]).with_columns(NT.alias("nt"),NUM.alias("num"))
s1all=pl.read_parquet(D+"train_source1.parquet").with_columns(NT.alias("nt"))
dfq=pl.concat([s1all.select("country","nt"),o.select("country","nt")]).explode("nt").drop_nulls().group_by("country","nt").len().rename({"len":"df"})
pairs=pl.read_parquet(D+"train_pairs.parquet").rename({"source1_entity_id":"a","m":"b"}).filter(pl.col("a").is_in(s1["entity_id"].implode()))
T=pairs.height
r1=s1.select("entity_id","country","nt").explode("nt").drop_nulls().join(dfq,on=["country","nt"]).sort("df").with_columns(pl.int_range(pl.len()).over("entity_id").alias("rk"))
oe=o.select(pl.col("entity_id").alias("b"),"country","nt").explode("nt").drop_nulls()
on=o.select(pl.col("entity_id").alias("b"),"num").explode("num").drop_nulls()
n1=s1.select(pl.col("entity_id").alias("a"),"num").explode("num").drop_nulls()
print(f"prep {time.time()-t:.0f}s  T={T}",flush=True)
res={}
def evalc(name,c):
    c=c.unique(); hit=c.join(pairs,on=["a","b"]).height
    print(f"{name:60s} per-S1={c.height/s1.height:8.1f}  recall={hit/T:.4f}",flush=True); res[name]=c; return c
for k,cap in [(1,5000),(2,5000),(3,5000),(2,50000)]:
    rr=r1.filter((pl.col("rk")<k)&(pl.col("df")<cap)).select(pl.col("entity_id").alias("a"),"country","nt")
    c=rr.join(oe,on=["country","nt"]).select("a","b")
    evalc(f"S1 {k} rarest tok (df<{cap}) vs ALL O tok",c)
    evalc(f"  ... AND shared addr number",c.join(n1,on="a").join(on,on=["b","num"]).select("a","b"))
# any-token-overlap ceiling
rr=r1.select(pl.col("entity_id").alias("a"),"country","nt").filter(pl.col("nt").is_in(r1.filter(pl.col("df")<200000)["nt"].unique().implode()))
full=pairs.join(s1.select(pl.col("entity_id").alias("a"),pl.col("nt").alias("n1")),on="a").join(o.select(pl.col("entity_id").alias("b"),pl.col("nt").alias("n2")),on="b")
print("ceiling: true pairs sharing >=1 name token:", (full.select(pl.col("n1").list.set_intersection("n2").list.len()>0).to_series()).mean())
print("true pairs with addr number shared:",full.join(n1,on="a").join(on,on=["b","num"]).select("a","b").unique().height/T)
print(f"{time.time()-t:.0f}s")
