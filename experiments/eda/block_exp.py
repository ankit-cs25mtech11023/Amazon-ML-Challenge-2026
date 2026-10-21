import polars as pl, time
D="work/data/"
SUF=r"\b(the|inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|incorporated|plc|llp|sarl|sas|sa|group)\b"
def norm(c): return pl.col(c).str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}","").str.replace_all(r"[^a-z0-9 ]"," ")
NT=norm("business_name").str.replace_all(SUF," ").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique()
NUM=norm("business_address").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0")).list.unique()
t=time.time()
S1=pl.scan_parquet(D+"train_source1.parquet"); O=pl.concat([pl.scan_parquet(D+"train_source2.parquet"),pl.scan_parquet(D+"train_source3.parquet")])
# doc freq over all records (streaming)
dfq=pl.concat([S1,O]).select("country",NT.alias("nt")).explode("nt").drop_nulls().group_by("country","nt").len().rename({"len":"df"}).collect(engine="streaming")
print("df table",dfq.shape,f"{time.time()-t:.0f}s",flush=True)
s1=S1.collect().sample(100000,seed=0).with_columns(NT.alias("nt"),NUM.alias("num"))
pairs=pl.read_parquet(D+"train_pairs.parquet").rename({"source1_entity_id":"a","m":"b"}).filter(pl.col("a").is_in(s1["entity_id"].implode()))
T=pairs.height
r1=s1.select("entity_id","country","nt").explode("nt").drop_nulls().join(dfq,on=["country","nt"]).sort("df").group_by("entity_id",maintain_order=True).head(2)
toks=r1.select("country","nt").unique()
# candidate-side: only records containing a sample token
o=O.with_columns(NT.alias("nt"),NUM.alias("num")).select("entity_id","country","nt","num")
oe=o.select("entity_id","country","nt").explode("nt").join(toks.lazy(),on=["country","nt"]).collect(engine="streaming")
print("o token rows",oe.height,f"{time.time()-t:.0f}s",flush=True)
# restrict to o records' 2 rarest tokens overall
oall=o.filter(pl.col("entity_id").is_in(oe["entity_id"].unique().implode())).collect(engine="streaming")
ro=oall.select("entity_id","country","nt").explode("nt").drop_nulls().join(dfq,on=["country","nt"]).sort("df").group_by("entity_id",maintain_order=True).head(2)
def evalc(name,c):
    c=c.unique(); hit=c.join(pairs,on=["a","b"]).height
    print(f"{name:45s} per-S1={c.height/s1.height:8.1f}  recall={hit/T:.4f}",flush=True); return c
cA=evalc("share one of 2 rarest name tokens (df<5000)",r1.filter(pl.col("df")<5000).join(ro,on=["country","nt"]).select(pl.col("entity_id").alias("a"),pl.col("entity_id_right").alias("b")))
cA2=evalc("share one of 2 rarest name tokens (no cap)",r1.join(ro,on=["country","nt"]).select(pl.col("entity_id").alias("a"),pl.col("entity_id_right").alias("b")))
n1=s1.select("entity_id","num").explode("num").drop_nulls(); no=oall.select("entity_id","num").explode("num").drop_nulls()
x1=r1.join(n1,on="entity_id"); xo=ro.join(no,on="entity_id")
cB=evalc("rare name token AND shared addr number",x1.join(xo,on=["country","nt","num"]).select(pl.col("entity_id").alias("a"),pl.col("entity_id_right").alias("b")))
cU=evalc("union A(cap)+B",pl.concat([cA,cB]))
# what do misses look like
miss=pairs.join(cU,on=["a","b"],how="anti")
print("miss: S2 frac",miss["b"].str.starts_with("S2").mean())
full=pl.concat([pl.read_parquet(D+"train_source2.parquet"),pl.read_parquet(D+"train_source3.parquet")])
mm=miss.sample(min(12,miss.height),seed=1).join(s1.select("entity_id","business_name"),left_on="a",right_on="entity_id").join(full.select("entity_id","business_name","business_address"),left_on="b",right_on="entity_id",suffix="_o")
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_rows(20)
print(mm.select("business_name","business_name_o","business_address"))
print(f"{time.time()-t:.0f}s")
