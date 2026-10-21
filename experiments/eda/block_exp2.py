import polars as pl, time
exec(open("work/block_exp.py").read().split("# restrict to o records")[0])
def evalc(name,c):
    c=c.unique(); hit=c.join(pairs,on=["a","b"]).height
    print(f"{name:55s} per-S1={c.height/s1.height:8.1f}  recall={hit/T:.4f}",flush=True); return c
for cap in [2000,20000]:
    for k in [1,2]:
        rr=r1.group_by("entity_id",maintain_order=True).head(k).filter(pl.col("df")<cap)
        evalc(f"S1 {k} rarest tok (df<{cap}) vs ALL O tokens",rr.join(oe,on=["country","nt"]).select(pl.col("entity_id").alias("a"),pl.col("entity_id_right").alias("b")))
# add number constraint
n1=s1.select("entity_id","num").explode("num").drop_nulls()
num_o=O.filter(pl.col("entity_id").is_in(oe["entity_id"].unique().implode())).select("entity_id",NUM.alias("num")).explode("num").drop_nulls().collect(engine="streaming")
rr=r1.group_by("entity_id",maintain_order=True).head(2)
c=rr.join(oe,on=["country","nt"]).select(pl.col("entity_id").alias("a"),pl.col("entity_id_right").alias("b")).unique()
c=c.join(n1.rename({"entity_id":"a"}),on="a").join(num_o.rename({"entity_id":"b"}),on=["b","num"]).select("a","b")
evalc("S1 2 rarest (no cap) vs ALL O tok AND shared number",c)
# recall ceiling of any-name-token-overlap (upper bound for token blocking)
print(f"{time.time()-t:.0f}s")
