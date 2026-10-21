import polars as pl
pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(40)
D="work/data/"
SUF=r"\b(the|inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|incorporated|plc|llp|sarl|sas|sa|group)\b"
def norm(c): return pl.col(c).str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}","").str.replace_all(r"[^a-z0-9 ]"," ")
NT=norm("business_name").str.replace_all(SUF," ").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique()
AT=norm("business_address").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique()
NUM=norm("business_address").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0")).list.unique()
s1=pl.read_parquet(D+"train_source1.parquet").sample(200000,seed=0)
o=pl.concat([pl.read_parquet(D+"train_source2.parquet"),pl.read_parquet(D+"train_source3.parquet")])
p=pl.read_parquet(D+"train_pairs.parquet").rename({"source1_entity_id":"a","m":"b"}).join(s1,left_on="a",right_on="entity_id").join(o,left_on="b",right_on="entity_id",suffix="_o")
f=lambda e,c: e.replace_all  # noop
p=p.with_columns(NT.alias("n1"),AT.alias("a1"),NUM.alias("u1"),
  norm("business_name_o").str.replace_all(SUF," ").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique().alias("n2"),
  norm("business_address_o").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars()>1)).list.unique().alias("a2"),
  norm("business_address_o").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0")).list.unique().alias("u2"))
p=p.with_columns(pl.col("n1").list.set_intersection("n2").list.len().alias("nn"),pl.col("a1").list.set_intersection("a2").list.len().alias("aa"),pl.col("u1").list.set_intersection("u2").list.len().alias("uu"),
  (pl.col("n2").list.len()==0).alias("n2empty"))
print("share >=1 name token:",(p["nn"]>0).mean())
print("share >=1 addr token:",(p["aa"]>0).mean())
print("share >=2 addr tokens:",(p["aa"]>1).mean())
print("share addr number:",(p["uu"]>0).mean())
print("name OR >=2 addr:",((p["nn"]>0)|(p["aa"]>1)).mean())
print("O-name has no latin tokens:",p["n2empty"].mean())
print("by country: share name tok", p.group_by("country").agg((pl.col("nn")>0).mean()).to_dicts())
print(p.filter(pl.col("nn")==0).sample(25,seed=5).select("business_name","business_name_o","business_address","business_address_o"))
