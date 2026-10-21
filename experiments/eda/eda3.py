import polars as pl
pl.Config.set_fmt_str_lengths(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(30)
D="work/data/"
SUF=r"\b(the|inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|incorporated|l\.?l\.?c|plc|llp|sarl|sas|sa)\b"
def key(c):
    return (pl.col(c).str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}","")
            .str.replace_all(r"[^a-z0-9 ]"," ").str.replace_all(SUF," ").str.replace_all(r"\s+"," ").str.strip_chars()
            .str.split(" ").list.sort().list.join(" "))
s1=pl.read_parquet(D+"train_source1.parquet").with_columns(key("business_name").alias("k"))
o=pl.concat([pl.read_parquet(D+"train_source2.parquet"),pl.read_parquet(D+"train_source3.parquet")]).with_columns(key("business_name").alias("k"))
pairs=pl.read_parquet(D+"train_pairs.parquet")
gtn=pl.read_parquet(D+"gt_n.parquet")
matched=set(pairs["m"].to_list())
o=o.with_columns(pl.col("entity_id").is_in(pairs["m"].implode()).alias("is_m"))
s1k=s1.group_by("k").len().rename({"len":"n_s1"})
print("S1 keys dup frac (same name key shared by >1 S1):", (s1.join(s1k,on="k")["n_s1"]>1).mean())
oo=o.join(s1k,on="k",how="left").with_columns(pl.col("n_s1").fill_null(0))
print("matched S2/3: frac whose name-key hits some S1:", (oo.filter("is_m")["n_s1"]>0).mean())
print("UNmatched S2/3: frac whose name-key hits some S1:", (oo.filter(~pl.col("is_m"))["n_s1"]>0).mean())
# true pair key-eq
j=pairs.join(s1.select("entity_id","k"),left_on="source1_entity_id",right_on="entity_id").join(o.select("entity_id","k"),left_on="m",right_on="entity_id",suffix="_o")
print("true pair name-key equal:",(j["k"]==j["k_o"]).mean())
# singletons: does an S2/S3 record share their key?
sg=s1.join(gtn.filter(pl.col("n")==0),left_on="entity_id",right_on="source1_entity_id")
ok=o.group_by("k").len()
print("singleton S1 with key present in S2/3:", sg.join(ok,on="k",how="left")["len"].is_not_null().mean())
print("\n--- unmatched S2/3 whose key hits an S1 (decoys) sample")
d=oo.filter((~pl.col("is_m"))&(pl.col("n_s1")>0)).sample(8,seed=1).join(s1.select("k","business_name","business_address"),on="k",suffix="_s1")
print(d.select("entity_id","business_name","business_address","business_name_s1","business_address_s1").head(10))
print("\n--- random unmatched S2/3")
print(oo.filter(~pl.col("is_m")).sample(8,seed=2).select("entity_id","business_name","business_address","country"))
print("\n--- singleton S1 examples")
print(sg.sample(6,seed=1).select("entity_id","business_name","business_address"))
