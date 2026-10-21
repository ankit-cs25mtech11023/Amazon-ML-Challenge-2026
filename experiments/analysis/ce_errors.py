"""Where does v8 still lose on fold-1 entities? Split errors by whether the pair was inside the cross-encoder band."""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
oof = pl.read_parquet("work/model_v6s1/oof.parquet").rename({"p": "p1"})
band = pl.read_parquet("work/ce/train_ce.parquet")
p = band["p1"].to_numpy().clip(1e-6, 1 - 1e-6)
Xb = np.column_stack([np.log(p / (1 - p)), band["ce"].to_numpy()])
band = band.with_columns(pl.Series("pb", LogisticRegression(C=10.0).fit(Xb, band["y"].to_numpy()).predict_proba(Xb)[:, 1]))
s1_eval = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
s1_eval = s1_eval.filter(s1_eval.hash(11) % 2 == 1)
pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
d = oof.join(band.select("o_id", "s1_id", "pb"), on=["o_id", "s1_id"], how="left").with_columns(
    pl.col("pb").is_not_null().alias("in_band"), pl.coalesce("pb", "p1").alias("p"))
best = d.sort("p", descending=True).unique("o_id", keep="first").filter(pl.col("p") >= 0.80)
best = best.filter(pl.col("s1_id").is_in(s1_eval.implode()))
tp = best.join(pe, on=["o_id", "s1_id"])
fp = best.join(pe, on=["o_id", "s1_id"], how="anti")
fn = pe.join(best, on=["o_id", "s1_id"], how="anti").join(d.select("o_id", "s1_id", "p1", "in_band", "p"), on=["o_id", "s1_id"], how="left")
print(f"eval entities {s1_eval.len()}  TP {tp.height}  FP {fp.height}  FN {fn.height}")
print("FP by in_band:", fp.group_by("in_band").len().to_dicts())
print("FN: not a candidate", fn["p1"].is_null().sum(),
      "| candidate, in band", (fn["in_band"] == True).sum(),
      "| candidate, NOT in band", ((fn["in_band"] == False) & fn["p1"].is_not_null()).sum())
nb = fn.filter((pl.col("in_band") == False) & pl.col("p1").is_not_null())
print("  not-in-band FN p1 buckets:", nb.select(pl.col("p1").cut([0.005, 0.02, 0.98]).alias("b")).group_by("b").len().sort("b").to_dicts())
