"""Break down OOF macro-F0.5 loss: missing candidates vs wrong decisions, by country / singleton / source."""
import os

import numpy as np
import polars as pl

M = os.environ.get("MODEL_DIR", "work/model")

tau = float(open(f"{M}/threshold.txt").read())
oof = pl.read_parquet(f"{M}/oof.parquet")
pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
s1 = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
if os.environ.get("EXCLUDE_FT"):
    s1 = s1.filter(pl.col("s1_id").hash(11) % 4 != 0)
    pairs = pairs.filter(pl.col("s1_id").hash(11) % 4 != 0)
    oof = oof.filter(~pl.col("o_id").is_in(pl.read_parquet("work/data/train_pairs.parquet").filter(pl.col("source1_entity_id").hash(11) % 4 == 0)["m"].implode()))
best = oof.sort("p", descending=True).unique("o_id", keep="first").filter(pl.col("p") >= tau).select("s1_id", "o_id")
incand = oof.join(pairs, on=["o_id", "s1_id"]).select("o_id", "s1_id")
tp = best.join(pairs, on=["s1_id", "o_id"])
e = s1.join(best.group_by("s1_id").len().rename({"len": "npred"}), on="s1_id", how="left") \
    .join(pairs.group_by("s1_id").len().rename({"len": "ntrue"}), on="s1_id", how="left") \
    .join(tp.group_by("s1_id").len().rename({"len": "tp"}), on="s1_id", how="left") \
    .join(incand.group_by("s1_id").len().rename({"len": "ncand_true"}), on="s1_id", how="left").fill_null(0)
p = e["tp"] / e["npred"].clip(lower_bound=1); r = e["tp"] / e["ntrue"].clip(lower_bound=1)
f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0.0).to_numpy()
f = np.where(e["ntrue"].to_numpy() == 0, (e["npred"] == 0).to_numpy().astype(float), f)
e = e.with_columns(pl.Series("f", f), (pl.col("ntrue") == 0).alias("singleton"))
N = e.height
print(f"macro F0.5 = {e['f'].mean():.5f}   total loss = {(1-e['f']).sum()/N:.5f}")
print(e.group_by("country", "singleton").agg(pl.len(), pl.col("f").mean().alias("meanF"), ((1 - pl.col("f")).sum() / N).alias("loss_share")).sort("loss_share", descending=True))
fp = best.join(pairs, on=["s1_id", "o_id"], how="anti")
fn = pairs.join(best, on=["s1_id", "o_id"], how="anti")
fn_nocand = fn.join(incand, on=["o_id", "s1_id"], how="anti").height
print(f"pred pairs {best.height}  TP {tp.height}  FP {fp.height}  FN {fn.height} (of which not in candidates {fn_nocand})")
wrong_s1 = fp.join(pairs.select("o_id"), on="o_id").height
print(f"FP where record truly belongs to ANOTHER S1: {wrong_s1}; FP where record is a true distractor: {fp.height-wrong_s1}")
