"""Cross-encoder data: uncertain candidate pairs (stage-1 p in [PLO, PHI], top-3 per record) with raw texts.

train: stage-1 OOF p (MODEL1/oof.parquet), labels, fold = hash(true S1 or own id) % 2 (same groups as train.py)
test : stage-1 p = mean of the OOF fold models (same calibration as train)
Writes work/ce/{split}_band.parquet: o_id, s1_id, p1, [y, fold], text_s1, text_o
"""
import glob
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
FEAT = os.environ.get("FEAT_DIR", "work/feat_v4")
M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
PLO, PHI = float(os.environ.get("PLO", "0.02")), float(os.environ.get("PHI", "0.98"))
TOPN = int(os.environ.get("TOPN", "3"))
BTAG = os.environ.get("BAND_TAG", "band")  # output name, e.g. band_hi / band_lo
CAP = int(os.environ.get("CAND_CAP", "0"))   # final blocking: top-CAP records per S1 entity
CAP_BY = os.environ.get("CAP_BY", "escore")  # "escore" (embedding cosine) or "p1" (learned filter: LightGBM score)
PMIN_BLOCK = float(os.environ.get("PMIN_BLOCK", "0"))


def texts(split, s):
    df = pl.read_csv(f"{DATA}/{split}/{split}_{s}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    return df.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")
                                    + " | " + pl.col("country").fill_null("")).alias("t"))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    os.makedirs("work/ce", exist_ok=True)
    if split == "train":
        p = pl.read_parquet(f"{M1}/oof.parquet").rename({"p": "p1"})
    elif os.path.exists("work/ce/test_p1_all.parquet"):
        p = pl.read_parquet("work/ce/test_p1_all.parquet")
    else:
        ms = [lgb.Booster(model_file=f) for f in sorted(glob.glob(f"{M1}/lgb_fold*.txt"))]
        cols = ms[0].feature_name()
        parts = []
        keep = None
        if CAP and CAP_BY == "escore":  # the matcher only runs on the final (capped) candidate set
            u = pl.read_parquet("work/cand/test_union4.parquet", columns=["o_id", "s1_id", "escore"])
            keep = u.filter(pl.col("escore").rank("ordinal", descending=True).over("s1_id") <= CAP).select("o_id", "s1_id")
        for f in sorted(glob.glob(f"{FEAT}/test/part_*.parquet")):
            d = pl.read_parquet(f)
            if keep is not None:
                d = d.join(keep, on=["o_id", "s1_id"], how="semi")
            X = d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
            parts.append(d.select("o_id", "s1_id").with_columns(pl.Series("p1", np.mean([m.predict(X, num_threads=64) for m in ms], axis=0))))
        p = pl.concat(parts)
        p.write_parquet("work/ce/test_p1_all.parquet")  # full stage-1 test probabilities (for the final blend)
    if CAP and CAP_BY == "p1" and split == "test":  # learned blocking filter -> final candidate set (test)
        p = p.filter(pl.col("p1") >= PMIN_BLOCK)
        p = p.filter(pl.col("p1").rank("ordinal", descending=True).over("s1_id") <= CAP)
    band = p.with_columns(pl.col("p1").rank("ordinal", descending=True).over("o_id").alias("r")) \
            .filter((pl.col("r") <= TOPN) & (pl.col("p1") >= PLO) & (pl.col("p1") <= PHI)).drop("r")
    if split == "train":
        pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
        band = band.join(pairs.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["o_id", "s1_id"], how="left") \
                   .with_columns(pl.col("y").fill_null(0))
        g = band.select("o_id").unique().join(pairs.select("o_id", pl.col("s1_id").alias("g")), on="o_id", how="left") \
                .with_columns((pl.coalesce("g", "o_id").hash(11) % 2).alias("fold"))
        band = band.join(g.select("o_id", "fold"), on="o_id")
    t1 = texts(split, "source1").rename({"entity_id": "s1_id", "t": "text_s1"})
    to = pl.concat([texts(split, "source2"), texts(split, "source3")]).rename({"entity_id": "o_id", "t": "text_o"})
    band = band.join(t1, on="s1_id").join(to, on="o_id")
    band.write_parquet(f"work/ce/{split}_{BTAG}.parquet")
    msg = f"{split} {BTAG}: rows {band.height}"
    if split == "train":
        msg += f", pos rate {band['y'].mean():.3f}, fold0 {(band['fold'] == 0).sum()}, fold1 {(band['fold'] == 1).sum()}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
