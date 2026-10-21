"""Score test candidate pairs, assign each S2/S3 record to its best S1, write the two submission TSVs."""
import glob
import os
import subprocess
import sys

import lightgbm as lgb
import polars as pl

FEAT = os.environ.get("FEAT_DIR", "work/feat")
MODEL = os.environ.get("MODEL_DIR", "work/model")
OUT = os.environ.get("OUT_DIR", "output")
TEST_DIR = os.environ.get("TEST_DIR", "student_resource/dataset/test")


def id_lists(df: pl.DataFrame, s1: pl.Series, col: str) -> pl.DataFrame:
    g = df.group_by("s1_id").agg(pl.col("o_id").unique().sort().str.join(",").alias(col))
    return pl.DataFrame({"source1_entity_id": s1}).join(g.rename({"s1_id": "source1_entity_id"}),
                                                        on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))


def main():
    os.makedirs(OUT, exist_ok=True)
    m = lgb.Booster(model_file=f"{MODEL}/lgb.txt")
    tau = float(os.environ.get("TAU") or open(f"{MODEL}/threshold.txt").read())
    cols = m.feature_name()
    parts = []
    for p in sorted(glob.glob(f"{FEAT}/test/part_*.parquet")):
        d = pl.read_parquet(p)
        prob = m.predict(d.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy(), num_threads=64)
        parts.append(d.select("o_id", "s1_id").with_columns(pl.Series("p", prob)))
    pred = pl.concat(parts)
    s1 = pl.read_csv(f"{TEST_DIR}/test_source1.tsv", separator="\t", quote_char=None, infer_schema_length=0)["entity_id"]

    cand = id_lists(pred, s1, "candidate_entity_ids")
    best = pred.sort("p", descending=True).unique("o_id", keep="first").filter(pl.col("p") >= tau)
    match = id_lists(best, s1, "matched_entity_ids")
    cand.write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    match.write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    n_nonempty = (match["matched_entity_ids"] != "").sum()
    print(f"tau={tau:.3f} matched pairs={best.height} S1 with matches={n_nonempty}/{match.height}")
    sys.exit(subprocess.call([sys.executable, "student_resource/utils/validate_submission.py",
                              "--matching", f"{OUT}/matching_results.tsv", "--candidate", f"{OUT}/candidate_pairs.tsv",
                              "--test-dir", TEST_DIR]))


if __name__ == "__main__":
    main()
