"""Stage -1: explode train ground truth into (source1_entity_id, m) pairs -> work/data/train_pairs.parquet."""
import os

import polars as pl

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
os.makedirs("work/data", exist_ok=True)
gt = pl.read_csv(f"{DATA}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
pairs = gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m")) \
          .select("source1_entity_id", "m").explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
pairs.write_parquet("work/data/train_pairs.parquet")
print("pairs", pairs.height)
