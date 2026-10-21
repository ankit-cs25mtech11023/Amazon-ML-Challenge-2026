"""Fine-tune the multilingual encoder on train pairs (contrastive, in-batch negatives).

Uses only pairs whose Source 1 entity falls in the "embedding fold" (hash % 4 == 0), so the
remaining 75% of train entities stay unseen for honest LightGBM training / evaluation.
Saves to work/emb_model_ft/.
"""
import os
import random

import polars as pl
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
BASE = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
OUT = os.environ.get("FT_DIR", "work/emb_model_ft")
NPAIRS = int(os.environ.get("NPAIRS", "1500000"))
BS = int(os.environ.get("BS", "384"))


def text(df):
    return df.with_columns(("query: " + pl.col("business_name").fill_null("") + ", " + pl.col("business_address").fill_null("")).alias("t")) \
             .select("entity_id", "country", "t")


def main():
    rd = lambda s: pl.read_csv(f"{DATA}/train/train_{s}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    s1 = text(rd("source1"))
    o = text(pl.concat([rd("source2"), rd("source3")]))
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    pairs = pairs.filter(pl.col("s1_id").hash(11) % 4 == 0)
    pairs = pairs.join(s1.rename({"entity_id": "s1_id", "t": "a"}), on="s1_id").join(o.select(pl.col("entity_id").alias("o_id"), pl.col("t").alias("b")), on="o_id")
    # one pair per S1 per batch is ideal (other positives of the same S1 are false negatives): keep <=1 pair per S1 per pass
    pairs = pairs.with_columns(pl.int_range(pl.len()).over("s1_id").alias("k")).filter(pl.col("k") < 2).sample(fraction=1.0, shuffle=True, seed=0)
    pairs = pairs.head(NPAIRS)
    # batches within a country give harder in-batch negatives
    pairs = pairs.sort(["k", "country"])
    ex = [InputExample(texts=[a, b]) for a, b in zip(pairs["a"].to_list(), pairs["b"].to_list())]
    # shuffle in blocks of one batch, keeping batches country-pure
    blocks = [ex[i:i + BS] for i in range(0, len(ex), BS)]
    random.Random(0).shuffle(blocks)
    ex = [e for b in blocks for e in b]
    print("training pairs", len(ex), flush=True)
    model = SentenceTransformer(BASE, device="cuda")
    model.max_seq_length = 96
    dl = DataLoader(ex, batch_size=BS, shuffle=False, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model, scale=30.0)
    model.fit(train_objectives=[(dl, loss)], epochs=1, warmup_steps=200, optimizer_params={"lr": 5e-5},
              use_amp=True, show_progress_bar=False, checkpoint_path=None)
    model.save(OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
