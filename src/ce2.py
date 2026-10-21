"""Second cross-encoder: trained on both entity folds (minus a 10% entity holdout), scores ALL final test candidates.

  python src/ce2.py data     # build train/holdout pairs and the test candidate pairs (with texts)
  python src/ce2.py train    # fine-tune BAAI/bge-reranker-v2-m3 on the training pairs
  python src/ce2.py score holdout|test
Holdout = entities whose group hash % 10 == 0 (group = true S1 of the record, or the record itself if unmatched).
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
BASE = os.environ.get("CE_BASE", "BAAI/bge-reranker-v2-m3")
OUT = os.environ.get("CE2_DIR", "work/ce2/model")
MAXLEN = int(os.environ.get("CE_MAXLEN", "160"))
BS = int(os.environ.get("CE_BS", "64"))
LR = float(os.environ.get("CE_LR", "2e-5"))
HI_POS = int(os.environ.get("HI_POS", "800000"))     # down-sample near-certain positives
M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
CAP, PMIN = int(os.environ.get("CAND_CAP", "12")), float(os.environ.get("PMIN_BLOCK", "0.001"))
torch.manual_seed(0)
os.makedirs("work/ce2", exist_ok=True)


def texts(split, s):
    df = pl.read_csv(f"{DATA}/{split}/{split}_{s}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    return df.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")
                                    + " | " + pl.col("country").fill_null("")).alias("t"))


def with_texts(d, split):
    t1 = texts(split, "source1").rename({"entity_id": "s1_id", "t": "text_s1"})
    to = pl.concat([texts(split, "source2"), texts(split, "source3")]).rename({"entity_id": "o_id", "t": "text_o"})
    return d.join(t1, on="s1_id").join(to, on="o_id")


def data():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    bands = [pl.read_parquet(f"work/ce/train_{b}.parquet").select("o_id", "s1_id", "p1", "y") for b in ("band", "band_hi", "band_lo")]
    tr = pl.concat(bands).unique(["o_id", "s1_id"])
    g = tr.select("o_id").unique().join(pairs.select("o_id", pl.col("s1_id").alias("g")), on="o_id", how="left") \
          .with_columns((pl.coalesce("g", "o_id").hash(13) % 10 == 0).alias("holdout"))
    tr = tr.join(g.select("o_id", "holdout"), on="o_id")
    hold = tr.filter("holdout")
    fit = tr.filter(~pl.col("holdout"))
    easy = fit.filter((pl.col("p1") > 0.98) & (pl.col("y") == 1))
    fit = pl.concat([fit.filter(~((pl.col("p1") > 0.98) & (pl.col("y") == 1))), easy.sample(min(HI_POS, easy.height), seed=0)])
    print(f"train pairs {fit.height} (pos rate {fit['y'].mean():.3f}), holdout pairs {hold.height}", flush=True)
    with_texts(fit.sample(fraction=1.0, shuffle=True, seed=1), "train").write_parquet("work/ce2/train.parquet")
    with_texts(hold, "train").write_parquet("work/ce2/holdout.parquet")
    # test: every final candidate (learned blocking filter)
    p = pl.read_parquet("work/ce/test_p1_all.parquet").filter(pl.col("p1") >= PMIN)
    p = p.filter(pl.col("p1").rank("ordinal", descending=True).over("s1_id") <= CAP)
    with_texts(p, "test").write_parquet("work/ce2/test.parquet")
    print(f"test candidates {p.height}", flush=True)


def train():
    d = pl.read_parquet("work/ce2/train.parquet")
    a, b, y = d["text_s1"].to_list(), d["text_o"].to_list(), d["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lens = np.array([len(x) + len(z) for x, z in zip(a, b)])
    order = np.argsort(lens + np.random.default_rng(0).uniform(0, 40, len(lens)))
    chunks = [order[i:i + BS] for i in range(0, len(order), BS)]
    np.random.default_rng(1).shuffle(chunks)
    steps = len(chunks)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * max(0.0, 1 - s / steps))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t = time.time()
    for s, idx in enumerate(chunks):
        enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(**enc).logits.squeeze(-1)
        loss = lossf(logit.float(), torch.from_numpy(y[idx]).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if s % 1000 == 0:
            print(f"step {s}/{steps} loss {loss.item():.4f} {time.time()-t:.0f}s", flush=True)
        if s and s % 10000 == 0:
            model.save_pretrained(OUT); tok.save_pretrained(OUT)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved", OUT, flush=True)


@torch.no_grad()
def score(which):
    d = pl.read_parquet(f"work/ce2/{which}.parquet")
    a, b = d["text_s1"].to_list(), d["text_o"].to_list()
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT, torch_dtype=torch.bfloat16).cuda().eval()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    t = time.time()
    for n, i in enumerate(range(0, len(order), 512)):
        idx = order[i:i + 512]
        enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if n % 500 == 0:
            print(f"scored {min(i + 512, len(a))}/{len(a)} {time.time()-t:.0f}s", flush=True)
    d.select([c for c in d.columns if not c.startswith("text_")]).with_columns(pl.Series("ce2", out)) \
     .write_parquet(f"work/ce2/{which}_scored.parquet")
    print("done", which, len(a), flush=True)


if __name__ == "__main__":
    {"data": data, "train": train}.get(sys.argv[1], lambda: score(sys.argv[2]))()
