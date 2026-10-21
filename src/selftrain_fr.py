"""Self-training (pseudo-labelling) of second cross-encoder for a country without training labels (France).

Uses only the unlabeled test records and the pipeline's own predictions (no labels, no external data):
  python src/selftrain_fr.py data    # pseudo-labelled French pairs + US/India replay pairs
  python src/selftrain_fr.py train   # continue fine-tuning second cross-encoder from work/ce2/model
  python src/selftrain_fr.py score   # re-score this country's test candidates -> work/ce2/test_scored_fr.parquet
Pseudo-positives : records whose top candidate has ensemble p >= POS_P  (label 1)
Pseudo-negatives : the losing candidates of those confident records, and any candidate with p <= NEG_P  (label 0)
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

COUNTRY = os.environ.get("ST_COUNTRY", "France")          # one country or a comma-separated list, e.g. "US,India"
COUNTRIES = [c.strip() for c in COUNTRY.split(",")]
BASE_SCORED = os.environ.get("ST_BASE_SCORED", "work/ce2/test_scored.parquet")   # scores kept for everything not re-scored
BAND = os.environ.get("ST_SCORE_BAND", "")                                    # e.g. "0.005,0.995": only re-score uncertain pairs
POS_P, NEG_P = float(os.environ.get("POS_P", "0.97")), float(os.environ.get("NEG_P", "0.03"))
NPOS, NNEG, NREPLAY = int(os.environ.get("NPOS", "300000")), int(os.environ.get("NNEG", "300000")), int(os.environ.get("NREPLAY", "300000"))
SRC = os.environ.get("CE2_DIR", "work/ce2/model")
OUT = os.environ.get("CE2_FR_DIR", "work/ce2/model_fr")
MAXLEN, BS, LR = int(os.environ.get("CE_MAXLEN", "160")), int(os.environ.get("CE_BS", "64")), float(os.environ.get("CE_LR", "1e-5"))
PRED = os.environ.get("ST_PRED", "work/ce2/ens_test_pred.parquet")        # ensemble predictions used for pseudo-labels
SCORED_OUT = os.environ.get("ST_SCORED_OUT", "work/ce2/test_scored_fr.parquet")
MIX = os.environ.get("ST_MIX", "work/ce2/selftrain_fr.parquet")
torch.manual_seed(0)


def data():
    pred = pl.read_parquet(PRED).with_columns(((pl.col("pa") + pl.col("pv")) / 2).alias("p"))
    cands = pl.read_parquet("work/ce2/test.parquet")  # final candidates with texts
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    d = cands.join(s1c, on="s1_id").filter(pl.col("country").is_in(COUNTRIES)).join(pred.select("o_id", "s1_id", "p"), on=["o_id", "s1_id"])
    d = d.with_columns(pl.col("p").rank("ordinal", descending=True).over("o_id").alias("r"),
                       pl.col("p").max().over("o_id").alias("pmax"))
    pos = d.filter((pl.col("r") == 1) & (pl.col("p") >= POS_P))
    neg = d.filter(((pl.col("r") > 1) & (pl.col("pmax") >= POS_P)) | (pl.col("p") <= NEG_P))
    pos = pos.sample(min(NPOS, pos.height), seed=0).with_columns(pl.lit(1.0).alias("y"))
    neg = neg.sample(min(NNEG, neg.height), seed=0).with_columns(pl.lit(0.0).alias("y"))
    replay = pl.read_parquet("work/ce2/train.parquet").sample(NREPLAY, seed=2).with_columns(pl.col("y").cast(pl.Float64))
    cols = ["text_s1", "text_o", "y"]
    mix = pl.concat([pos.select(cols), neg.select(cols), replay.select(cols)]).sample(fraction=1.0, shuffle=True, seed=3)
    mix.write_parquet(MIX)
    print(f"{COUNTRY}: candidates {d.height}, pseudo-pos {pos.height}, pseudo-neg {neg.height}, replay {replay.height}", flush=True)


def train():
    d = pl.read_parquet(MIX)
    a, b, y = d["text_s1"].to_list(), d["text_o"].to_list(), d["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lens = np.array([len(x) + len(z) for x, z in zip(a, b)])
    order = np.argsort(lens + np.random.default_rng(0).uniform(0, 40, len(lens)))
    chunks = [order[i:i + BS] for i in range(0, len(order), BS)]
    np.random.default_rng(1).shuffle(chunks)
    steps = len(chunks)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / steps))
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
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved", OUT, flush=True)


@torch.no_grad()
def score():
    base = pl.read_parquet(BASE_SCORED)
    cands = pl.read_parquet("work/ce2/test.parquet")
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    fr = cands.join(s1c, on="s1_id").filter(pl.col("country").is_in(COUNTRIES))
    if BAND:
        lo, hi = (float(x) for x in BAND.split(","))
        pr = pl.read_parquet(PRED).with_columns(((pl.col("pa") + pl.col("pv")) / 2).alias("p"))
        fr = fr.join(pr.filter((pl.col("p") >= lo) & (pl.col("p") <= hi)).select("o_id", "s1_id"), on=["o_id", "s1_id"], how="semi")
    a, b = fr["text_s1"].to_list(), fr["text_o"].to_list()
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
    new = fr.select("o_id", "s1_id").with_columns(pl.Series("ce2_new", out))
    res = base.join(new, on=["o_id", "s1_id"], how="left").with_columns(pl.coalesce("ce2_new", "ce2").alias("ce2")).drop("ce2_new")
    res.write_parquet(SCORED_OUT)
    print(f"done: re-scored {len(a)} {COUNTRY} pairs", flush=True)


if __name__ == "__main__":
    {"data": data, "train": train, "score": score}[sys.argv[1]]()
