"""Cross-encoder re-scorer for uncertain pairs.

  python src/ce_train.py train          # fine-tune on fold-0 uncertain train pairs -> work/ce/model
  python src/ce_train.py score train    # score fold-1 train pairs  -> work/ce/train_ce.parquet
  python src/ce_train.py score test     # score test pairs          -> work/ce/test_ce.parquet
Model: BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M params), input = (S1 text, S2/S3 text), output = match logit.
"""
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = os.environ.get("CE_BASE", "BAAI/bge-reranker-v2-m3")
OUT = os.environ.get("CE_DIR", "work/ce/model")
MAXLEN = int(os.environ.get("CE_MAXLEN", "128"))
BS = int(os.environ.get("CE_BS", "64"))
LR = float(os.environ.get("CE_LR", "2e-5"))
NMAX = int(os.environ.get("CE_NMAX", "1000000"))
FOLD = int(os.environ.get("CE_FOLD", "0"))          # train on this fold, score the other
TAG = os.environ.get("CE_TAG", "")                    # output suffix, e.g. "_b"
BAND = os.environ.get("CE_BAND", "band")              # which band file to score
torch.manual_seed(0)


def batches(a, b, bs, order):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        yield idx, [a[j] for j in idx], [b[j] for j in idx]


def train():
    d = pl.read_parquet("work/ce/train_band.parquet").filter(pl.col("fold") == FOLD).sample(fraction=1.0, shuffle=True, seed=0).head(NMAX)
    a, b, y = d["text_s1"].to_list(), d["text_o"].to_list(), d["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # length-bucketed shuffled batches: fast padding, still random across the epoch
    lens = np.array([len(x) + len(z) for x, z in zip(a, b)])
    order = np.argsort(lens + np.random.default_rng(0).uniform(0, 40, len(lens)))
    chunks = [order[i:i + BS] for i in range(0, len(order), BS)]
    np.random.default_rng(1).shuffle(chunks)
    steps = len(chunks)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.0, 1 - s / steps))
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
        if s % 500 == 0:
            print(f"step {s}/{steps} loss {loss.item():.4f} {time.time()-t:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print("saved", OUT, flush=True)


@torch.no_grad()
def score(split):
    d = pl.read_parquet(f"work/ce/{split}_{BAND}.parquet")
    if split == "train":
        d = d.filter(pl.col("fold") == 1 - FOLD)
    a, b = d["text_s1"].to_list(), d["text_o"].to_list()
    tok = AutoTokenizer.from_pretrained(OUT)
    model = AutoModelForSequenceClassification.from_pretrained(OUT, torch_dtype=torch.bfloat16).cuda().eval()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    t = time.time()
    for n, (idx, xa, xb) in enumerate(batches(a, b, 512, order)):
        enc = tok(xa, xb, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if n % 200 == 0:
            print(f"scored {min((n + 1) * 512, len(a))}/{len(a)} {time.time()-t:.0f}s", flush=True)
    d.select([c for c in d.columns if not c.startswith("text_")]).with_columns(pl.Series("ce", out)) \
     .write_parquet(f"work/ce/{split}_ce{TAG}.parquet")
    print("done", split, len(a), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        score(sys.argv[2])
