"""LLM matcher: Qwen2.5-7B-Instruct (Apache-2.0, 7.6B params) fine-tuned with LoRA as a pair classifier.

  python src/llm_match.py data            # hard training pairs, holdout slice, uncertain test pairs
  python src/llm_match.py train           # LoRA fine-tune (bf16, gradient checkpointing)
  python src/llm_match.py score holdout|test
Training pairs come only from non-holdout entities (the second-cross-encoder holdout stays clean for blending).
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = os.environ.get("LLM_BASE", "Qwen/Qwen2.5-7B-Instruct")
OUT = os.environ.get("LLM_DIR", "work/llm/lora")
NTRAIN = int(os.environ.get("LLM_NTRAIN", "250000"))
HOLD_FRAC = float(os.environ.get("LLM_HOLD_FRAC", "0.25"))
TEST_PRED = os.environ.get("LLM_TEST_PRED", "work/ce2/ens_test_pred_fr1.parquet")
PLO, PHI = float(os.environ.get("LLM_PLO", "0.03")), float(os.environ.get("LLM_PHI", "0.97"))
BS, LR, MAXLEN = int(os.environ.get("LLM_BS", "16")), float(os.environ.get("LLM_LR", "1e-4")), int(os.environ.get("LLM_MAXLEN", "192"))
os.makedirs("work/llm", exist_ok=True)
torch.manual_seed(0)


def prompt(a, b):
    return (f"Do these two records describe the same real-world business?\n"
            f"Record A: {a}\nRecord B: {b}\nAnswer:")


def data():
    tr = pl.read_parquet("work/ce2/train.parquet")
    mid = tr.filter((pl.col("p1") >= 0.02) & (pl.col("p1") <= 0.98))
    hi = tr.filter(pl.col("p1") > 0.98)
    lo = tr.filter(pl.col("p1") < 0.02)
    n_mid = int(NTRAIN * 0.6)
    sel = pl.concat([mid.sample(min(n_mid, mid.height), seed=0),
                     hi.sample(min(int(NTRAIN * 0.2), hi.height), seed=1),
                     lo.sample(min(int(NTRAIN * 0.2), lo.height), seed=2)]).sample(fraction=1.0, shuffle=True, seed=3)
    sel.select("text_s1", "text_o", "y").write_parquet("work/llm/train.parquet")
    h = pl.read_parquet("work/ce2/holdout.parquet")
    hs = h.filter(pl.col("s1_id").hash(21) % 1000 < HOLD_FRAC * 1000)
    hs.write_parquet("work/llm/holdout.parquet")
    pred = pl.read_parquet(TEST_PRED).with_columns(((pl.col("pa") + pl.col("pv")) / 2).alias("p"))
    unc = pred.filter((pl.col("p") >= PLO) & (pl.col("p") <= PHI)).select("o_id", "s1_id")
    t = pl.read_parquet("work/ce2/test.parquet").join(unc, on=["o_id", "s1_id"], how="semi")
    t.write_parquet("work/llm/test.parquet")
    print(f"train {sel.height} (pos {sel['y'].mean():.3f}) | holdout slice {hs.height} | uncertain test pairs {t.height}", flush=True)


def load_base():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    return tok, model


def train():
    d = pl.read_parquet(os.environ.get("LLM_TRAIN_FILE", "work/llm/train.parquet"))
    a, b, y = d["text_s1"].to_list(), d["text_o"].to_list(), d["y"].to_numpy().astype(np.float32)
    tok, model = load_base()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.cuda()
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
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
        enc = tok([prompt(a[j], b[j]) for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        logit = model(**enc).logits.squeeze(-1).float()
        loss = lossf(logit, torch.from_numpy(y[idx]).cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if s % 200 == 0:
            el = time.time() - t
            print(f"step {s}/{steps} loss {loss.item():.4f} {el:.0f}s eta {el / max(s, 1) * (steps - s) / 60:.0f}min", flush=True)
        if s and s % 3000 == 0:
            model.save_pretrained(OUT)
    model.save_pretrained(OUT)
    print("saved", OUT, flush=True)


@torch.no_grad()
def score(which):
    d = pl.read_parquet(f"work/llm/{which}.parquet")
    a, b = d["text_s1"].to_list(), d["text_o"].to_list()
    tok, model = load_base()
    model = PeftModel.from_pretrained(model, OUT).cuda().eval()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    t = time.time()
    B = int(os.environ.get("LLM_SCORE_BS", "64"))
    for n, i in enumerate(range(0, len(order), B)):
        idx = order[i:i + B]
        enc = tok([prompt(a[j], b[j]) for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if n % 200 == 0:
            el = time.time() - t
            print(f"scored {min(i + B, len(a))}/{len(a)} {el:.0f}s eta {el / max(i + B, 1) * (len(a) - i) / 60:.0f}min", flush=True)
    d.select([c for c in d.columns if not c.startswith("text_")]).with_columns(pl.Series("llm", out)) \
     .write_parquet(f"work/llm/{which}_scored.parquet")
    print("done", which, len(a), flush=True)


def data_wide():
    """Extra uncertain pairs (wider band) not already LLM-scored -> work/llm/test2.parquet."""
    lo, hi = float(os.environ.get("LLM_WIDE_LO", "0.005")), float(os.environ.get("LLM_WIDE_HI", "0.995"))
    pred = pl.read_parquet(os.environ.get("LLM_WIDE_PRED", "work/ce2/ens_test_pred_fr2.parquet")) \
             .with_columns(((pl.col("pa") + pl.col("pv")) / 2).alias("p"))
    band = pred.filter((pl.col("p") >= lo) & (pl.col("p") <= hi)).select("o_id", "s1_id")
    done = pl.read_parquet("work/llm/test.parquet").select("o_id", "s1_id")
    t = pl.read_parquet("work/ce2/test.parquet").join(band, on=["o_id", "s1_id"], how="semi").join(done, on=["o_id", "s1_id"], how="anti")
    t.write_parquet("work/llm/test2.parquet")
    print(f"wide band [{lo}, {hi}]: {t.height} additional pairs to score", flush=True)


if __name__ == "__main__":
    {"data": data, "data_wide": data_wide, "train": train}.get(sys.argv[1], lambda: score(sys.argv[2]))()
