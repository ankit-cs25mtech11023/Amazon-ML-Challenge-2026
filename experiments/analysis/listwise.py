"""Listwise LLM matcher ("select" framing): one S2/S3 record + its top-4 candidate entities -> A/B/C/D or E (none).

  python src/listwise.py data     # train / holdout / test prompts (top-4 candidates, shuffled order)
  python src/listwise.py train    # LoRA fine-tune Qwen2.5-7B-Instruct (causal LM, loss on the 5 option letters)
  python src/listwise.py score holdout|test   # per-candidate choice probability + none probability
Train examples: non-holdout entities, weighted to hard cases. Holdout: cross-encoder-v2 holdout entities.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
BASE = os.environ.get("LLM_BASE", "Qwen/Qwen2.5-7B-Instruct")
OUT = os.environ.get("LW_DIR", "work/lw/lora")
M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
NTRAIN = int(os.environ.get("LW_NTRAIN", "80000"))
TEST_PRED = os.environ.get("LW_TEST_PRED", "work/ce2/ens_test_pred_v14.parquet")
BS, LR, MAXLEN = int(os.environ.get("LW_BS", "8")), float(os.environ.get("LW_LR", "1e-4")), int(os.environ.get("LW_MAXLEN", "384"))
LETTERS = ["A", "B", "C", "D", "E"]
os.makedirs("work/lw", exist_ok=True)
torch.manual_seed(0)


def texts(split, s):
    df = pl.read_csv(f"{DATA}/{split}/{split}_{s}.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    return df.select("entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")
                                    + " | " + pl.col("country").fill_null("")).alias("t"))


def build(split, cand, pairs=None, seed=0):
    """cand: (o_id, s1_id, p). Keep top-4 per record, shuffle, attach texts and the answer letter."""
    top = cand.with_columns(pl.col("p").rank("ordinal", descending=True).over("o_id").alias("r")).filter(pl.col("r") <= 4)
    rng = np.random.default_rng(seed)
    top = top.with_columns(pl.Series("sh", rng.random(top.height))).sort(["o_id", "sh"]) \
             .with_columns(pl.int_range(pl.len()).over("o_id").alias("slot"))
    t1 = texts(split, "source1").rename({"entity_id": "s1_id", "t": "ts"})
    to = pl.concat([texts(split, "source2"), texts(split, "source3")]).rename({"entity_id": "o_id", "t": "to"})
    top = top.join(t1, on="s1_id")
    if pairs is not None:
        top = top.join(pairs.with_columns(pl.lit(1).alias("y")), on=["o_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
    g = top.sort(["o_id", "slot"]).group_by("o_id", maintain_order=True).agg(pl.col("s1_id"), pl.col("ts"), pl.col("p"),
                                                                            *( [pl.col("y")] if pairs is not None else []))
    g = g.join(to, on="o_id")
    if pairs is not None:
        g = g.with_columns(pl.col("y").list.arg_max().alias("ans"), pl.col("y").list.sum().alias("has"))
        g = g.with_columns(pl.when(pl.col("has") > 0).then(pl.col("ans")).otherwise(4).alias("ans")).drop("has")
    return g


def prompt(o, cands):
    lines = "\n".join(f"{LETTERS[i]}: {c}" for i, c in enumerate(cands))
    return ("Which candidate business is the same real-world business as the record? Businesses with similar names at "
            "different street numbers are usually different businesses.\n"
            f"Record: {o}\nCandidates:\n{lines}\nE: none of the above\nAnswer:")


def data():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    oof = pl.read_parquet(f"{M1}/oof.parquet")
    hold_o = pl.read_parquet("work/ce2/holdout.parquet").select("o_id").unique()
    # hard cases: close top-2, or best candidate below 0.98, plus some easy ones
    st = oof.with_columns(pl.col("p").rank("ordinal", descending=True).over("o_id").alias("r"))
    agg = st.group_by("o_id").agg(pl.col("p").filter(pl.col("r") == 1).first().alias("p1"),
                                  pl.col("p").filter(pl.col("r") == 2).first().fill_null(0).alias("p2"))
    tr_o = agg.join(hold_o, on="o_id", how="anti")
    hard = tr_o.filter((pl.col("p1") < 0.98) | (pl.col("p1") - pl.col("p2") < 0.3)).filter(pl.col("p1") > 0.02)
    easy = tr_o.join(hard.select("o_id"), on="o_id", how="anti")
    pick = pl.concat([hard.sample(min(int(NTRAIN * 0.8), hard.height), seed=0).select("o_id"),
                      easy.sample(int(NTRAIN * 0.2), seed=1).select("o_id")])
    g = build("train", oof.join(pick, on="o_id", how="semi"), pairs, seed=0)
    g.write_parquet("work/lw/train.parquet")
    print(f"train examples {g.height}, answer dist {g['ans'].value_counts().sort('ans').to_dicts()}", flush=True)
    # holdout: records of cross-encoder-v2 holdout entities that are uncertain
    h = agg.join(hold_o, on="o_id", how="semi").filter((pl.col("p1") < 0.98) | (pl.col("p1") - pl.col("p2") < 0.3)).filter(pl.col("p1") > 0.02)
    gh = build("train", oof.join(h.select("o_id"), on="o_id", how="semi"), pairs, seed=5)
    gh.write_parquet("work/lw/holdout.parquet")
    # test: uncertain records under the current ensemble
    pr = pl.read_parquet(TEST_PRED).select("o_id", "s1_id", ((pl.col("pa") + pl.col("pv")) / 2).alias("p"))
    cand = pl.read_parquet("work/ce2/test.parquet", columns=["o_id", "s1_id"])
    pr = pr.join(cand, on=["o_id", "s1_id"], how="semi")
    s = pr.with_columns(pl.col("p").rank("ordinal", descending=True).over("o_id").alias("r"))
    sa = s.group_by("o_id").agg(pl.col("p").filter(pl.col("r") == 1).first().alias("p1"),
                                pl.col("p").filter(pl.col("r") == 2).first().fill_null(0).alias("p2"))
    unc = sa.filter((pl.col("p1") > 0.05) & ((pl.col("p1") < 0.97) | (pl.col("p1") - pl.col("p2") < 0.2)))
    gt = build("test", pr.join(unc.select("o_id"), on="o_id", how="semi"), None, seed=9)
    gt.write_parquet("work/lw/test.parquet")
    print(f"holdout records {gh.height}, test records {gt.height}", flush=True)


def load(adapter=None):
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
    return tok, model, torch.tensor(ids)


def option_logits(model, enc, ids):
    h = model(**enc, output_hidden_states=False).logits[:, -1, :]
    return h.float()[:, ids]


def rows(g):
    ps, ans, n = [], [], []
    for o, ts in zip(g["to"].to_list(), g["ts"].to_list()):
        ps.append(prompt(o, ts)); n.append(len(ts))
    return ps, n


def train():
    g = pl.read_parquet("work/lw/train.parquet").sample(fraction=1.0, shuffle=True, seed=2).head(int(os.environ.get("LW_TRAIN_N", "40000")))
    ps, n = rows(g)
    y = g["ans"].to_numpy().astype(np.int64)
    tok, model, ids = load()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.cuda()
    ids = ids.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    steps = (len(ps) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / steps))
    model.train(); t = time.time()
    for s in range(steps):
        idx = slice(s * BS, (s + 1) * BS)
        enc = tok(ps[idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        lg = option_logits(model, enc, ids)
        mask = torch.full_like(lg, 0.0)
        for i, k in enumerate(n[idx]):          # unavailable slots (fewer than 4 candidates) can't be chosen
            mask[i, k:4] = -1e4
        loss = torch.nn.functional.cross_entropy(lg + mask, torch.from_numpy(y[idx]).cuda())
        loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if s % 200 == 0:
            el = time.time() - t
            print(f"step {s}/{steps} loss {loss.item():.4f} {el:.0f}s eta {el / max(s, 1) * (steps - s) / 60:.0f}min", flush=True)
        if s and s % 2000 == 0:
            model.save_pretrained(OUT)
    model.save_pretrained(OUT); print("saved", OUT, flush=True)


@torch.no_grad()
def score(which):
    g = pl.read_parquet(f"work/lw/{which}.parquet")
    ps, n = rows(g)
    tok, model, ids = load()
    model = PeftModel.from_pretrained(model, OUT).cuda().eval(); ids = ids.cuda()
    order = np.argsort([len(x) for x in ps])
    probs = np.zeros((len(ps), 5), dtype=np.float32)
    B = int(os.environ.get("LW_SCORE_BS", "32")); t = time.time()
    for k, i in enumerate(range(0, len(order), B)):
        idx = order[i:i + B]
        enc = tok([ps[j] for j in idx], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
        lg = option_logits(model, enc, ids)
        for r, j in enumerate(idx):
            lg[r, n[j]:4] = -1e4
        probs[idx] = torch.softmax(lg, -1).cpu().numpy()
        if k % 200 == 0:
            el = time.time() - t
            print(f"scored {min(i + B, len(ps))}/{len(ps)} {el:.0f}s eta {el / max(i + B, 1) * (len(ps) - i) / 60:.0f}min", flush=True)
    g = g.with_columns(pl.Series("pl", [probs[j, :n[j]].tolist() for j in range(len(ps))]), pl.Series("pnone", probs[:, 4]))
    out = g.select("o_id", "s1_id", "pl", "pnone").explode(["s1_id", "pl"]).rename({"pl": "p_list"})
    out.write_parquet(f"work/lw/{which}_scored.parquet")
    print("done", which, len(ps), flush=True)


if __name__ == "__main__":
    {"data": data, "train": train}.get(sys.argv[1], lambda: score(sys.argv[2]))()
