"""Stage 1b: dense retrieval with a multilingual sentence encoder (GPU).

Encodes "name, address" for every record with intfloat/multilingual-e5-small (MIT license, 118M params),
then exact top-K cosine search S2/S3 -> S1 within each country on the GPU.
Writes work/cand/{split}_emb.parquet (o_id, s1_id, escore, erank).
"""
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

DATA = os.environ.get("DATA_DIR", "student_resource/dataset")
OUT = os.environ.get("CAND_DIR", "work/cand")
EMB = os.environ.get("EMB_DIR", "work/emb")
MODEL = os.environ.get("EMB_MODEL", "intfloat/multilingual-e5-small")
K = int(os.environ.get("TOPK_EMB", "20"))


def encode(model, texts, path):
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    t = time.time()
    # sort by length for fast batching, then restore order
    order = np.argsort([len(x) for x in texts])
    emb = model.encode([texts[i] for i in order], batch_size=1024, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
    out = np.empty_like(emb)
    out[order] = emb
    np.save(path, out)
    print(f"  encoded {len(texts)} in {time.time()-t:.0f}s", flush=True)
    return out


def load(split, s):
    df = pl.read_csv(f"{DATA}/{split}/{split}_{s}.tsv", separator="\t", quote_char=None, infer_schema_length=0) \
        .with_columns(pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""))
    txt = ("query: " + df["business_name"] + ", " + df["business_address"]).to_list()
    return df.select("entity_id", "country"), txt


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(EMB, exist_ok=True)
    model = SentenceTransformer(MODEL, device="cuda")
    model.half()
    print("model", MODEL, flush=True)
    model.max_seq_length = 96
    ids, embs = {}, {}
    for s in ("source1", "source2", "source3"):
        df, txt = load(split, s)
        ids[s] = df
        embs[s] = encode(model, txt, f"{EMB}/{split}_{s}.npy")
    o_ids = pl.concat([ids["source2"], ids["source3"]])
    o_emb = np.concatenate([embs["source2"], embs["source3"]])
    res = []
    for c in sorted(set(ids["source1"]["country"].unique().to_list())):
        si = np.where((ids["source1"]["country"] == c).to_numpy())[0]
        oi = np.where((o_ids["country"] == c).to_numpy())[0]
        if len(si) == 0 or len(oi) == 0:
            continue
        S = torch.from_numpy(np.ascontiguousarray(embs["source1"][si])).cuda()
        t = time.time()
        for i in range(0, len(oi), 4096):
            Q = torch.from_numpy(np.ascontiguousarray(o_emb[oi[i:i + 4096]])).cuda()
            sc, ix = torch.topk(Q @ S.T, K, dim=1)
            n = sc.shape[0]
            res.append(pl.DataFrame({
                "orow": np.repeat(oi[i:i + n], K).astype(np.int64),
                "srow": si[ix.cpu().numpy().ravel()].astype(np.int64),
                "escore": sc.float().cpu().numpy().ravel(),
                "erank": np.tile(np.arange(K, dtype=np.int16), n),
            }))
        print(f"  [{c}] knn {len(oi)} x {len(si)} in {time.time()-t:.0f}s", flush=True)
        del S
    r = pl.concat(res)
    r = r.with_columns(o_ids["entity_id"].gather(r["orow"]).alias("o_id"),
                       ids["source1"]["entity_id"].gather(r["srow"]).alias("s1_id"))
    r.select("o_id", "s1_id", "escore", "erank").write_parquet(f"{OUT}/{split}_{os.environ.get('EMB_TAG', 'emb')}.parquet")
    print("done", r.height, flush=True)


if __name__ == "__main__":
    main()
