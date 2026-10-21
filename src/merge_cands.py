"""Stage 1c: union of token candidates (top KT) and embedding candidates (top KE).

Every pair gets both a token score (null if not retrieved by tokens) and an exact embedding cosine
(recomputed on the GPU for all pairs). Writes work/cand/{split}_union.parquet.
"""
import os
import sys

import numpy as np
import polars as pl
import torch

CAND = os.environ.get("CAND_DIR", "work/cand")
EMB = os.environ.get("EMB_DIR", "work/emb")
NORM = os.environ.get("NORM_DIR", "work/norm")
KT = int(os.environ.get("KT", "10"))
KE = int(os.environ.get("KE", "10"))
ETAG = os.environ.get("ETAG", "emb")            # which embedding candidate file to union in
EMB2 = os.environ.get("EMB2_DIR", "")           # optional second embedding space -> extra cosine feature
OUTTAG = os.environ.get("OUTTAG", "union")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t = pl.read_parquet(f"{CAND}/{split}_tok.parquet").filter(pl.col("rank") < KT)
    e = pl.read_parquet(f"{CAND}/{split}_{ETAG}.parquet").filter(pl.col("erank") < KE).drop("escore")
    u = t.join(e, on=["o_id", "s1_id"], how="full", coalesce=True)
    u = u.with_columns(pl.col("rank").fill_null(99).cast(pl.Int16), pl.col("erank").fill_null(99).cast(pl.Int16),
                       pl.col("score").fill_null(0.0))
    # exact cosine for every pair
    ids1 = pl.read_parquet(f"{NORM}/{split}_source1.parquet", columns=["entity_id"]).with_row_index("srow")
    ido = pl.concat([pl.read_parquet(f"{NORM}/{split}_source{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("orow")
    u = u.join(ids1.rename({"entity_id": "s1_id"}), on="s1_id").join(ido.rename({"entity_id": "o_id"}), on="o_id")
    sr, orr = u["srow"].to_numpy().astype(np.int64), u["orow"].to_numpy().astype(np.int64)

    def cosine(d):
        E1 = torch.from_numpy(np.load(f"{d}/{split}_source1.npy")).cuda()
        EO = torch.from_numpy(np.concatenate([np.load(f"{d}/{split}_source2.npy"), np.load(f"{d}/{split}_source3.npy")])).cuda()
        out = np.empty(len(sr), dtype=np.float32)
        step = 2_000_000
        for i in range(0, len(sr), step):
            a = torch.from_numpy(sr[i:i + step]).cuda(); b = torch.from_numpy(orr[i:i + step]).cuda()
            out[i:i + step] = (E1[a].float() * EO[b].float()).sum(1).cpu().numpy()
        del E1, EO
        torch.cuda.empty_cache()
        return out

    u = u.with_columns(pl.Series("escore", cosine(EMB)))
    if EMB2:
        u = u.with_columns(pl.Series("escore2", cosine(EMB2)))
    u = u.drop("srow", "orow")
    u.write_parquet(f"{CAND}/{split}_{OUTTAG}.parquet")
    print(split, "union rows", u.height, "tok-only", (u["erank"] == 99).sum(), "emb-only", (u["rank"] == 99).sum(), flush=True)


if __name__ == "__main__":
    main()
