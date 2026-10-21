"""Name-only candidate retrieval for S2/S3 records with an EMPTY address (address-driven retrievers miss them).

Char 3-gram TF-IDF on the cleaned business name (legal forms removed), cosine top-K Source 1 entities per record,
within country. Writes work/cand/{split}_name.parquet (o_id, s1_id, nscore, nrank).
"""
import os
import sys

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

NORM = os.environ.get("NORM_DIR", "work/norm")
K = int(os.environ.get("NAME_K", "10"))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    cols = ["entity_id", "country", "name_core", "name_ns", "addr_empty"]
    s1 = pl.read_parquet(f"{NORM}/{split}_source1.parquet", columns=cols)
    o = pl.concat([pl.read_parquet(f"{NORM}/{split}_source{k}.parquet", columns=cols) for k in (2, 3)]).filter(pl.col("addr_empty"))
    out = []
    for c in sorted(set(o["country"].unique().to_list())):
        a, b = s1.filter(pl.col("country") == c), o.filter(pl.col("country") == c)
        if a.height == 0 or b.height == 0:
            continue
        txt = lambda d: (d["name_core"].fill_null("") + " " + d["name_ns"].fill_null("")).to_list()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
        A = vec.fit_transform(txt(a))
        B = vec.transform(txt(b))
        C = sp_matmul_topn(B, A.T.tocsr(), top_n=K, threshold=0.2, sort=True, n_threads=48).tocoo()
        r = pl.DataFrame({"o_id": b["entity_id"].gather(C.row), "s1_id": a["entity_id"].gather(C.col),
                          "nscore": C.data.astype(np.float32)})
        out.append(r)
        print(f"  [{c}] empty-address records {b.height} -> pairs {r.height}", flush=True)
    r = pl.concat(out).with_columns(pl.col("nscore").rank("ordinal", descending=True).over("o_id").cast(pl.Int16).alias("nrank"))
    r.write_parquet(f"work/cand/{split}_name.parquet")
    if split == "train":
        p = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"}) \
              .join(o.select(pl.col("entity_id").alias("o_id")), on="o_id", how="semi")
        u = pl.read_parquet("work/cand/train_union4.parquet", columns=["o_id", "s1_id"])
        h = p.join(u.with_columns(pl.lit(1).alias("old")), on=["o_id", "s1_id"], how="left") \
             .join(r.select("o_id", "s1_id", "nrank"), on=["o_id", "s1_id"], how="left")
        T = h.height
        print(f"empty-address true pairs {T}: current recall {h['old'].is_not_null().mean():.4f}", flush=True)
        for k in (1, 3, 5, 10):
            new = (h["nrank"] < k).fill_null(False)
            print(f"  + name top-{k:<2d}: recall {(h['old'].is_not_null() | new).mean():.4f} "
                  f"(+{(new & h['old'].is_null()).sum()} pairs) extra cands {(r['nrank'] < k).sum()}", flush=True)


if __name__ == "__main__":
    main()
