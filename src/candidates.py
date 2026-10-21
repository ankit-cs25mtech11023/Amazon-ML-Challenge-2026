"""Stage 1: candidate generation via IDF-weighted sparse token retrieval.

For every Source 2/3 record, retrieve the top-K Source 1 entities of the same country by cosine
similarity over a bag of prefixed tokens (address tokens, name tokens, name sound-keys, no-space name).
Writes work/cand/{split}_tok.parquet with columns: o_id, s1_id, score, rank.
"""
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

NORM = os.environ.get("NORM_DIR", "work/norm")
OUT = os.environ.get("CAND_DIR", "work/cand")
K = int(os.environ.get("TOPK", "30"))
DF_CAP = float(os.environ.get("DF_CAP", "0.01"))  # drop tokens present in more than this fraction of S1
THREADS = int(os.environ.get("THREADS", "64"))
FIELD_W = {"a": 1.0, "n": 1.0, "s": 0.5, "z": 1.0}


def token_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Long frame (row, tok) of prefixed, de-duplicated tokens."""
    parts = [
        df.select(pl.col("row"), pl.col("addr_tok").list.eval(pl.lit("a:") + pl.element()).alias("tok")),
        df.select(pl.col("row"), pl.col("name_tok").list.eval(pl.lit("n:") + pl.element()).alias("tok")),
        df.select(pl.col("row"), pl.col("name_sdx").list.eval(pl.lit("s:") + pl.element()).alias("tok")),
        df.select(pl.col("row"), pl.concat_list(pl.lit("z:") + pl.col("name_ns")).alias("tok")),
    ]
    return pl.concat(parts).explode("tok").drop_nulls().filter(pl.col("tok").str.len_chars() > 2).unique()


def build_matrix(tf: pl.DataFrame, vocab: pl.DataFrame, nrows: int) -> sp.csr_matrix:
    j = tf.join(vocab, on="tok", how="inner")
    m = sp.csr_matrix(
        (j["w"].to_numpy().astype(np.float32), (j["row"].to_numpy(), j["col"].to_numpy())),
        shape=(nrows, vocab.height),
    )
    # L2 normalize rows
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms).dot(m).tocsr()


def run_country(s1: pl.DataFrame, o: pl.DataFrame, country: str) -> pl.DataFrame:
    t = time.time()
    s1 = s1.with_row_index("row")
    o = o.with_row_index("row")
    tf1 = token_frame(s1)
    tfo = token_frame(o)
    n = s1.height
    df = tf1.group_by("tok").len().rename({"len": "df"})
    vocab = (
        df.filter(pl.col("df") <= max(50, int(DF_CAP * n)))
        .with_columns(
            (np.log(n / pl.col("df")) + 1.0).alias("idf"),
            pl.col("tok").str.slice(0, 1).replace_strict(FIELD_W, default=1.0).alias("fw"),
        )
        .with_columns((pl.col("idf") * pl.col("fw")).alias("w"))
        .with_row_index("col")
        .select("tok", pl.col("col").cast(pl.Int64), "w")
    )
    A = build_matrix(tfo, vocab, o.height)
    B = build_matrix(tf1, vocab, n)
    print(f"  [{country}] S1={n} O={o.height} vocab={vocab.height} nnzA={A.nnz} nnzB={B.nnz} prep {time.time()-t:.0f}s", flush=True)
    BT = B.T.tocsr()
    res = []
    step = 500_000
    for i in range(0, A.shape[0], step):
        C = sp_matmul_topn(A[i:i + step], BT, top_n=K, threshold=0.05, sort=True, n_threads=THREADS)
        C = C.tocoo()
        res.append(pl.DataFrame({"orow": (C.row + i).astype(np.int64), "srow": C.col.astype(np.int64), "score": C.data.astype(np.float32)}))
        print(f"    {i+step:>9}/{A.shape[0]} {time.time()-t:.0f}s", flush=True)
    r = pl.concat(res)
    r = r.join(o.select(pl.col("row").cast(pl.Int64).alias("orow"), pl.col("entity_id").alias("o_id")), on="orow") \
         .join(s1.select(pl.col("row").cast(pl.Int64).alias("srow"), pl.col("entity_id").alias("s1_id")), on="srow")
    r = r.sort(["o_id", "score"], descending=[False, True]).with_columns(
        pl.int_range(pl.len()).over("o_id").cast(pl.Int16).alias("rank"))
    return r.select("o_id", "s1_id", "score", "rank")


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    os.makedirs(OUT, exist_ok=True)
    cols = ["entity_id", "country", "addr_tok", "name_tok", "name_sdx", "name_ns"]
    s1 = pl.read_parquet(f"{NORM}/{split}_source1.parquet", columns=cols)
    o = pl.concat([pl.read_parquet(f"{NORM}/{split}_source{k}.parquet", columns=cols) for k in (2, 3)])
    out = []
    for c in sorted(set(s1["country"].unique().to_list()) | set(o["country"].unique().to_list())):
        a, b = s1.filter(pl.col("country") == c), o.filter(pl.col("country") == c)
        if a.height == 0 or b.height == 0:
            continue
        out.append(run_country(a, b, c))
    res = pl.concat(out)
    res.write_parquet(f"{OUT}/{split}_tok.parquet")
    print("done", res.shape, flush=True)


if __name__ == "__main__":
    main()
