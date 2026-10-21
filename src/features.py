"""Stage 2: pairwise + context features for candidate pairs.

Input : work/cand/{split}_tok.parquet (o_id, s1_id, score, rank), work/norm/*.parquet
Output: work/feat/{split}/part_XX.parquet
"""
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

NORM = os.environ.get("NORM_DIR", "work/norm")
CAND = os.environ.get("CAND_DIR", "work/cand")
OUT = os.environ.get("FEAT_DIR", "work/feat")
KMAX = int(os.environ.get("KMAX", "10"))
NPARTS = int(os.environ.get("NPARTS", "8"))

COLS = ["entity_id", "name_core", "name_ns", "name_tok", "name_sdx", "legal_form", "addr_n", "addr_tok",
        "addr_num", "postal", "addr_empty", "name_is_url", "name_is_handle", "name_nonlatin", "name_ntok", "addr_ntok"]


def pdist(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def list_overlap(df, a, b, name):
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    la, lb = pl.col(a).list.len(), pl.col(b).list.len()
    return [
        inter.alias(f"{name}_inter"),
        (inter / (la + lb - inter).clip(lower_bound=1)).alias(f"{name}_jac"),
        (inter / la.clip(lower_bound=1)).alias(f"{name}_cont_o"),
        (inter / lb.clip(lower_bound=1)).alias(f"{name}_cont_s"),
    ]


def context(c: pl.DataFrame) -> pl.DataFrame:
    """Features describing a candidate relative to its competitors (both directions)."""
    c = c.with_columns(
        pl.col("score").max().over("o_id").alias("o_top"),
        pl.col("score").len().over("o_id").alias("o_ncand"),
        pl.col("score").sort(descending=True).slice(1, 1).first().over("o_id").fill_null(0).alias("o_second"),
        pl.col("score").rank("ordinal", descending=True).over("s1_id").alias("s_rank"),
        pl.col("score").len().over("s1_id").alias("s_ncand"),
        (pl.col("rank") == 0).sum().over("s1_id").alias("s_n_top1"),
        pl.col("score").max().over("s1_id").alias("s_top"),
    )
    c = c.with_columns(
        pl.col("escore").max().over("o_id").alias("e_top"),
        pl.col("escore").sort(descending=True).slice(1, 1).first().over("o_id").fill_null(0).alias("e_second"),
        pl.col("escore").rank("ordinal", descending=True).over("o_id").alias("e_rank_o"),
        pl.col("escore").rank("ordinal", descending=True).over("s1_id").alias("e_rank_s"),
        pl.col("escore").max().over("s1_id").alias("e_s_top"),
    )
    return c.with_columns(
        (pl.col("escore") - pl.col("e_top")).alias("egap_top"),
        (pl.col("escore") - pl.col("e_second")).alias("egap_second"),
        (pl.col("escore") - pl.col("e_s_top")).alias("egap_s_top"),
        (pl.col("score") - pl.col("o_top")).alias("gap_top"),
        (pl.col("score") - pl.col("o_second")).alias("gap_second"),
        (pl.col("score") - pl.col("s_top")).alias("gap_s_top"),
    )


def build_part(c: pl.DataFrame, s1: pl.DataFrame, o: pl.DataFrame, name_freq: pl.DataFrame) -> pl.DataFrame:
    d = c.join(o, left_on="o_id", right_on="entity_id").join(s1, left_on="s1_id", right_on="entity_id", suffix="_s")
    d = d.join(name_freq, left_on="name_core_s", right_on="name_core", how="left")
    f = {}
    no, ns = d["name_core"].to_list(), d["name_core_s"].to_list()
    f["n_ratio"] = pdist(no, ns, fuzz.ratio)
    f["n_tsort"] = pdist(no, ns, fuzz.token_sort_ratio)
    f["n_tset"] = pdist(no, ns, fuzz.token_set_ratio)
    f["n_partial"] = pdist(no, ns, fuzz.partial_ratio)
    f["n_jw"] = pdist(no, ns, distance.JaroWinkler.normalized_similarity)
    nso, nss = d["name_ns"].to_list(), d["name_ns_s"].to_list()
    f["ns_ratio"] = pdist(nso, nss, fuzz.ratio)
    f["ns_partial"] = pdist(nso, nss, fuzz.partial_ratio)
    ao, as_ = d["addr_n"].to_list(), d["addr_n_s"].to_list()
    f["a_ratio"] = pdist(ao, as_, fuzz.ratio)
    f["a_tsort"] = pdist(ao, as_, fuzz.token_sort_ratio)
    f["a_tset"] = pdist(ao, as_, fuzz.token_set_ratio)
    f["a_partial_tset"] = pdist(ao, as_, fuzz.partial_token_set_ratio)
    d = d.with_columns([pl.Series(k, v) for k, v in f.items()])
    d = d.with_columns(
        *list_overlap(d, "name_tok", "name_tok_s", "ntok"),
        *list_overlap(d, "name_sdx", "name_sdx_s", "nsdx"),
        *list_overlap(d, "addr_tok", "addr_tok_s", "atok"),
        *list_overlap(d, "addr_num", "addr_num_s", "anum"),
        (pl.col("postal").list.set_intersection(pl.col("postal_s")).list.len() > 0).alias("postal_eq"),
        ((pl.col("postal").list.len() > 0) & (pl.col("postal_s").list.len() > 0)
         & (pl.col("postal").list.set_intersection(pl.col("postal_s")).list.len() == 0)).alias("postal_conflict"),
        ((pl.col("addr_num").list.len() > 0) & (pl.col("addr_num_s").list.len() > 0)
         & (pl.col("addr_num").list.set_intersection(pl.col("addr_num_s")).list.len() == 0)).alias("num_conflict"),
        (pl.col("addr_num").list.first() == pl.col("addr_num_s").list.first()).fill_null(False).alias("num_first_eq"),
        (pl.col("legal_form") == pl.col("legal_form_s")).alias("legal_eq"),
        ((pl.col("legal_form") != "") & (pl.col("legal_form_s") != "") & (pl.col("legal_form") != pl.col("legal_form_s"))).alias("legal_conflict"),
        (pl.col("legal_form") == "").alias("legal_empty_o"),
        (pl.col("o_id").str.starts_with("S3")).alias("is_s3"),
        pl.col("name_freq").fill_null(1),
        (pl.col("name_ntok") - pl.col("name_ntok_s")).alias("name_ntok_diff"),
        (pl.col("addr_ntok") / pl.col("addr_ntok_s").clip(lower_bound=1)).alias("addr_len_ratio"),
    )
    drop = [c for c in d.columns if c in COLS or (c.endswith("_s") and c[:-2] in COLS)]
    return d.drop([c for c in drop if c not in ("addr_empty", "name_is_url", "name_is_handle", "name_nonlatin",
                                                   "name_ntok", "addr_ntok", "name_ntok_s", "addr_ntok_s")])


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    os.makedirs(f"{OUT}/{split}", exist_ok=True)
    t = time.time()
    cand = pl.read_parquet(f"{CAND}/{split}_{os.environ.get('CAND_KIND', 'union')}.parquet")
    cand = context(cand)
    s1 = pl.read_parquet(f"{NORM}/{split}_source1.parquet", columns=COLS)
    o = pl.concat([pl.read_parquet(f"{NORM}/{split}_source{k}.parquet", columns=COLS) for k in (2, 3)])
    name_freq = s1.group_by("name_core").len().rename({"len": "name_freq"})
    print(f"loaded {cand.height} cand rows {time.time()-t:.0f}s", flush=True)
    cand = cand.with_columns((pl.col("o_id").hash(7) % NPARTS).alias("part"))
    for p in range(NPARTS):
        c = cand.filter(pl.col("part") == p).drop("part")
        out = build_part(c, s1, o, name_freq)
        out.write_parquet(f"{OUT}/{split}/part_{p:02d}.parquet")
        print(f"part {p} {out.shape} {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
