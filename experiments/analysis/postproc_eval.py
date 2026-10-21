"""Evaluate two decision-stage ideas on the holdout slice, on top of the v15 (ensemble + LLM) probabilities.

MFC  (mutual-first-choice tie-break): when a record's top-2 candidates are within DELTA, prefer the one for which
      this record is that entity's highest-probability record.
TCF  (transitive-consistency filter): for entities with >= 3 assigned records, drop an assigned record whose
      embedding cosine to the mean of the entity's OTHER assigned records is below COS and whose p is below PMAX.
Also exposes apply_mfc / apply_tcf for building test outputs.
"""
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import llm_blend as lb  # noqa: E402
from train import macro_f05  # noqa: E402


def apply_mfc(pred: pl.DataFrame, delta: float) -> pl.DataFrame:
    """Return one (o_id, s1_id, p) per record: best candidate, or the mutual-first-choice runner-up on near-ties."""
    d = pred.with_columns(pl.col("p").rank("ordinal", descending=True).over("o_id").alias("r_o"),
                          pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("r_s"))
    top = d.filter(pl.col("r_o") <= 2).sort(["o_id", "r_o"])
    g = top.group_by("o_id", maintain_order=True).agg(pl.col("s1_id"), pl.col("p"), pl.col("r_s"))
    g = g.with_columns(pl.col("p").list.get(0).alias("p1"), pl.col("p").list.get(1, null_on_oob=True).alias("p2"),
                       pl.col("r_s").list.get(0).alias("rs1"), pl.col("r_s").list.get(1, null_on_oob=True).alias("rs2"))
    swap = (pl.col("p2").is_not_null() & (pl.col("p1") - pl.col("p2") < delta) & (pl.col("rs2") == 1) & (pl.col("rs1") > 1))
    return g.select("o_id",
                    pl.when(swap).then(pl.col("s1_id").list.get(1)).otherwise(pl.col("s1_id").list.get(0)).alias("s1_id"),
                    pl.when(swap).then(pl.col("p2")).otherwise(pl.col("p1")).alias("p"))


def apply_tcf(best: pl.DataFrame, tau: float, emb_o: np.ndarray, row_of: dict, cos_min: float, pmax: float) -> pl.DataFrame:
    """Drop weakly-consistent members of large clusters (among pairs that pass tau)."""
    acc = best.filter(pl.col("p") >= tau).with_columns(pl.len().over("s1_id").alias("n"))
    big = acc.filter(pl.col("n") >= 3)
    if big.height == 0:
        return best
    rows = np.array([row_of[x] for x in big["o_id"].to_list()])
    E = emb_o[rows].astype(np.float32)
    big = big.with_row_index("i")
    sums = big.group_by("s1_id").agg(pl.col("i"))
    S = np.zeros_like(E)
    for ids in sums["i"].to_list():
        idx = np.array(ids)
        S[idx] = E[idx].sum(0)
    others = (S - E) / (big["n"].to_numpy()[:, None] - 1)
    others /= np.linalg.norm(others, axis=1, keepdims=True) + 1e-9
    cos = (E * others).sum(1)
    drop = big.filter(pl.Series((cos < cos_min) & (big["p"].to_numpy() < pmax))).select("o_id")
    return best.join(drop, on="o_id", how="anti")


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    base = lb.holdout_ens()
    hs = lb.comp(pl.read_parquet("work/llm/holdout_scored.parquet")).join(base, on=["o_id", "s1_id"], how="left") \
           .with_columns(pl.col("p_ens").fill_null(pl.col("p1"))).with_columns((pl.col("o_id").hash(7) % 2).alias("half"))
    pb = np.zeros(hs.height)
    for k in (0, 1):
        tr, te = (hs["half"] != k).to_numpy(), (hs["half"] == k).to_numpy()
        pb[te] = LogisticRegression(C=10.0).fit(lb.X(hs.filter(pl.Series(tr))), hs["y"].to_numpy()[tr]) \
                                           .predict_proba(lb.X(hs.filter(pl.Series(te))))[:, 1]
    pred = base.join(hs.select("o_id", "s1_id").with_columns(pl.Series("pb", pb)), on=["o_id", "s1_id"], how="left") \
               .select("o_id", "s1_id", pl.coalesce("pb", "p_ens").alias("p"))
    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    s1_eval = s1_all.filter((s1_all.hash(13) % 10 == 0) & (s1_all.hash(11) % 4 != 0) & (s1_all.hash(21) % 1000 < 250))
    pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    tau = 0.75
    ref = apply_mfc(pred, 0.0)
    print(f"v15 reference (tau {tau}): {macro_f05(ref, tau, pe, s1_eval):.5f}", flush=True)
    for delta in (0.02, 0.05, 0.10, 0.20):
        print(f"  MFC delta={delta:.2f}: {macro_f05(apply_mfc(pred, delta), tau, pe, s1_eval):.5f}", flush=True)
    ido = pl.concat([pl.read_parquet(f"work/norm/train_source{k}.parquet", columns=["entity_id"]) for k in (2, 3)])
    row_of = dict(zip(ido["entity_id"].to_list(), range(ido.height)))
    emb_o = np.concatenate([np.load("work/emb_ftb/train_source2.npy", mmap_mode="r"), np.load("work/emb_ftb/train_source3.npy", mmap_mode="r")])
    for cos_min in (0.5, 0.6, 0.7):
        for pmax in (0.9, 0.97):
            f = macro_f05(apply_tcf(ref, tau, emb_o, row_of, cos_min, pmax), tau, pe, s1_eval)
            print(f"  TCF cos<{cos_min} & p<{pmax}: {f:.5f}", flush=True)


if __name__ == "__main__":
    main()
