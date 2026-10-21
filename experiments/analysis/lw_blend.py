"""Blend the listwise ("select") LLM into the final probabilities for uncertain records.

For every candidate of a listwise-scored record: p_final = LR([logit(p_base), p_list, p_none, is_list_top]).
Fitted (cross-fitted for evaluation) on holdout records of cross-encoder-v2 holdout entities; p_base on the holdout is
the ensemble probability. Test p_base = BASE_PRED (pred.parquet written by llm_blend.py for v16/v17).
"""
import os
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import llm_blend as lb  # noqa: E402
from predict import id_lists  # noqa: E402
from train import assign, macro_f05  # noqa: E402

BASE_PRED = os.environ.get("BASE_PRED", "output/v17_all/pred.parquet")
OUT = os.environ.get("OUT_DIR", "output/v18_lw")


def feats(d):
    d = d.with_columns((pl.col("p_list") >= pl.col("p_list").max().over("o_id")).cast(pl.Float64).alias("lw_top"))
    p = d["p_base"].to_numpy().clip(1e-6, 1 - 1e-6)
    return d, np.column_stack([np.log(p / (1 - p)), d["p_list"].to_numpy(), d["pnone"].to_numpy(), d["lw_top"].to_numpy()])


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    base = lb.holdout_ens().rename({"p_ens": "p_base"})
    h = pl.read_parquet("work/lw/holdout_scored.parquet").join(base, on=["o_id", "s1_id"], how="inner")
    h = h.join(pairs.with_columns(pl.lit(1).alias("y")), on=["o_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
    h, Xh = feats(h)
    half = (h["o_id"].hash(9) % 2).to_numpy()
    pb = np.zeros(h.height)
    for k in (0, 1):
        pb[half == k] = LogisticRegression(C=10.0).fit(Xh[half != k], h["y"].to_numpy()[half != k]).predict_proba(Xh[half == k])[:, 1]
    lr = LogisticRegression(C=10.0).fit(Xh, h["y"].to_numpy())
    print("listwise blend coef [logit p_base, p_list, p_none, lw_top]:", lr.coef_.round(3), "int", lr.intercept_.round(3), flush=True)
    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    s1_eval = s1_all.filter((s1_all.hash(13) % 10 == 0) & (s1_all.hash(11) % 4 != 0))
    if os.environ.get("TOUCHED_ONLY"):  # only entities that appear among listwise-scored candidates
        s1_eval = s1_eval.filter(s1_eval.is_in(h["s1_id"].unique().implode()))
        print(f"TOUCHED_ONLY: evaluating on {s1_eval.len()} entities", flush=True)
    pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    b0 = assign(base.rename({"p_base": "p"}))
    b1 = assign(base.join(h.select("o_id", "s1_id").with_columns(pl.Series("pb", pb)), on=["o_id", "s1_id"], how="left")
                .select("o_id", "s1_id", pl.coalesce("pb", "p_base").alias("p")))
    best = (0.0, 0.0)
    print(f"{'tau':>5} {'base':>9} {'+listwise':>10}")
    for t in np.arange(0.55, 0.96, 0.05):
        f0, f1 = macro_f05(b0, t, pe, s1_eval), macro_f05(b1, t, pe, s1_eval)
        best = max(best, (f1, t))
        print(f"{t:5.2f} {f0:9.5f} {f1:10.5f}", flush=True)
    print(f"holdout best +listwise {best[0]:.5f} at tau {best[1]:.2f}", flush=True)
    if os.environ.get("EVAL_ONLY"):
        return
    tau = float(os.environ.get("TAU", "0.75"))
    bp = pl.read_parquet(BASE_PRED).rename({"p": "p_base"})
    t = pl.read_parquet("work/lw/test_scored.parquet").join(bp, on=["o_id", "s1_id"], how="inner")
    t, Xt = feats(t)
    tb = t.select("o_id", "s1_id").with_columns(pl.Series("pb", lr.predict_proba(Xt)[:, 1]))
    pred = bp.join(tb, on=["o_id", "s1_id"], how="left").select("o_id", "s1_id", pl.coalesce("pb", "p_base").alias("p"))
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    a = assign(pred).join(s1c, on="s1_id")
    r = pl.read_csv(os.environ.get("FR_MATCH_REF", "output/v13_fr_st/matching_results.tsv"), separator="\t", quote_char=None,
                    infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
    r = r.join(s1c.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id").filter(pl.col("country") == "France")
    n_ref = int(r.select(pl.when(pl.col("matched_entity_ids") == "").then(0)
                         .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)).sum().item())
    tau_fr = float(np.sort(a.filter(pl.col("country") == "France")["p"].to_numpy())[::-1][n_ref - 1])
    bestp = a.filter(pl.col("p") >= pl.when(pl.col("country") == "France").then(pl.lit(tau_fr)).otherwise(pl.lit(tau))).drop("country")
    os.makedirs(OUT, exist_ok=True)
    pred.write_parquet(f"{OUT}/pred.parquet")
    id_lists(pred, s1c["s1_id"], "candidate_entity_ids").write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    id_lists(bestp, s1c["s1_id"], "matched_entity_ids").write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    print(f"test: tau {tau} tau_fr {tau_fr:.4f} listwise records {t['o_id'].n_unique()} matched {bestp.height}", flush=True)
    subprocess.call([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                     "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", "student_resource/dataset/test"])


if __name__ == "__main__":
    main()
