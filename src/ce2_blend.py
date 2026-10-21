"""Blend LightGBM (p1) with second cross-encoder on ALL final candidates; evaluate on holdout entities; write outputs.

Features: logit(p1), ce2 logit, ce2 gap to the best other candidate of the same record, is-top, #scored.
Blend fitted with 2-fold cross-fitting on the 10% holdout entities (never used to train second cross-encoder).
Decision: exclusive assignment, threshold tau (US/India) and tau + FR_DELTA for France (leaderboard-calibrated).
"""
import os
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from ce_blend import add_comp  # noqa: E402
from predict import id_lists  # noqa: E402
from train import assign, macro_f05  # noqa: E402

M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
OUT = os.environ.get("OUT_DIR", "output/ce2_blend")
FR_DELTA = float(os.environ.get("FR_DELTA", "0.15"))
OLD = os.environ.get("CE2_WITH_OLD", "0") == "1"   # also average in cross-encoder A/B logits where available


def X(d):
    p = d["p1"].to_numpy().clip(1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1 - p)), d["ce"].to_numpy(), d["ce_gap"].to_numpy().clip(-10, 10),
                            (d["ce_rank"] == 1).to_numpy().astype(float), np.minimum(d["n_scored"].to_numpy(), 5).astype(float)])


def prep(d):
    return d.rename({"ce2": "ce"}).join(add_comp(d.select("o_id", "s1_id", pl.col("ce2").alias("ce"))).drop("ce"),
                                         on=["o_id", "s1_id"], how="left")


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    h = prep(pl.read_parquet("work/ce2/holdout_scored.parquet"))
    h = h.with_columns((pl.col("o_id").hash(5) % 2).alias("half"))
    pb = np.zeros(h.height)
    for k in (0, 1):
        tr, te = (h["half"] != k).to_numpy(), (h["half"] == k).to_numpy()
        pb[te] = LogisticRegression(C=10.0).fit(X(h.filter(pl.Series(tr))), h["y"].to_numpy()[tr]).predict_proba(X(h.filter(pl.Series(te))))[:, 1]
    lr = LogisticRegression(C=10.0).fit(X(h), h["y"].to_numpy())
    print("blend coef [logit p1, ce2, gap, top, n]:", lr.coef_.round(3), "int", lr.intercept_.round(3), flush=True)

    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    s1_eval = s1_all.filter((s1_all.hash(13) % 10 == 0) & (s1_all.hash(11) % 4 != 0))
    pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    oof = pl.read_parquet(f"{M1}/oof.parquet")
    blended = oof.join(h.select("o_id", "s1_id").with_columns(pl.Series("pb", pb)), on=["o_id", "s1_id"], how="left") \
                 .with_columns(pl.coalesce("pb", "p").alias("p")).select("o_id", "s1_id", "p")
    b0, b1 = assign(oof), assign(blended)
    best = (0.0, 0.0)
    print(f"{'tau':>5} {'LightGBM':>9} {'+ce2':>9}")
    for tau in np.arange(0.5, 0.985, 0.02):
        f0, f1 = macro_f05(b0, tau, pe, s1_eval), macro_f05(b1, tau, pe, s1_eval)
        best = max(best, (f1, tau))
        print(f"{tau:5.2f} {f0:9.5f} {f1:9.5f}", flush=True)
    f, tau = best
    tau = float(os.environ.get("TAU", tau))
    print(f"holdout best {f:.5f} at tau {best[1]:.2f}; using tau {tau:.2f}", flush=True)
    if os.environ.get("EVAL_ONLY"):
        return

    t = prep(pl.read_parquet("work/ce2/test_scored.parquet"))
    pred = t.select("o_id", "s1_id").with_columns(pl.Series("p", lr.predict_proba(X(t))[:, 1]))
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    a = assign(pred).join(s1c, on="s1_id")
    ref = os.environ.get("FR_MATCH_REF", "")
    if ref:  # France has no labels: match the France match count of the leaderboard-calibrated reference submission
        r = pl.read_csv(ref, separator="\t", quote_char=None, infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
        r = r.join(s1c.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id").filter(pl.col("country") == "France")
        n_ref = int(r.select(pl.when(pl.col("matched_entity_ids") == "").then(0).otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)).sum().item())
        fr_p = np.sort(a.filter(pl.col("country") == "France")["p"].to_numpy())[::-1]
        tau_fr = float(fr_p[min(n_ref, len(fr_p)) - 1]) if n_ref else 1.0
        print(f"France: reference has {n_ref} matches -> tau_fr {tau_fr:.4f}", flush=True)
    else:
        tau_fr = min(tau + FR_DELTA, 0.99)
    bestp = a.filter(pl.col("p") >= pl.when(pl.col("country") == "France").then(pl.lit(tau_fr)).otherwise(pl.lit(tau))).drop("country")
    print(f"test decision: tau {tau:.3f}, France tau {tau_fr:.4f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    s1 = s1c["s1_id"]
    id_lists(pred, s1, "candidate_entity_ids").write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    id_lists(bestp, s1, "matched_entity_ids").write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    print(f"test: candidates {pred.height} ({pred.height / s1.len():.2f}/S1), matched pairs {bestp.height}", flush=True)
    subprocess.call([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                     "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", "student_resource/dataset/test"])


if __name__ == "__main__":
    main()
