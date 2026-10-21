"""Blend the LLM matcher into the ensemble on uncertain pairs; evaluate on the holdout slice; write outputs.

p_final = LR([logit(p_ens), llm_logit, llm gap to best other scored candidate, is-top]) on LLM-scored pairs, p_ens elsewhere.
LR cross-fitted on the holdout slice (entities never used to train the LLM or second cross-encoder).
Test p_ens comes from ENS_PRED (pa/pv columns saved by ensemble.py). France threshold: match count of FR_MATCH_REF.
"""
import os
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import ce2_blend  # noqa: E402
from ensemble import band_probs  # noqa: E402
from predict import id_lists  # noqa: E402
from train import assign, macro_f05  # noqa: E402

M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
ENS_PRED = os.environ.get("ENS_PRED", "work/ce2/ens_test_pred_fr1.parquet")
OUT = os.environ.get("OUT_DIR", "output/final")


def comp(d):
    top2 = d.sort("llm", descending=True).group_by("o_id", maintain_order=True).agg(
        pl.col("llm").first().alias("l1"), pl.col("llm").slice(1, 1).first().alias("l2"))
    return d.join(top2, on="o_id").with_columns(
        pl.when(pl.col("llm") >= pl.col("l1")).then(pl.col("llm") - pl.col("l2").fill_null(-10.0))
        .otherwise(pl.col("llm") - pl.col("l1")).alias("llm_gap"),
        (pl.col("llm") >= pl.col("l1")).cast(pl.Float64).alias("llm_top")).drop("l1", "l2")


def X(d):
    p = d["p_ens"].to_numpy().clip(1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1 - p)), d["llm"].to_numpy(), d["llm_gap"].to_numpy().clip(-10, 10), d["llm_top"].to_numpy()])


def holdout_ens():
    """p_ens on the second-cross-encoder holdout: 0.5 * band blend (out-of-fold) + 0.5 * cross-fitted second-CE blend."""
    h = ce2_blend.prep(pl.read_parquet("work/ce2/holdout_scored.parquet")).with_columns((pl.col("o_id").hash(5) % 2).alias("half"))
    pv = np.zeros(h.height)
    for k in (0, 1):
        tr, te = (h["half"] != k).to_numpy(), (h["half"] == k).to_numpy()
        pv[te] = LogisticRegression(C=10.0).fit(ce2_blend.X(h.filter(pl.Series(tr))), h["y"].to_numpy()[tr]) \
                                           .predict_proba(ce2_blend.X(h.filter(pl.Series(te))))[:, 1]
    hv = h.select("o_id", "s1_id").with_columns(pl.Series("pv", pv))
    oof = pl.read_parquet(f"{M1}/oof.parquet")
    base = oof.join(band_probs("train"), on=["o_id", "s1_id"], how="left").join(hv, on=["o_id", "s1_id"], how="left") \
              .with_columns(pl.coalesce("pa", "p").alias("pa"), pl.coalesce("pv", "pa", "p").alias("pv"))
    return base.select("o_id", "s1_id", ((pl.col("pa") + pl.col("pv")) / 2).alias("p_ens"))


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    base = holdout_ens()
    hs = comp(pl.read_parquet("work/llm/holdout_scored.parquet")).join(base, on=["o_id", "s1_id"], how="left") \
         .with_columns(pl.col("p_ens").fill_null(pl.col("p1"))).with_columns((pl.col("o_id").hash(7) % 2).alias("half"))
    pb = np.zeros(hs.height)
    for k in (0, 1):
        tr, te = (hs["half"] != k).to_numpy(), (hs["half"] == k).to_numpy()
        pb[te] = LogisticRegression(C=10.0).fit(X(hs.filter(pl.Series(tr))), hs["y"].to_numpy()[tr]).predict_proba(X(hs.filter(pl.Series(te))))[:, 1]
    lr = LogisticRegression(C=10.0).fit(X(hs), hs["y"].to_numpy())
    print("LLM blend coef [logit p_ens, llm, llm_gap, llm_top]:", lr.coef_.round(3), "int", lr.intercept_.round(3), flush=True)
    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    s1_eval = s1_all.filter((s1_all.hash(13) % 10 == 0) & (s1_all.hash(11) % 4 != 0) & (s1_all.hash(21) % 1000 < 250))
    pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    b0 = assign(base.rename({"p_ens": "p"}))
    b1 = assign(base.join(hs.select("o_id", "s1_id").with_columns(pl.Series("pb", pb)), on=["o_id", "s1_id"], how="left")
                .select("o_id", "s1_id", pl.coalesce("pb", "p_ens").alias("p")))
    best = (0.0, 0.0)
    print(f"{'tau':>5} {'ensemble':>9} {'+LLM':>9}")
    for t in np.arange(0.55, 0.96, 0.05):
        f0, f1 = macro_f05(b0, t, pe, s1_eval), macro_f05(b1, t, pe, s1_eval)
        best = max(best, (f1, t))
        print(f"{t:5.2f} {f0:9.5f} {f1:9.5f}", flush=True)
    tau = float(os.environ.get("TAU", best[1]))
    print(f"holdout best +LLM {best[0]:.5f} at tau {best[1]:.2f}; using tau {tau:.2f}", flush=True)
    if os.environ.get("EVAL_ONLY"):
        return

    ens = pl.read_parquet(ENS_PRED).select("o_id", "s1_id", ((pl.col("pa") + pl.col("pv")) / 2).alias("p_ens"))
    files = os.environ.get("LLM_TEST_FILES", "work/llm/test_scored.parquet").split(",")
    lt = pl.concat([pl.read_parquet(f).select("o_id", "s1_id", "llm") for f in files]).unique(["o_id", "s1_id"])
    ts = comp(lt).join(ens, on=["o_id", "s1_id"])
    tb = ts.select("o_id", "s1_id").with_columns(pl.Series("pb", lr.predict_proba(X(ts))[:, 1]))
    pred = ens.join(tb, on=["o_id", "s1_id"], how="left").select("o_id", "s1_id", pl.coalesce("pb", "p_ens").alias("p"))
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    a = assign(pred).join(s1c, on="s1_id")
    r = pl.read_csv(os.environ.get("FR_MATCH_REF", "output/stage8_fr1/matching_results.tsv"), separator="\t", quote_char=None,
                    infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
    r = r.join(s1c.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id").filter(pl.col("country") == "France")
    n_ref = int(r.select(pl.when(pl.col("matched_entity_ids") == "").then(0)
                         .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)).sum().item())
    tau_fr = float(np.sort(a.filter(pl.col("country") == "France")["p"].to_numpy())[::-1][n_ref - 1])
    bestp = a.filter(pl.col("p") >= pl.when(pl.col("country") == "France").then(pl.lit(tau_fr)).otherwise(pl.lit(tau))).drop("country")
    os.makedirs(OUT, exist_ok=True)
    pred.write_parquet(f"{OUT}/pred.parquet")   # per-pair final probabilities (for later blending stages)
    id_lists(pred, s1c["s1_id"], "candidate_entity_ids").write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    id_lists(bestp, s1c["s1_id"], "matched_entity_ids").write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    print(f"test: tau {tau:.2f} tau_fr {tau_fr:.4f} LLM-scored {ts.height} matched {bestp.height}", flush=True)
    subprocess.call([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                     "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", "student_resource/dataset/test"])


if __name__ == "__main__":
    main()
