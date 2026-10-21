"""Meta-stacker (Foursquare-winner style): LightGBM over the ~70 pair features + every model's score, trained on
holdout entities (never used to train any cross-encoder/LLM), applied to LLM-scored test pairs.

Inputs per pair: stage-2 features (work/feat_v4), LightGBM p1, per-band blend pa (cross-encoders A/B),
cross-encoder-v2 blend pv, pairwise-LLM logit (+ gap / is-top). Rows = pairs with an LLM score.
"""
import glob
import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import llm_blend as lb  # noqa: E402
from ensemble import v10c_probs  # noqa: E402
from predict import id_lists  # noqa: E402
from train import assign, macro_f05  # noqa: E402

FEAT = os.environ.get("FEAT_DIR", "work/feat_v4")
BASE_PRED = os.environ.get("BASE_PRED", "output/v16_llmwide/pred.parquet")
ENS_PRED = os.environ.get("ENS_PRED", "work/ce2/ens_test_pred_v14.parquet")
OUT = os.environ.get("OUT_DIR", "output/v19_meta")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=48, verbose=-1)
ROUNDS = int(os.environ.get("META_ROUNDS", "400"))


def load_feats(split, keys):
    parts = []
    for f in sorted(glob.glob(f"{FEAT}/{split}/part_*.parquet")):
        parts.append(pl.read_parquet(f).join(keys, on=["o_id", "s1_id"], how="semi"))
    return pl.concat(parts)


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    # ---- holdout rows (pairs with an LLM score on holdout entities)
    h = lb.comp(pl.read_parquet("work/llm/holdout_scored.parquet"))
    base = lb.holdout_ens()                                            # p_ens (0.5 pa + 0.5 pv) for all holdout pairs
    pa = v10c_probs("train").rename({"pa": "pa_"})
    h = h.join(base, on=["o_id", "s1_id"], how="left").join(pa, on=["o_id", "s1_id"], how="left")
    fh = load_feats("train", h.select("o_id", "s1_id"))
    h = h.join(fh, on=["o_id", "s1_id"], how="inner", suffix="_f")
    drop = {"o_id", "s1_id", "y", "text_s1", "text_o", "country", "holdout", "fold"}
    cols = [c for c in h.columns if c not in drop and h[c].dtype != pl.Utf8 and not c.endswith("_f")]
    X = h.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    y = h["y"].to_numpy()
    grp = (h["o_id"].hash(3) % 2).to_numpy()
    oof = np.zeros(h.height)
    for k in (0, 1):
        m = lgb.train(PARAMS, lgb.Dataset(X[grp != k], y[grp != k], feature_name=cols), ROUNDS)
        oof[grp == k] = m.predict(X[grp == k])
    final = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cols), ROUNDS)
    imp = sorted(zip(cols, final.feature_importance("gain")), key=lambda x: -x[1])[:12]
    print(f"meta rows {h.height}, feats {len(cols)}; top: " + ", ".join(f"{c}={v:.0f}" for c, v in imp), flush=True)

    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    s1_eval = s1_all.filter((s1_all.hash(13) % 10 == 0) & (s1_all.hash(11) % 4 != 0) & (s1_all.hash(21) % 1000 < 250))
    pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    # reference: v15-style LR blend on the same rows (cross-fitted)
    from sklearn.linear_model import LogisticRegression
    Xl = lb.X(h)
    plr = np.zeros(h.height)
    for k in (0, 1):
        plr[grp == k] = LogisticRegression(C=10.0).fit(Xl[grp != k], y[grp != k]).predict_proba(Xl[grp == k])[:, 1]
    bl = assign(base.join(h.select("o_id", "s1_id").with_columns(pl.Series("pb", plr)), on=["o_id", "s1_id"], how="left")
                .select("o_id", "s1_id", pl.coalesce("pb", "p_ens").alias("p")))
    bm = assign(base.join(h.select("o_id", "s1_id").with_columns(pl.Series("pb", oof)), on=["o_id", "s1_id"], how="left")
                .select("o_id", "s1_id", pl.coalesce("pb", "p_ens").alias("p")))
    best = (0.0, 0.0)
    print(f"{'tau':>5} {'LR blend(v15)':>14} {'meta-stack':>11}")
    for t in np.arange(0.55, 0.96, 0.05):
        f0, f1 = macro_f05(bl, t, pe, s1_eval), macro_f05(bm, t, pe, s1_eval)
        best = max(best, (f1, t))
        print(f"{t:5.2f} {f0:14.5f} {f1:11.5f}", flush=True)
    print(f"meta best {best[0]:.5f} at tau {best[1]:.2f}", flush=True)
    if os.environ.get("EVAL_ONLY"):
        return

    # ---- test: LLM-scored pairs get the meta probability; all others keep BASE_PRED
    lt = pl.concat([pl.read_parquet(f).select("o_id", "s1_id", "llm") for f in
                    os.environ.get("LLM_TEST_FILES", "work/llm/test_scored.parquet,work/llm/test2_scored.parquet").split(",")]).unique(["o_id", "s1_id"])
    ens = pl.read_parquet(ENS_PRED).select("o_id", "s1_id", ((pl.col("pa") + pl.col("pv")) / 2).alias("p_ens"), pl.col("pa").alias("pa_"))
    t = lb.comp(lt).join(ens, on=["o_id", "s1_id"])
    t = t.join(load_feats("test", t.select("o_id", "s1_id")), on=["o_id", "s1_id"], how="inner", suffix="_f")
    t = t.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in cols if c not in t.columns])
    pt = final.predict(t.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy())
    bp = pl.read_parquet(BASE_PRED)
    pred = bp.join(t.select("o_id", "s1_id").with_columns(pl.Series("pm", pt)), on=["o_id", "s1_id"], how="left") \
             .select("o_id", "s1_id", pl.coalesce("pm", "p").alias("p"))
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    a = assign(pred).join(s1c, on="s1_id")
    r = pl.read_csv("output/v13_fr_st/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0) \
          .with_columns(pl.col("matched_entity_ids").fill_null("")).join(s1c.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id") \
          .filter(pl.col("country") == "France")
    n_ref = int(r.select(pl.when(pl.col("matched_entity_ids") == "").then(0)
                         .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)).sum().item())
    tau = float(os.environ.get("TAU", best[1]))
    tau_fr = float(np.sort(a.filter(pl.col("country") == "France")["p"].to_numpy())[::-1][n_ref - 1])
    bestp = a.filter(pl.col("p") >= pl.when(pl.col("country") == "France").then(pl.lit(tau_fr)).otherwise(pl.lit(tau))).drop("country")
    os.makedirs(OUT, exist_ok=True)
    pred.write_parquet(f"{OUT}/pred.parquet")
    id_lists(pred, s1c["s1_id"], "candidate_entity_ids").write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    id_lists(bestp, s1c["s1_id"], "matched_entity_ids").write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    print(f"test: tau {tau:.2f} tau_fr {tau_fr:.4f} meta-scored {t.height} matched {bestp.height}", flush=True)
    subprocess.call([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                     "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", "student_resource/dataset/test"])


if __name__ == "__main__":
    main()
