"""Ensemble of the band blend (cross-encoders A/B per band) and the second-cross-encoder blend (second cross-encoder on all candidates).

p_ens = w * p_band blend + (1 - w) * p_v11 on every final candidate (learned blocking filter, 6.1 per entity).
Evaluated on the second-cross-encoder holdout entities (both components are out-of-fold there), then applied to test.
Decision: exclusive assignment; US/India threshold TAU; France threshold chosen so France keeps the same number of
matches as the leaderboard-best reference file (FR_MATCH_REF), unless TAU_FR is given.
"""
import os
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
import ce2_blend  # noqa: E402
from ce_blend import add_comp  # noqa: E402
from predict import id_lists  # noqa: E402
from train import assign, macro_f05  # noqa: E402

M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
W = [float(x) for x in os.environ.get("ENS_W", "0.3,0.5,0.7").split(",")]
OUT = os.environ.get("OUT_DIR", "output/ens")


def band_probs(split: str) -> pl.DataFrame:
    """band-blend per-band blend probabilities (A on fold 1 + B on fold 0 for train; A/B average on test mid band)."""
    def X(d):
        p = d["p1"].to_numpy().clip(1e-6, 1 - 1e-6)
        return np.column_stack([np.log(p / (1 - p)), d["ce"].to_numpy(), d["ce_gap"].to_numpy().clip(-10, 10),
                                (d["ce_rank"] == 1).to_numpy().astype(float), np.minimum(d["n_scored"].to_numpy(), 5).astype(float)])

    def load(sp, b):
        d = pl.read_parquet(f"work/ce/{sp}_ce{b}.parquet")
        fb = f"work/ce/{sp}_ce{b}_b.parquet"
        if b == "" and os.path.exists(fb):   # band blend: model B only on the mid band
            e = pl.read_parquet(fb)
            if sp == "train":
                d = pl.concat([d, e.select(d.columns)])
            else:
                d = d.join(e.select("o_id", "s1_id", pl.col("ce").alias("ce_b")), on=["o_id", "s1_id"], how="left") \
                     .with_columns(((pl.col("ce") + pl.coalesce("ce_b", "ce")) / 2).alias("ce")).drop("ce_b")
        return d

    bands = ["", "_hi", "_lo"]
    tr = {b: load("train", b) for b in bands}
    trc = add_comp(pl.concat([d.select("o_id", "s1_id", "ce") for d in tr.values()]).unique(["o_id", "s1_id"])).drop("ce")
    lrs = {b: LogisticRegression(C=10.0).fit(X(d.join(trc, on=["o_id", "s1_id"])), d.join(trc, on=["o_id", "s1_id"])["y"].to_numpy())
           for b, d in tr.items()}
    src = tr if split == "train" else {b: load("test", b) for b in bands}
    comp = trc if split == "train" else add_comp(pl.concat([d.select("o_id", "s1_id", "ce") for d in src.values()]).unique(["o_id", "s1_id"])).drop("ce")
    parts = []
    for b, d in src.items():
        d = d.join(comp, on=["o_id", "s1_id"])
        parts.append(d.select("o_id", "s1_id").with_columns(pl.Series("pa", lrs[b].predict_proba(X(d))[:, 1])))
    return pl.concat(parts).unique(["o_id", "s1_id"], keep="first")


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    # second-cross-encoder component: cross-fitted on the holdout (same as ce2_blend), full-fit for test
    h = ce2_blend.prep(pl.read_parquet("work/ce2/holdout_scored.parquet")).with_columns((pl.col("o_id").hash(5) % 2).alias("half"))
    pb = np.zeros(h.height)
    for k in (0, 1):
        tr, te = (h["half"] != k).to_numpy(), (h["half"] == k).to_numpy()
        pb[te] = LogisticRegression(C=10.0).fit(ce2_blend.X(h.filter(pl.Series(tr))), h["y"].to_numpy()[tr]) \
                                            .predict_proba(ce2_blend.X(h.filter(pl.Series(te))))[:, 1]
    lr2 = LogisticRegression(C=10.0).fit(ce2_blend.X(h), h["y"].to_numpy())
    hv11 = h.select("o_id", "s1_id").with_columns(pl.Series("pv", pb))
    ha = band_probs("train")
    oof = pl.read_parquet(f"{M1}/oof.parquet")
    base = oof.join(ha, on=["o_id", "s1_id"], how="left").join(hv11, on=["o_id", "s1_id"], how="left") \
              .with_columns(pl.coalesce("pa", "p").alias("pa"), pl.coalesce("pv", "pa", "p").alias("pv"))
    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    SHIFT = float(os.environ.get("SHIFT_FRAC", "0"))
    if SHIFT:  # test-like distractor density: drop S1 entities; their records become distractors for the rest
        drop = s1_all.filter(s1_all.hash(17) % 1000 < SHIFT * 1000)
        base = base.filter(~pl.col("s1_id").is_in(drop.implode()))
        s1_all = s1_all.filter(~s1_all.is_in(drop.implode()))
        print(f"SHIFT_FRAC={SHIFT}: dropped {drop.len()} S1 entities -> records per S1 rises accordingly", flush=True)
    if os.environ.get("EVAL_CAP"):  # apply the same learned blocking filter as on test
        base = base.filter(pl.col("p") >= 0.001).filter(pl.col("p").rank("ordinal", descending=True).over("s1_id") <= 12)
    s1_eval = s1_all.filter((s1_all.hash(13) % 10 == 0) & (s1_all.hash(11) % 4 != 0))
    pe = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    res = {}
    for w in [1.0, 0.0] + W:
        pred = base.select("o_id", "s1_id", (w * pl.col("pa") + (1 - w) * pl.col("pv")).alias("p"))
        b = assign(pred)
        sc = [(macro_f05(b, t, pe, s1_eval), t) for t in np.arange(0.5, 0.96, 0.05)]
        print(f"   w_band blend={w:.1f} curve: " + " ".join(f"{t:.2f}:{f:.5f}" for f, t in sc), flush=True)
        best = max(sc)
        res[w] = best
        print(f"w_band blend={w:.1f}: best holdout F0.5 {best[0]:.5f} at tau {best[1]:.2f} | at 0.70: {dict((round(t,2),f) for f,t in sc)[0.7]:.5f}", flush=True)
    if os.environ.get("EVAL_ONLY"):
        return

    w = float(os.environ.get("W", "0.5"))
    tau = float(os.environ.get("TAU", "0.70"))
    ta = band_probs("test")
    tv = ce2_blend.prep(pl.read_parquet(os.environ.get("CE2_TEST_SCORED", "work/ce2/test_scored.parquet")))
    tv = tv.select("o_id", "s1_id").with_columns(pl.Series("pv", lr2.predict_proba(ce2_blend.X(tv))[:, 1]))
    p1 = pl.read_parquet("work/ce/test_p1_all.parquet")
    pred = tv.join(p1, on=["o_id", "s1_id"], how="left").join(ta, on=["o_id", "s1_id"], how="left") \
             .with_columns(pl.coalesce("pa", "p1").alias("pa"))
    if os.environ.get("SAVE_PRED"):
        pred.select("o_id", "s1_id", "pa", "pv").write_parquet(os.environ["SAVE_PRED"])
    pred = pred.select("o_id", "s1_id", (w * pl.col("pa") + (1 - w) * pl.col("pv")).alias("p"))
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    a = assign(pred).join(s1c, on="s1_id")
    if os.environ.get("TAU_FR"):
        tau_fr = float(os.environ["TAU_FR"])
    else:
        r = pl.read_csv(os.environ.get("FR_MATCH_REF", "output/stage5_blend/matching_results.tsv"), separator="\t", quote_char=None,
                        infer_schema_length=0).with_columns(pl.col("matched_entity_ids").fill_null(""))
        r = r.join(s1c.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id").filter(pl.col("country") == "France")
        n_ref = int(r.select(pl.when(pl.col("matched_entity_ids") == "").then(0)
                             .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)).sum().item())
        fr_p = np.sort(a.filter(pl.col("country") == "France")["p"].to_numpy())[::-1]
        tau_fr = float(fr_p[n_ref - 1])
    bestp = a.filter(pl.col("p") >= pl.when(pl.col("country") == "France").then(pl.lit(tau_fr)).otherwise(pl.lit(tau))).drop("country")
    os.makedirs(OUT, exist_ok=True)
    id_lists(pred, s1c["s1_id"], "candidate_entity_ids").write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    id_lists(bestp, s1c["s1_id"], "matched_entity_ids").write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    print(f"test: w={w} tau={tau} tau_fr={tau_fr:.4f} candidates {pred.height} matched {bestp.height}", flush=True)
    subprocess.call([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                     "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", "student_resource/dataset/test"])


if __name__ == "__main__":
    main()
