"""Blend stage-1 LightGBM probability with the cross-encoder logit on uncertain pairs; evaluate + write submission.

Evaluation uses fold-1 train entities only (the cross-encoder never trained on them).
The 3-parameter logistic blend is fitted with 2-fold cross-fitting inside fold 1, so the reported score is honest.
"""
import os
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from predict import id_lists  # noqa: E402
from train import assign, macro_f05  # noqa: E402

M1 = os.environ.get("MODEL1_DIR", "work/model_v6s1")
OUT = os.environ.get("OUT_DIR", "output/band_blend")


def decide_expf(best: pl.DataFrame, pmin: float = 0.3) -> pl.DataFrame:
    """Per S1, keep the top-k assigned records (by p) that maximise approximate expected F0.5.
    E[F](k) ~ 1.25*sum(top-k p) / (0.25*k + sum(all p));  E[F](0) = prod(1-p)  (empty prediction for a singleton)."""
    b = best.filter(pl.col("p") >= pmin).sort(["s1_id", "p"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("k"),
        pl.col("p").cum_sum().over("s1_id").alias("cs"),
        pl.col("p").sum().over("s1_id").alias("T"),
        (1 - pl.col("p")).log().sum().over("s1_id").exp().alias("p_none"))
    b = b.with_columns((1.25 * pl.col("cs") / (0.25 * pl.col("k") + pl.col("T"))).alias("ef"))
    best_k = b.group_by("s1_id").agg(pl.col("ef").max().alias("ef_max"), pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"),
                                     pl.col("p_none").first())
    keep = best_k.filter(pl.col("ef_max") > pl.col("p_none")).select("s1_id", "k_best")
    return b.join(keep, on="s1_id").filter(pl.col("k") <= pl.col("k_best")).select("o_id", "s1_id", "p")


COMP = os.environ.get("CE_COMP", "1") == "1"
PGAP = os.environ.get("CE_PGAP", "0") == "1"
CAP = int(os.environ.get("CAND_CAP", "0"))  # final blocking: keep top-CAP records per S1 entity by embedding cosine


CAP_BY = os.environ.get("CAP_BY", "escore")   # "escore" (embedding cosine) or "p1" (learned blocking: LightGBM prob)
PMIN_BLOCK = float(os.environ.get("PMIN_BLOCK", "0"))  # learned blocking: also drop pairs with p1 below this


def capped(split: str) -> pl.DataFrame:
    if CAP_BY == "p1":
        u = pl.read_parquet(f"{M1}/oof.parquet").rename({"p": "s"}) if split == "train" else \
            pl.read_parquet("work/ce/test_p1_all.parquet").rename({"p1": "s"})
        u = u.filter(pl.col("s") >= PMIN_BLOCK)
    else:
        u = pl.read_parquet(f"work/cand/{split}_union4.parquet", columns=["o_id", "s1_id", "escore"]).rename({"escore": "s"})
    return u.filter(pl.col("s").rank("ordinal", descending=True).over("s1_id") <= CAP).select("o_id", "s1_id")


def p1_gap(all_p: pl.DataFrame) -> pl.DataFrame:
    """LightGBM probability minus the best OTHER candidate of the same record (all candidates)."""
    top2 = all_p.sort("p1", descending=True).group_by("o_id", maintain_order=True).agg(
        pl.col("p1").first().alias("q1"), pl.col("p1").slice(1, 1).first().fill_null(0.0).alias("q2"))
    return all_p.join(top2, on="o_id").with_columns(
        pl.when(pl.col("p1") >= pl.col("q1")).then(pl.col("p1") - pl.col("q2")).otherwise(pl.col("p1") - pl.col("q1")).alias("p1_gap")
    ).select("o_id", "s1_id", "p1_gap")


def add_comp(d: pl.DataFrame) -> pl.DataFrame:
    """ce gap to the best OTHER scored candidate of the same record, and ce rank within the record."""
    d = d.with_columns(pl.col("ce").rank("ordinal", descending=True).over("o_id").alias("ce_rank"),
                       pl.len().over("o_id").alias("n_scored"))
    top2 = d.sort("ce", descending=True).group_by("o_id", maintain_order=True).agg(
        pl.col("ce").first().alias("c1"), pl.col("ce").slice(1, 1).first().alias("c2"))
    d = d.join(top2, on="o_id").with_columns(
        pl.when(pl.col("ce_rank") == 1).then(pl.col("ce") - pl.col("c2").fill_null(-10.0))
        .otherwise(pl.col("ce") - pl.col("c1")).alias("ce_gap"))
    return d.drop("c1", "c2")


def X(df):
    p = df["p1"].to_numpy().clip(1e-6, 1 - 1e-6)
    cols = [np.log(p / (1 - p)), df["ce"].to_numpy()]
    if COMP and "ce_gap" in df.columns:
        cols += [df["ce_gap"].to_numpy().clip(-10, 10), (df["ce_rank"] == 1).to_numpy().astype(float),
                 np.minimum(df["n_scored"].to_numpy(), 5).astype(float)]
    if PGAP and "p1_gap" in df.columns:
        cols += [df["p1_gap"].to_numpy()]
    return np.column_stack(cols)


def main():
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    oof = pl.read_parquet(f"{M1}/oof.parquet").rename({"p": "p1"})
    keep_tr = capped("train") if CAP else None
    if CAP:
        oof = oof.join(keep_tr, on=["o_id", "s1_id"], how="semi")
        print(f"CAND_CAP={CAP}: train candidates kept {oof.height}", flush=True)
    keep_te = capped("test") if CAP else None
    BANDS = os.environ.get("CE_BANDS", ",_hi,_lo").split(",")   # "" = mid band, "_hi" confident, "_lo" low
    SFX = os.environ.get("CE_SFX", "")                                # model suffix to average, e.g. ",_b"
    TWO = os.environ.get("CE_TWO", "0") == "1"   # use cross-encoder B too (trained on fold 1)
    def load_band(split, b):
        fa, fb = f"work/ce/{split}_ce{b}.parquet", f"work/ce/{split}_ce{b}_b.parquet"
        if not os.path.exists(fa):
            return None
        d = pl.read_parquet(fa)
        if TWO and os.path.exists(fb):
            e = pl.read_parquet(fb)
            if split == "train":   # A scored fold 1, B scored fold 0 -> full out-of-fold coverage
                d = pl.concat([d, e.select(d.columns)])
            else:                  # test: average the two models' logits
                d = d.join(e.select("o_id", "s1_id", pl.col("ce").alias("ce_b")), on=["o_id", "s1_id"], how="left") \
                     .with_columns(((pl.col("ce") + pl.coalesce("ce_b", "ce")) / 2).alias("ce")).drop("ce_b")
        if CAP:
            d = d.join(keep_tr if split == "train" else keep_te, on=["o_id", "s1_id"], how="semi")
        return d
    parts, lrs = [], {}
    loaded = {b: load_band("train", b) for b in BANDS}
    allb = pl.concat([d.select("o_id", "s1_id", "ce") for d in loaded.values() if d is not None]).unique(["o_id", "s1_id"])
    comp = add_comp(allb).drop("ce")
    if PGAP:
        comp = comp.join(p1_gap(oof), on=["o_id", "s1_id"], how="left")
    for b in BANDS:
        band = loaded[b]
        if band is None:
            continue
        band = band.join(comp, on=["o_id", "s1_id"], how="left")
        band = band.with_columns((pl.col("o_id").hash(5) % 2).alias("half"))
        pb = np.zeros(band.height)
        for h in (0, 1):
            tr, te = (band["half"] != h).to_numpy(), (band["half"] == h).to_numpy()
            lr = LogisticRegression(C=10.0).fit(X(band.filter(pl.Series(tr))), band["y"].to_numpy()[tr])
            pb[te] = lr.predict_proba(X(band.filter(pl.Series(te))))[:, 1]
        lrs[b] = LogisticRegression(C=10.0).fit(X(band), band["y"].to_numpy())
        print(f"band '{b or 'mid'}': {band.height} rows, coef {lrs[b].coef_.round(3)} int {lrs[b].intercept_.round(3)}", flush=True)
        parts.append(band.select("o_id", "s1_id").with_columns(pl.Series("pb", pb)))
    band = pl.concat(parts).unique(["o_id", "s1_id"], keep="first")
    pb = band["pb"].to_numpy()

    # evaluation universe: fold-1 entities (never seen by the cross-encoder, never used to fine-tune encoders)
    s1_all = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    s1_eval = s1_all.filter(s1_all.hash(11) % 4 != 0) if (TWO and os.environ.get("EVAL_SET", "all") == "all") else s1_all.filter(s1_all.hash(11) % 2 == 1)
    pairs_eval = pairs.filter(pl.col("s1_id").is_in(s1_eval.implode()))
    base = oof  # all records compete; fold-0 records keep their LightGBM probability
    blended = base.join(band.select("o_id", "s1_id", "pb"), on=["o_id", "s1_id"], how="left") \
                  .with_columns(pl.coalesce("pb", "p1").alias("p")).select("o_id", "s1_id", "p")
    base = base.rename({"p1": "p"})
    b0, b1 = assign(base), assign(blended)
    print(f"{'tau':>5} {'LightGBM only':>14} {'+ cross-enc':>12}")
    best = (0, 0.0)
    for tau in np.arange(0.4, 0.86, 0.05):
        f0, f1 = macro_f05(b0, tau, pairs_eval, s1_eval), macro_f05(b1, tau, pairs_eval, s1_eval)
        best = max(best, (f1, tau))
        print(f"{tau:5.2f} {f0:14.5f} {f1:12.5f}", flush=True)
    f_best, tau = best
    print(f"best blended F0.5 {f_best:.5f} at tau {tau:.2f}", flush=True)

    # test submission
    if not all(os.path.exists(f"work/ce/test_ce{b}.parquet") for b in lrs) or os.environ.get("EVAL_ONLY"):
        print("test scores not ready -> evaluation only", flush=True)
        return
    p1 = pl.read_parquet("work/ce/test_p1_all.parquet")
    if CAP:
        p1 = p1.join(keep_te, on=["o_id", "s1_id"], how="semi")
        print(f"CAND_CAP={CAP}: test candidates kept {p1.height} ({p1.height / 1732544:.1f} per S1)", flush=True)
    tparts = []
    tloaded = {b: load_band("test", b) for b in lrs}
    tcomp = add_comp(pl.concat([d.select("o_id", "s1_id", "ce") for d in tloaded.values()]).unique(["o_id", "s1_id"])).drop("ce")
    if PGAP:
        tcomp = tcomp.join(p1_gap(pl.read_parquet("work/ce/test_p1_all.parquet")), on=["o_id", "s1_id"], how="left")
    for b, lr in lrs.items():
        tb = tloaded[b].join(tcomp, on=["o_id", "s1_id"], how="left")
        tparts.append(tb.with_columns(pl.Series("pb", lr.predict_proba(X(tb))[:, 1])).select("o_id", "s1_id", "pb"))
    tb = pl.concat(tparts).unique(["o_id", "s1_id"], keep="first")
    pred = p1.join(tb, on=["o_id", "s1_id"], how="left").with_columns(pl.coalesce("pb", "p1").alias("p")).select("o_id", "s1_id", "p")
    s1 = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id"])["entity_id"]
    os.makedirs(OUT, exist_ok=True)
    tau = float(os.environ.get("TAU", tau))
    tau_fr = float(os.environ.get("TAU_FR", tau))
    s1c = pl.read_parquet("work/norm/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1_id"})
    bestp = assign(pred).join(s1c, on="s1_id").filter(
        pl.col("p") >= pl.when(pl.col("country") == "France").then(pl.lit(tau_fr)).otherwise(pl.lit(tau))).drop("country")
    print(f"test decision: tau {tau:.3f}, France tau {tau_fr:.3f}", flush=True)
    id_lists(pred, s1, "candidate_entity_ids").write_csv(f"{OUT}/candidate_pairs.tsv", separator="\t", quote_style="never")
    id_lists(bestp, s1, "matched_entity_ids").write_csv(f"{OUT}/matching_results.tsv", separator="\t", quote_style="never")
    base_n = assign(p1.rename({"p1": "p"})).filter(pl.col("p") >= tau).height
    print(f"test matched pairs: blended {bestp.height}  (LightGBM-only at same tau: {base_n})", flush=True)
    subprocess.call([sys.executable, "student_resource/utils/validate_submission.py", "--matching", f"{OUT}/matching_results.tsv",
                     "--candidate", f"{OUT}/candidate_pairs.tsv", "--test-dir", "student_resource/dataset/test"])


if __name__ == "__main__":
    main()
