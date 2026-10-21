"""Stage 3+4: train LightGBM matcher, out-of-fold macro-F0.5 evaluation, threshold tuning, final model.

Outputs: work/model/lgb.txt, work/model/threshold.txt, work/model/oof.parquet
"""
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

FEAT = os.environ.get("FEAT_DIR", "work/feat")
MODEL = os.environ.get("MODEL_DIR", "work/model")
ROUNDS = int(os.environ.get("ROUNDS", "600"))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.5"))  # fraction of O records used for fitting
ID_COLS = ["o_id", "s1_id"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=int(os.environ.get("THREADS", "64")), verbose=-1, max_bin=255)


def load():
    df = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f"{FEAT}/train/part_*.parquet"))])
    pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
    df = df.join(pairs.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["o_id", "s1_id"], how="left") \
           .with_columns(pl.col("y").fill_null(0))
    # group = true S1 of the O record (or the O record itself if unmatched) -> leakage-free folds
    grp = df.select("o_id").unique().join(pairs.select("o_id", pl.col("s1_id").alias("g")), on="o_id", how="left") \
            .with_columns(pl.coalesce("g", "o_id").hash(11).alias("gh"))
    df = df.join(grp.select("o_id", "gh", "g"), on="o_id")
    if os.environ.get("EXCLUDE_FT"):
        ft = pairs.filter(pl.col("s1_id").hash(11) % 4 == 0)
        n0 = df.height
        df = df.filter(~pl.col("o_id").is_in(ft["o_id"].implode()))
        pairs = pairs.filter(pl.col("s1_id").hash(11) % 4 != 0)
        print(f"EXCLUDE_FT: dropped {n0 - df.height} rows of records owned by fine-tune entities", flush=True)
    return df.drop("g"), pairs


def feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS + ["y", "gh", "country"]]


def to_np(df, cols):
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def assign(pred: pl.DataFrame) -> pl.DataFrame:
    """Each O record goes to its single best S1 (exclusivity)."""
    return pred.sort("p", descending=True).unique("o_id", keep="first")


def macro_f05(best: pl.DataFrame, tau: float, pairs: pl.DataFrame, all_s1: pl.Series) -> float:
    pr = best.filter(pl.col("p") >= tau).select("s1_id", "o_id")
    tp = pr.join(pairs, on=["s1_id", "o_id"]).group_by("s1_id").len().rename({"len": "tp"})
    npred = pr.group_by("s1_id").len().rename({"len": "npred"})
    ntrue = pairs.group_by("s1_id").len().rename({"len": "ntrue"})
    e = pl.DataFrame({"s1_id": all_s1}).join(npred, on="s1_id", how="left").join(ntrue, on="s1_id", how="left") \
        .join(tp, on="s1_id", how="left").fill_null(0)
    p = e["tp"] / e["npred"].clip(lower_bound=1)
    r = e["tp"] / e["ntrue"].clip(lower_bound=1)
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0.0)
    f = pl.Series(np.where((e["ntrue"] == 0).to_numpy(), (e["npred"] == 0).to_numpy().astype(float), f.to_numpy()))
    return float(f.mean())


def main():
    os.makedirs(MODEL, exist_ok=True)
    t = time.time()
    df, pairs = load()
    cols = feature_cols(df)
    all_s1 = pl.read_parquet("work/norm/train_source1.parquet", columns=["entity_id"])["entity_id"]
    if os.environ.get("EXCLUDE_FT"):
        all_s1 = all_s1.filter(all_s1.hash(11) % 4 != 0)
    print(f"rows {df.height} pos {df['y'].sum()} feats {len(cols)} load {time.time()-t:.0f}s", flush=True)

    fold = (df["gh"] % 2).to_numpy()
    fit_mask = ((df["gh"] // 2) % 1000 < TRAIN_FRAC * 1000).to_numpy()
    oof = np.zeros(df.height, dtype=np.float32)
    for k in (0, 1):
        tr = (fold != k) & fit_mask
        X, y = to_np(df.filter(pl.Series(tr)), cols), df["y"].to_numpy()[tr]
        m = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=cols, free_raw_data=True), ROUNDS)
        te = fold == k
        oof[te] = m.predict(to_np(df.filter(pl.Series(te)), cols), num_threads=PARAMS["num_threads"])
        m.save_model(f"{MODEL}/lgb_fold{k}.txt")  # fold models: test-time stacking uses their average
        print(f"fold {k} done {time.time()-t:.0f}s", flush=True)
    pred = df.select("o_id", "s1_id").with_columns(pl.Series("p", oof))
    pred.write_parquet(f"{MODEL}/oof.parquet")
    best = assign(pred)
    cand_recall = df.filter(pl.col("y") == 1).height / pairs.height
    print(f"candidate recall ceiling {cand_recall:.4f}", flush=True)
    res = [(tau, macro_f05(best, tau, pairs, all_s1)) for tau in np.arange(0.2, 0.95, 0.05)]
    for tau, f in res:
        print(f"  tau={tau:.2f}  macroF0.5={f:.5f}")
    tau = max(res, key=lambda x: x[1])[0]
    open(f"{MODEL}/threshold.txt", "w").write(f"{tau:.3f}\n")

    tr = fit_mask
    m = lgb.train(PARAMS, lgb.Dataset(to_np(df.filter(pl.Series(tr)), cols), df["y"].to_numpy()[tr], feature_name=cols), ROUNDS)
    m.save_model(f"{MODEL}/lgb.txt")
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])[:25]
    print("top features:", ", ".join(f"{c}={v:.0f}" for c, v in imp))
    print(f"best tau {tau:.2f}, total {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
