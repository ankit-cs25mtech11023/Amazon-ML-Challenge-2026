"""Stage 5: second-stage matcher using stage-1 probabilities and cluster consistency.

For every candidate (o, s1) with stage-1 probability p, adds:
  - probability context: rank of s1 among o's candidates, gap to o's runner-up, s1's claimed-record count / mass
  - cluster consistency: embedding similarity of o to the other records confidently assigned to s1 ("anchors")
Train: uses out-of-fold stage-1 probabilities (MODEL1/oof.parquet); 2-fold OOF eval like train.py.
Test : stage-1 model predicts p, then the stage-2 model decides.
Usage: python src/stage2.py train|test
"""
import glob
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(__file__))
from predict import id_lists  # noqa: E402
from train import PARAMS, assign, macro_f05, to_np  # noqa: E402

FEAT = os.environ.get("FEAT_DIR", "work/feat_v4")
M1 = os.environ.get("MODEL1_DIR", "work/model_v4")
M2 = os.environ.get("MODEL2_DIR", "work/model_v5")
EMB = os.environ.get("EMB_DIR", "work/emb_ftb")
NORM = os.environ.get("NORM_DIR", "work/norm")
PMIN = float(os.environ.get("PMIN", "0.01"))
ANCHOR_P = float(os.environ.get("ANCHOR_P", "0.5"))
ROUNDS = int(os.environ.get("ROUNDS", "500"))


def load_feats(split):
    return pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f"{FEAT}/{split}/part_*.parquet"))])


def add_context(d: pl.DataFrame, split: str) -> pl.DataFrame:
    t = time.time()
    d = d.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("o_id").alias("p_rank_o"),
        pl.col("p").sort(descending=True).slice(1, 1).first().over("o_id").fill_null(0).alias("p_second_o"),
        pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("p_rank_s"),
    ).with_columns(
        (pl.col("p") - pl.when(pl.col("p_rank_o") == 1).then(pl.col("p_second_o"))
         .otherwise(pl.col("p").max().over("o_id"))).alias("p_gap_o"),
    )
    best = d.filter(pl.col("p_rank_o") == 1).select("o_id", "s1_id", "p")
    claims = best.group_by("s1_id").agg(
        (pl.col("p") >= ANCHOR_P).sum().alias("s_n_claim"), pl.col("p").sum().alias("s_claim_mass"))
    d = d.join(claims, on="s1_id", how="left").with_columns(pl.col("s_n_claim").fill_null(0), pl.col("s_claim_mass").fill_null(0))
    # this row's own contribution removed (so a record doesn't count itself)
    own = (pl.col("p_rank_o") == 1)
    d = d.with_columns(
        (pl.col("s_n_claim") - (own & (pl.col("p") >= ANCHOR_P)).cast(pl.Int32)).alias("s_n_claim_other"),
        (pl.col("s_claim_mass") - pl.when(own).then(pl.col("p")).otherwise(0)).alias("s_claim_mass_other"),
    )
    # cluster consistency via embeddings: anchors = records confidently assigned to s1
    anchors = best.filter(pl.col("p") >= ANCHOR_P).select("s1_id", pl.col("o_id").alias("a_id"))
    ido = pl.concat([pl.read_parquet(f"{NORM}/{split}_source{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("r")
    rix = dict(zip(ido["entity_id"].to_list(), ido["r"].to_list()))
    q = d.select("o_id", "s1_id").with_row_index("row").join(anchors, on="s1_id").filter(pl.col("o_id") != pl.col("a_id"))
    EO = torch.from_numpy(np.concatenate([np.load(f"{EMB}/{split}_source2.npy"), np.load(f"{EMB}/{split}_source3.npy")])).cuda()
    oi = np.fromiter((rix[x] for x in q["o_id"].to_list()), dtype=np.int64, count=q.height)
    ai = np.fromiter((rix[x] for x in q["a_id"].to_list()), dtype=np.int64, count=q.height)
    cos = np.empty(q.height, dtype=np.float32)
    step = 1_000_000
    for i in range(0, q.height, step):
        a = torch.from_numpy(oi[i:i + step]).cuda(); b = torch.from_numpy(ai[i:i + step]).cuda()
        cos[i:i + step] = (EO[a] * EO[b]).float().sum(1).cpu().numpy()
    del EO
    torch.cuda.empty_cache()
    same_src = (q["o_id"].str.slice(0, 2) == q["a_id"].str.slice(0, 2)).to_numpy()
    q = q.select("row").with_columns(pl.Series("c", cos), pl.Series("same", same_src))
    agg = q.group_by("row").agg(pl.col("c").max().alias("anc_cos_max"), pl.col("c").mean().alias("anc_cos_mean"),
                                pl.col("c").min().alias("anc_cos_min"), pl.len().alias("anc_n"),
                                pl.col("c").filter(pl.col("same")).max().alias("anc_cos_same_src"),
                                pl.col("c").filter(~pl.col("same")).max().alias("anc_cos_other_src"))
    d = d.with_row_index("row").join(agg, on="row", how="left").drop("row").with_columns(pl.col("anc_n").fill_null(0))
    print(f"  context features {time.time()-t:.0f}s  anchor pairs {q.height}", flush=True)
    return d


def stage1_p(d, split):
    if split == "train":
        oof = pl.read_parquet(f"{M1}/oof.parquet")
        d = d.join(oof, on=["o_id", "s1_id"], how="left")
        miss = d.filter(pl.col("p").is_null())
        if miss.height:
            # records excluded from stage-1 training (fine-tune entities): no fold model ever saw them,
            # so the fold-model average is an honest probability -> keeps competition context complete
            ms = [lgb.Booster(model_file=f) for f in sorted(glob.glob(f"{M1}/lgb_fold*.txt"))]
            X = miss.select([pl.col(c).cast(pl.Float32) for c in ms[0].feature_name()]).to_numpy()
            pm = np.mean([m.predict(X, num_threads=64) for m in ms], axis=0)
            d = pl.concat([d.filter(pl.col("p").is_not_null()), miss.with_columns(pl.Series("p", pm.astype(np.float32)))])
            print(f"  filled stage-1 p for {miss.height} excluded rows with fold-model average", flush=True)
        return d
    # use the same fold models that produced the train OOF probabilities -> identical calibration train/test
    files = sorted(glob.glob(f"{M1}/lgb_fold*.txt")) or [f"{M1}/lgb.txt"]
    ms = [lgb.Booster(model_file=f) for f in files]
    X = d.select([pl.col(c).cast(pl.Float32) for c in ms[0].feature_name()]).to_numpy()
    p = np.mean([m.predict(X, num_threads=64) for m in ms], axis=0)
    print(f"  stage-1 test p from {len(ms)} model(s): {[os.path.basename(f) for f in files]}", flush=True)
    return d.with_columns(pl.Series("p", p))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    os.makedirs(M2, exist_ok=True)
    t = time.time()
    d = stage1_p(load_feats(split), split)
    n_all = d.height
    low = d.filter(pl.col("p") < PMIN).select("o_id", "s1_id", "p")  # untouched: stays rejected
    d = add_context(d, split)  # context uses all candidates (full competition)
    d = d.filter(pl.col("p") >= PMIN)
    print(f"{split}: {n_all} rows, stage-2 rows {d.height}, {time.time()-t:.0f}s", flush=True)

    if split == "train":
        pairs = pl.read_parquet("work/data/train_pairs.parquet").rename({"source1_entity_id": "s1_id", "m": "o_id"})
        d = d.join(pairs.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["o_id", "s1_id"], how="left").with_columns(pl.col("y").fill_null(0))
        grp = d.select("o_id").unique().join(pairs.select("o_id", pl.col("s1_id").alias("g")), on="o_id", how="left") \
               .with_columns(pl.coalesce("g", "o_id").hash(11).alias("gh"))
        d = d.join(grp.select("o_id", "gh"), on="o_id")
        all_s1 = pl.read_parquet(f"{NORM}/train_source1.parquet", columns=["entity_id"])["entity_id"]
        if os.environ.get("EXCLUDE_FT"):
            ft_o = pairs.filter(pl.col("s1_id").hash(11) % 4 == 0)["o_id"]
            d = d.filter(~pl.col("o_id").is_in(ft_o.implode()))
            low = low.filter(~pl.col("o_id").is_in(ft_o.implode()))
            pairs = pairs.filter(pl.col("s1_id").hash(11) % 4 != 0)
            all_s1 = all_s1.filter(all_s1.hash(11) % 4 != 0)
        cols = [c for c in d.columns if c not in ("o_id", "s1_id", "y", "gh")]
        fold = (d["gh"] % 2).to_numpy()
        oof = np.zeros(d.height, dtype=np.float32)
        y = d["y"].to_numpy()
        for k in (0, 1):
            tr, te = fold != k, fold == k
            m = lgb.train(PARAMS, lgb.Dataset(to_np(d.filter(pl.Series(tr)), cols), y[tr], feature_name=cols), ROUNDS)
            oof[te] = m.predict(to_np(d.filter(pl.Series(te)), cols), num_threads=64)
            print(f"fold {k} {time.time()-t:.0f}s", flush=True)
        pred = pl.concat([d.select("o_id", "s1_id").with_columns(pl.Series("p", oof)), low])
        best = assign(pred)
        stage1_best = assign(pl.concat([d.select("o_id", "s1_id", "p"), low]))
        res = [(tau, macro_f05(best, tau, pairs, all_s1)) for tau in np.arange(0.3, 0.9, 0.05)]
        base = max(macro_f05(stage1_best, tau, pairs, all_s1) for tau in (0.55, 0.6, 0.65))
        for tau, f in res:
            print(f"  tau={tau:.2f}  macroF0.5={f:.5f}")
        tau = max(res, key=lambda x: x[1])[0]
        print(f"stage-1 best {base:.5f} -> stage-2 best {max(r[1] for r in res):.5f} at tau {tau:.2f}", flush=True)
        open(f"{M2}/threshold.txt", "w").write(f"{tau:.3f}\n")
        m = lgb.train(PARAMS, lgb.Dataset(to_np(d, cols), y, feature_name=cols), ROUNDS)
        m.save_model(f"{M2}/lgb.txt")
        imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])[:15]
        print("top features:", ", ".join(f"{c}={v:.0f}" for c, v in imp), flush=True)
    else:
        m = lgb.Booster(model_file=f"{M2}/lgb.txt")
        tau = float(os.environ.get("TAU") or open(f"{M2}/threshold.txt").read())
        p2 = m.predict(d.select([pl.col(c).cast(pl.Float32) for c in m.feature_name()]).to_numpy(), num_threads=64)
        pred = pl.concat([d.select("o_id", "s1_id").with_columns(pl.Series("p", p2)), low])
        out = os.environ.get("OUT_DIR", "output/v5")
        os.makedirs(out, exist_ok=True)
        s1 = pl.read_parquet(f"{NORM}/test_source1.parquet", columns=["entity_id"])["entity_id"]
        best = assign(pred).filter(pl.col("p") >= tau)
        id_lists(pred, s1, "candidate_entity_ids").write_csv(f"{out}/candidate_pairs.tsv", separator="\t", quote_style="never")
        id_lists(best, s1, "matched_entity_ids").write_csv(f"{out}/matching_results.tsv", separator="\t", quote_style="never")
        print(f"tau={tau:.3f} matched pairs={best.height}", flush=True)


if __name__ == "__main__":
    main()
