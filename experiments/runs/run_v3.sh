#!/bin/bash
# v3: token top-10 U fine-tuned-embedding top-10, dual cosine features, fine-tune leakage guard
set -e
export POLARS_MAX_THREADS=64 THREADS=64 CUDA_VISIBLE_DEVICES=2
PY=.venv/bin/python
until grep -q "^done" logs/embft_test.log; do sleep 15; done
for s in train test; do
  ETAG=embft EMB_DIR=work/emb_ft EMB2_DIR=work/emb OUTTAG=union3 $PY -W ignore src/merge_cands.py $s >> logs/v3_merge.log 2>&1
done
for s in train test; do
  FEAT_DIR=work/feat_v3 CAND_KIND=union3 NPARTS=12 $PY -W ignore src/features.py $s > logs/v3_feat_$s.log 2>&1
done
FEAT_DIR=work/feat_v3 MODEL_DIR=work/model_v3 EXCLUDE_FT=1 ROUNDS=700 TRAIN_FRAC=0.35 $PY -W ignore src/train.py > logs/v3_train.log 2>&1
FEAT_DIR=work/feat_v3 MODEL_DIR=work/model_v3 OUT_DIR=output/v3 $PY -W ignore src/predict.py > logs/v3_predict.log 2>&1
echo V3_DONE >> logs/v3_predict.log
