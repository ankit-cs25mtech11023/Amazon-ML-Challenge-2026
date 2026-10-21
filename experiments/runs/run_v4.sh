#!/bin/bash
# v4: token top-10 U fine-tuned e5-base top-10; cosine from ft-base (escore) and ft-small (escore2)
set -e
export POLARS_MAX_THREADS=64 THREADS=64 CUDA_VISIBLE_DEVICES=2
PY=.venv/bin/python
for s in train test; do
  FEAT_DIR=work/feat_v4 CAND_KIND=union4 NPARTS=12 $PY -W ignore src/features.py $s > logs/v4_feat_$s.log 2>&1
done
FEAT_DIR=work/feat_v4 MODEL_DIR=work/model_v4 EXCLUDE_FT=1 ROUNDS=700 TRAIN_FRAC=0.35 $PY -W ignore src/train.py > logs/v4_train.log 2>&1
FEAT_DIR=work/feat_v4 MODEL_DIR=work/model_v4 OUT_DIR=output/v4 $PY -W ignore src/predict.py > logs/v4_predict.log 2>&1
echo V4_DONE >> logs/v4_predict.log
