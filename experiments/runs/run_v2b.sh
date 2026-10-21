#!/bin/bash
set -e
export POLARS_MAX_THREADS=64 THREADS=64 PROCS=64 CUDA_VISIBLE_DEVICES=2
PY=.venv/bin/python
until grep -q "^done" logs/embed_test.log; do sleep 10; done
$PY -W ignore src/merge_cands.py train                 > logs/v2_merge.log 2>&1
$PY -W ignore src/merge_cands.py test                  >> logs/v2_merge.log 2>&1
rm -rf work/feat/train work/feat/test
NPARTS=12 $PY -W ignore src/features.py train          > logs/v2_feat_train.log 2>&1
NPARTS=12 $PY -W ignore src/features.py test           > logs/v2_feat_test.log 2>&1
MODEL_DIR=work/model_v2 ROUNDS=600 TRAIN_FRAC=0.3 $PY -W ignore src/train.py > logs/v2_train.log 2>&1
MODEL_DIR=work/model_v2 OUT_DIR=output/v2 $PY -W ignore src/predict.py > logs/v2_predict.log 2>&1
echo V2_DONE >> logs/v2_predict.log
