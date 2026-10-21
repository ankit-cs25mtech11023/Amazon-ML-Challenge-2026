#!/bin/bash
# second cross-encoder: trained on fold 1, scores fold 0 + test (for a 2-model ensemble and full out-of-fold blend)
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=32 CE_FOLD=1 CE_TAG=_b CE_DIR=work/ce/model_b
PY=.venv/bin/python
$PY -W ignore src/ce_train.py train > logs/ceb_train.log 2>&1
$PY -W ignore src/ce_train.py score train > logs/ceb_score_train.log 2>&1
$PY -W ignore src/ce_train.py score test > logs/ceb_score_test.log 2>&1
echo CEB_DONE >> logs/ceb_score_test.log
