#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=24
PY=.venv/bin/python
$PY -W ignore src/ce2.py train > logs/ce2_train.log 2>&1
$PY -W ignore src/ce2.py score holdout > logs/ce2_score_holdout.log 2>&1
$PY -W ignore src/ce2.py score test > logs/ce2_score_test.log 2>&1
echo CE2_DONE >> logs/ce2_score_test.log
