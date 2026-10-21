#!/bin/bash
# Cross-encoder: build test band (CPU) in parallel with fine-tuning (GPU 2), then score fold-1 train + test.
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=32
PY=.venv/bin/python
$PY -W ignore src/ce_data.py test > logs/ce_data_test.log 2>&1 &
DATA_PID=$!
$PY -W ignore src/ce_train.py train > logs/ce_train.log 2>&1
wait $DATA_PID
$PY -W ignore src/ce_train.py score train > logs/ce_score_train.log 2>&1
$PY -W ignore src/ce_train.py score test > logs/ce_score_test.log 2>&1
echo CE_DONE >> logs/ce_score_test.log
