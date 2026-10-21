#!/bin/bash
# listwise LLM: waits until US/India re-scoring is done (GPU free), smoke test, then train + score
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=16 HF_HOME=/tmp/amlc_aayush/hf
PY=.venv/bin/python
until grep -q LLM2_DONE logs/llm_score_test2.log 2>/dev/null; do sleep 20; done
# smoke: 64 examples, 8 steps
.venv/bin/python -c "import polars as pl; pl.read_parquet('work/lw/train.parquet').head(64).write_parquet('work/lw/smoke_train.parquet')"
cp work/lw/train.parquet work/lw/train_full.parquet
cp work/lw/smoke_train.parquet work/lw/train.parquet
LW_DIR=work/lw/smoke LW_TRAIN_N=64 $PY -W ignore src/listwise.py train > logs/lw_smoke.log 2>&1 || { echo SMOKE_FAILED >> logs/lw_smoke.log; cp work/lw/train_full.parquet work/lw/train.parquet; exit 1; }
cp work/lw/train_full.parquet work/lw/train.parquet
set -e
LW_TRAIN_N=40000 $PY -W ignore src/listwise.py train > logs/lw_train.log 2>&1
$PY -W ignore src/listwise.py score holdout > logs/lw_score_holdout.log 2>&1
$PY -W ignore src/listwise.py score test > logs/lw_score_test.log 2>&1
echo LW_DONE >> logs/lw_score_test.log
