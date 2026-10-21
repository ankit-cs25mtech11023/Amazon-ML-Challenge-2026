#!/bin/bash
# France self-training round 2: pseudo-labels from v13 (round-1-adapted) predictions, continue from model_fr
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=32
PY=.venv/bin/python
ENS="import sys; sys.path.insert(0, \"src\"); import ensemble; ensemble.W = []; ensemble.main()"
W=0.5 TAU=0.75 CE2_TEST_SCORED=work/ce2/test_scored_fr.parquet SAVE_PRED=work/ce2/ens_test_pred_r2.parquet OUT_DIR=output/v13_check $PY -W ignore -c "$ENS" > logs/r2_pred.log 2>&1
export ST_PRED=work/ce2/ens_test_pred_r2.parquet ST_MIX=work/ce2/selftrain_fr_r2.parquet ST_SCORED_OUT=work/ce2/test_scored_fr2.parquet CE2_DIR=work/ce2/model_fr CE2_FR_DIR=work/ce2/model_fr2 NPOS=400000 NNEG=400000
$PY -W ignore src/selftrain_fr.py data > logs/r2_data.log 2>&1
$PY -W ignore src/selftrain_fr.py train > logs/r2_train.log 2>&1
$PY -W ignore src/selftrain_fr.py score > logs/r2_score.log 2>&1
W=0.5 TAU=0.75 CE2_TEST_SCORED=work/ce2/test_scored_fr2.parquet OUT_DIR=output/v14_fr_r2 $PY -W ignore -c "$ENS" > logs/v14_build.log 2>&1
echo V14_DONE >> logs/v14_build.log
