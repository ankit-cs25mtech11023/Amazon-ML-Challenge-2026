#!/bin/bash
# v6 = v5 with calibration-consistent stacking (test stage-1 p = mean of the OOF fold models)
set -e
export POLARS_MAX_THREADS=64 THREADS=64 CUDA_VISIBLE_DEVICES=2
PY=.venv/bin/python
FEAT_DIR=work/feat_v4 MODEL_DIR=work/model_v6s1 EXCLUDE_FT=1 ROUNDS=700 TRAIN_FRAC=0.35 $PY -W ignore src/train.py > logs/v6_stage1.log 2>&1
export FEAT_DIR=work/feat_v4 MODEL1_DIR=work/model_v6s1 MODEL2_DIR=work/model_v6 EMB_DIR=work/emb_ftb
EXCLUDE_FT=1 $PY -W ignore src/stage2.py train > logs/v6_train.log 2>&1
OUT_DIR=output/v6 $PY -W ignore src/stage2.py test > logs/v6_test.log 2>&1
$PY student_resource/utils/validate_submission.py --matching output/v6/matching_results.tsv --candidate output/v6/candidate_pairs.tsv --test-dir student_resource/dataset/test >> logs/v6_test.log 2>&1
echo V6_DONE >> logs/v6_test.log
