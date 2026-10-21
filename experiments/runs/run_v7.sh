#!/bin/bash
# v6 = v5 with calibration-consistent stacking (test stage-1 p = mean of the OOF fold models)
set -e
export POLARS_MAX_THREADS=64 THREADS=64 CUDA_VISIBLE_DEVICES=2
PY=.venv/bin/python
export FEAT_DIR=work/feat_v4 MODEL1_DIR=work/model_v6s1 MODEL2_DIR=work/model_v7 EMB_DIR=work/emb_ftb
EXCLUDE_FT=1 $PY -W ignore src/stage2.py train > logs/v7_train.log 2>&1
OUT_DIR=output/v7 $PY -W ignore src/stage2.py test > logs/v7_test.log 2>&1
$PY student_resource/utils/validate_submission.py --matching output/v7/matching_results.tsv --candidate output/v7/candidate_pairs.tsv --test-dir student_resource/dataset/test >> logs/v7_test.log 2>&1
echo V7_DONE >> logs/v7_test.log
