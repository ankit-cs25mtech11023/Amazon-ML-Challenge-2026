#!/bin/bash
set -e
export POLARS_MAX_THREADS=64 THREADS=64 CUDA_VISIBLE_DEVICES=2 FEAT_DIR=work/feat_v4 MODEL1_DIR=work/model_v4 MODEL2_DIR=work/model_v5 EMB_DIR=work/emb_ftb
PY=.venv/bin/python
EXCLUDE_FT=1 $PY -W ignore src/stage2.py train > logs/v5_train.log 2>&1
OUT_DIR=output/v5 $PY -W ignore src/stage2.py test > logs/v5_test.log 2>&1
$PY student_resource/utils/validate_submission.py --matching output/v5/matching_results.tsv --candidate output/v5/candidate_pairs.tsv --test-dir student_resource/dataset/test >> logs/v5_test.log 2>&1
echo V5_DONE >> logs/v5_test.log
