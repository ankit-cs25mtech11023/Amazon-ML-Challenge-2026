#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=32
PY=.venv/bin/python
$PY -W ignore src/selftrain_fr.py train > logs/st_train.log 2>&1
$PY -W ignore src/selftrain_fr.py score > logs/st_score.log 2>&1
W=0.5 TAU=0.75 CE2_TEST_SCORED=work/ce2/test_scored_fr.parquet OUT_DIR=output/v13_fr_st $PY -W ignore -c "
import sys; sys.path.insert(0, \"src\")
import ensemble; ensemble.W = []; ensemble.main()
" > logs/v13_build.log 2>&1
echo V13_DONE >> logs/v13_build.log
