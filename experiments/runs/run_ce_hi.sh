#!/bin/bash
# score confident accepts (p1>0.98, top-1) and low band (0.005-0.02, top-3) with cross-encoder A
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=24 CE_DIR=work/ce/model CE_FOLD=0
PY=.venv/bin/python
for sp in train test; do
  PLO=0.98 PHI=1.01 TOPN=1 BAND_TAG=band_hi $PY -W ignore src/ce_data.py $sp >> logs/ce_hi_data.log 2>&1
  PLO=0.005 PHI=0.02 TOPN=3 BAND_TAG=band_lo $PY -W ignore src/ce_data.py $sp >> logs/ce_hi_data.log 2>&1
done
for b in band_hi band_lo; do
  for sp in train test; do
    CE_BAND=$b CE_TAG=_${b#band_} $PY -W ignore src/ce_train.py score $sp >> logs/ce_hi_score.log 2>&1
  done
done
echo CEHI_DONE >> logs/ce_hi_score.log
