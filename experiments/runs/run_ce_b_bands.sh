#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=24 CE_FOLD=1 CE_DIR=work/ce/model_b
PY=.venv/bin/python
for b in band_hi band_lo; do for sp in train test; do CE_BAND=$b CE_TAG=_${b#band_}_b $PY -W ignore src/ce_train.py score $sp >> logs/ceb_bands.log 2>&1; done; done
echo CEB_BANDS_DONE >> logs/ceb_bands.log
