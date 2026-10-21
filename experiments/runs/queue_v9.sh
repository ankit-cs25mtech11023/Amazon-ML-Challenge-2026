#!/bin/bash
until [ -f work/ce/test_ce_hi.parquet ]; do sleep 15; done
sleep 10
CE_COMP=1 CE_BANDS=",_hi" OUT_DIR=output/v9 POLARS_MAX_THREADS=24 .venv/bin/python -W ignore src/ce_blend.py > logs/v9_blend.log 2>&1
echo V9_DONE >> logs/v9_blend.log
