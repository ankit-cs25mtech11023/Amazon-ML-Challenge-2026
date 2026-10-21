#!/bin/bash
# queued after France round 2 (v14): LLM LoRA fine-tune + scoring on GPU 2
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=24 HF_HOME=/tmp/amlc_aayush/hf
PY=.venv/bin/python
until grep -q V14_DONE logs/v14_build.log 2>/dev/null; do sleep 30; done
until grep -q DOWNLOADED logs/qwen_download.log 2>/dev/null; do sleep 30; done
set -e
$PY -W ignore src/llm_match.py train > logs/llm_train.log 2>&1
$PY -W ignore src/llm_match.py score holdout > logs/llm_score_holdout.log 2>&1
$PY -W ignore src/llm_match.py score test > logs/llm_score_test.log 2>&1
echo LLM_DONE >> logs/llm_score_test.log
