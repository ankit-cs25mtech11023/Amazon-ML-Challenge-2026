#!/bin/bash
export CUDA_VISIBLE_DEVICES=2 HF_HOME=/tmp/amlc_aayush/hf POLARS_MAX_THREADS=16 LLM_SCORE_BS=96
.venv/bin/python -W ignore src/llm_match.py score test2 > logs/llm_score_test2.log 2>&1
echo LLM2_DONE >> logs/llm_score_test2.log
