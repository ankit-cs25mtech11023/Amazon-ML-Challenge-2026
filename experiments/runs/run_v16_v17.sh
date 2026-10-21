#!/bin/bash
# US/India self-training starts now (parallel to LLM scoring); v16 after LLM scoring; v17 after both
export CUDA_VISIBLE_DEVICES=2 POLARS_MAX_THREADS=24 HF_HOME=/tmp/amlc_aayush/hf
PY=.venv/bin/python
export ST_COUNTRY="US,India" ST_PRED=work/ce2/ens_test_pred_v14.parquet ST_MIX=work/ce2/selftrain_usin.parquet \
       CE2_DIR=work/ce2/model CE2_FR_DIR=work/ce2/model_usin ST_BASE_SCORED=work/ce2/test_scored_fr2.parquet \
       ST_SCORED_OUT=work/ce2/test_scored_usin.parquet ST_SCORE_BAND=0.005,0.995
$PY -W ignore src/selftrain_fr.py train > logs/usin_train.log 2>&1
until grep -q LLM2_DONE logs/llm_score_test2.log 2>/dev/null; do sleep 30; done
LLM_TEST_FILES=work/llm/test_scored.parquet,work/llm/test2_scored.parquet ENS_PRED=work/ce2/ens_test_pred_v14.parquet \
  FR_MATCH_REF=output/v13_fr_st/matching_results.tsv OUT_DIR=output/v16_llmwide TAU=0.75 $PY -W ignore src/llm_blend.py > logs/v16_build.log 2>&1
echo V16_DONE >> logs/v16_build.log
$PY -W ignore src/selftrain_fr.py score > logs/usin_score.log 2>&1
unset ST_COUNTRY ST_PRED ST_MIX CE2_DIR CE2_FR_DIR ST_BASE_SCORED ST_SCORED_OUT ST_SCORE_BAND
W=0.5 TAU=0.75 TAU_FR=0.8893 CE2_TEST_SCORED=work/ce2/test_scored_usin.parquet \
  SAVE_PRED=work/ce2/ens_test_pred_v17.parquet OUT_DIR=output/v17_ens \
  $PY -W ignore -c "import sys; sys.path.insert(0, 'src'); import ensemble; ensemble.W = []; ensemble.main()" > logs/v17_ens.log 2>&1
LLM_TEST_FILES=work/llm/test_scored.parquet,work/llm/test2_scored.parquet ENS_PRED=work/ce2/ens_test_pred_v17.parquet \
  FR_MATCH_REF=output/v13_fr_st/matching_results.tsv OUT_DIR=output/v17_all TAU=0.75 $PY -W ignore src/llm_blend.py > logs/v17_build.log 2>&1
echo V17_DONE >> logs/v17_build.log
