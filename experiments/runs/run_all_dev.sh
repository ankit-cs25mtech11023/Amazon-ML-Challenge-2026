#!/bin/bash
# End-to-end reproduction: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
# Run from the directory that contains student_resource/ (the challenge data) and this repo's src/.
# Needs one CUDA GPU (>= 24 GB), ~200 GB RAM, ~150 GB free disk. Total ~4-5 h on 64 cores + RTX 6000 Ada.
set -e
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-64} THREADS=${THREADS:-64} PROCS=${PROCS:-64}
PY=${PY:-python}
mkdir -p logs work output
$PY -W ignore src/prep_labels.py
$PY -W ignore src/normalize.py                                     # stage 0: normalization
for s in train test; do $PY -W ignore src/candidates.py $s; done   # stage 1a: IDF token retrieval (top-30)
# stage 1b: contrastive fine-tuning of two multilingual encoders on 25% of train entities
EMB_MODEL=intfloat/multilingual-e5-small FT_DIR=work/emb_model_ft      BS=384 $PY -W ignore src/finetune_embed.py
EMB_MODEL=intfloat/multilingual-e5-base  FT_DIR=work/emb_model_ft_base BS=256 $PY -W ignore src/finetune_embed.py
for s in train test; do
  EMB_MODEL=work/emb_model_ft      EMB_DIR=work/emb_ft  EMB_TAG=embft  $PY -W ignore src/embed.py $s
  EMB_MODEL=work/emb_model_ft_base EMB_DIR=work/emb_ftb EMB_TAG=embftb $PY -W ignore src/embed.py $s
  # stage 1c: union of token top-10 and fine-tuned-e5-base top-10, exact cosine in both embedding spaces
  ETAG=embftb EMB_DIR=work/emb_ftb EMB2_DIR=work/emb_ft OUTTAG=union4 $PY -W ignore src/merge_cands.py $s
  FEAT_DIR=work/feat_v4 CAND_KIND=union4 NPARTS=12 $PY -W ignore src/features.py $s   # stage 2: pair features
done
# stage 3: LightGBM matcher (records owned by fine-tuning entities excluded to avoid leakage); saves fold models
FEAT_DIR=work/feat_v4 MODEL_DIR=work/model_v6s1 EXCLUDE_FT=1 ROUNDS=700 TRAIN_FRAC=0.35 $PY -W ignore src/train.py
# stage 4: cross-encoder re-scoring (bge-reranker-v2-m3) of mid / confident / low bands
export MODEL1_DIR=work/model_v6s1 FEAT_DIR=work/feat_v4 CAP_BY=p1 CAND_CAP=12 PMIN_BLOCK=0.001
for sp in train test; do
  PLO=0.02  PHI=0.98 TOPN=3 BAND_TAG=band    $PY -W ignore src/ce_data.py $sp
  PLO=0.98  PHI=1.01 TOPN=1 BAND_TAG=band_hi $PY -W ignore src/ce_data.py $sp
  PLO=0.005 PHI=0.02 TOPN=3 BAND_TAG=band_lo $PY -W ignore src/ce_data.py $sp
done
CE_FOLD=0 CE_DIR=work/ce/model $PY -W ignore src/ce_train.py train       # trained on entity fold 0
for b in band band_hi band_lo; do
  t=${b#band}; for sp in train test; do CE_FOLD=0 CE_DIR=work/ce/model CE_BAND=$b CE_TAG=$t $PY -W ignore src/ce_train.py score $sp; done
done
# second cross-encoder, trained on entity fold 1: scores fold-0 train pairs and the test mid band (2-model average)
CE_FOLD=1 CE_DIR=work/ce/model_b $PY -W ignore src/ce_train.py train
for sp in train test; do CE_FOLD=1 CE_DIR=work/ce/model_b CE_BAND=band CE_TAG=_b $PY -W ignore src/ce_train.py score $sp; done
# stage 5: per-band logistic blend, exclusive assignment, F0.5-tuned threshold, write + validate outputs
# final blocking = learned filter: per S1 entity keep top-12 records by LightGBM probability, and only p >= 0.001
OUT_DIR=output/ref_v10c CE_TWO=1 TAU=0.70 TAU_FR=0.85 CAP_BY=p1 CAND_CAP=12 PMIN_BLOCK=0.001 CE_COMP=1 CE_BANDS=",_hi,_lo" $PY -W ignore src/ce_blend.py
# stage 6: cross-encoder v2 (both folds, 10% entity holdout) scores every final candidate
$PY -W ignore src/ce2.py data
$PY -W ignore src/ce2.py train
$PY -W ignore src/ce2.py score holdout
$PY -W ignore src/ce2.py score test
# stage 7: ensemble (0.5 * per-band blend + 0.5 * cross-encoder-v2 blend); France match count calibrated to ref_v10c
W=0.5 TAU=0.75 FR_MATCH_REF=output/ref_v10c/matching_results.tsv SAVE_PRED=work/ce2/ens_test_pred.parquet OUT_DIR=output/ens \
  $PY -W ignore -c "import sys; sys.path.insert(0, 'src'); import ensemble; ensemble.W = []; ensemble.main()"
# stage 8: self-training for France (no labels in train): pseudo-labels from the ensemble's confident French predictions,
#          continue fine-tuning cross-encoder v2, re-score French candidates, rebuild the ensemble -> final outputs
ST_COUNTRY=France $PY -W ignore src/selftrain_fr.py data
ST_COUNTRY=France $PY -W ignore src/selftrain_fr.py train
ST_COUNTRY=France $PY -W ignore src/selftrain_fr.py score
W=0.5 TAU=0.75 FR_MATCH_REF=output/ref_v10c/matching_results.tsv CE2_TEST_SCORED=work/ce2/test_scored_fr.parquet \
  SAVE_PRED=work/ce2/ens_test_pred_r2.parquet OUT_DIR=output/v13 \
  $PY -W ignore -c "import sys; sys.path.insert(0, 'src'); import ensemble; ensemble.W = []; ensemble.main()"
# stage 9: France self-training round 2 (pseudo-labels from the round-1-adapted ensemble, continue from round-1 model)
export ST_PRED=work/ce2/ens_test_pred_r2.parquet ST_MIX=work/ce2/selftrain_fr_r2.parquet ST_SCORED_OUT=work/ce2/test_scored_fr2.parquet \
       CE2_DIR=work/ce2/model_fr CE2_FR_DIR=work/ce2/model_fr2 NPOS=400000 NNEG=400000
for st in data train score; do ST_COUNTRY=France $PY -W ignore src/selftrain_fr.py $st; done
unset ST_PRED ST_MIX ST_SCORED_OUT CE2_DIR CE2_FR_DIR NPOS NNEG
W=0.5 TAU=0.75 FR_MATCH_REF=output/ref_v10c/matching_results.tsv CE2_TEST_SCORED=work/ce2/test_scored_fr2.parquet \
  SAVE_PRED=work/ce2/ens_test_pred_v14.parquet OUT_DIR=output/v14 \
  $PY -W ignore -c "import sys; sys.path.insert(0, 'src'); import ensemble; ensemble.W = []; ensemble.main()"
# stage 10: LLM matcher — Qwen2.5-7B-Instruct (Apache-2.0, 7.6B) + LoRA on 250k hard pairs, scores uncertain pairs
export HF_HOME=${HF_HOME:-work/hf} LLM_TEST_PRED=work/ce2/ens_test_pred_r2.parquet
$PY -W ignore src/llm_match.py data
$PY -W ignore src/llm_match.py train
$PY -W ignore src/llm_match.py score holdout
$PY -W ignore src/llm_match.py score test
# stage 11: LLM also scores a wider uncertainty band (p in [0.005, 0.995], 663k more pairs)
LLM_WIDE_PRED=work/ce2/ens_test_pred_v14.parquet $PY -W ignore src/llm_match.py data_wide
$PY -W ignore src/llm_match.py score test2
# stage 12: self-training for US + India (pseudo-labels from the ensemble's confident predictions), re-score their
#           uncertain candidates (p in [0.005, 0.995]) on top of the France-adapted scores
export ST_COUNTRY="US,India" ST_PRED=work/ce2/ens_test_pred_v14.parquet ST_MIX=work/ce2/selftrain_usin.parquet \
       CE2_DIR=work/ce2/model CE2_FR_DIR=work/ce2/model_usin ST_BASE_SCORED=work/ce2/test_scored_fr2.parquet \
       ST_SCORED_OUT=work/ce2/test_scored_usin.parquet ST_SCORE_BAND=0.005,0.995 NPOS=400000 NNEG=400000
for st in data train score; do $PY -W ignore src/selftrain_fr.py $st; done
unset ST_COUNTRY ST_PRED ST_MIX CE2_DIR CE2_FR_DIR ST_BASE_SCORED ST_SCORED_OUT ST_SCORE_BAND NPOS NNEG
W=0.5 TAU=0.75 TAU_FR=0.8893 CE2_TEST_SCORED=work/ce2/test_scored_usin.parquet SAVE_PRED=work/ce2/ens_test_pred_v17.parquet \
  OUT_DIR=output/v17_ens $PY -W ignore -c "import sys; sys.path.insert(0, 'src'); import ensemble; ensemble.W = []; ensemble.main()"
# stage 13: blend both LLM score sets into the ensemble (fitted on holdout), France count calibrated -> final outputs
LLM_TEST_FILES=work/llm/test_scored.parquet,work/llm/test2_scored.parquet ENS_PRED=work/ce2/ens_test_pred_v17.parquet \
  FR_MATCH_REF=output/v13/matching_results.tsv OUT_DIR=output TAU=0.75 $PY -W ignore src/llm_blend.py
