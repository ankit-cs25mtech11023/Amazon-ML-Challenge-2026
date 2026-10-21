# Business Entity Resolution — Amazon ML Challenge 2026

Team **moov** — Akshay Bagde, Ankit Kr Sinha, Aayush Ranjan, Apoorv Singh.
**Top 100 of 32,000+ teams.** Macro F0.5: **0.9911** on held-out train entities, **0.9879** on the public leaderboard.

## Task
For every Source 1 business (name, address, country), find all records in Source 2 / Source 3 that refer to the same
real-world business. A business can have zero matches. Train data covers US and India; test also has **France, with no
labels**. Data size: 2.2M Source 1 entities and about 10M Source 2/3 records per split. Metric: macro F0.5 per Source 1
entity (precision-heavy). Only MIT/Apache-2.0 models up to 8B parameters were allowed, and no external data or APIs.

## Key findings from the data
- No Source 2/3 record matches more than one Source 1 entity, so the task becomes **exclusive assignment**.
- The **address is the backbone**: 99.97% of true pairs share an address token. Names are often replaced, written in
  Indic scripts, or shared by chains (48% of Source 1 name keys are not unique).
- About 26% of Source 2/3 records match nothing (distractors). True pairs never cross countries.

## Pipeline
```
normalize -> retrieve (IDF tokens + fine-tuned multilingual-e5) -> learned blocking filter (~6 candidates / entity)
          -> LightGBM (~70 features) -> bge-reranker-v2-m3 cross-encoders -> ensemble
          -> self-training on unlabeled test data (France, then US/India)
          -> Qwen2.5-7B + LoRA on uncertain pairs -> exclusive assignment + F0.5-tuned threshold
```

| Stage | What it does | Result |
| --- | --- | --- |
| Normalization | transliteration to Latin, legal-form split, address abbreviations, state/region canonicalization, house numbers, PIN codes, sound keys | — |
| Retrieval | IDF token search ∪ contrastively fine-tuned `multilingual-e5-small/base` (exact GPU kNN) | **99.24% recall** |
| Blocking filter | LightGBM score, top-12 per entity, p ≥ 0.001 | 6.1 candidates per entity |
| Matcher | LightGBM, 2-fold grouped out-of-fold | — |
| Cross-encoders | `BAAI/bge-reranker-v2-m3` fine-tuned on pairs, per-band logistic blends | — |
| Self-training | pseudo-labels from confident test predictions + train replay (France 2 rounds, US/India 1) | handles the unlabeled country |
| LLM matcher | `Qwen2.5-7B-Instruct` + LoRA (r=16) on 250k hard pairs, scores 1.36M uncertain pairs | — |
| Decision | each record goes to its best entity or none; τ = 0.75 (France calibrated separately) | F0.5 0.991 / 0.988 |

The full write-up (features, error analysis, thresholds) is in [docs/SOLUTION_REPORT.md](docs/SOLUTION_REPORT.md).

## Repository layout
```
src/                 final pipeline (one script per stage, see the table in docs/SOLUTION_REPORT.md)
run_all.sh           end-to-end run: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
requirements.txt     pinned versions (Python 3.11, torch 2.6.0 + CUDA 12.4)
docs/
  SOLUTION_REPORT.md submitted solution documentation
  PLAN.md            data findings and plan made during the challenge
  CHALLENGE_NOTES.md rules, file sizes, submission format
experiments/
  runs/              scripts for each leaderboard version (v2 ... v17, cross-encoder and LLM runs)
  analysis/          recall checks, error analysis, France diagnostics, listwise / meta-stacking trials
  eda/               early data exploration and blocking experiments
```
Scripts in `experiments/` were written against earlier versions of `src/` and are kept for reference; `run_all.sh`
is the one that reproduces the final submission.

## How to run
```bash
# 1. put the challenge data next to src/:  student_resource/dataset/{train,test}/*.tsv
#                                          student_resource/utils/validate_submission.py
# 2. environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# 3. full pipeline (writes output/ and runs the official validator)
CUDA_VISIBLE_DEVICES=0 PY=python bash run_all.sh
```
Hardware used: 96-core Xeon, 251 GB RAM, one NVIDIA RTX 6000 Ada (48 GB). Intermediate files go to `work/`.

## Not included
The dataset, intermediate files, trained models and submission TSVs are not in this repo (several GB, and the data
belongs to the challenge organisers).

## Models used
`intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`, `BAAI/bge-reranker-v2-m3`,
`Qwen/Qwen2.5-7B-Instruct` (all MIT / Apache-2.0), LightGBM.
