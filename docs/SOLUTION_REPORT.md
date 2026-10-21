# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** moov  
**Team Members:** Akshay Bagde, Ankit Kr Sinha, Aayush Ranjan, Apoorv Singh  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We treat the task as an **assignment problem**: every Source 2/3 record belongs to at most one Source 1 entity, so each
record is assigned to its single best Source 1 candidate or to nobody. Candidates come from IDF-weighted token search and
**contrastively fine-tuned multilingual sentence encoders** (99.2% recall), are narrowed by a **learned blocking filter to
6.1 candidates per Source 1 entity**, and are scored by a LightGBM matcher, fine-tuned **multilingual cross-encoders** and a
**LoRA-fine-tuned 7B LLM**. The unlabeled country (France) is handled by **self-training on unlabeled test records**.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the 2.2M training entities (7.6M true pairs):
- **No Source 2/3 record matches more than one Source 1 entity** → exclusive assignment.
- ~26% of Source 2/3 records match nothing (distractors); 5.6% of Source 1 entities have no match (singletons).
- A true pair never crosses countries → country is a hard partition.
- **48% of Source 1 name keys are shared by another entity** (chains, common names), so names alone cannot identify an
  entity. Names are often replaced: Indic scripts (18% of Indian Source 2/3 names — Devanagari, Tamil, Telugu, Kannada,
  Gujarati, Bengali, Malayalam), URLs, social handles, initials, random strings.
- **The address is the backbone**: 99.97% of true pairs share an address token (vs. 85% sharing a name token).
- House-number conflict occurs in 6% of true pairs vs. 66% of non-matches (decoys on the same street).
- Records with an empty address are matched 97.7% of the time.
- France appears only in test (15% of test entities, 3 regions / 4 departments / ~15 cities). Source 1 writes the region,
  Sources 2/3 often the department.
- No leakage in IDs or row order.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — multi-retriever blocking + learned blocking filter + GBDT matcher + transformer cross-encoders
+ 7B LLM matcher + self-training + exclusive assignment.  
**Core Innovation:** (1) fine-tuned multilingual retrieval that handles scripts, typos and replaced names; (2) layered
matchers (GBDT → cross-encoders → LLM) applied where each is most useful; (3) self-training to adapt the matcher to a
country with no labels, using only unlabeled test records.

---

## 3. Candidate Generation (Blocking)

**Normalization.** Unicode NFKD, accent stripping, transliteration of any script to Latin, legal forms split into a separate
field (Pvt/Private, Ltd/Limited, LLC, Inc, Corp, LLP, SARL, SAS, EURL, SCI, …), address abbreviations (St/Street, Rd/Road,
Ave, Blvd, Nagar, Colony, Near, Opp, R./Rue, Av., Bd, Imp., Allée, …), US/Indian state names → abbreviations, French
region/department names removed, "N°" and St/Ste → Saint/Sainte, house numbers without leading zeros, 6-digit PIN codes,
no-space name, consonant-class sound keys.

**Stage A — retrieval (two complementary retrievers, per country):**
1. *IDF token search* — prefixed tokens (address, name, sound keys, no-space name); tokens in >1% of Source 1 dropped;
   top-30 Source 1 entities per record by IDF-weighted cosine (sparse matrix product).
2. *Fine-tuned dense search* — `multilingual-e5-small` / `-base` fine-tuned with an in-batch-negative contrastive loss on
   true pairs of 25% of train entities (country-pure batches); exact GPU top-20 cosine search.

| Retriever (held-out entities) | @1 | @5 | @10 | @20 |
| --- | --- | --- | --- | --- |
| Token IDF | 93.1% | 96.3% | 96.9% | 97.5% |
| Fine-tuned multilingual-e5 | 97.3% | 98.7% | 99.0% | 99.3% |

The union of token top-10 and embedding top-10 reaches **99.24% recall** of true pairs.

**Stage B — learned blocking filter.** A LightGBM scorer on retrieval and string features keeps, for each Source 1 entity,
at most its **12** highest-scoring records and only pairs with score ≥ 0.001 (no train entity has more than 11 matches).

- **Blocking keys used:** IDF-weighted name/address/number/sound-key tokens; fine-tuned multilingual dense embeddings;
  country as a hard partition; learned blocking score.
- **Candidate pairs generated:** **10.6M test pairs = 6.1 candidates per Source 1 entity (median 6, max 12)** —
  exactly the set scored by the matchers and contained in `candidate_pairs.tsv`.
- **How we ensured true matches were not lost:** two complementary retrievers (only ~11% of pairs found by both), union
  recall 99.24%; the learned filter only removes pairs the matchers would not accept (it changes 0.06% of final matches
  compared with scoring every retrieved pair).

---

## 4. Matching Model

**Features used (~70, all country-agnostic):**
- *Name:* ratio, token-sort, token-set, partial ratio, Jaro-Winkler on cleaned names; ratio/partial on no-space names;
  token and sound-key Jaccard/containment; legal-form equal/conflict/empty; name frequency in Source 1; length difference.
- *Address:* ratio, token-sort, token-set, partial-token-set; token Jaccard and containment both ways; house-number overlap,
  first-number equality, number conflict; PIN equality/conflict; length ratio; empty-address flags.
- *Retrieval / competition context:* token score and rank, cosine in two embedding spaces and ranks, the record's best and
  second-best scores and the gap to the best competing entity, the entity's rank among records that retrieved it.

**Model type (layered):**
1. **LightGBM** (binary, 255 leaves, 700 rounds), 2-fold grouped out-of-fold training; fine-tuning entities excluded.
2. **Cross-encoders** — `BAAI/bge-reranker-v2-m3` (568M) fine-tuned on raw `"name | address | country"` pairs (BCE, bf16):
   one per entity fold on uncertain / confident / low-probability bands, and one on 2.85M pairs from both folds scoring
   every final candidate. Blended with logistic regressions on [logit(p_LightGBM), cross-encoder logit, gap to the best
   other candidate, is-top, #candidates], fitted on held-out entities; the two blends are averaged.
3. **Self-training for the unlabeled country.** France has no training labels. The ensemble's most confident French test
   pairs become pseudo-labels (p ≥ 0.97 top choices as matches; their losing competitors and p ≤ 0.03 as non-matches,
   400k each), mixed with 300k training pairs as replay; the cross-encoder is fine-tuned further and all French
   candidates are re-scored (two rounds). The same procedure is applied to US and India on their uncertain candidates.
   Only unlabeled test records and the pipeline's own predictions are used — no ground truth, no external data.
4. **LLM matcher** — `Qwen/Qwen2.5-7B-Instruct` (7.6B, Apache-2.0) as a sequence classifier with LoRA (r=16, α=32, all
   attention and MLP projections, 40M trainable parameters), 1 epoch on 250k hard training pairs (bf16, gradient
   checkpointing). Prompt: *"Do these two records describe the same real-world business? Record A: … Record B: …"*.
   It scores the 1.36M test pairs with ensemble probability in [0.005, 0.995]; its logit, its gap to the record's best
   other candidate and an is-top flag are blended with the ensemble probability (logistic regression on held-out entities).

**Threshold selection method:** exclusive assignment (each Source 2/3 record → only its highest-probability Source 1
entity), kept if the final probability ≥ τ. τ = 0.75 for US/India, chosen by maximising macro F0.5 (including
singletons) on held-out entities. France has no labels, so its threshold is calibrated separately (on the public
leaderboard) and applied as a fixed number of French matches.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9911 on held-out train entities (US/India); public leaderboard 0.9879.
- **Common false positives (wrong merges):** same-street decoys with a nearby house number (609 vs 611); chain names
  shared by many entities in one city; records whose name was replaced by a generic or random string and matched on
  address alone.
- **Common false negatives (missed matches):** records with an **empty address** and a common name (candidate recall
  84% for these vs. 99.9% for records with an address — the name alone cannot say which chain branch they belong to);
  heavily truncated addresses; native-script names with abbreviated addresses.

---

## 6. Conclusion
Treating the task as exclusive assignment and building blocking around the address, with fine-tuned multilingual
retrieval, takes candidate recall to 99.2% while a learned filter keeps only 6.1 candidates per entity. Layering
cross-encoders and a 7B LLM on top of a feature-based LightGBM, and adapting the matcher to the unseen country with
self-training on unlabeled test records, gave the largest gains. Lesson: keep features country-agnostic and use
unlabeled data to adapt to distribution shift.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source in `src/`, entry point `run_all.sh` (raw TSVs →
`output/matching_results.tsv` + `output/candidate_pairs.tsv`, then runs the official validator). `README.md` lists every
stage; `requirements.txt` pins versions (Python 3.11, torch 2.6.0 + CUDA 12.4).

| Stage | Script |
| --- | --- |
| Labels, normalization | `prep_labels.py`, `normalize.py` |
| Retrieval | `candidates.py`, `finetune_embed.py`, `embed.py`, `merge_cands.py` |
| Features, LightGBM | `features.py`, `train.py` |
| Cross-encoders, blocking filter, blends | `ce_data.py`, `ce_train.py`, `ce_blend.py`, `ce2.py`, `ce2_blend.py`, `ensemble.py` |
| Self-training | `selftrain_fr.py` |
| LLM matcher, final decision | `llm_match.py`, `llm_blend.py`, `predict.py` |

Models (all MIT / Apache-2.0, ≤ 8B parameters): `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`,
`BAAI/bge-reranker-v2-m3`, `Qwen/Qwen2.5-7B-Instruct`, LightGBM. No external data, APIs or lookups; the only hand-written
resources are abbreviation, legal-form and state/region tables.

### B. Additional Results
- Candidate recall: token retrieval 96.95% (top-10) → with fine-tuned embeddings 99.24%; final candidate set 6.1 per
  Source 1 entity.
- Hardware: 96-core Xeon, 251 GB RAM, one RTX 6000 Ada (48 GB).
