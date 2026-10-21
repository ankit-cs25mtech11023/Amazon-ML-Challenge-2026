# Team moov — Entity Resolution Plan

Target: top of the leaderboard (currently #1 = 0.9859, #30 ≈ 0.974). Deadline 27 Sep, 11:59 PM IST.

## 1. What the data tells us (measured on train, 25 Sep)

| Finding | Number | Consequence |
| --- | --- | --- |
| Each S2/S3 record matches **at most one** S1 | 0 exceptions in 7.6M pairs | Treat as **assignment**: each S2/S3 record picks its single best S1 or "none". Huge precision lever. |
| S2/S3 records that match nothing (distractors) | ~26% | Need a strong "no match" decision, not just nearest neighbour. |
| S1 singletons (no match) | 5.6% | Each is worth a full 1.0 if we predict empty, 0 if we predict anything. |
| Matches per S1 | mostly 2–6, max 11 | Recall per entity matters too; don't stop at top-1. |
| Country always equal inside a true pair | 100% | Hard-block on country. France (test only) is ~15% of test S1. |
| S1 name keys shared by another S1 | 48% | **Name alone cannot identify an entity.** |
| True pairs sharing ≥1 name token | 85% (US 91%, India 76%) | Names are replaced by: Indic script (Hindi/Tamil/Telugu/Kannada/Gujarati/Bengali/Malayalam, 18% of India S2/S3), URLs (`hwsprime.com`), handles (`@dhanview`), initials (`CA`), or random strings (`Miratavoio`). |
| True pairs sharing ≥1 / ≥2 address tokens | 99.97% / 99.05% | **Address is the backbone.** |
| Hard decoys | e.g. Peters Automotive, 609 vs 611 Crossing Drive, Durham | House-number conflict must be a feature. |
| S2/S3 with empty address | 3.3%, of which 97.7% are matched | Empty address is a positive signal, not a negative one. |
| ID / row-order leakage | correlation ≈ 0 | None to exploit. |

Blocking experiments (100k S1 sample vs full pool):

| Candidate method | Candidates per S1 | Recall |
| --- | --- | --- |
| Rarest name token(s) | 400–900 | 0.55–0.57 |
| Rarest address tokens, share ≥2 | 290 | 0.83 |

Neither is enough alone. Plain key-sharing is too blunt, so candidates must be **ranked by weighted similarity** from several complementary retrievers.

## 2. Pipeline

```
normalize ─► candidate generation (3 retrievers, union) ─► pair features ─► LightGBM
          ─► (optional) cross-encoder rerank ─► assignment + F0.5-tuned thresholds ─► TSVs
```

### Stage 0 — Normalization (drives everything else)
- Unicode NFKD, strip accents, lowercase, punctuation → space, collapse whitespace.
- **Indic transliteration** to Latin (`indic-transliteration`, `unidecode` fallback), plus a phonetic skeleton so
  `डिजिटल लॉजिस्टिक्स` ≈ `digital logistics`. Map Indic legal words (`प्राइवेट लिमिटेड`, `प्रा. लि.`) and native-script
  state names (`उत्तर प्रदेश`, `ગુજરાત`) to Latin via a table **learned from train pairs** (provided data only).
- Canonicalize legal suffixes (pvt/private, ltd/limited, corp, llc, llp, inc; France: sarl, sas, eurl, sa, sci, cie)
  and keep "legal form" as a separate field.
- Address abbreviations: US (st, rd, ave, dr, ln, ct, blvd, hwy, ste/unit), India (no/h.no/door no, flr/floor,
  nr/near, opp, rd, ngr/nagar, cly/colony), **France (r./rue, av./avenue, bd/boulevard, imp./impasse, all./allée,
  pl./place, rte/route, chem./chemin, st/saint, bis/ter)**. State abbreviations (OH↔Ohio, TN↔Tamil Nadu, MH↔Maharashtra).
- Name variants: remove `.com`, `@`, `#`, brackets, phone numbers (`- 9091179558`); also a **no-space** form so
  `systemsgurumantraprojects` matches `Systems Gurumantra Projects`.
- Extract structured parts: house/unit numbers (strip leading zeros: `0061`→`61`), PIN/ZIP, street tokens, city, state.

### Stage 1 — Candidate generation (target ≥99% recall; this caps our score)
Main direction is **S2/S3 record → top-K S1** (the S1 pool is small at 2.2M, and each record has exactly one true S1).
Also run S1 → top-K S2/S3 and take the union. All steps are partitioned by country.

1. **Sparse TF-IDF, char 3–4-grams** on normalized address (+ name): cosine top-K via chunked sparse matmul on 96 cores.
   Robust to typos and reordering.
2. **Weighted token retrieval (BM25-style)** on address tokens + transliterated name tokens + house numbers and PIN.
   Cheap, and catches cases where char n-grams are diluted.
3. **Dense multilingual embeddings** (GPU 2) of `name | address`: `intfloat/multilingual-e5-small` or `-base` (MIT),
   or `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0). Exact brute-force kNN on the GPU in chunks
   (India: 0.8M × 4.7M × 384-d takes about 1–2 min in fp16). Handles Indic scripts and garbled names.
   **Fine-tune with a contrastive loss on train pairs** (MultipleNegativesRanking with hard negatives). Likely the
   single biggest recall win.
4. Exact-key passes: (name key, city), (house no. + street token, city), (no-space name), (PIN + name token).

Measure recall on the validation split after every change; the aim is the union reaching ≥99% with K ≈ 10–30 per record.
`candidate_pairs.tsv` = exactly the pairs the final model scores.

### Stage 2 — Pair features (~80)
- **Name**: Jaro-Winkler, Levenshtein ratio, token-set/sort ratio, partial ratio, no-space ratio, phonetic match,
  transliterated-name similarity, IDF-weighted token Jaccard, legal-form agree/conflict, name "garbled" flags
  (URL/handle/initials/non-Latin), name-key frequency in S1 (how common the name is).
- **Address**: IDF-weighted token Jaccard, char-TF-IDF cosine, **house-number exact / conflict / numeric distance**,
  PIN/ZIP match or conflict, city match, state match (after canonicalization), token containment both ways,
  missing-component flags, address length ratio.
- **Model scores**: embedding cosine (base + fine-tuned), retriever ranks from each of the 3 methods.
- **Assignment context** (key for precision): this S1's rank among the record's candidates, score gap to the
  best competing S1, the record's rank within the S1's candidates, number of candidates, the S1's best score,
  and whether the other S1s near it share the same name (chain-store ambiguity).
- Source (S2 vs S3). **Country is not a feature**; all features are country-agnostic so they carry over to France.

### Stage 3 — Matching model
- **LightGBM** binary classifier on candidate pairs, GroupKFold by S1 entity, trained on all 2.2M train S1
  (lab machine: 238 GB RAM, 96 cores).
- Second pass: recompute context features using stage-1 probabilities (e.g. "best competing probability") and
  retrain. Cheap, and usually a clear gain for assignment-style problems.
- Optional **cross-encoder reranker** (fine-tuned `xlm-roberta-base` / `multilingual-e5-base`, both MIT)
  on the uncertain pairs only (0.2 < p < 0.9), blended with LightGBM.

### Stage 4 — Decision (optimizes macro F0.5 directly)
- **Exclusivity**: each S2/S3 record goes only to its argmax S1, and only if p ≥ τ.
- τ tuned on validation for **macro F0.5**, possibly separate τ for S2 vs S3 and by candidate ambiguity.
- Per-S1 expected-F0.5 maximization: given calibrated probabilities, choose the subset that maximizes expected F0.5.
  This handles singletons naturally (empty set wins when every p is lowish).

### France (test only, no labels)
All features are country-agnostic, plus French abbreviation/legal-form dictionaries and multilingual embeddings.
Checks: compare score distributions for France vs US/India, and look at matched-count-per-S1 (train ≈3.5 on average).
Optional pseudo-labelling on high-confidence France pairs. **Grey area under the fair-play rule; team decides before we use it.**

## 3. Validation
Hold out 10% of train S1 entities (grouped). Their S2/S3 pool is the **full train pool**, so distractors are realistic.
A local scorer implements the exact macro F0.5, including singletons. Every submission is chosen by local score first.
Leaderboard uploads are used sparingly.

## 4. Compliance
- Only provided data plus hand-written abbreviation tables. No APIs, geocoding, or internet lookups.
- Pretrained models: MIT/Apache-2.0 only, all ≤ 600M params (limit is 8B). Licenses listed in the doc.
- Every run is reproducible from `code/business_entity_resolution/` with a pinned `requirements.txt`.
- `utils/validate_submission.py` must PASS before every upload.

## 5. Timeline

| When | Milestone |
| --- | --- |
| **25 Sep night** | Normalization v1, TF-IDF + token retrieval, LightGBM baseline, assignment + threshold. **First leaderboard submission.** |
| **26 Sep** | Embeddings (base, then fine-tuned), full feature set, second-pass context features, per-entity F0.5 decision. **Strong submission before 27 Sep 00:00** (48-hour mark: top 500 get +$100 credits). |
| **27 Sep** | France checks, cross-encoder/ensemble if time allows, final run, documentation, zip. Submit final package by **~9 PM** to leave margin. |

## 6. Suggested team split
- **Aayush + Claude**: pipeline, candidate generation, model, submissions (lab machine `msd2`, GPU 2).
- **Akshay (leader)**: leaderboard uploads and final zip upload (only his account can), check submission limits in the portal.
- **Ankit**: spot-check errors (false merges and misses) from validation output and suggest normalization rules, esp. India addresses.
- **Apoorv**: French abbreviation/legal-form dictionary; draft `Documentation_template.md` as we go.
