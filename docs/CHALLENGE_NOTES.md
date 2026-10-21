# Amazon ML Challenge 2026 — Team "moov" notes

Saved 25 Sep 2026. Full problem statement: `student_resource/README.md`.

## Team
Akshay Bagde (leader) · Ankit Kr Sinha · Aayush Ranjan · Apoorv Singh — all verified on Unstop.
Only the leader's account can open the round page, download data, and submit.

## Deadline
Round closes **27 Sep 2026, 11:59 PM IST**.
- Leaderboard submissions: upload `matching_results.tsv` on the round page
  (https://unstop.com/competitions/1743604/round/1593683/play/code).
- Final package (required from every team): `moov_submission.zip`
  ```
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/{src/, README.md, requirements.txt}
  Documentation_template.md   (filled in)
  ```

## Task: Business Entity Resolution
For every Source 1 entity, list all matching Source 2 / Source 3 records (can be zero).
- Columns: `entity_id, business_name, business_address, country` (TSV — always `sep="\t"`).
- Train: US + India. **Test also has France** — never hard-code countries; every test S1 row must appear.
- Metric: **macro F0.5 per S1 entity**, precision-heavy. Singletons with an empty prediction score 1.0;
  any false match on a singleton scores 0.

| File | Rows |
| --- | --- |
| train_source1 / ground_truth | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

## Rules that can disqualify you
- No external lookups: no entity-resolution APIs, government registries, geocoding, or internet data.
- Any model used must be MIT/Apache-2.0 licensed and at most 8B parameters.
- Validate before uploading:
  `cd student_resource && python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Leaderboard snapshot (25 Sep, ~4:50 PM IST)
Top score 0.9859 (Banana, BITS Pilani); #30 ≈ 0.974; 490 teams. moov: no submission yet.

## Reference files
- `reference/problem_statement_video.mp4` — official explainer video
- `reference/aws_builder_setup_guide.pdf` — official AWS setup guide (Step 3 = Free Tier, $200 credits)
- AWS prep blog: https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
- Prep session recording: https://www.twitch.tv/videos/2880101957
