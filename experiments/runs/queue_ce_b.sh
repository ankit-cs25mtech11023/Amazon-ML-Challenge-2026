#!/bin/bash
until grep -q CEHI_DONE logs/ce_hi_score.log 2>/dev/null; do sleep 20; done
bash run_ce_b.sh
