#!/usr/bin/env bash
set -euo pipefail

EXP_LIST=("default_3_random_0" "default_3_random_1" "default_3_random_2")
OUT="outputs_swarm_batch"

for E in "${EXP_LIST[@]}"; do
  echo "Running $E ..."
  PYTHONPATH=$(pwd) miluv_env/bin/python swarm_target_tracking.py \
    --exp "$E" --target ifo003 \
    --use_height --use_height_tf --height_std 0.07 \
    --uwb_std 0.28 --pair_corr 0.7 --gate_sigma 3.0 \
    --sigma_a_xy 1.0 --sigma_a_z 0.45 \
    --use_los \
    --ci_method grid --ci_objective trace --ci_grid 0.1 \
    --out "$OUT"
done

echo "Done. See $OUT/*/summary.csv"

