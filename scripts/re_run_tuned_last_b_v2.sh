#!/usr/bin/env bash
set -euo pipefail

# Re-run tuned condition B for the combos completed in the last baseline lap,
# with Q-adaptation enabled and a slightly larger UWB std (0.33), writing to outputs_online_tuned_v2.

OUT_A=${1:-outputs_baseline_v1}
OUT_B=${2:-outputs_online_tuned_v2}
PY=${PY:-miluv_env/bin/python}

mapfile -t DONE_A < <(ls -1 "${OUT_A}"/*/target_estimate.csv 2>/dev/null | sed "s|^${OUT_A}/||; s|/target_estimate.csv$||" | sort)
if (( ${#DONE_A[@]} == 0 )); then
  echo "No completed baseline runs found under ${OUT_A}" >&2
  exit 0
fi

mkdir -p "${OUT_B}"
common=(
  --use_height --use_height_tf
  --uwb_std 0.33
  --pair_corr 0.75
  --sigma_a_xy 1.0 --sigma_a_z 0.45
  --ci_method grid --ci_objective trace
  --use_los --los_influence 0.2 --geom_influence 0.4
  --ema_alpha 0.0 --online_tune --online_r_min_scale 0.75 --online_r_max_scale 10.0
  --gate_target 0.97 --gate_sigma_init 3.0 --q_adapt
)

COUNT=0
for combo in "${DONE_A[@]}"; do
  exp=${combo%_*}
  tgt=${combo##*_}
  # Limit to stress experiments if desired; otherwise run all from baseline
  if [[ "$exp" != "default_3_random3_2" && "$exp" != "default_3_random3_1" ]]; then
    continue
  fi
  out_csv="${OUT_B}/${exp}_${tgt}/target_estimate.csv"
  if [[ -f "$out_csv" ]]; then echo "[SKIP-B2] ${exp} ${tgt}"; continue; fi
  cmd=( env PYTHONPATH=$(pwd) ${PY} swarm_target_tracking.py --exp "$exp" --target "$tgt" "${common[@]}" --out "$OUT_B" )
  echo "[RUN-B2] ${exp} ${tgt}"; echo "         ${cmd[*]}"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "${cmd[@]}"; fi
  COUNT=$((COUNT+1))
done

echo "[DONE] Scheduled ${COUNT} tuned v2 runs to match last baseline stress combos."

