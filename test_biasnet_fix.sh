#!/usr/bin/env bash
# test_biasnet_fix.sh
# Test BiasNet feature fix with proper train/test split on multiple default experiments
# Follows same pattern as run_fair_byexp_cv_zig.sh but focuses on BiasNet validation

set -euo pipefail

ROOT="${ROOT:-$PWD}"

# Optional: activate venv
if [ -f "$ROOT/miluv_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/miluv_env/bin/activate"
fi

# Test output directory
TEST_OUT="$ROOT/test_biasnet_fix_output"
mkdir -p "$TEST_OUT"

echo "===== BiasNet Feature Fix Test ====="
echo "Output directory: $TEST_OUT"

# Discover specific default experiments: random, random2, random3 variants only
EXPDIR="$ROOT/data/three_robots"
declare -a ALL_DEFAULT_EXPS=()
while IFS= read -r -d '' d; do
  bn="$(basename "$d")"
  shopt -s nocasematch
  # Only include default_*_random, default_*_random2, default_*_random3 (no moving triangle)
  if [[ "$bn" =~ ^default_.*_random[23]?[^a-z] ]] || [[ "$bn" =~ ^default_.*_random[23]?$ ]]; then
    ALL_DEFAULT_EXPS+=("$bn")
  fi
done < <(find "$EXPDIR" -mindepth 1 -maxdepth 1 -type d -print0)

if [ "${#ALL_DEFAULT_EXPS[@]}" -lt 3 ]; then
  echo "ERROR: Need at least 3 default random experiments for proper train/test split" >&2
  echo "Found: ${ALL_DEFAULT_EXPS[*]}" >&2
  exit 1
fi

echo "Available default random experiments: ${#ALL_DEFAULT_EXPS[@]}"
printf '  - %s\n' "${ALL_DEFAULT_EXPS[@]}"

# Use same parameters as main script
COMMON_ARGS=(
  --use_height
  --use_height_tf
  --uwb_std 0.8
  --pair_corr 0.3
  --sigma_a_xy 3.0
  --sigma_a_z 1.5
  --ci_method grid
  --ci_objective trace
  --los_influence 0
  --geom_influence 0
  --ema_alpha 0.0
  --online_tune
  --online_r_min_scale 0.75
  --online_r_max_scale 3.0
  --gate_target 0.90
  --gate_sigma_init 4.0
  --q_adapt
)

# Use first 3 experiments for training, last one for testing
TRAIN_EXPS=("${ALL_DEFAULT_EXPS[@]:0:3}")
TEST_EXP="${ALL_DEFAULT_EXPS[3]}"
TARGET="ifo001"

echo
echo "Training experiments: ${TRAIN_EXPS[*]}"
echo "Test experiment: $TEST_EXP"
echo "Target: $TARGET"

# Step 1: Collect bias samples from training experiments
echo
echo "===== Step 1: Collecting BiasNet training data ====="
COLLECT_DIR="$TEST_OUT/collect"
mkdir -p "$COLLECT_DIR"

for exp in "${TRAIN_EXPS[@]}"; do
  echo "Collecting from: $exp"
  python "$ROOT/swarm_target_tracking.py" \
    --exp "$exp" \
    --target "$TARGET" \
    "${COMMON_ARGS[@]}" \
    --collect_bias \
    --out "$COLLECT_DIR"
  
  samples_file="$COLLECT_DIR/${exp}_${TARGET}/bias_samples.jsonl"
  if [ -f "$samples_file" ]; then
    echo "[COLLECT] $exp -> $(wc -l < "$samples_file" | tr -d ' ') samples"
  else
    echo "[ERROR] No samples collected from $exp" >&2
  fi
done

# Merge all training samples
echo
echo "Merging training samples..."
MERGED_SAMPLES="$COLLECT_DIR/merged_train_samples.jsonl"
rm -f "$MERGED_SAMPLES"
for exp in "${TRAIN_EXPS[@]}"; do
  samples_file="$COLLECT_DIR/${exp}_${TARGET}/bias_samples.jsonl"
  if [ -f "$samples_file" ]; then
    cat "$samples_file" >> "$MERGED_SAMPLES"
  fi
done

if [ ! -f "$MERGED_SAMPLES" ] || [ ! -s "$MERGED_SAMPLES" ]; then
  echo "ERROR: No training samples collected" >&2
  exit 1
fi

echo "[MERGE] Total training samples: $(wc -l < "$MERGED_SAMPLES" | tr -d ' ')"

# Step 2: Train BiasNet on merged samples
echo
echo "===== Step 2: Training BiasNet ====="
MODEL_DIR="$TEST_OUT/models"
mkdir -p "$MODEL_DIR"

python -m swarm_ml.train_biasnet_cli \
  --samples "$MERGED_SAMPLES" \
  --out "$MODEL_DIR/biasnet_fixed" \
  --split_mode by_exp \
  --seed 42

echo "[TRAIN] BiasNet trained -> $MODEL_DIR/biasnet_fixed"

# Step 3: Test BiasNet performance on unseen experiment
echo
echo "===== Step 3: Testing BiasNet on Unseen Experiment ====="
echo "Test experiment: $TEST_EXP (not used in training)"
EVAL_DIR="$TEST_OUT/eval"
mkdir -p "$EVAL_DIR"

echo
echo "--- Baseline (no BiasNet) ---"
python "$ROOT/swarm_target_tracking.py" \
  --exp "$TEST_EXP" \
  --target "$TARGET" \
  "${COMMON_ARGS[@]}" \
  --out "$EVAL_DIR/baseline"

echo
echo "--- BiasNet enabled ---"
python "$ROOT/swarm_target_tracking.py" \
  --exp "$TEST_EXP" \
  --target "$TARGET" \
  "${COMMON_ARGS[@]}" \
  --bias_gain 0.6 \
  --biasnet_dir "$MODEL_DIR/biasnet_fixed" \
  --out "$EVAL_DIR/biasnet"

# Step 4: Compare results
echo
echo "===== Step 4: Results Comparison ====="

echo "--- Baseline Results ---"
python "$ROOT/swarm_eval_table.py" --root "$EVAL_DIR/baseline" 2>/dev/null || echo "Baseline eval failed"

echo
echo "--- BiasNet Results ---"
python "$ROOT/swarm_eval_table.py" --root "$EVAL_DIR/biasnet" 2>/dev/null || echo "BiasNet eval failed"

# Extract key metrics for quick comparison
echo
echo "===== Training/Test Data Summary ====="
echo "Training experiments: ${TRAIN_EXPS[*]}"
echo "Training samples: $(wc -l < "$MERGED_SAMPLES" | tr -d ' ')"
echo "Test experiment: $TEST_EXP"
echo
echo "===== BiasNet Feature Fix Validation ====="
echo "Key things to check:"
echo "1. BiasNet RMSE should be <= Baseline RMSE"
echo "2. BiasNet NEES should be closer to 1.0 and not showing 'OUT-OF-BAND' warnings"
echo "3. This validates the height feature (Δz) fix for train/test parity"
echo "4. Model trained on ${#TRAIN_EXPS[@]} experiments, tested on unseen data"
echo
echo "Results saved in: $TEST_OUT"