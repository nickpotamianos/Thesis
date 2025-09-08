#!/usr/bin/env bash
# run_fair_byexp_cv_zig.sh
# Strict fair CV on NON-ZIGZAG experiments + dedicated Zigzag sections.
# Caches collection once per (experiment, target) and reuses across folds/setups.

set -euo pipefail

ROOT="${ROOT:-$PWD}"

# Optional: activate venv
if [ -f "$ROOT/miluv_env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/miluv_env/bin/activate"
fi

EXPDIR="$ROOT/data/three_robots"
if [ ! -d "$EXPDIR" ]; then
  echo "ERROR: $EXPDIR not found" >&2
  exit 1
fi

# ---------- Discover eligible experiments ----------
declare -a ALL_EXPS=()
while IFS= read -r -d '' d; do
  bn="$(basename "$d")"
  shopt -s nocasematch
  # Exclude cir/obstacle/noAprilTag/oneTag cohorts from this pipeline
  if [[ "$bn" =~ cir|obstacle|noapriltag|onetag ]]; then
    continue
  fi
  ALL_EXPS+=("$bn")
done < <(find "$EXPDIR" -mindepth 1 -maxdepth 1 -type d -print0)

if [ "${#ALL_EXPS[@]}" -eq 0 ]; then
  echo "ERROR: no eligible experiments under $EXPDIR" >&2
  exit 1
fi

# Partition into Zigzag vs Non-Zigzag (case-insensitive)
declare -a ZIG_EXPS=()
declare -a BASE_EXPS=()
for e in "${ALL_EXPS[@]}"; do
  shopt -s nocasematch
  if [[ "$e" =~ zigzag ]]; then
    ZIG_EXPS+=("$e")
  else
    BASE_EXPS+=("$e")
  fi
done

echo "[DATASET] Eligible experiments: ${#ALL_EXPS[@]}"
printf '  - %s\n' "${ALL_EXPS[@]}"
echo
echo "[PARTITION] Non-Zigzag experiments: ${#BASE_EXPS[@]}"
printf '  - %s\n' "${BASE_EXPS[@]}"
echo "[PARTITION] Zigzag experiments:     ${#ZIG_EXPS[@]}"
printf '  - %s\n' "${ZIG_EXPS[@]}"

if [ "${#BASE_EXPS[@]}" -eq 0 ]; then
  echo "ERROR: No non-zigzag experiments remain after filtering." >&2
  exit 1
fi
if [ "${#ZIG_EXPS[@]}" -eq 0 ]; then
  echo "WARNING: No zigzag experiments found. Zigzag sections will be skipped." >&2
fi

# ---------- Targets: loop all three by default; can override via env: TARGETS="ifo001 ifo003" ----------
TARGETS_STR="${TARGETS:-ifo001 ifo002 ifo003}"
read -r -a TARGETS_ARR <<< "$TARGETS_STR"
echo
echo "[TARGETS] Will process targets: ${TARGETS_ARR[*]}"

# ---------- Common runtime args (without --target!) ----------
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

# ---------- Global cache: collect once per (target, experiment) ----------
CACHE_ROOT="$ROOT/outputs_collect_cache"
mkdir -p "$CACHE_ROOT"

ensure_collected() {
  local EXP="$1"
  local TGT="$2"
  local OUTDIR="$CACHE_ROOT/$TGT/$EXP"
  mkdir -p "$OUTDIR"

  # Check both possible locations for cached files
  local SUBDIR="$OUTDIR/${EXP}_${TGT}"
  local need=0
  
  # Check if files exist in either location
  if [ ! -s "$OUTDIR/bias_samples.jsonl" ] && [ ! -s "$SUBDIR/bias_samples.jsonl" ]; then 
    need=1
  fi
  if [ ! -s "$OUTDIR/fusion_snaps.jsonl" ] && [ ! -s "$OUTDIR/fusion_snaps.jsonl.gz" ] && \
     [ ! -s "$SUBDIR/fusion_snaps.jsonl" ] && [ ! -s "$SUBDIR/fusion_snaps.jsonl.gz" ]; then 
    need=1
  fi

  if [ "$need" -eq 0 ]; then
    echo "  [CACHE HIT] $EXP ($TGT)"
    return 0
  fi

  echo "  [COLLECT] $EXP ($TGT) -> $OUTDIR"
  python "$ROOT/swarm_target_tracking.py" \
    --exp "$EXP" \
    --target "$TGT" \
    "${COMMON_ARGS[@]}" \
    --collect_bias --collect_fusion \
    --out "$OUTDIR"
}

# Merge utility: build (bias, fusion) datasets for a list of EXPS across all TARGETS
build_dataset_for_set() {
  local OUTDIR="$1"; shift
  local -a EXPS_LIST=("$@")

  mkdir -p "$OUTDIR"
  local BIAS="$OUTDIR/bias_samples.jsonl"
  local FUSE="$OUTDIR/fusion_snaps.jsonl"
  rm -f "$BIAS" "$FUSE"

  echo "[DATASET] Building dataset at $OUTDIR"
  
  # Initialize empty files
  touch "$BIAS" "$FUSE"
  
  for tgt in "${TARGETS_ARR[@]}"; do
    for exp in "${EXPS_LIST[@]}"; do
      ensure_collected "$exp" "$tgt"
      
      # Look for the actual output directory with experiment name
      local SRC_BASE="$CACHE_ROOT/$tgt/$exp"
      local SRC_DIR="$SRC_BASE/${exp}_${tgt}"
      
      # Try both locations (direct and with subdirectory)
      for SRC_TRY in "$SRC_DIR" "$SRC_BASE"; do
        if [ -s "$SRC_TRY/bias_samples.jsonl" ]; then
          cat "$SRC_TRY/bias_samples.jsonl" >> "$BIAS"
          break
        fi
      done
      
      for SRC_TRY in "$SRC_DIR" "$SRC_BASE"; do
        if [ -s "$SRC_TRY/fusion_snaps.jsonl" ]; then
          cat "$SRC_TRY/fusion_snaps.jsonl" >> "$FUSE"
          break
        elif [ -s "$SRC_TRY/fusion_snaps.jsonl.gz" ]; then
          gzip -cd "$SRC_TRY/fusion_snaps.jsonl.gz" >> "$FUSE"
          break
        fi
      done
    done
  done

  echo "  [STATS] $(wc -l < "$BIAS" | tr -d ' ') lines in bias_samples.jsonl"
  echo "  [STATS] $(wc -l < "$FUSE" | tr -d ' ') lines in fusion_snaps.jsonl"
}

# Train BN/FN (by_exp + by_time) on given dataset dir
train_models() {
  local DS_DIR="$1"
  local OUTDIR="$2"
  mkdir -p "$OUTDIR"

  local BIAS="$DS_DIR/bias_samples.jsonl"
  local FUSE="$DS_DIR/fusion_snaps.jsonl"

  # Check if BiasNet models already exist
  if [ -f "$OUTDIR/biasnet_by_exp/biasnet.pt" ] && [ -f "$OUTDIR/biasnet_by_time/biasnet.pt" ]; then
    echo "[SKIP] BiasNet models already trained -> $OUTDIR"
  else
    echo "[TRAIN] BiasNet (by_exp/by_time) -> $OUTDIR"
    python -m swarm_ml.train_biasnet_cli   --samples "$BIAS" --out "$OUTDIR/biasnet_by_exp"   --split_mode by_exp   --seed 0
    python -m swarm_ml.train_biasnet_cli   --samples "$BIAS" --out "$OUTDIR/biasnet_by_time"  --split_mode by_time --seed 0
  fi

  # Check if FusionNet models already exist
  if [ -f "$OUTDIR/fusionnet_by_exp/fusionnet.pt" ] && [ -f "$OUTDIR/fusionnet_by_time/fusionnet.pt" ]; then
    echo "[SKIP] FusionNet models already trained -> $OUTDIR"
  else
    echo "[TRAIN] FusionNet (by_exp/by_time) -> $OUTDIR"
    python -m swarm_ml.train_fusionnet_cli --snaps   "$FUSE" --out "$OUTDIR/fusionnet_by_exp" --split_mode by_exp  --epochs 20 --seed 0
    python -m swarm_ml.train_fusionnet_cli --snaps   "$FUSE" --out "$OUTDIR/fusionnet_by_time" --split_mode by_time --epochs 20 --seed 0
  fi
}

# Evaluate models (and baseline) on a single TEST_EXP for all TARGETS
evaluate_on_exp() {
  local TEST_EXP="$1"
  local MODEL_DIR="$2"
  local OUTROOT="$3"
  mkdir -p "$OUTROOT"

  echo "[EVAL] Test on $TEST_EXP -> $OUTROOT"
  for tgt in "${TARGETS_ARR[@]}"; do
    local EVAL_DIR="$OUTROOT/$tgt"
    mkdir -p "$EVAL_DIR"

    # Baseline (no ML) - uses grid CI from COMMON_ARGS
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" --out "$EVAL_DIR/baseline"

    # BN only - uses grid CI with bias correction
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" \
      --biasnet_dir "$MODEL_DIR/biasnet_by_exp"   --out "$EVAL_DIR/bn_by_exp"
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" \
      --biasnet_dir "$MODEL_DIR/biasnet_by_time"  --out "$EVAL_DIR/bn_by_time"

    # FN only - OVERRIDE to learned CI
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" \
      --ci_method learned --fusionnet_dir "$MODEL_DIR/fusionnet_by_exp" --out "$EVAL_DIR/fn_by_exp"
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" \
      --ci_method learned --fusionnet_dir "$MODEL_DIR/fusionnet_by_time" --out "$EVAL_DIR/fn_by_time"

    # BN+FN (matched splits) - OVERRIDE to learned CI
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" \
      --ci_method learned --biasnet_dir "$MODEL_DIR/biasnet_by_exp"   --fusionnet_dir "$MODEL_DIR/fusionnet_by_exp"   --out "$EVAL_DIR/bnfn_by_exp"
    python "$ROOT/swarm_target_tracking.py" --exp "$TEST_EXP" --target "$tgt" "${COMMON_ARGS[@]}" \
      --ci_method learned --biasnet_dir "$MODEL_DIR/biasnet_by_time"  --fusionnet_dir "$MODEL_DIR/fusionnet_by_time"  --out "$EVAL_DIR/bnfn_by_time"

    # Per-scenario summaries
    for SCEN in "$EVAL_DIR"/*; do
      [ -d "$SCEN" ] || continue
      python "$ROOT/swarm_eval_table.py" --root "$SCEN" || true
    done
  done
}

# =====================================================================
# (A) Normal CV (NON-ZIGZAG only) — strict unseen per fold
# =====================================================================
CVROOT="$ROOT/outputs_cv_byexp_nozig"
mkdir -p "$CVROOT"

fold=0
for TEST_EXP in "${BASE_EXPS[@]}"; do
  ((fold+=1))
  echo
  echo "===================== NON-ZIG FOLD $fold / ${#BASE_EXPS[@]} : TEST = $TEST_EXP ====================="

  # TRAIN = all non-zigzag except TEST
  TRAIN_EXPS=()
  for e in "${BASE_EXPS[@]}"; do
    [[ "$e" == "$TEST_EXP" ]] || TRAIN_EXPS+=("$e")
  done

  FOLD_DIR="$CVROOT/fold_${fold}"
  DS_DIR="$FOLD_DIR/datasets";  mkdir -p "$DS_DIR"
  MODEL_DIR="$FOLD_DIR/models"; mkdir -p "$MODEL_DIR"
  EVAL_DIR="$FOLD_DIR/eval";    mkdir -p "$EVAL_DIR"

  # 1) Dataset for this fold (TRAIN_EXPS across all targets)
  build_dataset_for_set "$DS_DIR" "${TRAIN_EXPS[@]}"

  # 2) Train BN/FN for this fold
  train_models "$DS_DIR" "$MODEL_DIR"

  # 3) Evaluate on held-out TEST_EXP for all targets
  evaluate_on_exp "$TEST_EXP" "$MODEL_DIR" "$EVAL_DIR"
done

echo
echo "===== NON-ZIGZAG CV COMPLETE -> $CVROOT ====="

# =====================================================================
# (B) Zigzag Sections
# =====================================================================
ZIGROOT="$ROOT/outputs_zig_sections"
mkdir -p "$ZIGROOT"

if [ "${#ZIG_EXPS[@]}" -gt 0 ]; then
  # ---------- B1) Generalization to Zigzag: Train on NON-ZIG only, test on all Zigzag ----------
  echo
  echo "===================== ZIG-GEN: Train on NON-ZIG, test on Zigzag ====================="
  GEN_DIR="$ZIGROOT/generalization"
  GEN_DS="$GEN_DIR/datasets";   mkdir -p "$GEN_DS"
  GEN_MD="$GEN_DIR/models";     mkdir -p "$GEN_MD"
  GEN_EV="$GEN_DIR/eval";       mkdir -p "$GEN_EV"

  build_dataset_for_set "$GEN_DS" "${BASE_EXPS[@]}"
  train_models "$GEN_DS" "$GEN_MD"
  for ZTEST in "${ZIG_EXPS[@]}"; do
    evaluate_on_exp "$ZTEST" "$GEN_MD" "$GEN_EV/$ZTEST"
  done

  # ---------- B2) Zigzag-only specialization (LOEO within Zigzag) ----------
  echo
  echo "===================== ZIG-ONLY LOEO ====================="
  zfold=0
  for ZTEST in "${ZIG_EXPS[@]}"; do
    ((zfold+=1))
    echo "---- ZIG FOLD $zfold / ${#ZIG_EXPS[@]} : TEST = $ZTEST ----"
    ZTRAIN=()
    for e in "${ZIG_EXPS[@]}"; do
      [[ "$e" == "$ZTEST" ]] || ZTRAIN+=("$e")
    done
    ZF_DIR="$ZIGROOT/zigloo_fold_${zfold}"
    ZF_DS="$ZF_DIR/datasets";  mkdir -p "$ZF_DS"
    ZF_MD="$ZF_DIR/models";    mkdir -p "$ZF_MD"
    ZF_EV="$ZF_DIR/eval";      mkdir -p "$ZF_EV"

    build_dataset_for_set "$ZF_DS" "${ZTRAIN[@]}"
    train_models "$ZF_DS" "$ZF_MD"
    evaluate_on_exp "$ZTEST" "$ZF_MD" "$ZF_EV"
  done

  # ---------- B3) Hybrid (Recommended): Train on ALL except held-out Zigzag ----------
  echo
  echo "===================== ZIG-HYBRID (recommended): Train on ALL minus held-out Zigzag ====================="
  hfold=0
  for ZTEST in "${ZIG_EXPS[@]}"; do
    ((hfold+=1))
    echo "---- HYBRID FOLD $hfold / ${#ZIG_EXPS[@]} : TEST = $ZTEST ----"
    HTRAIN=()
    for e in "${ALL_EXPS[@]}"; do
      [[ "$e" == "$ZTEST" ]] || HTRAIN+=("$e")
    done
    HF_DIR="$ZIGROOT/hybrid_fold_${hfold}"
    HF_DS="$HF_DIR/datasets";  mkdir -p "$HF_DS"
    HF_MD="$HF_DIR/models";    mkdir -p "$HF_MD"
    HF_EV="$HF_DIR/eval";      mkdir -p "$HF_EV"

    build_dataset_for_set "$HF_DS" "${HTRAIN[@]}"
    train_models "$HF_DS" "$HF_MD"
    evaluate_on_exp "$ZTEST" "$HF_MD" "$HF_EV"
  done
fi

echo
echo "===== DONE ====="
echo "Cache of single-pass collections per (experiment,target): $CACHE_ROOT"
echo "Non-Zigzag CV outputs:                                   $CVROOT"
echo "Zigzag sections (GEN / ZigLOEO / Hybrid):                $ZIGROOT"