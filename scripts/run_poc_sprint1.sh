#!/usr/bin/env zsh
# scripts/run_poc_sprint1.sh — Sprint 1: Baseline Transformer (exp_poc_a)
#
# Runs 3 seeds of the baseline transformer (no Mamba, no ESN, no HDC).
# Logs to results/poc/sprint1_seed{N}.log
# Checkpoints to checkpoints/poc_a_s{N}/
#
# Usage:
#   zsh scripts/run_poc_sprint1.sh
#   zsh scripts/run_poc_sprint1.sh --seed 42   # single seed
#
# Expected wall clock: ~20-30 min per seed on M3 MPS (10K steps, d=128)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

RESULTS_DIR="results/poc"
mkdir -p "$RESULTS_DIR"
mkdir -p checkpoints/poc_a_s42 checkpoints/poc_a_s43 checkpoints/poc_a_s44

# Sprint config
D_MODEL=128
N_HEADS=4
N_LAYERS=4
FF_MULT=4
SEGMENT_LEN=128
BATCH_SIZE=8
STEPS=10000
LR=3e-4
WARMUP=500
VAL_EVERY=500
LOG_EVERY=100
SAVE_EVERY=5000

# Determine which seeds to run (default: all three)
SEEDS=(42 43 44)
if [[ $# -eq 2 && "$1" == "--seed" ]]; then
  SEEDS=($2)
fi

echo "================================================================"
echo "POC Sprint 1 — Baseline Transformer"
echo "Config: d=${D_MODEL}  layers=${N_LAYERS}  seg=${SEGMENT_LEN}  steps=${STEPS}"
echo "Seeds: ${SEEDS[*]}"
echo "================================================================"

for SEED in "${SEEDS[@]}"; do
  LOG_FILE="${RESULTS_DIR}/sprint1_seed${SEED}.log"
  CKPT_DIR="checkpoints/poc_a_s${SEED}"

  echo ""
  echo "──────────────────────────────────────────────────────────────"
  echo "Seed ${SEED}  →  ${LOG_FILE}"
  echo "Started: $(date)"
  echo "──────────────────────────────────────────────────────────────"

  T_START=$(date +%s)

  PYTHONPATH=python python3 scripts/train.py \
    --d-model    "$D_MODEL"   \
    --n-heads    "$N_HEADS"   \
    --n-layers   "$N_LAYERS"  \
    --ff-mult    "$FF_MULT"   \
    --segment-len "$SEGMENT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --steps      "$STEPS"    \
    --lr         "$LR"       \
    --warmup-steps "$WARMUP" \
    --val-every  "$VAL_EVERY" \
    --log-every  "$LOG_EVERY" \
    --save-every "$SAVE_EVERY" \
    --ckpt-dir   "$CKPT_DIR" \
    --seed       "$SEED"     \
    2>&1 | tee "$LOG_FILE"

  T_END=$(date +%s)
  ELAPSED=$(( T_END - T_START ))
  echo ""
  echo "Seed ${SEED} complete.  Elapsed: ${ELAPSED}s  ($(( ELAPSED / 60 ))m)"

  # Extract final val_ppl from log
  FINAL_VAL=$(grep '\[val\]' "$LOG_FILE" | tail -1 | grep -oE 'val_ppl\s+[0-9]+\.[0-9]+' | awk '{print $2}')
  if [[ -n "$FINAL_VAL" ]]; then
    echo "  Final val_ppl: ${FINAL_VAL}"
  fi
done

echo ""
echo "================================================================"
echo "Sprint 1 complete.  Results in ${RESULTS_DIR}/"
echo ""
echo "Summary:"
for SEED in "${SEEDS[@]}"; do
  LOG_FILE="${RESULTS_DIR}/sprint1_seed${SEED}.log"
  if [[ -f "$LOG_FILE" ]]; then
    FINAL_VAL=$(grep '\[val\]' "$LOG_FILE" | tail -1 | grep -oE 'val_ppl\s+[0-9]+\.[0-9]+' | awk '{print $2}')
    FINAL_LOSS=$(grep '^\bstep\b' "$LOG_FILE" | grep -E 'step\s+10000' | grep -oE 'loss\s+[0-9]+\.[0-9]+' | awk '{print $2}')
    echo "  seed ${SEED}: val_ppl=${FINAL_VAL:-N/A}  loss=${FINAL_LOSS:-N/A}"
  fi
done
echo ""
echo "Update results/poc/sprint1_baseline.md with these numbers."
echo "Gate: val_ppl < 2.5 to proceed to Sprint 2."
echo "================================================================"
