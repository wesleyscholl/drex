#!/usr/bin/env bash
# =============================================================================
# lightning_run.sh — DREX-UNIFIED POC runner for Linux GPU VMs
#
# Compatible with:
#   - Lightning AI Studios (Terminal tab)
#   - RunPod (https://runpod.io)  — Jupyter terminal or SSH
#   - Vast.ai                     — SSH terminal
#   - Lambda Labs                 — SSH terminal
#   - Any Ubuntu/Debian GPU instance with Python 3.9+
#
# Usage:
#   # Run a single sprint (default: sprint 1, seeds 42 43 44):
#   bash lightning_run.sh
#
#   # Run sprint 2:
#   SPRINT=2 bash lightning_run.sh
#
#   # Run all 5 sprints sequentially:
#   ALL_SPRINTS=1 bash lightning_run.sh
#
#   # Run sprint 5 (scale run, seed 42 only):
#   SPRINT=5 bash lightning_run.sh
#
# Environment variables:
#   SPRINT            Sprint number to run (1–5).  Default: 1.
#   SEEDS             Space-separated seed list.   Default: 42 43 44 (42 for sprint 5).
#   BATCH_SIZE        Training batch size.         Default: 32.
#   DEVICE            PyTorch device.              Default: auto (picks CUDA).
#   RESULTS_ROOT      Output directory.            Default: /workspace/drex_poc.
#   REPO_DIR          Path to cloned drex repo.    Default: /workspace/drex.
#   ALL_SPRINTS       Set to 1 to run all sprints. Default: 0.
#   SKIP_SETUP        Set to 1 to skip clone + pip install (re-runs only). Default: 0.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
SPRINT="${SPRINT:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
REPO_URL="https://github.com/wesleyscholl/drex.git"
REPO_DIR="${REPO_DIR:-/workspace/drex}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/drex_poc}"
ALL_SPRINTS="${ALL_SPRINTS:-0}"
SKIP_SETUP="${SKIP_SETUP:-0}"

# Default seeds per sprint (sprint 5 = single seed 50k-step scale run)
if [[ "${SPRINT}" == "5" ]]; then
    SEEDS="${SEEDS:-42}"
else
    SEEDS="${SEEDS:-42 43 44}"
fi

# ---------------------------------------------------------------------------
# Helper: pretty section banner
# ---------------------------------------------------------------------------
banner() {
    echo ""
    echo "======================================================================="
    echo "  $*"
    echo "======================================================================="
    echo ""
}

# ---------------------------------------------------------------------------
# Setup: clone repo + install Python deps
# ---------------------------------------------------------------------------
if [[ "${SKIP_SETUP}" != "1" ]]; then
    banner "Setup: cloning drex repo"

    if [[ -d "${REPO_DIR}/.git" ]]; then
        echo "Repo already cloned at ${REPO_DIR} — pulling latest..."
        git -C "${REPO_DIR}" pull --ff-only
    else
        git clone "${REPO_URL}" "${REPO_DIR}"
    fi

    banner "Setup: installing Python dependencies"
    python3 -m pip install --upgrade pip
    python3 -m pip install torch datasets safetensors tqdm
fi

mkdir -p "${RESULTS_ROOT}"

# Verify GPU is available
banner "GPU check"
python3 - << 'PY_EOF'
import torch, sys
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  ({p.total_memory / 1024**3:.1f} GB VRAM)")
else:
    print("WARNING: No CUDA GPU found. Training will be very slow.")
    print("Make sure you selected a GPU pod/instance.")
PY_EOF

# ---------------------------------------------------------------------------
# Build the sprint runner command
# ---------------------------------------------------------------------------
build_cmd() {
    local sprint="$1"
    local seeds_str="$2"
    local seed_args=()
    read -r -a seed_args <<< "${seeds_str}"

    echo python3 "${REPO_DIR}/scripts/poc/run_poc_cloud.py" \
        --sprint "${sprint}" \
        --seeds "${seed_args[@]}" \
        --out-dir "${RESULTS_ROOT}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --repo-root "${REPO_DIR}"
}

# ---------------------------------------------------------------------------
# Run sprints
# ---------------------------------------------------------------------------
cd "${REPO_DIR}"

if [[ "${ALL_SPRINTS}" == "1" ]]; then
    banner "Running all 5 sprints sequentially"
    python3 "${REPO_DIR}/scripts/poc/run_poc_cloud.py" \
        --all \
        --out-dir "${RESULTS_ROOT}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --repo-root "${REPO_DIR}"
else
    banner "Running Sprint ${SPRINT} (seeds: ${SEEDS})"
    # Build seed array from space-separated string
    seed_array=()
    read -r -a seed_array <<< "${SEEDS}"

    python3 "${REPO_DIR}/scripts/poc/run_poc_cloud.py" \
        --sprint "${SPRINT}" \
        --seeds "${seed_array[@]}" \
        --out-dir "${RESULTS_ROOT}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --repo-root "${REPO_DIR}"
fi

# ---------------------------------------------------------------------------
# Sprint 3/4/5: run passkey eval after training
# ---------------------------------------------------------------------------
if [[ "${SPRINT}" == "3" || "${SPRINT}" == "4" || "${SPRINT}" == "5" ]]; then
    _SPRINT_NAMES=([1]="exp_poc_a" [2]="exp_poc_b" [3]="exp_poc_c" [4]="exp_poc_d" [5]="exp_poc_e")
    name="${_SPRINT_NAMES[${SPRINT}]}"

    banner "Sprint ${SPRINT}: passkey retrieval eval"

    for seed in ${SEEDS}; do
        ckpt_dir="${RESULTS_ROOT}/checkpoints/${name}_s${seed}"
        # Find the latest safetensors checkpoint
        ckpt=$(find "${ckpt_dir}" -name "*.safetensors" 2>/dev/null | sort | tail -1 || true)
        if [[ -z "${ckpt}" ]]; then
            echo "No checkpoint found in ${ckpt_dir} — skipping passkey eval for seed ${seed}"
            continue
        fi
        echo "Evaluating: ${ckpt}"
        max_ctx=1024
        if [[ "${SPRINT}" == "5" ]]; then
            max_ctx=4096
        fi
        python3 -m drex.eval.passkey \
            --checkpoint "${ckpt}" \
            --max-context "${max_ctx}" \
            2>&1 | tee "${RESULTS_ROOT}/passkey_s${sprint}_s${seed}.log" || \
            echo "WARN: passkey eval failed — continuing"
    done
fi

if [[ "${SPRINT}" == "5" ]]; then
    banner "Sprint 5: BABILong eval"
    for seed in ${SEEDS}; do
        name="exp_poc_e"
        ckpt_dir="${RESULTS_ROOT}/checkpoints/${name}_s${seed}"
        ckpt=$(find "${ckpt_dir}" -name "*.safetensors" 2>/dev/null | sort | tail -1 || true)
        if [[ -z "${ckpt}" ]]; then
            echo "No checkpoint found in ${ckpt_dir} — skipping BABILong eval for seed ${seed}"
            continue
        fi
        python3 -m drex.eval.babilong \
            --checkpoint "${ckpt}" \
            2>&1 | tee "${RESULTS_ROOT}/babilong_s5_s${seed}.log" || \
            echo "WARN: babilong eval failed — continuing"
    done
fi

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
banner "Results saved to: ${RESULTS_ROOT}"
find "${RESULTS_ROOT}" -name "*_summary.json" -exec echo "  {}" \;

echo ""
echo "Done."
