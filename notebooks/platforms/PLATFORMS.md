# DREX-UNIFIED POC — Cloud Platform Quick-Start Guide

This guide covers running all 5 POC sprints on free and low-cost cloud GPU platforms
so nothing bogs down your local machine.

---

## Sprint overview

| Sprint | Name | Config | Seeds | Steps | Est. T4 time |
|---|---|---|---|---|---|
| 1 | `exp_poc_a` | Baseline transformer | 42, 43, 44 | 10k | ~25 min |
| 2 | `exp_poc_b` | + Mamba SSM backbone | 42, 43, 44 | 10k | ~25 min |
| 3 | `exp_poc_c` | + ESN episodic memory | 42, 43, 44 | 10k | ~30 min |
| 4 | `exp_poc_d` | + HDC encoder (full DREX-UNIFIED core) | 42, 43, 44 | 10k | ~30 min |
| 5 | `exp_poc_e` | Scale: d=256, 8L, 512-seg | 42 | 50k | ~90 min |

**Recommended strategy:** Run Sprints 1–4 in parallel across 4 separate Colab/Kaggle tabs
(each tab = one sprint). Sprint 5 is single-seed and runs best on Lightning AI, RunPod,
or any paid GPU VM (cost << $1 at spot prices).

---

## 1. Google Colab (recommended for Sprints 1–4)

**File:** `notebooks/platforms/colab_drex_poc.ipynb`

**Setup:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `colab_drex_poc.ipynb`
3. **Runtime → Change runtime type → GPU → T4** (free) or A100 (Colab Pro)
4. In Cell 2, set `SPRINT = 1` (or 2, 3, 4, 5)
5. **Runtime → Run all** (`Ctrl+F9`)

**What it does:**
- Mounts Google Drive as `/content/drive/MyDrive/drex_poc/` (results persist across sessions)
- Clones the drex repo to `/content/drex`
- Installs `datasets` and `safetensors`
- Runs `scripts/poc/run_poc_cloud.py --sprint N` with the selected seeds
- Prints a results table with per-seed `val_ppl`

**Time limit:** Colab Free sessions are limited to ~12 h. Sprints 1–4 (~25–30 min/sprint × 3 seeds) fit comfortably within the limit.

**Tip:** Open 4 separate browser tabs → upload the same notebook 4 times → set `SPRINT=1/2/3/4` in each → run all simultaneously to collect all 4 sprints in ~30 minutes.

---

## 2. Kaggle (recommended for Sprints 1–4, parallel backup)

**File:** `notebooks/platforms/kaggle_drex_poc.ipynb`

**Setup:**
1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. **File → Import Notebook** → upload `kaggle_drex_poc.ipynb`
3. In the right-side **Settings** panel: **Accelerator → GPU T4 x2** (or P100)
4. Edit Cell 2: set `SPRINT = 1` (or 2, 3, 4, 5)
5. Click **Run All** (⏵▾)

**What it does:**
- Outputs to `/kaggle/working/drex_poc/` (auto-saved as Kaggle outputs)
- Clones drex repo to `/kaggle/working/drex`
- Runs the sprint runner and saves `*_summary.json` to Kaggle outputs

**Quota:** 30 GPU hours/week. Sprint 1–4 each use ~1.5 h (3 seeds × 30 min). Running 4 sprints in parallel uses ~6 GPU-hours total from your quota.

**Download results:** After the run, go to **Output** tab → download `*_summary.json`.

---

## 3. Lightning AI Studios (recommended for Sprint 5)

**File:** `notebooks/platforms/lightning_run.sh`

**Setup:**
1. Go to [lightning.ai](https://lightning.ai) → create account (free 15 credits/month)
2. **New Studio** → choose **PyTorch** template → select **GPU: T4**
3. Open the built-in **Terminal** tab
4. Paste and run:

```bash
# Option A: clone and run directly
git clone https://github.com/wesleyscholl/drex.git /workspace/drex
SPRINT=5 bash /workspace/drex/notebooks/platforms/lightning_run.sh

# Option B: run all sprints sequentially
ALL_SPRINTS=1 bash /workspace/drex/notebooks/platforms/lightning_run.sh

# Option C: resume a specific sprint, skip setup
SPRINT=3 SKIP_SETUP=1 bash /workspace/drex/notebooks/platforms/lightning_run.sh
```

**Results** are written to `/workspace/drex_poc/`. Download via **Files** panel or `scp`.

---

## 4. RunPod (recommended for Sprint 5 or if Colab/Kaggle are unavailable)

**File:** `notebooks/platforms/lightning_run.sh`

**Setup:**
1. Go to [runpod.io](https://runpod.io) → **Deploy**
2. Select a **RTX 3090** or **A100** pod → PyTorch template → click **Deploy On-Demand**
3. Connect via **Web Terminal** or SSH
4. Run:

```bash
git clone https://github.com/wesleyscholl/drex.git /workspace/drex
SPRINT=5 bash /workspace/drex/notebooks/platforms/lightning_run.sh
```

**Cost:** ~$0.20–$0.50 for Sprint 5 (90 min on RTX 3090 at ~$0.22/h spot).
**Stop the pod immediately** after the run to avoid idle charges.

---

## 5. Vast.ai (cheapest GPU option)

Same as RunPod. Use `lightning_run.sh` via SSH terminal.

```bash
# After SSH into your Vast.ai instance:
git clone https://github.com/wesleyscholl/drex.git /workspace/drex
SPRINT=5 RESULTS_ROOT=/root/drex_poc bash /workspace/drex/notebooks/platforms/lightning_run.sh
```

---

## Environment variables for `lightning_run.sh`

| Variable | Default | Description |
|---|---|---|
| `SPRINT` | `1` | Sprint number (1–5) |
| `SEEDS` | `42 43 44` | Space-separated seeds (sprint 5 uses `42` only) |
| `BATCH_SIZE` | `32` | Training batch size (safe on T4/P100 for d=128/256) |
| `DEVICE` | `auto` | PyTorch device: `auto`, `cuda`, `mps`, `cpu` |
| `RESULTS_ROOT` | `/workspace/drex_poc` | Output directory for logs, checkpoints, JSON |
| `REPO_DIR` | `/workspace/drex` | Path to cloned drex repo |
| `ALL_SPRINTS` | `0` | Set to `1` to run all 5 sprints sequentially |
| `SKIP_SETUP` | `0` | Set to `1` to skip `git clone` + `pip install` |

---

## Collecting results

All platforms write a per-sprint JSON summary:

```
{RESULTS_ROOT}/exp_poc_a_summary.json   # Sprint 1
{RESULTS_ROOT}/exp_poc_b_summary.json   # Sprint 2
{RESULTS_ROOT}/exp_poc_c_summary.json   # Sprint 3
{RESULTS_ROOT}/exp_poc_d_summary.json   # Sprint 4
{RESULTS_ROOT}/exp_poc_e_summary.json   # Sprint 5
{RESULTS_ROOT}/overall_summary.json     # Combined (only when --all is used)
```

Each JSON looks like:
```json
{
  "sprint": 2,
  "name": "exp_poc_b",
  "description": "Mamba SSM backbone replacing L1 SWA",
  "median_val_ppl": 1.3421,
  "results": {
    "42": {"seed": 42, "returncode": 0, "elapsed_s": 1620, "final_val_ppl": 1.33},
    "43": {"seed": 43, "returncode": 0, "elapsed_s": 1590, "final_val_ppl": 1.36},
    "44": {"seed": 44, "returncode": 0, "elapsed_s": 1609, "final_val_ppl": 1.33}
  }
}
```

Copy these files back to `results/poc/` in the local repo and update `DREX_UNIFIED_PLAN.md`
sprint checklist when each gate is passed.

---

## Sprint gate criteria

| Sprint | Gate | Action if fails |
|---|---|---|
| 1 | val_ppl < 2.5 at 10k steps | Check for NaN in logs; reduce LR to 1e-4 |
| 2 | val_ppl ≤ Sprint 1 + 0.20 | Reduce `--mamba-d-state` to 8; try `--mamba-expand 4` |
| 3 | val_ppl < Sprint 2 | If wr > 0.85, lower `--episodic-gate-thresh` to 0.50 |
| 4 | val_ppl ≤ Sprint 3 | Try `--hdc-dim 256` (closer to d_model) |
| 5 | val_ppl ≤ Sprint 1 AND passkey depth ≥ 2× Sprint 1 | Debug which component is hurting |

---

## Troubleshooting

**`ModuleNotFoundError: drex`** — The drex Python package path wasn't added to `sys.path`.
The runner does this automatically via `--repo-root`. If running `train.py` manually,
add `export PYTHONPATH=/workspace/drex/python:$PYTHONPATH` before running.

**CUDA out of memory** — Reduce `--batch-size` to 16 or 8. Sprint 5 (d=256) may need
`--batch-size 16` on T4 (16 GB).

**HuggingFace download fails / SSL error** — Add `--no-ssl-verify` to the `train.py`
call. On Colab/Kaggle this is rarely needed.

**Session timeout before sprint finishes** — Each sprint completion writes a checkpoint
to `{RESULTS_ROOT}/checkpoints/{name}_s{seed}/`. Use `--resume path/to/checkpoint.safetensors`
to continue from the last saved step.

**Kaggle kernel dies without output** — Kaggle kills kernels that produce no output for
~20 min. The runner streams logs live so this shouldn't happen. If it does, reduce seeds
to just `[42]` for a smoke test first.
