# results/TRAINING_RUNS.md — Drex Experiment A/B Baseline Comparison

**Purpose:** End-to-end comparison of DrexTransformer with and without episodic memory
(`MemoryModule`) on TinyStories, evaluated on passkey recall and BABILong. This is the
first published training result for the Phase 13–15 validated architecture.

---

## Setup

### Hardware

| Field | Value |
|---|---|
| Machine | (fill in: e.g., "MacBook Pro M3 Max, 128GB") |
| Compute | (fill in: e.g., "MPS / CUDA / CPU") |
| PyTorch | (fill in: e.g., "2.5.1") |
| Python | (fill in: e.g., "3.12.8") |
| Date | (fill in) |

### Data

TinyStories (roneneldan/TinyStories on HuggingFace Hub).

- Train split: ~50M characters (~2.1M stories)
- Validation split: ~500k characters cap (--val-max-chars 500000)
- Tokenisation: char-level, vocab size 256 (raw byte values)
- Segment length: 512 tokens TBPTT chunks

If direct download is blocked by SSL (self-signed certificate error):

```bash
# Option 1: disable SSL verification for the download session (dev only)
# Pass --no-ssl-verify to train.py

# Option 2: download TinyStories as a local text file first
python3 -c "
import ssl, httpx, sys
# patch SSL (dev only)
import httpx as hx
class _C(hx.Client):
    def __init__(self, *a, **kw): kw['verify']=False; super().__init__(*a,**kw)
hx.Client = _C
class _CA(hx.AsyncClient):
    def __init__(self, *a, **kw): kw['verify']=False; super().__init__(*a,**kw)
hx.AsyncClient = _CA
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)
with open('/tmp/tinystories_train.txt', 'w') as f:
    for row in ds: f.write(row['text'] + '\n')
print('done')
"
# Then use: python scripts/train.py --data-file /tmp/tinystories_train.txt ...
```

---

## Experiment A — Baseline (no episodic memory)

DrexTransformer without MemoryModule. This is the control condition.

### Config

| Parameter | Value |
|---|---|
| d_model | 256 |
| n_layers | 4 |
| n_heads | 4 |
| ff_mult | 4 |
| vocab_size | 256 (char-level) |
| window_size | 512 |
| batch_size | 8 |
| segment_len | 512 |
| dropout | 0.1 |
| lr | 3e-4 |
| warmup_steps | 2000 |
| steps | 50000 |
| grad_clip | 1.0 |
| use_episodic_memory | False |
| reset_on_boundary | True |
| ~parameters | ~6.3M (estimated) |

### Command

```bash
PYTHONPATH=python python3.12 scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --ckpt-dir checkpoints/exp_a \
  --save-every 5000 \
  --no-ssl-verify \
  2>&1 | tee results/exp_a_train.log
```

### Training log (fill in when run)

| Metric | Value |
|---|---|
| Final train loss | — |
| Final train ppl | — |
| Final val loss | — |
| Final val ppl | — |
| Wallclock time | — |
| Effective throughput | — tok/s |
| NaN skips total | — |

---

## Experiment B — Episodic Memory (thresh\*=0.70)

DrexTransformer with MemoryModule enabled. Identical config to Exp A except
`--use-episodic-memory --episodic-gate-thresh 0.70`.

### Config

Same as Experiment A, plus:

| Parameter | Value |
|---|---|
| use_episodic_memory | True |
| episodic_gate_thresh | 0.70 (thresh* confirmed exp_48_1) |
| ~parameters | ~8.1M (estimated: +4 × 2 × (256/2)² × 5 params per MemoryModule) |

### Command

```bash
PYTHONPATH=python python3.12 scripts/train.py \
  --steps 50000 --log-every 200 \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --batch-size 8 --segment-len 512 --dropout 0.1 \
  --lr 3e-4 --warmup-steps 2000 --grad-clip 1.0 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --val-every 1000 --val-max-chars 500000 \
  --reset-on-boundary \
  --ckpt-dir checkpoints/exp_b \
  --save-every 5000 \
  --no-ssl-verify \
  2>&1 | tee results/exp_b_train.log
```

### Training log (fill in when run)

| Metric | Value |
|---|---|
| Final train loss | — |
| Final train ppl | — |
| Final val loss | — |
| Final val ppl | — |
| Wallclock time | — |
| Effective throughput | — tok/s |
| NaN skips total | — |
| Mean write rate (final 1000 steps) | — |
| Write rate range (final 1000 steps) | — |

---

## Evaluation

Run after both training experiments complete.

### Passkey Recall

```bash
# Experiment A baseline
PYTHONPATH=python python3.12 scripts/eval_passkey.py \
  --checkpoint checkpoints/exp_a/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --lengths 512 1024 2048 4096 8192 16384 \
  --trials 20 \
  --device mps \
  2>&1 | tee results/exp_a_passkey.log

# Experiment B episodic memory
PYTHONPATH=python python3.12 scripts/eval_passkey.py \
  --checkpoint checkpoints/exp_b/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --report-write-rate \
  --lengths 512 1024 2048 4096 8192 16384 \
  --trials 20 \
  --device mps \
  2>&1 | tee results/exp_b_passkey.log
```

### BABILong

```bash
# Experiment A baseline
PYTHONPATH=python python3.12 scripts/eval_babilong.py \
  --checkpoint checkpoints/exp_a/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --lengths 2048 4096 8192 \
  --tasks 1 2 3 4 5 --trials 20 \
  --device mps \
  2>&1 | tee results/exp_a_babilong.log

# Experiment B episodic memory
PYTHONPATH=python python3.12 scripts/eval_babilong.py \
  --checkpoint checkpoints/exp_b/step_0050000_final.safetensors \
  --d-model 256 --n-layers 4 --n-heads 4 --ff-mult 4 \
  --window-size 512 \
  --use-episodic-memory --episodic-gate-thresh 0.70 \
  --report-write-rate \
  --lengths 2048 4096 8192 \
  --tasks 1 2 3 4 5 --trials 20 \
  --device mps \
  2>&1 | tee results/exp_b_babilong.log
```

---

## Results Table (fill in when run)

### Passkey Recall Accuracy

| Model | 512 ctx | 1k ctx | 2k ctx | 4k ctx | 8k ctx | 16k ctx |
|---|---|---|---|---|---|---|
| Exp A (baseline) | — | — | — | — | — | — |
| Exp B (episodic) | — | — | — | — | — | — |
| Δ (B − A) | — | — | — | — | — | — |

### BABILong Accuracy at 8k context

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Mean |
|---|---|---|---|---|---|---|
| Exp A (baseline) | — | — | — | — | — | — |
| Exp B (episodic) | — | — | — | — | — | — |

### Write Rate Validation (Exp B, post-training)

| Context Length | mean_wr | min_wr | max_wr | In range? |
|---|---|---|---|---|
| 2048 | — | — | — | — |
| 4096 | — | — | — | — |
| 8192 | — | — | — | — |

*Target write rate range: [0.10, 0.85] per Hard Constraint #5 in ARCHITECTURE_FINDINGS.md.*

---

## Interpretation Guide

**What a successful result looks like:**
- Exp B outperforms Exp A on passkey recall specifically at context lengths ≥ 2k (where
  the passkey reward is outside the 512-token sliding window)
- Exp B shows any improvement on BABILong Task 5 (count after drop) — the hardest task
  requiring multi-step reasoning over distractors
- Write rates at eval time fall within [0.10, 0.85]

**What a null result would mean:**
- If Exp A ≈ Exp B on both benchmarks: episodic memory provides no benefit at this
  model size / training scale. This is important to know before the paper goes out.
  It would motivate ablations at larger model sizes or different training data.

**What a concerning result would look like:**
- Write rate outside [0.10, 0.85] at eval time (after training): the trained model may
  have learned to suppress or saturate the gate. Check with `--report-write-rate`.
- Val loss significantly worse for Exp B than Exp A: MemoryModule may be adding
  harmful gradient noise at this scale. Check NaN skip counts in the training log.

---

## Ablations (Phase 16)

After Exp A/B complete, run these ablations to elevate medium-confidence components
(§12.2 in ARCHITECTURE_FINDINGS.md) to high confidence or refute them:

```bash
# Ablation C — no null retrieval gate
# Requires: add --no-null-gate flag to train.py (not yet implemented)

# Ablation D — full-sequence residual (not last-token-only)
# Requires: change x[:, -1] = x[:, -1] + mem_r to a full-sequence residual

# Ablation E — L4 only (disable L2 InfiniAttention)
# Requires: add --no-l2 flag to train.py
```
