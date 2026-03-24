# Sprint 1 — Baseline Transformer (exp_poc_a)

**Goal:** Establish the floor. Vanilla transformer + InfiniAttention L2. No episodic memory.

**Config:**
- d_model=128, n_heads=4, n_layers=4, ff_mult=4
- segment_len=128, batch_size=8, steps=10,000
- optimizer: AdamW lr=3e-4, wd=0.1, warmup=500
- val_every=500, log_every=100

**Command (reference):**
```bash
scripts/run_poc_sprint1.sh
```

---

## Results

| seed | val_ppl (step 5k) | val_ppl (step 10k) | tok/s | status |
|------|-------------------|---------------------|-------|--------|
| 42   | —                 | —                   | —     | RUNNING |
| 43   | —                 | —                   | —     | PENDING |
| 44   | —                 | —                   | —     | PENDING |

**Median val_ppl at step 10k:** TBD

**Gate to proceed to Sprint 2:** val_ppl < 2.5 at step 10k

---

## Logs

- `results/poc/sprint1_seed42.log`
- `results/poc/sprint1_seed43.log`
- `results/poc/sprint1_seed44.log`

## Checkpoints

- `checkpoints/poc_a_s42/step_0010000_final.safetensors`
- `checkpoints/poc_a_s43/step_0010000_final.safetensors`
- `checkpoints/poc_a_s44/step_0010000_final.safetensors`

---

## Notes

Fill this in after runs complete:
- Convergence behaviour:
- Wall clock time per seed:
- Any anomalies:
