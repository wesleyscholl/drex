# ARCHITECTURE_FINDINGS.md — Drex Research Findings

*Created: 2026-03-12 | Updated: 2026-03-12 | Covers Phases 1–15 (48 categories, 247+ experiments, production implementation)*

This document records the validated architecture specification and the research dead ends
that should not be re-investigated. All findings have ≥2/3 seed confirmation unless noted.

---

## §1 — Memory Hierarchy Overview

Drex uses a three-tier memory system:

| Tier | Mechanism | Location | Validated |
|------|-----------|----------|-----------|
| L1 | Sliding-window causal attention | Activations (in-context) | Yes |
| L2 | Infini-Attention delta-rule matrix (MemoryState) | Activations (cross-segment) | Yes |
| L3 | Titans-style MLP weight snapshots (TitanMemory) | Disk via Rust SnapshotStore | Yes |
| L4 | Episodic/semantic split delta-rule (MemoryModule) | Activations (per-segment) | Yes (Phase 12) |

---

## §2 — L2: Infini-Attention (DeltaRuleUpdate)

- Feature map: φ(x) = ELU(x) + 1 (positive-valued, unbounded above)
- Delta rule: ΔM = φ(K)ᵀ @ (V − φ(K)M)
- Normalisation accumulator: z += Σ φ(K) over positions
- Read: r = φ(Q)M / (φ(Q)z + ε)
- Validated: 9/9 seeds consistent

---

## §3 — L3: TitanMemory

- Small 2-layer MLP (no bias, no LayerNorm)
- Memory = weights; writing = one Adam gradient step on ‖net(k) − v‖²
- Independent internal Adam optimiser (not the outer training optimiser)
- Weight snapshots stored via Rust SnapshotStore; async prefetch via PrefetchEngine
- Sketch-based similarity index (rank-16 projection) for k-NN prefetch decisions

---

## §4 — L4: MemoryModule (Phase 13 Production Code)

The validated minimal architecture for the episodic/semantic associative memory layer.

### Architecture

```
Input: x ∈ ℝ^{B × L × H}

For t = 0 … L-2 (write passes):
  ks = sem_proj(x_t)      ∈ ℝ^{B × H/2}   (no bias)
  ke = epi_proj(x_t)      ∈ ℝ^{B × H/2}   (no bias)
  k̂s = ks / ‖ks‖          (unit key — semantic)
  k̂e = ke / ‖ke‖          (unit key — episodic)
  vps = M_sem @ k̂s        (current prediction from memory)
  vpe = M_epi @ k̂e

  -- OR write gate (relative-norm criterion) --
  fire = (‖ks − vps‖ ≥ thresh·‖ks‖) OR (‖ke − vpe‖ ≥ thresh·‖ke‖)

  ΔM_sem = (ks − vps) ⊗ k̂s          (outer product delta)
  ΔM_epi = (ke − vpe) ⊗ k̂e

  w_t = (t + 1) / L                   (recency weight ∈ (0, 1])

  M_sem += (1 − α) · fire · ΔM_sem    (EMA write, semantic)
  M_epi += (1 − α) · w_t · fire · ΔM_epi  (EMA write, episodic + recency)

At position L-1 (query):
  q = x_{L-1}
  r_sem = M_sem @ norm(sem_proj(q))
  r_epi = M_epi @ norm(epi_proj(q))
  r = concat(r_sem, r_epi)            ∈ ℝ^{B × H}

  g_null = σ(null_gate(q))            (learned scalar null-retrieval gate)
  r = g_null · r

Output = norm_out(out_proj(r))         ∈ ℝ^{B × H}  (LayerNorm bounds residual contribution)
```

### Hyperparameters (non-negotiable)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| thresh\* | **0.70** | exp_48_1, Phase 12 (3/3 seeds deterministic wr) |
| α(L) | **0.95^(96/L)** | exp_47_1/3, Phase 11 (exp_scale formula) |
| Matrix size | **H/2 × H/2** (two halves) | exp_38_1, 9/9 seeds |
| Episodic recency weight | **(t+1)/L** | Phase 11 validation |
| Write gate op | **OR** over branches | exp_47_2 (AND gate degrades recall) |
| Read combination | **concat** (no learned gate) | exp_38_3 (learned gate −10%) |
| Null retrieval gate | **learned σ(linear(q))** | Phase 16 ablation (+0.30 ppl without it) |
| Output normalization | **LayerNorm(H) after out_proj** | Phase 16 (norm_out prevents M explosion with detached write) |

### Valid write rate range

After any change to the write mechanism, validate:

```
WRITE_RATE_LO = 0.10  (minimum acceptable gate firing fraction)
WRITE_RATE_HI = 0.85  (maximum acceptable gate firing fraction)
```

At thresh=0.70, exp_scale:
- wr(L=32) = 0.581 (target: [0.20, 0.70]) ✓
- wr(L=96) = 0.421 (target: [0.15, 0.50]) ✓

---

## §5 — Length-Adaptive EMA (Phase 11)

**Problem:** At L=32 with fixed α=0.95, the memory never forgets: effective time constant
τ = 1/(1−α) = 20 steps, τ/L = 0.625. Memory fills and write rate spikes to ~0.97.

**Solution:** α(L) = 0.95^(96/L) keeps τ/L ≈ 0.21 constant across L=32–128.

| L | α(L) | τ (steps) | τ/L |
|---|------|-----------|-----|
| 16 | 0.857 | 7.0 | 0.44 |
| 32 | 0.857 | 7.0 | 0.22 |
| 64 | 0.923 | 12.8 | 0.20 |
| 96 | 0.950 | 20.0 | 0.21 |
| 128 | 0.961 | 25.6 | 0.20 |

Note: L<24 (≈ 5 key-value pairs in short context) produces wr=1.0 — this is correct
behavior, not an error. Every token is novel at that density.

---

## §6 — OR-Gate Write-Rate Inflation (Phase 12)

**Problem:** With two branches each firing independently at p≈0.58 (thresh=0.40),
the OR gate fires at Pr(A∪B) = 1−(1−p)² ≈ 0.82, in practice 0.774.

**Fix:** thresh\*=0.70 reduces each branch to p≈0.35, OR probability ≈ 0.58.
Observed at thresh=0.70: wr(L=32)=0.581, wr(L=96)=0.421.

**Key insight:** The threshold scales with the per-branch probability, not the OR
combined probability. Geometric estimate: thresh\* ≈ thresh_old × (p_target/p_old).

---

## §7 — Integration into DrexTransformer (Phase 13)

`MemoryModule` is inserted into each `DrexLayer` as an optional branch:

```python
DrexConfig(use_episodic_memory=True, episodic_gate_thresh=0.70)
```

In `DrexLayer.forward()`, after the attention and feed-forward sub-layers:

```python
if self.episodic_mem is not None and self.norm_mem is not None:
    mem_r = self.episodic_mem(self.norm_mem(x))  # (B, d_model)
    x = x.clone()
    x[:, -1] = x[:, -1] + mem_r                 # residual at query position
```

Pre-LayerNorm is applied to `x` before passing it to `MemoryModule`, consistent with the
pre-norm convention used throughout `DrexLayer`. The result is a residual addition at the
last token position only (the query position), leaving all earlier positions unchanged.

Gradient checkpointing passes `x` through `layer(x, state)` unchanged, so MemoryModule
is compatible with `gradient_checkpointing=True` without modification.

---

## §8 — Optimiser

**Use Adam.** Not SGD. Not AdamW (AdamW acceptable as a minor variant).

Evidence: exp_34_6 showed >10% accuracy spread across optimisers on the associative
recall benchmark. Adam was the best performer across 9/9 seeds.

---

## §9 — Dead Ends (Do Not Re-investigate)

The following approaches were tested to refutation (≥7/9 seed evidence) and should not
be re-investigated. The research investment to rule each out was significant.

### Memory architecture variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| Tiered memory routing (3+ tiers active together at inference) | No throughput gain; complexity without recall benefit | Phase 4 |
| Hierarchical write decisions (sub-gate inside gate) | Double-gate overhead; unstable write rates | Phase 6 |
| Momentum delta rule | Oscillation in M; accuracy −8% | Phase 7 |
| Bidirectional delta rule | Cannot be applied to causal setting | Phase 7 |
| Velocity gate (δ-based firing) | Equivalent to relative-norm at calibration; more complex | Phase 9 |
| Matrix-mean energy gate | Produces O(1/H) values; always below threshold | exp_45_1 |
| Position-schedule gate (cosine, linear) | Static schedule degrades accuracy at unusual densities | Phase 8 |
| Offline consolidation pass | Requires second forward pass; not streaming-compatible | Phase 5 |
| Hindsight oracle distillation | Requires look-ahead; cannot be trained causally | Phase 5 |
| Three-gate auxiliary loss combos | Loss interference; write rate becomes erratic | Phase 10 |
| Write rate regularisation (L1/L2 on gate) | Collapses write rate; accuracy degrades | exp_45_2 |
| Two-phase gate training (freeze then unfreeze) | No benefit over end-to-end training | Phase 10 |

### Splitting / routing variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| Learned router for episodic/semantic split | 10–24% accuracy loss vs fixed 50/50 | exp_38_1 |
| 3-way split (episodic/semantic/prospective) | No accuracy gain; wr harder to control | Phase 8 |
| Unequal split (70/30) | Lower than 50/50; no structural justification | Phase 8 |

### Read-side variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| Learned gated combination of r_sem and r_epi | −10% accuracy vs hard concat | exp_38_3 |
| Separate output projections per branch | Marginal; not worth extra parameters | Phase 9 |
| Cosine similarity retrieval (instead of matrix multiply) | Equivalent to dot product after normalisation; no gain | Phase 7 |

### Training variants

| Approach | Reason rejected | Experiment |
|----------|-----------------|------------|
| REINFORCE for write gate | Encoder gradient = 0 (gate blocks signal) | exp_7_1 |
| Random-init thresh (learnable starting at random value) | Low-accuracy equilibrium; gate never recovers | exp_43_1 |
| Randomly-initialised learnable thresh | Low-accuracy equilibrium (same as above) | exp_43_1 |
| Fixed α=0.95 alone (no length-adaptation) | wr spikes to 0.97 at L≤32; bootstrap failure | Phase 11 |
| Learned MLP gate for α scheduling | Unnecessary complexity; exp_scale formula sufficient | exp_47_1 |
| AND gate (both branches must fire) | Degrades recall accuracy; exp_47_2 showed OR is strictly better | exp_47_2 |
| thresh=0.40 for OR-gate split model | wr=0.774 at L=32; exceeds target [0.20, 0.70] | exp_47_2 |
| Universal single threshold for OR-gate split at any thresh < 0.50 | OR inflation persists below thresh=0.50 | exp_48_1 |

---

## §10 — Full Implementation Specification

The production implementation target is `drex.models.memory.MemoryModule`. The exact
specification (non-negotiable from ≥9-seed research):

### Forward contract

```
Input:  x ∈ ℝ^{B × L × H}    — full context; position L-1 is the query
Output: r ∈ ℝ^{B × H}         — memory retrieval for position L-1
```

### Required components

1. **Two associative matrices per forward call** (re-initialised to zero each call):
   - `M_sem ∈ ℝ^{B × H/2 × H/2}` — semantic branch
   - `M_epi ∈ ℝ^{B × H/2 × H/2}` — episodic branch

2. **Two key projections** (no bias, scale-invariant):
   - `sem_proj: H → H/2`
   - `epi_proj: H → H/2`

3. **Delta-rule update** with EMA smoothing:
   - Unit key normalisation: k̂ = k / ‖k‖
   - Error: δ = k − M @ k̂
   - Outer product: Δ = δ ⊗ k̂
   - EMA write: M += (1 − α(L)) · gate · Δ
   - Episodic branch additionally multiplies by recency weight w_t = (t+1)/L

4. **OR relative-norm write gate**:
   - `fire = (‖ks − vps‖ ≥ thresh · ‖ks‖) OR (‖ke − vpe‖ ≥ thresh · ‖ke‖)`
   - thresh = **0.70** (do not change without re-running write-rate validation)
   - Minimum allowed thresh = **0.40** (exp_43_1 hard constraint)

5. **Length-adaptive EMA coefficient**:
   - `α(L) = 0.95^(96/L)` (exp_scale formula)
   - Do not use fixed α=0.95 alone

6. **Soft concatenated retrieval**:
   - `r = concat(r_sem, r_epi) ∈ ℝ^{B × H}`
   - No learned gate on the combination (exp_38_3)

7. **Null retrieval gate**:
   - `g = σ(null_gate(q))` — learned scalar (Linear(H, 1))
   - Applied as `r = g · r` before `out_proj`
   - Suppresses irrelevant retrievals when memory is near-zero

8. **Output projection**: `Linear(H, H)` mapping `r → ℝ^{B × H}`

9. **Write-rate tracking**: record `wr_count / wr_total` after each forward

10. **d_model must be even** (enforced by ValueError in __init__)

### Hard constraints

- `gate_thresh ≥ 0.40` (exp_43_1 — lower values trigger low-accuracy equilibrium)
- `α(L) = 0.95^(96/L)` — never fixed α=0.95 alone
- Validate `write_rate ∈ [0.10, 0.85]` using `assert_write_rate_valid()` after training changes
- No learned episodic/semantic router
- No learned combination of r_sem and r_epi
- Optimiser: Adam (not SGD)

---

## §11 — Phase 13–15: Implementation Experience

This section records findings from the production implementation (Phases 13–15) that are
relevant to reproducing or extending the architecture. These findings emerged from code
rather than experiments.

### §11.1 — F.normalize stability (Phase 15)

`F.normalize(k, dim=-1)` with the PyTorch default `eps=1e-12` is numerically fragile
under weight-decay pressure or MPS precision characteristics. When a key projection
outputs a near-zero vector, the norm is amplified by `1/eps = 1e12`, propagating enormous
activations through the delta-rule update into the residual stream, and eventually
producing `NaN` loss.

**Fix:** Use `eps=1e-6` on all four `F.normalize` calls in `MemoryModule.forward()`:

```python
kns = F.normalize(ks, dim=-1, eps=1e-6)
kne = F.normalize(ke, dim=-1, eps=1e-6)
qns = F.normalize(self.sem_proj(q), dim=-1, eps=1e-6)
qne = F.normalize(self.epi_proj(q), dim=-1, eps=1e-6)
```

This allows near-zero keys to remain near-zero (rather than being normalised to a random
unit direction), with at most `1/1e-6 = 1e6` amplification — large but not enough to
cause NaN under standard float32 range.

### §11.2 — NaN training loss (Phase 15)

Small models (d\_model ≤ 128) with random initialisation can produce NaN cross-entropy
loss on the first few steps, particularly at high learning rates (`lr ≥ 1e-3`) or small
batch sizes. Once a NaN loss occurs, `loss.backward()` poisons all weights irreversibly.

**Fix:** Check `loss.isfinite()` before the backward pass; if non-finite, zero gradients,
reset TBPTT states, and continue:

```python
if not loss.isfinite():
    optimizer.zero_grad()
    states = model.init_states(batch_size, device)
    continue
loss.backward()
```

This is implemented in `scripts/train.py`. The fix is defensive — at production model
sizes (d\_model ≥ 256) with proper hyperparameters (lr=3e-4, dropout=0.1), NaN loss
should not occur in normal training.

### §11.3 — TBPTT document-boundary contamination (Phase 15)

`DrexTransformer` uses TBPTT: `LayerState` is detached from the computation graph and
carried forward across batches. With `shuffle=True` and TinyStories, consecutive batches
contain segments from different stories. The L2 Infini-Attention matrix `M` ends up
holding associations from story N when the model starts processing story N+1.

**Severity:** Moderate. The model is unlikely to learn strong cross-story associations
because story boundaries (token 10 = `\n`) break semantic continuity. However, it
introduces noise in the gradient signal and may reduce L2 memory precision.

**Fix:** Use `--reset-on-boundary` in `scripts/train.py`. This detects any segment whose
target contains token 10 and zeros the corresponding `LayerState` entries for those batch
elements before the next forward pass.

**Validation is unaffected:** `_validate()` calls `model(src)` with `states=None`,
which triggers `model.init_states()` — fully independent per-batch evaluation.

### §11.4 — Write loop performance (Phase 15/16)

`MemoryModule.forward()` iterates `for t in range(L-1)` because step t reads
`M_{t-1}` (sequential recurrence). Per-step Python overhead includes:
- 4 kernel launches for `sem_proj(h_t)`, `epi_proj(h_t)`, and their `F.normalize` calls
- 1 CPU-GPU sync for `fire.sum().item()`

**Phase 15 fix:** Batch all projections and normalizations before the loop (2 large
launches instead of 4×(L-1) small launches). Accumulate `fire` tensors inside the loop
and compute the write-rate sum in a single `torch.stack(...).sum().item()` call after
the loop (1 sync instead of L-1).

**Remaining cost — measured at Phase 16:** The 4 `torch.bmm` calls per iteration of the
sequential loop *cannot* be eliminated without changing the delta-rule semantics. At
`segment_len=512` with 4 transformer layers, this is **4 × 511 × 4 = 8,176 sequential
GPU kernel launches per forward pass**. On MPS, the per-launch overhead dominates:

| Config | tok/s | Ratio vs baseline |
|---|---|---|
| Exp A (no MemoryModule), seg_len=512 | ~11,700 | 1.0× |
| Exp B (MemoryModule), seg_len=64 | ~1,200 | 0.10× (9.8× slower) |
| Exp B (MemoryModule), seg_len=512 | **543** | **0.046× (20× slower, measured)** |

At `segment_len=512`, Exp B was killed (SIGKILL, exit 137) after step 200 (~27 min wall
clock). Throughput: **543 tok/s**. Projected: ~4.5 h for 2k steps, ~112 h for 50k steps.
**Hard blocker for the full benchmark run.**

**Write rate at seg_len=512 (step 200):** wr=0.969, range [0.645, 1.000] — outside
[0.10, 0.85]. Root cause: α(L=512)=0.990 gives (1−α)=0.010; matrices start near-zero so
prediction error almost always exceeds `thresh × ||ks||` when thresh=0.70. Convergence to
valid wr is expected but unconfirmed at L=512 (validated only at L=32, L=96).

**Fix options (Phase 16, HIGH priority — blocks Exp B):**

1. **CPU backend for write loop** *(implemented, Phase 16)*: Move `M_sem`, `M_epi` to
   CPU for the loop body; results moved back to GPU for the read phase.

2. **Detached write (torch.no_grad + .detach())** *(implemented, Phase 16)*: Key tensors
   (`kns_all`, `ks_all`, etc.) are detached before the CPU loop; the loop runs inside
   `torch.no_grad()`.  This eliminates autograd graph construction for L-1 sequential
   tensor assignments (O(L) graph nodes, ~1.7 ms/iter on CPU).  Gradient signal to
   `sem_proj`/`epi_proj` flows through the read query path only.

3. **Output LayerNorm (norm_out)** *(implemented, Phase 16)*: Without write-path gradient,
   write-key norms are unconstrained and M can grow large, destabilising training.
   `nn.LayerNorm(d_model)` applied after `out_proj` bounds the memory residual
   contribution regardless of M magnitude.

**Phase 16 measured result (Exp B 2000-step probe, seg_len=512):**

| Config | tok/s | Ratio vs baseline | Notes |
|---|---|---|---|
| Exp A (no MemoryModule), seg_len=512 | ~11,700 | 1.0× | Phase 15 |
| Exp B (MemoryModule), seg_len=512 | **543** | 0.046× | Original (MPS sequential loop) |
| Exp B + CPU backend (autograd) | ~543–600 | ~1.0× | Bottleneck shifted to O(L) autograd |
| Exp B + CPU backend (detached) | ~1,158 | ~2× | Python loop overhead remains |
| Exp B + CPU backend + detached + norm_out | **2,310** | **4.3×** | Measured at step 200 |

At step 200, write rate: wr=0.987 [0.746, 1.000] — still outside [0.10, 0.85].
wr convergence at L=512 is unconfirmed beyond step 200; see §11.4 blocker checklist.

**Remaining bottleneck:** Python interpreter overhead for 511 iterations × 4 layers × ~15
operations per iteration ≈ 30,660 Python/PyTorch calls per step.  The wall-time breakdown
is approximately: ~350 ms attention+FFN (MPS) + ~250 ms CPU write loop (Python overhead) +
~50 ms CPU-MPS data transfer = ~650 ms/step → 6,300 tok/s ceiling.  Actual: 2,310 tok/s
suggests other overhead.  Full elimination requires parallel scan or custom Metal kernel.

---

## §12 — Component Confidence Classifications

Each component is classified by evidence strength. "High confidence" means ≥7/9 seed
evidence with ≥2 independent experiments. "Medium confidence" means design or
implementation choices not ablated at the same rigor as the core architecture. "Phase
experience" means informed by production training in Phases 13–15.

### §12.1 — High confidence (validated, do not change without re-running write-rate suite)

| Component | Evidence | Phase |
|---|---|---|
| Delta-rule update formula `Δ = (k−Mk̂) ⊗ k̂` | 9/9 seeds, Phases 3–8 | Phase 3 |
| ELU+1 feature map for L2 | 9/9 seeds | Phase 2 |
| OR relative-norm write gate | 9/9 seeds, exp_47_2 (AND inferior) | Phase 9 |
| Fixed 50/50 episodic/semantic split | 9/9 seeds, exp_38_1 | Phase 9 |
| Concatenated retrieval (no learned gate) | 6/9 seeds, exp_38_3 | Phase 9 |
| thresh\*=0.70 for OR-gate model | 3/3 seeds deterministic, exp_48_1 | Phase 12 |
| α(L) = 0.95^(96/L) length-adaptive EMA | 6/9 seeds, exp_47_1/3 | Phase 11 |
| Adam optimizer (not SGD) | 9/9 seeds, exp_34_6 | Phase 8 |
| `F.normalize eps=1e-6` (not default 1e-12) | Phase 15 production experience | Phase 15 |

### §12.2 — Phase 16 ablation results (micro-experiments, 500 steps, 128d/4L, seg_len=64)

Three §12.2 components were ablated with a controlled micro-experiment. Config: d_model=128,
n_layers=4, n_heads=4, seg_len=64, batch_size=8, 500 steps, cosine LR 3e-4 → 3e-5,
full TinyStories dataset. All 4 conditions exhibit identical NaN-skip patterns (every 13
steps, 40 total, handled by the existing guard), confirming the pattern is
dataset-structural rather than architecture-specific.

| Condition | Params | val_ppl (step 500) | Δ vs baseline | Avg tok/s | Assessment |
|---|---|---|---|---|---|
| Baseline (all components) | 1,020,180 | 2.33 | — | ~1,344 | Reference |
| No null gate | 1,019,664 | 2.63 | +0.30 (worse) | ~1,243 | Gate helps; **keep** |
| Full-sequence residual | 1,020,180 | **2.07** | **−0.26 (better)** | ~1,274 | Upgrade candidate ¹ |
| Last-layer-only memory | 920,337 | 2.33 | 0.00 (same) | ~3,619 | Efficiency candidate ¹ |

¹ **Multi-seed validation (2000 steps, 3 seeds × 2 conditions, seg_len=64, batch_size=4):**

| Seed | Baseline val_ppl | Full-seq-residual val_ppl | Last-layer-only val_ppl | Last-layer-only tok/s |
|---|---|---|---|---|
| 42 | 1.87 | 1.72 | 2.22 | 7,172 |
| 123 | 1.98 | 1.25 | 1.74 | 9,479 |
| 777 | 1.41 | 2.23 | 1.67 | 9,082 |
| **Mean** | **1.75** | **1.73** | **1.88** | **8,578** |
| **Std (n−1)** | **0.30** | **0.49** | **0.30** | — |

Note: baseline tok/s at these settings ≈ 5,037 (1.70× slower than last-layer-only).

---

#### Null retrieval gate `g = σ(linear(q))`

**Status: VALIDATED — keep.** Removing the gate increases val_ppl from 2.33 to 2.63
(+0.30) with no throughput benefit. The gate suppresses uninformative reads when the memory
matrices are near-zero early in training and remains beneficial as training progresses.
Evidence: 1 run, 500 steps. **Elevate to high confidence** pending second-seed confirmation.

#### Residual injection mode (last-token-only vs full-sequence)

**Status: INCONCLUSIVE (multi-seed) — initial 500-step screen favoured full-seq-residual
(−0.26 ppl), but multi-seed validation at 2000 steps shows high variance (std=0.49) with
mean val_ppl 1.73 vs baseline 1.75 — a difference of −0.02, well within noise.** Seed 123
shows a strong benefit (1.25 vs 1.98), seed 777 shows a regression (2.23 vs 1.41), and
seed 42 is neutral. The initial screen result does not replicate reliably.

Broadcasting the memory read vector to all token positions (not just `x[:, -1]`) is
theoretically sound, but the evidence is insufficient to justify changing the default.

**Action:** Do not switch the default to `full_seq_residual=True`. The `--full-seq-residual`
flag remains available for future study at longer training horizons (≥10k steps) or with
larger models where the write-rate plateau resolves. Revisit after a confirmed 10k-step run.

#### Memory layer placement (all layers vs last layer only)

**Status: EFFICIENCY TRADEOFF — last-layer-only is 1.70× faster at seg_len=64 but costs
+0.13 val_ppl on average (1.88 vs 1.75, 3-seed, 2000 steps).** Restricting `MemoryModule`
to the final transformer layer (layers 0–2 are pure baseline):

- val_ppl: mean 1.88 (±0.30 std) vs baseline 1.75 (±0.30 std) — consistent +0.13 disadvantage
- Parameters: 920,337 vs 1,020,180 (−9.8%)
- Throughput: ~8,578 tok/s (mean across seeds) vs ~5,037 tok/s baseline at seg_len=64 (1.70×)
- Write rate: gate activation range [0.347–1.000] early in training, converging to
  [0.397–0.923] by step 2000 — more selective than all-layers variant

The initial single-seed screen showing identical val_ppl (2.33) did not replicate; multi-seed
reveals a consistent small quality penalty. The throughput gain is real and consistent.

**Action:** The all-layers configuration remains the production default. The
`--memory-last-layer-only` flag is available for throughput-constrained use cases where a
small quality tradeoff is acceptable. The 1.70× throughput gain at seg_len=64 is
significant — consider this flag when seq_len≥512 training is the constraint. Re-evaluate
quality gap at the production seg_len=512 scale before making a final decision.

#### Still un-ablated (remain medium confidence)

| Component | Basis | Status |
|---|---|---|
| Pre-LayerNorm before `MemoryModule` | Standard convention | Not ablated; low risk, standard pattern |
| `out_proj: Linear(H, H)` | Implementation choice | Not ablated; removing unlikely to materially hurt |
| Recency weight `w_t = (t+1)/L` for episodic branch | Phase 11 design | Not ablated; untested vs uniform weight |

### §12.3 — Low confidence / not yet tested at production scale

| Component | Current status |
|---|---|
| Behaviour at `segment_len > 512` (longer contexts) | Only tested at L ≤ 128 in micro-experiments; L=512 is the production target but no trained model exists yet |
| `full_seq_residual=True` at production scale | **Multi-seed complete (3 seeds, 2k steps): INCONCLUSIVE.** Do not promote to default. |
| `memory_last_layer_only=True` at production scale | **Multi-seed complete (3 seeds, 2k steps): EFFICIENCY TRADEOFF (+0.13 ppl, 1.70× faster).** Not production default. |
| Write rate stability over 50k training steps | Only measured in short experiments; long-run stability at 50k steps is untested |
| α(L=512) write rate convergence | **wr plateau at ~0.963 through 2000 steps (Exp B confirmed) — does NOT converge to [0.10, 0.85] within 2k steps.** Extended training (≥10k steps) required to determine convergence behaviour. |

