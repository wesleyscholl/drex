# DREX_UNIFIED_SPEC.md
# Konjo AI Research · DREX Architecture Lab
# Version 0.2 · March 2026
# Status: Living Document — update before any implementation sprint

---

## PURPOSE

This document is the contract between all components in the DREX-UNIFIED
architecture.  No component gets built without its interface being defined here
first.  No integration sprint begins without all upstream component specs being
locked.  This is the single source of truth for tensor shapes, validation
criteria, training methods, and phase gates.

Rule: always reference this plan before building.  If the plan is wrong, update
the plan first, then build.

---

## ARCHITECTURE OVERVIEW

DREX-UNIFIED is a post-transformer hybrid architecture combining:

    1. HDC Encoder          — fixed random hyperdimensional projection, zero training
    2. Mamba SSM Backbone   — selective state space sequence processor, Predictive Coding
    3. DREX Controller      — small RL policy for memory routing, REINFORCE
    4. Working Memory (L1)  — Echo State Network reservoir, zero training
    5. Episodic Memory (L2) — ESN with EMA feedback controller, near-zero training
    6. Semantic Memory (L3) — small SSM trained with NoProp local block denoising
    7. Sparse Router        — top-k gating across memory tiers
    8. KAN Readout          — learnable spline output projection, closed-form fitting
    9. Reward Feedback Loop — output signal back to controller and ESN tiers

Data flow:

    INPUT → HDC ENCODER → MAMBA SSM → CONTROLLER → [L1, L2, L3]
         → SPARSE ROUTER → KAN READOUT → OUTPUT → (reward → CONTROLLER, ESN feedback)

---

## GLOBAL CONVENTIONS

    Language:       Python 3.12 primary, Rust/Candle for performance-critical ops
    Hardware:       Apple M3 16GB (primary), CUDA RTX 3090 (secondary)
    Framework:      MLX for Apple Silicon training; NumPy for reservoir and HDC
    dtype:          bfloat16 for trained components, float32 for reservoir and HDC
    Batch dims:     always (batch, sequence, features) — B × S × D
    Sequence:       variable length, no fixed maximum
    Seeds:          all fixed random components use seed=42 for reproducibility
    Testing:        100% test coverage, per-file test files, pytest
    Commits:        Conventional Commits format
    Logging:        never suppress output; log shapes at component boundaries in
                    debug mode

---

## IMPLEMENTATION STATUS

| # | Component         | Spec File               | Actual File                             | Status      |
|---|-------------------|-------------------------|-----------------------------------------|-------------|
| 1 | Input Layer       | src/input/tokenizer.py  | python/drex/training/data.py            | ✅ DONE      |
| 2 | HDC Encoder       | src/hdc/encoder.py      | python/drex/models/hdc_encoder.py       | ✅ DONE Ph24 |
| 3 | Mamba SSM         | src/backbone/mamba.py   | python/drex/models/mamba.py (PENDING)   | 🔲 Phase 25  |
| 4 | DREX Controller   | src/controller/policy.py| python/drex/models/controller.py (–)    | 🔲 Phase 26  |
| 5 | Working Mem (L1)  | src/memory/reservoir.py | python/drex/models/memory_esn.py        | ✅ DONE Ph23 |
| 6 | Episodic Mem (L2) | src/memory/episodic.py  | python/drex/models/memory.py            | ✅ DONE Ph13 |
| 7 | Semantic Mem (L3) | src/memory/semantic.py  | python/drex/models/semantic.py (–)      | 🔲 Phase 27  |
| 8 | Sparse Router     | src/router/sparse.py    | python/drex/models/router.py (–)        | 🔲 Phase 29  |
| 9 | KAN Readout       | src/readout/kan.py      | python/drex/models/kan_readout.py (–)   | 🔲 Phase 28  |
|10 | Reward Loop       | src/controller/reward.py| python/drex/models/reward.py (–)        | 🔲 Phase 26  |
|11 | Integration       | src/drex_unified.py     | python/drex/models/drex_unified.py (–)  | 🔲 Phase 30  |

Note: all new components live in `python/drex/models/`, not `src/`.
The spec uses `src/` as a logical namespace, not a filesystem path.

---

## COMPONENT 1: INPUT LAYER

    File (logical): src/input/tokenizer.py
    Actual file:    python/drex/training/data.py
    Test:           tests/python/test_data.py
    Status:         DONE

Description:
    Accepts raw bytes or BPE tokens.  Byte-level input is preferred — Mamba
    outperforms transformers on byte-level tasks and eliminates the vocabulary
    bottleneck.  BPE mode retained for compatibility benchmarking.

Input:
    raw string or byte sequence of arbitrary length

Output:
    tensor of shape (B, S) dtype int32
      B = batch size
      S = sequence length in tokens or bytes

Parameters:
    mode:        "byte" | "bpe"
    vocab_size:  int, only used in bpe mode, default 32000
    max_length:  int | None, default None (no truncation)

Validation criteria:
    [ ] byte mode: output vocab range is 0–255, no out-of-bounds values
    [ ] bpe mode: output vocab range is 0 to vocab_size - 1
    [ ] round-trip test: encode then decode recovers original string
    [ ] no padding added silently — document padding behavior explicitly

Training cost: None


---

## COMPONENT 2: HDC ENCODER

    File (logical): src/hdc/encoder.py
    Actual file:    python/drex/models/hdc_encoder.py
    Test:           tests/python/test_hdc_encoder.py (44 tests, 100% coverage)
    Status:         DONE — Phase 24, commit 999d067

Description:
    Projects token embeddings into a high-dimensional random hypervector space.
    Uses three operations: binding (element-wise multiply, encodes associations),
    bundling (element-wise add + sign normalisation, encodes composition), and
    permutation (cyclic rotation, encodes sequence position).  All projection
    matrices are fixed at initialisation and never updated.

    Current implementation: lift (d_model → hdc_dim) + readdown (hdc_dim → d_model)
    with residual merge + LayerNorm.  Training mode uses tanh; eval mode uses sign
    threshold for true bipolar HDC.

Input:
    token embeddings of shape (B, S, d_model) dtype float32

Output:
    HDC-enriched embeddings of shape (B, S, d_model) dtype float32
    (same shape as input — residual merge preserves original signal)

    Optional diagnostic output:
    hypervectors of shape (B, S, hdc_dim) — via .hypervector() method

Parameters:
    d_model:      int, input/output embedding dimension
    hdc_dim:      int, hypervector dimension, must be > d_model, default 4096
    normalize:    bool, L2-normalise hypervectors before readdown, default True
    seed:         int, random seed for projection matrices, default 0

Internal matrices (fixed, never trained — registered as buffers):
    W_lift:  shape (d_model, hdc_dim)
    W_down:  shape (hdc_dim, d_model)

Trainable parameters:
    LayerNorm: 2 × d_model parameters (weight + bias)
    Total trainable: ZERO (only LayerNorm, no projection training)

Primitive functions (also implemented in hdc_encoder.py):
    hdc_bind(a, b)          → a * b (element-wise)
    hdc_bundle(a, b)        → normalise(a + b)
    hdc_permute(x, shifts)  → roll(x, shifts, dim=-1)

Validation criteria (all passing):
    [x] similarity test: cosine_sim(encode(A), encode(A)) ≈ 1.0
    [x] dimensionality test: output shape is (B, S, d_model)
    [x] zero trainable params from projections
    [x] buffers not updated by optimizer
    [x] reproducibility: same seed → same weights
    [x] training/eval outputs differ (tanh vs sign)
    [x] no NaN at L=1 or L=512
    [x] gradient flows through LayerNorm

Open question:
    Optimal d_hdc for language tasks.  10,000 theoretically motivated but
    untested at this stack depth.  Start at 4096 (current default) and scale
    if HDC orthogonality tests degrade past threshold.

Training cost: Zero — random initialisation only, no gradient operations


---

## COMPONENT 3: MAMBA SSM BACKBONE

    File (logical): src/backbone/mamba.py
    Actual file:    python/drex/models/mamba.py
    Test:           tests/python/test_mamba.py
    Status:         PENDING — Phase 25

Description:
    Selective state space model for sequence processing.  Replaces transformer
    self-attention.  O(n) training via parallel scan, O(1) inference memory per
    token.  Trained using Predictive Coding — each layer independently minimises
    its local prediction error against the layer above.  No global backward pass
    through the full network.

Input:
    HDC-enriched embeddings: shape (B, S, d_model)
    (d_hdc → d_model projection handled internally if d_hdc ≠ d_model)

Output:
    hidden state tensor:       shape (B, S, d_model)
    final recurrent state:     shape (B, d_state × n_layers)
    (recurrent state used by controller and ESN feedback)

Parameters:
    d_model:   int, model dimension, default 256
    d_state:   int, SSM state dimension, default 16
    d_conv:    int, local convolution width, default 4
    expand:    int, inner dimension multiplier, default 2
    n_layers:  int, number of Mamba blocks, default 4
    dt_rank:   "auto" | int, delta rank, default "auto"

Predictive Coding training:
    Each layer l maintains a local target: the representation produced by
    layer l+1 on the previous step.
    Local loss for layer l: MSE(output_l, sg(target_l))  (sg = stop_gradient)
    Layers train in parallel; no sequential gradient dependency.
    No global loss signal flows backward through more than one layer.
    Top layer uses task loss as its local target.

Validation criteria:
    [ ] shape test: output shape is (B, S, d_model)
    [ ] causality test: output at t depends only on 0..t, never t+1..S
    [ ] recurrence test: final state changes when input changes
    [ ] PC convergence test: all local layer losses decrease simultaneously
    [ ] equivalence test: PC-trained Mamba within 10% perplexity of same-size
        backprop-trained Mamba on WikiText-2 at 10M tokens

Training cost: Low — local per-layer losses only, no full backward graph
Training method: Predictive Coding, local MSE targets, parallel layer updates

Implementation notes:
    - Use mamba-ssm or MLX port (mamba.mlx) for Apple Silicon
    - TBPTT boundary reset: Mamba has its own hidden state; apply same boundary
      reset logic as existing LayerState in the transformer
    - Gradient checkpointing: Mamba blocks support this


---

## COMPONENT 4: DREX CONTROLLER

    File (logical): src/controller/policy.py
    Actual file:    python/drex/models/controller.py
    Test:           tests/python/test_controller.py
    Status:         PENDING — Phase 26

Description:
    A small RL policy that decides what to write to each memory tier, what to
    read, and which sparse execution paths to activate.  Operates on concatenated
    HDC hypervectors and Mamba hidden state.  Output is a discrete action vector.
    Trained via REINFORCE with reward from downstream prediction accuracy.

    Discrete actions used specifically because Phase 7 multi-stability finding
    showed continuous differentiable write gates develop initialization-dependent
    equilibria.  A discrete RL policy sidesteps this entirely.
    Key difference from exp_7_1 REINFORCE failure: the controller here operates
    on DETACHED representations (HDC + Mamba state) and trains via RL reward, NOT
    backprop through the gate.  The encoder gradient does not go to zero.

Input:
    hdc_state:   shape (B, d_hdc) — HDC encoding of last input token
    mamba_state: shape (B, d_model) — last Mamba recurrent state
    concatenated: shape (B, d_hdc + d_model) — controller input

Output:
    write_decisions: shape (B, n_tiers) dtype int32, values in {0=skip, 1=write, 2=overwrite}
    read_weights:    shape (B, n_tiers) dtype float32, softmax over tiers
    sparse_gates:    shape (B, n_modules) dtype bool

Parameters:
    d_hdc:       int, must match HDC encoder d_hdc
    d_model:     int, must match Mamba backbone d_model
    n_tiers:     int, number of memory tiers, default 3
    n_modules:   int, number of sparse execution modules, default 4
    hidden_dim:  int, policy network hidden size, default 128
    n_layers:    int, policy network depth, default 2
    gamma:       float, REINFORCE discount factor, default 0.99
    lr:          float, policy learning rate, default 1e-4

Reward signal:
    quality_reward  = -(current_loss - previous_loss) * lambda_quality
    sparsity_reward = -sum(write_decisions) * lambda_sparse
    total_reward    = quality_reward + sparsity_reward

    lambda_sparse:  float, sparsity penalty, default 0.01
    lambda_quality: float, quality weight, default 1.0

Validation criteria:
    [ ] action space test: all outputs valid discrete values within defined ranges
    [ ] learning test: better-than-random routing within 1000 episodes on synthetic
        task where correct tier is known
    [ ] sparsity test: average write operations per step decreases over training
    [ ] stability test: policy gradient variance bounded, no reward collapse

Training cost: Tiny — 2-layer MLP, ~50K parameters, CPU-trainable
Training method: REINFORCE with baseline subtraction


---

## COMPONENT 5: WORKING MEMORY — L1 ESN RESERVOIR

    File (logical): src/memory/reservoir.py
    Actual file:    python/drex/models/memory_esn.py (EchoStateMemory)
    Test:           tests/python/test_memory_esn.py (Phase 23)
    Status:         DONE — Phase 23, commit 170ff80

Description:
    Sparsely connected (~1% density) recurrent network with fixed random weights.
    Never updated after initialisation.  Creates a high-dimensional echo of recent
    inputs.  Linear readout is the only trained component.  Output feedback from
    the controller creates attractor states.

Input:
    write signal from controller: shape (B, d_model)
    read request: bool

Output:
    reservoir state: shape (B, N_reservoir)
    read output:     shape (B, d_read)

Parameters:
    n_reservoir:      int, default = d_model × esn_reservoir_mult
    spectral_radius:  float, default 0.95, must be < 1.0
    sparsity:         float, connection density, default 0.01
    d_read:           int, readout dimension, must match d_model
    feedback:         bool, output-to-reservoir feedback, default True

Trained component:
    W_readout fitted via ridge regression (closed form, no gradient)

Validation criteria (all passing — Phase 23):
    [x] echo state property: two runs with same input converge within washout steps
    [x] spectral radius < 1.0
    [x] ridge regression solve < 10s for N=2000 on CPU
    [x] write rate in [0.10, 0.85] during training (from MemoryModule integration)
    [x] reservoir buffers receive zero gradient from optimizer

Pending experiments:
    exp_53: ESN vs MemoryModule associative recall (gated on Exp A/B baseline)
    exp_54: output feedback → attractor states (expected 30–60% error reduction)

Training cost: Zero for reservoir, milliseconds for readout
Training method: Ridge regression (closed form), readout only


---

## COMPONENT 6: EPISODIC MEMORY — L2

    File (logical): src/memory/episodic.py
    Actual file:    python/drex/models/memory.py (MemoryModule, Phase 13)
    Test:           tests/python/test_memory.py
    Status:         DONE — Phases 11–13

Description:
    Extends the L1 reservoir with learned feedback and EMA delta writes.
    Stores episode-level context.  EMA decay with alpha(L) = 0.95^(96/L)
    (validated Phase 11).  Write gate at thresh* = 0.70 (validated Phase 12).

Input:
    write signal: shape (B, d_model)
    previous episodic state: shape (B, d_model)

Output:
    episodic state: shape (B, d_model)
    read output:    shape (B, d_model)

Key formula (EMA delta write):
    delta = new_input - previous_state
    new_state = alpha(L) * previous_state + (1 - alpha(L)) * delta
    where alpha(L) = 0.95^(96/L)

OR write gate:
    fire when: ||k - vp|| >= thresh* * ||k||  where thresh* = 0.70

Validation criteria (all passing — Phase 11–12):
    [x] EMA stability: converges to stable attractor under repeated identical input
    [x] write rate in [0.10, 0.85] for L=32 (wr≈0.58) and L=96 (wr≈0.42)
    [x] alpha(L) formula produces tau/L ≈ 0.21 across L=32–128
    [x] thresh* = 0.70 confirmed on ≥2/3 seeds (exp_48_1, Phase 12)

Training cost: Near Zero — EMA alpha is a scalar hyperparameter


---

## COMPONENT 7: SEMANTIC MEMORY — L3

    File (logical): src/memory/semantic.py
    Actual file:    python/drex/models/semantic.py
    Test:           tests/python/test_semantic.py
    Status:         PENDING — Phase 27

Description:
    A small trained SSM storing compressed world knowledge.  Trained using NoProp
    — each block independently denoises a noisy version of its target label.  No
    global backpropagation.  Updates its own weights during inference for
    continual learning without catastrophic forgetting.

Input:
    write signal: shape (B, d_model)
    query:        shape (B, d_model)

Output:
    retrieved knowledge: shape (B, d_model)

Parameters:
    d_model:           int, default 256
    n_blocks:          int, number of NoProp blocks, default 4
    noise_std:         float, denoising noise level, default 0.1
    inference_lr:      float, inference-time update rate, default 1e-5
    update_at_inference: bool, default True

NoProp training per block (reference: arXiv 2503.24322):
    y_noisy = y_clean + Normal(0, noise_std)
    block_loss = MSE(block_output, y_noisy)
    blocks train independently in parallel — no inter-block gradients

Phase 22 engineering notes (directly applicable):
    - fix from Phase 22 APPLIES: block optimisers must own ONLY block-specific
      params (not shared head params)
    - shared head optimizer updated once per global step
    - verify via gradient graph inspection: no gradient between blocks

Validation criteria:
    [ ] NoProp convergence: all block losses decrease independently
    [ ] no-backprop test: zero gradient between blocks (assert via grad graph)
    [ ] accuracy test: NoProp L3 within 5% of backprop baseline on CIFAR-100
    [ ] continual learning: after 10 sequential tasks, task 1 accuracy within 10%
    [ ] inference update: new content improves retrieval without degrading old

Training cost: Low — local block losses, no full backward graph
Training method: NoProp (local denoising per block), Phase 22 result applicable


---

## COMPONENT 8: SPARSE ROUTER

    File (logical): src/router/sparse.py
    Actual file:    python/drex/models/router.py
    Test:           tests/python/test_sparse_router.py
    Status:         PENDING — Phase 29

Description:
    Gates which memory tiers and downstream modules activate per input.
    Top-k gating with load-balancing auxiliary loss.  Dead modules receive zero
    compute and zero gradient.

Input:
    memory outputs: list of tensors (B, d_model), one per tier
    controller sparse_gates: shape (B, n_modules) bool
    query: shape (B, d_model)

Output:
    merged representation: shape (B, d_model)
    routing weights: shape (B, n_tiers) float

Parameters:
    n_tiers:             int, default 3
    top_k:               int, active tiers per token, default 2
    load_balance_coeff:  float, default 0.01

Gating mechanism:
    score_i = dot(query, tier_output_i)
    select top_k by score
    apply softmax over selected tiers
    output = sum(weight_i * tier_output_i)  for active tiers i
    inactive tiers: detached from computation graph, zero gradient

Load-balancing loss:
    loss_lb = load_balance_coeff * variance(fraction_routed_per_tier)
    added to total training loss to prevent tier collapse

Validation criteria:
    [ ] sparsity: exactly top_k tiers activate per token
    [ ] gradient isolation: inactive tier params receive zero gradient
    [ ] load balance: routing fraction per tier within 10% of uniform over 1000 steps
    [ ] throughput: sparse > dense by ≥20% at n_tiers=3, top_k=2

Training cost: Tiny — gating parameters only, ~10K parameters


---

## COMPONENT 9: KAN READOUT

    File (logical): src/readout/kan.py
    Actual file:    python/drex/models/kan_readout.py
    Test:           tests/python/test_kan.py
    Status:         PENDING — Phase 28

Description:
    Replaces final linear projection with a Kolmogorov-Arnold Network.
    Learnable spline functions on edges — interpretable, auditable.
    Smaller KANs match or exceed larger MLPs.

Input:
    merged representation from sparse router: shape (B, d_model)

Output:
    logits: shape (B, vocab_size) for LM, (B, n_classes) for classification

Parameters:
    d_in:          int, must match d_model, default 256
    d_out:         int, vocab_size or n_classes
    n_grid:        int, spline grid points, default 5
    spline_order:  int, B-spline order, default 3
    n_kan_layers:  int, default 2
    fit_method:    "closed_form" | "gradient", default "closed_form"

Validation criteria:
    [ ] approximation: within 2% of MLP readout accuracy on validation set
    [ ] interpretability: spline functions plottable, non-trivial learned transformations
    [ ] size: fewer parameters than equivalent MLP for same accuracy
    [ ] closed form: fit completes < 60s for d_model=256, d_out=32000 on CPU


---

## COMPONENT 10: REWARD FEEDBACK LOOP

    File (logical): src/controller/reward.py
    Actual file:    python/drex/models/reward.py
    Test:           tests/python/test_reward.py
    Status:         PENDING — Phase 26

Description:
    Computes the reward signal for the DREX controller from output quality.
    Closes the loop between output and controller.  Also provides the feedback
    signal that creates ESN attractor states in L1 and L2 memory tiers.

Input:
    predicted output: shape (B, vocab_size) or task-specific
    ground truth: shape (B,) token IDs or labels
    previous_loss: float, loss at previous step
    write_decisions: shape (B, n_tiers)

Output:
    reward: shape (B,) float
    esn_feedback: shape (B, d_read) — fed back into L1 and L2 reservoirs

Reward computation:
    quality_reward  = -(current_loss - previous_loss) * lambda_quality
    sparsity_reward = -sum(write_decisions) * lambda_sparse
    total_reward    = quality_reward + sparsity_reward

ESN feedback:
    esn_feedback = linear_projection(output_logits, d_read)
    injected into L1 and L2 reservoirs as feedback input
    this is what creates attractor states and lifts the memory ceiling

Validation criteria:
    [ ] reward sign: better predictions produce higher rewards consistently
    [ ] feedback shape: esn_feedback shape is exactly (B, d_read)
    [ ] sparsity incentive: higher write_decisions → lower reward
    [ ] attractor test: with feedback, L1 reservoir develops stable attractor
        states — measured by state convergence speed


---

## INTEGRATION SPEC

    File (logical): src/drex_unified.py
    Actual file:    python/drex/models/drex_unified.py
    Test:           tests/python/test_integration.py
    Status:         PENDING — Phase 30

Full pipeline pseudocode:

    tokens = InputLayer(raw_text)                                    # (B, S)
    hdc = HDCEncoder(tokens)                                         # (B, S, d_model)
    mamba_out, mamba_state = MambaBackbone(hdc)                      # (B,S,dm), (B,dm)
    write_decisions, read_weights, sparse_gates = Controller(
        hdc[:,-1,:], mamba_state)
    l1_out = WorkingMemory.step(mamba_state, write_decisions[:,0])   # (B, d_model)
    l2_out = EpisodicMemory.step(mamba_state, write_decisions[:,1])  # (B, d_model)
    l3_out = SemanticMemory.query(mamba_state, write_decisions[:,2]) # (B, d_model)
    merged = SparseRouter([l1_out, l2_out, l3_out], read_weights)    # (B, d_model)
    logits = KANReadout(merged)                                      # (B, vocab_size)
    reward, feedback = RewardLoop(logits, targets, write_decisions)
    WorkingMemory.receive_feedback(feedback)
    EpisodicMemory.receive_feedback(feedback)
    Controller.update(reward)

Integration validation criteria:
    [ ] shape propagation: all intermediate tensors are correct shapes end-to-end
    [ ] gradient isolation: L1 and L2 receive zero gradient from task loss
    [ ] memory tier independence: each tier can be ablated without crashing
    [ ] baseline: integrated system < bag-of-words perplexity on WikiText-2
    [ ] transformer comparison: at d_model=256, 4 Mamba layers, perplexity
        within 20% of same-parameter-count transformer on WikiText-2


---

## PHASE GATES

    Phase 1 — Free components (DONE)
    ─────────────────────────────────────────────────────────────────────────
    [x] ESN (L1) validation criteria pass                (Phase 23)
    [x] HDC Encoder validation criteria pass             (Phase 24)
    [ ] Combined HDC+ESN beats bag-of-words on POS tagging (exp_55/56 pending)
    Gate status: 2/3 — code done, evaluation experiments pending

    Phase 2 — Backbone and semantic tier (PENDING)
    ─────────────────────────────────────────────────────────────────────────
    [ ] Mamba PC convergence test passes                 (Phase 25)
    [ ] NoProp convergence test passes                   (Phase 27)
    [ ] L3 accuracy within 5% of backprop baseline
    Gate status: 0/3

    Phase 3 — Controller (PENDING)
    ─────────────────────────────────────────────────────────────────────────
    [ ] Controller achieves better-than-random routing within 1000 episodes (Phase 26)
    [ ] Sparsity incentive works (write ops decrease over training)
    [ ] Policy gradient variance bounded, no reward collapse
    Gate status: 0/3

    Phase 4 — Integration (PENDING)
    ─────────────────────────────────────────────────────────────────────────
    [ ] All integration validation criteria pass         (Phase 30)
    [ ] Ablation results documented
    [ ] Perplexity comparison to transformer baseline completed
    Gate status: 0/3

Do not proceed to the next phase until the current phase gate is met.
Document all failures — failed experiments are as important as successes.


---

## OPEN QUESTIONS

1. Optimal d_hdc for language tasks — 10,000 theoretically motivated but
   untested at this depth.  Start at 4096 (current default) and scale if
   HDC orthogonality tests degrade past cosine_sim threshold of 0.1.

2. NoProp noise_std sensitivity — the paper tested image tasks.  For language,
   the right noise level for denoising targets is unknown.  Treat as tunable
   hyperparameter; sweep {0.05, 0.1, 0.2}.

3. Controller reward delay — the quality reward requires a forward pass,
   meaning the controller gets delayed reward.  Investigate whether a learned
   value function (actor-critic instead of pure REINFORCE) is necessary for
   stable learning.

4. Inference-time semantic memory update frequency — updating every token is
   expensive.  Investigate update-every-N-tokens schedule.

5. ESN spectral radius tuning for language — 0.95 standard for time series.
   Language has different temporal correlation structure.  May need values
   closer to 0.99 for long-range dependencies.  Sweep {0.90, 0.95, 0.97, 0.99}.

6. HDC d_hdc vs d_model gap — current design requires hdc_dim > d_model.  At
   very large hdc_dim (10,000+), the readdown projection is a significant
   bottleneck.  Consider factored projections or learned compression.


---

## HARDWARE FEASIBILITY NOTES

Apple M3 16GB viability for a 1B-parameter DREX-UNIFIED model:
    ESN + HDC components:         CPU, zero training cost
    Mamba backbone + NoProp L3:   M3 GPU via MLX
    Controller:                   CPU (~50K params)
    Active training memory / step: estimated < 6GB
    (no full backward graph stored at any step)

    Comparison: transformer equivalent at 1B params → 40–80GB for same count

Realistic performance claim (honest):
    Narrow tasks (long-context reasoning, continual learning, structured
    prediction): 1–3B DREX-UNIFIED plausibly matches 7B+ transformer on specific
    benchmarks if architectural advantages hold.  Evidence base: Titans result.

    General LM benchmarks (MMLU, HellaSwag): transformer pretraining scale
    advantage will not be overcome by architecture alone at 1B params.

    Correct claim for publications: "matches or beats 7B transformers on
    long-context and continual learning tasks at 1B parameters and a fraction
    of the training cost."

POC validation targets (sufficient for a publishable paper):
    1. ESN episodic tier achieves competitive recall with zero training cost
       vs trained attention layer baseline
    2. NoProp semantic tier converges without global backprop (loss curves)
    3. Controller learns to route to the correct memory tier (routing accuracy
       on synthetic task where correct tier is known)
    4. Full integrated system achieves competitive perplexity on WikiText-2
       compared to same-size transformer


---

## CHANGELOG

    v0.1  2026-03-24  Initial spec drafted (architecture planning session)
    v0.2  2026-03-24  Adapted to actual repo structure (python/drex/models/);
                      marked Phase 23 and Phase 24 as DONE with actual commit
                      hashes; added implementation status table; expanded
                      Phase 22 NoProp engineering notes under Component 7.
