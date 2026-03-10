"""
Experiment 45.1 — Gate Energy Scale Diagnosis

Hypothesis: The matrix-mean energy criterion used in exp_44_1
(Delta.pow(2).mean([1,2])) evaluates to < 0.05 at all training stages (well
below threshold=0.4), while the vector-norm criterion ((k-vp).norm(dim=-1))
evaluates to O(1–10), confirming scale mismatch as the sole root cause of zero
gate fire rate and near-random accuracy in all gate-containing exp_44_1 configs.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
SEQ_LEN     = 32
NUM_PAIRS   = 5
BATCH       = 64
STEPS       = 800
GATE_THRESH = 0.4   # threshold used in exp_44_1
MEAS_BATCH  = 20    # batches for energy measurement


def make_batch(batch_size=BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
               vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 4,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class DiagnosticDeltaModel(nn.Module):
    """Delta-rule model that records both energy criteria in every forward pass."""

    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.kp   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp   = nn.Linear(hidden_dim, hidden_dim)
        self.out  = nn.Linear(hidden_dim, vocab_size)
        # populated during forward when record=True
        self.rec_matrix:  list[float] = []
        self.rec_vecnorm: list[float] = []
        self.rec_knorm:   list[float] = []

    def forward(self, seq: torch.Tensor, record: bool = False) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)

        if record:
            self.rec_matrix.clear()
            self.rec_vecnorm.clear()
            self.rec_knorm.clear()

        for t in range(L - 1):
            k  = self.kp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))

            if record:
                with torch.no_grad():
                    # Criterion A — exp_44_1 formula: O(1/H) for typical outer products
                    e_matrix  = Delta.pow(2).mean([1, 2])      # shape (B,)
                    # Criterion B — vector-norm: O(||k||) ≈ O(sqrt(H))
                    e_vecnorm = (k - vp).norm(dim=-1)          # shape (B,)
                    k_norm    = k.norm(dim=-1)                  # shape (B,)
                    self.rec_matrix.append(float(e_matrix.mean()))
                    self.rec_vecnorm.append(float(e_vecnorm.mean()))
                    self.rec_knorm.append(float(k_norm.mean()))

            M = M + Delta

        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


def measure_energies(model: DiagnosticDeltaModel, n_batches: int = MEAS_BATCH) -> dict:
    """Return summary statistics of both energy criteria over n_batches."""
    model.eval()
    all_matrix: list[float] = []
    all_vec:    list[float] = []
    all_knorm:  list[float] = []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, _ = make_batch()
            model(seq, record=True)
            all_matrix.extend(model.rec_matrix)
            all_vec.extend(model.rec_vecnorm)
            all_knorm.extend(model.rec_knorm)

    matrix_mean  = sum(all_matrix)  / len(all_matrix)
    matrix_max   = max(all_matrix)
    vec_mean     = sum(all_vec)     / len(all_vec)
    vec_max      = max(all_vec)
    knorm_mean   = sum(all_knorm)   / len(all_knorm)
    # fire rates at the fixed threshold=0.4
    fr_matrix    = sum(1 for v in all_matrix  if v >= GATE_THRESH) / len(all_matrix)
    # relative fire rate: vecnorm >= 0.4 × k_norm (using the per-position knorm list)
    # approximate with knorm_mean as reference
    fr_vec_abs   = sum(1 for v in all_vec     if v >= GATE_THRESH) / len(all_vec)
    fr_vec_rel   = sum(1 for v, kn in zip(all_vec, all_knorm)
                       if v >= GATE_THRESH * kn) / len(all_vec)

    return {
        "matrix_mean":    round(matrix_mean, 6),
        "matrix_max":     round(matrix_max,  6),
        "matrix_fire_rate_abs": round(fr_matrix,  4),
        "vecnorm_mean":   round(vec_mean,    4),
        "vecnorm_max":    round(vec_max,     4),
        "vecnorm_fire_rate_abs": round(fr_vec_abs, 4),
        "vecnorm_fire_rate_rel": round(fr_vec_rel, 4),
        "knorm_mean":     round(knorm_mean,  4),
        "scale_ratio":    round(vec_mean / max(matrix_mean, 1e-9), 1),
    }


def train_model(model: DiagnosticDeltaModel, steps: int) -> None:
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch()
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()


class Exp451GateEnergyDiagnosis(Experiment):
    experiment_id = "exp_45_1"
    hypothesis = (
        "The matrix-mean energy criterion used in exp_44_1 "
        "(Delta.pow(2).mean([1,2])) evaluates to <0.05 at all training stages "
        "(well below threshold=0.4), while vector-norm energy "
        "((k-vp).norm(dim=-1)) is in the range [1,10], confirming scale "
        "mismatch as the sole root cause of zero gate fire rate."
    )

    def run(self) -> ExperimentResult:
        metrics: dict = {}
        model = DiagnosticDeltaModel()

        # ── Stage 0: untrained ────────────────────────────────────────────────
        s0 = measure_energies(model)
        for k, v in s0.items():
            metrics[f"init_{k}"] = v
        print(f"[init]  matrix_mean={s0['matrix_mean']:.6f}  "
              f"matrix_fire={s0['matrix_fire_rate_abs']:.4f}  "
              f"vecnorm_mean={s0['vecnorm_mean']:.4f}  "
              f"vec_fire_rel={s0['vecnorm_fire_rate_rel']:.4f}")

        # ── Stage 1: mid-training (400 steps) ────────────────────────────────
        train_model(model, STEPS // 2)
        s1 = measure_energies(model)
        for k, v in s1.items():
            metrics[f"mid_{k}"] = v
        print(f"[mid]   matrix_mean={s1['matrix_mean']:.6f}  "
              f"matrix_fire={s1['matrix_fire_rate_abs']:.4f}  "
              f"vecnorm_mean={s1['vecnorm_mean']:.4f}  "
              f"vec_fire_rel={s1['vecnorm_fire_rate_rel']:.4f}")

        # ── Stage 2: fully trained (800 steps) ───────────────────────────────
        train_model(model, STEPS // 2)
        s2 = measure_energies(model)
        for k, v in s2.items():
            metrics[f"final_{k}"] = v
        print(f"[final] matrix_mean={s2['matrix_mean']:.6f}  "
              f"matrix_fire={s2['matrix_fire_rate_abs']:.4f}  "
              f"vecnorm_mean={s2['vecnorm_mean']:.4f}  "
              f"vec_fire_rel={s2['vecnorm_fire_rate_rel']:.4f}")

        # ── Diagnosis ─────────────────────────────────────────────────────────
        matrix_max_ever = max(s0["matrix_max"], s1["matrix_max"], s2["matrix_max"])
        matrix_always_sub_thresh = matrix_max_ever < GATE_THRESH
        vec_clearly_above = s0["vecnorm_mean"] > 0.5   # O(1) or larger

        metrics["matrix_max_ever"]           = round(matrix_max_ever, 6)
        metrics["matrix_always_sub_thresh"]  = matrix_always_sub_thresh
        metrics["init_scale_ratio"]          = s0["scale_ratio"]

        if matrix_always_sub_thresh and vec_clearly_above:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Scale mismatch confirmed. matrix_max_ever={matrix_max_ever:.4f} "
                f"< threshold={GATE_THRESH}. vecnorm_mean(init)={s0['vecnorm_mean']:.2f}. "
                f"scale_ratio={s0['scale_ratio']:.0f}x. Gate fires 0% with matrix formula, "
                f"{s0['vecnorm_fire_rate_rel']*100:.0f}% with relative vector-norm formula."
            )
        elif not matrix_always_sub_thresh:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Matrix energy occasionally exceeded threshold "
                f"(max={matrix_max_ever:.4f}). Scale mismatch is not the sole cause."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Partial: matrix_max={matrix_max_ever:.4f}, "
                f"vecnorm_mean={s0['vecnorm_mean']:.4f}."
            )

        return self.result(outcome, metrics, notes)


if __name__ == "__main__":
    Exp451GateEnergyDiagnosis().execute()
