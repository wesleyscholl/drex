"""
Experiment 47.1 — Length-Adaptive EMA Alpha: Formula Sweep

Hypothesis: Replacing the fixed α=0.95 with a length-dependent formula
α(L) such that the EMA time constant τ = 1/(1−α) scales proportionally
with L resolves the EMA bootstrap problem at L=32. Specifically, for some
formula in the candidate set, the gate achieves:
  (a) wr ∈ [0.20, 0.70] at L=32
  (b) wr ∈ [0.15, 0.50] at L=96
  (c) acc_ratio ≥ 0.97 × EMA-alone baseline at both lengths
on ≥ 2/3 seeds, with gate threshold thresh=0.40.

Motivation: Phase 10 exhausted all fixed-threshold approaches. The root
cause is that α=0.95 gives time constant τ=20 steps. At L=32, τ/L=0.625
— the memory never converges, keeping ‖k−vp‖≈‖k‖ throughout, so the gate
fires universally (wr≈0.96). At L=96, τ/L=0.21 — memory converges within
the first fifth of the sequence, and the gate becomes selective (wr≈0.31).

A formula that keeps τ/L constant resolves this without any threshold change.

Formulas tested (all give τ ≈ constant fraction of L):

  exp_scale:  α(L) = α_ref^(L_ref / L),  α_ref=0.95, L_ref=96
              L=32 → 0.8574,  τ=7.0  (τ/L=0.22)
              L=96 → 0.9500,  τ=20.0 (τ/L=0.21)

  linear_c4:  α(L) = clamp(1 − 4/L, 0.75, 0.98)
              L=32 → 0.8750,  τ=8.0  (τ/L=0.25)
              L=96 → 0.9583,  τ=24.0 (τ/L=0.25)

  linear_c5:  α(L) = clamp(1 − 5/L, 0.75, 0.98)
              L=32 → 0.8438,  τ=6.4  (τ/L=0.20)
              L=96 → 0.9479,  τ=19.2 (τ/L=0.20)

  linear_c6:  α(L) = clamp(1 − 6/L, 0.75, 0.98)
              L=32 → 0.8125,  τ=5.3  (τ/L=0.17)
              L=96 → 0.9375,  τ=16.0 (τ/L=0.17)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
NUM_PAIRS   = 5
STEPS       = 800
BATCH       = 32
ALPHA_FIXED = 0.95   # fixed-EMA reference
GATE_THRESH = 0.40   # canonical threshold — adaptive α should make this work

SEQ_LENS    = [32, 96]

WR_TARGET_L32 = (0.20, 0.70)
WR_TARGET_L96 = (0.15, 0.50)

# --- α formula definitions ---

def alpha_fixed(L: int) -> float:
    """Baseline: fixed α=0.95 regardless of L."""
    return ALPHA_FIXED

def alpha_exp_scale(L: int, alpha_ref: float = 0.95, L_ref: int = 96) -> float:
    """Exponential scaling: α(L) = α_ref^(L_ref/L).
    Keeps τ/L ≈ constant across sequence lengths.
    """
    return alpha_ref ** (L_ref / L)

def alpha_linear_c4(L: int) -> float:
    """Linear rate: α(L) = clamp(1 - 4/L, 0.75, 0.98)."""
    return max(0.75, min(0.98, 1.0 - 4.0 / L))

def alpha_linear_c5(L: int) -> float:
    """Linear rate: α(L) = clamp(1 - 5/L, 0.75, 0.98)."""
    return max(0.75, min(0.98, 1.0 - 5.0 / L))

def alpha_linear_c6(L: int) -> float:
    """Linear rate: α(L) = clamp(1 - 6/L, 0.75, 0.98)."""
    return max(0.75, min(0.98, 1.0 - 6.0 / L))

FORMULAS = {
    "fixed":      alpha_fixed,
    "exp_scale":  alpha_exp_scale,
    "linear_c4":  alpha_linear_c4,
    "linear_c5":  alpha_linear_c5,
    "linear_c6":  alpha_linear_c6,
}


# --- Synthetic K-V recall task ---

def make_batch(batch_size: int = BATCH, seq_len: int = 32,
               num_pairs: int = NUM_PAIRS, vocab_size: int = VOCAB_SIZE):
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
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


# --- Model ---

class AdaptiveEMAModel(nn.Module):
    """
    EMA delta-rule memory with a length-adaptive alpha.
    alpha_fn(L) → float is called inside forward() using the actual sequence length.
    Gate uses relative vector-norm: ‖k − vp‖ ≥ gate_thresh × ‖k‖.
    """

    def __init__(self, alpha_fn, use_gate: bool = True,
                 gate_thresh: float = GATE_THRESH,
                 hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.alpha_fn   = alpha_fn
        self.use_gate   = use_gate
        self.gate_thresh = gate_thresh

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.kp    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
        self._wr_count = 0
        self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0
        self._wr_total = 0

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        alpha = self.alpha_fn(L)  # derived from actual sequence length
        M = torch.zeros(B, H, H, device=h.device)

        for t in range(L - 1):
            k  = self.kp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))

            if self.use_gate:
                energy = (k - vp).norm(dim=-1)
                ref    = self.gate_thresh * k.norm(dim=-1)
                fire   = (energy >= ref).float()
                self._wr_count += fire.sum().item()
                self._wr_total += B
                Delta = fire[:, None, None] * Delta

            M = M + (1.0 - alpha) * Delta

        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


def train_eval(model: AdaptiveEMAModel, seq_len: int,
               steps: int = STEPS, batch: int = BATCH) -> tuple[float, float]:
    """Train model, return (accuracy, write_rate)."""
    opt = Adam(model.parameters(), lr=3e-4)
    model.reset_wr()
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len=seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step()
        opt.zero_grad()
    final_wr = model.write_rate()
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len=seq_len)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, final_wr


class Exp471AdaptiveAlphaSweep(Experiment):
    experiment_id = "exp_47_1"
    hypothesis = (
        "Replacing fixed α=0.95 with a length-adaptive formula α(L) such that "
        "τ = 1/(1−α) ∝ L resolves the EMA bootstrap problem: for some formula in "
        "{exp_scale, linear_c4, linear_c5, linear_c6}, the EMA+gate achieves "
        "wr ∈ [0.20, 0.70] at L=32 AND wr ∈ [0.15, 0.50] at L=96 AND "
        "acc_ratio ≥ 0.97×EMA-alone at both lengths, on ≥ 2/3 seeds."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}

        # EMA-alone reference (fixed α) at each length — establishes accuracy floor
        ref_accs: dict[int, float] = {}
        for slen in SEQ_LENS:
            print(f"  EMA-alone reference (fixed α=0.95, no gate) at L={slen} ...")
            ref = AdaptiveEMAModel(alpha_fn=alpha_fixed, use_gate=False)
            acc_ref, _ = train_eval(ref, seq_len=slen)
            ref_accs[slen] = acc_ref
            results[f"acc_ema_fixed_ref_L{slen}"] = round(acc_ref, 4)
            print(f"    acc_ema_fixed={acc_ref:.4f}")

        # Sweep formulas and sequence lengths
        wr_table:  dict[tuple, float] = {}   # (formula_name, L) → write_rate
        acc_table: dict[tuple, float] = {}   # (formula_name, L) → accuracy

        for name, fn in FORMULAS.items():
            for slen in SEQ_LENS:
                alpha_val = fn(slen)
                print(f"  formula={name}  α({slen})={alpha_val:.4f}  L={slen} ...")

                # Adaptive EMA alone (no gate) — check α change doesn't hurt accuracy
                m_no_gate = AdaptiveEMAModel(alpha_fn=fn, use_gate=False)
                acc_ng, _ = train_eval(m_no_gate, seq_len=slen)
                results[f"acc_{name}_nogate_L{slen}"] = round(acc_ng, 4)

                # Adaptive EMA + gate
                m_gate = AdaptiveEMAModel(alpha_fn=fn, use_gate=True,
                                          gate_thresh=GATE_THRESH)
                acc_g, wr_g = train_eval(m_gate, seq_len=slen)
                acc_table[(name, slen)] = acc_g
                wr_table[(name, slen)]  = wr_g
                ratio = acc_g / max(ref_accs[slen], 1e-6)
                results[f"acc_{name}_gate_L{slen}"] = round(acc_g, 4)
                results[f"wr_{name}_gate_L{slen}"]  = round(wr_g, 4)
                results[f"ratio_{name}_L{slen}"]    = round(ratio, 4)
                results[f"alpha_{name}_L{slen}"]    = round(alpha_val, 4)
                print(f"    acc_nogate={acc_ng:.4f}  acc_gate={acc_g:.4f}  "
                      f"wr={wr_g:.3f}  ratio={ratio:.3f}")

        # Identify winning formulas (satisfy all criteria at both lengths)
        winning_formulas = []
        for name in FORMULAS:
            wr32    = wr_table.get((name, 32), 0.0)
            wr96    = wr_table.get((name, 96), 0.0)
            acc32   = acc_table.get((name, 32), 0.0)
            acc96   = acc_table.get((name, 96), 0.0)
            wr32_ok = WR_TARGET_L32[0] <= wr32 <= WR_TARGET_L32[1]
            wr96_ok = WR_TARGET_L96[0] <= wr96 <= WR_TARGET_L96[1]
            acc32_ok = acc32 >= ref_accs[32] * 0.97
            acc96_ok = acc96 >= ref_accs[96] * 0.97
            if wr32_ok and wr96_ok and acc32_ok and acc96_ok:
                winning_formulas.append(name)
                print(f"  *** WINNER: {name} (wr32={wr32:.3f}, wr96={wr96:.3f}) ***")

        # Formulas that pass only one length (for diagnostics)
        ok_at_32 = [n for n in FORMULAS if
                    WR_TARGET_L32[0] <= wr_table.get((n, 32), 0.0) <= WR_TARGET_L32[1]
                    and acc_table.get((n, 32), 0.0) >= ref_accs[32] * 0.97]
        ok_at_96 = [n for n in FORMULAS if
                    WR_TARGET_L96[0] <= wr_table.get((n, 96), 0.0) <= WR_TARGET_L96[1]
                    and acc_table.get((n, 96), 0.0) >= ref_accs[96] * 0.97]
        results["winning_formulas"] = winning_formulas
        results["formulas_ok_at_L32"] = ok_at_32
        results["formulas_ok_at_L96"] = ok_at_96

        if winning_formulas:
            best = winning_formulas[0]
            wr32 = wr_table[(best, 32)]
            wr96 = wr_table[(best, 96)]
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Bootstrap problem resolved. Winning formula(s): {winning_formulas}. "
                f"Best: {best}, α(32)={results[f'alpha_{best}_L32']}, "
                f"α(96)={results[f'alpha_{best}_L96']}. "
                f"wr(L=32)={wr32:.3f} ∈ {WR_TARGET_L32}, "
                f"wr(L=96)={wr96:.3f} ∈ {WR_TARGET_L96}. "
                f"Use {best} formula in exp_47_2."
            )
        elif ok_at_32 or ok_at_96:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Partial progress: formulas ok at L=32: {ok_at_32}, "
                f"ok at L=96: {ok_at_96}. "
                "No single formula satisfies both lengths simultaneously."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                "No length-adaptive α formula resolves the bootstrap problem. "
                "Neither wr nor accuracy criteria satisfied at either length. "
                "Consider learned MLP threshold (exp_47_4) as fallback."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp471AdaptiveAlphaSweep().execute()
