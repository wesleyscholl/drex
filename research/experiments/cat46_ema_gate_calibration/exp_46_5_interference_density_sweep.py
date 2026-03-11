"""
Experiment 46.5 — Interference Density Sweep (Gate Advantage vs ρ)

Hypothesis: The gate accuracy advantage — acc(EMA+gate, thresh*) − acc(EMA-alone) —
is positive and increases monotonically with interference density ρ = N_pairs / H,
measured across ρ ∈ {0.08, 0.12, 0.19, 0.31, 0.50, 0.75}.

Motivation: At the standard task (N_pairs=5, H=64 → ρ=0.078), the gate adds
minimal accuracy beyond EMA because the task is easy. Write selectivity buys real
gains when the memory is under capacity pressure (many pairs competing for the same
H×H matrix space). This sweep tests whether gate selectivity is a high-ρ mechanism
and whether there is a minimum ρ threshold below which the gate adds no value.

ρ grid realised as: (H=64, N_pairs ∈ {5,8,12}) + (H=32, N_pairs ∈ {10,16,24})
  H=64,  N=5  →  ρ=0.078
  H=64,  N=8  →  ρ=0.125
  H=64,  N=12 →  ρ=0.188
  H=32,  N=10 →  ρ=0.313
  H=32,  N=16 →  ρ=0.500
  H=32,  N=24 →  ρ=0.750
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

VOCAB_SIZE  = 128    # large enough for all hidden-dim variants
SEQ_LEN     = 32    # fixed sequence length for this sweep
STEPS       = 800
BATCH       = 32
ALPHA       = 0.95
CALIB_THRESH = 0.70  # from exp_46_1: best compromise with selectivity at both L=32 and L=96

# (hidden_dim, num_pairs) → interference density ρ = num_pairs / hidden_dim
DENSITY_CONFIGS = [
    (64,  5,  0.078),   # standard drex task
    (64,  8,  0.125),
    (64,  12, 0.188),
    (32,  10, 0.313),
    (32,  16, 0.500),
    (32,  24, 0.750),
]


def make_batch(batch_size=BATCH, seq_len=SEQ_LEN, num_pairs=5, vocab_size=VOCAB_SIZE):
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


class DensityModel(nn.Module):
    """EMA-alone or EMA+gate model for a given (hidden_dim, num_pairs) density config."""

    def __init__(self, use_gate=False, gate_thresh=CALIB_THRESH,
                 alpha=ALPHA, hidden_dim=64, num_pairs=5, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_gate    = use_gate
        self.gate_thresh = gate_thresh
        self.alpha       = alpha
        self.num_pairs   = num_pairs

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.kp    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
        self._wr_count = 0; self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0; self._wr_total = 0

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
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

            M = M + (1.0 - self.alpha) * Delta

        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


def train_eval(model: DensityModel, num_pairs: int,
               steps=STEPS, batch=BATCH) -> tuple[float, float]:
    opt = Adam(model.parameters(), lr=3e-4)
    model.reset_wr()
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, num_pairs=num_pairs)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    final_wr = model.write_rate()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, num_pairs=num_pairs)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, final_wr


class Exp465InterferenceDensitySweep(Experiment):
    experiment_id = "exp_46_5"
    hypothesis = (
        "The gate accuracy advantage (acc(EMA+gate) − acc(EMA-alone)) is positive "
        "and increases monotonically with interference density ρ = N_pairs/H "
        "across ρ ∈ {0.08, 0.12, 0.19, 0.31, 0.50, 0.75}."
    )

    def run(self) -> ExperimentResult:
        results: dict = {"calib_thresh_used": CALIB_THRESH}
        advantages: list[float] = []
        rhos:       list[float] = []

        for hidden_dim, num_pairs, rho in DENSITY_CONFIGS:
            tag = f"H{hidden_dim}_N{num_pairs}"
            print(f"  ρ={rho:.3f}  (H={hidden_dim}, N={num_pairs}) ...")

            ema_ref = DensityModel(
                use_gate=False, hidden_dim=hidden_dim, num_pairs=num_pairs,
            )
            acc_ema, _ = train_eval(ema_ref, num_pairs=num_pairs)

            ema_gate = DensityModel(
                use_gate=True, gate_thresh=CALIB_THRESH,
                hidden_dim=hidden_dim, num_pairs=num_pairs,
            )
            acc_gate, wr_gate = train_eval(ema_gate, num_pairs=num_pairs)

            advantage = acc_gate - acc_ema
            results[f"rho_{tag}"]         = round(rho, 4)
            results[f"acc_ema_{tag}"]     = round(acc_ema, 4)
            results[f"acc_gate_{tag}"]    = round(acc_gate, 4)
            results[f"wr_gate_{tag}"]     = round(wr_gate, 4)
            results[f"advantage_{tag}"]   = round(advantage, 4)
            advantages.append(advantage)
            rhos.append(rho)
            print(f"    acc_ema={acc_ema:.4f}  acc_gate={acc_gate:.4f}  "
                  f"adv={advantage:+.4f}  wr={wr_gate:.3f}")

        results["advantages"] = [round(a, 4) for a in advantages]
        results["rhos"]       = rhos

        # Test monotonicity: advantage should be non-decreasing with rho
        n_consistent = sum(
            1 for i in range(len(advantages) - 1)
            if advantages[i + 1] >= advantages[i] - 0.01   # 0.01 tolerance for noise
        )
        results["n_monotonic_pairs"] = n_consistent
        results["total_rho_pairs"]   = len(advantages) - 1
        all_positive = all(a > 0 for a in advantages)
        monotone     = n_consistent >= len(advantages) - 1  # all pairs consistent

        results["all_positive"]  = all_positive
        results["monotone_trend"] = monotone

        if all_positive and monotone:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Gate advantage is positive at all ρ and monotonically increases. "
                f"advantages={advantages}. "
                "Write selectivity is a universal benefit, stronger at high capacity pressure."
            )
        elif all_positive and not monotone:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Gate advantage is positive at all ρ but NOT monotone "
                f"({n_consistent}/{len(advantages)-1} pairs). "
                f"advantages={advantages}."
            )
        elif not all_positive and monotone:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Trending monotone but gate advantage is negative at low ρ "
                f"({[r for r, a in zip(rhos, advantages) if a <= 0]}). "
                f"Gate only pays off above a minimum density. advantages={advantages}."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Gate advantage is neither universally positive nor monotone with ρ. "
                f"advantages={advantages}. "
                "Write selectivity does not scale with capacity pressure at thresh*."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp465InterferenceDensitySweep().execute()
