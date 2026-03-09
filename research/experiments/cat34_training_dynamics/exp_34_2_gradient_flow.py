"""
Experiment 34.2 — Gradient Flow to Memory vs Encoder Parameters

Hypothesis: Memory-specific parameters (k_proj, v_proj, out) receive larger
gradient magnitude early in training than encoder parameters, indicating memory
shapes the loss landscape first.  Gradient ratio (memory/encoder) > 2.0 at
step 100 and < 1.5 at step 1000.

Literature basis: Hinton's "learning to think fast" — peripheral memory
modules adapt rapidly while core representations stabilise.
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

VOCAB_SIZE = 64
HIDDEN_DIM = 64
NUM_PAIRS  = 4
SEQ_LEN    = 24
STEPS      = 1200
BATCH      = 8
LR         = 3e-4
MEASURE_AT = [100, 300, 600, 1000]


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class DeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = nn.Sequential(
            nn.Embedding(VOCAB_SIZE, HIDDEN_DIM),
        )
        self.embed  = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff     = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                    nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm   = nn.LayerNorm(HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def encode(self, x):
        e = self.embed(x); return self.norm(e + self.ff(e))

    def forward(self, seq):
        hs = self.encode(seq); B, L, H = hs.shape
        M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = self.k_proj(hs[:, t, :]); v = self.v_proj(hs[:, t, :])
            k_n = F.normalize(k, dim=-1)
            vp  = torch.bmm(M, k_n.unsqueeze(-1)).squeeze(-1)
            M   = M + torch.bmm((v - vp).unsqueeze(-1), k_n.unsqueeze(1))
        q = self.q_proj(hs[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))

    def grad_norms(self):
        """Return (encoder_gnorm, memory_gnorm)."""
        enc_params   = list(self.embed.parameters()) + list(self.ff.parameters()) + list(self.norm.parameters())
        mem_params   = [self.k_proj.weight, self.v_proj.weight, self.q_proj.weight]
        enc_gnorm    = sum(p.grad.pow(2).sum().item()
                          for p in enc_params if p.grad is not None) ** 0.5
        mem_gnorm    = sum(p.grad.pow(2).sum().item()
                          for p in mem_params if p.grad is not None) ** 0.5
        return enc_gnorm, mem_gnorm


class Exp342GradientFlow(Experiment):
    experiment_id = "exp_34_2"
    hypothesis = ("Memory projection parameters (k/v/q) receive larger gradient "
                  "than encoder early in training (ratio > 2.0 at step 100) "
                  "and converge to similar magnitudes later (ratio < 1.5 at step 1000).")

    def run(self) -> ExperimentResult:
        model = DeltaModel(); opt = Adam(model.parameters(), lr=LR); model.train()
        param_bytes = sum(p.numel() * 4 for p in model.parameters())
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, measure_at=MEASURE_AT,
            param_bytes=param_bytes,
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        metrics = {}
        for step in range(1, STEPS + 1):
            seq, tgt = make_batch(BATCH)
            F.cross_entropy(model(seq), tgt).backward()
            if step in MEASURE_AT:
                enc_gn, mem_gn = model.grad_norms()
                ratio = mem_gn / max(enc_gn, 1e-8)
                metrics[f"enc_gnorm_s{step}"]  = round(enc_gn, 4)
                metrics[f"mem_gnorm_s{step}"]  = round(mem_gn, 4)
                metrics[f"ratio_s{step}"]       = round(ratio, 4)
                print(f"    step={step}: enc={enc_gn:.4f} mem={mem_gn:.4f} ratio={ratio:.3f}")
            opt.step(); opt.zero_grad()

        ratio_early = metrics.get("ratio_s100", 0.0)
        ratio_late  = metrics.get("ratio_s1000", 0.0)

        if ratio_early > 2.0 and ratio_late < 1.5:
            outcome = OUTCOME_SUPPORTED
        elif ratio_early <= 1.0:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Gradient ratio (mem/enc): early(s100)={ratio_early:.3f}, "
                 f"late(s1000)={ratio_late:.3f}. "
                 f"Hypothesis: ratio>2.0 early and <1.5 late.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp342GradientFlow().execute()
