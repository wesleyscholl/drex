"""
Experiment 32.3 — Deep Seed Validation: Three-Hop Chain (exp_13_2 replication)

Replicates exp_13_2 with 9 seeds {0,1,2,7,13,42,99,123,777}.
Stricter criterion: degradation_ratio > 2.0 (three-hop exceeds two-hop by 2×).
This validates the surprising Phase 3 result that three-hop *beats* two-hop.
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

KB_SIZE    = 32
HIDDEN_DIM = 64
VOCAB_SIZE = 128
STEPS      = 500
BATCH      = 32
LR         = 3e-4


def build_chains():
    A = torch.arange(0, KB_SIZE)
    B = torch.randperm(KB_SIZE) + KB_SIZE
    C = torch.randperm(KB_SIZE) + 2 * KB_SIZE
    D = torch.randint(3 * KB_SIZE, VOCAB_SIZE, (KB_SIZE,))
    return A, B, C, D


class MultiHopMemory(nn.Module):
    def __init__(self, hidden_dim, num_slots, max_hops):
        super().__init__()
        self.embed    = nn.Embedding(VOCAB_SIZE, hidden_dim)
        self.mem_keys = nn.ParameterList([nn.Parameter(torch.randn(num_slots, hidden_dim))
                                          for _ in range(max_hops)])
        self.mem_vals = nn.ParameterList([nn.Parameter(torch.randn(num_slots, hidden_dim))
                                          for _ in range(max_hops)])
        self.hop_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                        for _ in range(max_hops)])
        self.out = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.max_hops = max_hops

    def read_hop(self, query, hop_idx):
        k = F.normalize(self.mem_keys[hop_idx], dim=-1)
        q = F.normalize(self.hop_projs[hop_idx](query), dim=-1)
        attn = torch.softmax(torch.mm(q, k.t()), dim=-1)
        return torch.mm(attn, self.mem_vals[hop_idx])

    def forward(self, start_ids, num_hops):
        h = self.embed(start_ids)
        for i in range(num_hops): h = self.read_hop(h, i)
        return self.out(h)


class Exp323ThreehopChain9Seeds(Experiment):
    experiment_id = "exp_32_3"
    hypothesis = ("Deep seed validation of three-hop chain (exp_13_2): "
                  "degradation_ratio > 2.0 (three-hop exceeds two-hop by 2×).")

    def run(self) -> ExperimentResult:
        torch.manual_seed(self.seed)
        A, B, C, D = build_chains()
        a_to_b = {A[i].item(): B[i].item() for i in range(KB_SIZE)}
        b_to_c = {B[i].item(): C[i].item() for i in range(KB_SIZE)}
        c_to_d = {C[i].item(): D[i].item() for i in range(KB_SIZE)}

        model = MultiHopMemory(HIDDEN_DIM, num_slots=KB_SIZE*3, max_hops=3)
        opt   = Adam(model.parameters(), lr=LR)
        a_tensor = A; b_targets = B; c_targets = C; d_targets = D

        for step in range(STEPS):
            idx  = torch.randint(0, KB_SIZE, (BATCH,))
            task = torch.randint(0, 3, (1,)).item()
            a_ids = A[idx]
            if task == 0:   targets = B[idx]; logits = model(a_ids, num_hops=1)
            elif task == 1: targets = C[idx]; logits = model(a_ids, num_hops=2)
            else:           targets = D[idx]; logits = model(a_ids, num_hops=3)
            F.cross_entropy(logits, targets).backward(); opt.step(); opt.zero_grad()

        model.eval()
        with torch.no_grad():
            acc_single = (model(a_tensor, 1).argmax(-1) == b_targets).float().mean().item()
            acc_two    = (model(a_tensor, 2).argmax(-1) == c_targets).float().mean().item()
            acc_three  = (model(a_tensor, 3).argmax(-1) == d_targets).float().mean().item()

        degradation_ratio = acc_three / max(acc_two, 0.001)
        print(f"  Seed={self.seed}: 1-hop={acc_single:.3f} 2-hop={acc_two:.3f} "
              f"3-hop={acc_three:.3f} ratio={degradation_ratio:.3f}")

        config = dict(
            KB_SIZE=KB_SIZE, HIDDEN_DIM=HIDDEN_DIM, VOCAB_SIZE=VOCAB_SIZE,
            STEPS=STEPS, BATCH=BATCH, seed=self.seed, replicates="exp_13_2",
            param_bytes=sum(p.numel()*4 for p in model.parameters()),
            activation_bytes=BATCH * KB_SIZE * HIDDEN_DIM * 4,
        )
        metrics = dict(
            acc_single=round(acc_single, 4), acc_two=round(acc_two, 4),
            acc_three=round(acc_three, 4), degradation_ratio=round(degradation_ratio, 4),
            seed=self.seed,
        )

        if degradation_ratio > 2.0:   outcome = OUTCOME_SUPPORTED
        elif degradation_ratio < 0.50: outcome = OUTCOME_REFUTED
        else:                           outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Seed={self.seed}. degradation_ratio={degradation_ratio:.3f}. "
                 f"Accs: 1-hop={acc_single:.3f}, 2-hop={acc_two:.3f}, 3-hop={acc_three:.3f}. "
                 f"Required ratio > 2.0 to confirm three-hop beats two-hop finding.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp323ThreehopChain9Seeds().execute()
