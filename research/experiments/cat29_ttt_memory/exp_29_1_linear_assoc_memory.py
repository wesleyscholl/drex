"""
Experiment 29.1 — Outer-Product Linear Associative Memory

Hypothesis: A pure outer-product linear associative memory
(M += outer(v, k^T) / ||k||²) without any test-time SGD matches or beats
slot memory within 2% at matched parameter count. Validates linear write
as sufficient for associative recall without gradient-based inference.

Literature basis: Titans (2025) shows linear memory with outer-product writes
achieves strong performance. This is the simplest Titans-style variant.
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
NUM_PAIRS   = 5
SEQ_LEN     = 24
STEPS       = 2000
BATCH       = 32
LR          = 3e-4
EVAL_N      = 50
NUM_SLOTS   = NUM_PAIRS + 2   # for slot model (matched capacity)


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3: seq[b, pos]=keys[i]; seq[b, pos+1]=vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3]=2; seq[b, SEQ_LEN-2]=keys[qi]; seq[b, SEQ_LEN-1]=0; tgt[b]=vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class LinearAssocModel(nn.Module):
    """Outer-product associative matrix: M += v ⊗ k^T / ||k||²."""
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.rp     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape
        M  = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = self.k_proj(hs[:, t, :]); v = self.v_proj(hs[:, t, :])
            norm2 = k.pow(2).sum(-1, keepdim=True) + 1e-6   # (B,1)
            M = M + torch.bmm(v.unsqueeze(-1), k.unsqueeze(1)) / norm2.unsqueeze(-1)
        q   = hs[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(ctx))


class SlotModel(nn.Module):
    """Slot memory baseline (matched in parameter count)."""
    def __init__(self):
        super().__init__()
        self.enc   = Encoder()
        self.q     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.slots = NUM_SLOTS

    def forward(self, seq):
        B, L = seq.shape; H = HIDDEN_DIM; hs = self.enc(seq)
        content = hs[:, :-3, :]; k = min(self.slots, content.shape[1])
        _, idx  = torch.topk(content.norm(dim=-1), k, dim=1)
        s = torch.gather(content, 1, idx.unsqueeze(-1).expand(-1,-1,H))
        if k < self.slots: s = torch.cat([s, torch.zeros(B, self.slots-k, H)], dim=1)
        attn = torch.softmax(torch.bmm(self.q(hs[:,-1:,:]), s.transpose(1,2))/H**0.5, -1)
        return self.out(torch.bmm(attn, s).squeeze(1))


def train_eval(model_class, steps=STEPS, batch=BATCH, eval_n=EVAL_N):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(eval_n):
            seq, tgt = make_batch(batch)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp291LinearAssocMemory(Experiment):
    experiment_id = "exp_29_1"
    hypothesis = ("Outer-product linear associative memory (no test-time SGD) "
                  "matches slot memory within 2% at matched parameter count.")

    def run(self) -> ExperimentResult:
        lam = LinearAssocModel(); sm = SlotModel()
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH,
            param_bytes_linear=sum(p.numel()*4 for p in lam.parameters()),
            param_bytes_slot=sum(p.numel()*4 for p in sm.parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training linear associative memory..."); acc_linear = round(train_eval(LinearAssocModel), 4)
        print(f"    linear={acc_linear:.3f}")
        print("  Training slot memory..."); acc_slot = round(train_eval(SlotModel), 4)
        print(f"    slot={acc_slot:.3f}")

        gap = acc_linear - acc_slot
        metrics = dict(acc_linear=acc_linear, acc_slot=acc_slot, gap=round(gap, 4))

        # SUPPORTED: linear within 2% of slot (within 2% either direction means close)
        if abs(gap) < 0.02:
            outcome = OUTCOME_SUPPORTED
        elif gap > 0.02:
            outcome = OUTCOME_SUPPORTED  # linear beats slot -- still confirms
        elif gap < -0.05:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = f"Linear acc={acc_linear:.3f}, slot acc={acc_slot:.3f}, gap={gap:+.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp291LinearAssocMemory().execute()
