"""
Experiment 30.3 — Bidirectional Delta Rule

Hypothesis: Bidirectional delta rule (forward + retroactive backward pass) improves
accuracy by >8% on late-query tasks vs unidirectional, without harming early-query tasks
by more than 2%.
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

VOCAB_SIZE     = 64
HIDDEN_DIM     = 64
NUM_PAIRS      = 6
SEQ_LEN        = 30
STEPS          = 1500
BATCH          = 32
LR             = 3e-4
EVAL_N         = 50
EARLY_READ_POS = 5


def make_batch_late(batch_size: int):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


def make_batch_early(batch_size: int):
    seq = torch.full((batch_size, SEQ_LEN), 3, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, 3] = 2; seq[b, 4] = keys[qi]; seq[b, 5] = 0
        pos = 6
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN: seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, h: int = HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class UniDirDelta(nn.Module):
    def __init__(self, h: int = HIDDEN_DIM, v: int = VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(v, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, v)
    def forward(self, seq, read_pos: int = -1):
        h_all = self.enc(seq); B, L, H = h_all.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        q = h_all[:, read_pos, :]
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


class BiDirDelta(nn.Module):
    def __init__(self, h: int = HIDDEN_DIM, v: int = VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(v, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, v)
    def forward(self, seq, read_pos: int = -1):
        h_all = self.enc(seq); B, L, H = h_all.shape
        M_fwd = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp    = torch.bmm(M_fwd, k.unsqueeze(-1)).squeeze(-1)
            M_fwd = M_fwd + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                                      k.unsqueeze(1))
        M_bwd = torch.zeros(B, H, H)
        for t in range(L - 2, -1, -1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp    = torch.bmm(M_bwd, k.unsqueeze(-1)).squeeze(-1)
            M_bwd = M_bwd + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                                      k.unsqueeze(1))
        q = h_all[:, read_pos, :]
        ctx = (torch.bmm(M_fwd, q.unsqueeze(-1)).squeeze(-1) +
               torch.bmm(M_bwd, q.unsqueeze(-1)).squeeze(-1)) / 2
        return self.out(self.rp(ctx))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


def train_eval(model_class, make_batch_fn, read_pos: int = -1) -> float:
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch_fn(BATCH)
        F.cross_entropy(model(seq, read_pos), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch_fn(BATCH)
            c += (model(seq, read_pos).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp303BidirectionalDelta(Experiment):
    experiment_id = "exp_30_3"
    hypothesis = ("Bidirectional delta rule improves accuracy by >8% on late-query tasks "
                  "vs unidirectional, without harming early-query tasks by more than 2%.")

    def run(self) -> ExperimentResult:
        ud = UniDirDelta(); bd = BiDirDelta()
        config = dict(
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, early_read_pos=EARLY_READ_POS,
            param_bytes_unidir=ud.param_bytes(), param_bytes_bidir=bd.param_bytes(),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training UniDir (late)...")
        acc_unidir_late = round(train_eval(UniDirDelta, make_batch_late, -1), 4)
        print(f"    unidir_late={acc_unidir_late:.3f}")
        print("  Training BiDir (late)...")
        acc_bidir_late = round(train_eval(BiDirDelta, make_batch_late, -1), 4)
        print(f"    bidir_late={acc_bidir_late:.3f}")
        print("  Training UniDir (early)...")
        acc_unidir_early = round(train_eval(UniDirDelta, make_batch_early, EARLY_READ_POS), 4)
        print(f"    unidir_early={acc_unidir_early:.3f}")
        print("  Training BiDir (early)...")
        acc_bidir_early = round(train_eval(BiDirDelta, make_batch_early, EARLY_READ_POS), 4)
        print(f"    bidir_early={acc_bidir_early:.3f}")

        late_improvement = acc_bidir_late  - acc_unidir_late
        early_change     = acc_bidir_early - acc_unidir_early
        metrics = dict(
            acc_unidir_late=acc_unidir_late, acc_bidir_late=acc_bidir_late,
            acc_unidir_early=acc_unidir_early, acc_bidir_early=acc_bidir_early,
            late_improvement=round(late_improvement, 4), early_change=round(early_change, 4),
        )

        if late_improvement > 0.08 and early_change >= -0.02:
            outcome = OUTCOME_SUPPORTED
        elif late_improvement < 0.0 or early_change < -0.10:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Late: bidir={acc_bidir_late:.3f}, unidir={acc_unidir_late:.3f}, "
                 f"improvement={late_improvement:.3f}. "
                 f"Early: bidir={acc_bidir_early:.3f}, unidir={acc_unidir_early:.3f}, "
                 f"change={early_change:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp303BidirectionalDelta().execute()
