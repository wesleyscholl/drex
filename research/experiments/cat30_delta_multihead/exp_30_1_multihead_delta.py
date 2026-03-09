"""
Experiment 30.1 — Multi-Head Delta Rule

Hypothesis: Multi-head delta rule (4 heads × H/4 dims each, total params ≈ single-head)
outperforms single-head by >5% on 8-pair associative recall.
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
NUM_PAIRS  = 8
SEQ_LEN    = 32
STEPS      = 1500
BATCH      = 32
LR         = 3e-4
EVAL_N     = 50
NUM_HEADS  = 4


def make_batch(batch_size: int):
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


class Encoder(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, h: int = HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x):
        e = self.embed(x); return self.norm(e + self.ff(e))


class SingleHeadDelta(nn.Module):
    def __init__(self, h: int = HIDDEN_DIM, v: int = VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(v, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, v)
    def forward(self, seq):
        h_all = self.enc(seq); B, L, H = h_all.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        return self.out(self.rp(torch.bmm(M, h_all[:, -1, :].unsqueeze(-1)).squeeze(-1)))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


class MultiHeadDelta(nn.Module):
    def __init__(self, h: int = HIDDEN_DIM, v: int = VOCAB_SIZE, num_heads: int = NUM_HEADS):
        super().__init__()
        assert h % num_heads == 0
        self.num_heads = num_heads; self.h_head = h // num_heads
        self.enc      = Encoder(v, h)
        self.k_projs  = nn.ModuleList([nn.Linear(h, self.h_head) for _ in range(num_heads)])
        self.v_projs  = nn.ModuleList([nn.Linear(h, self.h_head) for _ in range(num_heads)])
        self.q_projs  = nn.ModuleList([nn.Linear(h, self.h_head) for _ in range(num_heads)])
        self.out_proj = nn.Linear(h, h)
        self.out      = nn.Linear(h, v)
    def forward(self, seq):
        h_all = self.enc(seq); B, L, H = h_all.shape; head_ctxs = []
        for hi in range(self.num_heads):
            Hh = self.h_head; M = torch.zeros(B, Hh, Hh)
            for t in range(L - 1):
                k = self.k_projs[hi](h_all[:, t, :]); v = self.v_projs[hi](h_all[:, t, :])
                vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
                M  = M + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                                   k.unsqueeze(1))
            q_h = self.q_projs[hi](h_all[:, -1, :])
            head_ctxs.append(torch.bmm(M, q_h.unsqueeze(-1)).squeeze(-1))
        ctx = torch.cat(head_ctxs, dim=-1)
        return self.out(self.out_proj(ctx))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


def train_eval(model_class) -> float:
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp301MultiheadDelta(Experiment):
    experiment_id = "exp_30_1"
    hypothesis = ("Multi-head delta rule (4 heads x H/4 dims) outperforms single-head "
                  "by >5% on 8-pair associative recall.")

    def run(self) -> ExperimentResult:
        shd = SingleHeadDelta(); mhd = MultiHeadDelta()
        config = dict(
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, num_heads=NUM_HEADS,
            h_head=HIDDEN_DIM // NUM_HEADS,
            param_bytes_single=shd.param_bytes(), param_bytes_multi=mhd.param_bytes(),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training SingleHeadDelta...")
        acc_single = round(train_eval(SingleHeadDelta), 4)
        print(f"    single={acc_single:.3f}")
        print("  Training MultiHeadDelta...")
        acc_multi = round(train_eval(MultiHeadDelta), 4)
        print(f"    multi={acc_multi:.3f}")

        gap = acc_multi - acc_single
        metrics = dict(acc_single=acc_single, acc_multi=acc_multi, gap=round(gap, 4))

        if gap > 0.05:   outcome = OUTCOME_SUPPORTED
        elif gap < -0.05: outcome = OUTCOME_REFUTED
        else:             outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Multi-head acc={acc_multi:.3f}, Single-head acc={acc_single:.3f}, gap={gap:.3f}. "
                 f"Params: single={shd.param_bytes()//1024}KB, multi={mhd.param_bytes()//1024}KB.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp301MultiheadDelta().execute()
