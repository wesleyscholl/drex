"""
Experiment 34.4 — Curriculum Learning for Associative Memory

Hypothesis: Training with an easy-first curriculum (start with 2 pairs, gradually
increase to 8 pairs over training) improves final accuracy vs random mixed
training, with a gap > 0.08 at 1500 steps.

Literature basis: Curriculum learning (Bengio et al., 2009) improves generalisation
by presenting examples in order of difficulty; we test whether memory systems
benefit from a stable low-interference early phase.
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
SEQ_LEN    = 24
STEPS      = 1500
BATCH      = 8
LR         = 3e-4
EVAL_N     = 60
EVAL_PAIRS = 4
MIN_PAIRS  = 2
MAX_PAIRS  = 8


def make_batch(batch_size, num_pairs):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (num_pairs * 4,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:num_pairs]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (num_pairs,)); pos = 0
        for i in range(num_pairs):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class DeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape
        M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = self.k_proj(hs[:, t, :]); v = self.v_proj(hs[:, t, :])
            k_n = F.normalize(k, dim=-1)
            vp  = torch.bmm(M, k_n.unsqueeze(-1)).squeeze(-1)
            M   = M + torch.bmm((v - vp).unsqueeze(-1), k_n.unsqueeze(1))
        q = self.q_proj(hs[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def eval_model(model):
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH, EVAL_PAIRS)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    model.train(); return c / t


def train_random():
    model = DeltaModel(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        n = torch.randint(MIN_PAIRS, MAX_PAIRS + 1, (1,)).item()
        seq, tgt = make_batch(BATCH, n)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    return eval_model(model)


def train_easy_first():
    model = DeltaModel(); opt = Adam(model.parameters(), lr=LR); model.train()
    for s in range(STEPS):
        frac = s / max(STEPS - 1, 1)
        n = max(MIN_PAIRS, round(MIN_PAIRS + frac * (MAX_PAIRS - MIN_PAIRS)))
        seq, tgt = make_batch(BATCH, n)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    return eval_model(model)


def train_hard_first():
    model = DeltaModel(); opt = Adam(model.parameters(), lr=LR); model.train()
    for s in range(STEPS):
        frac = s / max(STEPS - 1, 1)
        n = max(MIN_PAIRS, round(MAX_PAIRS - frac * (MAX_PAIRS - MIN_PAIRS)))
        seq, tgt = make_batch(BATCH, n)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    return eval_model(model)


class Exp344CurriculumLearning(Experiment):
    experiment_id = "exp_34_4"
    hypothesis = ("Easy-first curriculum (2→8 pairs) improves final accuracy "
                  "vs random mixed training by >0.08 on a 4-pair evaluation task.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
            min_pairs=MIN_PAIRS, max_pairs=MAX_PAIRS, eval_pairs=EVAL_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH,
            param_bytes=sum(p.numel() * 4 for p in DeltaModel().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training with random curriculum...")
        acc_random = round(train_random(), 4); print(f"    random={acc_random:.3f}")
        print("  Training with easy-first curriculum...")
        acc_easy   = round(train_easy_first(), 4); print(f"    easy_first={acc_easy:.3f}")
        print("  Training with hard-first curriculum...")
        acc_hard   = round(train_hard_first(), 4); print(f"    hard_first={acc_hard:.3f}")

        gap = acc_easy - acc_random
        metrics = dict(
            acc_random=acc_random, acc_easy_first=acc_easy, acc_hard_first=acc_hard,
            gap_easy_vs_random=round(gap, 4),
            gap_easy_vs_hard=round(acc_easy - acc_hard, 4),
        )
        if gap > 0.08:
            outcome = OUTCOME_SUPPORTED
        elif gap < -0.03:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Random={acc_random:.3f}, Easy-first={acc_easy:.3f}, "
                 f"Hard-first={acc_hard:.3f}. Gap(easy-random)={gap:+.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp344CurriculumLearning().execute()
