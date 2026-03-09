"""
Experiment 31.2 — Retroactive Re-encoding Gap at 8× Sequence Length

Hypothesis: Retroactive re-encoding gap (+0.133 at SEQ_LEN=24) persists above +0.08
at 8× length (SEQ_LEN=192, 10 KV pairs).
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
NUM_PAIRS  = 10
SEQ_LENS   = [24, 48, 96, 192]
STEPS      = 600
BATCH      = 16
LR         = 3e-4
EVAL_N     = 40


def make_batch(batch_size, seq_len):
    seq    = torch.full((batch_size, seq_len), 3, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 4, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (NUM_PAIRS,))]).unique()[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


class ForwardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        h = self.enc(x); B, L, H = h.shape; M = torch.zeros(B, H, H)
        for t in range(L):
            k = F.normalize(self.k_proj(h[:, t, :]), dim=-1); v = self.v_proj(h[:, t, :])
            M = M + torch.bmm((v - torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                              k.unsqueeze(1))
        return self.out(torch.bmm(M, self.q_proj(h[:, -1, :]).unsqueeze(-1)).squeeze(-1))


class RetroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc        = Encoder()
        self.k_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.retro_attn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out        = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        h = self.enc(x); B, L, H = h.shape; M = torch.zeros(B, H, H)
        for t in range(L):
            k = F.normalize(self.k_proj(h[:, t, :]), dim=-1); v = self.v_proj(h[:, t, :])
            M = M + torch.bmm((v - torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                              k.unsqueeze(1))
        # Top-3 column re-encoding of first half
        half = L // 2; k_s = 3; slots = M.transpose(1, 2)
        q = self.retro_attn(h[:, :half, :])
        scores = torch.bmm(q, slots.transpose(1, 2)) / H**0.5
        topk_s, topk_i = scores.topk(min(k_s, H), dim=-1)
        topk_a = torch.softmax(topk_s, dim=-1)
        topk_sl = torch.stack([slots[b, topk_i[b]] for b in range(B)])
        ctx = (topk_a.unsqueeze(-1) * topk_sl).sum(-2)
        M2 = M.clone()
        for t in range(half):
            k = F.normalize(self.k_proj(ctx[:, t, :]), dim=-1); v = self.v_proj(ctx[:, t, :])
            M2 = M2 + torch.bmm((v - torch.bmm(M2, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                                 k.unsqueeze(1))
        return self.out(torch.bmm(M2, self.q_proj(h[:, -1, :]).unsqueeze(-1)).squeeze(-1))


def train_eval(model_class, seq_len):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, seq_len)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH, seq_len)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += BATCH
    return c / t


class Exp312RetroLengthScaling(Experiment):
    experiment_id = "exp_31_2"
    hypothesis = ("Retroactive re-encoding gap persists above +0.08 at SEQ_LEN=192 (8× baseline).")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_lens=SEQ_LENS, steps=STEPS, batch=BATCH,
            param_bytes=sum(p.numel()*4 for p in RetroModel().parameters()),
            activation_bytes=BATCH * max(SEQ_LENS) * HIDDEN_DIM * 4,
        )
        metrics = {}; gaps = {}
        for sl in SEQ_LENS:
            print(f"  SEQ_LEN={sl}: forward...")
            acc_fwd   = round(train_eval(ForwardModel, sl), 4)
            print(f"    fwd={acc_fwd:.3f}  retro...")
            acc_retro = round(train_eval(RetroModel, sl), 4)
            gap = acc_retro - acc_fwd
            print(f"    retro={acc_retro:.3f}  gap={gap:+.3f}")
            metrics[f"acc_forward_len{sl}"] = acc_fwd
            metrics[f"acc_retro_len{sl}"]   = acc_retro
            metrics[f"gap_len{sl}"]          = round(gap, 4)
            gaps[sl] = gap
        retro_gap_192 = gaps[192]
        if retro_gap_192 > 0.08:   outcome = OUTCOME_SUPPORTED
        elif retro_gap_192 > 0.0:  outcome = OUTCOME_INCONCLUSIVE
        else:                       outcome = OUTCOME_REFUTED
        notes = (f"Gaps: len24={gaps[24]:.3f}, len48={gaps[48]:.3f}, "
                 f"len96={gaps[96]:.3f}, len192={gaps[192]:.3f}. "
                 f"192 gap > 0.08: {retro_gap_192 > 0.08}")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp312RetroLengthScaling().execute()
