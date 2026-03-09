"""
Experiment 31.4 — Parametric Memory + Learned Eviction Policy

Hypothesis: When parametric MLP inner capacity is exceeded (>8 KV pairs), learned
importance-based eviction achieves >10% better accuracy than FIFO eviction.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

VOCAB_SIZE     = 64
HIDDEN_DIM     = 64
CAPACITY_LIMIT = 8
NUM_PAIRS      = 12
SEQ_LEN        = 32
STEPS          = 1200
BATCH          = 16
LR             = 3e-4
INFERENCE_LR   = 0.05
INNER_DIM      = 24
EVAL_N         = 50


def make_batch(batch_size):
    seq    = torch.full((batch_size, SEQ_LEN), 3, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 4, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (NUM_PAIRS,))]).unique()[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


class InnerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(HIDDEN_DIM, INNER_DIM), nn.ReLU(),
                                 nn.Linear(INNER_DIM, HIDDEN_DIM))
    def forward(self, x): return self.net(x)


def finetune_mlp_on_buffer(base_mlp, buf_keys, buf_vals):
    if not buf_keys: return base_mlp
    tmp = InnerMLP(); tmp.load_state_dict(base_mlp.state_dict())
    opt = Adam(tmp.parameters(), lr=INFERENCE_LR)
    for kh, vh in zip(buf_keys, buf_vals):
        with torch.enable_grad():
            F.mse_loss(tmp(kh.unsqueeze(0)).squeeze(0), vh).backward()
        opt.step(); opt.zero_grad()
    return tmp


class ParametricFIFO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(); self.mlp = InnerMLP()
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def _process_sample(self, token_ids):
        buf: deque = deque(maxlen=CAPACITY_LIMIT); key_h = None
        for t in range(SEQ_LEN - 1):
            h = self.encoder(token_ids[t].unsqueeze(0)).squeeze(0)
            if t % 2 == 0: key_h = h
            elif key_h is not None: buf.append((key_h.detach(), h.detach())); key_h = None
        tmp = finetune_mlp_on_buffer(self.mlp, [kh for kh,_ in buf], [vh for _,vh in buf])
        q_h = self.encoder(token_ids[SEQ_LEN - 2].unsqueeze(0)).squeeze(0)
        return self.out(tmp(q_h.unsqueeze(0)))
    def forward(self, seqs):
        return torch.cat([self._process_sample(seqs[b]) for b in range(seqs.size(0))], dim=0)


class ImportanceScorer(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Linear(HIDDEN_DIM * 2, 1)
    def forward(self, k, v): return self.net(torch.cat([k, v], dim=-1)).squeeze(-1)


class ParametricLearned(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(); self.mlp = InnerMLP()
        self.scorer  = ImportanceScorer()
        self.out     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def _process_sample(self, token_ids):
        buf_k: list = []; buf_v: list = []; buf_s: list = []; key_h = None
        for t in range(SEQ_LEN - 1):
            h = self.encoder(token_ids[t].unsqueeze(0)).squeeze(0)
            if t % 2 == 0: key_h = h
            elif key_h is not None:
                vh = h; score = self.scorer(key_h.detach().unsqueeze(0), vh.detach().unsqueeze(0)).item()
                if len(buf_k) >= CAPACITY_LIMIT:
                    mi = min(range(len(buf_s)), key=lambda i: buf_s[i])
                    buf_k.pop(mi); buf_v.pop(mi); buf_s.pop(mi)
                buf_k.append(key_h.detach()); buf_v.append(vh.detach()); buf_s.append(score)
                key_h = None
        tmp = finetune_mlp_on_buffer(self.mlp, buf_k, buf_v)
        q_h = self.encoder(token_ids[SEQ_LEN - 2].unsqueeze(0)).squeeze(0)
        return self.out(tmp(q_h.unsqueeze(0)))
    def forward(self, seqs):
        return torch.cat([self._process_sample(seqs[b]) for b in range(seqs.size(0))], dim=0)


def train_model(model_class, steps):
    model = model_class()
    enc_params = list(model.encoder.parameters()) + list(model.out.parameters())
    if hasattr(model, 'scorer'): enc_params += list(model.scorer.parameters())
    opt = Adam(enc_params, lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        loss = F.cross_entropy(model(seq), tgt)
        loss.backward(); opt.step(); opt.zero_grad()
    return model


def eval_model(model, n):
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(n):
            seq, tgt = make_batch(BATCH)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += BATCH
    model.train(); return c / t


class Exp314ParametricEviction(Experiment):
    experiment_id = "exp_31_4"
    hypothesis = ("Learned importance-based eviction achieves >10% better accuracy "
                  "than FIFO eviction when NUM_PAIRS (12) exceeds CAPACITY_LIMIT (8).")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, capacity_limit=CAPACITY_LIMIT,
            num_pairs=NUM_PAIRS, seq_len=SEQ_LEN, steps=STEPS, batch=BATCH,
            inference_lr=INFERENCE_LR, inner_dim=INNER_DIM,
            param_bytes=sum(p.numel()*4 for p in ParametricLearned().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training ParametricFIFO...")
        fifo_m = train_model(ParametricFIFO, STEPS)
        print("  Training ParametricLearned...")
        learned_m = train_model(ParametricLearned, STEPS)
        acc_fifo    = round(eval_model(fifo_m,    EVAL_N), 4)
        acc_learned = round(eval_model(learned_m, EVAL_N), 4)
        gap = round(acc_learned - acc_fifo, 4)
        print(f"  fifo={acc_fifo:.3f} learned={acc_learned:.3f} gap={gap:+.3f}")
        metrics = dict(acc_fifo=acc_fifo, acc_learned=acc_learned, gap=gap,
                       capacity_limit=CAPACITY_LIMIT, num_pairs=NUM_PAIRS)
        if acc_learned > acc_fifo + 0.10:  outcome = OUTCOME_SUPPORTED
        elif acc_learned > acc_fifo:        outcome = OUTCOME_INCONCLUSIVE
        else:                               outcome = OUTCOME_REFUTED
        notes = (f"FIFO={acc_fifo:.3f}, Learned={acc_learned:.3f}, Gap={gap:+.3f}. "
                 f"Capacity={CAPACITY_LIMIT}, NumPairs={NUM_PAIRS}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp314ParametricEviction().execute()
