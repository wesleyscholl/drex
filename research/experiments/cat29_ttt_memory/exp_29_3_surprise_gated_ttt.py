"""
Experiment 29.3 — Surprise-Weighted Write Gate for TTT Memory

Hypothesis: A surprise-weighted gate that only updates the memory MLP when the
gradient-norm ratio (||∇||² / EMA_||∇||²) > 1.5 achieves >90% of full-update
accuracy at <50% actual update rate.

Literature basis: exp_1_5 found surprise anti-correlates with memory importance
(r = −0.503). Gradient ratio measures surprise *relative to expectation* (novelty
vs. difficulty), which should be a better gate than raw loss magnitude.
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

VOCAB_SIZE       = 64
HIDDEN_DIM       = 64
NUM_PAIRS        = 5
SEQ_LEN          = 24
STEPS            = 1500
BATCH            = 8
LR               = 3e-4
INFERENCE_LR     = 0.05
INNER_DIM        = 32
EMA_DECAY        = 0.9
SURPRISE_THRESH  = 1.5
EVAL_N           = 60


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class InnerMLP(nn.Module):
    def __init__(self, h=HIDDEN_DIM, inner=INNER_DIM):
        super().__init__()
        self.fc1 = nn.Linear(h, inner); self.fc2 = nn.Linear(inner, h)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


def inner_forward(w1, b1, w2, b2, x):
    return F.relu(x @ w1.T + b1) @ w2.T + b2


class FullTTT(nn.Module):
    """Always updates on every KV pair (full update rate = 1.0)."""
    def __init__(self):
        super().__init__()
        self.enc  = Encoder()
        self.base = InnerMLP()
        self.out  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
        for b in range(B):
            w1 = self.base.fc1.weight.clone().detach().requires_grad_(True)
            b1 = self.base.fc1.bias.clone().detach().requires_grad_(True)
            w2 = self.base.fc2.weight.clone().detach().requires_grad_(True)
            b2 = self.base.fc2.bias.clone().detach().requires_grad_(True)
            for t in range(0, L - 3, 2):
                kh = hs[b, t, :].detach(); vh = hs[b, t + 1, :].detach()
                with torch.enable_grad():
                    pred  = inner_forward(w1, b1, w2, b2, kh.unsqueeze(0)).squeeze(0)
                    loss  = F.mse_loss(pred, vh)
                    grads = torch.autograd.grad(loss, [w1, b1, w2, b2], create_graph=False)
                with torch.no_grad():
                    w1 = w1 - INFERENCE_LR * grads[0]; b1 = b1 - INFERENCE_LR * grads[1]
                    w2 = w2 - INFERENCE_LR * grads[2]; b2 = b2 - INFERENCE_LR * grads[3]
                w1 = w1.detach().requires_grad_(True); b1 = b1.detach().requires_grad_(True)
                w2 = w2.detach().requires_grad_(True); b2 = b2.detach().requires_grad_(True)
            with torch.no_grad():
                ctx = inner_forward(w1, b1, w2, b2, hs[b, -1, :].detach().unsqueeze(0)).squeeze(0)
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


class SurpriseTTT(nn.Module):
    """Updates only when gradient-norm ratio > SURPRISE_THRESH."""
    def __init__(self):
        super().__init__()
        self.enc  = Encoder()
        self.base = InnerMLP()
        self.out  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []; total_steps = 0; write_steps = 0
        for b in range(B):
            w1 = self.base.fc1.weight.clone().detach().requires_grad_(True)
            b1 = self.base.fc1.bias.clone().detach().requires_grad_(True)
            w2 = self.base.fc2.weight.clone().detach().requires_grad_(True)
            b2 = self.base.fc2.bias.clone().detach().requires_grad_(True)
            ema_grad_sq = 1.0  # initialize EMA of ||∇||²
            for t in range(0, L - 3, 2):
                total_steps += 1
                kh = hs[b, t, :].detach(); vh = hs[b, t + 1, :].detach()
                with torch.enable_grad():
                    pred  = inner_forward(w1, b1, w2, b2, kh.unsqueeze(0)).squeeze(0)
                    loss  = F.mse_loss(pred, vh)
                    grads = torch.autograd.grad(loss, [w1, b1, w2, b2], create_graph=False)
                grad_sq = sum(g.pow(2).sum().item() for g in grads)
                # Update EMA
                ema_grad_sq = EMA_DECAY * ema_grad_sq + (1 - EMA_DECAY) * grad_sq
                ratio = grad_sq / max(ema_grad_sq, 1e-8)
                if ratio > SURPRISE_THRESH:
                    write_steps += 1
                    with torch.no_grad():
                        w1 = w1 - INFERENCE_LR * grads[0]; b1 = b1 - INFERENCE_LR * grads[1]
                        w2 = w2 - INFERENCE_LR * grads[2]; b2 = b2 - INFERENCE_LR * grads[3]
                    w1 = w1.detach().requires_grad_(True); b1 = b1.detach().requires_grad_(True)
                    w2 = w2.detach().requires_grad_(True); b2 = b2.detach().requires_grad_(True)
            with torch.no_grad():
                ctx = inner_forward(w1, b1, w2, b2, hs[b, -1, :].detach().unsqueeze(0)).squeeze(0)
            ctxs.append(ctx)
        self._last_write_rate = write_steps / max(total_steps, 1)
        return self.out(torch.stack(ctxs))


class NoTTT(nn.Module):
    """Never updates — frozen MLP (baseline lower bound)."""
    def __init__(self):
        super().__init__()
        self.enc  = Encoder()
        self.base = InnerMLP()
        self.out  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
        for b in range(B):
            with torch.no_grad():
                ctx = self.base(hs[b, -1, :].detach())
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


def train_eval_model(model_class, steps=STEPS, batch=BATCH, eval_n=EVAL_N):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0; last_write_rate = 1.0
    with torch.no_grad():
        for _ in range(eval_n):
            seq, tgt = make_batch(batch)
            out = model(seq)
            c += (out.argmax(-1) == tgt).sum().item(); t += tgt.size(0)
            if hasattr(model, '_last_write_rate'):
                last_write_rate = model._last_write_rate
    acc = c / t
    wr  = last_write_rate if hasattr(model, '_last_write_rate') else 1.0
    return acc, wr


class Exp293SurpriseGatedTTT(Experiment):
    experiment_id = "exp_29_3"
    hypothesis = ("Surprise-gated TTT (update when gradient-norm ratio > 1.5) achieves "
                  ">90% of full-update accuracy at <50% update rate.")

    def run(self) -> ExperimentResult:
        noTTT = NoTTT(); full = FullTTT(); surp = SurpriseTTT()
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, inner_dim=INNER_DIM,
            ema_decay=EMA_DECAY, surprise_thresh=SURPRISE_THRESH,
            param_bytes=sum(p.numel()*4 for p in full.parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training NoTTT model...")
        acc_no, _ = train_eval_model(NoTTT); print(f"    no_ttt={acc_no:.3f}")
        print("  Training FullTTT model...")
        acc_full, _ = train_eval_model(FullTTT); print(f"    full_ttt={acc_full:.3f}")
        print("  Training SurpriseTTT model...")
        acc_surp, write_rate = train_eval_model(SurpriseTTT)
        print(f"    surprise_ttt={acc_surp:.3f} write_rate={write_rate:.3f}")

        acc_ratio = acc_surp / max(acc_full, 1e-6)
        metrics = dict(
            acc_full=round(acc_full, 4), acc_surprise=round(acc_surp, 4),
            acc_no_ttt=round(acc_no, 4), write_rate=round(write_rate, 4),
            acc_ratio=round(acc_ratio, 4),
        )

        if acc_ratio > 0.90 and write_rate < 0.50:
            outcome = OUTCOME_SUPPORTED
        elif acc_ratio < 0.70:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Full acc={acc_full:.3f}, Surprise acc={acc_surp:.3f}, "
                 f"No-TTT acc={acc_no:.3f}. acc_ratio={acc_ratio:.3f}, "
                 f"write_rate={write_rate:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp293SurpriseGatedTTT().execute()
