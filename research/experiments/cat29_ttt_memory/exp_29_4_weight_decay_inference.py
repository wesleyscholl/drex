"""
Experiment 29.4 — Weight Decay at Inference Prevents Memory Saturation

Hypothesis: Weight decay (L2 regularization) applied during test-time MLP updates
prevents memory saturation on long sequences: accuracy at SEQ_LEN=192 is >20%
higher with weight_decay=0.01 than without.

Literature basis: Titans (2025) uses weight decay as an explicit forgetting mechanism
during test-time gradient descent, preventing old associations from saturating memory.
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

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
NUM_PAIRS     = 8
SEQ_LENS      = [24, 96, 192]
STEPS         = 800
BATCH         = 8
LR            = 3e-4
INFERENCE_LR  = 0.05
WEIGHT_DECAY  = 0.01
INNER_DIM     = 32
EVAL_N        = 40


def make_batch(batch_size, seq_len, num_pairs=NUM_PAIRS):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (num_pairs * 4,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:num_pairs]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (num_pairs,)); pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3): seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len-3] = 2; seq[b, seq_len-2] = keys[qi]; seq[b, seq_len-1] = 0
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


def inner_step(w1, b1, w2, b2, kh, vh, lr, wd=0.0):
    """One gradient step on MLP weights; returns updated (w1,b1,w2,b2)."""
    with torch.enable_grad():
        h_mid = F.relu(kh @ w1.T + b1)
        pred  = h_mid @ w2.T + b2
        loss  = F.mse_loss(pred, vh)
        grads = torch.autograd.grad(loss, [w1, b1, w2, b2], create_graph=False)
    with torch.no_grad():
        w1_new = (1 - wd) * w1 - lr * grads[0]
        b1_new = b1 - lr * grads[1]
        w2_new = (1 - wd) * w2 - lr * grads[2]
        b2_new = b2 - lr * grads[3]
    return (w1_new.detach().requires_grad_(True), b1_new.detach().requires_grad_(True),
            w2_new.detach().requires_grad_(True), b2_new.detach().requires_grad_(True))


class PlainTTT(nn.Module):
    """Test-time MLP updates without weight decay."""
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
                w1, b1, w2, b2 = inner_step(w1, b1, w2, b2, kh, vh, INFERENCE_LR, wd=0.0)
            with torch.no_grad():
                ctx = F.relu(hs[b, -1, :].detach() @ w1.T + b1) @ w2.T + b2
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


class WeightDecayTTT(nn.Module):
    """Test-time MLP updates with weight decay (forgetting)."""
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
                w1, b1, w2, b2 = inner_step(w1, b1, w2, b2, kh, vh, INFERENCE_LR, wd=WEIGHT_DECAY)
            with torch.no_grad():
                ctx = F.relu(hs[b, -1, :].detach() @ w1.T + b1) @ w2.T + b2
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


def train_eval(model_class, seq_len, steps=STEPS, batch=BATCH, eval_n=EVAL_N):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(eval_n):
            seq, tgt = make_batch(batch, seq_len)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp294WeightDecayInference(Experiment):
    experiment_id = "exp_29_4"
    hypothesis = ("Weight decay at inference prevents saturation: at SEQ_LEN=192, "
                  "accuracy with weight_decay=0.01 is >20% higher than without.")

    def run(self) -> ExperimentResult:
        plain_m = PlainTTT(); wd_m = WeightDecayTTT()
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_lens=SEQ_LENS, steps=STEPS, batch=BATCH, inner_dim=INNER_DIM,
            weight_decay=WEIGHT_DECAY, inference_lr=INFERENCE_LR,
            param_bytes_plain=sum(p.numel()*4 for p in plain_m.parameters()),
            param_bytes_wd=sum(p.numel()*4 for p in wd_m.parameters()),
            activation_bytes=BATCH * max(SEQ_LENS) * HIDDEN_DIM * 4,
        )

        metrics = {}
        for sl in SEQ_LENS:
            print(f"  SEQ_LEN={sl}: plain...")
            acc_plain = round(train_eval(PlainTTT, sl), 4)
            print(f"    plain={acc_plain:.3f}  weight_decay...")
            acc_wd    = round(train_eval(WeightDecayTTT, sl), 4)
            print(f"    wd={acc_wd:.3f}")
            metrics[f"acc_plain_len{sl}"] = acc_plain
            metrics[f"acc_wd_len{sl}"]    = acc_wd
            metrics[f"gap_len{sl}"]       = round(acc_wd - acc_plain, 4)

        gain_192 = metrics["acc_wd_len192"] - metrics["acc_plain_len192"]
        metrics["acc_gain_wd_192"] = round(gain_192, 4)

        if gain_192 > 0.20:
            outcome = OUTCOME_SUPPORTED
        elif gain_192 < 0.0:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Gain at SEQ_LEN=192: {gain_192:+.3f}. "
                 f"Gaps: len24={metrics['gap_len24']:+.3f}, "
                 f"len96={metrics['gap_len96']:+.3f}, "
                 f"len192={metrics['gap_len192']:+.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp294WeightDecayInference().execute()
