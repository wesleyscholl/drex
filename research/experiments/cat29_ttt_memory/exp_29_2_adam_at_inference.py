"""
Experiment 29.2 — Adam-at-Inference for Parametric Memory

Hypothesis: Adam-at-inference (storing m1, m2 momentum states per weight tensor)
improves accuracy by >5% over SGD at the same number of inner steps (INFERENCE_STEPS=3)
for the parametric memory MLP. Validates the utility of adaptive learning rates
for test-time gradient-based memory updates.

Literature basis: Titans (2025) uses momentum-enhanced gradient descent for the
test-time memory update. Adam's per-parameter adaptive rates should help when
gradient directions vary across KV pairs during a single forward pass.
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
NUM_PAIRS      = 5
SEQ_LEN        = 24
STEPS          = 1500
BATCH          = 8
LR             = 3e-4
INFERENCE_LR   = 0.05
INFERENCE_STEPS= 3
INNER_DIM      = 32
EVAL_N         = 60


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


def apply_adam_step(params, grads, m1_states, m2_states, step_n, lr=INFERENCE_LR, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step in-place given explicit m1/m2 state lists."""
    new_params = []
    new_m1 = []; new_m2 = []
    for p, g, m1, m2 in zip(params, grads, m1_states, m2_states):
        m1_new = b1 * m1 + (1 - b1) * g
        m2_new = b2 * m2 + (1 - b2) * g.pow(2)
        m1_hat = m1_new / (1 - b1 ** step_n)
        m2_hat = m2_new / (1 - b2 ** step_n)
        new_p  = p - lr * m1_hat / (m2_hat.sqrt() + eps)
        new_params.append(new_p); new_m1.append(m1_new); new_m2.append(m2_new)
    return new_params, new_m1, new_m2


class ParamMemSGD(nn.Module):
    """Parametric memory with SGD at inference (baseline)."""
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
            for _ in range(INFERENCE_STEPS):
                for t in range(0, L - 3, 2):
                    kh = hs[b, t, :].detach(); vh = hs[b, t + 1, :].detach()
                    with torch.enable_grad():
                        h_mid = F.relu(kh @ w1.T + b1)
                        pred  = h_mid @ w2.T + b2
                        loss  = F.mse_loss(pred, vh)
                        grads = torch.autograd.grad(loss, [w1, b1, w2, b2], create_graph=False)
                    with torch.no_grad():
                        w1 = w1 - INFERENCE_LR * grads[0]
                        b1 = b1 - INFERENCE_LR * grads[1]
                        w2 = w2 - INFERENCE_LR * grads[2]
                        b2 = b2 - INFERENCE_LR * grads[3]
                    w1 = w1.detach().requires_grad_(True)
                    b1 = b1.detach().requires_grad_(True)
                    w2 = w2.detach().requires_grad_(True)
                    b2 = b2.detach().requires_grad_(True)
            with torch.no_grad():
                q_h = hs[b, -1, :].detach()
                h_mid = F.relu(q_h @ w1.T + b1)
                ctx   = h_mid @ w2.T + b2
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


class ParamMemAdam(nn.Module):
    """Parametric memory with Adam at inference (hypothesis)."""
    def __init__(self):
        super().__init__()
        self.enc  = Encoder()
        self.base = InnerMLP()
        self.out  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
        for b in range(B):
            w1 = self.base.fc1.weight.clone().detach().requires_grad_(True)
            b1s= self.base.fc1.bias.clone().detach().requires_grad_(True)
            w2 = self.base.fc2.weight.clone().detach().requires_grad_(True)
            b2s= self.base.fc2.bias.clone().detach().requires_grad_(True)
            # Initialize Adam moment states (zeros)
            m1 = [torch.zeros_like(p) for p in [w1, b1s, w2, b2s]]
            m2 = [torch.zeros_like(p) for p in [w1, b1s, w2, b2s]]
            step_n = 0
            for _ in range(INFERENCE_STEPS):
                for t in range(0, L - 3, 2):
                    kh = hs[b, t, :].detach(); vh = hs[b, t + 1, :].detach()
                    with torch.enable_grad():
                        h_mid = F.relu(kh @ w1.T + b1s)
                        pred  = h_mid @ w2.T + b2s
                        loss  = F.mse_loss(pred, vh)
                        grads = torch.autograd.grad(loss, [w1, b1s, w2, b2s], create_graph=False)
                    step_n += 1
                    params_new, m1, m2 = apply_adam_step(
                        [w1, b1s, w2, b2s], list(grads), m1, m2, step_n)
                    w1, b1s, w2, b2s = params_new
                    w1  = w1.detach().requires_grad_(True)
                    b1s = b1s.detach().requires_grad_(True)
                    w2  = w2.detach().requires_grad_(True)
                    b2s = b2s.detach().requires_grad_(True)
            with torch.no_grad():
                q_h  = hs[b, -1, :].detach()
                h_mid= F.relu(q_h @ w1.T + b1s)
                ctx  = h_mid @ w2.T + b2s
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


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


class Exp292AdamAtInference(Experiment):
    experiment_id = "exp_29_2"
    hypothesis = ("Adam-at-inference for parametric memory MLP improves accuracy "
                  "by >5% over SGD at the same number of inner steps (INFERENCE_STEPS=3).")

    def run(self) -> ExperimentResult:
        sgd_m = ParamMemSGD(); adam_m = ParamMemAdam()
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, inference_steps=INFERENCE_STEPS,
            inner_dim=INNER_DIM,
            param_bytes_sgd=sum(p.numel()*4 for p in sgd_m.parameters()),
            param_bytes_adam=sum(p.numel()*4 for p in adam_m.parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training SGD-at-inference model...")
        acc_sgd = round(train_eval(ParamMemSGD), 4)
        print(f"    sgd={acc_sgd:.3f}")
        print("  Training Adam-at-inference model...")
        acc_adam = round(train_eval(ParamMemAdam), 4)
        print(f"    adam={acc_adam:.3f}")

        gap = acc_adam - acc_sgd
        metrics = dict(acc_sgd=acc_sgd, acc_adam=acc_adam, gap=round(gap, 4))

        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
        elif gap < -0.02:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = f"Adam acc={acc_adam:.3f}, SGD acc={acc_sgd:.3f}, gap={gap:+.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp292AdamAtInference().execute()
