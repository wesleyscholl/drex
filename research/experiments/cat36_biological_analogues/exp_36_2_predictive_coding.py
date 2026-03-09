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

experiment_id = "exp_36_2"
hypothesis = (
    "Storing prediction residuals (what the model predicted wrong) rather than "
    "full token representations produces equivalent or better associative recall "
    "accuracy with the same memory capacity — the predictive coding hypothesis."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 600
BATCH = 32


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class FullRepDeltaModel(nn.Module):
    """Condition A: store full representations using standard delta rule."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom   # residual (delta rule)
            M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))


class ResidualCodingModel(nn.Module):
    """Condition B: store prediction errors against a learned running mean (predictive coding).

    Each step, update a running mean prediction and store only the residual
    (surprise = actual - predicted). The residual is smaller and more information-dense
    if the predictive coding hypothesis is correct.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        # Predictor: takes current token and predicts next representation
        self.predictor = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 2):
            k = h[:, t, :]
            predicted_next = self.predictor(k)            # predict h[t+1]
            actual_next = h[:, t + 1, :]
            residual = actual_next - predicted_next       # prediction error
            # Write residual (surprise) instead of full representation
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = residual - v_pred / denom
            M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))


def train_and_eval(model_class):
    model = model_class()
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model(seq).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp362PredictiveCoding(Experiment):
    experiment_id = "exp_36_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        print("Training Condition A: full representation delta rule ...")
        acc_full = train_and_eval(FullRepDeltaModel)
        print(f"  acc_full={acc_full:.4f}")

        print("Training Condition B: predictive coding (residual storage) ...")
        acc_residual = train_and_eval(ResidualCodingModel)
        print(f"  acc_residual={acc_residual:.4f}")

        gap = round(acc_residual - acc_full, 4)

        metrics = dict(
            acc_full_representation=round(acc_full, 4),
            acc_residual_coding=round(acc_residual, 4),
            residual_advantage=gap,
        )

        if gap > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Predictive coding wins: full={acc_full:.3f}, "
                     f"residual={acc_residual:.3f}, advantage={gap:.3f}>0.03.")
        elif gap < -0.05:
            outcome = OUTCOME_REFUTED
            notes = (f"Full representation beats residual: full={acc_full:.3f}, "
                     f"residual={acc_residual:.3f}, gap={gap:.3f}<-0.05.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Roughly equivalent: full={acc_full:.3f}, residual={acc_residual:.3f}, "
                     f"gap={gap:.3f}. Neither clearly superior.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, num_pairs=NUM_PAIRS))


if __name__ == "__main__":
    Exp362PredictiveCoding().execute()
