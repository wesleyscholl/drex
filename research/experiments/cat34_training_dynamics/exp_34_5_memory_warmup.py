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

experiment_id = "exp_34_5"
hypothesis = (
    "Gradual memory warmup (write scale 0→1 over 200 of 600 steps) improves "
    "final accuracy by >3% over full-memory training from step 0, because "
    "the backbone learns representations before the memory becomes active."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 600
WARMUP_STEPS = 200
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


class DeltaModel(nn.Module):
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

    def forward(self, seq, mem_scale=1.0):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            dM = torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
            M = M + mem_scale * dM  # scale memory writes
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))


def train_and_eval(use_warmup):
    model = DeltaModel()
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for step in range(STEPS):
        if use_warmup:
            mem_scale = min(1.0, step / WARMUP_STEPS)
        else:
            mem_scale = 1.0
        seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq, mem_scale=mem_scale), tgt)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(50):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model(seq, mem_scale=1.0).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp345MemoryWarmup(Experiment):
    experiment_id = "exp_34_5"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        print("Training Condition A: full memory from step 0 ...")
        acc_full = train_and_eval(use_warmup=False)
        print(f"  acc_full={acc_full:.4f}")

        print("Training Condition B: warmup ramp over first 200 steps ...")
        acc_warmup = train_and_eval(use_warmup=True)
        print(f"  acc_warmup={acc_warmup:.4f}")

        gain = round(acc_warmup - acc_full, 4)

        metrics = dict(
            acc_full_immediate=round(acc_full, 4),
            acc_warmup=round(acc_warmup, 4),
            warmup_gain=gain,
            warmup_steps=WARMUP_STEPS,
            total_steps=STEPS,
        )

        if gain > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Warmup improves accuracy: full={acc_full:.3f} → warmup={acc_warmup:.3f}, "
                     f"gain={gain:.3f}>0.03. Backbone benefits from pre-memory initialization.")
        elif gain < -0.02:
            outcome = OUTCOME_REFUTED
            notes = (f"Warmup hurts: full={acc_full:.3f} → warmup={acc_warmup:.3f}, "
                     f"gain={gain:.3f}<-0.02. Delayed memory access is harmful.")
        elif abs(gain) <= 0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"No significant warmup benefit: gain={gain:.3f} (threshold >0.03). "
                     f"full={acc_full:.3f}, warmup={acc_warmup:.3f}.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = f"Marginal warmup effect: gain={gain:.3f}."

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, warmup_steps=WARMUP_STEPS))


if __name__ == "__main__":
    Exp345MemoryWarmup().execute()
