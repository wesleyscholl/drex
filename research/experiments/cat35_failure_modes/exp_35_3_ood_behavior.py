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

experiment_id = "exp_35_3"
hypothesis = (
    "Out-of-distribution inputs cause abnormal write gate behavior: "
    "the write rate for OOD tokens deviates from in-distribution write rate "
    "by more than 2×, revealing that the gate is not robust to OOD inputs."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
STEPS = 600
BATCH = 32
# Training token range: [0, VOCAB_SIZE//2); OOD range: [VOCAB_SIZE//2, VOCAB_SIZE)
TRAIN_RANGE = VOCAB_SIZE // 2


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs, ood_frac=0.0):
    """ood_frac fraction of context tokens replaced with OOD tokens."""
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 4, (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 4, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 4, vocab_size // 2, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]

    # Replace a fraction of context tokens with OOD token IDs
    if ood_frac > 0.0:
        context_len = seq_len - 3
        n_replace = int(context_len * ood_frac)
        for b in range(batch_size):
            positions = torch.randperm(context_len)[:n_replace]
            seq[b, positions] = torch.randint(TRAIN_RANGE, vocab_size, (n_replace,))
    return seq, target


class LearnedGateDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.gate_net = nn.Sequential(nn.Linear(HIDDEN_DIM, 16), nn.ReLU(),
                                       nn.Linear(16, 1), nn.Sigmoid())
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        write_rates = []
        for t in range(L - 1):
            k = h[:, t, :]
            g = self.gate_net(k)  # (B, 1)
            write_rates.append((g > 0.5).float().mean().item())
            v_pred = torch.bmm(M.detach(), k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            M = M + g.unsqueeze(-1) * torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        avg_wr = sum(write_rates) / len(write_rates) if write_rates else 0.0
        return self.output(self.read_proj(ctx)), avg_wr


def evaluate_write_rate(model, ood_frac, n_batches=80):
    model.eval()
    wrs = []
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, ood_frac)
            logits, wr = model(seq)
            wrs.append(wr)
            pred = logits.argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return sum(wrs) / len(wrs), correct / total


class Exp353OODBehavior(Experiment):
    experiment_id = "exp_35_3"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        model = LearnedGateDeltaModel()
        opt = Adam(model.parameters(), lr=3e-4)
        model.train()
        for _ in range(STEPS):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, ood_frac=0.0)
            logits, _ = model(seq)
            loss = F.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

        ood_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
        write_rates = {}
        accs = {}
        for frac in ood_fracs:
            wr, acc = evaluate_write_rate(model, frac)
            write_rates[frac] = round(wr, 4)
            accs[frac] = round(acc, 4)
            print(f"  OOD_frac={frac:.2f} → write_rate={wr:.4f}  acc={acc:.4f}")

        baseline_wr = write_rates[0.0]
        ood_wr = write_rates[1.0]
        wr_ratio = ood_wr / max(baseline_wr, 1e-6)

        metrics = {f"wr_ood_{int(f*100)}": v for f, v in write_rates.items()}
        metrics.update({f"acc_ood_{int(f*100)}": v for f, v in accs.items()})
        metrics["baseline_write_rate"] = round(baseline_wr, 4)
        metrics["full_ood_write_rate"] = round(ood_wr, 4)
        metrics["wr_ratio_ood_vs_baseline"] = round(wr_ratio, 4)

        if wr_ratio > 2.0 or wr_ratio < 0.5:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Abnormal OOD write rate: baseline={baseline_wr:.3f}, "
                     f"OOD={ood_wr:.3f}, ratio={wr_ratio:.2f} (outside [0.5, 2.0]). "
                     f"Gate not robust to OOD inputs.")
        elif 0.75 <= wr_ratio <= 1.33:
            outcome = OUTCOME_REFUTED
            notes = (f"Stable write rate under OOD: baseline={baseline_wr:.3f}, "
                     f"OOD={ood_wr:.3f}, ratio={wr_ratio:.2f} within [0.75, 1.33].")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate OOD effect: wr_ratio={wr_ratio:.2f}. "
                     f"Baseline={baseline_wr:.3f}, OOD={ood_wr:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, train_range=TRAIN_RANGE))


if __name__ == "__main__":
    Exp353OODBehavior().execute()
