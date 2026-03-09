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

experiment_id = "exp_35_1"
hypothesis = (
    "Delta rule memory degrades gracefully under post-hoc noise injection: "
    "accuracy at 30% additive noise to M is >60% of clean baseline, "
    "not a catastrophic cliff (>70% accuracy drop)."
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

    def forward(self, seq, noise_scale=0.0):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
        # Inject noise proportional to M's Frobenius norm
        if noise_scale > 0.0:
            m_norm = M.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            noise = torch.randn_like(M) * noise_scale * m_norm
            M = M + noise
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))


def evaluate_at_noise(model, noise_scale, n_batches=60):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model(seq, noise_scale=noise_scale).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp351MemoryPoisoning(Experiment):
    experiment_id = "exp_35_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        model = DeltaModel()
        opt = Adam(model.parameters(), lr=3e-4)
        model.train()
        for _ in range(STEPS):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            loss = F.cross_entropy(model(seq), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

        noise_levels = [0.0, 0.10, 0.25, 0.50, 1.0, 2.0]
        accs = {}
        for ns in noise_levels:
            acc = evaluate_at_noise(model, ns)
            accs[ns] = round(acc, 4)
            print(f"  noise={ns:.2f} → acc={acc:.4f}")

        baseline = accs[0.0]
        acc_at_30 = accs.get(0.25, accs[0.50])  # closest to 30% noise
        acc_at_50 = accs[0.50]
        retention_30 = acc_at_30 / max(baseline, 1e-6)
        retention_50 = acc_at_50 / max(baseline, 1e-6)

        # Check for cliff: >50% drop between two consecutive noise levels
        sorted_ns = sorted(accs.keys())
        cliff_detected = False
        cliff_at = None
        for i in range(1, len(sorted_ns)):
            drop = accs[sorted_ns[i - 1]] - accs[sorted_ns[i]]
            if drop > 0.5 * baseline:
                cliff_detected = True
                cliff_at = sorted_ns[i]
                break

        metrics = {f"acc_noise_{int(ns*100)}pct": v for ns, v in accs.items()}
        metrics["baseline_acc"] = round(baseline, 4)
        metrics["retention_at_25pct_noise"] = round(retention_30, 4)
        metrics["retention_at_50pct_noise"] = round(retention_50, 4)
        metrics["cliff_detected"] = cliff_detected
        metrics["cliff_at_noise"] = cliff_at

        if not cliff_detected and retention_30 > 0.60:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Graceful degradation: baseline={baseline:.3f}, "
                     f"@25%noise={acc_at_30:.3f} (retention={retention_30:.3f}>0.60). "
                     f"No cliff detected.")
        elif cliff_detected or retention_30 < 0.30:
            outcome = OUTCOME_REFUTED
            notes = (f"Catastrophic degradation: baseline={baseline:.3f}, "
                     f"@25%noise={acc_at_30:.3f} (retention={retention_30:.3f}). "
                     f"Cliff at noise={cliff_at}.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate degradation: retention@25%={retention_30:.3f}. "
                     f"Cliff={cliff_detected}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, noise_levels=noise_levels))


if __name__ == "__main__":
    Exp351MemoryPoisoning().execute()
