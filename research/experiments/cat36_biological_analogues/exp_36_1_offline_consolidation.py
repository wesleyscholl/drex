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

experiment_id = "exp_36_1"
hypothesis = (
    "An offline consolidation phase — replaying all written key-value pairs "
    "through the memory without new input — improves associative recall accuracy "
    "by >3% over single-pass writing, analogous to hippocampal-cortical consolidation."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 6   # slightly larger to stress memory more
STEPS = 600
BATCH = 32
CONSOLIDATION_PASSES = 2  # how many extra consolidation passes through stored pairs


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    stored_kv = []   # (keys, vals) tuples for consolidation
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
        stored_kv.append((keys[:num_pairs], vals[:num_pairs]))
    return seq, target, stored_kv


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

    def encode(self, tok_ids):
        h = self.embed(tok_ids)
        return self.norm(h + self.ff(h))

    def forward_with_consolidation(self, seq, consolidate=False):
        """Forward pass. If consolidate=True, replay the K/V pairs an extra time."""
        h = self.encode(seq)
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)

        # Initial write pass
        for t in range(L - 1):
            k = h[:, t, :]
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))

        if consolidate:
            # Offline consolidation: re-pass the context tokens (excluding query)
            for _ in range(CONSOLIDATION_PASSES):
                for t in range(L - 3):   # only context tokens, not the query triplet
                    k = h[:, t, :].detach()
                    v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
                    denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
                    dv = k - v_pred / denom
                    M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))

        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))

    def forward(self, seq):
        return self.forward_with_consolidation(seq, consolidate=False)


def train_epoch(model, opt, steps):
    model.train()
    for _ in range(steps):
        seq, tgt, _ = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()


def evaluate(model, consolidate, n_batches=60):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt, _ = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            pred = model.forward_with_consolidation(seq, consolidate=consolidate).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp361OfflineConsolidation(Experiment):
    experiment_id = "exp_36_1"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        model = DeltaModel()
        opt = Adam(model.parameters(), lr=3e-4)

        print("Training model ...")
        train_epoch(model, opt, STEPS)

        print("Evaluating without consolidation ...")
        acc_no_consol = evaluate(model, consolidate=False)
        print(f"  acc_no_consolidation={acc_no_consol:.4f}")

        print(f"Evaluating WITH consolidation ({CONSOLIDATION_PASSES} extra passes) ...")
        acc_with_consol = evaluate(model, consolidate=True)
        print(f"  acc_with_consolidation={acc_with_consol:.4f}")

        gain = round(acc_with_consol - acc_no_consol, 4)

        metrics = dict(
            acc_no_consolidation=round(acc_no_consol, 4),
            acc_with_consolidation=round(acc_with_consol, 4),
            consolidation_gain=gain,
            consolidation_passes=CONSOLIDATION_PASSES,
        )

        if gain > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Consolidation improves accuracy: {acc_no_consol:.3f} → {acc_with_consol:.3f}, "
                     f"gain={gain:.3f}>0.03. Offline replay is beneficial.")
        elif gain < -0.01:
            outcome = OUTCOME_REFUTED
            notes = (f"Consolidation hurts: {acc_no_consol:.3f} → {acc_with_consol:.3f}, "
                     f"gain={gain:.3f}. Extra passes overwrite useful information.")
        else:
            outcome = OUTCOME_REFUTED
            notes = (f"No consolidation benefit: gain={gain:.3f} (threshold >0.03). "
                     f"Offline replay does not improve retrieval.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, consolidation_passes=CONSOLIDATION_PASSES,
                                       num_pairs=NUM_PAIRS))


if __name__ == "__main__":
    Exp361OfflineConsolidation().execute()
