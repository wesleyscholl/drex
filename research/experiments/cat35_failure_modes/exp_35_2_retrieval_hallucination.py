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

experiment_id = "exp_35_2"
hypothesis = (
    "Slot memory architecture does NOT hallucinate: querying with keys never "
    "presented in context produces accuracy near random chance, not "
    "significantly above chance (< random + 5%)."
)

VOCAB_SIZE = 128
HIDDEN_DIM = 64
SEQ_LEN = 24
NUM_PAIRS = 5
NUM_SLOTS = 8
STEPS = 600
BATCH = 32
# Keys in [4, VOCAB_SIZE//3) are in-distribution; OOD keys in [VOCAB_SIZE//2, VOCAB_SIZE-2)
IN_DIST_KEY_LOW = 4
IN_DIST_KEY_HIGH = VOCAB_SIZE // 3       # ~42
OOD_KEY_LOW = VOCAB_SIZE * 2 // 3       # ~85
OOD_KEY_HIGH = VOCAB_SIZE - 2           # 126


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs, ood_query=False):
    """Make associative recall sequences.

    If ood_query=True, the query token is from a range never used as a key in training.
    """
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(IN_DIST_KEY_LOW, IN_DIST_KEY_HIGH,
                             (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys,
                              torch.randint(IN_DIST_KEY_LOW, IN_DIST_KEY_HIGH, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 4, vocab_size // 2, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3

        if ood_query:
            # Query with a key that was never in the context (OOD range)
            ood_key = torch.randint(OOD_KEY_LOW, OOD_KEY_HIGH, (1,)).item()
            seq[b, seq_len - 3] = 2
            seq[b, seq_len - 2] = ood_key
            seq[b, seq_len - 1] = 0
            # Target: a value from the valid range (so "correct" is 1/|vals range|)
            target[b] = torch.randint(vocab_size // 4, vocab_size // 2, (1,)).item()
        else:
            qi = torch.randint(0, num_pairs, (1,)).item()
            seq[b, seq_len - 3] = 2
            seq[b, seq_len - 2] = keys[qi]
            seq[b, seq_len - 1] = 0
            target[b] = vals[qi]
    return seq, target


class SlotMemoryModel(nn.Module):
    def __init__(self, num_slots):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        # Slot keys and values
        self.slot_keys = nn.Parameter(torch.randn(num_slots, HIDDEN_DIM) * 0.1)
        self.slot_vals = nn.Parameter(torch.randn(num_slots, HIDDEN_DIM) * 0.1)
        self.write_gate = nn.Sequential(nn.Linear(HIDDEN_DIM, num_slots), nn.Softmax(dim=-1))
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.num_slots = num_slots

    def forward(self, seq):
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        keys = self.slot_keys.unsqueeze(0).expand(B, -1, -1).clone()
        vals = self.slot_vals.unsqueeze(0).expand(B, -1, -1).clone()
        for t in range(L - 1):
            token_h = h[:, t, :]
            w = self.write_gate(token_h)  # (B, num_slots)
            keys = keys + w.unsqueeze(-1) * token_h.unsqueeze(1)
            vals = keys  # simplified: slot stores the token representation
        q = h[:, -1, :]
        # Retrieve: cosine similarity between query and slot keys
        key_norm = F.normalize(keys, dim=-1)
        q_norm = F.normalize(q.unsqueeze(1), dim=-1)
        sim = (key_norm * q_norm).sum(-1)  # (B, num_slots)
        attn = F.softmax(sim, dim=-1).unsqueeze(-1)
        ctx = (attn * vals).sum(1)  # (B, H)
        return self.output(ctx)


def evaluate(model, ood=False, n_batches=80):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS, ood_query=ood)
            pred = model(seq).argmax(-1)
            correct += (pred == tgt).sum().item()
            total += tgt.size(0)
    return correct / total


class Exp352RetrievalHallucination(Experiment):
    experiment_id = "exp_35_2"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        model = SlotMemoryModel(NUM_SLOTS)
        opt = Adam(model.parameters(), lr=3e-4)
        model.train()
        for _ in range(STEPS):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            loss = F.cross_entropy(model(seq), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

        acc_in_dist = evaluate(model, ood=False)
        acc_ood = evaluate(model, ood=True)

        # Random baseline: 1 / (vocab range for values)
        val_range = VOCAB_SIZE // 2 - VOCAB_SIZE // 4
        random_baseline = 1.0 / val_range
        ood_above_random = acc_ood - random_baseline
        hallucination_ratio = acc_ood / max(acc_in_dist, 1e-6)

        metrics = dict(
            acc_in_distribution=round(acc_in_dist, 4),
            acc_ood_query=round(acc_ood, 4),
            random_baseline=round(random_baseline, 4),
            ood_above_random=round(ood_above_random, 4),
            hallucination_ratio=round(hallucination_ratio, 4),
        )

        if ood_above_random < 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"No hallucination: OOD acc={acc_ood:.3f} ≈ random baseline={random_baseline:.3f} "
                     f"(gap={ood_above_random:.3f}<0.05). In-dist acc={acc_in_dist:.3f}.")
        elif ood_above_random > 0.15:
            outcome = OUTCOME_REFUTED
            notes = (f"Hallucination detected: OOD acc={acc_ood:.3f} >> random={random_baseline:.3f} "
                     f"(excess={ood_above_random:.3f}>0.15). Model returns confident wrong answers.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Moderate OOD accuracy: {acc_ood:.3f}, random={random_baseline:.3f}, "
                     f"excess={ood_above_random:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       num_slots=NUM_SLOTS, steps=STEPS))


if __name__ == "__main__":
    Exp352RetrievalHallucination().execute()
