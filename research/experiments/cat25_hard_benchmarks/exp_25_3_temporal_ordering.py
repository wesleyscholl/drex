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

experiment_id = "exp_25_3"
hypothesis = ("Temporal ordering task (retrieve k-th event in temporal sequence): "
              "accuracy drops monotonically with k, revealing the read bottleneck "
              "for ordered memory access.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 32
MAX_K         = 6     # max k-th event to recall
N_EVENTS      = 6     # number of distinct events in the sequence
MEMORY_SLOTS  = 8
STEPS         = 2000
BATCH         = 32
LR            = 3e-4


def make_temporal_batch(batch_size, query_k):
    """
    Sequence: [e1, e2, ..., eN, ..., PAD, MARKER, k_token, 0]
    Target: event at position k (1-indexed).
    Each event is a unique token; position is implicit via token order.
    """
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        # Sample N_EVENTS distinct event tokens
        events = torch.randperm(VOCAB_SIZE - 4)[:N_EVENTS] + 4
        pos = 0
        for i in range(N_EVENTS):
            if pos < SEQ_LEN - 4:
                seq[b, pos] = events[i]; pos += 1
        for p in range(pos, SEQ_LEN - 4):
            seq[b, p] = 3
        # Query: retrieve event at position k (1-indexed)
        k = min(query_k, N_EVENTS)
        seq[b, SEQ_LEN - 4] = 2      # MARKER
        seq[b, SEQ_LEN - 3] = k      # query: which position
        seq[b, SEQ_LEN - 2] = 0
        seq[b, SEQ_LEN - 1] = 0
        target[b] = events[k - 1]    # 1-indexed
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.pos   = nn.Embedding(SEQ_LEN, HIDDEN_DIM)  # position embeddings
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        B, L = x.shape
        pos  = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h    = self.embed(x) + self.pos(pos)
        return self.norm(h + self.ff(h))


class MemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.gate    = nn.Linear(HIDDEN_DIM, 1)
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        _, idx = self.gate(h).squeeze(-1).topk(min(MEMORY_SLOTS, L), dim=-1)
        memory = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        q    = self.q_proj(h[:, -1, :])
        attn = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return self.output((attn.unsqueeze(-1) * memory).sum(1))


def train_model(steps=STEPS, k_values=None):
    """Train on all k values simultaneously."""
    if k_values is None:
        k_values = list(range(1, MAX_K + 1))
    model = MemoryModel()
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        # Cycle through k values
        k = k_values[step % len(k_values)]
        seq, tgt = make_temporal_batch(BATCH, k)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, n_eval=50):
    model.eval()
    accs = {}
    with torch.no_grad():
        for k in range(1, MAX_K + 1):
            correct = total = 0
            for _ in range(n_eval):
                seq, tgt = make_temporal_batch(BATCH, k)
                correct += (model(seq).argmax(-1) == tgt).sum().item()
                total   += tgt.size(0)
            accs[k] = round(correct / total, 4)
    return accs


def is_monotone_decreasing(accs_dict):
    """Check if accuracy decreases with k."""
    vals = [accs_dict[k] for k in sorted(accs_dict)]
    return all(vals[i] >= vals[i+1] - 0.01 for i in range(len(vals) - 1))


class Exp253TemporalOrdering(Experiment):
    experiment_id = "exp_25_3"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                      seq_len=SEQ_LEN, n_events=N_EVENTS, max_k=MAX_K,
                      memory_slots=MEMORY_SLOTS, steps=STEPS, batch=BATCH)
        print("Training temporal ordering model...")
        model = train_model()
        accs  = evaluate(model)
        for k, a in sorted(accs.items()):
            print(f"  k={k}: acc={a:.3f}")

        vals = [accs[k] for k in sorted(accs)]
        monotone = is_monotone_decreasing(accs)
        first_to_last_drop = vals[0] - vals[-1]

        metrics = dict(
            acc_by_k=accs, monotone_decreasing=monotone,
            first_to_last_drop=round(first_to_last_drop, 4),
            k1_acc=vals[0], k_max_acc=vals[-1],
        )
        if monotone and first_to_last_drop > 0.20:
            outcome = OUTCOME_SUPPORTED
            notes   = (f"Temporal accuracy monotone-decreasing with k. "
                       f"k=1 to k={MAX_K} drop={first_to_last_drop:.3f}>0.20.")
        elif vals[-1] > vals[0] - 0.05:
            outcome = OUTCOME_REFUTED
            notes   = f"Temporal ordering is nearly flat (drop={first_to_last_drop:.3f}<0.05)."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = (f"Drop={first_to_last_drop:.3f} but not monotone ({monotone}). "
                       f"Partial temporal bottleneck.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp253TemporalOrdering().execute()
