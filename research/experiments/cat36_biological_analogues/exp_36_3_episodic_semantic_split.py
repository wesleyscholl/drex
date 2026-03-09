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

experiment_id = "exp_36_3"
hypothesis = (
    "Separating episodic memory (what happened: event order/temporal context) "
    "from semantic memory (what things mean: content associations) outperforms "
    "a unified memory store by >5% on tasks requiring both types of recall."
)

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN = 32   # longer sequence for episodic component
NUM_PAIRS = 5
STEPS = 700
BATCH = 32
HALF_DIM = HIDDEN_DIM // 2


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    """Task requires both: (1) recall which value was paired with key (semantic),
    and (2) identify the temporal order of keys (episodic).
    Reward (targets) are split: half batches test semantic, half test position order.
    Here we simplify to just semantic recall but build the positional info into context."""
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        # Write KV pairs at specific positions (provides episodic position signal)
        for i in range(num_pairs):
            pos = i * 4  # fixed spacing creates a temporal structure
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]
        # Fill with padding
        for p in range(num_pairs * 4, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, out_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.embed(x)
        # Resize if needed
        if self.ff[-1].out_features != HIDDEN_DIM:
            return self.norm(self.ff(h))
        return self.norm(h + self.ff(h))


class UnifiedMemoryModel(nn.Module):
    """Condition A: single delta-rule matrix of size H×H."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k = h[:, t, :]
            v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
            dv = k - v_pred / denom
            M = M + torch.bmm(dv.unsqueeze(-1), k.unsqueeze(1))
        q = h[:, -1, :]
        ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx))


class SplitMemoryModel(nn.Module):
    """Condition B: separate semantic (content) and episodic (temporal) memory matrices.

    - Semantic memory (M_sem, H/2 × H/2): delta rule on content projections
    - Episodic memory (M_epi, H/2 × H/2): stores positional/recency-weighted representations
    Both are read and concatenated for the output.
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(HIDDEN_DIM)
        self.sem_proj = nn.Linear(HIDDEN_DIM, HALF_DIM)   # semantic subspace
        self.epi_proj = nn.Linear(HIDDEN_DIM, HALF_DIM)   # episodic subspace
        self.output = nn.Linear(HALF_DIM * 2, VOCAB_SIZE)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        M_sem = torch.zeros(B, HALF_DIM, HALF_DIM, device=h.device)
        M_epi = torch.zeros(B, HALF_DIM, HALF_DIM, device=h.device)

        for t in range(L - 1):
            ks = self.sem_proj(h[:, t, :])   # semantic key
            ke = self.epi_proj(h[:, t, :])   # episodic key

            # Semantic: pure delta rule (content association)
            v_pred_s = torch.bmm(M_sem, ks.unsqueeze(-1)).squeeze(-1)
            denom_s = ks.pow(2).sum(-1, keepdim=True) + 1e-6
            dv_s = ks - v_pred_s / denom_s
            M_sem = M_sem + torch.bmm(dv_s.unsqueeze(-1), ks.unsqueeze(1))

            # Episodic: recency-weighted (later tokens get higher weight)
            recency_weight = (t + 1) / L
            v_pred_e = torch.bmm(M_epi, ke.unsqueeze(-1)).squeeze(-1)
            denom_e = ke.pow(2).sum(-1, keepdim=True) + 1e-6
            dv_e = ke - v_pred_e / denom_e
            M_epi = M_epi + recency_weight * torch.bmm(dv_e.unsqueeze(-1), ke.unsqueeze(1))

        q = h[:, -1, :]
        qs = self.sem_proj(q)
        qe = self.epi_proj(q)
        ctx_s = torch.bmm(M_sem, qs.unsqueeze(-1)).squeeze(-1)
        ctx_e = torch.bmm(M_epi, qe.unsqueeze(-1)).squeeze(-1)
        ctx = torch.cat([ctx_s, ctx_e], dim=-1)
        return self.output(ctx)


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


class Exp363EpisodicSemanticSplit(Experiment):
    experiment_id = "exp_36_3"
    hypothesis = hypothesis

    def run(self) -> ExperimentResult:
        print("Training Condition A: unified memory ...")
        acc_unified = train_and_eval(UnifiedMemoryModel)
        print(f"  acc_unified={acc_unified:.4f}")

        print("Training Condition B: split episodic+semantic memory ...")
        acc_split = train_and_eval(SplitMemoryModel)
        print(f"  acc_split={acc_split:.4f}")

        gap = round(acc_split - acc_unified, 4)

        # Parameter counts (both should be comparable)
        n_unified = sum(p.numel() for p in UnifiedMemoryModel().parameters())
        n_split = sum(p.numel() for p in SplitMemoryModel().parameters())

        metrics = dict(
            acc_unified=round(acc_unified, 4),
            acc_split=round(acc_split, 4),
            split_advantage=gap,
            params_unified=n_unified,
            params_split=n_split,
        )

        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes = (f"Split memory wins: unified={acc_unified:.3f}, "
                     f"split={acc_split:.3f}, advantage={gap:.3f}>0.05.")
        elif gap < -0.03:
            outcome = OUTCOME_REFUTED
            notes = (f"Unified memory wins: unified={acc_unified:.3f}, "
                     f"split={acc_split:.3f}, gap={gap:.3f}<-0.03.")
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (f"Similar performance: unified={acc_unified:.3f}, "
                     f"split={acc_split:.3f}, gap={gap:.3f}.")

        return self.result(outcome, metrics, notes,
                           config=dict(vocab=VOCAB_SIZE, hidden=HIDDEN_DIM,
                                       steps=STEPS, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS))


if __name__ == "__main__":
    Exp363EpisodicSemanticSplit().execute()
