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

experiment_id = "exp_22_3"
hypothesis = ("Orthogonal slot initialization via Gram-Schmidt plus orthogonality "
              "regularization prevents slot collapse and improves read accuracy by "
              ">5% without changing the read mechanism.")

VOCAB_SIZE   = 64
HIDDEN_DIM   = 64
SEQ_LEN      = 32
NUM_PAIRS    = 4
MEMORY_SLOTS = 8
STEPS        = 2000
BATCH        = 32
LR           = 3e-4
ORTH_LAMBDA  = 0.1   # orthogonality regularization weight


def make_assoc_batch(batch_size):
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, use_orthogonal_init=False):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
        if use_orthogonal_init:
            nn.init.orthogonal_(self.embed.weight)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


def orth_reg_loss(slots):
    """Regularize slot embeddings to be mutually orthogonal.
    slots: (B, k, H) → returns scalar loss."""
    norm_s = F.normalize(slots, dim=-1)           # (B, k, H)
    gram   = torch.bmm(norm_s, norm_s.transpose(1, 2))   # (B, k, k)
    k      = gram.size(1)
    I      = torch.eye(k, device=slots.device).unsqueeze(0)
    return (gram - I).pow(2).mean()


class StandardSlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(use_orthogonal_init=False)
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
        return self.output((attn.unsqueeze(-1) * memory).sum(1)), memory


class OrthSlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(use_orthogonal_init=True)
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
        return self.output((attn.unsqueeze(-1) * memory).sum(1)), memory


def slot_collapse_score(model, n_batches=20):
    """Mean pairwise cosine similarity across slots — lower = more diverse."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, _ = make_assoc_batch(BATCH)
            _, memory = model(seq)
            norm_m = F.normalize(memory, dim=-1)           # (B, k, H)
            gram   = torch.bmm(norm_m, norm_m.transpose(1, 2))  # (B, k, k)
            k      = gram.size(1)
            mask   = 1 - torch.eye(k, device=gram.device)
            total += (gram * mask).abs().mean().item()
    return total / n_batches


def train_and_eval(model, use_orth_reg=False, n_eval=50):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH)
        logits, memory = model(seq)
        loss = F.cross_entropy(logits, tgt)
        if use_orth_reg:
            loss = loss + ORTH_LAMBDA * orth_reg_loss(memory)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_assoc_batch(BATCH)
            logits, _ = model(seq)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp223OrthogonalSlots(Experiment):
    experiment_id = "exp_22_3"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, memory_slots=MEMORY_SLOTS,
                      steps=STEPS, batch=BATCH, orth_lambda=ORTH_LAMBDA)
        print("Training standard model (A)...")
        model_A = StandardSlotModel()
        acc_A = train_and_eval(model_A, use_orth_reg=False)
        collapse_A = slot_collapse_score(model_A)
        print(f"  A: acc={acc_A:.3f}, collapse={collapse_A:.3f}")

        print("Training orthogonal-slot model (B)...")
        model_B = OrthSlotModel()
        acc_B = train_and_eval(model_B, use_orth_reg=True)
        collapse_B = slot_collapse_score(model_B)
        print(f"  B: acc={acc_B:.3f}, collapse={collapse_B:.3f}")

        gap = acc_B - acc_A
        collapse_reduction = collapse_A - collapse_B
        metrics = dict(acc_A=round(acc_A, 4), acc_B=round(acc_B, 4),
                       gap=round(gap, 4),
                       collapse_A=round(collapse_A, 4), collapse_B=round(collapse_B, 4),
                       collapse_reduction=round(collapse_reduction, 4))
        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Orth-slot improved by {gap:.3f} > 0.05. Collapse reduced by {collapse_reduction:.3f}."
        elif gap < -0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Standard model outperforms (gap={gap:.3f})."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Gap={gap:.3f}, collapse_reduction={collapse_reduction:.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp223OrthogonalSlots().execute()
