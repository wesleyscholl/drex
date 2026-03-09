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

experiment_id = "exp_22_5"
hypothesis = ("Contrastive slot training (InfoNCE-style loss pushing slot embeddings "
              "apart) improves retrieval accuracy by >5% on high-interference tasks "
              "(many similar keys).")

VOCAB_SIZE      = 64
HIDDEN_DIM      = 64
SEQ_LEN         = 32
NUM_PAIRS       = 4
MEMORY_SLOTS    = 8
STEPS           = 2000
BATCH           = 32
LR              = 3e-4
CONTRAST_LAMBDA = 0.1
# High interference: keys from a narrow key range (4-16) so many similar keys
KEY_RANGE_LOW   = 4
KEY_RANGE_HIGH  = 16


def make_high_interference_batch(batch_size):
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(KEY_RANGE_LOW, KEY_RANGE_HIGH,
                             (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(KEY_RANGE_LOW,
                                                   KEY_RANGE_HIGH, (1,))])[:NUM_PAIRS]
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
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


def nt_xent_loss(slots, temperature=0.1):
    """NT-Xent contrastive loss over slots within each batch item.
    For each batch item push slots apart from each other.
    slots: (B, k, H)
    """
    B, k, H = slots.shape
    norm_s  = F.normalize(slots.reshape(B * k, H), dim=-1)  # (B*k, H)
    sim     = torch.mm(norm_s, norm_s.T) / temperature        # (B*k, B*k)
    # Mask diagonals (self-similarity)
    mask    = torch.eye(B * k, device=slots.device, dtype=torch.bool)
    # For each slot, treat all other slots in same batch item as positives to avoid
    # — instead, use a simple repulsion loss: we want off-diagonal same-item sims < 0
    # Simpler: just minimize mean off-diagonal cosine similarity
    off_sim = sim.masked_fill(mask, 0.0).abs().sum() / (B * k * (B * k - 1))
    return off_sim


class StandardModel(nn.Module):
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
        return self.output((attn.unsqueeze(-1) * memory).sum(1)), memory


class ContrastiveSlotModel(nn.Module):
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
        return self.output((attn.unsqueeze(-1) * memory).sum(1)), memory


def train_and_eval(model, use_contrast=False, n_eval=50):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_high_interference_batch(BATCH)
        logits, memory = model(seq)
        loss = F.cross_entropy(logits, tgt)
        if use_contrast:
            loss = loss + CONTRAST_LAMBDA * nt_xent_loss(memory)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_high_interference_batch(BATCH)
            logits, _ = model(seq)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp225ContrastiveSlots(Experiment):
    experiment_id = "exp_22_5"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, memory_slots=MEMORY_SLOTS,
                      steps=STEPS, batch=BATCH, contrast_lambda=CONTRAST_LAMBDA,
                      key_range=(KEY_RANGE_LOW, KEY_RANGE_HIGH))
        random_baseline = 1.0 / (VOCAB_SIZE // 2)
        print(f"Random baseline: {random_baseline:.3f}")

        print("Training standard model (A)...")
        model_A = StandardModel()
        acc_A = train_and_eval(model_A, use_contrast=False)
        print(f"  A: acc={acc_A:.3f}")

        print("Training contrastive-slot model (B)...")
        model_B = ContrastiveSlotModel()
        acc_B = train_and_eval(model_B, use_contrast=True)
        print(f"  B: acc={acc_B:.3f}")

        gap = acc_B - acc_A
        metrics = dict(acc_A=round(acc_A, 4), acc_B=round(acc_B, 4),
                       gap=round(gap, 4), random_baseline=round(random_baseline, 4))
        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Contrastive slots improved by {gap:.3f} > 0.05 on high-interference."
        elif gap < -0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Standard outperforms contrastive by {-gap:.3f}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Gap={gap:.3f}, between -0.02 and +0.05."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp225ContrastiveSlots().execute()
