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

experiment_id = "exp_22_1"
hypothesis = ("Slot-conditioned read (soft attention over slots -> query refinement) "
              "reduces read error by >5% vs fixed linear-projection query on "
              "multi-fact retrieval (4 KV pairs, random baseline <10%).")

VOCAB_SIZE = 64
HIDDEN_DIM = 64
SEQ_LEN    = 32
NUM_PAIRS  = 4
MEMORY_SLOTS = 8
STEPS      = 2000
BATCH      = 32
LR         = 3e-4


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
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class StandardReadModel(nn.Module):
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
        memory = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))       # (B, k, H)
        query  = self.q_proj(h[:, -1, :])                                # (B, H)
        attn   = F.softmax(torch.bmm(memory, query.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return self.output((attn.unsqueeze(-1) * memory).sum(1))


class SlotConditionedReadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder       = Encoder()
        self.gate          = nn.Linear(HIDDEN_DIM, 1)
        self.q_init_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.slot_key_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.q_refine_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output        = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        _, idx = self.gate(h).squeeze(-1).topk(min(MEMORY_SLOTS, L), dim=-1)
        memory    = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))    # (B, k, H)
        q_init    = self.q_init_proj(h[:, -1, :])                       # (B, H)
        slot_keys = self.slot_key_proj(memory)                           # (B, k, H)
        # refine query via soft attention over slot keys
        r_attn    = F.softmax(torch.bmm(slot_keys, q_init.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        q_refined = q_init + self.q_refine_proj((r_attn.unsqueeze(-1) * memory).sum(1))
        attn      = F.softmax(torch.bmm(memory, q_refined.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return self.output((attn.unsqueeze(-1) * memory).sum(1))


def train_and_eval(model_class, n_eval=50):
    model = model_class()
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_assoc_batch(BATCH)
            correct += (model(seq).argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp221SlotConditionedRead(Experiment):
    experiment_id = "exp_22_1"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, memory_slots=MEMORY_SLOTS, steps=STEPS, batch=BATCH)
        print("Training standard read model (A)...")
        acc_A = train_and_eval(StandardReadModel)
        print(f"  A: acc={acc_A:.3f}")
        print("Training slot-conditioned read model (B)...")
        acc_B = train_and_eval(SlotConditionedReadModel)
        print(f"  B: acc={acc_B:.3f}")
        gap = acc_B - acc_A
        metrics = dict(acc_A=round(acc_A, 4), acc_B=round(acc_B, 4), gap=round(gap, 4))
        if gap > 0.05:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Slot-conditioned read improved by {gap:.3f} > 0.05."
        elif gap < -0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Standard read outperforms slot-conditioned by {-gap:.3f}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Gap={gap:.3f}, between -0.02 and +0.05."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp221SlotConditionedRead().execute()
