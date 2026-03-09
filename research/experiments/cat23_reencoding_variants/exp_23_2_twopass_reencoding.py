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

experiment_id = "exp_23_2"
hypothesis = ("Two-pass re-encoding yields diminishing returns: the second re-encoding "
              "pass provides <20% of the accuracy gain from the first pass.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 24
NUM_PAIRS     = 4
FORWARD_SLOTS = 4
STEPS         = 1500
BATCH         = 32
LR            = 3e-4


def make_assoc_batch(batch_size):
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 2, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 2, (1,))])[:NUM_PAIRS]
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


class ForwardGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
                                  nn.Linear(HIDDEN_DIM // 2, 1), nn.Sigmoid())

    def forward(self, h):
        return self.gate(h).squeeze(-1)


class CrossAttentionReenc(nn.Module):
    """Single cross-attention re-encoding layer."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.norm   = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, slots, context):
        """slots: (B,k,H), context: (B,L,H) → refined slots (B,k,H)"""
        H = slots.size(-1)
        Sq = self.q_proj(slots);  Sk = self.k_proj(context);  Sv = self.v_proj(context)
        attn = F.softmax(torch.bmm(Sq, Sk.transpose(1, 2)) / H**0.5, dim=-1)
        return self.norm(slots + torch.bmm(attn, Sv))


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h, memory):
        H = query_h.size(-1)
        q    = self.q_proj(query_h)
        attn = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return self.out((attn.unsqueeze(-1) * memory).sum(1))


class NPassReencModel(nn.Module):
    """Re-encoding N times (0 = no re-encoding baseline)."""
    def __init__(self, n_passes: int):
        super().__init__()
        self.encoder   = Encoder()
        self.fwd_gate  = ForwardGate()
        self.reenc_layers = nn.ModuleList([CrossAttentionReenc() for _ in range(n_passes)])
        self.read_head = ReadHead()
        self.n_passes  = n_passes

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        scores = self.fwd_gate(hidden)
        _, idx = scores.topk(min(FORWARD_SLOTS, L), dim=-1)
        slots  = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        for reenc in self.reenc_layers:
            slots = reenc(slots, hidden)
        return self.read_head(hidden[:, -1, :], slots)


def train_and_eval(n_passes, n_eval=50):
    model = NPassReencModel(n_passes)
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


class Exp232TwoPassReencoding(Experiment):
    experiment_id = "exp_23_2"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, forward_slots=FORWARD_SLOTS,
                      steps=STEPS, batch=BATCH)
        print("Training 0-pass (no re-encoding) model (A)...")
        acc_0 = train_and_eval(0)
        print(f"  0-pass: acc={acc_0:.3f}")
        print("Training 1-pass re-encoding model (B)...")
        acc_1 = train_and_eval(1)
        print(f"  1-pass: acc={acc_1:.3f}")
        print("Training 2-pass re-encoding model (C)...")
        acc_2 = train_and_eval(2)
        print(f"  2-pass: acc={acc_2:.3f}")

        gain_1 = acc_1 - acc_0
        gain_2 = acc_2 - acc_1
        diminish_ratio = gain_2 / max(abs(gain_1), 1e-6)

        metrics = dict(acc_0pass=round(acc_0, 4), acc_1pass=round(acc_1, 4),
                       acc_2pass=round(acc_2, 4),
                       gain_pass1=round(gain_1, 4), gain_pass2=round(gain_2, 4),
                       diminishment_ratio=round(diminish_ratio, 4))
        if abs(gain_1) > 0.01 and diminish_ratio < 0.20:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Gain1={gain_1:.3f}, Gain2={gain_2:.3f}, ratio={diminish_ratio:.3f}<0.20."
        elif diminish_ratio > 0.80:
            outcome = OUTCOME_REFUTED
            notes   = f"No diminishing returns: ratio={diminish_ratio:.3f}>0.80."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Gain1={gain_1:.3f}, Gain2={gain_2:.3f}, ratio={diminish_ratio:.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp232TwoPassReencoding().execute()
