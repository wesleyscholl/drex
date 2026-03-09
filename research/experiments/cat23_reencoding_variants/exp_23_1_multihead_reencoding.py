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

experiment_id = "exp_23_1"
hypothesis = ("Multi-head re-encoding (MHA num_heads=4 over slots) outperforms "
              "single-head cross-attention re-encoding by >3% due to richer "
              "slot-context interaction.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 24
NUM_PAIRS     = 4
MEMORY_SLOTS  = 6
FORWARD_SLOTS = 4   # top-k selected by forward gate
NUM_HEADS     = 4
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


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, query_h, memory):
        H = query_h.size(-1)
        q    = self.q_proj(query_h)
        attn = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        ctx  = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


def select_fwd_slots(hidden, fwd_gate):
    B, L, H = hidden.shape
    scores   = fwd_gate(hidden)
    _, idx   = scores.topk(min(FORWARD_SLOTS, L), dim=-1)
    return hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))


# ── Single-head re-encoding ───────────────────────────────────────────────────

class SingleHeadReencModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fwd_gate = ForwardGate()
        self.reenc_q  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.reenc_k  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.reenc_v  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.read_head = ReadHead()

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        slots  = select_fwd_slots(hidden, self.fwd_gate)     # (B, k, H)
        # Cross-attention: slots (as queries) over all token hiddens (as keys/values)
        Sq = self.reenc_q(slots)     # (B, k, H)
        Sk = self.reenc_k(hidden)    # (B, L, H)
        Sv = self.reenc_v(hidden)    # (B, L, H)
        attn = F.softmax(torch.bmm(Sq, Sk.transpose(1, 2)) / H**0.5, dim=-1)  # (B, k, L)
        reenc_slots = slots + torch.bmm(attn, Sv)             # (B, k, H) residual
        return self.read_head(hidden[:, -1, :], reenc_slots)


# ── Multi-head re-encoding ────────────────────────────────────────────────────

class MultiHeadReencModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.fwd_gate  = ForwardGate()
        self.mha       = nn.MultiheadAttention(HIDDEN_DIM, NUM_HEADS, batch_first=True)
        self.norm      = nn.LayerNorm(HIDDEN_DIM)
        self.read_head = ReadHead()

    def forward(self, seq):
        hidden = self.encoder(seq)
        slots  = select_fwd_slots(hidden, self.fwd_gate)   # (B, k, H)
        # MHA: query=slots, key=value=hidden
        reenc, _ = self.mha(slots, hidden, hidden)          # (B, k, H)
        reenc_slots = self.norm(slots + reenc)
        return self.read_head(hidden[:, -1, :], reenc_slots)


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


class Exp231MultiHeadReencoding(Experiment):
    experiment_id = "exp_23_1"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, memory_slots=MEMORY_SLOTS,
                      forward_slots=FORWARD_SLOTS, num_heads=NUM_HEADS,
                      steps=STEPS, batch=BATCH)
        print("Training single-head re-encoding model (A)...")
        acc_A = train_and_eval(SingleHeadReencModel)
        print(f"  A: acc={acc_A:.3f}")
        print("Training multi-head re-encoding model (B)...")
        acc_B = train_and_eval(MultiHeadReencModel)
        print(f"  B: acc={acc_B:.3f}")
        gap = acc_B - acc_A
        metrics = dict(acc_A=round(acc_A, 4), acc_B=round(acc_B, 4), gap=round(gap, 4))
        if gap > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes   = f"MHA re-encoding improved by {gap:.3f} > 0.03."
        elif gap < -0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Single-head re-encoding outperforms MHA by {-gap:.3f}."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Gap={gap:.3f}, between -0.02 and +0.03."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp231MultiHeadReencoding().execute()
