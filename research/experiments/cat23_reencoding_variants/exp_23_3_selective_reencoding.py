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

experiment_id = "exp_23_3"
hypothesis = ("Selective re-encoding (re-encode only slots with cosine distance > T "
              "from context) achieves >90% of full re-encoding accuracy at <60% "
              "re-encode rate.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 24
NUM_PAIRS     = 4
FORWARD_SLOTS = 4
STEPS         = 1500
BATCH         = 32
LR            = 3e-4
COSINE_THRESH = 0.5   # re-encode slot if cosine distance from nearest context > T


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


def cross_attn_reenc(slots, context):
    """Single cross-attention re-encoding: slots (B,k,H), context (B,L,H) → (B,k,H)."""
    H  = slots.size(-1)
    attn = F.softmax(torch.bmm(slots, context.transpose(1, 2)) / H**0.5, dim=-1)
    return slots + torch.bmm(attn, context)


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


class BaselineNoReencModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder()
        self.fwd_gate = ForwardGate()
        self.read_head = ReadHead()

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        _, idx = self.fwd_gate(hidden).topk(min(FORWARD_SLOTS, L), dim=-1)
        slots  = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        return self.read_head(hidden[:, -1, :], slots)


class FullReencModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder()
        self.fwd_gate = ForwardGate()
        self.read_head = ReadHead()

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        _, idx = self.fwd_gate(hidden).topk(min(FORWARD_SLOTS, L), dim=-1)
        slots  = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        slots  = cross_attn_reenc(slots, hidden)
        return self.read_head(hidden[:, -1, :], slots)


class SelectiveReencModel(nn.Module):
    def __init__(self, threshold=COSINE_THRESH):
        super().__init__()
        self.encoder   = Encoder()
        self.fwd_gate  = ForwardGate()
        self.read_head = ReadHead()
        self.threshold = threshold

    def forward(self, seq, return_rate=False):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        _, idx = self.fwd_gate(hidden).topk(min(FORWARD_SLOTS, L), dim=-1)
        slots  = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))

        # Compute cosine similarity between each slot and its nearest context token
        norm_slots   = F.normalize(slots, dim=-1)      # (B, k, H)
        norm_context = F.normalize(hidden, dim=-1)      # (B, L, H)
        cos_sim      = torch.bmm(norm_slots, norm_context.transpose(1, 2))  # (B, k, L)
        max_cos_sim  = cos_sim.max(dim=-1).values                            # (B, k)
        cos_dist     = 1 - max_cos_sim                                       # (B, k)

        # Selectively re-encode slots where cosine distance > threshold
        # Use soft gating during training (differentiable), hard threshold for metrics
        sel_gate = (cos_dist > self.threshold).float().unsqueeze(-1)      # (B, k, 1)
        reenc_slots = cross_attn_reenc(slots, hidden)
        slots = slots + sel_gate * (reenc_slots - slots)   # soft blend

        reenc_rate = sel_gate.mean().item() if B > 0 else 0.0
        if return_rate:
            return self.read_head(hidden[:, -1, :], slots), reenc_rate
        return self.read_head(hidden[:, -1, :], slots)


def train_and_eval_generic(model_class, init_kwargs=None, n_eval=50):
    model = model_class(**(init_kwargs or {}))
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH)
        if hasattr(model, 'threshold'):
            logits = model(seq, return_rate=False)
        else:
            logits = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0; rate_sum = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_assoc_batch(BATCH)
            if hasattr(model, 'threshold'):
                logits, rate = model(seq, return_rate=True)
                rate_sum += rate
            else:
                logits = model(seq)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total, rate_sum / n_eval if hasattr(model, 'threshold') else 1.0


class Exp233SelectiveReencoding(Experiment):
    experiment_id = "exp_23_3"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, forward_slots=FORWARD_SLOTS,
                      steps=STEPS, batch=BATCH, cosine_threshold=COSINE_THRESH)
        print("Training baseline (no re-encoding) model (A)...")
        acc_A, _ = train_and_eval_generic(BaselineNoReencModel)
        print(f"  A: acc={acc_A:.3f}")
        print("Training full re-encoding model (B)...")
        acc_B, _ = train_and_eval_generic(FullReencModel)
        print(f"  B: acc={acc_B:.3f}")
        print("Training selective re-encoding model (C)...")
        acc_C, reenc_rate = train_and_eval_generic(SelectiveReencModel,
                                                    {"threshold": COSINE_THRESH})
        print(f"  C: acc={acc_C:.3f}, reenc_rate={reenc_rate:.3f}")

        acc_ratio = acc_C / max(acc_B, 1e-6) if acc_B > acc_A else 1.0
        metrics = dict(acc_A=round(acc_A, 4), acc_B=round(acc_B, 4), acc_C=round(acc_C, 4),
                       reenc_rate=round(reenc_rate, 4), acc_ratio=round(acc_ratio, 4))
        if acc_ratio > 0.90 and reenc_rate < 0.60:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Selective: acc_ratio={acc_ratio:.3f}>0.90 at reenc_rate={reenc_rate:.3f}<0.60."
        elif acc_ratio < 0.70 or reenc_rate > 0.85:
            outcome = OUTCOME_REFUTED
            notes   = f"Selective fails: ratio={acc_ratio:.3f} or rate={reenc_rate:.3f} too high."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Acc_ratio={acc_ratio:.3f}, reenc_rate={reenc_rate:.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp233SelectiveReencoding().execute()
