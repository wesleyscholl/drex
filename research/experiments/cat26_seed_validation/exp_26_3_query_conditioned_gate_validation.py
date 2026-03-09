"""
Experiment 26.3 — Query-Conditioned Gate Validation (Seed Stability for exp_17_1)

Hypothesis: Query-conditioned write gate (exp_17_1) improvement over context-only
gate is seed-stable: additional seeds confirm >5% accuracy improvement on
multi-query-type tasks.
"""
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

VOCAB_SIZE       = 64
HIDDEN_DIM       = 64
SEQ_LEN          = 32
NUM_QUERY_TYPES  = 4
MEMORY_SLOTS     = 8
STEPS            = 400
BATCH            = 32
NUM_PAIRS        = 4
LR               = 3e-4


def make_typed_batch(batch_size):
    seq         = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target      = torch.zeros(batch_size, dtype=torch.long)
    query_types = torch.randint(0, NUM_QUERY_TYPES, (batch_size,))
    for b in range(batch_size):
        qt        = query_types[b].item()
        k_start   = 4 + qt * 10; k_end = k_start + 10
        keys      = torch.randint(k_start, k_end, (NUM_PAIRS,))
        vals      = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 4:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 4):
            seq[b, p] = 3
        seq[b, SEQ_LEN - 4] = qt
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        target[b] = vals[qi]
    return seq, target, query_types


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q    = self.q_proj(query_h)
        sc   = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1).masked_fill(mask == 0, -1e9)
        attn = torch.softmax(sc, dim=-1)
        return self.out((attn.unsqueeze(-1) * memory).sum(1))


class ContextOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder()
        self.gate_net = nn.Linear(HIDDEN_DIM, 1)
        self.rh       = ReadHead()

    def forward(self, seq):
        B, L = seq.shape
        h = self.encoder(seq)
        gate = torch.sigmoid(self.gate_net(h)).squeeze(-1)
        _, idx = gate.topk(min(MEMORY_SLOTS, L), dim=-1)
        mem = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
        return self.rh(h[:, -1, :], mem, torch.ones(B, mem.size(1)))


class QueryConditionedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder      = Encoder()
        self.type_pred    = nn.Linear(HIDDEN_DIM, NUM_QUERY_TYPES)
        self.type_proj    = nn.Linear(NUM_QUERY_TYPES, HIDDEN_DIM // 2)
        self.gate_net     = nn.Linear(HIDDEN_DIM + HIDDEN_DIM // 2, 1)
        self.rh           = ReadHead()

    def forward(self, seq):
        B, L = seq.shape
        h = self.encoder(seq)
        type_dist = torch.softmax(self.type_pred(h.mean(1)), dim=-1)     # (B, T)
        type_emb  = self.type_proj(type_dist).unsqueeze(1).expand(-1, L, -1)
        gate = torch.sigmoid(self.gate_net(torch.cat([h, type_emb], dim=-1))).squeeze(-1)
        _, idx = gate.topk(min(MEMORY_SLOTS, L), dim=-1)
        mem = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, HIDDEN_DIM))
        return self.rh(h[:, -1, :], mem, torch.ones(B, mem.size(1))), type_dist


def train_context_only(steps=STEPS):
    m = ContextOnlyModel(); opt = Adam(m.parameters(), lr=LR); m.train()
    for _ in range(steps):
        seq, tgt, _ = make_typed_batch(BATCH)
        loss = F.cross_entropy(m(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def eval_context_only(m, steps=200):
    m.eval(); c = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, tgt, _ = make_typed_batch(BATCH)
            c += (m(seq).argmax(-1) == tgt).sum().item()
    return c / (steps * BATCH)


def train_querycond(steps=STEPS):
    m = QueryConditionedModel(); opt = Adam(m.parameters(), lr=LR); m.train()
    for _ in range(steps):
        seq, tgt, qt = make_typed_batch(BATCH)
        logits, type_dist = m(seq)
        loss = F.cross_entropy(logits, tgt) + 0.3 * F.cross_entropy(type_dist, qt)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def eval_querycond(m, steps=200):
    m.eval(); c = tc = 0
    with torch.no_grad():
        for _ in range(steps):
            seq, tgt, qt = make_typed_batch(BATCH)
            logits, td = m(seq)
            c += (logits.argmax(-1) == tgt).sum().item()
            tc += (td.argmax(-1) == qt).sum().item()
    n = steps * BATCH
    return c / n, tc / n


class Exp263QueryConditionedGateValidation(Experiment):
    experiment_id = "exp_26_3"
    hypothesis = ("Query-conditioned write gate (exp_17_1) improvement over context-only "
                  "gate is seed-stable: additional seeds confirm >5% accuracy improvement "
                  "on multi-query-type tasks.")

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_query_types=NUM_QUERY_TYPES, memory_slots=MEMORY_SLOTS,
                      steps=STEPS, batch=BATCH)
        print("Training context-only gate (A)...")
        model_a  = train_context_only()
        acc_a    = eval_context_only(model_a)
        print(f"  acc_A={acc_a:.4f}")
        print("Training query-conditioned gate (B)...")
        model_b  = train_querycond()
        acc_b, qp = eval_querycond(model_b)
        print(f"  acc_B={acc_b:.4f}, query_pred_acc={qp:.4f}")
        gap = acc_b - acc_a
        if acc_b > acc_a + 0.05 and qp > 0.40:
            outcome = OUTCOME_SUPPORTED
        elif acc_a >= acc_b - 0.02:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE
        metrics = dict(acc_A=round(acc_a,4), acc_B=round(acc_b,4),
                       gap=round(gap,4), query_pred_acc=round(qp,4))
        notes = (f"Context-only={acc_a:.3f}, QueryCond={acc_b:.3f}, "
                 f"gap={gap:.3f}, qtype_pred={qp:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp263QueryConditionedGateValidation().execute()
