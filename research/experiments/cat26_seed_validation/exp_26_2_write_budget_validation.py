"""
Experiment 26.2 — Write Budget Validation (Seed Stability for exp_9_5)

Hypothesis: Write budget with positionally biased task (exp_9_5) is seed-stable:
additional seeds confirm non-uniform (oracle) budget outperforms uniform allocation.
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

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 32
MEMORY_SLOTS  = 6
N_BLOCKS      = 4
BLOCK_SIZE    = SEQ_LEN // N_BLOCKS   # 8
NUM_PAIRS     = 4
BATCH_SIZE    = 32
TRAIN_STEPS   = 2000
LR            = 3e-4
DEVICE        = "cpu"
ORACLE_SLOTS_BLOCK0 = 4   # force 4 of 6 slots to block-0 positions


def make_assoc_batch_block0(batch_size=BATCH_SIZE):
    """All KV pairs placed in block 0 (positions 0 to BLOCK_SIZE-1)."""
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 2, (NUM_PAIRS * 2,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 2, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < BLOCK_SIZE:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        target[b] = vals[qi]
    return seq, target


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
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1).masked_fill(mask == 0, -1e9)
        return self.out((torch.softmax(scores, dim=-1).unsqueeze(-1) * memory).sum(1))


class WriteGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


def select_uniform(hidden):
    """Take first MEMORY_SLOTS tokens."""
    B, L, H = hidden.shape
    k = min(MEMORY_SLOTS, L - 3)
    return hidden[:, :k, :], torch.ones(B, k, device=hidden.device)


def select_oracle(hidden):
    """Take ORACLE_SLOTS_BLOCK0 tokens from block0, rest from remaining."""
    B, L, H = hidden.shape
    b0 = min(ORACLE_SLOTS_BLOCK0, BLOCK_SIZE, L)
    b1 = min(MEMORY_SLOTS - b0, L - b0)
    block0 = hidden[:, :b0, :]
    block1 = hidden[:, b0: b0 + b1, :]
    memory = torch.cat([block0, block1], dim=1)
    mask   = torch.ones(B, memory.size(1), device=hidden.device)
    return memory, mask


def select_learned(hidden, gate_scores):
    B, L, H = hidden.shape
    k   = min(MEMORY_SLOTS, L)
    _, idx = gate_scores.topk(k, dim=-1)
    return hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H)), torch.ones(B, k)


def train_eval(mode, steps=TRAIN_STEPS, n_eval=50):
    enc = Encoder().to(DEVICE); rh = ReadHead().to(DEVICE)
    gate = WriteGate().to(DEVICE) if mode == "learned" else None
    params = list(enc.parameters()) + list(rh.parameters())
    if gate:
        params += list(gate.parameters())
    opt = Adam(params, lr=LR)
    enc.train(); rh.train()
    if gate: gate.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch_block0()
        hidden   = enc(seq)
        if mode == "uniform":
            mem, mask = select_uniform(hidden)
        elif mode == "oracle":
            mem, mask = select_oracle(hidden)
        else:
            gs = gate(hidden); mem, mask = select_learned(hidden, gs)
        loss = F.cross_entropy(rh(hidden[:, -1, :], mem, mask), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); rh.eval()
    if gate: gate.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_assoc_batch_block0()
            hidden = enc(seq)
            if mode == "uniform":
                mem, mask = select_uniform(hidden)
            elif mode == "oracle":
                mem, mask = select_oracle(hidden)
            else:
                gs = gate(hidden); mem, mask = select_learned(hidden, gs)
            total += (rh(hidden[:, -1, :], mem, mask).argmax(-1) == tgt).float().mean().item()
    return total / n_eval


class Exp262WriteBudgetValidation(Experiment):
    experiment_id = "exp_26_2"
    hypothesis = ("Write budget with positionally biased task (exp_9_5) is seed-stable: "
                  "additional seeds confirm non-uniform (oracle) budget outperforms uniform.")

    def run(self) -> ExperimentResult:
        print("  Uniform allocation...")
        acc_A = train_eval("uniform"); print(f"    acc_A={acc_A:.3f}")
        print("  Oracle block-0 allocation...")
        acc_B = train_eval("oracle"); print(f"    acc_B={acc_B:.3f}")
        print("  Learned gate allocation...")
        acc_C = train_eval("learned"); print(f"    acc_C={acc_C:.3f}")
        gap_oracle = acc_B - acc_A; gap_learned = acc_C - acc_A
        if acc_B > acc_A + 0.05 and acc_C > acc_A + 0.03:
            outcome = OUTCOME_SUPPORTED
        elif acc_B <= acc_A:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE
        metrics = dict(acc_uniform=round(acc_A,4), acc_oracle=round(acc_B,4),
                       acc_learned=round(acc_C,4),
                       uniform_to_oracle_gap=round(gap_oracle,4),
                       uniform_to_learned_gap=round(gap_learned,4))
        notes = (f"Uniform={acc_A:.3f}, Oracle={acc_B:.3f}, Learned={acc_C:.3f}. "
                 f"Oracle gap={gap_oracle:.3f}, Learned gap={gap_learned:.3f}.")
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      memory_slots=MEMORY_SLOTS, n_blocks=N_BLOCKS, num_pairs=NUM_PAIRS,
                      oracle_block0_slots=ORACLE_SLOTS_BLOCK0, train_steps=TRAIN_STEPS)
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp262WriteBudgetValidation().execute()
