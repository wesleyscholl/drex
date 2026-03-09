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

experiment_id = "exp_23_4"
hypothesis = ("Re-encoding gain is task-type specific: factual recall tasks show "
              ">2x the accuracy benefit of pattern-completion tasks from cross-attention "
              "re-encoding.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 24
NUM_PAIRS     = 4
FORWARD_SLOTS = 4
STEPS         = 1500
BATCH         = 32
LR            = 3e-4


# ── Task A: factual recall (random KV pairs) ──────────────────────────────────

def make_factual_batch(batch_size):
    """Standard random KV associative recall."""
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


# ── Task B: pattern completion (each val = key + 1 mod vocab; "count up" task) ─

def make_pattern_batch(batch_size):
    """Sequential pattern: val = (key + 1) mod VOCAB_SIZE for each KV pair."""
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE - 2, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE - 2, (1,))])[:NUM_PAIRS]
        vals = (keys + 1) % VOCAB_SIZE  # deterministic successor function
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


# ── Model components ──────────────────────────────────────────────────────────

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
        return self.out((attn.unsqueeze(-1) * memory).sum(1))


class SlotModel(nn.Module):
    def __init__(self, with_reenc=False):
        super().__init__()
        self.encoder   = Encoder()
        self.fwd_gate  = ForwardGate()
        self.read_head = ReadHead()
        self.with_reenc = with_reenc

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        _, idx = self.fwd_gate(hidden).topk(min(FORWARD_SLOTS, L), dim=-1)
        slots  = hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        if self.with_reenc:
            attn   = F.softmax(torch.bmm(slots, hidden.transpose(1, 2)) / H**0.5, dim=-1)
            slots  = slots + torch.bmm(attn, hidden)
        return self.read_head(hidden[:, -1, :], slots)


def train_and_eval(task, with_reenc=False, n_eval=50):
    model = SlotModel(with_reenc=with_reenc)
    opt   = Adam(model.parameters(), lr=LR)
    data_fn = make_factual_batch if task == "factual" else make_pattern_batch
    model.train()
    for _ in range(STEPS):
        seq, tgt = data_fn(BATCH)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = data_fn(BATCH)
            correct += (model(seq).argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp234ReencodingTaskType(Experiment):
    experiment_id = "exp_23_4"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, forward_slots=FORWARD_SLOTS,
                      steps=STEPS, batch=BATCH)
        print("Factual task — no re-encoding (A_fact)...")
        acc_fact_base = train_and_eval("factual", with_reenc=False)
        print(f"  {acc_fact_base:.3f}")
        print("Factual task — with re-encoding (B_fact)...")
        acc_fact_reenc = train_and_eval("factual", with_reenc=True)
        print(f"  {acc_fact_reenc:.3f}")
        print("Pattern task — no re-encoding (A_patt)...")
        acc_patt_base = train_and_eval("pattern", with_reenc=False)
        print(f"  {acc_patt_base:.3f}")
        print("Pattern task — with re-encoding (B_patt)...")
        acc_patt_reenc = train_and_eval("pattern", with_reenc=True)
        print(f"  {acc_patt_reenc:.3f}")

        gain_factual = acc_fact_reenc - acc_fact_base
        gain_pattern = acc_patt_reenc - acc_patt_base
        specificity_ratio = gain_factual / max(abs(gain_pattern), 1e-6)

        metrics = dict(
            acc_factual_base=round(acc_fact_base, 4), acc_factual_reenc=round(acc_fact_reenc, 4),
            acc_pattern_base=round(acc_patt_base, 4), acc_pattern_reenc=round(acc_patt_reenc, 4),
            gain_factual=round(gain_factual, 4), gain_pattern=round(gain_pattern, 4),
            specificity_ratio=round(specificity_ratio, 4),
        )
        if gain_factual > 0.02 and specificity_ratio > 2.0:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Factual gain={gain_factual:.3f}, pattern gain={gain_pattern:.3f}, ratio={specificity_ratio:.3f}>2.0."
        elif specificity_ratio < 0.5:
            outcome = OUTCOME_REFUTED
            notes   = f"No task-type specificity: ratio={specificity_ratio:.3f}<0.5."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Factual gain={gain_factual:.3f}, pattern gain={gain_pattern:.3f}, ratio={specificity_ratio:.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp234ReencodingTaskType().execute()
