"""
Experiment 32.1 — Deep Seed Validation: Retroactive Writing (exp_3_6 replication)

Replicates exp_3_6 with 9 seeds {0,1,2,7,13,42,99,123,777}.
Stricter criterion: gap > 0.09 (vs original's gap > 0.02).
SUPPORTED if two-pass accuracy exceeds forward-only by more than 9%.
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
SEQ_LEN       = 24
HIDDEN_DIM    = 64
MEMORY_SLOTS  = 6
BATCH_SIZE    = 32
TRAIN_STEPS   = 1500
LR            = 3e-4
NUM_PAIRS     = 4
QUERY_MARKER  = 2
FORWARD_SLOTS = 4


def make_assoc_batch(batch_size: int):
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 3, (NUM_PAIRS,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (1,))])[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = QUERY_MARKER; seq[b, SEQ_LEN - 2] = keys[qi]
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


class ForwardGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM//2), nn.ReLU(),
                                  nn.Linear(HIDDEN_DIM//2, 1))
    def forward(self, hidden): return torch.sigmoid(self.gate(hidden)).squeeze(-1)


class RevisionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.gate = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ReLU(),
                                  nn.Linear(HIDDEN_DIM, 1))
    def forward(self, hidden, skipped_mask):
        summary = self.summary_proj(hidden.mean(dim=1))
        combined = torch.cat([hidden, summary.unsqueeze(1).expand_as(hidden)], dim=-1)
        scores = torch.sigmoid(self.gate(combined)).squeeze(-1)
        return scores * skipped_mask


class ReadHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, query_h, memory, mask):
        q = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1).masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx  = (attn.unsqueeze(-1) * memory).sum(1)
        return self.out(ctx)


def assemble_fwd_only(hidden, gate_scores):
    B, L, H = hidden.shape; k = FORWARD_SLOTS
    _, topk_idx = torch.topk(gate_scores, k, dim=1)
    memory = torch.gather(hidden, 1, topk_idx.unsqueeze(-1).expand(-1,-1,H))
    mask   = torch.ones(B, k); pad = MEMORY_SLOTS - k
    memory = F.pad(memory, (0,0,0,pad)); mask = F.pad(mask, (0,pad))
    return memory, mask, topk_idx


def assemble_two_pass(hidden, fwd_gate, rev_gate):
    B, L, H = hidden.shape
    fwd_scores = fwd_gate(hidden)
    _, fwd_idx = torch.topk(fwd_scores, FORWARD_SLOTS, dim=1)
    written = torch.zeros(B, L); written.scatter_(1, fwd_idx, 1.0)
    skipped = 1.0 - written
    k_retro = MEMORY_SLOTS - FORWARD_SLOTS
    rev_scores = rev_gate(hidden, skipped)
    _, rev_idx = torch.topk(rev_scores, k_retro, dim=1)
    all_mem = []; all_mask = []; retro_rates = []
    for b in range(B):
        fwd_pos = fwd_idx[b].tolist(); rev_pos = rev_idx[b].tolist()
        retro_rates.append(sum(1 for p in rev_pos if written[b,p].item() < 0.5) / L)
        idx_t = torch.tensor(fwd_pos + rev_pos)
        all_mem.append(hidden[b, idx_t, :])
        all_mask.append(torch.ones(MEMORY_SLOTS))
    return torch.stack(all_mem), torch.stack(all_mask), sum(retro_rates)/len(retro_rates)


def train_model(is_two_pass):
    enc = Encoder(); fwd_gate = ForwardGate(); read_head = ReadHead()
    rev_gate = RevisionGate() if is_two_pass else None
    params = list(enc.parameters()) + list(fwd_gate.parameters()) + list(read_head.parameters())
    if rev_gate: params += list(rev_gate.parameters())
    opt = Adam(params, lr=LR)
    for _ in range(TRAIN_STEPS):
        seq, tgt = make_assoc_batch(BATCH_SIZE)
        hidden = enc(seq)
        if is_two_pass:
            memory, mask, _ = assemble_two_pass(hidden, fwd_gate, rev_gate)
        else:
            gate_scores = fwd_gate(hidden)
            memory, mask, _ = assemble_fwd_only(hidden, gate_scores)
        loss = F.cross_entropy(read_head(hidden[:,-1,:], memory, mask), tgt)
        loss.backward(); opt.step(); opt.zero_grad()
    return enc, read_head, fwd_gate, rev_gate


def eval_accuracy(enc, read_head, fwd_gate, rev_gate, is_two_pass):
    total_acc = total_retro = 0.0; N = 20
    with torch.no_grad():
        for _ in range(N):
            seq, tgt = make_assoc_batch(BATCH_SIZE)
            hidden = enc(seq)
            if is_two_pass:
                memory, mask, retro = assemble_two_pass(hidden, fwd_gate, rev_gate)
                total_retro += retro
            else:
                gate_scores = fwd_gate(hidden)
                memory, mask, _ = assemble_fwd_only(hidden, gate_scores)
            logits = read_head(hidden[:,-1,:], memory, mask)
            total_acc += (logits.argmax(-1) == tgt).float().mean().item()
    return total_acc/N, total_retro/N


class Exp321RetroactiveWriting9Seeds(Experiment):
    experiment_id = "exp_32_1"
    hypothesis = ("Deep seed validation of retroactive writing (exp_3_6): "
                  "gap > 0.09 (stricter criterion for robust confirmation).")

    def run(self) -> ExperimentResult:
        torch.manual_seed(self.seed)
        config = dict(
            vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM,
            memory_slots=MEMORY_SLOTS, forward_slots=FORWARD_SLOTS,
            batch_size=BATCH_SIZE, train_steps=TRAIN_STEPS, num_pairs=NUM_PAIRS,
            seed=self.seed, replicates="exp_3_6",
            param_bytes=sum(p.numel()*4 for p in list(Encoder().parameters()) +
                           list(ForwardGate().parameters()) + list(ReadHead().parameters()) +
                           list(RevisionGate().parameters())),
            activation_bytes=BATCH_SIZE * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training forward-only model...")
        enc_fwd, head_fwd, fwd_g_fwd, _ = train_model(False)
        print("  Training two-pass model...")
        enc_two, head_two, fwd_g_two, rev_g = train_model(True)
        fwd_acc, _ = eval_accuracy(enc_fwd, head_fwd, fwd_g_fwd, None, False)
        two_acc, retro_rate = eval_accuracy(enc_two, head_two, fwd_g_two, rev_g, True)
        gap = two_acc - fwd_acc
        print(f"  forward={fwd_acc:.3f} two_pass={two_acc:.3f} gap={gap:+.3f} retro_rate={retro_rate:.3f}")
        metrics = dict(
            forward_acc=round(fwd_acc, 4), two_pass_acc=round(two_acc, 4),
            acc_gap=round(gap, 4), retroactive_write_rate=round(retro_rate, 4), seed=self.seed,
        )
        if gap > 0.09:         outcome = OUTCOME_SUPPORTED
        elif gap < 0.0:         outcome = OUTCOME_REFUTED
        else:                   outcome = OUTCOME_INCONCLUSIVE
        notes = (f"Seed={self.seed}. Two-pass vs forward gap={gap:+.3f}. "
                 f"Retroactive write rate={retro_rate:.3f}. Required gap > 0.09.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp321RetroactiveWriting9Seeds().execute()
