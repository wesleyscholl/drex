"""
Experiment 28.4 — Memory Slot Count Scaling

Hypothesis: Slot memory accuracy peaks when NUM_SLOTS is 1.5–2× NUM_PAIRS,
then degrades with excess slots (slot collapse: collapse_score increases).
NUM_PAIRS=6; sweep NUM_SLOTS ∈ {2,4,6,8,12,16,24}.
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

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
NUM_PAIRS   = 6
SEQ_LEN     = 24
STEPS       = 500
BATCH       = 32
LR          = 3e-4
EVAL_N      = 40
SLOT_COUNTS = [2, 4, 6, 8, 12, 16, 24]


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3: seq[b, pos] = keys[i]; seq[b, pos+1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0; tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class SlotModel(nn.Module):
    def __init__(self, num_slots):
        super().__init__()
        self.num_slots = num_slots; self.enc = Encoder()
        self.q = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        B, L = seq.shape; H = HIDDEN_DIM; hs = self.enc(seq)
        content = hs[:, :-3, :]; k = min(self.num_slots, content.shape[1])
        gate_scores = content.norm(dim=-1)   # (B, L-3)
        _, idx = torch.topk(gate_scores, k, dim=1)
        slots = torch.gather(content, 1, idx.unsqueeze(-1).expand(-1,-1,H))
        if k < self.num_slots:
            slots = torch.cat([slots, torch.zeros(B, self.num_slots-k, H)], dim=1)
        q = self.q(hs[:, -1, :]).unsqueeze(1)
        attn = torch.softmax(torch.bmm(q, slots.transpose(1,2)) / H**0.5, -1)
        return self.out(torch.bmm(attn, slots).squeeze(1))


def train_eval_slots(num_slots):
    model = SlotModel(num_slots)
    opt   = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()

    # Evaluate accuracy
    model.eval(); c = t = 0
    slot_vecs_all = []
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH)
            B, L = seq.shape; hs = model.enc(seq); content = hs[:, :-3, :]
            k  = min(num_slots, content.shape[1])
            _, idx = torch.topk(content.norm(dim=-1), k, dim=1)
            slots = torch.gather(content, 1, idx.unsqueeze(-1).expand(-1,-1,HIDDEN_DIM))
            slot_vecs_all.append(slots.reshape(-1, HIDDEN_DIM))
            logits = model(seq)
            c += (logits.argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    acc = c / t

    # Collapse score: mean pairwise cosine similarity among slots (higher = more collapse)
    if slot_vecs_all:
        sv = torch.cat(slot_vecs_all[:5], dim=0)  # sample 5 batches
        sv_n = F.normalize(sv, dim=-1)
        gram = torch.mm(sv_n, sv_n.t())
        n_s  = sv_n.shape[0]
        off_diag = gram - torch.eye(n_s)
        collapse = off_diag.abs().mean().item()
    else:
        collapse = 0.0
    return acc, collapse


class Exp284SlotCountScaling(Experiment):
    experiment_id = "exp_28_4"
    hypothesis = ("Slot accuracy peaks at NUM_SLOTS=1.5–2×NUM_PAIRS (9–12 slots), "
                  "then degrades with excess slots due to slot collapse.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, slot_counts=SLOT_COUNTS,
            param_bytes=sum(p.numel()*4 for p in SlotModel(8).parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )

        accs = {}; collapses = {}
        for k in SLOT_COUNTS:
            print(f"  num_slots={k}...")
            acc, col = train_eval_slots(k)
            accs[k] = round(acc, 4); collapses[k] = round(col, 4)
            print(f"    acc={acc:.3f}, collapse={col:.3f}")

        peak_k = max(SLOT_COUNTS, key=lambda k: accs[k])
        acc_peak = accs[peak_k]; acc_24 = accs[24]
        metrics = {f"acc_k{k}":  accs[k]  for k in SLOT_COUNTS}
        metrics.update({f"collapse_k{k}": collapses[k] for k in SLOT_COUNTS})
        metrics["peak_k"]  = peak_k
        metrics["acc_peak"] = round(acc_peak, 4)
        metrics["acc_k24_vs_peak"] = round(acc_24 - acc_peak, 4)

        # SUPPORTED: peak in [9,14] range AND k=24 is <peak-0.05
        peak_in_range = 8 <= peak_k <= 14
        degradation   = acc_peak - acc_24 > 0.05

        if peak_in_range and degradation:
            outcome = OUTCOME_SUPPORTED
        elif peak_k <= 4 or (not degradation and acc_24 > acc_peak * 0.98):
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Peak at k={peak_k} (acc={acc_peak:.3f}). "
                 f"k=24 acc={acc_24:.3f} (drop={acc_peak-acc_24:.3f}).")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp284SlotCountScaling().execute()
