"""
Experiment 26.1 — Protected Slots Validation (Seed Stability for exp_9_4)

Hypothesis: Protected slot interior optimum (exp_9_4) is seed-stable: 3 additional
seeds confirm an interior peak at K=3-6 with MEMORY_SLOTS=12.
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
MEMORY_SLOTS  = 12
NUM_PAIRS     = 5
NUM_CRITICAL  = 3
BATCH_SIZE    = 32
TRAIN_STEPS   = 500
LR            = 3e-4
DEVICE        = "cpu"
K_VALUES      = [0, 2, 4, 6, 8, 10]


def make_assoc_batch_protected(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                                num_pairs=NUM_PAIRS, num_critical=NUM_CRITICAL):
    seq               = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target            = torch.zeros(batch_size, dtype=torch.long)
    critical_mask     = torch.zeros(batch_size, seq_len)
    query_is_critical = torch.zeros(batch_size, dtype=torch.bool)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 2, (num_pairs * 2,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 2, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]
                if i < num_critical:
                    critical_mask[b, pos] = 1.0; critical_mask[b, pos + 1] = 1.0
                pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]; query_is_critical[b] = (qi < num_critical)
    return seq, target, critical_mask, query_is_critical


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class WriteGate(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h)).squeeze(-1)


class ReadHead(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, query_h, memory, mask):
        q      = self.q_proj(query_h)
        scores = torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn   = torch.softmax(scores, dim=-1)
        return self.out((attn.unsqueeze(-1) * memory).sum(1))


def select_memory_slots(hidden, write_scores, critical_mask, k_protected):
    B, L, H     = hidden.shape
    total_slots = min(MEMORY_SLOTS, L)
    k_open      = total_slots - k_protected
    all_indices = []
    for b in range(B):
        crit_pos = (critical_mask[b] > 0.5).nonzero(as_tuple=True)[0]
        nc_pos   = (critical_mask[b] <= 0.5).nonzero(as_tuple=True)[0]
        if k_protected > 0 and len(crit_pos) > 0:
            n   = min(k_protected, len(crit_pos))
            idx = write_scores[b, crit_pos].topk(n).indices
            prot_idx = crit_pos[idx]
        else:
            prot_idx = torch.tensor([], dtype=torch.long)
        if k_open > 0 and len(nc_pos) > 0:
            n   = min(k_open, len(nc_pos))
            idx = write_scores[b, nc_pos].topk(n).indices
            open_idx = nc_pos[idx]
        else:
            open_idx = torch.tensor([], dtype=torch.long)
        combined = torch.cat([prot_idx, open_idx])
        if len(combined) < total_slots:
            all_pos   = torch.arange(L)
            used_mask = torch.zeros(L, dtype=torch.bool)
            if len(combined) > 0:
                used_mask[combined] = True
            unused    = all_pos[~used_mask][:total_slots - len(combined)]
            combined  = torch.cat([combined, unused])
        all_indices.append(combined[:total_slots])
    indices = torch.stack(all_indices)
    memory  = hidden.gather(1, indices.unsqueeze(-1).expand(-1, -1, H))
    mask    = torch.ones(B, total_slots, device=hidden.device)
    return memory, mask


def train_and_eval_k(k_protected):
    enc  = Encoder().to(DEVICE); gate = WriteGate().to(DEVICE); rh = ReadHead().to(DEVICE)
    opt  = Adam(list(enc.parameters()) + list(gate.parameters()) + list(rh.parameters()), lr=LR)
    enc.train(); gate.train(); rh.train()
    for _ in range(TRAIN_STEPS):
        seq, target, crit_mask, _ = make_assoc_batch_protected(BATCH_SIZE)
        hidden = enc(seq); ws = gate(hidden)
        memory, mask = select_memory_slots(hidden, ws, crit_mask, k_protected)
        loss = F.cross_entropy(rh(hidden[:, -1, :], memory, mask), target)
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); gate.eval(); rh.eval()
    total = crit = nc = 0.0; cn = ncn = 0; n_eval = 50
    with torch.no_grad():
        for _ in range(n_eval):
            seq, target, crit_mask, qic = make_assoc_batch_protected(BATCH_SIZE)
            hidden = enc(seq); ws = gate(hidden)
            memory, mask = select_memory_slots(hidden, ws, crit_mask, k_protected)
            preds   = rh(hidden[:, -1, :], memory, mask).argmax(-1)
            correct = (preds == target)
            total += correct.float().mean().item()
            if qic.any():
                crit += correct[qic].float().mean().item(); cn += 1
            if (~qic).any():
                nc += correct[~qic].float().mean().item(); ncn += 1
    return dict(acc=total/n_eval, crit_acc=crit/max(cn,1), nc_acc=nc/max(ncn,1))


class Exp261ProtectedSlotsValidation(Experiment):
    experiment_id = "exp_26_1"
    hypothesis = ("Protected slot interior optimum (exp_9_4) is seed-stable: 3 additional "
                  "seeds confirm an interior peak at K=3-6 with MEMORY_SLOTS=12.")

    def run(self) -> ExperimentResult:
        accs = []; crit_accs = []; nc_accs = []
        for k in K_VALUES:
            print(f"  K={k}...")
            r = train_and_eval_k(k)
            accs.append(r["acc"]); crit_accs.append(r["crit_acc"]); nc_accs.append(r["nc_acc"])
            print(f"    acc={r['acc']:.3f}, crit_acc={r['crit_acc']:.3f}, nc_acc={r['nc_acc']:.3f}")
        acc_tensor = torch.tensor(accs)
        max_acc = acc_tensor.max().item(); min_acc = acc_tensor.min().item()
        k_opt   = int(acc_tensor.argmax().item())
        interior = (2 <= k_opt <= 8 and accs[0] < max_acc - 0.02 and accs[-1] < max_acc - 0.02)
        if interior:
            outcome = OUTCOME_SUPPORTED
        elif k_opt == 10:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE
        metrics = dict(
            accuracies=[round(a,4) for a in accs], critical_accuracies=[round(a,4) for a in crit_accs],
            noncritical_accuracies=[round(a,4) for a in nc_accs],
            k_opt=k_opt, max_acc=round(max_acc,4), min_acc=round(min_acc,4),
            acc_range=round(max_acc-min_acc,4), interior_peak_exists=interior,
        )
        notes = (f"K values {K_VALUES}, accs {[round(a,3) for a in accs]}. "
                 f"Optimal K={k_opt}, max_acc={max_acc:.3f}. Interior peak: {interior}.")
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      memory_slots=MEMORY_SLOTS, num_pairs=NUM_PAIRS,
                      num_critical=NUM_CRITICAL, train_steps=TRAIN_STEPS, k_values=K_VALUES)
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp261ProtectedSlotsValidation().execute()
