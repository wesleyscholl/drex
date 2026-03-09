"""
Experiment 33.4 — Training Budget Compensates for High Interference Density

Hypothesis: At ρ=1.0 (N_pairs=64, H=64), tripling training steps from 400 to 1200
recovers >50% of the accuracy lost vs ρ=0.5 (N_pairs=32) for at least one architecture.

recovery_fraction = (acc_rho1_1200 - acc_rho1_400) / max(acc_rho05_400 - acc_rho1_400, 1e-6)
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

HIDDEN_DIM   = 64
VOCAB_SIZE   = 256
BATCH        = 32
LR           = 3e-4
EVAL_BATCHES = 40
N_HIGH       = 64    # ρ = 1.0
N_MID        = 32    # ρ = 0.5 (reference "good" accuracy)
STEPS_REF    = 400
STEPS_LIST   = [400, 800, 1200]


def make_assoc_batch(batch_size, n_pairs, seq_len, vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    key_range = max(8, vocab_size // 3); val_base = vocab_size // 2
    for b in range(batch_size):
        keys = torch.randint(4, key_range, (n_pairs * 4,)).unique()[:n_pairs]
        while len(keys) < n_pairs:
            keys = torch.cat([keys, torch.randint(4, key_range, (1,))])[:n_pairs]
        vals = torch.randint(val_base, vocab_size, (n_pairs,)); pos = 0
        for i in range(n_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3): seq[b, p] = 3
        qi = torch.randint(0, n_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class SlotMemoryModel(nn.Module):
    def __init__(self, num_slots, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.num_slots = num_slots
        self.enc    = Encoder(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)
    def forward(self, seq):
        B, L = seq.shape; H = self.enc.embed.embedding_dim
        h = self.enc(seq); norms = h[:, :-3, :].norm(dim=-1)
        k = min(self.num_slots, L - 3); _, idx = torch.topk(norms, k, dim=-1)
        slots = torch.gather(h[:, :-3, :], 1, idx.unsqueeze(-1).expand(-1, -1, H))
        if k < self.num_slots:
            slots = torch.cat([slots, torch.zeros(B, self.num_slots - k, H)], dim=1)
        q    = self.q_proj(h[:, -1, :]).unsqueeze(1)
        attn = torch.softmax(torch.bmm(q, slots.transpose(1, 2)) / H**0.5, -1)
        return self.out(torch.bmm(attn, slots).squeeze(1))


class DeltaModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(vocab_size, hidden_dim)
        self.rp  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = hs[:, t, :]; v = hs[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        return self.out(self.rp(torch.bmm(M, hs[:, -1:, :].transpose(1, 2)).squeeze(-1)))


def train_and_eval(model_class, n_pairs, steps):
    seq_len = max(24, 2 * n_pairs + 8)
    num_slots = n_pairs + 2
    if model_class is SlotMemoryModel:
        model = SlotMemoryModel(num_slots)
    else:
        model = DeltaModel()
    opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(BATCH, n_pairs, seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, tgt = make_assoc_batch(BATCH, n_pairs, seq_len)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp334BudgetRecovery(Experiment):
    experiment_id = "exp_33_4"
    hypothesis = ("Tripling training steps (400→1200) at ρ=1.0 recovers >50% of accuracy "
                  "lost vs ρ=0.5 for at least one architecture.")

    def run(self) -> ExperimentResult:
        ref_slot  = sum(p.numel() * 4 for p in SlotMemoryModel(N_MID + 2).parameters())
        ref_delta = sum(p.numel() * 4 for p in DeltaModel().parameters())
        config = dict(
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, batch=BATCH,
            n_high=N_HIGH, n_mid=N_MID, steps_list=STEPS_LIST,
            param_bytes_slot=ref_slot, param_bytes_delta=ref_delta,
            activation_bytes=BATCH * max(24, 2 * N_HIGH + 8) * HIDDEN_DIM * 4,
        )

        # Reference accuracy at ρ=0.5 (good capacity regime)
        print(f"  Reference ρ=0.5 (N={N_MID})...")
        acc_slot_ref  = round(train_and_eval(SlotMemoryModel, N_MID, STEPS_REF), 4)
        acc_delta_ref = round(train_and_eval(DeltaModel, N_MID, STEPS_REF), 4)
        print(f"    slot_ref={acc_slot_ref:.4f} delta_ref={acc_delta_ref:.4f}")

        # Accuracy at ρ=1.0 with varying steps
        accs_slot = {}; accs_delta = {}
        for s in STEPS_LIST:
            print(f"  ρ=1.0 (N={N_HIGH}), steps={s}...")
            accs_slot[s]  = round(train_and_eval(SlotMemoryModel, N_HIGH, s), 4)
            accs_delta[s] = round(train_and_eval(DeltaModel, N_HIGH, s), 4)
            print(f"    slot={accs_slot[s]:.4f} delta={accs_delta[s]:.4f}")

        denom_slot  = max(acc_slot_ref  - accs_slot[400],  1e-6)
        denom_delta = max(acc_delta_ref - accs_delta[400], 1e-6)
        rec_slot  = round((accs_slot[1200]  - accs_slot[400])  / denom_slot,  4)
        rec_delta = round((accs_delta[1200] - accs_delta[400]) / denom_delta, 4)

        print(f"  Recovery: slot={rec_slot:.3f} delta={rec_delta:.3f}")

        metrics = dict(
            acc_slot_rho05=acc_slot_ref, acc_delta_rho05=acc_delta_ref,
        )
        for s in STEPS_LIST:
            metrics[f"acc_slot_{s}"]  = accs_slot[s]
            metrics[f"acc_delta_{s}"] = accs_delta[s]
        metrics["recovery_slot"]  = rec_slot
        metrics["recovery_delta"] = rec_delta

        if rec_slot > 0.50 or rec_delta > 0.50:
            outcome = OUTCOME_SUPPORTED
        elif rec_slot < 0.10 and rec_delta < 0.10:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Slot recovery={rec_slot:.3f}, Delta recovery={rec_delta:.3f}. "
                 f"ρ=0.5 ref: slot={acc_slot_ref:.3f}, delta={acc_delta_ref:.3f}. "
                 f"ρ=1.0@400: slot={accs_slot[400]:.3f}, delta={accs_delta[400]:.3f}. "
                 f"ρ=1.0@1200: slot={accs_slot[1200]:.3f}, delta={accs_delta[1200]:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp334BudgetRecovery().execute()
