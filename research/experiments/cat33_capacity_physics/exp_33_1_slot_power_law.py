"""
Experiment 33.1 — Interference Density Power Law (Slot Memory)

Hypothesis: Slot memory accuracy follows acc ~ ρ^(-γ) where ρ = N_pairs/hidden_dim,
with power-law fit R² > 0.90 across ρ ∈ {0.03–1.0}.

This experiment establishes the baseline interference exponent γ for slot memory,
the first step toward the Interference Density Law (a novel publishable finding).
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

HIDDEN_DIM  = 64
VOCAB_SIZE  = 256
STEPS       = 400
BATCH       = 32
LR          = 3e-4
N_PAIRS_LIST = [2, 4, 8, 16, 32, 64]   # ρ = n/H ∈ {0.031, 0.063, 0.125, 0.25, 0.5, 1.0}
EVAL_BATCHES = 40


def make_assoc_batch(batch_size, n_pairs, seq_len, vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    key_range = max(8, vocab_size // 3)
    val_base  = vocab_size // 2
    for b in range(batch_size):
        keys = torch.randint(4, key_range, (n_pairs * 4,)).unique()[:n_pairs]
        while len(keys) < n_pairs:
            keys = torch.cat([keys, torch.randint(4, key_range, (1,))])[:n_pairs]
        vals = torch.randint(val_base, vocab_size, (n_pairs,))
        pos = 0
        for i in range(n_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
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
        h = self.enc(seq)
        # Write top-num_slots by L2-norm magnitude (dense write)
        norms = h[:, :-3, :].norm(dim=-1)                        # (B, L-3)
        k     = min(self.num_slots, L - 3)
        _, idx = torch.topk(norms, k, dim=-1)                    # (B, k)
        slots  = torch.gather(h[:, :-3, :], 1,
                              idx.unsqueeze(-1).expand(-1, -1, H))  # (B, k, H)
        if k < self.num_slots:
            pad   = torch.zeros(B, self.num_slots - k, H)
            slots = torch.cat([slots, pad], dim=1)
        q     = self.q_proj(h[:, -1, :]).unsqueeze(1)            # (B,1,H)
        attn  = torch.softmax(torch.bmm(q, slots.transpose(1, 2)) / H**0.5, -1)
        ctx   = torch.bmm(attn, slots).squeeze(1)
        return self.out(ctx)


def train_and_eval(n_pairs):
    seq_len = max(24, 2 * n_pairs + 8)
    num_slots = n_pairs + 2          # slightly over-provision slots
    batch_sz  = min(BATCH, 32)

    model = SlotMemoryModel(num_slots=num_slots)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(batch_sz, n_pairs, seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()

    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, tgt = make_assoc_batch(batch_sz, n_pairs, seq_len)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


def fit_power_law(rhos, accs):
    """Fit log(acc) = -γ*log(ρ) + c. Returns (gamma, intercept, r_squared)."""
    import math
    xs = [math.log(r) for r in rhos]
    ys = [math.log(max(a, 1e-6)) for a in accs]
    n   = len(xs)
    mx  = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12:
        return 0.0, my, 0.0
    slope = sxy / sxx
    inter = my - slope * mx
    r2    = (sxy ** 2) / (sxx * syy) if syy > 1e-12 else 1.0
    return -slope, inter, r2   # gamma = -slope since acc ~ ρ^(-γ) → log(acc) = -γ*log(ρ)


class Exp331SlotPowerLaw(Experiment):
    experiment_id = "exp_33_1"
    hypothesis = ("Slot memory accuracy follows acc ~ ρ^(-γ) with R² > 0.90 across "
                  "ρ ∈ {0.031, 0.063, 0.125, 0.25, 0.5, 1.0} (N_pairs/hidden_dim).")

    def run(self) -> ExperimentResult:
        config = dict(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, steps=STEPS,
                      batch=BATCH, n_pairs_list=N_PAIRS_LIST,
                      param_bytes=sum(p.numel() * 4 for p in
                                      SlotMemoryModel(N_PAIRS_LIST[2] + 2).parameters()),
                      activation_bytes=BATCH * 24 * HIDDEN_DIM * 4)

        accs = {}; rhos = {}
        for n in N_PAIRS_LIST:
            rho = n / HIDDEN_DIM
            print(f"  n_pairs={n}, ρ={rho:.3f}...")
            acc = train_and_eval(n)
            accs[n] = round(acc, 4); rhos[n] = round(rho, 4)
            print(f"    acc={acc:.4f}")

        acc_list = [accs[n] for n in N_PAIRS_LIST]
        rho_list = [rhos[n] for n in N_PAIRS_LIST]
        gamma, intercept, r2 = fit_power_law(rho_list, acc_list)

        print(f"  Power law: acc ~ ρ^(-{gamma:.3f}), R²={r2:.3f}")

        if r2 > 0.90:
            outcome = OUTCOME_SUPPORTED
        elif r2 > 0.70:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED

        metrics = {f"acc_n{n}": accs[n] for n in N_PAIRS_LIST}
        metrics.update({f"rho_n{n}": rhos[n] for n in N_PAIRS_LIST})
        metrics["gamma_slot"] = round(gamma, 4)
        metrics["intercept"]  = round(intercept, 4)
        metrics["r_squared"]  = round(r2, 4)

        notes = (f"Slot memory γ={gamma:.3f}, R²={r2:.3f}. "
                 f"Accs: {[accs[n] for n in N_PAIRS_LIST]}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp331SlotPowerLaw().execute()
