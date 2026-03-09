"""
Experiment 33.3 — Gamma (Interference Exponent) Dimension Invariance

Hypothesis: The interference exponent γ is independent of hidden dimension:
fitting γ at H=32, H=64, H=128 produces values within ±0.1 for each architecture.
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

VOCAB_SIZE   = 256
STEPS        = 300
BATCH        = 32
LR           = 3e-4
EVAL_BATCHES = 40
HIDDEN_DIMS  = [32, 64, 128]

# ρ values kept constant across dims: [0.031, 0.063, 0.125, 0.25, 0.5, 1.0]
RHO_VALUES = [1/32, 2/32, 4/32, 8/32, 16/32, 32/32]


def n_pairs_for(rho, hidden_dim):
    return max(1, round(rho * hidden_dim))


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


def fit_power_law(rhos, accs):
    xs = [math.log(r) for r in rhos]
    ys = [math.log(max(a, 1e-6)) for a in accs]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12: return 0.0, my, 0.0
    slope = sxy / sxx; inter = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy) if syy > 1e-12 else 1.0
    return -slope, inter, r2


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class SlotMemoryModel(nn.Module):
    def __init__(self, num_slots, hidden_dim=64, vocab_size=VOCAB_SIZE):
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
    def __init__(self, hidden_dim=64, vocab_size=VOCAB_SIZE):
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


def train_and_eval(model, n_pairs):
    seq_len = max(24, 2 * n_pairs + 8)
    opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH, n_pairs, seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, tgt = make_assoc_batch(BATCH, n_pairs, seq_len)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp333GammaDimInvariance(Experiment):
    experiment_id = "exp_33_3"
    hypothesis = ("Interference exponent γ is independent of hidden dimension: "
                  "γ values at H=32, H=64, H=128 are within ±0.1 for each architecture.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, steps=STEPS, batch=BATCH,
            hidden_dims=HIDDEN_DIMS, rho_values=[round(r, 4) for r in RHO_VALUES],
            activation_bytes=BATCH * 24 * 128 * 4,
            param_bytes_ref=sum(p.numel() * 4 for p in SlotMemoryModel(10, 64).parameters()),
        )

        gammas = {"slot": {}, "delta": {}}
        r2s    = {"slot": {}, "delta": {}}

        for H in HIDDEN_DIMS:
            ns = [n_pairs_for(rho, H) for rho in RHO_VALUES]
            actual_rhos = [n / H for n in ns]
            print(f"  H={H}, n_pairs={ns}")
            accs_slot = []; accs_delta = []
            for n in ns:
                print(f"    n={n}...", end=" ", flush=True)
                a_s = round(train_and_eval(SlotMemoryModel(n + 2, H), n), 4)
                a_d = round(train_and_eval(DeltaModel(H), n), 4)
                accs_slot.append(a_s); accs_delta.append(a_d)
                print(f"slot={a_s:.4f} delta={a_d:.4f}")
            g_s, _, r2_s = fit_power_law(actual_rhos, accs_slot)
            g_d, _, r2_d = fit_power_law(actual_rhos, accs_delta)
            gammas["slot"][H] = round(g_s, 4); gammas["delta"][H] = round(g_d, 4)
            r2s["slot"][H]   = round(r2_s, 4); r2s["delta"][H]   = round(r2_d, 4)
            print(f"  γ_slot={g_s:.3f} R²={r2_s:.3f} | γ_delta={g_d:.3f} R²={r2_d:.3f}")

        spread_slot  = max(abs(gammas["slot"][H1] - gammas["slot"][H2])
                           for H1 in HIDDEN_DIMS for H2 in HIDDEN_DIMS)
        spread_delta = max(abs(gammas["delta"][H1] - gammas["delta"][H2])
                           for H1 in HIDDEN_DIMS for H2 in HIDDEN_DIMS)
        max_spread   = max(spread_slot, spread_delta)

        metrics = {}
        for arch in ["slot", "delta"]:
            for H in HIDDEN_DIMS:
                metrics[f"gamma_{arch}_H{H}"] = gammas[arch][H]
                metrics[f"r2_{arch}_H{H}"]    = r2s[arch][H]
        metrics["max_gamma_spread_slot"]  = round(spread_slot, 4)
        metrics["max_gamma_spread_delta"] = round(spread_delta, 4)
        metrics["max_spread_overall"]     = round(max_spread, 4)

        if max_spread < 0.10:
            outcome = OUTCOME_SUPPORTED
        elif max_spread > 0.20:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Slot γ spread={spread_slot:.3f}, Delta γ spread={spread_delta:.3f}. "
                 f"Max of both={max_spread:.3f}. "
                 f"Slot γ: H32={gammas['slot'][32]:.3f} H64={gammas['slot'][64]:.3f} "
                 f"H128={gammas['slot'][128]:.3f}. "
                 f"Delta γ: H32={gammas['delta'][32]:.3f} H64={gammas['delta'][64]:.3f} "
                 f"H128={gammas['delta'][128]:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp333GammaDimInvariance().execute()
