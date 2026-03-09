"""
Experiment 30.4 — Energy-Gated Delta Rule Pareto Frontier

Hypothesis: The accuracy–write-rate Pareto frontier of the energy-gated delta rule
has its knee at 40–60% write rate across all tested dimensions (32, 64, 128),
confirming a universal optimal operating point.
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

VOCAB_SIZE        = 64
NUM_PAIRS         = 6
SEQ_LEN           = 24
STEPS             = 800
BATCH             = 32
LR                = 3e-4
EVAL_N            = 40
ENERGY_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
HIDDEN_DIMS       = [32, 64, 128]


def make_batch(batch_size: int):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, h: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class EnergyGatedDeltaModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, energy_threshold: float = 0.4, vocab: int = VOCAB_SIZE):
        super().__init__()
        self.energy_threshold = energy_threshold
        self.enc = Encoder(vocab, hidden_dim)
        self.rp  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab)
        self.last_write_rate = 0.0
    def forward(self, seq):
        h_all = self.enc(seq); B, L, H = h_all.shape; M = torch.zeros(B, H, H)
        writes = 0; total = 0
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp    = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            error = v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)
            gate  = (error.norm(dim=-1) > self.energy_threshold * v.norm(dim=-1)).float()
            writes += gate.sum().item(); total += B
            M = M + gate.view(B, 1, 1) * torch.bmm(error.unsqueeze(-1), k.unsqueeze(1))
        self.last_write_rate = writes / max(total, 1)
        return self.out(self.rp(torch.bmm(M, h_all[:, -1, :].unsqueeze(-1)).squeeze(-1)))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


def train_eval_pareto(hidden_dim: int, energy_threshold: float):
    model = EnergyGatedDeltaModel(hidden_dim=hidden_dim, energy_threshold=energy_threshold)
    opt   = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0; total_wr = 0.0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
            total_wr += model.last_write_rate
    return c / t, total_wr / EVAL_N


def find_pareto_knee(write_rates: list[float], accs: list[float]) -> float:
    best_t = 0; best_gain = -float("inf")
    for t in range(len(accs) - 1):
        wr_diff = max(write_rates[t] - write_rates[t + 1], 0.01)
        gain    = (accs[t] - accs[t + 1]) / wr_diff
        if gain > best_gain: best_gain = gain; best_t = t
    return write_rates[best_t]


class Exp304ParetoSweep(Experiment):
    experiment_id = "exp_30_4"
    hypothesis = ("Energy-gated delta rule Pareto knee lies at 40-60% write rate "
                  "for all tested hidden dimensions (32, 64, 128).")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS, seq_len=SEQ_LEN,
            steps=STEPS, batch=BATCH, energy_thresholds=ENERGY_THRESHOLDS,
            hidden_dims=HIDDEN_DIMS,
            param_bytes_h64=EnergyGatedDeltaModel(hidden_dim=64).param_bytes(),
            activation_bytes=BATCH * SEQ_LEN * max(HIDDEN_DIMS) * 4,
        )

        metrics: dict = {}; knee_wrs: dict[int, float] = {}
        for H in HIDDEN_DIMS:
            accs: list[float] = []; wrs: list[float] = []
            for i, th in enumerate(ENERGY_THRESHOLDS):
                print(f"  H={H}  threshold={th:.1f}...")
                acc, wr = train_eval_pareto(H, th)
                accs.append(acc); wrs.append(wr)
                metrics[f"pareto_acc_H{H}_th{i}"] = round(acc, 4)
                metrics[f"pareto_wr_H{H}_th{i}"]  = round(wr, 4)
                print(f"    acc={acc:.3f}  wr={wr:.3f}")
            knee = find_pareto_knee(wrs, accs)
            metrics[f"knee_wr_H{H}"] = round(knee, 4); knee_wrs[H] = knee
            print(f"  H={H} knee write-rate={knee:.3f}")

        all_in_range = all(0.40 <= knee_wrs[H] <= 0.60 for H in HIDDEN_DIMS)
        metrics["all_knees_in_range"] = int(all_in_range)

        if all_in_range:                                            outcome = OUTCOME_SUPPORTED
        elif any(0.40 <= knee_wrs[H] <= 0.60 for H in HIDDEN_DIMS): outcome = OUTCOME_INCONCLUSIVE
        else:                                                        outcome = OUTCOME_REFUTED

        notes = (f"Knee write rates: H32={knee_wrs[32]:.3f}, H64={knee_wrs[64]:.3f}, "
                 f"H128={knee_wrs[128]:.3f}. All in [0.40,0.60]: {all_in_range}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp304ParetoSweep().execute()
