"""
Experiment 30.2 — Momentum-Accelerated Delta Rule

Hypothesis: M_t = β×M_{t-1} + (1−β)×ΔM achieves the same accuracy-to-write-rate ratio
as energy gating, but with more stable training (lower loss variance over the final 100 steps).
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
NUM_PAIRS        = 6
SEQ_LEN          = 24
STEPS            = 1500
BATCH            = 32
LR               = 3e-4
EVAL_N           = 50
BETA             = 0.9
ENERGY_THRESHOLD = 0.4


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
    def __init__(self, vocab: int = VOCAB_SIZE, h: int = HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class EnergyGatedDelta(nn.Module):
    def __init__(self, h: int = HIDDEN_DIM, v: int = VOCAB_SIZE, threshold: float = ENERGY_THRESHOLD):
        super().__init__()
        self.threshold = threshold
        self.enc = Encoder(v, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, v)
        self.last_write_rate = 0.0
    def forward(self, seq):
        h_all = self.enc(seq); B, L, H = h_all.shape; M = torch.zeros(B, H, H)
        writes = 0; total = 0
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp    = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            error = v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)
            gate  = (error.norm(dim=-1) > self.threshold * v.norm(dim=-1)).float()
            writes += gate.sum().item(); total += B
            M = M + gate.view(B, 1, 1) * torch.bmm(error.unsqueeze(-1), k.unsqueeze(1))
        self.last_write_rate = writes / max(total, 1)
        return self.out(self.rp(torch.bmm(M, h_all[:, -1, :].unsqueeze(-1)).squeeze(-1)))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


class MomentumDelta(nn.Module):
    def __init__(self, h: int = HIDDEN_DIM, v: int = VOCAB_SIZE, beta: float = BETA):
        super().__init__()
        self.beta = beta
        self.enc = Encoder(v, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, v)
    def forward(self, seq):
        h_all = self.enc(seq); B, L, H = h_all.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp    = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            delta = torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                              k.unsqueeze(1))
            M = self.beta * M + (1 - self.beta) * delta
        return self.out(self.rp(torch.bmm(M, h_all[:, -1, :].unsqueeze(-1)).squeeze(-1)))
    def param_bytes(self): return sum(p.numel() * 4 for p in self.parameters())


def train_eval_with_var(model_class):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    recent_losses: list[float] = []
    for step in range(STEPS):
        seq, tgt = make_batch(BATCH)
        loss = F.cross_entropy(model(seq), tgt)
        loss.backward(); opt.step(); opt.zero_grad()
        if step >= STEPS - 100: recent_losses.append(loss.item())
    loss_var = float(torch.tensor(recent_losses).var().item()) if len(recent_losses) > 1 else 0.0
    model.eval(); c = t = 0; total_wr = 0.0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH)
            logits = model(seq)
            c += (logits.argmax(-1) == tgt).sum().item(); t += tgt.size(0)
            if hasattr(model, "last_write_rate"): total_wr += model.last_write_rate
    acc = c / t
    wr  = (total_wr / EVAL_N) if hasattr(model, "last_write_rate") else 1.0
    return acc, loss_var, wr


class Exp302MomentumDelta(Experiment):
    experiment_id = "exp_30_2"
    hypothesis = ("Momentum delta (β=0.9) matches energy-gated accuracy within 5% "
                  "while achieving lower final-100-step loss variance (ratio < 0.8).")

    def run(self) -> ExperimentResult:
        eg = EnergyGatedDelta(); md = MomentumDelta()
        config = dict(
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, beta=BETA,
            energy_threshold=ENERGY_THRESHOLD,
            param_bytes_energy=eg.param_bytes(), param_bytes_momentum=md.param_bytes(),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training EnergyGatedDelta...")
        acc_energy, lv_energy, wr_energy = train_eval_with_var(EnergyGatedDelta)
        print(f"    energy: acc={acc_energy:.3f} loss_var={lv_energy:.5f} wr={wr_energy:.3f}")
        print("  Training MomentumDelta...")
        acc_mom, lv_mom, wr_mom = train_eval_with_var(MomentumDelta)
        print(f"    momentum: acc={acc_mom:.3f} loss_var={lv_mom:.5f} wr={wr_mom:.3f}")

        gap_acc   = acc_mom - acc_energy
        var_ratio = lv_mom / max(lv_energy, 1e-9)
        metrics   = dict(
            acc_energy=round(acc_energy, 4), acc_momentum=round(acc_mom, 4),
            write_rate_energy=round(wr_energy, 4), write_rate_momentum=round(wr_mom, 4),
            loss_var_energy=round(lv_energy, 6), loss_var_momentum=round(lv_mom, 6),
            gap_acc=round(gap_acc, 4), var_ratio=round(var_ratio, 4),
        )

        if abs(gap_acc) < 0.05 and lv_mom < lv_energy * 0.8:
            outcome = OUTCOME_SUPPORTED
        elif abs(gap_acc) >= 0.10 or lv_mom >= lv_energy:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Momentum acc={acc_mom:.3f}, Energy acc={acc_energy:.3f}, gap={gap_acc:.3f}. "
                 f"Var ratio={var_ratio:.3f}. Write rates: energy={wr_energy:.3f}, "
                 f"momentum={wr_mom:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp302MomentumDelta().execute()
