"""
Experiment 34.3 — Write Gate Adaptation During Training

Hypothesis: An energy-gated delta model's write rate naturally decreases from
high (≥0.70) in early training to low (≤0.40) by training completion, showing
that the gate becomes more selective as the model learns.

Literature basis: Gating mechanisms in HTM (Hawkins & George 2006) become
selective as representations stabilise; initial high write rates correspond to
high uncertainty (surprise).
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

VOCAB_SIZE      = 64
HIDDEN_DIM      = 64
NUM_PAIRS       = 5
SEQ_LEN         = 24
STEPS           = 1500
BATCH           = 8
LR              = 3e-4
ENERGY_THRESH   = 0.4
EVAL_N          = 40
MEASURE_STEPS   = [100, 300, 600, 900, 1200, 1500]


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class EnergyGatedDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self._last_write_rate = 1.0

    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape
        M = torch.zeros(B, H, H); writes = 0; total = 0
        for t in range(L - 1):
            k = self.k_proj(hs[:, t, :]); v = self.v_proj(hs[:, t, :])
            k_n = F.normalize(k, dim=-1)
            vp  = torch.bmm(M, k_n.unsqueeze(-1)).squeeze(-1)
            error = (v - vp).norm(dim=-1)    # (B,)
            gate  = (error > ENERGY_THRESH * v.norm(dim=-1)).float()  # (B,)
            writes += gate.sum().item(); total += B
            M = M + gate.unsqueeze(-1).unsqueeze(-1) * \
                torch.bmm((v - vp).unsqueeze(-1), k_n.unsqueeze(1))
        self._last_write_rate = writes / max(total, 1)
        q = self.q_proj(hs[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


def measure_write_rate(model, n=30):
    model.eval(); total_wr = 0
    with torch.no_grad():
        for _ in range(n):
            seq, _ = make_batch(BATCH)
            model(seq)
            total_wr += model._last_write_rate
    model.train()
    return total_wr / n


class Exp343WriteGateAdaptation(Experiment):
    experiment_id = "exp_34_3"
    hypothesis = ("Energy-gated delta write rate decreases naturally during training: "
                  "≥0.70 at step 100 and ≤0.40 at step 1500.")

    def run(self) -> ExperimentResult:
        model = EnergyGatedDelta(); opt = Adam(model.parameters(), lr=LR)
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH,
            energy_threshold=ENERGY_THRESH, measure_steps=MEASURE_STEPS,
            param_bytes=sum(p.numel() * 4 for p in model.parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        model.train(); metrics = {}
        measurement_set = set(MEASURE_STEPS)
        for step in range(1, STEPS + 1):
            seq, tgt = make_batch(BATCH)
            F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
            if step in measurement_set:
                wr = measure_write_rate(model)
                metrics[f"write_rate_s{step}"] = round(wr, 4)
                print(f"    step={step}: write_rate={wr:.3f}")

        wr_early = metrics.get("write_rate_s100", 1.0)
        wr_late  = metrics.get("write_rate_s1500", 1.0)

        if wr_early >= 0.70 and wr_late <= 0.40:
            outcome = OUTCOME_SUPPORTED
        elif wr_early < 0.50:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Write rate: early(s100)={wr_early:.3f}, late(s1500)={wr_late:.3f}. "
                 f"Full trajectory: {metrics}. "
                 f"Hypothesis: ≥0.70 early → ≤0.40 late.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp343WriteGateAdaptation().execute()
