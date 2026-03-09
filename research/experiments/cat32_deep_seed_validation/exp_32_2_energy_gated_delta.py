"""
Experiment 32.2 — Deep Seed Validation: Energy-Gated Delta Rule (exp_15_3 replication)

Replicates exp_15_3 with 9 seeds {0,1,2,7,13,42,99,123,777}.
Criterion: acc_ratio_B > 0.90 AND write_rate_B < 0.70 (same as original).
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

VOCAB_SIZE = 64
HIDDEN_DIM = 32
SEQ_LEN    = 24
NUM_PAIRS  = 5
STEPS      = 400
BATCH      = 32
LR         = 3e-4


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size//3), (num_pairs*3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size//3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size//2, vocab_size, (num_pairs,)); pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos+1] = vals[i]; pos += 2
        for p in range(pos, seq_len-3): seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len-3] = 2; seq[b, seq_len-2] = keys[qi]; target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
                                   nn.Linear(hidden_dim*2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


def hopfield_energy(M):
    return -0.5 * (M * M).sum(dim=(-2,-1))


def delta_update(M, k, v):
    v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
    denom  = k.pow(2).sum(-1, keepdim=True) + 1e-6
    return torch.bmm((v - v_pred/denom).unsqueeze(-1), k.unsqueeze(1))


class ContinuousDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hidden = self.encoder(seq); B, L, H = hidden.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = hidden[:, t, :]; M = M + delta_update(M, k, k)
        ctx = torch.bmm(M, hidden[:,-1,:].unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx)), None


class EnergyGatedDeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.read_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hidden = self.encoder(seq); B, L, H = hidden.shape; M = torch.zeros(B, H, H)
        total_writes = total_tokens = 0
        for t in range(L - 1):
            k = hidden[:, t, :]; dM = delta_update(M, k, k)
            gate = (hopfield_energy(M + dM) < hopfield_energy(M)).float().view(B,1,1)
            M = M + gate * dM
            total_writes += gate.sum().item(); total_tokens += B
        write_rate = total_writes / max(total_tokens, 1)
        ctx = torch.bmm(M, hidden[:,-1,:].unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx)), write_rate


def train_model(model, steps, batch_size):
    opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(batch_size, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
        logits, _ = model(seq)
        F.cross_entropy(logits, tgt).backward(); opt.step(); opt.zero_grad()
    return model


def evaluate(model, n_batches=50):
    model.eval(); c = t = 0; wrs = []
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, SEQ_LEN, VOCAB_SIZE, NUM_PAIRS)
            logits, wr = model(seq)
            c += (logits.argmax(-1) == tgt).sum().item(); t += tgt.size(0)
            if wr is not None: wrs.append(wr)
    return c/t, (sum(wrs)/len(wrs) if wrs else 1.0)


class Exp322EnergyGatedDelta9Seeds(Experiment):
    experiment_id = "exp_32_2"
    hypothesis = ("Deep seed validation of energy-gated delta rule (exp_15_3): "
                  "acc_ratio > 0.90 AND write_rate < 0.70.")

    def run(self) -> ExperimentResult:
        torch.manual_seed(self.seed)
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
            num_pairs=NUM_PAIRS, steps=STEPS, batch=BATCH, seed=self.seed,
            replicates="exp_15_3",
            param_bytes=sum(p.numel()*4 for p in EnergyGatedDeltaModel().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training continuous delta model...")
        model_A = train_model(ContinuousDeltaModel(), STEPS, BATCH)
        acc_A, _ = evaluate(model_A)
        print(f"    continuous: acc={acc_A:.3f}")
        print("  Training energy-gated delta model...")
        model_B = train_model(EnergyGatedDeltaModel(), STEPS, BATCH)
        acc_B, wr_B = evaluate(model_B)
        print(f"    energy-gated: acc={acc_B:.3f} wr={wr_B:.3f}")
        acc_ratio_B = acc_B / max(acc_A, 1e-6)
        metrics = dict(
            acc_A_continuous=round(acc_A, 4), write_rate_A=1.0,
            acc_B_energy_gated=round(acc_B, 4), write_rate_B=round(wr_B, 4),
            acc_ratio_B=round(acc_ratio_B, 4), seed=self.seed,
        )
        supported = acc_ratio_B > 0.90 and wr_B < 0.70
        refuted   = wr_B > 0.85 or (acc_A - acc_B)/max(acc_A, 1e-6) > 0.10
        if supported:   outcome = OUTCOME_SUPPORTED
        elif refuted:   outcome = OUTCOME_REFUTED
        else:           outcome = OUTCOME_INCONCLUSIVE
        notes = (f"Seed={self.seed}. acc_ratio={acc_ratio_B:.3f}, write_rate={wr_B:.3f}. "
                 f"acc_A={acc_A:.3f}, acc_B={acc_B:.3f}. Required: ratio>0.90 AND wr<0.70.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp322EnergyGatedDelta9Seeds().execute()
