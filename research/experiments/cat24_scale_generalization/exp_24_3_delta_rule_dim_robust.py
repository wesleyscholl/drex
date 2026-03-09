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

experiment_id = "exp_24_3"
hypothesis = ("Energy-gated delta rule achieves the same accuracy-to-write-rate ratio "
              "at HIDDEN_DIM=128 as at HIDDEN_DIM=32 (within 5%) — the mechanism is "
              "dimension-robust.")

VOCAB_SIZE  = 64
SEQ_LEN     = 24
NUM_PAIRS   = 5
STEPS       = 400
BATCH       = 32
LR          = 3e-4
HIDDEN_DIMS = [32, 64, 128]


def make_assoc_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                     num_pairs=NUM_PAIRS):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


def hopfield_energy(M):
    return -0.5 * (M * M).sum(dim=(-2, -1))


def delta_update(M, k, v):
    v_pred  = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
    denom   = k.pow(2).sum(-1, keepdim=True) + 1e-6
    delta_v = v - v_pred / denom
    return torch.bmm(delta_v.unsqueeze(-1), k.unsqueeze(1))


class EnergyGatedDeltaModel(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.encoder    = Encoder(VOCAB_SIZE, hidden_dim)
        self.read_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.output     = nn.Linear(hidden_dim, VOCAB_SIZE)
        self.hidden_dim = hidden_dim

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        M = torch.zeros(B, H, H)
        total_writes = 0; total_tokens = 0
        for t in range(L - 1):
            k = hidden[:, t, :]; v = hidden[:, t, :]
            dM = delta_update(M, k, v)
            E_before = hopfield_energy(M); E_after = hopfield_energy(M + dM)
            gate = (E_after < E_before).float().unsqueeze(-1).unsqueeze(-1)
            M = M + gate * dM
            total_writes += gate.sum().item()
            total_tokens += B
        write_rate = total_writes / max(total_tokens, 1)
        query  = hidden[:, -1, :]
        ctx    = torch.bmm(M, query.unsqueeze(-1)).squeeze(-1)
        return self.output(self.read_proj(ctx)), write_rate


def train_and_eval(hidden_dim, n_eval=50):
    model = EnergyGatedDeltaModel(hidden_dim)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0; write_rates = []
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_assoc_batch(BATCH)
            logits, wr = model(seq)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
            write_rates.append(wr)
    acc = correct / total
    avg_wr = sum(write_rates) / len(write_rates) if write_rates else 1.0
    ratio  = acc / max(avg_wr, 1e-6)
    return acc, avg_wr, ratio


class Exp243DeltaRuleDimRobust(Experiment):
    experiment_id = "exp_24_3"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                      steps=STEPS, batch=BATCH, hidden_dims=HIDDEN_DIMS)
        results = {}
        for h_dim in HIDDEN_DIMS:
            print(f"\n--- HIDDEN_DIM={h_dim} ---")
            acc, wr, ratio = train_and_eval(h_dim)
            results[h_dim] = dict(acc=round(acc, 4), write_rate=round(wr, 4),
                                  acc_wr_ratio=round(ratio, 4))
            print(f"  acc={acc:.3f}, write_rate={wr:.3f}, ratio={ratio:.3f}")

        ratios = [results[h]['acc_wr_ratio'] for h in HIDDEN_DIMS]
        min_r  = min(ratios); max_r = max(ratios)
        spread = (max_r - min_r) / max(max_r, 1e-6)

        metrics = dict(results_by_dim=results, ratio_spread=round(spread, 4),
                       min_ratio=round(min_r, 4), max_ratio=round(max_r, 4))
        if spread < 0.05:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Ratio spread={spread:.3f}<0.05 across dims {HIDDEN_DIMS}. Dimension-robust."
        elif spread > 0.15:
            outcome = OUTCOME_REFUTED
            notes   = f"Ratio spread={spread:.3f}>0.15. Mechanism is dim-sensitive."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Ratio spread={spread:.3f}, between 0.05 and 0.15."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp243DeltaRuleDimRobust().execute()
