"""
Experiment 28.5 — Vocabulary Size Scaling

Hypothesis: Delta rule capacity scales as O(hidden_dim²) regardless of vocab size
(vocab is just an indexing space). Accuracy variance across VOCAB_SIZE ∈ {32,64,128,256}
is <2% at each NUM_PAIRS level.
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

HIDDEN_DIM  = 64
SEQ_LEN     = 24
STEPS       = 400
BATCH       = 32
LR          = 3e-4
EVAL_N      = 40
VOCAB_SIZES = [32, 64, 128, 256]
N_PAIRS_LIST= [2, 4, 8]      # three difficulty levels


def make_batch(batch_size, n_pairs, vocab_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        key_ub = max(8, vocab_size // 3)
        keys = torch.randint(4, key_ub, (n_pairs * 4,)).unique()[:n_pairs]
        while len(keys) < n_pairs:
            keys = torch.cat([keys, torch.randint(4, key_ub, (1,))])[:n_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (n_pairs,)); pos = 0
        for i in range(n_pairs):
            if pos + 1 < SEQ_LEN - 3: seq[b, pos]=keys[i]; seq[b, pos+1]=vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, n_pairs, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab, h):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h*2), nn.ReLU(), nn.Linear(h*2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class DeltaModel(nn.Module):
    def __init__(self, vocab, h=HIDDEN_DIM):
        super().__init__()
        self.enc = Encoder(vocab, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, vocab)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = hs[:, t, :]; v = hs[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp/(k.pow(2).sum(-1,keepdim=True)+1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        return self.out(self.rp(torch.bmm(M, hs[:,-1:,:].transpose(1,2)).squeeze(-1)))


def train_eval(vocab, n_pairs):
    model = DeltaModel(vocab); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_batch(BATCH, n_pairs, vocab)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(BATCH, n_pairs, vocab)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp285VocabScaling(Experiment):
    experiment_id = "exp_28_5"
    hypothesis = ("Delta rule accuracy variance across VOCAB_SIZE ∈ {32,64,128,256} "
                  "is <2% at each NUM_PAIRS level, confirming vocab-independence of capacity.")

    def run(self) -> ExperimentResult:
        config = dict(
            hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN, steps=STEPS, batch=BATCH,
            vocab_sizes=VOCAB_SIZES, n_pairs_list=N_PAIRS_LIST,
            param_bytes_V32=sum(p.numel()*4 for p in DeltaModel(32).parameters()),
            param_bytes_V256=sum(p.numel()*4 for p in DeltaModel(256).parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )

        results = {}
        for n in N_PAIRS_LIST:
            results[n] = {}
            for v in VOCAB_SIZES:
                print(f"  n_pairs={n}, vocab={v}...")
                acc = round(train_eval(v, n), 4)
                results[n][v] = acc
                print(f"    acc={acc:.4f}")

        metrics = {}
        max_var_across_pairs = 0.0
        variance_ok = True
        for n in N_PAIRS_LIST:
            acc_list = [results[n][v] for v in VOCAB_SIZES]
            mx = max(acc_list); mn = min(acc_list); var = mx - mn
            metrics[f"acc_variance_n{n}"] = round(var, 4)
            metrics[f"acc_mean_n{n}"]     = round(sum(acc_list)/len(acc_list), 4)
            for v in VOCAB_SIZES:
                metrics[f"acc_n{n}_v{v}"] = results[n][v]
            if var > 0.02: variance_ok = False
            max_var_across_pairs = max(max_var_across_pairs, var)
        metrics["max_variance"] = round(max_var_across_pairs, 4)

        if variance_ok:
            outcome = OUTCOME_SUPPORTED
        elif max_var_across_pairs > 0.05:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Max variance across vocab sizes = {max_var_across_pairs:.4f} "
                 f"({'<' if variance_ok else '>'}0.02). "
                 f"Vocab-independence {'confirmed' if variance_ok else 'NOT confirmed'}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp285VocabScaling().execute()
