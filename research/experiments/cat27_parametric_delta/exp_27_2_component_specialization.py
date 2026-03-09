"""
Experiment 27.2 — Component Specialization in Hybrid Memory

Hypothesis: In the hybrid model, the delta-rule component handles short-term
within-sequence interference while the parametric component handles long-range
retrieval (measurable via per-component accuracy partitioned by within-sequence
token distance): delta_short > param_short + 5% AND param_long > delta_long + 5%.
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

VOCAB_SIZE   = 64
HIDDEN_DIM   = 32
SEQ_LEN      = 32
NUM_PAIRS    = 4
STEPS        = 300
BATCH        = 8
LR           = 3e-4
LR_JOINT     = 1e-4
STEPS_PRE    = 100
STEPS_JOINT  = 200
INFERENCE_LR = 0.01
MLP_INNER    = 8

# Short-range: KV pairs within first 8 tokens; long-range: spread across full seq
SHORT_FILLER = 2   # ≤2 filler tokens between pairs
LONG_FILLER  = 10  # ≥10 filler tokens between pairs


def make_range_batch(batch_size, filler_tokens, num_pairs=NUM_PAIRS,
                     seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE):
    """Batch with a controllable gap between KV pairs (filler_tokens padding)."""
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size // 2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
                for _ in range(filler_tokens):
                    if pos < seq_len - 3:
                        seq[b, pos] = 3; pos += 1
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


def make_mixed_batch(batch_size):
    """Half short-range, half long-range."""
    half = batch_size // 2
    s_seq, s_tgt = make_range_batch(half, SHORT_FILLER)
    l_seq, l_tgt = make_range_batch(batch_size - half, LONG_FILLER)
    return (torch.cat([s_seq, l_seq], 0), torch.cat([s_tgt, l_tgt], 0),
            [True] * half + [False] * (batch_size - half))


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


# ── Delta-rule helpers ─────────────────────────────────────────────────────────

def hopfield_energy(M):
    return -0.5 * (M * M).sum(dim=(-2, -1))


def delta_update(M, k, v):
    v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
    denom  = k.pow(2).sum(-1, keepdim=True) + 1e-6
    return torch.bmm((v - v_pred / denom).unsqueeze(-1), k.unsqueeze(1))


class DeltaMemoryModule(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        B, L, H = h.shape
        M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = h[:, t, :]; v = h[:, t, :]
            dM   = delta_update(M, k, v)
            gate = (hopfield_energy(M + dM) < hopfield_energy(M)).float().view(B, 1, 1)
            M    = M + gate * dM
        return self.read_proj(torch.bmm(M, h[:, -1, :].unsqueeze(-1)).squeeze(-1))


class InnerMLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ParamMemModule(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.base = InnerMLP(hidden_dim, MLP_INNER)

    def forward(self, h):
        B, L, H = h.shape
        ctxs = []
        for b in range(B):
            mlp = InnerMLP(H, MLP_INNER)
            mlp.load_state_dict(self.base.state_dict())
            opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L - 3, 2):
                k_h = h[b, t, :].detach(); v_h = h[b, t + 1, :].detach()
                with torch.enable_grad():
                    opt.zero_grad()
                    F.mse_loss(mlp(k_h.unsqueeze(0)), v_h.unsqueeze(0)).backward()
                opt.step()
            with torch.no_grad():
                ctxs.append(mlp(h[b, -1, :].detach().unsqueeze(0)).squeeze(0))
        return torch.stack(ctxs)


class DeltaOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.mem = DeltaMemoryModule()
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        return self.out(self.mem(self.enc(seq)))


class ParamOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.mem = ParamMemModule()
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        return self.out(self.mem(self.enc(seq)))


def _param_list(model):
    if isinstance(model, ParamOnly):
        return (list(model.enc.parameters()) + list(model.mem.base.parameters()) +
                list(model.out.parameters()))
    return list(model.parameters())


def train(model, steps, lr, filler=None):
    opt = Adam(_param_list(model), lr=lr); model.train()
    for _ in range(steps):
        if filler is not None:
            seq, tgt = make_range_batch(BATCH, filler)
        else:
            # mixed
            half = BATCH // 2
            s, st = make_range_batch(half, SHORT_FILLER)
            l, lt = make_range_batch(BATCH - half, LONG_FILLER)
            seq, tgt = torch.cat([s, l]), torch.cat([st, lt])
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_by_range(model, n=20):
    """Returns (acc_short, acc_long)."""
    model.eval()
    cs = ts = cl = tl = 0
    with torch.no_grad():
        for _ in range(n):
            s_seq, s_tgt = make_range_batch(BATCH, SHORT_FILLER)
            cs += (model(s_seq).argmax(-1) == s_tgt).sum().item(); ts += s_tgt.size(0)
            l_seq, l_tgt = make_range_batch(BATCH, LONG_FILLER)
            cl += (model(l_seq).argmax(-1) == l_tgt).sum().item(); tl += l_tgt.size(0)
    return cs / ts, cl / tl


class Exp272ComponentSpecialization(Experiment):
    experiment_id = "exp_27_2"
    hypothesis = ("In the hybrid model, the delta-rule component specializes in "
                  "short-range retrieval while the parametric component specializes "
                  "in long-range retrieval: delta_short > param_short + 5% AND "
                  "param_long > delta_long + 5%.")

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, steps_pre=STEPS_PRE, steps_joint=STEPS_JOINT,
                      short_filler=SHORT_FILLER, long_filler=LONG_FILLER, batch=BATCH)

        print("  Training DeltaOnly on mixed..."); delta_m = train(DeltaOnly(), STEPS, LR)
        d_short, d_long = eval_by_range(delta_m)
        print(f"    delta: short={d_short:.3f}, long={d_long:.3f}")

        print("  Training ParamOnly on mixed..."); param_m = train(ParamOnly(), STEPS, LR)
        p_short, p_long = eval_by_range(param_m)
        print(f"    param: short={p_short:.3f}, long={p_long:.3f}")

        short_gap = d_short - p_short   # positive → delta wins short
        long_gap  = p_long  - d_long    # positive → param wins long

        if short_gap > 0.05 and long_gap > 0.05:
            outcome = OUTCOME_SUPPORTED
        elif short_gap < -0.01 and long_gap < -0.01:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        metrics = dict(
            delta_short=round(d_short, 4), delta_long=round(d_long, 4),
            param_short=round(p_short, 4), param_long=round(p_long, 4),
            short_gap=round(short_gap, 4), long_gap=round(long_gap, 4),
        )
        notes = (f"Delta: short={d_short:.3f}, long={d_long:.3f}. "
                 f"Param: short={p_short:.3f}, long={p_long:.3f}. "
                 f"short_gap={short_gap:.3f}, long_gap={long_gap:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp272ComponentSpecialization().execute()
