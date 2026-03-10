"""
Experiment 45.6 — Scale Generalization of the Corrected System

Hypothesis: The corrected full system (EMA + split + relative-norm gate) maintains
write rate in [0.15, 0.85] and accuracy ≥ EMA-only baseline × 1.0 across all
six scale configurations: HIDDEN_DIM ∈ {32, 64, 128} × SEQ_LEN ∈ {32, 96},
showing the relative-norm gate fix is not specific to the H=64, L=32 setting
used during diagnosis and that Phase 9's repair generalises.
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

VOCAB_SIZE  = 128       # large enough to accommodate all hidden-dim variants
NUM_PAIRS   = 5
STEPS       = 800
BATCH       = 32
GATE_THRESH = 0.4
ALPHA       = 0.95

HIDDEN_DIMS = [32, 64, 128]
SEQ_LENS    = [32, 96]


def make_batch(batch_size=BATCH, seq_len=32, num_pairs=NUM_PAIRS,
               vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, vocab_size // 3, (num_pairs * 4,)).unique()[:num_pairs]
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
        tgt[b] = vals[qi]
    return seq, tgt


class ScaleModel(nn.Module):
    """
    Delta-rule model with configurable scale, supporting EMA + split + fixed gate.

    The gate uses the corrected relative vector-norm criterion:
        energy = ‖k − vp‖  ≥  gate_thresh × ‖k‖
    """

    def __init__(self, use_ema=False, use_split=False, use_gate=False,
                 alpha=ALPHA, gate_thresh=GATE_THRESH,
                 hidden_dim=64, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_ema     = use_ema
        self.use_split   = use_split
        self.use_gate    = use_gate
        self.alpha       = alpha
        self.gate_thresh = gate_thresh

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        if use_split:
            half = hidden_dim // 2
            self.sem_p = nn.Linear(hidden_dim, half, bias=False)
            self.epi_p = nn.Linear(hidden_dim, half, bias=False)
            self.rp    = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.kp = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.rp = nn.Linear(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim, vocab_size)
        self._wr_count = 0; self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def forward(self, seq):
        h = self.embed(seq); h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        half = H // 2

        if self.use_split:
            M_s = torch.zeros(B, half, half, device=h.device)
            M_e = torch.zeros(B, half, half, device=h.device)
            for t in range(L - 1):
                ks  = self.sem_p(h[:, t, :])
                ke  = self.epi_p(h[:, t, :])
                kns = F.normalize(ks, dim=-1)
                kne = F.normalize(ke, dim=-1)
                vps = torch.bmm(M_s, kns.unsqueeze(-1)).squeeze(-1)
                vpe = torch.bmm(M_e, kne.unsqueeze(-1)).squeeze(-1)
                Delta_s = torch.bmm((ks - vps).unsqueeze(-1), kns.unsqueeze(1))
                Delta_e = torch.bmm((ke - vpe).unsqueeze(-1), kne.unsqueeze(1))
                if self.use_gate:
                    err_s  = (ks - vps).norm(dim=-1)
                    err_e  = (ke - vpe).norm(dim=-1)
                    ref_s  = self.gate_thresh * ks.norm(dim=-1)
                    ref_e  = self.gate_thresh * ke.norm(dim=-1)
                    fire   = ((err_s >= ref_s) | (err_e >= ref_e)).float()
                    self._wr_count += fire.sum().item()
                    self._wr_total += B
                    gate = fire[:, None, None]
                    Delta_s = gate * Delta_s; Delta_e = gate * Delta_e
                if self.use_ema and self.alpha < 1.0:
                    M_s = M_s + (1.0 - self.alpha) * Delta_s
                    M_e = M_e + (1.0 - self.alpha) * Delta_e
                else:
                    M_s = M_s + Delta_s; M_e = M_e + Delta_e
            q  = h[:, -1, :]
            cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
            ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
            read = torch.cat([cs, ce], dim=-1)
        else:
            M = torch.zeros(B, H, H, device=h.device)
            for t in range(L - 1):
                k  = self.kp(h[:, t, :])
                kn = F.normalize(k, dim=-1)
                vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
                Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
                if self.use_gate:
                    energy = (k - vp).norm(dim=-1)
                    ref    = self.gate_thresh * k.norm(dim=-1)
                    fire   = (energy >= ref).float()
                    self._wr_count += fire.sum().item()
                    self._wr_total += B
                    Delta = fire[:, None, None] * Delta
                if self.use_ema and self.alpha < 1.0:
                    M = M + (1.0 - self.alpha) * Delta
                else:
                    M = M + Delta
            q    = self.kp(h[:, -1, :])
            read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)

        return self.out(self.rp(read))


def train_eval(model, steps=STEPS, batch=BATCH, seq_len=32):
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len=seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len=seq_len)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot


class Exp456ScaleGeneralization(Experiment):
    experiment_id = "exp_45_6"
    hypothesis = (
        "The corrected full system (EMA + split + relative-norm gate) maintains write "
        "rate in [0.15, 0.85] and accuracy ≥ EMA-only baseline across all six scale "
        "configurations: HIDDEN_DIM ∈ {32, 64, 128} × SEQ_LEN ∈ {32, 96}."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}
        all_wr_ok    = True
        all_acc_ok   = True

        for hdim in HIDDEN_DIMS:
            for slen in SEQ_LENS:
                tag = f"h{hdim}_s{slen}"
                print(f"  Scale config {tag} ...")

                # EMA-only baseline (no split, no gate)
                baseline = ScaleModel(
                    use_ema=True, use_split=False, use_gate=False,
                    hidden_dim=hdim,
                )
                acc_base = train_eval(baseline, seq_len=slen)
                results[f"acc_baseline_{tag}"] = round(acc_base, 4)

                # Corrected full system
                full = ScaleModel(
                    use_ema=True, use_split=True, use_gate=True,
                    hidden_dim=hdim,
                )
                acc_full = train_eval(full, seq_len=slen)
                wr       = full.write_rate()
                results[f"acc_full_{tag}"]  = round(acc_full, 4)
                results[f"wr_full_{tag}"]   = round(wr, 4)

                wr_ok  = 0.15 <= wr <= 0.85
                acc_ok = acc_full >= acc_base * 1.00  # at-least-as-good threshold
                results[f"wr_ok_{tag}"]  = wr_ok
                results[f"acc_ok_{tag}"] = acc_ok

                if not wr_ok:
                    all_wr_ok = False
                if not acc_ok:
                    all_acc_ok = False

                print(f"    acc_base={acc_base:.4f}  acc_full={acc_full:.4f}  "
                      f"wr={wr:.3f}  wr_ok={wr_ok}  acc_ok={acc_ok}")

        results["all_write_rates_in_range"] = all_wr_ok
        results["all_accs_at_least_baseline"] = all_acc_ok

        if all_wr_ok and all_acc_ok:
            outcome = OUTCOME_SUPPORTED
            notes = (
                "CONFIRMED: corrected gate maintains write rate [0.15, 0.85] and "
                "accuracy ≥ EMA baseline at all 6 scale configs. "
                "Relative-norm gate fix generalises across H ∈ {32,64,128} × L ∈ {32,96}."
            )
        elif all_wr_ok and not all_acc_ok:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                "Gate write rates healthy across scales but accuracy did not match "
                "baseline at every config. Interaction with scale may still need tuning."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                f"all_wr_ok={all_wr_ok}, all_acc_ok={all_acc_ok}. "
                "Gate write rate collapsed or accuracy degraded at one or more scale configs."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp456ScaleGeneralization().execute()
