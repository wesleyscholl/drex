"""
Experiment 45.2 — Corrected Gate Re-Integration (2³ Ablation)

Hypothesis: Replacing the matrix-mean energy criterion from exp_44_1 with a
relative vector-norm criterion (‖k − Mk_n‖ ≥ thresh × ‖k‖, thresh=0.4) in
the full 2³ ablation restores acc_gate to >0.18 (out of random ~0.016) and
enables the full system (EMA + split + gate) to achieve accuracy within 5% of
acc_ema_split, eliminating the catastrophic 0.27→0.03 collapse in exp_44_1.
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
HIDDEN_DIM = 64
SEQ_LEN    = 32
NUM_PAIRS  = 5
STEPS      = 800
BATCH      = 32
GATE_THRESH = 0.4   # relative threshold: fire when ‖err‖ ≥ thresh × ‖k‖


def make_batch(batch_size=BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
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


class FixedGateModel(nn.Module):
    """
    Integrated model (EMA / split / gate, any combination).

    Gate fix: energy = ‖k − vp‖ compared to thresh × ‖k‖ (relative, scale-
    invariant).  The original exp_44_1 used Delta.pow(2).mean([1,2]) which is
    O(1/H) and always below threshold=0.4, causing write_rate→0.
    """

    def __init__(self, use_ema=False, use_split=False, use_gate=False,
                 alpha=0.95, gate_thresh=GATE_THRESH,
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_ema     = use_ema
        self.use_split   = use_split
        self.use_gate    = use_gate
        self.alpha       = alpha
        self.gate_thresh = gate_thresh
        self.hidden_dim  = hidden_dim

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
        # write-rate tracking (cumulative, reset externally)
        self._wr_count = 0
        self._wr_total = 0

    def _apply_gate(self, err: torch.Tensor, ref_key: torch.Tensor) -> torch.Tensor:
        """Return gate mask shape (B,1,1) using relative vector-norm criterion."""
        energy = err.norm(dim=-1)                          # ‖k − vp‖
        ref    = self.gate_thresh * ref_key.norm(dim=-1)   # thresh × ‖k‖
        gate   = (energy >= ref).float()
        self._wr_count += gate.sum().item()
        self._wr_total += gate.shape[0]
        return gate[:, None, None]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape

        if self.use_split:
            half = H // 2
            M_s = torch.zeros(B, half, half, device=h.device)
            M_e = torch.zeros(B, half, half, device=h.device)
            for t in range(L - 1):
                ks   = self.sem_p(h[:, t, :])
                ke   = self.epi_p(h[:, t, :])
                kns  = F.normalize(ks, dim=-1)
                kne  = F.normalize(ke, dim=-1)
                vps  = torch.bmm(M_s, kns.unsqueeze(-1)).squeeze(-1)
                vpe  = torch.bmm(M_e, kne.unsqueeze(-1)).squeeze(-1)
                d_s  = torch.bmm((ks - vps).unsqueeze(-1), kns.unsqueeze(1))
                d_e  = torch.bmm((ke - vpe).unsqueeze(-1), kne.unsqueeze(1))
                if self.use_gate:
                    # ── FIXED: relative vector-norm gate, per-matrix ──────────
                    g_s = self._apply_gate(ks - vps, ks)
                    g_e = self._apply_gate(ke - vpe, ke)
                    d_s = g_s * d_s
                    d_e = g_e * d_e
                if self.use_ema and self.alpha < 1.0:
                    M_s = M_s + (1.0 - self.alpha) * d_s
                    M_e = M_e + (1.0 - self.alpha) * d_e
                else:
                    M_s = M_s + d_s
                    M_e = M_e + d_e
            q    = h[:, -1, :]
            cs   = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
            ce   = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
            read = torch.cat([cs, ce], dim=-1)
        else:
            M = torch.zeros(B, H, H, device=h.device)
            for t in range(L - 1):
                k   = self.kp(h[:, t, :])
                kn  = F.normalize(k, dim=-1)
                vp  = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
                d   = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
                if self.use_gate:
                    # ── FIXED: relative vector-norm gate ─────────────────────
                    g  = self._apply_gate(k - vp, k)
                    d  = g * d
                if self.use_ema and self.alpha < 1.0:
                    M  = M + (1.0 - self.alpha) * d
                else:
                    M  = M + d
            q    = self.kp(h[:, -1, :])
            read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)

        return self.out(self.rp(read))

    def write_rate(self) -> float:
        wr = self._wr_count / max(self._wr_total, 1)
        self._wr_count = 0; self._wr_total = 0
        return wr


def train_eval(model: FixedGateModel, steps=STEPS, batch=BATCH,
               seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
               vocab_size=VOCAB_SIZE) -> tuple[float, float]:
    """Train the model; return (accuracy, write_rate)."""
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    model._wr_count = 0; model._wr_total = 0
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()

    final_wr = model.write_rate()

    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len, num_pairs, vocab_size)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, final_wr


# All 8 combinations of (EMA, split, gate)
CONFIGS = [
    ("baseline",   False, False, False),
    ("ema",        True,  False, False),
    ("split",      False, True,  False),
    ("gate",       False, False, True),
    ("ema_split",  True,  True,  False),
    ("ema_gate",   True,  False, True),
    ("split_gate", False, True,  True),
    ("full",       True,  True,  True),
]


class Exp452CorrectedGateIntegration(Experiment):
    experiment_id = "exp_45_2"
    hypothesis = (
        "Replacing matrix-mean energy (exp_44_1's broken formula) with relative "
        "vector-norm energy (‖k−Mk_n‖ ≥ thresh × ‖k‖, thresh=0.4) in the full "
        "2³ ablation restores acc_gate to >0.18 and enables acc_full ≥ "
        "acc_ema_split × 0.95, eliminating the catastrophic collapse of exp_44_1."
    )

    def run(self) -> ExperimentResult:
        accs: dict[str, float] = {}
        wrs:  dict[str, float] = {}

        for name, use_ema, use_split, use_gate in CONFIGS:
            print(f"Training config={name} ...")
            model = FixedGateModel(use_ema=use_ema, use_split=use_split,
                                   use_gate=use_gate, alpha=0.95,
                                   gate_thresh=GATE_THRESH)
            acc, wr = train_eval(model)
            accs[name] = round(acc, 4)
            wrs[name]  = round(wr,  4)
            print(f"  acc={acc:.4f}  write_rate={wr:.4f}")

        acc_full      = accs["full"]
        acc_ema_split = accs["ema_split"]
        acc_gate      = accs["gate"]
        ratio_full    = round(acc_full / max(acc_ema_split, 1e-6), 4)
        gap_full      = round(acc_full - acc_ema_split, 4)

        metrics = {
            **{f"acc_{k}": v for k, v in accs.items()},
            **{f"wr_{k}":  v for k, v in wrs.items()},
            "ratio_full_vs_ema_split": ratio_full,
            "gap_full_vs_ema_split":   gap_full,
        }

        gate_restored = acc_gate > 0.18
        collapse_fixed = ratio_full >= 0.95

        if gate_restored and collapse_fixed:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Collapse fixed. acc_gate={acc_gate:.4f}>0.18. "
                f"acc_full={acc_full:.4f}, acc_ema_split={acc_ema_split:.4f}, "
                f"ratio={ratio_full:.3f}≥0.95. write_rate_gate={wrs['gate']:.3f}, "
                f"write_rate_full={wrs['full']:.3f}."
            )
        elif not gate_restored:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Gate still broken: acc_gate={acc_gate:.4f}≤0.18. "
                f"write_rate_gate={wrs['gate']:.3f}. Fix insufficient."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Gate partly fixed (acc_gate={acc_gate:.4f}) but full system "
                f"still degraded: ratio={ratio_full:.3f}<0.95."
            )

        return self.result(outcome, metrics, notes)


if __name__ == "__main__":
    Exp452CorrectedGateIntegration().execute()
