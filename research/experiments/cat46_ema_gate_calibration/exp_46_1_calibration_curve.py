"""
Experiment 46.1 — EMA–Gate (α, L) Calibration Curve

Hypothesis: There exists a universal threshold thresh* such that EMA+gate
with thresh* achieves write rate in [0.20, 0.70] at SEQ_LEN=32 AND write rate
in [0.15, 0.50] at SEQ_LEN=96, using the same scalar thresh* for both lengths.

Motivation: exp_45_4 showed wr≈0.96 at L=32 with thresh=0.40 under EMA
(gate inert), while exp_45_6 showed wr≈0.31 at L=96 with the same threshold
(gate healthy). The gate is already well-calibrated at long contexts; the
problem is exclusively the short-sequence bootstrap regime where vp≈0 so
‖k−vp‖/‖k‖ ≈ 1.0 >> thresh=0.40. This experiment maps the full (thresh, L)
calibration curve to determine whether a single scalar can serve both regimes.
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
NUM_PAIRS  = 5
STEPS      = 800
BATCH      = 32
ALPHA      = 0.95

THRESHOLDS = [0.40, 0.70, 0.90, 1.20, 1.50, 1.80, 2.50]
SEQ_LENS   = [32, 96]

# Target write-rate windows per length
WR_TARGET_L32 = (0.20, 0.70)   # gate should be selective at short contexts
WR_TARGET_L96 = (0.15, 0.50)   # gate should remain selective at long contexts


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


class EMAGateModel(nn.Module):
    """
    Minimal EMA+gate model for threshold calibration sweep.
    Gate: energy = ‖k − vp‖ ≥ gate_thresh × ‖k‖  (relative vector-norm, scale-invariant)
    use_gate=False → EMA-alone reference (thresh ignored).
    """

    def __init__(self, use_gate=True, gate_thresh=0.40,
                 alpha=ALPHA, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_gate    = use_gate
        self.gate_thresh = gate_thresh
        self.alpha       = alpha

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.kp    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
        self._wr_count = 0
        self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0; self._wr_total = 0

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
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

            M = M + (1.0 - self.alpha) * Delta

        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


def train_eval(model: EMAGateModel, seq_len: int,
               steps=STEPS, batch=BATCH) -> tuple[float, float]:
    """Train, return (accuracy, write_rate)."""
    opt = Adam(model.parameters(), lr=3e-4)
    model.reset_wr()
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len=seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    final_wr = model.write_rate()
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len=seq_len)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, final_wr


class Exp461CalibrationCurve(Experiment):
    experiment_id = "exp_46_1"
    hypothesis = (
        "There exists a universal threshold thresh* such that EMA+gate achieves "
        "write rate in [0.20, 0.70] at SEQ_LEN=32 AND [0.15, 0.50] at SEQ_LEN=96 "
        "using the same threshold, with accuracy ≥ EMA-alone × 0.97 at both lengths."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}

        # EMA-alone references (one per sequence length, trained fresh)
        ref_accs: dict[int, float] = {}
        for slen in SEQ_LENS:
            print(f"  EMA-alone reference at L={slen} ...")
            ref = EMAGateModel(use_gate=False)
            acc_ref, _ = train_eval(ref, seq_len=slen)
            ref_accs[slen] = acc_ref
            results[f"acc_ema_ref_L{slen}"] = round(acc_ref, 4)
            print(f"    acc_ema={acc_ref:.4f}")

        # Sweep thresholds × lengths
        wr_table: dict[tuple, float] = {}   # (thresh, L) → write_rate
        acc_table: dict[tuple, float] = {}

        for thresh in THRESHOLDS:
            for slen in SEQ_LENS:
                tag = f"thresh{thresh}_L{slen}"
                print(f"  EMA+gate thresh={thresh} L={slen} ...")
                model = EMAGateModel(use_gate=True, gate_thresh=thresh)
                acc, wr = train_eval(model, seq_len=slen)
                results[f"acc_{tag}"] = round(acc, 4)
                results[f"wr_{tag}"]  = round(wr, 4)
                wr_table[(thresh, slen)]  = wr
                acc_table[(thresh, slen)] = acc
                ratio = acc / max(ref_accs[slen], 1e-6)
                results[f"ratio_{tag}"] = round(ratio, 4)
                print(f"    acc={acc:.4f}  wr={wr:.3f}  ratio={ratio:.3f}")

        # Identify universal thresh* candidates
        universal_candidates = []
        for thresh in THRESHOLDS:
            wr32 = wr_table[(thresh, 32)]
            wr96 = wr_table[(thresh, 96)]
            wr32_ok = WR_TARGET_L32[0] <= wr32 <= WR_TARGET_L32[1]
            wr96_ok = WR_TARGET_L96[0] <= wr96 <= WR_TARGET_L96[1]
            acc32_ok = acc_table[(thresh, 32)] >= ref_accs[32] * 0.97
            acc96_ok = acc_table[(thresh, 96)] >= ref_accs[96] * 0.97
            if wr32_ok and wr96_ok and acc32_ok and acc96_ok:
                universal_candidates.append(thresh)

        results["universal_thresh_candidates"] = universal_candidates
        results["n_universal"] = len(universal_candidates)
        # Also record which thresholds work at each individual length
        ok_at_32 = [t for t in THRESHOLDS
                    if WR_TARGET_L32[0] <= wr_table[(t, 32)] <= WR_TARGET_L32[1]
                    and acc_table[(t, 32)] >= ref_accs[32] * 0.97]
        ok_at_96 = [t for t in THRESHOLDS
                    if WR_TARGET_L96[0] <= wr_table[(t, 96)] <= WR_TARGET_L96[1]
                    and acc_table[(t, 96)] >= ref_accs[96] * 0.97]
        results["thresholds_ok_at_L32"] = ok_at_32
        results["thresholds_ok_at_L96"] = ok_at_96

        if len(universal_candidates) >= 1:
            outcome = OUTCOME_SUPPORTED
            best = universal_candidates[0]
            notes = (
                f"Universal thresh* found: {universal_candidates}. "
                f"Best candidate: thresh*={best}. "
                f"wr at L=32={wr_table[(best, 32)]:.3f}, "
                f"wr at L=96={wr_table[(best, 96)]:.3f}. "
                f"Use thresh*={best} for exp_46_4."
            )
        elif len(ok_at_32) >= 1 or len(ok_at_96) >= 1:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"No single universal threshold found. "
                f"ok_at_L32={ok_at_32}, ok_at_L96={ok_at_96}. "
                "Gate calibration is length-specific; a fixed scalar cannot serve both regimes."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                "No threshold achieves healthy write rate AND accuracy ≥ 0.97 × EMA "
                "at either sequence length. EMA+gate may be fundamentally uncalibratable "
                "with a fixed scalar threshold."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp461CalibrationCurve().execute()
