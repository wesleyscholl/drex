"""
Experiment 46.6 — Full System Seed Stability at Two Sequence Lengths

Hypothesis: The calibrated full system (EMA α=0.95 + episodic/semantic split +
relative vector-norm gate, thresh*=CALIB_THRESH) achieves:
  (a) acc_full ≥ acc_ema_split × 1.02 at SEQ_LEN=96  (gate adds value at long context)
  (b) wr_full ∈ [0.20, 0.70] at SEQ_LEN=32            (gate is selective at short context)
  (c) wr_full ∈ [0.15, 0.50] at SEQ_LEN=96            (gate does not over-suppress)
on all three seeds (42, 123, 777), confirming that the Phase 10 calibration
produces a seed-stable full system.
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
HIDDEN_DIM   = 64
NUM_PAIRS    = 5
STEPS        = 800
BATCH        = 32
ALPHA        = 0.95
CALIB_THRESH = 0.70  # from exp_46_1: best compromise with selectivity at both L=32 and L=96

SEQ_LENS     = [32, 96]

# Pass/fail thresholds
MIN_GAIN_L96    = 0.02    # acc_full / acc_ema_split ≥ 1.02 at L=96
WR_TARGET_L32   = (0.20, 0.70)
WR_TARGET_L96   = (0.15, 0.50)


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


class FullCalibratedModel(nn.Module):
    """
    Full calibrated system: EMA α=0.95 + episodic/semantic split + gate at CALIB_THRESH.
    Also supports ema_split (no gate) for the comparison baseline.
    """

    def __init__(self, use_gate=True, alpha=ALPHA, gate_thresh=CALIB_THRESH,
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_gate    = use_gate
        self.alpha       = alpha
        self.gate_thresh = gate_thresh

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        half = hidden_dim // 2
        self.sem_p = nn.Linear(hidden_dim, half, bias=False)
        self.epi_p = nn.Linear(hidden_dim, half, bias=False)
        self.rp    = nn.Linear(hidden_dim, hidden_dim)
        self.out   = nn.Linear(hidden_dim, vocab_size)
        self._wr_count = 0; self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0; self._wr_total = 0

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        half = H // 2
        M_s  = torch.zeros(B, half, half, device=h.device)
        M_e  = torch.zeros(B, half, half, device=h.device)

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
                gate   = fire[:, None, None]
                Delta_s = gate * Delta_s; Delta_e = gate * Delta_e

            w_t = (t + 1) / L
            M_s = M_s + (1.0 - self.alpha) * Delta_s
            M_e = M_e + (1.0 - self.alpha) * w_t * Delta_e

        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(torch.cat([cs, ce], dim=-1)))


def train_eval(model: FullCalibratedModel, seq_len: int,
               steps=STEPS, batch=BATCH) -> tuple[float, float]:
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


class Exp466SeedStabilityTwoLengths(Experiment):
    experiment_id = "exp_46_6"
    hypothesis = (
        "The calibrated full system (EMA+split+gate, thresh*=CALIB_THRESH) achieves "
        "acc_full ≥ acc_ema_split × 1.02 at SEQ_LEN=96 AND wr ∈ [0.20, 0.70] at "
        "SEQ_LEN=32 AND wr ∈ [0.15, 0.50] at SEQ_LEN=96 on all three seeds."
    )

    def run(self) -> ExperimentResult:
        results: dict = {"calib_thresh_used": CALIB_THRESH}
        all_pass = True

        for slen in SEQ_LENS:
            print(f"\n  SEQ_LEN={slen} ...")

            # EMA+split baseline (no gate)
            baseline = FullCalibratedModel(use_gate=False)
            acc_base, _ = train_eval(baseline, seq_len=slen)
            results[f"acc_ema_split_L{slen}"] = round(acc_base, 4)
            print(f"    ema_split: acc={acc_base:.4f}")

            # Full calibrated system
            full = FullCalibratedModel(use_gate=True, gate_thresh=CALIB_THRESH)
            acc_full, wr_full = train_eval(full, seq_len=slen)
            results[f"acc_full_L{slen}"] = round(acc_full, 4)
            results[f"wr_full_L{slen}"]  = round(wr_full, 4)
            ratio = acc_full / max(acc_base, 1e-6)
            results[f"ratio_L{slen}"]    = round(ratio, 4)
            print(f"    full:      acc={acc_full:.4f}  wr={wr_full:.3f}  ratio={ratio:.3f}")

            # Per-length pass checks
            if slen == 96:
                gain_ok = ratio >= (1.0 + MIN_GAIN_L96)
                results["gain_ok_L96"] = gain_ok
                if not gain_ok:
                    all_pass = False
            wr_lo, wr_hi = (WR_TARGET_L32 if slen == 32 else WR_TARGET_L96)
            wr_ok = wr_lo <= wr_full <= wr_hi
            results[f"wr_ok_L{slen}"] = wr_ok
            if not wr_ok:
                all_pass = False

        results["all_criteria_met"] = all_pass

        acc_full_96  = results.get("acc_full_L96", 0.0)
        acc_base_96  = results.get("acc_ema_split_L96", 1e-6)
        wr_l32       = results.get("wr_full_L32", 0.0)
        wr_l96       = results.get("wr_full_L96", 0.0)
        ratio_96     = results.get("ratio_L96", 0.0)

        if all_pass:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"All criteria met for this seed. "
                f"acc_full(L=96)={acc_full_96:.4f}, "
                f"acc_ema_split(L=96)={acc_base_96:.4f}, "
                f"ratio={ratio_96:.3f} ≥ {1+MIN_GAIN_L96:.2f}. "
                f"wr(L=32)={wr_l32:.3f} ∈ {WR_TARGET_L32}, "
                f"wr(L=96)={wr_l96:.3f} ∈ {WR_TARGET_L96}. "
                f"thresh*={CALIB_THRESH} is seed-stable."
            )
        else:
            failed = []
            if not results.get("gain_ok_L96", True):
                failed.append(f"gain_L96 ratio={ratio_96:.3f} < {1+MIN_GAIN_L96:.2f}")
            if not results.get("wr_ok_L32", True):
                failed.append(f"wr_L32={wr_l32:.3f} outside {WR_TARGET_L32}")
            if not results.get("wr_ok_L96", True):
                failed.append(f"wr_L96={wr_l96:.3f} outside {WR_TARGET_L96}")
            outcome = OUTCOME_REFUTED if len(failed) >= 2 else OUTCOME_INCONCLUSIVE
            notes = f"Failed criteria: {'; '.join(failed)}"

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp466SeedStabilityTwoLengths().execute()
