"""
Experiment 46.4 — Full 2³ Ablation at Calibrated Threshold

Hypothesis: Using the calibrated threshold thresh* (identified in exp_46_1) in
the full 2³ ablation (all 8 combinations of EMA × split × gate) achieves
acc(EMA+gate) > acc(EMA-alone) + 0.005 at SEQ_LEN=32, confirming that a
calibrated gate adds genuine value beyond EMA alone. Additionally, write rate
at SEQ_LEN=96 must stay in [0.15, 0.50] (no over-suppression at long context).

CALIB_THRESH: update this constant from the best universal candidate reported
by exp_46_1 before running. Default is 1.20, a conservative initial estimate
based on the write-rate distribution under EMA at L=32. If exp_46_1 finds no
universal thresh*, use the best L=32-specific value and annotate the result.
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

# Updated from exp_46_1 seed 42: thresh=0.70 is the only value with selectivity
# at BOTH L=32 (wr=0.61) and L=96 (wr=0.20) while ratio_L32=1.022 ≥ 0.97.
# No universal thresh* with ratio_L96 ≥ 0.97 exists; 0.70 is the best compromise.
CALIB_THRESH = 0.70

SEQ_LENS     = [32, 96]

# Minimum gate advantage required to call the gate useful
MIN_ACC_GAIN     = 0.005    # acc(ema+gate) − acc(ema) > 0.005 at L=32
WR_RANGE_L96     = (0.15, 0.50)  # gate must not over-suppress at long context


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


class CalibratedGateModel(nn.Module):
    """
    Full 2³ ablation model using calibrated threshold CALIB_THRESH.

    write gate: ‖k − vp‖ ≥ gate_thresh × ‖k‖  (relative vector-norm, scale-invariant)
    """

    def __init__(self, use_ema=False, use_split=False, use_gate=False,
                 alpha=ALPHA, gate_thresh=CALIB_THRESH,
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
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

        if self.use_split:
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
                if self.use_ema and self.alpha < 1.0:
                    w_t = (t + 1) / L
                    M_s = M_s + (1.0 - self.alpha) * Delta_s
                    M_e = M_e + (1.0 - self.alpha) * w_t * Delta_e
                else:
                    w_t = (t + 1) / L
                    M_s = M_s + Delta_s
                    M_e = M_e + w_t * Delta_e
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


def train_eval(model: CalibratedGateModel, seq_len: int,
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


class Exp464FullAblationCalibrated(Experiment):
    experiment_id = "exp_46_4"
    hypothesis = (
        "Using calibrated threshold thresh* in the full 2³ ablation: "
        "acc(ema_gate) > acc(ema) + 0.005 at SEQ_LEN=32, confirming the gate "
        "adds genuine value beyond EMA alone; wr(ema_gate, L=96) ∈ [0.15, 0.50], "
        "confirming no over-suppression at long context."
    )

    def run(self) -> ExperimentResult:
        results: dict = {"calib_thresh_used": CALIB_THRESH}
        accs: dict[str, dict[int, float]] = {name: {} for name, *_ in CONFIGS}
        wrs:  dict[str, dict[int, float]] = {name: {} for name, *_ in CONFIGS}

        for slen in SEQ_LENS:
            print(f"\n  SEQ_LEN={slen} ...")
            for name, use_ema, use_split, use_gate in CONFIGS:
                model = CalibratedGateModel(
                    use_ema=use_ema, use_split=use_split, use_gate=use_gate,
                    gate_thresh=CALIB_THRESH,
                )
                acc, wr = train_eval(model, seq_len=slen)
                accs[name][slen] = acc
                wrs[name][slen]  = wr
                results[f"acc_{name}_L{slen}"] = round(acc, 4)
                results[f"wr_{name}_L{slen}"]  = round(wr, 4)
                print(f"    {name:12s}  acc={acc:.4f}  wr={wr:.3f}")

        # Primary hypothesis checks
        gain_ema_gate_L32 = accs["ema_gate"][32] - accs["ema"][32]
        wr_ema_gate_L96   = wrs["ema_gate"][96]
        wr_full_L32       = wrs["full"][32]
        wr_full_L96       = wrs["full"][96]

        results["gain_ema_gate_L32"]  = round(gain_ema_gate_L32, 4)
        results["wr_ema_gate_L96"]    = round(wr_ema_gate_L96, 4)
        results["wr_full_L32"]        = round(wr_full_L32, 4)
        results["wr_full_L96"]        = round(wr_full_L96, 4)

        gate_adds_value   = gain_ema_gate_L32 > MIN_ACC_GAIN
        no_oversuppress   = WR_RANGE_L96[0] <= wr_ema_gate_L96 <= WR_RANGE_L96[1]

        results["gate_adds_value_L32"] = gate_adds_value
        results["wr_ok_L96"]           = no_oversuppress

        # Secondary: does the full system improve on ema_split?
        gain_full_L32 = accs["full"][32] - accs["ema_split"][32]
        gain_full_L96 = accs["full"][96] - accs["ema_split"][96]
        results["gain_full_vs_ema_split_L32"] = round(gain_full_L32, 4)
        results["gain_full_vs_ema_split_L96"] = round(gain_full_L96, 4)

        if gate_adds_value and no_oversuppress:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Calibrated gate (thresh*={CALIB_THRESH}) adds genuine value. "
                f"gain(ema_gate vs ema, L=32)={gain_ema_gate_L32:+.4f} > {MIN_ACC_GAIN}. "
                f"wr(ema_gate, L=96)={wr_ema_gate_L96:.3f} ∈ {WR_RANGE_L96}. "
                f"gain_full_L32={gain_full_L32:+.4f}, gain_full_L96={gain_full_L96:+.4f}."
            )
        elif gate_adds_value and not no_oversuppress:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Gate adds value at L=32 (gain={gain_ema_gate_L32:+.4f}) but "
                f"wr at L=96={wr_ema_gate_L96:.3f} outside {WR_RANGE_L96}. "
                f"Threshold over-suppresses at long context."
            )
        elif not gate_adds_value and no_oversuppress:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"wr at L=96 is healthy ({wr_ema_gate_L96:.3f}) but gate does not "
                f"improve accuracy: gain={gain_ema_gate_L32:+.4f} ≤ {MIN_ACC_GAIN}. "
                f"thresh*={CALIB_THRESH} may still be too high or task is too easy."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Calibrated gate (thresh*={CALIB_THRESH}) provides no accuracy gain "
                f"(gain={gain_ema_gate_L32:+.4f}) AND wr at L=96={wr_ema_gate_L96:.3f} "
                f"is outside target {WR_RANGE_L96}."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp464FullAblationCalibrated().execute()
