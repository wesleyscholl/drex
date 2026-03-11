"""
Experiment 46.3 — Position-Conditioned Threshold Gate

Hypothesis: A position-conditioned write gate with a deterministic threshold
schedule — thresh(t) = thresh_min + (thresh_max − thresh_min) × (1 − t/L) —
achieves write rate in [0.20, 0.60] at SEQ_LEN=32 under EMA while retaining
write rate in [0.15, 0.50] at SEQ_LEN=96, without any learned parameters.

Motivation: The error gate fires almost universally at small t because
vp ≈ 0 → ‖k−vp‖/‖k‖ ≈ 1.0. The position schedule compensates: early tokens
(t≈0) face a high threshold (thresh_max) requiring a near-perfect novelty signal
to fire; late tokens converge to thresh_min=0.40 (the value that already works
at L=96). This is parameter-free and directly addresses the bootstrap problem
identified in exp_45_6 without relying on multi-stable learned scalars (exp_43_1).

Three thresh_max values are tested: {1.0, 1.5, 2.0}.
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
THRESH_MIN = 0.40   # converged value — already healthy at L=96

THRESH_MAX_VALUES = [1.0, 1.5, 2.0]
SEQ_LENS          = [32, 96]

WR_TARGET_L32 = (0.20, 0.60)
WR_TARGET_L96 = (0.15, 0.50)


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


class PosGateModel(nn.Module):
    """
    Delta-rule model with a position-conditioned threshold schedule.

    Position schedule: thresh(t, L) = thresh_min + (thresh_max − thresh_min) × (1 − t/L)
        t=0       → thresh = thresh_max  (strictest; vp is near zero, block most writes)
        t=L-1     → thresh = thresh_min  (0.40; same as the fixed threshold that works at L=96)

    use_gate=False → EMA-alone reference (schedule ignored).
    use_split → episodic/semantic split.
    """

    def __init__(self, use_ema=True, use_split=False, use_gate=False,
                 alpha=ALPHA, thresh_min=THRESH_MIN, thresh_max=1.50,
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_ema    = use_ema
        self.use_split  = use_split
        self.use_gate   = use_gate
        self.alpha      = alpha
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

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

    def _schedule(self, t: int, L: int) -> float:
        """Return position-conditioned threshold at step t of sequence length L."""
        return self.thresh_min + (self.thresh_max - self.thresh_min) * (1.0 - t / max(L - 1, 1))

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0; self._wr_total = 0

    def _gate(self, err: torch.Tensor, k: torch.Tensor,
              thresh: float) -> torch.Tensor:
        """Relative vector-norm gate with position-scheduled threshold."""
        energy = err.norm(dim=-1)
        ref    = thresh * k.norm(dim=-1)
        fire   = (energy >= ref).float()
        self._wr_count += fire.sum().item()
        self._wr_total += fire.shape[0]
        return fire[:, None, None]

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
                    thresh_t = self._schedule(t, L)
                    g_s = self._gate(ks - vps, ks, thresh_t)
                    g_e = self._gate(ke - vpe, ke, thresh_t)
                    Delta_s = g_s * Delta_s
                    Delta_e = g_e * Delta_e
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
                    thresh_t = self._schedule(t, L)
                    g  = self._gate(k - vp, k, thresh_t)
                    Delta = g * Delta
                if self.use_ema and self.alpha < 1.0:
                    M = M + (1.0 - self.alpha) * Delta
                else:
                    M = M + Delta
            q    = self.kp(h[:, -1, :])
            read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)

        return self.out(self.rp(read))


def train_eval(model: PosGateModel, seq_len: int,
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


class Exp463PositionConditionedThreshold(Experiment):
    experiment_id = "exp_46_3"
    hypothesis = (
        "A position-conditioned threshold schedule thresh(t) = thresh_min + "
        "(thresh_max − thresh_min) × (1 − t/L) achieves write rate in [0.20, 0.60] "
        "at SEQ_LEN=32 under EMA while retaining wr in [0.15, 0.50] at SEQ_LEN=96, "
        "for at least one thresh_max ∈ {1.0, 1.5, 2.0}."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}

        # EMA-alone references
        ref_accs: dict[int, float] = {}
        for slen in SEQ_LENS:
            ref = PosGateModel(use_ema=True, use_split=False, use_gate=False)
            acc_ref, _ = train_eval(ref, seq_len=slen)
            ref_accs[slen] = acc_ref
            results[f"acc_ema_ref_L{slen}"] = round(acc_ref, 4)
            print(f"  EMA ref L={slen}: acc={acc_ref:.4f}")

        any_universal = False
        best_thresh_max = None

        for thresh_max in THRESH_MAX_VALUES:
            wr_ok_both = True
            acc_ok_both = True
            for slen in SEQ_LENS:
                tag = f"ema_pos_tmax{thresh_max}_L{slen}"
                print(f"  EMA+pos_gate (thresh_max={thresh_max}) L={slen} ...")
                model = PosGateModel(
                    use_ema=True, use_split=False, use_gate=True,
                    thresh_max=thresh_max,
                )
                acc, wr = train_eval(model, seq_len=slen)
                results[f"acc_{tag}"] = round(acc, 4)
                results[f"wr_{tag}"]  = round(wr, 4)
                ratio = acc / max(ref_accs[slen], 1e-6)
                results[f"ratio_{tag}"] = round(ratio, 4)
                wt_lo, wt_hi = (WR_TARGET_L32 if slen == 32 else WR_TARGET_L96)
                wr_ok  = wt_lo <= wr <= wt_hi
                acc_ok = acc >= ref_accs[slen] * 0.97
                results[f"wr_ok_{tag}"]  = wr_ok
                results[f"acc_ok_{tag}"] = acc_ok
                if not wr_ok:
                    wr_ok_both = False
                if not acc_ok:
                    acc_ok_both = False
                print(f"    acc={acc:.4f}  wr={wr:.3f}  wr_ok={wr_ok}  acc_ok={acc_ok}")

            # Also test with split for the best-looking thresh_max configuration
            for slen in SEQ_LENS:
                tag_sp = f"ema_split_pos_tmax{thresh_max}_L{slen}"
                print(f"  EMA+split+pos_gate (thresh_max={thresh_max}) L={slen} ...")
                model_sp = PosGateModel(
                    use_ema=True, use_split=True, use_gate=True,
                    thresh_max=thresh_max,
                )
                acc_sp, wr_sp = train_eval(model_sp, seq_len=slen)
                results[f"acc_{tag_sp}"] = round(acc_sp, 4)
                results[f"wr_{tag_sp}"]  = round(wr_sp, 4)
                print(f"    acc={acc_sp:.4f}  wr={wr_sp:.3f}")

            if wr_ok_both and acc_ok_both:
                any_universal = True
                if best_thresh_max is None:
                    best_thresh_max = thresh_max

        results["any_universal_thresh_max"] = any_universal
        results["best_thresh_max"] = best_thresh_max

        if any_universal:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Position schedule works! best_thresh_max={best_thresh_max}. "
                "Deterministic schedule solves the bootstrap problem at both lengths "
                "without any learned parameters."
            )
        else:
            # Check if any thresh_max works at L=32 alone
            ok_l32 = [
                tm for tm in THRESH_MAX_VALUES
                if results.get(f"wr_ok_ema_pos_tmax{tm}_L32", False)
                and results.get(f"acc_ok_ema_pos_tmax{tm}_L32", False)
            ]
            if ok_l32:
                outcome = OUTCOME_INCONCLUSIVE
                notes = (
                    f"Position schedule reduces wr at L=32 (ok: {ok_l32}) but "
                    "fails to maintain accuracy or wr target at L=96. "
                    "Schedule is length-dependent, not universal."
                )
            else:
                outcome = OUTCOME_REFUTED
                notes = (
                    "No thresh_max in {1.0, 1.5, 2.0} achieves healthy wr AND "
                    "accuracy ≥ 0.97 ×EMA at both lengths. Position schedule "
                    "insufficient to solve bootstrap problem."
                )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp463PositionConditionedThreshold().execute()
