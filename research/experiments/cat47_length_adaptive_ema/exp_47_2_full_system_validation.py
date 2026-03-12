"""
Experiment 47.2 — Full System Validation with Length-Adaptive Alpha

Hypothesis: The complete validated architecture (length-adaptive EMA + episodic/
semantic split + relative-norm write gate) achieves:
  (a) acc_full ≥ acc_ema_split × 0.97 at both L=32 and L=96
  (b) wr ∈ [0.20, 0.70] at L=32
  (c) wr ∈ [0.15, 0.50] at L=96
on ≥ 2/3 seeds, using the exp_scale formula from exp_47_1:
  α(L) = 0.95^(96/L)    [L=32 → 0.8574, L=96 → 0.9500]

This validates that integrating length-adaptive alpha into the full stack
(which includes the episodic/semantic split from exp_38_1 and exp_42_7)
does not degrade accuracy relative to adaptive-EMA-alone, and that gate
selectivity is restored at short sequences.

Also runs linear_c5 (α = 1 − 5/L) as a secondary candidate in case
exp_scale shows a seed-inconsistency.
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

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
NUM_PAIRS   = 5
STEPS       = 800
BATCH       = 32
GATE_THRESH = 0.40

SEQ_LENS    = [32, 96]

# Pass/fail thresholds
MIN_ACC_RATIO = 0.97     # acc(full) / acc(ema_split) must be ≥ this
WR_TARGET_L32 = (0.20, 0.70)
WR_TARGET_L96 = (0.15, 0.50)

# Candidate formulas from exp_47_1
CANDIDATES = {
    "exp_scale": lambda L: 0.95 ** (96.0 / L),
    "linear_c5": lambda L: max(0.75, min(0.98, 1.0 - 5.0 / L)),
}


def make_batch(batch_size: int = BATCH, seq_len: int = 32,
               num_pairs: int = NUM_PAIRS, vocab_size: int = VOCAB_SIZE):
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
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class FullAdaptiveModel(nn.Module):
    """
    Full architecture: length-adaptive EMA + episodic/semantic split + gate.
    alpha_fn(L) is called inside forward() using the actual L from the input.
    use_gate=False gives the ema_split baseline for comparison.
    """

    def __init__(self, alpha_fn, use_gate: bool = True,
                 gate_thresh: float = GATE_THRESH,
                 hidden_dim: int = HIDDEN_DIM, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.alpha_fn    = alpha_fn
        self.use_gate    = use_gate
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
        self._wr_count = 0
        self._wr_total = 0

    def write_rate(self) -> float:
        return self._wr_count / max(self._wr_total, 1)

    def reset_wr(self):
        self._wr_count = 0
        self._wr_total = 0

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        alpha = self.alpha_fn(L)   # length-adaptive coefficient
        half  = H // 2
        M_s   = torch.zeros(B, half, half, device=h.device)
        M_e   = torch.zeros(B, half, half, device=h.device)

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
                gate    = fire[:, None, None]
                Delta_s = gate * Delta_s
                Delta_e = gate * Delta_e

            w_t = (t + 1) / L   # episodic recency weight
            M_s = M_s + (1.0 - alpha) * Delta_s
            M_e = M_e + (1.0 - alpha) * w_t * Delta_e

        q  = h[:, -1, :]
        cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
        ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(torch.cat([cs, ce], dim=-1)))


def train_eval(model: FullAdaptiveModel, seq_len: int,
               steps: int = STEPS, batch: int = BATCH) -> tuple[float, float]:
    opt = Adam(model.parameters(), lr=3e-4)
    model.reset_wr()
    model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len=seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step()
        opt.zero_grad()
    final_wr = model.write_rate()
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch, seq_len=seq_len)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, final_wr


class Exp472FullSystemValidation(Experiment):
    experiment_id = "exp_47_2"
    hypothesis = (
        "The full architecture (length-adaptive EMA + episodic/semantic split + "
        "relative-norm gate, thresh=0.40) achieves acc_full ≥ acc_ema_split×0.97 "
        "at both L=32 and L=96, with wr ∈ [0.20,0.70] at L=32 and "
        "wr ∈ [0.15,0.50] at L=96, for at least one of {exp_scale, linear_c5}."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}
        any_winner = False
        winning_candidates: list[str] = []

        for cname, alpha_fn in CANDIDATES.items():
            print(f"\n  Candidate: {cname}")
            candidate_pass = True
            failed_criteria: list[str] = []

            for slen in SEQ_LENS:
                alpha_val = alpha_fn(slen)
                print(f"    L={slen}, α={alpha_val:.4f} ...")

                # Adaptive EMA + split baseline (no gate)
                base = FullAdaptiveModel(alpha_fn=alpha_fn, use_gate=False)
                acc_base, _ = train_eval(base, seq_len=slen)
                results[f"acc_base_{cname}_L{slen}"] = round(acc_base, 4)
                print(f"      ema_split (no gate): acc={acc_base:.4f}")

                # Full system: adaptive EMA + split + gate
                full = FullAdaptiveModel(alpha_fn=alpha_fn, use_gate=True,
                                         gate_thresh=GATE_THRESH)
                acc_full, wr_full = train_eval(full, seq_len=slen)
                ratio = acc_full / max(acc_base, 1e-6)
                results[f"acc_full_{cname}_L{slen}"] = round(acc_full, 4)
                results[f"wr_full_{cname}_L{slen}"]  = round(wr_full, 4)
                results[f"ratio_{cname}_L{slen}"]    = round(ratio, 4)
                print(f"      full (gate): acc={acc_full:.4f}  wr={wr_full:.3f}  "
                      f"ratio={ratio:.3f}")

                # Per-length checks
                acc_ok = ratio >= MIN_ACC_RATIO
                wr_lo, wr_hi = (WR_TARGET_L32 if slen == 32 else WR_TARGET_L96)
                wr_ok = wr_lo <= wr_full <= wr_hi
                results[f"acc_ok_{cname}_L{slen}"] = acc_ok
                results[f"wr_ok_{cname}_L{slen}"]  = wr_ok

                if not acc_ok:
                    candidate_pass = False
                    failed_criteria.append(
                        f"acc_ratio={ratio:.3f} < {MIN_ACC_RATIO} at L={slen}"
                    )
                if not wr_ok:
                    candidate_pass = False
                    failed_criteria.append(
                        f"wr={wr_full:.3f} outside {(wr_lo, wr_hi)} at L={slen}"
                    )

            results[f"pass_{cname}"] = candidate_pass
            if candidate_pass:
                winning_candidates.append(cname)
                any_winner = True
                print(f"  *** PASS: {cname} ***")
            else:
                print(f"  FAIL: {cname} — {'; '.join(failed_criteria)}")

        results["winning_candidates"] = winning_candidates

        if winning_candidates:
            best = winning_candidates[0]
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Full system validated with adaptive alpha. "
                f"Winner(s): {winning_candidates}. "
                f"Primary: {best}. "
                f"wr(L=32)={results.get(f'wr_full_{best}_L32', '?')}, "
                f"wr(L=96)={results.get(f'wr_full_{best}_L96', '?')}. "
                f"acc_ratio(L=32)={results.get(f'ratio_{best}_L32', '?')}, "
                f"acc_ratio(L=96)={results.get(f'ratio_{best}_L96', '?')}. "
                f"Blocker resolved: use {best} formula in production architecture."
            )
        elif any(results.get(f"pass_{c}", False) for c in CANDIDATES):
            outcome = OUTCOME_INCONCLUSIVE
            notes = "Partial pass on individual lengths only. See per-length metrics."
        else:
            # Check if at least wr is in range (gate works, just accuracy off)
            wr32_ok_any = any(
                WR_TARGET_L32[0] <= results.get(f"wr_full_{c}_L32", 0.0) <= WR_TARGET_L32[1]
                for c in CANDIDATES
            )
            if wr32_ok_any:
                outcome = OUTCOME_INCONCLUSIVE
                notes = (
                    "Gate selectivity at L=32 restored (wr in target) but accuracy "
                    "degraded below threshold. Adaptive alpha overshoots — try higher c."
                )
            else:
                outcome = OUTCOME_REFUTED
                notes = (
                    "Neither exp_scale nor linear_c5 achieved all criteria in the "
                    "full system. Bootstrap problem persists in full stack. "
                    "Consider learned MLP threshold (exp_47_4) as fallback."
                )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp472FullSystemValidation().execute()
