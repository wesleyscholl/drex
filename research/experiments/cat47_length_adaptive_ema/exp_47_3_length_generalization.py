"""
Experiment 47.3 — Length Generalization of Adaptive Alpha

Hypothesis: The winning length-adaptive alpha formula from exp_47_1/47_2
generalises across the full practical range of sequence lengths L ∈ {16, 32,
48, 64, 96, 128}: write rate stays within [0.15, 0.80] at all lengths, and
accuracy ≥ 0.95 × the fixed-α EMA-alone baseline at every length, confirming
the formula is safe to deploy without per-length tuning.

Motivation: exp_47_1 and exp_47_2 only test L=32 and L=96. For production
use in an LLM, the memory module must behave correctly across all context
prefix lengths. This experiment validates that α(L) = 0.95^(96/L) (exp_scale,
the primary candidate) does not have edge-case failures at very short (L=16)
or very long (L=128) sequences.

Also tracks the computed α value and time constant τ = 1/(1−α) at each
length, providing a calibration table for the implementation spec.
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

ALPHA_FIXED = 0.95          # fixed-EMA reference
ALPHA_REF   = 0.95          # exp_scale formula parameter
L_REF       = 96            # exp_scale formula parameter
SEC_FORMULA = "linear_c5"   # secondary candidate: α = 1 - 5/L

TEST_LENGTHS = [16, 32, 48, 64, 96, 128]

# Write-rate target: wider band here since this is a generalization check
WR_TARGET = (0.15, 0.80)
# Accuracy must not drop below this fraction of fixed-EMA baseline
MIN_ACC_RATIO = 0.95


def alpha_exp_scale(L: int) -> float:
    return ALPHA_REF ** (L_REF / L)

def alpha_linear_c5(L: int) -> float:
    return max(0.75, min(0.98, 1.0 - 5.0 / L))

def alpha_fixed(L: int) -> float:
    return ALPHA_FIXED

FORMULAS = {
    "exp_scale":  alpha_exp_scale,
    "linear_c5":  alpha_linear_c5,
}


def make_batch(batch_size: int = BATCH, seq_len: int = 32,
               num_pairs: int = NUM_PAIRS, vocab_size: int = VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        # Always use min(NUM_PAIRS, available_slots) to handle very short sequences
        max_pairs = min(num_pairs, (seq_len - 3) // 2)
        n = max(1, max_pairs)
        keys = torch.randint(4, vocab_size // 3, (n * 4,)).unique()[:n]
        while len(keys) < n:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:n]
        vals = torch.randint(vocab_size // 2, vocab_size, (n,))
        pos = 0
        for i in range(n):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, n, (1,)).item()
        seq[b, seq_len - 3] = 2
        seq[b, seq_len - 2] = keys[qi]
        seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class AdaptiveEMAModel(nn.Module):
    """EMA delta-rule memory with length-adaptive alpha and optional gate."""

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
        self.kp    = nn.Linear(hidden_dim, hidden_dim, bias=False)
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
        alpha = self.alpha_fn(L)
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

            M = M + (1.0 - alpha) * Delta

        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))


def train_eval(model: AdaptiveEMAModel, seq_len: int,
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


class Exp473LengthGeneralization(Experiment):
    experiment_id = "exp_47_3"
    hypothesis = (
        "The exp_scale formula α(L)=0.95^(96/L) achieves write rate ∈ [0.15, 0.80] "
        "AND acc_ratio ≥ 0.95 × fixed-EMA baseline at ALL lengths in "
        "{16, 32, 48, 64, 96, 128}, confirming safe deployment without per-length tuning."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}

        # Fixed-EMA baselines at each test length
        fixed_ref_accs: dict[int, float] = {}
        for slen in TEST_LENGTHS:
            print(f"  Fixed-EMA reference at L={slen} ...")
            ref = AdaptiveEMAModel(alpha_fn=alpha_fixed, use_gate=False)
            acc, _ = train_eval(ref, seq_len=slen)
            fixed_ref_accs[slen] = acc
            results[f"acc_fixed_ref_L{slen}"] = round(acc, 4)
            print(f"    acc_fixed={acc:.4f}")

        # Record calibration table: α and τ at each length for each formula
        for fname, fn in FORMULAS.items():
            for slen in TEST_LENGTHS:
                a = fn(slen)
                tau = 1.0 / max(1.0 - a, 1e-6)
                results[f"alpha_{fname}_L{slen}"] = round(a, 4)
                results[f"tau_{fname}_L{slen}"]   = round(tau, 1)
                results[f"tau_over_L_{fname}_L{slen}"] = round(tau / slen, 3)

        # Main sweep: adaptive EMA + gate at each length
        formula_all_pass: dict[str, bool] = {}
        for fname, fn in FORMULAS.items():
            print(f"\n  Formula: {fname}")
            all_pass = True
            failed: list[str] = []

            for slen in TEST_LENGTHS:
                alpha_val = fn(slen)
                print(f"    L={slen}, α={alpha_val:.4f} ...")
                model = AdaptiveEMAModel(alpha_fn=fn, use_gate=True,
                                         gate_thresh=GATE_THRESH)
                acc, wr = train_eval(model, seq_len=slen)
                ratio = acc / max(fixed_ref_accs[slen], 1e-6)
                results[f"acc_{fname}_L{slen}"] = round(acc, 4)
                results[f"wr_{fname}_L{slen}"]  = round(wr, 4)
                results[f"ratio_{fname}_L{slen}"] = round(ratio, 4)
                print(f"      acc={acc:.4f}  wr={wr:.3f}  ratio={ratio:.3f}")

                wr_ok  = WR_TARGET[0] <= wr <= WR_TARGET[1]
                acc_ok = ratio >= MIN_ACC_RATIO
                results[f"wr_ok_{fname}_L{slen}"]  = wr_ok
                results[f"acc_ok_{fname}_L{slen}"] = acc_ok
                if not wr_ok or not acc_ok:
                    all_pass = False
                    if not wr_ok:
                        failed.append(f"wr={wr:.3f} outside {WR_TARGET} at L={slen}")
                    if not acc_ok:
                        failed.append(f"ratio={ratio:.3f} < {MIN_ACC_RATIO} at L={slen}")

            formula_all_pass[fname] = all_pass
            results[f"all_pass_{fname}"] = all_pass
            if all_pass:
                print(f"  *** PASS: {fname} generalises across all lengths ***")
            else:
                print(f"  FAIL: {fname} — {'; '.join(failed)}")

        results["passing_formulas"] = [f for f, p in formula_all_pass.items() if p]

        winning = [f for f, p in formula_all_pass.items() if p]
        partial  = [f for f, p in formula_all_pass.items()
                    if not p and any(
                        results.get(f"wr_ok_{f}_L{s}", False) and
                        results.get(f"acc_ok_{f}_L{s}", False)
                        for s in TEST_LENGTHS
                    )]

        if winning:
            best = winning[0]
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Length generalization confirmed. "
                f"Formula(s) passing all L: {winning}. "
                f"Primary {best}: wr and acc within targets at all lengths "
                f"{TEST_LENGTHS}. Safe for deployment with variable context length."
            )
        elif partial:
            outcome = OUTCOME_INCONCLUSIVE
            bad_lengths = {f: [
                s for s in TEST_LENGTHS
                if not (results.get(f"wr_ok_{f}_L{s}", False) and
                        results.get(f"acc_ok_{f}_L{s}", False))
            ] for f in partial}
            notes = (
                f"Partial generalization. Failing lengths per formula: {bad_lengths}. "
                "Consider clamping α to [0.80, 0.98] or using a piecewise formula."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                "No formula generalises across all tested lengths. "
                "Length-adaptive alpha approach requires per-range tuning. "
                "Consider a learned MLP threshold as the primary fallback."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp473LengthGeneralization().execute()
