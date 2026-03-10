"""
Experiment 45.3 — Gate Criterion Robustness Across Hidden Dimensions

Hypothesis: The relative vector-norm criterion (‖k−vp‖ ≥ thresh × ‖k‖) keeps
write rate in [0.20, 0.80] across HIDDEN_DIM ∈ {32, 64, 128}, while the
matrix-mean criterion (exp_44_1 original) gives write_rate ≤ 0.02 at every
dimension, and the absolute-norm criterion's write_rate varies by >0.30 across
dimensions when the same absolute threshold=0.4 is used.
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
SEQ_LEN     = 32
NUM_PAIRS   = 5
STEPS       = 600
BATCH       = 32
GATE_THRESH = 0.4
DIMS        = [32, 64, 128]
CRITERIA    = ["matrix_mean", "abs_norm", "rel_norm"]


def make_batch(batch_size=BATCH, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
               vocab_size=VOCAB_SIZE, hidden_dim=64):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    vs  = max(vocab_size, hidden_dim * 2)
    for b in range(batch_size):
        keys = torch.randint(4, vs // 3, (num_pairs * 4,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vs // 3, (1,))])[:num_pairs]
        vals = torch.randint(vs // 2, vs, (num_pairs,))
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


class GateCriterionModel(nn.Module):
    """Delta-rule model with a configurable gate energy criterion."""

    def __init__(self, criterion: str, hidden_dim: int = 64,
                 vocab_size: int = VOCAB_SIZE, gate_thresh: float = GATE_THRESH):
        super().__init__()
        self.criterion   = criterion
        self.hidden_dim  = hidden_dim
        self.gate_thresh = gate_thresh
        vs = max(vocab_size, hidden_dim * 2)

        self.embed = nn.Embedding(vs, hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.kp   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rp   = nn.Linear(hidden_dim, hidden_dim)
        self.out  = nn.Linear(hidden_dim, vs)
        self._wr_count = 0; self._wr_total = 0

    def _gate(self, k: torch.Tensor, vp: torch.Tensor,
              Delta: torch.Tensor) -> torch.Tensor:
        """Return gate mask (B,1,1) according to criterion."""
        if self.criterion == "matrix_mean":
            # exp_44_1 formula — O(1/H), breaks at high H
            energy = Delta.pow(2).mean([1, 2])
            gate   = (energy >= self.gate_thresh).float()
        elif self.criterion == "abs_norm":
            # absolute vector-norm — O(‖k‖) = O(sqrt(H))
            energy = (k - vp).norm(dim=-1)
            gate   = (energy >= self.gate_thresh).float()
        else:  # "rel_norm"
            # relative vector-norm — dimensionless, scale-invariant
            energy = (k - vp).norm(dim=-1)
            ref    = self.gate_thresh * k.norm(dim=-1)
            gate   = (energy >= ref).float()
        self._wr_count += gate.sum().item()
        self._wr_total += gate.shape[0]
        return gate[:, None, None]

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape
        M = torch.zeros(B, H, H, device=h.device)
        for t in range(L - 1):
            k  = self.kp(h[:, t, :])
            kn = F.normalize(k, dim=-1)
            vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
            d  = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))
            g  = self._gate(k, vp, d)
            M  = M + g * d
        q    = self.kp(h[:, -1, :])
        read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        return self.out(self.rp(read))

    def write_rate(self) -> float:
        wr = self._wr_count / max(self._wr_total, 1)
        self._wr_count = 0; self._wr_total = 0
        return wr


def run_config(criterion: str, hidden_dim: int,
               steps: int = STEPS) -> tuple[float, float]:
    """Train a GateCriterionModel and return (accuracy, final_write_rate)."""
    vs    = max(VOCAB_SIZE, hidden_dim * 2)
    model = GateCriterionModel(criterion, hidden_dim=hidden_dim, vocab_size=vs)
    opt   = Adam(model.parameters(), lr=3e-4)
    model.train()
    model._wr_count = 0; model._wr_total = 0

    for _ in range(steps):
        seq, tgt = make_batch(BATCH, SEQ_LEN, NUM_PAIRS, vs, hidden_dim)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()

    final_wr = model.write_rate()

    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(40):
            seq, tgt = make_batch(BATCH, SEQ_LEN, NUM_PAIRS, vs, hidden_dim)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return round(ok / tot, 4), round(final_wr, 4)


class Exp453GateCriterionComparison(Experiment):
    experiment_id = "exp_45_3"
    hypothesis = (
        "The relative vector-norm criterion (‖err‖ ≥ thresh × ‖k‖) keeps write "
        "rate in [0.20, 0.80] across HIDDEN_DIM ∈ {32, 64, 128}, while the "
        "matrix-mean criterion gives write_rate ≤ 0.02 at all dims and the "
        "absolute-norm write_rate varies by >0.30 across dims."
    )

    def run(self) -> ExperimentResult:
        metrics: dict = {}
        # {criterion → {dim → (acc, wr)}}
        results: dict[str, dict[int, tuple[float, float]]] = {c: {} for c in CRITERIA}

        for crit in CRITERIA:
            for dim in DIMS:
                print(f"  criterion={crit}  H={dim} ...")
                acc, wr = run_config(crit, dim)
                results[crit][dim] = (acc, wr)
                metrics[f"acc_{crit}_H{dim}"]  = acc
                metrics[f"wr_{crit}_H{dim}"]   = wr
                print(f"    acc={acc:.4f}  wr={wr:.4f}")

        # ── Evaluate hypothesis conditions ────────────────────────────────────
        # Cond 1: rel_norm write rates all in [0.20, 0.80]
        rel_wrs    = [results["rel_norm"][d][1] for d in DIMS]
        rel_in_band = all(0.20 <= w <= 0.80 for w in rel_wrs)
        metrics["rel_norm_wrs"]    = rel_wrs
        metrics["rel_norm_in_band"] = rel_in_band

        # Cond 2: matrix_mean write rates all ≤ 0.02
        mat_wrs    = [results["matrix_mean"][d][1] for d in DIMS]
        mat_all_dead = all(w <= 0.02 for w in mat_wrs)
        metrics["matrix_mean_wrs"]    = mat_wrs
        metrics["matrix_mean_all_dead"] = mat_all_dead

        # Cond 3: abs_norm write rate spread > 0.30
        abs_wrs    = [results["abs_norm"][d][1] for d in DIMS]
        abs_spread = round(max(abs_wrs) - min(abs_wrs), 4)
        metrics["abs_norm_wrs"]    = abs_wrs
        metrics["abs_norm_spread"] = abs_spread

        if rel_in_band and mat_all_dead and abs_spread > 0.30:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"All criterion predictions confirmed. rel_norm wrs={rel_wrs} "
                f"(all in [0.2,0.8]). matrix_mean wrs={mat_wrs} (all≤0.02). "
                f"abs_norm spread={abs_spread:.3f}>0.30."
            )
        elif not mat_all_dead:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Matrix-mean gate occasionally active: wrs={mat_wrs}. "
                f"Scale mismatch hypothesis weakened."
            )
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Partial. rel_in_band={rel_in_band}, mat_dead={mat_all_dead}, "
                f"abs_spread={abs_spread:.3f}. "
                f"rel_wrs={rel_wrs}, abs_wrs={abs_wrs}."
            )

        return self.result(outcome, metrics, notes)


if __name__ == "__main__":
    Exp453GateCriterionComparison().execute()
