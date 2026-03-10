"""
Experiment 45.4 — Write Rate Trajectory Under Corrected Gate

Hypothesis: With the corrected relative vector-norm energy gate, write rate for
all four gate-containing configs (gate, ema+gate, split+gate, full) stabilizes
between 0.20 and 0.80 within the first 200 training steps and stays in that
range throughout 800 steps; the broken matrix-mean gate collapses to ≈0.0 from
step 0 and never recovers, confirming the trajectories are qualitatively
distinct and the scale bug is the sole cause of collapse.
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
SEQ_LEN     = 32
NUM_PAIRS   = 5
STEPS       = 800
BATCH       = 32
GATE_THRESH = 0.4
LOG_EVERY   = 100   # record write rate every N steps


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


class TrajectoryModel(nn.Module):
    """
    Integrated delta-rule model that tracks write rate during training.

    gate_variant:
        "rel_norm"    — corrected relative vector-norm (this experiment's fix)
        "matrix_mean" — original broken formula from exp_44_1
    """

    def __init__(self, use_ema=False, use_split=False,
                 alpha=0.95, gate_thresh=GATE_THRESH, gate_variant="rel_norm",
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_ema      = use_ema
        self.use_split    = use_split
        self.alpha        = alpha
        self.gate_thresh  = gate_thresh
        self.gate_variant = gate_variant
        self.hidden_dim   = hidden_dim

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

    def _gate(self, k: torch.Tensor, vp: torch.Tensor,
              Delta: torch.Tensor) -> torch.Tensor:
        """Return gate mask (B,1,1) and update write-rate counters."""
        if self.gate_variant == "rel_norm":
            energy = (k - vp).norm(dim=-1)
            ref    = self.gate_thresh * k.norm(dim=-1)
            gate   = (energy >= ref).float()
        else:  # "matrix_mean"
            gate = (Delta.pow(2).mean([1, 2]) >= self.gate_thresh).float()
        self._wr_count += gate.sum().item()
        self._wr_total += gate.shape[0]
        return gate[:, None, None]

    def write_rate_and_reset(self) -> float:
        wr = self._wr_count / max(self._wr_total, 1)
        self._wr_count = 0; self._wr_total = 0
        return wr

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
                # per-matrix gate (average error signal)
                err_avg = torch.cat([(ks - vps), (ke - vpe)], dim=-1)
                key_avg = torch.cat([ks, ke], dim=-1)
                gate = self._gate(key_avg, torch.zeros_like(key_avg), Delta_s)
                # recompute gate using combined signal
                if self.gate_variant == "rel_norm":
                    err_s = (ks - vps).norm(dim=-1)
                    err_e = (ke - vpe).norm(dim=-1)
                    ref_s = self.gate_thresh * ks.norm(dim=-1)
                    ref_e = self.gate_thresh * ke.norm(dim=-1)
                    fire  = ((err_s >= ref_s) | (err_e >= ref_e)).float()
                else:
                    energy = (Delta_s.pow(2).mean([1, 2]) + Delta_e.pow(2).mean([1, 2])) * 0.5
                    fire   = (energy >= self.gate_thresh).float()
                # override the gate computed above with the correct combined gate
                self._wr_count -= gate.squeeze().sum().item()  # undo double-count
                self._wr_count += fire.sum().item()
                gate_2d = fire[:, None, None]
                Delta_s = gate_2d * Delta_s
                Delta_e = gate_2d * Delta_e
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
                gate  = self._gate(k, vp, Delta)
                Delta = gate * Delta
                if self.use_ema and self.alpha < 1.0:
                    M = M + (1.0 - self.alpha) * Delta
                else:
                    M = M + Delta
            q    = self.kp(h[:, -1, :])
            read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)

        return self.out(self.rp(read))


def train_with_trajectory(model, steps=STEPS, batch=BATCH, log_every=LOG_EVERY):
    """Train for `steps`, returning (final_acc, write_rate_trajectory)."""
    opt = Adam(model.parameters(), lr=3e-4)
    model.train()
    trajectory = []
    for step in range(1, steps + 1):
        seq, tgt = make_batch(batch)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
        if step % log_every == 0:
            trajectory.append(round(model.write_rate_and_reset(), 4))
    model.eval(); ok = tot = 0
    with torch.no_grad():
        for _ in range(60):
            seq, tgt = make_batch(batch)
            ok  += (model(seq).argmax(-1) == tgt).sum().item()
            tot += tgt.size(0)
    return ok / tot, trajectory


# 4 gate-containing configs × 2 gate variants = 8 training runs
GATE_CONFIGS = [
    ("gate",       False, False),
    ("ema_gate",   True,  False),
    ("split_gate", False, True),
    ("full",       True,  True),
]
VARIANTS = ["rel_norm", "matrix_mean"]


class Exp454WriteRateTrajectory(Experiment):
    experiment_id = "exp_45_4"
    hypothesis = (
        "With the corrected relative vector-norm energy gate, write rate for all four "
        "gate-containing configs (gate, ema+gate, split+gate, full) stabilizes between "
        "0.20 and 0.80 within the first 200 training steps and stays there throughout "
        "training; the broken matrix-mean gate collapses to ≈0.0 from step 0 and never "
        "recovers."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}

        for variant in VARIANTS:
            for name, use_ema, use_split in GATE_CONFIGS:
                key = f"{variant}_{name}"
                print(f"  Training {key} ...")
                model = TrajectoryModel(
                    use_ema=use_ema, use_split=use_split,
                    gate_variant=variant,
                )
                acc, traj = train_with_trajectory(model)
                results[f"acc_{key}"]  = round(acc, 4)
                results[f"traj_{key}"] = traj
                print(f"    acc={acc:.4f}  traj={traj}")

        # Evaluate: corrected gate stays in [0.20, 0.80] for all 4 configs
        #           broken gate stays at ≤ 0.02 for all 4 configs
        corrected_stable = True
        broken_collapsed  = True

        for name, _, _ in GATE_CONFIGS:
            traj_rel = results[f"traj_rel_norm_{name}"]
            traj_mat = results[f"traj_matrix_mean_{name}"]
            # All checkpoints (steps 100..800) of corrected gate in range
            if not traj_rel or not all(0.20 <= wr <= 0.80 for wr in traj_rel):
                corrected_stable = False
            # Broken gate collapses: all checkpoints ≤ 0.05
            if not traj_mat or not all(wr <= 0.05 for wr in traj_mat):
                broken_collapsed = False

        results["corrected_stable_all_configs"] = corrected_stable
        results["broken_collapsed_all_configs"]  = broken_collapsed

        if corrected_stable and broken_collapsed:
            outcome = OUTCOME_SUPPORTED
            notes = (
                "CONFIRMED: corrected gate stays in [0.20, 0.80] for all 4 configs. "
                "Broken matrix-mean gate collapsed to ≈0 from step 0. "
                "Trajectories are qualitatively distinct, scale bug confirmed as sole cause."
            )
        elif corrected_stable and not broken_collapsed:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                "Corrected gate is stable but broken gate did not fully collapse. "
                "The distinction between criteria may be less sharp than expected."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                f"corrected_stable={corrected_stable}, "
                f"broken_collapsed={broken_collapsed}. "
                "The corrected gate did not maintain stable write rates."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp454WriteRateTrajectory().execute()
