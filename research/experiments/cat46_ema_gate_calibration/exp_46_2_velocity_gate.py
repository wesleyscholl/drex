"""
Experiment 46.2 — Velocity Gate (EMA-Agnostic Write Signal)

Hypothesis: A velocity gate conditioned on how fast the memory prediction is
changing — gate_signal = ‖vp_t − vp_{t-1}‖ / ‖k‖ ≥ thresh_v — achieves
write rate in [0.20, 0.60] under EMA at BOTH SEQ_LEN=32 and SEQ_LEN=96,
unlike the error gate which gives wr≈0.96 at L=32 because vp≈0 throughout.

Motivation: The error gate fires when the prediction residual is large. Under
EMA, vp grows slowly (0.05× per step), so the residual stays near ‖k‖ for
the entire short sequence — causing wr→1. The velocity gate instead fires when
vp is *actively changing*, which happens at the start of every new key-value
pair presentation and stops once the memory has absorbed the association. This
should give natural selectivity even before vp has converged, because velocity
drops to zero once a KV pair is written regardless of how small vp itself is.
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
VEL_THRESH   = 0.10   # velocity gate threshold (tune via this constant)

SEQ_LENS     = [32, 96]
WR_TARGET    = (0.20, 0.60)   # target write rate window (same at both lengths)


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


class VelocityGateModel(nn.Module):
    """
    Delta-rule model with velocity-based write gate.

    Velocity gate criterion:
        vel_t = ‖vp_t − vp_{t-1}‖ / max(‖k_t‖, ε)   (normalised velocity)
        fire  = vel_t ≥ vel_thresh

    The gate fires when the memory prediction is actively changing (new information
    is being absorbed). Once a KV pair is written and vp stabilises, the gate
    closes naturally — independent of how small the absolute value of vp is.

    use_ema=False, use_split=False, use_gate=False → pure delta-rule baseline
    use_ema=True,  use_gate=True                  → EMA + velocity gate
    use_split=True                                → episodic/semantic split
    """

    def __init__(self, use_ema=False, use_split=False, use_gate=False,
                 alpha=ALPHA, vel_thresh=VEL_THRESH,
                 hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.use_ema    = use_ema
        self.use_split  = use_split
        self.use_gate   = use_gate
        self.alpha      = alpha
        self.vel_thresh = vel_thresh

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

    def _velocity_gate(self, vp: torch.Tensor, vp_prev: torch.Tensor,
                       k: torch.Tensor) -> torch.Tensor:
        """
        Return fire mask (B,) based on normalised memory-prediction velocity.

        vel = ‖vp − vp_prev‖ / max(‖k‖, ε)
        fire when vel ≥ vel_thresh.

        At step t=0, vp_prev = zeros → velocity equals ‖vp‖/‖k‖, which is
        initially small (M is zero), so the gate will NOT fire spuriously at
        step 0 (unlike the error gate which always fires at step 0).
        """
        vel  = (vp - vp_prev).norm(dim=-1) / k.norm(dim=-1).clamp(min=1e-6)
        fire = (vel >= self.vel_thresh).float()
        self._wr_count += fire.sum().item()
        self._wr_total += fire.shape[0]
        return fire

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.embed(seq)
        h = self.norm(h + self.ff(h))
        B, L, H = h.shape

        if self.use_split:
            half = H // 2
            M_s  = torch.zeros(B, half, half, device=h.device)
            M_e  = torch.zeros(B, half, half, device=h.device)
            vps_prev = torch.zeros(B, half, device=h.device)
            vpe_prev = torch.zeros(B, half, device=h.device)

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
                    # velocity gate: fires when vp is actively changing
                    w_t = (t + 1) / L   # recency weight for episodic
                    fire_s = self._velocity_gate(vps, vps_prev, ks)
                    fire_e = self._velocity_gate(vpe, vpe_prev, ke)
                    # combined gate: write if either matrix shows active change
                    gate   = torch.clamp(fire_s + fire_e, max=1.0)[:, None, None]
                    Delta_s = gate * Delta_s; Delta_e = gate * Delta_e

                if self.use_ema and self.alpha < 1.0:
                    w_t = (t + 1) / L
                    M_s = M_s + (1.0 - self.alpha) * Delta_s
                    M_e = M_e + (1.0 - self.alpha) * w_t * Delta_e
                else:
                    w_t = (t + 1) / L
                    M_s = M_s + Delta_s
                    M_e = M_e + w_t * Delta_e

                vps_prev = vps.detach()
                vpe_prev = vpe.detach()

            q  = h[:, -1, :]
            cs = torch.bmm(M_s, self.sem_p(q).unsqueeze(-1)).squeeze(-1)
            ce = torch.bmm(M_e, self.epi_p(q).unsqueeze(-1)).squeeze(-1)
            read = torch.cat([cs, ce], dim=-1)

        else:
            M       = torch.zeros(B, H, H, device=h.device)
            vp_prev = torch.zeros(B, H, device=h.device)

            for t in range(L - 1):
                k  = self.kp(h[:, t, :])
                kn = F.normalize(k, dim=-1)
                vp = torch.bmm(M, kn.unsqueeze(-1)).squeeze(-1)
                Delta = torch.bmm((k - vp).unsqueeze(-1), kn.unsqueeze(1))

                if self.use_gate:
                    fire  = self._velocity_gate(vp, vp_prev, k)
                    Delta = fire[:, None, None] * Delta

                if self.use_ema and self.alpha < 1.0:
                    M = M + (1.0 - self.alpha) * Delta
                else:
                    M = M + Delta

                vp_prev = vp.detach()

            q    = self.kp(h[:, -1, :])
            read = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)

        return self.out(self.rp(read))


def train_eval(model: VelocityGateModel, seq_len: int,
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


# Four mechanism configs tested at both sequence lengths
CONFIGS = [
    ("velocity_gate",       False, False, True),
    ("ema_velocity",        True,  False, True),
    ("split_velocity",      False, True,  True),
    ("ema_split_velocity",  True,  True,  True),
]
# EMA-alone references (one per length, no gate)
EMA_REFS = [
    ("ema_ref",             True,  False, False),
]


class Exp462VelocityGate(Experiment):
    experiment_id = "exp_46_2"
    hypothesis = (
        "A velocity gate (‖vp_t − vp_{t-1}‖ / ‖k‖ ≥ thresh_v) achieves write "
        "rate in [0.20, 0.60] under EMA at both SEQ_LEN=32 and SEQ_LEN=96, "
        "providing EMA-agnostic gate selectivity that the error gate lacks at L=32."
    )

    def run(self) -> ExperimentResult:
        results: dict = {}
        all_wr_ok  = True
        all_acc_ok = True

        # EMA-only references
        ref_accs: dict[int, float] = {}
        for slen in SEQ_LENS:
            ref = VelocityGateModel(use_ema=True, use_split=False, use_gate=False)
            acc_ref, _ = train_eval(ref, seq_len=slen)
            ref_accs[slen] = acc_ref
            results[f"acc_ema_ref_L{slen}"] = round(acc_ref, 4)
            print(f"  EMA ref L={slen}: acc={acc_ref:.4f}")

        for name, use_ema, use_split, use_gate in CONFIGS:
            for slen in SEQ_LENS:
                tag = f"{name}_L{slen}"
                print(f"  {tag} ...")
                model = VelocityGateModel(
                    use_ema=use_ema, use_split=use_split, use_gate=use_gate,
                )
                acc, wr = train_eval(model, seq_len=slen)
                results[f"acc_{tag}"] = round(acc, 4)
                results[f"wr_{tag}"]  = round(wr, 4)
                ratio = acc / max(ref_accs[slen], 1e-6)
                results[f"ratio_{tag}"] = round(ratio, 4)
                wr_ok  = WR_TARGET[0] <= wr <= WR_TARGET[1]
                acc_ok = acc >= ref_accs[slen] * 0.97
                results[f"wr_ok_{tag}"]  = wr_ok
                results[f"acc_ok_{tag}"] = acc_ok
                if not wr_ok:
                    all_wr_ok = False
                if not acc_ok:
                    all_acc_ok = False
                print(f"    acc={acc:.4f}  wr={wr:.3f}  wr_ok={wr_ok}  acc_ok={acc_ok}")

        # Focused check: does the EMA+velocity config specifically achieve the target?
        ema_vel_wr_32 = results.get("wr_ema_velocity_L32", 1.0)
        ema_vel_wr_96 = results.get("wr_ema_velocity_L96", 1.0)
        ema_vel_ok    = (WR_TARGET[0] <= ema_vel_wr_32 <= WR_TARGET[1] and
                         WR_TARGET[0] <= ema_vel_wr_96 <= WR_TARGET[1])
        results["ema_velocity_wr_both_lengths_ok"] = ema_vel_ok

        if ema_vel_ok and all_acc_ok:
            outcome = OUTCOME_SUPPORTED
            notes = (
                f"Velocity gate achieves target wr at both L=32 and L=96 under EMA. "
                f"wr(ema+vel, L=32)={ema_vel_wr_32:.3f}, "
                f"wr(ema+vel, L=96)={ema_vel_wr_96:.3f}. "
                "Gate signal is EMA-agnostic as hypothesised."
            )
        elif ema_vel_ok and not all_acc_ok:
            outcome = OUTCOME_INCONCLUSIVE
            notes = (
                f"Velocity gate achieves target wr at both lengths but accuracy "
                f"degraded in some configs. "
                f"wr(ema+vel, L=32)={ema_vel_wr_32:.3f}, "
                f"wr(ema+vel, L=96)={ema_vel_wr_96:.3f}."
            )
        else:
            outcome = OUTCOME_REFUTED
            notes = (
                f"Velocity gate FAILED to achieve wr ∈ [0.20, 0.60] at both lengths. "
                f"wr(ema+vel, L=32)={ema_vel_wr_32:.3f}, "
                f"wr(ema+vel, L=96)={ema_vel_wr_96:.3f}. "
                "Velocity signal does not provide length-agnostic selectivity."
            )

        return self.result(outcome, results, notes)


if __name__ == "__main__":
    Exp462VelocityGate().execute()
