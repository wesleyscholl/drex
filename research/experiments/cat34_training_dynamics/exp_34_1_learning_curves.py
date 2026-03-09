"""
Experiment 34.1 — Learning Curve Phase Transitions

Hypothesis: Delta rule memory exhibits a sharper learning-curve phase
transition (rapid acc gain over a short step window) than slot or parametric
memory.  We measure accuracy at 10 checkpoints and declare a phase transition
when ≥30% accuracy is gained within ≤20% of training (150 steps in a 750-step
window).

Literature basis: Phase transitions in neural networks (Saxe et al., 2013),
linear networks learn in distinct stages.  Our delta rule performs an implicit
outer-product gradient step; slot/param lack this algebraic self-consistency.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

VOCAB_SIZE  = 64
HIDDEN_DIM  = 64
NUM_PAIRS   = 4
SEQ_LEN     = 24
STEPS       = 1500
BATCH       = 8
LR          = 3e-4
EVAL_N      = 40
CHECKPOINTS = [50, 150, 300, 500, 750, 1000, 1250, 1500]


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, VOCAB_SIZE // 3, (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,)); pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN-3] = 2; seq[b, SEQ_LEN-2] = keys[qi]; seq[b, SEQ_LEN-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class DeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.rp     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape
        M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = self.k_proj(hs[:, t, :]); v = self.v_proj(hs[:, t, :])
            k_n = F.normalize(k, dim=-1)
            vp  = torch.bmm(M, k_n.unsqueeze(-1)).squeeze(-1)
            M   = M + torch.bmm((v - vp).unsqueeze(-1), k_n.unsqueeze(1))
        q = self.q_proj(hs[:, -1, :])
        return self.out(self.rp(torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)))


class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        B, L = seq.shape; H = HIDDEN_DIM
        h = self.enc(seq)
        norms = h[:, :-3, :].norm(dim=-1)
        k = min(NUM_PAIRS + 2, L - 3)
        _, idx = torch.topk(norms, k, dim=-1)
        slots = torch.gather(h[:, :-3, :], 1, idx.unsqueeze(-1).expand(-1, -1, H))
        q     = self.q_proj(h[:, -1, :]).unsqueeze(1)
        attn  = torch.softmax(torch.bmm(q, slots.transpose(1, 2)) / H**0.5, -1)
        return self.out(torch.bmm(attn, slots).squeeze(1))


def train_with_checkpoints(model_class):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    curve = {}
    step = 0
    chk_set = set(CHECKPOINTS)
    while step < STEPS:
        seq, tgt = make_batch(BATCH)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
        step += 1
        if step in chk_set:
            model.eval(); c = t = 0
            with torch.no_grad():
                for _ in range(EVAL_N):
                    s, g = make_batch(BATCH)
                    c += (model(s).argmax(-1) == g).sum().item(); t += g.size(0)
            curve[step] = round(c / t, 4)
            model.train()
    return curve


def detect_phase_transition(curve):
    """Look for 30% gain within any 150-step window."""
    steps = sorted(curve)
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            if steps[j] - steps[i] <= 200:
                gain = curve[steps[j]] - curve[steps[i]]
                if gain >= 0.30:
                    return True, steps[i], steps[j], gain
    return False, -1, -1, 0.0


class Exp341LearningCurves(Experiment):
    experiment_id = "exp_34_1"
    hypothesis = ("Delta rule memory shows a sharper phase transition "
                  "(≥30% accuracy gain within 200 steps) compared to slot memory.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, batch=BATCH, checkpoints=CHECKPOINTS,
            param_bytes=sum(p.numel() * 4 for p in DeltaModel().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Training DeltaModel with checkpoints...")
        curve_delta = train_with_checkpoints(DeltaModel)
        print(f"    delta curve: {curve_delta}")
        print("  Training SlotModel with checkpoints...")
        curve_slot = train_with_checkpoints(SlotModel)
        print(f"    slot curve: {curve_slot}")

        has_delta, d_start, d_end, d_gain = detect_phase_transition(curve_delta)
        has_slot, s_start, s_end, s_gain  = detect_phase_transition(curve_slot)

        delta_final = curve_delta.get(STEPS, 0)
        slot_final  = curve_slot.get(STEPS, 0)

        metrics = dict(
            delta_final=delta_final, slot_final=slot_final,
            delta_has_transition=int(has_delta), slot_has_transition=int(has_slot),
            delta_transition_gain=round(d_gain, 4),
            slot_transition_gain=round(s_gain, 4),
            delta_transition_window=f"{d_start}-{d_end}" if has_delta else "none",
        )
        for s, a in curve_delta.items():
            metrics[f"delta_chk{s}"] = a
        for s, a in curve_slot.items():
            metrics[f"slot_chk{s}"] = a

        if has_delta and not has_slot:
            outcome = OUTCOME_SUPPORTED
        elif not has_delta:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Delta transition: {has_delta} (gain={d_gain:.3f}, "
                 f"window {d_start}-{d_end}). Slot transition: {has_slot} "
                 f"(gain={s_gain:.3f}). Final: delta={delta_final:.3f} slot={slot_final:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp341LearningCurves().execute()
