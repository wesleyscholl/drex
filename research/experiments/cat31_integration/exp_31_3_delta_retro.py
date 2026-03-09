"""
Experiment 31.3 — Energy-Gated Delta Rule + Retroactive Slot Re-encoding

Hypothesis: DeltaRetroModel outperforms delta-only by >8% and outperforms retro-only by >2%.
Uses pre-training isolation: each component pre-trained 200 steps before 300-step joint fine-tune.
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

VOCAB_SIZE       = 64
HIDDEN_DIM       = 64
NUM_PAIRS        = 6
SEQ_LEN          = 26
STEPS            = 500
BATCH            = 32
LR               = 3e-4
EVAL_N           = 50
ENERGY_THRESHOLD = 0.4
STEPS_PRETRAIN   = 200
STEPS_FINETUNE   = 300


def make_batch(batch_size):
    seq    = torch.full((batch_size, SEQ_LEN), 3, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 4, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (NUM_PAIRS,))]).unique()[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


class DeltaOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        h = self.enc(x); B, L, H = h.shape; M = torch.zeros(B, H, H)
        for t in range(L):
            k = F.normalize(self.k_proj(h[:, t, :]), dim=-1); v = self.v_proj(h[:, t, :])
            pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1); delta = v - pred
            gate = (delta.norm(dim=-1, keepdim=True) > ENERGY_THRESHOLD * v.norm(dim=-1, keepdim=True)).float()
            M = M + gate.unsqueeze(-1) * torch.bmm(delta.unsqueeze(-1), k.unsqueeze(1))
        return self.out(F.relu(torch.bmm(M, self.q_proj(h[:, -1, :]).unsqueeze(-1)).squeeze(-1)))


class RetroOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc        = Encoder()
        self.k_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.retro_attn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out        = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        h = self.enc(x); B, L, H = h.shape; M = torch.zeros(B, H, H)
        for t in range(L):
            k = F.normalize(self.k_proj(h[:, t, :]), dim=-1); v = self.v_proj(h[:, t, :])
            M = M + torch.bmm((v - torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                              k.unsqueeze(1))
        q = self.q_proj(h[:, -1, :]); slots = M.transpose(1, 2)
        scores = torch.bmm(slots, q.unsqueeze(-1)).squeeze(-1) / H**0.5
        attn = torch.softmax(scores, dim=-1)
        ctx  = torch.bmm(attn.unsqueeze(1), slots).squeeze(1)
        return self.out(F.relu(ctx))


class DeltaRetroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc        = Encoder()
        self.k_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj     = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.retro_attn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.alpha      = nn.Parameter(torch.tensor(0.5))
        self.out        = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        h = self.enc(x); B, L, H = h.shape; M = torch.zeros(B, H, H)
        for t in range(L):
            k = F.normalize(self.k_proj(h[:, t, :]), dim=-1); v = self.v_proj(h[:, t, :])
            pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1); delta = v - pred
            gate = (delta.norm(dim=-1, keepdim=True) > ENERGY_THRESHOLD * v.norm(dim=-1, keepdim=True)).float()
            M = M + gate.unsqueeze(-1) * torch.bmm(delta.unsqueeze(-1), k.unsqueeze(1))
        q = self.q_proj(h[:, -1, :]); slots = M.transpose(1, 2)
        k_s = min(NUM_PAIRS + 2, H)
        _, topk_i = slots.norm(dim=-1).topk(k_s, dim=-1)
        sel = torch.stack([slots[b, topk_i[b]] for b in range(B)])
        q_r  = self.retro_attn(q)
        attn = torch.softmax(torch.bmm(sel, q_r.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        retro_ctx = torch.bmm(attn.unsqueeze(1), sel).squeeze(1)
        m_ctx = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
        alpha = torch.sigmoid(self.alpha)
        return self.out(F.relu(alpha * retro_ctx + (1 - alpha) * m_ctx))


def train_model(model, steps, lr, grad_clip=None):
    opt = Adam(model.parameters(), lr=lr); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(BATCH)
        loss = F.cross_entropy(model(seq), tgt); loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); opt.zero_grad()
    return model


def eval_model(model, n):
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(n):
            seq, tgt = make_batch(BATCH)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += BATCH
    model.train(); return c / t


class Exp313DeltaRetro(Experiment):
    experiment_id = "exp_31_3"
    hypothesis = ("DeltaRetroModel outperforms delta-only by >8% and outperforms retro-only by >2%.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps=STEPS, steps_pretrain=STEPS_PRETRAIN,
            steps_finetune=STEPS_FINETUNE, batch=BATCH, energy_threshold=ENERGY_THRESHOLD,
            param_bytes=sum(p.numel()*4 for p in DeltaRetroModel().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Pre-training DeltaOnlyModel...")
        delta_m = train_model(DeltaOnlyModel(), STEPS_PRETRAIN, LR)
        print("  Pre-training RetroOnlyModel...")
        retro_m = train_model(RetroOnlyModel(), STEPS_PRETRAIN, LR)
        print("  Fine-tuning DeltaRetroModel...")
        dr = DeltaRetroModel()
        dr.enc.load_state_dict(delta_m.enc.state_dict())
        dr.k_proj.load_state_dict(delta_m.k_proj.state_dict())
        dr.v_proj.load_state_dict(delta_m.v_proj.state_dict())
        dr.q_proj.load_state_dict(delta_m.q_proj.state_dict())
        dr = train_model(dr, STEPS_FINETUNE, LR, grad_clip=0.5)
        acc_delta = round(eval_model(delta_m, EVAL_N), 4)
        acc_retro = round(eval_model(retro_m, EVAL_N), 4)
        acc_dr    = round(eval_model(dr, EVAL_N), 4)
        print(f"  delta={acc_delta:.3f} retro={acc_retro:.3f} delta_retro={acc_dr:.3f}")
        gain_vs_delta = round(acc_dr - acc_delta, 4)
        gain_vs_retro = round(acc_dr - acc_retro, 4)
        metrics = dict(acc_delta=acc_delta, acc_retro=acc_retro, acc_delta_retro=acc_dr,
                       gain_vs_delta=gain_vs_delta, gain_vs_retro=gain_vs_retro)
        if acc_dr > acc_delta + 0.08 and acc_dr > acc_retro + 0.02:
            outcome = OUTCOME_SUPPORTED
        elif acc_dr > acc_delta and acc_dr > acc_retro:
            outcome = OUTCOME_INCONCLUSIVE
        else:
            outcome = OUTCOME_REFUTED
        notes = (f"Delta={acc_delta:.3f}, Retro={acc_retro:.3f}, DeltaRetro={acc_dr:.3f}. "
                 f"Gain vs delta: {gain_vs_delta:+.3f}, gain vs retro: {gain_vs_retro:+.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp313DeltaRetro().execute()
