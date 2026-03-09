"""
Experiment 31.1 — Retroactive Re-encoding + Two-Hop Retrieval Combined

Hypothesis: Combined model (retroactive re-encoding + two-hop retrieval) outperforms
both in isolation by >5% on two-hop + re-encoding tasks, using pre-training isolation.
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

VOCAB_SIZE     = 128
HIDDEN_DIM     = 64
NUM_PAIRS      = 5
SEQ_LEN        = 26
STEPS_PRETRAIN = 200
STEPS_FINETUNE = 200
BATCH          = 16
LR             = 3e-4
FINETUNE_LR    = 1e-4
EVAL_N         = 50


def make_batch(batch_size):
    seq    = torch.full((batch_size, SEQ_LEN), 3, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, 4 + NUM_PAIRS * 4, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, 32, (NUM_PAIRS,))]).unique()[:NUM_PAIRS]
        vals = torch.randint(32, VOCAB_SIZE // 2, (NUM_PAIRS,))
        two_hop_answer = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (1,)).item()
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 4:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        if pos + 1 < SEQ_LEN - 2:
            seq[b, pos] = vals[0]; seq[b, pos + 1] = two_hop_answer; pos += 2
        if pos < SEQ_LEN - 1:
            seq[b, pos] = 2; seq[b, pos + 1] = keys[0]
        target[b] = two_hop_answer
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


def build_delta_mem(hidden, k_proj, v_proj):
    B, L, H = hidden.shape; M = torch.zeros(B, H, H)
    for t in range(L):
        k = F.normalize(k_proj(hidden[:, t, :]), dim=-1)
        v = v_proj(hidden[:, t, :])
        M = M + torch.bmm((v - torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                          k.unsqueeze(1))
    return M


class RetroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.retro_attn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        hidden = self.enc(x); B, L, H = hidden.shape
        M = build_delta_mem(hidden, self.k_proj, self.v_proj)
        half = L // 2; slots = M.transpose(1, 2)
        q = self.retro_attn(hidden[:, :half, :])
        attn = torch.softmax(torch.bmm(q, slots.transpose(1, 2)) / H**0.5, -1)
        ctx = torch.bmm(attn, slots); M2 = M.clone()
        for t in range(half):
            k = F.normalize(self.k_proj(ctx[:, t, :]), dim=-1)
            v = self.v_proj(ctx[:, t, :])
            M2 = M2 + torch.bmm((v - torch.bmm(M2, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                                 k.unsqueeze(1))
        return self.out(torch.bmm(M2, self.q_proj(hidden[:, -1, :]).unsqueeze(-1)).squeeze(-1))


class TwoHopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.hop2_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        hidden = self.enc(x)
        M = build_delta_mem(hidden, self.k_proj, self.v_proj)
        q1 = self.q_proj(hidden[:, -1, :])
        hop1 = torch.bmm(M, q1.unsqueeze(-1)).squeeze(-1)
        hop2 = torch.bmm(M, self.hop2_proj(F.relu(hop1)).unsqueeze(-1)).squeeze(-1)
        return self.out(hop2)


class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.retro_attn = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.hop2_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, x):
        hidden = self.enc(x); B, L, H = hidden.shape
        M = build_delta_mem(hidden, self.k_proj, self.v_proj)
        half = L // 2; slots = M.transpose(1, 2)
        q_r = self.retro_attn(hidden[:, :half, :])
        attn = torch.softmax(torch.bmm(q_r, slots.transpose(1, 2)) / H**0.5, -1)
        ctx = torch.bmm(attn, slots); M2 = M.clone()
        for t in range(half):
            k = F.normalize(self.k_proj(ctx[:, t, :]), dim=-1)
            v = self.v_proj(ctx[:, t, :])
            M2 = M2 + torch.bmm((v - torch.bmm(M2, k.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1),
                                 k.unsqueeze(1))
        q1 = self.q_proj(hidden[:, -1, :])
        hop1 = torch.bmm(M2, q1.unsqueeze(-1)).squeeze(-1)
        hop2 = torch.bmm(M2, self.hop2_proj(F.relu(hop1)).unsqueeze(-1)).squeeze(-1)
        return self.out(hop2)


def train_model(model_class_or_instance, steps, lr, grad_clip=None):
    model = model_class_or_instance() if isinstance(model_class_or_instance, type) else model_class_or_instance
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


class Exp311RetroTwoHop(Experiment):
    experiment_id = "exp_31_1"
    hypothesis = ("Combined model (retroactive re-encoding + two-hop retrieval) outperforms "
                  "both in isolation by >5% using pre-training isolation.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps_pretrain=STEPS_PRETRAIN, steps_finetune=STEPS_FINETUNE,
            batch=BATCH, lr=LR, finetune_lr=FINETUNE_LR,
            param_bytes=sum(p.numel()*4 for p in CombinedModel().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )
        print("  Pre-training RetroModel...")
        retro_m  = train_model(RetroModel,  STEPS_PRETRAIN, LR)
        print("  Pre-training TwoHopModel...")
        twohop_m = train_model(TwoHopModel, STEPS_PRETRAIN, LR)
        print("  Fine-tuning CombinedModel...")
        combined = CombinedModel()
        combined.enc.load_state_dict(retro_m.enc.state_dict())
        combined.k_proj.load_state_dict(retro_m.k_proj.state_dict())
        combined.v_proj.load_state_dict(retro_m.v_proj.state_dict())
        combined.q_proj.load_state_dict(retro_m.q_proj.state_dict())
        combined.retro_attn.load_state_dict(retro_m.retro_attn.state_dict())
        combined.hop2_proj.load_state_dict(twohop_m.hop2_proj.state_dict())
        combined = train_model(combined, STEPS_FINETUNE, FINETUNE_LR, grad_clip=0.5)
        acc_retro    = round(eval_model(retro_m,  EVAL_N), 4)
        acc_twohop   = round(eval_model(twohop_m, EVAL_N), 4)
        acc_combined = round(eval_model(combined, EVAL_N), 4)
        print(f"  retro={acc_retro:.3f} twohop={acc_twohop:.3f} combined={acc_combined:.3f}")
        gain_vs_retro  = round(acc_combined - acc_retro,  4)
        gain_vs_twohop = round(acc_combined - acc_twohop, 4)
        metrics = dict(acc_retro=acc_retro, acc_twohop=acc_twohop, acc_combined=acc_combined,
                       gain_vs_retro=gain_vs_retro, gain_vs_twohop=gain_vs_twohop)
        baseline = max(acc_retro, acc_twohop)
        if acc_combined > baseline + 0.05:   outcome = OUTCOME_SUPPORTED
        elif acc_combined > baseline:         outcome = OUTCOME_INCONCLUSIVE
        else:                                 outcome = OUTCOME_REFUTED
        notes = (f"Retro={acc_retro:.3f}, TwoHop={acc_twohop:.3f}, Combined={acc_combined:.3f}. "
                 f"Gain vs retro: {gain_vs_retro:+.3f}, gain vs twohop: {gain_vs_twohop:+.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp311RetroTwoHop().execute()
