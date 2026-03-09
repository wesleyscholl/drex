"""
Experiment 27.1 — Hybrid Memory (Delta-Rule + Parametric MLP)

Hypothesis: Hybrid memory (delta-rule matrix + parametric MLP) outperforms either alone
when each component is pre-trained independently for 100 steps before joint fine-tuning.
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

VOCAB_SIZE     = 64
HIDDEN_DIM     = 32
SEQ_LEN        = 24
NUM_PAIRS      = 5
STEPS_PRE      = 100
STEPS_JOINT    = 200
BATCH          = 8
LR             = 3e-4
LR_JOINT       = 1e-4
INFERENCE_LR   = 0.01
MLP_INNER      = 8


def make_assoc_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                     num_pairs=NUM_PAIRS):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size//3), (num_pairs*3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size//3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size//2, vocab_size, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos+1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3): seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len-3] = 2; seq[b, seq_len-2] = keys[qi]; seq[b, seq_len-1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
                                   nn.Linear(hidden_dim*2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


# ── Delta-rule helpers ────────────────────────────────────────────────────────

def hopfield_energy(M):
    return -0.5 * (M*M).sum(dim=(-2,-1))


def delta_update(M, k, v):
    v_pred = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
    denom  = k.pow(2).sum(-1, keepdim=True) + 1e-6
    return torch.bmm((v - v_pred/denom).unsqueeze(-1), k.unsqueeze(1))


class DeltaMemoryModule(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.read_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        B, L, H = h.shape
        M = torch.zeros(B, H, H)
        for t in range(L-1):
            k = h[:, t, :]; v = h[:, t, :]
            dM = delta_update(M, k, v)
            gate = (hopfield_energy(M+dM) < hopfield_energy(M)).float().view(B,1,1)
            M = M + gate*dM
        return self.read_proj(torch.bmm(M, h[:,-1,:].unsqueeze(-1)).squeeze(-1))


class InnerMLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ParamMemModule(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.base = InnerMLP(hidden_dim, MLP_INNER)

    def forward(self, h):
        B, L, H = h.shape
        ctxs = []
        for b in range(B):
            mlp = InnerMLP(H, MLP_INNER)
            mlp.load_state_dict(self.base.state_dict())
            opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L-3, 2):
                k_h = h[b,t,:].detach(); v_h = h[b,t+1,:].detach()
                with torch.enable_grad():
                    opt.zero_grad()
                    F.mse_loss(mlp(k_h.unsqueeze(0)), v_h.unsqueeze(0)).backward()
                opt.step()
            with torch.no_grad():
                ctxs.append(mlp(h[b,-1,:].detach().unsqueeze(0)).squeeze(0))
        return torch.stack(ctxs)


class DeltaOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.mem = DeltaMemoryModule()
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        return self.out(self.mem(self.enc(seq)))


class ParamOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.mem = ParamMemModule()
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        return self.out(self.mem(self.enc(seq)))


class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc    = Encoder()
        self.delta  = DeltaMemoryModule()
        self.param  = ParamMemModule()
        self.comb   = nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM)
        self.out    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.enc(seq)
        return self.out(self.comb(torch.cat([self.delta(h), self.param(h)], dim=-1)))


def _param_list(model):
    if isinstance(model, ParamOnly):
        return (list(model.enc.parameters()) + list(model.mem.base.parameters()) +
                list(model.out.parameters()))
    if isinstance(model, Hybrid):
        return (list(model.enc.parameters()) + list(model.delta.parameters()) +
                list(model.param.base.parameters()) + list(model.comb.parameters()) +
                list(model.out.parameters()))
    return list(model.parameters())


def train(model, steps, lr):
    opt = Adam(_param_list(model), lr=lr); model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(BATCH)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evalu(model, n=30):
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(n):
            seq, tgt = make_assoc_batch(BATCH)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c/t


class Exp271HybridMemory(Experiment):
    experiment_id = "exp_27_1"
    hypothesis = ("Hybrid memory (delta-rule matrix + parametric MLP) outperforms either "
                  "alone when each component is pre-trained independently for 100 steps "
                  "before joint fine-tuning.")

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, steps_pre=STEPS_PRE, steps_joint=STEPS_JOINT,
                      batch=BATCH, mlp_inner=MLP_INNER)
        total = STEPS_PRE*2 + STEPS_JOINT

        print("  DeltaOnly..."); acc_delta = evalu(train(DeltaOnly(), total, LR))
        print(f"    acc_delta={acc_delta:.3f}")
        print("  ParamOnly..."); acc_param = evalu(train(ParamOnly(), total, LR))
        print(f"    acc_param={acc_param:.3f}")
        print("  ColdJoint..."); acc_cold = evalu(train(Hybrid(), total, LR))
        print(f"    acc_cold={acc_cold:.3f}")

        print("  PretrainedHybrid...")
        dpre  = train(DeltaOnly(), STEPS_PRE, LR)
        ppre  = train(ParamOnly(), STEPS_PRE, LR)
        hybrid = Hybrid()
        hybrid.enc.load_state_dict(dpre.enc.state_dict())
        hybrid.delta.load_state_dict(dpre.mem.state_dict())
        hybrid.param.base.load_state_dict(ppre.mem.base.state_dict())
        acc_hybrid = evalu(train(hybrid, STEPS_JOINT, LR_JOINT))
        print(f"    acc_hybrid={acc_hybrid:.3f}")

        best = max(acc_delta, acc_param)
        superadditive = acc_hybrid / max(best, 1e-6)
        if acc_hybrid > best*1.1 and acc_hybrid > acc_cold + 0.02:
            outcome = OUTCOME_SUPPORTED
        elif acc_hybrid <= best or acc_hybrid <= acc_cold:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE
        metrics = dict(acc_delta_only=round(acc_delta,4), acc_param_only=round(acc_param,4),
                       acc_cold_joint=round(acc_cold,4), acc_pretrained_hybrid=round(acc_hybrid,4),
                       superadditive_ratio=round(superadditive,4))
        notes = (f"Delta={acc_delta:.3f}, Param={acc_param:.3f}, "
                 f"Cold={acc_cold:.3f}, PretrainedHybrid={acc_hybrid:.3f}, "
                 f"ratio={superadditive:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp271HybridMemory().execute()
