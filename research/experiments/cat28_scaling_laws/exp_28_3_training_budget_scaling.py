"""
Experiment 28.3 — Training Budget Scaling (Sample Efficiency)

Hypothesis: Across STEPS ∈ {200, 400, 800, 1600, 3200}, parametric memory's
per-step accuracy gain exceeds slot and delta rule, confirming highest sample efficiency.
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
SEQ_LEN      = 24
STEPS_LIST   = [200, 400, 800, 1600, 3200]
BATCH        = 32
LR           = 3e-4
INFERENCE_LR = 0.01
INNER_DIM    = 16
EVAL_N       = 30


def make_batch(batch_size):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    H, V, N = HIDDEN_DIM, VOCAB_SIZE, NUM_PAIRS
    for b in range(batch_size):
        keys = torch.randint(4, max(8, V // 3), (N * 4,)).unique()[:N]
        while len(keys) < N: keys = torch.cat([keys, torch.randint(4, V // 3, (1,))])[:N]
        vals = torch.randint(V // 2, V, (N,)); pos = 0
        for i in range(N):
            if pos + 1 < SEQ_LEN - 3: seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3): seq[b, p] = 3
        qi = torch.randint(0, N, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM*2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)
    def forward(self, x): e = self.embed(x); return self.norm(e + self.ff(e))


class SlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.q = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE); self.slots = NUM_PAIRS + 2
    def forward(self, seq):
        B, L = seq.shape; H = HIDDEN_DIM; hs = self.enc(seq)
        content = hs[:, :-3, :]; k = min(self.slots, content.shape[1])
        _, idx = torch.topk(content.norm(dim=-1), k, dim=1)
        s = torch.gather(content, 1, idx.unsqueeze(-1).expand(-1, -1, H))
        if k < self.slots: s = torch.cat([s, torch.zeros(B, self.slots-k, H)], dim=1)
        attn = torch.softmax(torch.bmm(self.q(hs[:, -1:, :]), s.transpose(1,2))/H**0.5, -1)
        return self.out(torch.bmm(attn, s).squeeze(1))


class DeltaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.rp = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; M = torch.zeros(B, H, H)
        for t in range(L-1):
            k = hs[:, t, :]; v = hs[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp/(k.pow(2).sum(-1,keepdim=True)+1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        return self.out(self.rp(torch.bmm(M, hs[:,-1:,:].transpose(1,2)).squeeze(-1)))


class InnerMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(HIDDEN_DIM,INNER_DIM); self.fc2=nn.Linear(INNER_DIM,HIDDEN_DIM)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


class ParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder(); self.base = InnerMLP(); self.out = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
        for b in range(B):
            mlp = InnerMLP(); mlp.load_state_dict(self.base.state_dict())
            opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L-3, 2):
                kh = hs[b,t,:].detach(); vh = hs[b,t+1,:].detach()
                with torch.enable_grad():
                    opt.zero_grad(); F.mse_loss(mlp(kh.unsqueeze(0)), vh.unsqueeze(0)).backward()
                opt.step()
            with torch.no_grad():
                ctxs.append(mlp(hs[b,-1,:].detach().unsqueeze(0)).squeeze(0))
        return self.out(torch.stack(ctxs))


def train_checkpoint(model_class, total_steps, checkpoints, batch=BATCH, eval_n=EVAL_N):
    """Train model and evaluate at each checkpoint step."""
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    results = {}; ck_set = set(checkpoints); step = 0
    for target in sorted(checkpoints):
        for _ in range(target - step):
            seq, tgt = make_batch(batch)
            F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
        step = target
        model.eval(); c = t = 0
        with torch.no_grad():
            for _ in range(eval_n):
                seq, tgt = make_batch(batch)
                c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
        results[target] = c / t
        model.train()
    return results


def compute_slope(steps, accs):
    """Fit linear regression of acc vs steps. Returns slope per step."""
    n = len(steps); mx = sum(steps)/n; my = sum(accs)/n
    sxy = sum((x-mx)*(y-my) for x,y in zip(steps,accs))
    sxx = sum((x-mx)**2 for x in steps)
    return sxy / sxx if sxx > 1e-12 else 0.0


class Exp283TrainingBudgetScaling(Experiment):
    experiment_id = "exp_28_3"
    hypothesis = ("Parametric memory has the steepest per-step accuracy gain across "
                  "STEPS ∈ {200,400,800,1600,3200}, confirming highest sample efficiency.")

    def run(self) -> ExperimentResult:
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, num_pairs=NUM_PAIRS,
            seq_len=SEQ_LEN, steps_list=STEPS_LIST, batch=BATCH,
            param_bytes_slot=sum(p.numel()*4 for p in SlotModel().parameters()),
            param_bytes_delta=sum(p.numel()*4 for p in DeltaModel().parameters()),
            param_bytes_param=sum(p.numel()*4 for p in ParamModel().parameters()),
            activation_bytes=BATCH * SEQ_LEN * HIDDEN_DIM * 4,
        )

        print("  Training slot model at checkpoints...")
        slot_ck  = train_checkpoint(SlotModel,  max(STEPS_LIST), STEPS_LIST)
        print("  Training delta model at checkpoints...")
        delta_ck = train_checkpoint(DeltaModel, max(STEPS_LIST), STEPS_LIST)
        print("  Training param model at checkpoints...")
        param_ck = train_checkpoint(ParamModel, max(STEPS_LIST), STEPS_LIST)

        metrics = {}
        for s in STEPS_LIST:
            metrics[f"acc_slot_s{s}"]  = round(slot_ck[s],  4)
            metrics[f"acc_delta_s{s}"] = round(delta_ck[s], 4)
            metrics[f"acc_param_s{s}"] = round(param_ck[s], 4)

        slope_slot  = compute_slope(STEPS_LIST, [slot_ck[s]  for s in STEPS_LIST])
        slope_delta = compute_slope(STEPS_LIST, [delta_ck[s] for s in STEPS_LIST])
        slope_param = compute_slope(STEPS_LIST, [param_ck[s] for s in STEPS_LIST])
        metrics["slope_slot"]  = round(slope_slot  * 1000, 6)   # per 1000 steps
        metrics["slope_delta"] = round(slope_delta * 1000, 6)
        metrics["slope_param"] = round(slope_param * 1000, 6)

        if slope_param > slope_delta and slope_param > slope_slot:
            outcome = OUTCOME_SUPPORTED
        elif slope_slot > slope_param and slope_slot > slope_delta:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Slopes (×10³/step): slot={slope_slot*1e3:.4f}, "
                 f"delta={slope_delta*1e3:.4f}, param={slope_param*1e3:.4f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp283TrainingBudgetScaling().execute()
