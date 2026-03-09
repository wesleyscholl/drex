"""
Experiment 28.2 — Hidden Dimension Power Law

Hypothesis: Accuracy scales as dim^α with α > 0.3 for energy-gated delta rule,
confirmed by log-log fit (R² > 0.95) across HIDDEN_DIM ∈ {32, 64, 128, 256}.
Logs param_bytes and activation_bytes per dimension.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from experiments.base import (Experiment, ExperimentResult, OUTCOME_SUPPORTED,
                               OUTCOME_REFUTED, OUTCOME_INCONCLUSIVE)

VOCAB_SIZE  = 64
NUM_PAIRS   = 5
SEQ_LEN     = 24
STEPS       = 400
LR          = 3e-4
BATCH       = 32
EVAL_N      = 40
DIMS        = [32, 64, 128, 256]


def make_batch(batch_size, hidden_dim=64, vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (NUM_PAIRS * 4,)).unique()[:NUM_PAIRS]
        while len(keys) < NUM_PAIRS:
            keys = torch.cat([keys, torch.randint(4, vocab_size // 3, (1,))])[:NUM_PAIRS]
        vals = torch.randint(vocab_size // 2, vocab_size, (NUM_PAIRS,))
        pos = 0
        for i in range(NUM_PAIRS):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


class Encoder(nn.Module):
    def __init__(self, vocab, h):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x):
        e = self.embed(x); return self.norm(e + self.ff(e))


def make_slot_model(h, v):
    slots = NUM_PAIRS + 2
    enc   = Encoder(v, h)
    q_proj = nn.Linear(h, h)
    out   = nn.Linear(h, v)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc; self.q = q_proj; self.out = out; self.slots = slots
        def forward(self, seq):
            B, L = seq.shape; H = self.enc.embed.embedding_dim
            hs = self.enc(seq); content = hs[:, :-3, :]
            k  = min(self.slots, content.shape[1])
            _, idx = torch.topk(content.norm(dim=-1), k, dim=1)
            s = torch.gather(content, 1, idx.unsqueeze(-1).expand(-1, -1, H))
            if k < self.slots:
                s = torch.cat([s, torch.zeros(B, self.slots - k, H)], dim=1)
            attn = torch.softmax(torch.bmm(self.q(hs[:, -1:, :]), s.transpose(1, 2)) / H**0.5, -1)
            return self.out(torch.bmm(attn, s).squeeze(1))
    return M()


def make_delta_model(h, v):
    enc = Encoder(v, h); rp = nn.Linear(h, h); out = nn.Linear(h, v)
    class M(nn.Module):
        def __init__(self): super().__init__(); self.enc=enc; self.rp=rp; self.out=out
        def forward(self, seq):
            hs = self.enc(seq); B, L, H = hs.shape
            M2 = torch.zeros(B, H, H)
            for t in range(L - 1):
                k = hs[:, t, :]; vt = hs[:, t, :]
                vp = torch.bmm(M2, k.unsqueeze(-1)).squeeze(-1)
                denom = k.pow(2).sum(-1, keepdim=True) + 1e-6
                M2 = M2 + torch.bmm((vt - vp / denom).unsqueeze(-1), k.unsqueeze(1))
            # energy gate
            return self.out(self.rp(torch.bmm(M2, hs[:, -1:, :].transpose(1, 2)).squeeze(-1)))
    return M()


def make_param_model(h, v):
    enc = Encoder(v, h)
    inner_dim = max(4, h // 8)
    base_fc1 = nn.Linear(h, inner_dim); base_fc2 = nn.Linear(inner_dim, h)
    out = nn.Linear(h, v)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc; self.fc1 = base_fc1; self.fc2 = base_fc2; self.out = out
            self.inner = inner_dim
        def forward(self, seq):
            hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
            sd = {"fc1.weight": self.fc1.weight, "fc1.bias": self.fc1.bias,
                  "fc2.weight": self.fc2.weight, "fc2.bias": self.fc2.bias}
            for b in range(B):
                fc1 = nn.Linear(H, self.inner); fc2 = nn.Linear(self.inner, H)
                fc1.weight.data.copy_(self.fc1.weight.data)
                fc1.bias.data.copy_(self.fc1.bias.data)
                fc2.weight.data.copy_(self.fc2.weight.data)
                fc2.bias.data.copy_(self.fc2.bias.data)
                opt = torch.optim.SGD([*fc1.parameters(), *fc2.parameters()], lr=0.01)
                for t in range(0, L - 3, 2):
                    kh = hs[b, t, :].detach(); vh = hs[b, t + 1, :].detach()
                    with torch.enable_grad():
                        opt.zero_grad()
                        F.mse_loss(fc2(F.relu(fc1(kh.unsqueeze(0)))), vh.unsqueeze(0)).backward()
                    opt.step()
                with torch.no_grad():
                    ctxs.append(fc2(F.relu(fc1(hs[b, -1, :].detach().unsqueeze(0)))).squeeze(0))
            return self.out(torch.stack(ctxs))
    return M()


def param_bytes(m):
    return sum(p.numel() * 4 for p in m.parameters())


def train_eval(model, steps=STEPS, batch=BATCH):
    opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(batch)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


def fit_power_law(dims, accs):
    xs = [math.log(d) for d in dims]
    ys = [math.log(max(a, 1e-6)) for a in accs]
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sxx = sum((x-mx)**2 for x in xs); syy = sum((y-my)**2 for y in ys)
    if sxx < 1e-12: return 0.0, my, 0.0
    slope = sxy/sxx; inter = my - slope*mx
    r2 = (sxy**2)/(sxx*syy) if syy > 1e-12 else 1.0
    return slope, inter, r2


class Exp282HiddenDimPowerLaw(Experiment):
    experiment_id = "exp_28_2"
    hypothesis = ("Accuracy scales as dim^α with α > 0.3 for delta rule, "
                  "R² > 0.95 log-log fit over HIDDEN_DIM ∈ {32,64,128,256}.")

    def run(self) -> ExperimentResult:
        metrics = {}; accs_slot = {}; accs_delta = {}; accs_param = {}
        config = dict(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, num_pairs=NUM_PAIRS,
                      steps=STEPS, batch=BATCH, dims=DIMS)

        for h in DIMS:
            pbytes = BATCH * SEQ_LEN * h * 4
            print(f"  H={h}...")
            sm = make_slot_model(h, VOCAB_SIZE)
            dm = make_delta_model(h, VOCAB_SIZE)
            pm = make_param_model(h, VOCAB_SIZE)
            config[f"param_bytes_H{h}_slot"]  = param_bytes(sm)
            config[f"param_bytes_H{h}_delta"] = param_bytes(dm)
            config[f"param_bytes_H{h}_param"] = param_bytes(pm)
            config[f"activation_bytes_H{h}"]  = pbytes
            a_slot  = round(train_eval(sm), 4)
            a_delta = round(train_eval(dm), 4)
            a_param = round(train_eval(pm), 4)
            accs_slot[h] = a_slot; accs_delta[h] = a_delta; accs_param[h] = a_param
            metrics[f"acc_slot_H{h}"]  = a_slot
            metrics[f"acc_delta_H{h}"] = a_delta
            metrics[f"acc_param_H{h}"] = a_param
            print(f"    slot={a_slot:.3f}, delta={a_delta:.3f}, param={a_param:.3f}")

        dim_list = DIMS
        s_al, _, s_r2 = fit_power_law(dim_list, [accs_slot[d]  for d in dim_list])
        d_al, _, d_r2 = fit_power_law(dim_list, [accs_delta[d] for d in dim_list])
        p_al, _, p_r2 = fit_power_law(dim_list, [accs_param[d] for d in dim_list])
        metrics.update(dict(
            alpha_slot=round(s_al, 4), r2_slot=round(s_r2, 4),
            alpha_delta=round(d_al, 4), r2_delta=round(d_r2, 4),
            alpha_param=round(p_al, 4), r2_param=round(p_r2, 4),
        ))

        if d_r2 > 0.95 and d_al > 0.30:
            outcome = OUTCOME_SUPPORTED
        elif d_r2 < 0.70 or d_al < 0.10:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Delta: α={d_al:.3f}, R²={d_r2:.3f}. "
                 f"Slot: α={s_al:.3f}, R²={s_r2:.3f}. "
                 f"Param: α={p_al:.3f}, R²={p_r2:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp282HiddenDimPowerLaw().execute()
