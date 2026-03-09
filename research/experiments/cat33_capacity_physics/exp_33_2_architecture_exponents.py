"""
Experiment 33.2 — Architecture Interference Exponents Comparison

Hypothesis: Different architectures have distinct interference exponents:
γ_parametric < γ_slot < γ_delta (parametric most capacity-efficient per dim unit).
γ spread > 0.3 between fastest and slowest decay.
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

HIDDEN_DIM   = 64
VOCAB_SIZE   = 256
STEPS        = 400
BATCH        = 32
LR           = 3e-4
N_PAIRS_LIST = [2, 4, 8, 16, 32, 64]
EVAL_BATCHES = 40
INNER_DIM    = 16
INFERENCE_LR = 0.05


def make_assoc_batch(batch_size, n_pairs, seq_len, vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    key_range = max(8, vocab_size // 3); val_base = vocab_size // 2
    for b in range(batch_size):
        keys = torch.randint(4, key_range, (n_pairs * 4,)).unique()[:n_pairs]
        while len(keys) < n_pairs:
            keys = torch.cat([keys, torch.randint(4, key_range, (1,))])[:n_pairs]
        vals = torch.randint(val_base, vocab_size, (n_pairs,)); pos = 0
        for i in range(n_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3): seq[b, p] = 3
        qi = torch.randint(0, n_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        tgt[b] = vals[qi]
    return seq, tgt


def fit_power_law(rhos, accs):
    xs = [math.log(r) for r in rhos]
    ys = [math.log(max(a, 1e-6)) for a in accs]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx < 1e-12: return 0.0, my, 0.0
    slope = sxy / sxx; inter = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy) if syy > 1e-12 else 1.0
    return -slope, inter, r2


class Encoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                                   nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class SlotMemoryModel(nn.Module):
    def __init__(self, num_slots, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.num_slots = num_slots
        self.enc    = Encoder(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out    = nn.Linear(hidden_dim, vocab_size)
    def forward(self, seq):
        B, L = seq.shape; H = self.enc.embed.embedding_dim
        h = self.enc(seq); norms = h[:, :-3, :].norm(dim=-1)
        k = min(self.num_slots, L - 3); _, idx = torch.topk(norms, k, dim=-1)
        slots = torch.gather(h[:, :-3, :], 1, idx.unsqueeze(-1).expand(-1, -1, H))
        if k < self.num_slots:
            slots = torch.cat([slots, torch.zeros(B, self.num_slots - k, H)], dim=1)
        q    = self.q_proj(h[:, -1, :]).unsqueeze(1)
        attn = torch.softmax(torch.bmm(q, slots.transpose(1, 2)) / H**0.5, -1)
        return self.out(torch.bmm(attn, slots).squeeze(1))


class DeltaModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(vocab_size, hidden_dim)
        self.rp  = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = hs[:, t, :]; v = hs[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        return self.out(self.rp(torch.bmm(M, hs[:, -1:, :].transpose(1, 2)).squeeze(-1)))


class InnerMLP(nn.Module):
    def __init__(self, h=HIDDEN_DIM, inner=INNER_DIM):
        super().__init__()
        self.fc1 = nn.Linear(h, inner); self.fc2 = nn.Linear(inner, h)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


class ParamModel(nn.Module):
    """Test-time SGD parametric memory (1 inner step per pair for tractability)."""
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.enc  = Encoder(vocab_size, hidden_dim)
        self.base = InnerMLP(hidden_dim, INNER_DIM)
        self.out  = nn.Linear(hidden_dim, vocab_size)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
        for b in range(B):
            w1 = self.base.fc1.weight.clone().detach().requires_grad_(True)
            b1 = self.base.fc1.bias.clone().detach().requires_grad_(True)
            w2 = self.base.fc2.weight.clone().detach().requires_grad_(True)
            b2 = self.base.fc2.bias.clone().detach().requires_grad_(True)
            for t in range(0, L - 3, 2):
                kh = hs[b, t, :].detach(); vh = hs[b, t + 1, :].detach()
                with torch.enable_grad():
                    h_mid = F.relu(kh @ w1.T + b1); pred = h_mid @ w2.T + b2
                    loss  = F.mse_loss(pred, vh)
                    grads = torch.autograd.grad(loss, [w1, b1, w2, b2], create_graph=False)
                with torch.no_grad():
                    w1 = w1 - INFERENCE_LR * grads[0]; b1 = b1 - INFERENCE_LR * grads[1]
                    w2 = w2 - INFERENCE_LR * grads[2]; b2 = b2 - INFERENCE_LR * grads[3]
                w1 = w1.detach().requires_grad_(True); b1 = b1.detach().requires_grad_(True)
                w2 = w2.detach().requires_grad_(True); b2 = b2.detach().requires_grad_(True)
            with torch.no_grad():
                q_h = hs[b, -1, :].detach(); ctx = F.relu(q_h @ w1.T + b1) @ w2.T + b2
            ctxs.append(ctx)
        return self.out(torch.stack(ctxs))


def train_and_eval(model, n_pairs):
    seq_len = max(24, 2 * n_pairs + 8)
    opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH, n_pairs, seq_len)
        F.cross_entropy(model(seq), tgt).backward()
        opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            seq, tgt = make_assoc_batch(BATCH, n_pairs, seq_len)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp332ArchitectureExponents(Experiment):
    experiment_id = "exp_33_2"
    hypothesis = ("Different architectures have distinct interference exponents: "
                  "γ_parametric < γ_slot < γ_delta with spread > 0.3.")

    def run(self) -> ExperimentResult:
        config = dict(
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, steps=STEPS, batch=BATCH,
            n_pairs_list=N_PAIRS_LIST, inner_dim=INNER_DIM,
            param_bytes_slot=sum(p.numel() * 4 for p in
                                 SlotMemoryModel(N_PAIRS_LIST[2] + 2).parameters()),
            param_bytes_delta=sum(p.numel() * 4 for p in DeltaModel().parameters()),
            param_bytes_param=sum(p.numel() * 4 for p in ParamModel().parameters()),
            activation_bytes=BATCH * 24 * HIDDEN_DIM * 4,
        )

        accs_slot = {}; accs_delta = {}; accs_param = {}; rhos = {}
        for n in N_PAIRS_LIST:
            rho = n / HIDDEN_DIM; rhos[n] = round(rho, 4)
            print(f"  n_pairs={n}, ρ={rho:.3f}...")
            accs_slot[n]  = round(train_and_eval(SlotMemoryModel(n + 2), n), 4)
            accs_delta[n] = round(train_and_eval(DeltaModel(), n), 4)
            accs_param[n] = round(train_and_eval(ParamModel(), n), 4)
            print(f"    slot={accs_slot[n]:.4f} delta={accs_delta[n]:.4f} param={accs_param[n]:.4f}")

        rho_list = [rhos[n] for n in N_PAIRS_LIST]
        g_slot,  _, r2_slot  = fit_power_law(rho_list, [accs_slot[n]  for n in N_PAIRS_LIST])
        g_delta, _, r2_delta = fit_power_law(rho_list, [accs_delta[n] for n in N_PAIRS_LIST])
        g_param, _, r2_param = fit_power_law(rho_list, [accs_param[n] for n in N_PAIRS_LIST])

        print(f"  γ: slot={g_slot:.3f} delta={g_delta:.3f} param={g_param:.3f}")
        spread = max(g_slot, g_delta, g_param) - min(g_slot, g_delta, g_param)
        ordering_ok = g_param < g_slot < g_delta

        metrics = {f"acc_slot_n{n}": accs_slot[n] for n in N_PAIRS_LIST}
        metrics.update({f"acc_delta_n{n}": accs_delta[n] for n in N_PAIRS_LIST})
        metrics.update({f"acc_param_n{n}": accs_param[n] for n in N_PAIRS_LIST})
        metrics.update(dict(
            gamma_slot=round(g_slot, 4), gamma_delta=round(g_delta, 4),
            gamma_param=round(g_param, 4), gamma_spread=round(spread, 4),
            r2_slot=round(r2_slot, 4), r2_delta=round(r2_delta, 4),
            r2_param=round(r2_param, 4), ordering_ok=int(ordering_ok),
        ))

        if ordering_ok and spread > 0.3:
            outcome = OUTCOME_SUPPORTED
        elif spread < 0.1:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"γ ordering: param={g_param:.3f} < slot={g_slot:.3f} < delta={g_delta:.3f}, "
                 f"spread={spread:.3f}, R²: slot={r2_slot:.3f} delta={r2_delta:.3f} "
                 f"param={r2_param:.3f}. Ordering correct: {ordering_ok}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp332ArchitectureExponents().execute()
