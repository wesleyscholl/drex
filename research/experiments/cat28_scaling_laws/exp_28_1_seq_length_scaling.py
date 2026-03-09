"""
Experiment 28.1 — Sequence Length Scaling

Hypothesis: Parametric memory retains >90% accuracy at SEQ_LEN=192 (8× baseline)
while slot memory drops below 30%, confirming a qualitative crossover.

Tracks explicit param_bytes and activation_bytes per architecture.
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
NUM_PAIRS    = 6
SEQ_LENS     = [24, 48, 96, 192]
STEPS        = 500
BATCH        = 16
LR           = 3e-4
INFERENCE_LR = 0.01
INNER_DIM    = 16
EVAL_N       = 30


def make_batch(batch_size, seq_len, num_pairs, vocab_size=VOCAB_SIZE):
    seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    tgt = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 4,)).unique()[:num_pairs]
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


class Encoder(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, h=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab, h)
        self.ff    = nn.Sequential(nn.Linear(h, h * 2), nn.ReLU(), nn.Linear(h * 2, h))
        self.norm  = nn.LayerNorm(h)
    def forward(self, x):
        e = self.embed(x); return self.norm(e + self.ff(e))


class SlotModel(nn.Module):
    def __init__(self, num_slots=NUM_PAIRS + 2, h=HIDDEN_DIM, v=VOCAB_SIZE):
        super().__init__()
        self.slots = num_slots; self.enc = Encoder(v, h)
        self.q   = nn.Linear(h, h); self.out = nn.Linear(h, v)
    def forward(self, seq):
        B, L = seq.shape; H = self.enc.embed.embedding_dim
        hs = self.enc(seq); content = hs[:, :-3, :]
        k  = min(self.slots, content.shape[1])
        _, idx = torch.topk(content.norm(dim=-1), k, dim=1)
        slots  = torch.gather(content, 1, idx.unsqueeze(-1).expand(-1, -1, H))
        if k < self.slots:
            slots = torch.cat([slots, torch.zeros(B, self.slots - k, H)], dim=1)
        attn = torch.softmax(torch.bmm(self.q(hs[:, -1:, :]),
                                       slots.transpose(1, 2)) / H**0.5, -1)
        return self.out(torch.bmm(attn, slots).squeeze(1))
    def param_bytes(self):
        return sum(p.numel() * 4 for p in self.parameters())


class DeltaModel(nn.Module):
    def __init__(self, h=HIDDEN_DIM, v=VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(v, h); self.rp = nn.Linear(h, h); self.out = nn.Linear(h, v)
    def forward(self, seq):
        h_all = self.enc(seq); B, L, H = h_all.shape
        M = torch.zeros(B, H, H)
        for t in range(L - 1):
            k = h_all[:, t, :]; v = h_all[:, t, :]
            vp = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            M  = M + torch.bmm((v - vp / (k.pow(2).sum(-1, keepdim=True) + 1e-6)).unsqueeze(-1),
                               k.unsqueeze(1))
        return self.out(self.rp(torch.bmm(M, h_all[:, -1:, :].transpose(1, 2)).squeeze(-1)))
    def param_bytes(self):
        return sum(p.numel() * 4 for p in self.parameters())


class InnerMLP(nn.Module):
    def __init__(self, h=HIDDEN_DIM, inner=INNER_DIM):
        super().__init__()
        self.fc1 = nn.Linear(h, inner); self.fc2 = nn.Linear(inner, h)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


class ParamModel(nn.Module):
    def __init__(self, h=HIDDEN_DIM, v=VOCAB_SIZE):
        super().__init__()
        self.enc = Encoder(v, h); self.base = InnerMLP(h); self.out = nn.Linear(h, v)
    def forward(self, seq):
        hs = self.enc(seq); B, L, H = hs.shape; ctxs = []
        for b in range(B):
            mlp = InnerMLP(H, INNER_DIM); mlp.load_state_dict(self.base.state_dict())
            opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L - 3, 2):
                k_h = hs[b, t, :].detach(); v_h = hs[b, t + 1, :].detach()
                with torch.enable_grad():
                    opt.zero_grad(); F.mse_loss(mlp(k_h.unsqueeze(0)), v_h.unsqueeze(0)).backward()
                opt.step()
            with torch.no_grad():
                ctxs.append(mlp(hs[b, -1, :].detach().unsqueeze(0)).squeeze(0))
        return self.out(torch.stack(ctxs))
    def param_bytes(self):
        return sum(p.numel() * 4 for p in self.parameters())


def train_eval(model_class, seq_len, steps=STEPS, batch=BATCH):
    model = model_class(); opt = Adam(model.parameters(), lr=LR); model.train()
    for _ in range(steps):
        seq, tgt = make_batch(batch, seq_len, NUM_PAIRS)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(EVAL_N):
            seq, tgt = make_batch(batch, seq_len, NUM_PAIRS)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c / t


class Exp281SeqLengthScaling(Experiment):
    experiment_id = "exp_28_1"
    hypothesis = ("Parametric memory retains >90% accuracy at SEQ_LEN=192 while "
                  "slot memory drops below 30%, confirming a length-scaling crossover.")

    def run(self) -> ExperimentResult:
        sm = SlotModel(); dm = DeltaModel(); pm = ParamModel()
        config = dict(
            hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_pairs=NUM_PAIRS,
            seq_lens=SEQ_LENS, steps=STEPS, batch=BATCH,
            param_bytes_slot=sm.param_bytes(), param_bytes_delta=dm.param_bytes(),
            param_bytes_param=pm.param_bytes(),
            activation_bytes=BATCH * max(SEQ_LENS) * HIDDEN_DIM * 4,
        )

        metrics = {}
        for sl in SEQ_LENS:
            print(f"  SEQ_LEN={sl}: training slot...")
            metrics[f"acc_slot_len{sl}"] = round(train_eval(SlotModel, sl), 4)
            print(f"    slot={metrics[f'acc_slot_len{sl}']:.3f}  delta...")
            metrics[f"acc_delta_len{sl}"] = round(train_eval(DeltaModel, sl), 4)
            print(f"    delta={metrics[f'acc_delta_len{sl}']:.3f}  param...")
            metrics[f"acc_param_len{sl}"] = round(train_eval(ParamModel, sl), 4)
            print(f"    param={metrics[f'acc_param_len{sl}']:.3f}")

        base_slot  = metrics["acc_slot_len24"];  slot_192  = metrics["acc_slot_len192"]
        base_param = metrics["acc_param_len24"]; param_192 = metrics["acc_param_len192"]
        slot_ret   = slot_192  / max(base_slot,  1e-6)
        param_ret  = param_192 / max(base_param, 1e-6)
        metrics["slot_retention_8x"]  = round(slot_ret,  4)
        metrics["param_retention_8x"] = round(param_ret, 4)

        if param_ret > 0.90 and slot_ret < 0.30:
            outcome = OUTCOME_SUPPORTED
        elif param_ret < 0.50 and slot_ret > 0.60:
            outcome = OUTCOME_REFUTED
        else:
            outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Slot retention 8×={slot_ret:.3f}, param retention 8×={param_ret:.3f}. "
                 f"Slot@192={slot_192:.3f}, Param@192={param_192:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp281SeqLengthScaling().execute()
