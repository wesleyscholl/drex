"""
Experiment 32.4 — Deep Seed Validation: Parametric Seq Scaling (exp_16_3 replication)

Replicates exp_16_3 with 9 seeds {0,1,2,7,13,42,99,123,777}.
Stricter criterion: param_retention_gap > 0.35 (vs original's >0.15).
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

VOCAB_SIZE         = 64
HIDDEN_DIM         = 32
SEQ_LENS           = [24, 48]
NUM_PAIRS_PER_LEN  = [4, 8]
MEMORY_SLOTS       = 8
STEPS              = 300
BATCH              = 8
INFERENCE_LR       = 0.01
LR                 = 3e-4
MLP_INNER          = 8


def make_assoc_batch(batch_size, seq_len, vocab_size, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size//3), (num_pairs*3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, vocab_size//3, (1,))])[:num_pairs]
        vals = torch.randint(vocab_size//2, vocab_size, (num_pairs,)); pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos+1] = vals[i]; pos += 2
        for p in range(pos, seq_len-3): seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len-3] = 2; seq[b, seq_len-2] = keys[qi]; target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.ff    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(),
                                   nn.Linear(hidden_dim*2, hidden_dim))
        self.norm  = nn.LayerNorm(hidden_dim)
    def forward(self, x): h = self.embed(x); return self.norm(h + self.ff(h))


class SlotMemoryModel(nn.Module):
    def __init__(self, num_slots=MEMORY_SLOTS):
        super().__init__()
        self.encoder    = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.query_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.key_proj   = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output     = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.num_slots  = num_slots
    def forward(self, seq):
        hidden = self.encoder(seq); B, L, H = hidden.shape
        write_len = min(self.num_slots, L - 3)
        slots = hidden[:, :write_len, :]
        if write_len < self.num_slots:
            slots = torch.cat([slots, torch.zeros(B, self.num_slots-write_len, H)], dim=1)
        query = self.query_proj(hidden[:,-1,:]).unsqueeze(1)
        keys  = self.key_proj(slots)
        attn  = torch.softmax(torch.bmm(query, keys.transpose(1,2)) / H**0.5, dim=-1)
        return self.output(torch.bmm(attn, slots).squeeze(1))


class InnerMLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner); self.fc2 = nn.Linear(inner, hidden_dim)
    def forward(self, x): return self.fc2(F.relu(self.fc1(x)))


class ParametricMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder(VOCAB_SIZE, HIDDEN_DIM)
        self.base_mlp = InnerMLP()
        self.output   = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
    def forward(self, seq):
        hidden = self.encoder(seq); B, L, H = hidden.shape; ctxs = []
        for b in range(B):
            mlp = InnerMLP(); mlp.load_state_dict(self.base_mlp.state_dict())
            inner_opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L-3, 2):
                kh = hidden[b, t, :].detach(); vh = hidden[b, t+1, :].detach()
                with torch.enable_grad():
                    inner_opt.zero_grad()
                    F.mse_loss(mlp(kh.unsqueeze(0)), vh.unsqueeze(0)).backward()
                inner_opt.step()
            with torch.no_grad():
                ctxs.append(mlp(hidden[b,-1,:].detach().unsqueeze(0)).squeeze(0))
        return self.output(torch.stack(ctxs))


def train_model(model, steps, seq_len, num_pairs):
    if isinstance(model, ParametricMemoryModel):
        opt = Adam(list(model.encoder.parameters()) + list(model.base_mlp.parameters()) +
                   list(model.output.parameters()), lr=LR)
    else:
        opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(BATCH, seq_len, VOCAB_SIZE, num_pairs)
        F.cross_entropy(model(seq), tgt).backward(); opt.step(); opt.zero_grad()
    return model


def evaluate(model, seq_len, num_pairs, n_batches=40):
    model.eval(); c = t = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, seq_len, VOCAB_SIZE, num_pairs)
            c += (model(seq).argmax(-1) == tgt).sum().item(); t += tgt.size(0)
    return c/t


class Exp324ParametricSeqScaling9Seeds(Experiment):
    experiment_id = "exp_32_4"
    hypothesis = ("Deep seed validation of parametric seq scaling (exp_16_3): "
                  "param retention gap > 0.35 (stricter criterion).")

    def run(self) -> ExperimentResult:
        torch.manual_seed(self.seed)
        config = dict(
            vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_lens=SEQ_LENS,
            num_pairs_per_len=NUM_PAIRS_PER_LEN, memory_slots=MEMORY_SLOTS,
            steps=STEPS, batch=BATCH, inference_lr=INFERENCE_LR, seed=self.seed,
            replicates="exp_16_3",
            param_bytes=sum(p.numel()*4 for p in ParametricMemoryModel().parameters()),
            activation_bytes=BATCH * max(SEQ_LENS) * HIDDEN_DIM * 4,
        )
        acc_slot = {}; acc_param = {}
        for seq_len, num_pairs in zip(SEQ_LENS, NUM_PAIRS_PER_LEN):
            print(f"  seq_len={seq_len}, num_pairs={num_pairs}")
            model_A = train_model(SlotMemoryModel(), STEPS, seq_len, num_pairs)
            acc_slot[seq_len] = round(evaluate(model_A, seq_len, num_pairs), 4)
            print(f"    slot={acc_slot[seq_len]:.3f}")
            model_B = train_model(ParametricMemoryModel(), STEPS, seq_len, num_pairs)
            acc_param[seq_len] = round(evaluate(model_B, seq_len, num_pairs), 4)
            print(f"    param={acc_param[seq_len]:.3f}")

        retention_A = acc_slot[48]  / max(acc_slot[24],  0.001)
        retention_B = acc_param[48] / max(acc_param[24], 0.001)
        retention_diff = retention_B - retention_A

        metrics = dict(
            acc_slot_len24=acc_slot[24], acc_slot_len48=acc_slot[48],
            acc_param_len24=acc_param[24], acc_param_len48=acc_param[48],
            retention_slot=round(retention_A, 4), retention_parametric=round(retention_B, 4),
            retention_diff=round(retention_diff, 4), seed=self.seed,
        )

        if retention_diff > 0.35:   outcome = OUTCOME_SUPPORTED
        elif retention_diff < 0.10: outcome = OUTCOME_REFUTED
        else:                       outcome = OUTCOME_INCONCLUSIVE

        notes = (f"Seed={self.seed}. Param retention gap={retention_diff:.3f}. "
                 f"Required gap > 0.35. Slot retention={retention_A:.3f}, "
                 f"Param retention={retention_B:.3f}.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp324ParametricSeqScaling9Seeds().execute()
