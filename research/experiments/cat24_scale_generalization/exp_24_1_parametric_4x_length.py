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

experiment_id = "exp_24_1"
hypothesis = ("Parametric memory retains >80% accuracy at 4x training sequence length, "
              "while slot memory drops below 40%.")

VOCAB_SIZE   = 64
HIDDEN_DIM   = 32
SEQ_LENS     = [24, 48, 96]       # 1x, 2x, 4x
NUM_PAIRS_PER_LEN = [4, 8, 12]
MEMORY_SLOTS = 8
STEPS        = 300
BATCH        = 8
LR           = 3e-4
INFERENCE_LR = 0.01
MLP_INNER    = 8


def make_assoc_batch(batch_size, seq_len, num_pairs):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:num_pairs]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < seq_len - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, seq_len - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, seq_len - 3] = 2; seq[b, seq_len - 2] = keys[qi]; seq[b, seq_len - 1] = 0
        target[b] = vals[qi]
    return seq, target


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
    def __init__(self, num_slots=MEMORY_SLOTS, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder    = Encoder(vocab_size, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.output     = nn.Linear(hidden_dim, vocab_size)
        self.num_slots  = num_slots

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        write_len = min(self.num_slots, L - 3)
        slots = hidden[:, :write_len, :]
        if write_len < self.num_slots:
            pad   = torch.zeros(B, self.num_slots - write_len, H)
            slots = torch.cat([slots, pad], dim=1)
        query = self.query_proj(hidden[:, -1, :]).unsqueeze(1)
        keys  = self.key_proj(slots)
        attn  = F.softmax(torch.bmm(query, keys.transpose(1, 2)) / H**0.5, dim=-1)
        ctx   = torch.bmm(attn, slots).squeeze(1)
        return self.output(ctx)


class InnerMLP(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, inner=MLP_INNER):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class ParametricMemoryModel(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder  = Encoder(vocab_size, hidden_dim)
        self.base_mlp = InnerMLP(hidden_dim, MLP_INNER)
        self.output   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        contexts = []
        for b in range(B):
            mlp = InnerMLP(H, MLP_INNER)
            mlp.load_state_dict(self.base_mlp.state_dict())
            inner_opt = torch.optim.SGD(mlp.parameters(), lr=INFERENCE_LR)
            for t in range(0, L - 3, 2):
                key_h = hidden[b, t, :].detach()
                val_h = hidden[b, t + 1, :].detach()
                with torch.enable_grad():
                    inner_opt.zero_grad()
                    pred = mlp(key_h.unsqueeze(0))
                    loss = F.mse_loss(pred, val_h.unsqueeze(0))
                    loss.backward()
                inner_opt.step()
            with torch.no_grad():
                ctx = mlp(hidden[b, -1, :].detach().unsqueeze(0)).squeeze(0)
            contexts.append(ctx)
        return self.output(torch.stack(contexts, dim=0))


def train_model(model, steps, batch_size, seq_len, num_pairs):
    if isinstance(model, ParametricMemoryModel):
        opt = Adam(list(model.encoder.parameters()) +
                   list(model.base_mlp.parameters()) +
                   list(model.output.parameters()), lr=LR)
    else:
        opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(batch_size, seq_len, num_pairs)
        loss = F.cross_entropy(model(seq), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, seq_len, num_pairs, n_batches=40):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH, seq_len, num_pairs)
            correct += (model(seq).argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp241Parametric4xLength(Experiment):
    experiment_id = "exp_24_1"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                      seq_lens=SEQ_LENS, num_pairs_per_len=NUM_PAIRS_PER_LEN,
                      memory_slots=MEMORY_SLOTS, steps=STEPS, batch=BATCH)
        acc_slot = {}; acc_param = {}
        for seq_len, num_pairs in zip(SEQ_LENS, NUM_PAIRS_PER_LEN):
            print(f"\n--- seq_len={seq_len}, num_pairs={num_pairs} ---")
            print("  Training slot memory...")
            m_slot = SlotMemoryModel(MEMORY_SLOTS)
            m_slot = train_model(m_slot, STEPS, BATCH, seq_len, num_pairs)
            acc_slot[seq_len] = round(evaluate(m_slot, seq_len, num_pairs), 4)
            print(f"  Slot acc={acc_slot[seq_len]:.3f}")
            print("  Training parametric memory...")
            m_param = ParametricMemoryModel()
            m_param = train_model(m_param, STEPS, BATCH, seq_len, num_pairs)
            acc_param[seq_len] = round(evaluate(m_param, seq_len, num_pairs), 4)
            print(f"  Param acc={acc_param[seq_len]:.3f}")

        # 4x retention = acc at 96 / acc at 24
        retention_slot  = acc_slot[96]  / max(acc_slot[24],  0.001)
        retention_param = acc_param[96] / max(acc_param[24], 0.001)

        metrics = dict(
            acc_slot=acc_slot, acc_parametric=acc_param,
            retention_slot_4x=round(retention_slot, 4),
            retention_param_4x=round(retention_param, 4),
        )
        if retention_param > 0.80 and acc_param[96] > 0.40 and acc_slot[96] < 0.40:
            outcome = OUTCOME_SUPPORTED
            notes   = (f"Param retention_4x={retention_param:.3f}>0.80, "
                       f"slot acc_96={acc_slot[96]:.3f}<0.40.")
        elif acc_param[96] < acc_slot[96]:
            outcome = OUTCOME_REFUTED
            notes   = f"Parametric underperforms slot at 4x length."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = (f"Param retention_4x={retention_param:.3f}, "
                       f"slot acc_96={acc_slot[96]:.3f}. Threshold not fully met.")
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp241Parametric4xLength().execute()
