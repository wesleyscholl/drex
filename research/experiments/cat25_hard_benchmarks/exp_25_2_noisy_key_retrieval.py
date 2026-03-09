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

experiment_id = "exp_25_2"
hypothesis = ("Noisy-key retrieval (Gaussian noise added to query at test time): "
              "slot memory degrades >20% while parametric memory degrades <10%.")

VOCAB_SIZE   = 64
HIDDEN_DIM   = 32
SEQ_LEN      = 24
NUM_PAIRS    = 5
MEMORY_SLOTS = 8
STEPS        = 300
BATCH        = 8
LR           = 3e-4
INFERENCE_LR = 0.01
MLP_INNER    = 8
NOISE_SIGMAS = [0.0, 0.1, 0.2]   # noise standard deviations at test time


def make_assoc_batch(batch_size, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE,
                     num_pairs=NUM_PAIRS):
    seq    = torch.zeros(batch_size, seq_len, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, vocab_size // 3), (num_pairs * 3,)).unique()[:num_pairs]
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

    def forward(self, seq, noise_sigma=0.0):
        hidden = self.encoder(seq)
        B, L, H = hidden.shape
        write_len = min(self.num_slots, L - 3)
        slots = hidden[:, :write_len, :]
        if write_len < self.num_slots:
            pad = torch.zeros(B, self.num_slots - write_len, H)
            slots = torch.cat([slots, pad], dim=1)
        query = self.query_proj(hidden[:, -1, :])
        if noise_sigma > 0:
            query = query + torch.randn_like(query) * noise_sigma
        keys  = self.key_proj(slots)
        attn  = F.softmax(torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) / H**0.5, dim=-1)
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

    def forward(self, seq, noise_sigma=0.0):
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
            query_h = hidden[b, -1, :].detach()
            if noise_sigma > 0:
                query_h = query_h + torch.randn_like(query_h) * noise_sigma
            with torch.no_grad():
                ctx = mlp(query_h.unsqueeze(0)).squeeze(0)
            contexts.append(ctx)
        return self.output(torch.stack(contexts, dim=0))


def train_model(model, steps=STEPS):
    if isinstance(model, ParametricMemoryModel):
        opt = Adam(list(model.encoder.parameters()) +
                   list(model.base_mlp.parameters()) +
                   list(model.output.parameters()), lr=LR)
    else:
        opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, tgt = make_assoc_batch(BATCH)
        loss = F.cross_entropy(model(seq, noise_sigma=0.0), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, noise_sigma=0.0, n_batches=40):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_batches):
            seq, tgt = make_assoc_batch(BATCH)
            correct += (model(seq, noise_sigma=noise_sigma).argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp252NoisyKeyRetrieval(Experiment):
    experiment_id = "exp_25_2"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
                      num_pairs=NUM_PAIRS, memory_slots=MEMORY_SLOTS, steps=STEPS,
                      batch=BATCH, noise_sigmas=NOISE_SIGMAS)
        print("Training slot memory model...")
        slot_model = SlotMemoryModel()
        slot_model = train_model(slot_model)

        print("Training parametric memory model...")
        param_model = ParametricMemoryModel()
        param_model = train_model(param_model)

        slot_accs = {}; param_accs = {}
        for sigma in NOISE_SIGMAS:
            slot_accs[sigma]  = round(evaluate(slot_model,  sigma), 4)
            param_accs[sigma] = round(evaluate(param_model, sigma), 4)
            print(f"  sigma={sigma}: slot={slot_accs[sigma]:.3f}, param={param_accs[sigma]:.3f}")

        slot_clean  = slot_accs[0.0];  slot_noisy  = slot_accs[0.1]
        param_clean = param_accs[0.0]; param_noisy = param_accs[0.1]
        slot_deg  = (slot_clean  - slot_noisy)  / max(slot_clean,  1e-6)
        param_deg = (param_clean - param_noisy) / max(param_clean, 1e-6)

        metrics = dict(
            slot_accs_by_sigma=slot_accs,   param_accs_by_sigma=param_accs,
            slot_deg_at_sigma01=round(slot_deg, 4),
            param_deg_at_sigma01=round(param_deg, 4),
        )
        if slot_deg > 0.20 and param_deg < 0.10:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Slot deg={slot_deg:.3f}>0.20, Param deg={param_deg:.3f}<0.10."
        elif slot_deg < 0.10:
            outcome = OUTCOME_REFUTED
            notes   = f"Slot is noise-robust (deg={slot_deg:.3f}) — no architecture gap."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Slot deg={slot_deg:.3f}, Param deg={param_deg:.3f}. Partial support."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp252NoisyKeyRetrieval().execute()
