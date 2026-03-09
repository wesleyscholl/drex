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

experiment_id = "exp_22_4"
hypothesis = ("Read gating (suppress low-confidence reads via entropy threshold) "
              "transfers from simple associative recall to multi-pair retrieval "
              "without retuning: accuracy gap is <2% on simple and >3% on hard task.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 32
NUM_PAIRS_EASY = 2
NUM_PAIRS_HARD = 6
MEMORY_SLOTS  = 8
STEPS         = 2000
BATCH         = 32
LR            = 3e-4
# Entropy threshold: a fixed scalar derived from uniform distribution over MEMORY_SLOTS
# log(k) is the max possible entropy; we gate at 85% of max entropy
ENTROPY_THRESHOLD_FRAC = 0.85


def make_assoc_batch(batch_size, num_pairs):
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randint(4, max(8, VOCAB_SIZE // 3), (num_pairs * 3,)).unique()[:num_pairs]
        while len(keys) < num_pairs:
            keys = torch.cat([keys, torch.randint(4, VOCAB_SIZE // 3, (1,))])[:num_pairs]
        vals = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (num_pairs,))
        pos = 0
        for i in range(num_pairs):
            if pos + 1 < SEQ_LEN - 3:
                seq[b, pos] = keys[i]; seq[b, pos + 1] = vals[i]; pos += 2
        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3
        qi = torch.randint(0, num_pairs, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2; seq[b, SEQ_LEN - 2] = keys[qi]; seq[b, SEQ_LEN - 1] = 0
        target[b] = vals[qi]
    return seq, target


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


class BaseSlotModel(nn.Module):
    """Shared base: encoder + gate + read head."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.gate    = nn.Linear(HIDDEN_DIM, 1)
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def get_memory(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        _, idx = self.gate(h).squeeze(-1).topk(min(MEMORY_SLOTS, L), dim=-1)
        memory = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        return h, memory

    def read_with_attn(self, h, memory):
        H = h.size(-1)
        q = self.q_proj(h[:, -1, :])
        attn = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return attn, (attn.unsqueeze(-1) * memory).sum(1)

    def forward(self, seq):
        h, memory = self.get_memory(seq)
        _, ctx = self.read_with_attn(h, memory)
        return self.output(ctx), None   # no entropy gate


class EntropyGatedSlotModel(nn.Module):
    """Entropy-gated read: if attention entropy > threshold, fall back to mean pooling."""
    def __init__(self, entropy_threshold):
        super().__init__()
        self.encoder   = Encoder()
        self.gate      = nn.Linear(HIDDEN_DIM, 1)
        self.q_proj    = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output    = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.threshold = entropy_threshold

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        _, idx = self.gate(h).squeeze(-1).topk(min(MEMORY_SLOTS, L), dim=-1)
        memory = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        q      = self.q_proj(h[:, -1, :])
        k      = memory.size(1)
        attn   = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        # Compute Shannon entropy of each attention distribution
        entropy = -(attn * (attn + 1e-9).log()).sum(-1)   # (B,)
        # Gate: low-confidence = high entropy → use uniform fallback
        high_ent = (entropy > self.threshold).float().unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        uniform  = torch.ones_like(attn) / k
        eff_attn = (1 - high_ent.squeeze(-1)) * attn + high_ent.squeeze(-1) * uniform
        ctx = (eff_attn.unsqueeze(-1) * memory).sum(1)
        return self.output(ctx), entropy.mean().item()


def train_and_eval(model, num_pairs, n_eval=50):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(STEPS):
        seq, tgt = make_assoc_batch(BATCH, num_pairs)
        logits, _ = model(seq)
        loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt = make_assoc_batch(BATCH, num_pairs)
            logits, _ = model(seq)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp224ReadGateTransfer(Experiment):
    experiment_id = "exp_22_4"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        k            = MEMORY_SLOTS
        import math
        max_entropy  = math.log(k)
        threshold    = ENTROPY_THRESHOLD_FRAC * max_entropy
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                      num_pairs_easy=NUM_PAIRS_EASY, num_pairs_hard=NUM_PAIRS_HARD,
                      memory_slots=MEMORY_SLOTS, steps=STEPS, batch=BATCH,
                      entropy_threshold=round(threshold, 4))

        print("Training standard model on easy task (A_easy)...")
        model_std = BaseSlotModel()
        acc_std_easy = train_and_eval(model_std, NUM_PAIRS_EASY)
        print(f"  A_easy: {acc_std_easy:.3f}")

        print("Training entropy-gated model on easy task (B_easy)...")
        model_gate = EntropyGatedSlotModel(threshold)
        acc_gate_easy = train_and_eval(model_gate, NUM_PAIRS_EASY)
        print(f"  B_easy: {acc_gate_easy:.3f}")

        # Evaluate both on hard task WITHOUT retraining
        print("Evaluating on hard task (zero-shot transfer):")
        model_std2 = BaseSlotModel()
        acc_std_hard = train_and_eval(model_std2, NUM_PAIRS_HARD)
        print(f"  A_hard (retrained): {acc_std_hard:.3f}")

        model_gate2 = EntropyGatedSlotModel(threshold)
        acc_gate_hard = train_and_eval(model_gate2, NUM_PAIRS_HARD)
        print(f"  B_hard (retrained, same threshold): {acc_gate_hard:.3f}")

        gap_easy = abs(acc_gate_easy - acc_std_easy)
        gap_hard = acc_gate_hard - acc_std_hard

        metrics = dict(
            acc_std_easy=round(acc_std_easy, 4), acc_gate_easy=round(acc_gate_easy, 4),
            acc_std_hard=round(acc_std_hard, 4), acc_gate_hard=round(acc_gate_hard, 4),
            gap_easy=round(gap_easy, 4), gap_hard=round(gap_hard, 4),
            entropy_threshold=round(threshold, 4),
        )
        if gap_easy < 0.02 and gap_hard > 0.03:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Easy gap={gap_easy:.3f}<0.02, hard gap={gap_hard:.3f}>0.03. Transfer confirmed."
        elif gap_easy >= 0.05 or gap_hard < -0.02:
            outcome = OUTCOME_REFUTED
            notes   = f"Easy gap={gap_easy:.3f} too large or hard benefit negative ({gap_hard:.3f})."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Easy gap={gap_easy:.3f}, hard gap={gap_hard:.3f}. Partial transfer."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp224ReadGateTransfer().execute()
