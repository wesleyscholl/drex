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

experiment_id = "exp_25_1"
hypothesis = ("Multi-domain retrieval benchmark (facts + patterns + temporal chains "
              "in same sequence): any memory architecture achieves <70% joint accuracy "
              "without domain-specific slots.")

VOCAB_SIZE    = 64
HIDDEN_DIM    = 64
SEQ_LEN       = 48
NUM_PAIRS     = 3     # 3 pairs per domain × 3 domains
MEMORY_SLOTS  = 8
STEPS         = 2000
BATCH         = 32
LR            = 3e-4
NUM_DOMAINS   = 3


def make_multi_domain_batch(batch_size):
    """Sequence contains 3 domains of KV facts interleaved. Query from one domain."""
    seq    = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    target = torch.zeros(batch_size, dtype=torch.long)
    query_domain = torch.randint(0, NUM_DOMAINS, (batch_size,))

    for b in range(batch_size):
        dom  = query_domain[b].item()
        all_keys = []; all_vals = []
        # 3 domains: each with keys in different ranges
        for d in range(NUM_DOMAINS):
            k_low = 4 + d * 10; k_high = k_low + 10
            dk = torch.randint(k_low, k_high, (NUM_PAIRS * 3,)).unique()[:NUM_PAIRS]
            while len(dk) < NUM_PAIRS:
                dk = torch.cat([dk, torch.randint(k_low, k_high, (1,))])[:NUM_PAIRS]
            dv = torch.randint(VOCAB_SIZE // 2, VOCAB_SIZE, (NUM_PAIRS,))
            all_keys.append(dk); all_vals.append(dv)

        pos = 0
        for d in range(NUM_DOMAINS):
            for i in range(NUM_PAIRS):
                if pos + 1 < SEQ_LEN - 3:
                    seq[b, pos] = all_keys[d][i]
                    seq[b, pos + 1] = all_vals[d][i]
                    pos += 2

        for p in range(pos, SEQ_LEN - 3):
            seq[b, p] = 3
        qi = torch.randint(0, NUM_PAIRS, (1,)).item()
        seq[b, SEQ_LEN - 3] = 2
        seq[b, SEQ_LEN - 2] = all_keys[dom][qi]
        seq[b, SEQ_LEN - 1] = 0
        target[b] = all_vals[dom][qi]
    return seq, target, query_domain


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.ff    = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2), nn.ReLU(),
                                   nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM))
        self.norm  = nn.LayerNorm(HIDDEN_DIM)

    def forward(self, x):
        h = self.embed(x); return self.norm(h + self.ff(h))


# ── Model A: Generic slots (no domain knowledge) ──────────────────────────────

class GenericSlotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.gate    = nn.Linear(HIDDEN_DIM, 1)
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        _, idx = self.gate(h).squeeze(-1).topk(min(MEMORY_SLOTS, L), dim=-1)
        memory = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))
        q    = self.q_proj(h[:, -1, :])
        attn = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return self.output((attn.unsqueeze(-1) * memory).sum(1))


# ── Model B: Domain-specific slots ────────────────────────────────────────────

class DomainSlotModel(nn.Module):
    """Maintains separate slot sets per domain; selects domain via query marker."""
    def __init__(self, num_domains=NUM_DOMAINS, slots_per_domain=None):
        super().__init__()
        if slots_per_domain is None:
            slots_per_domain = max(1, MEMORY_SLOTS // num_domains)
        self.encoder = Encoder()
        self.gate    = nn.ModuleList([nn.Linear(HIDDEN_DIM, 1) for _ in range(num_domains)])
        self.domain_predictor = nn.Linear(HIDDEN_DIM, num_domains)
        self.q_proj  = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.output  = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.num_domains = num_domains
        self.sld = slots_per_domain

    def forward(self, seq):
        h = self.encoder(seq)
        B, L, H = h.shape
        # Predict domain from last token (query marker region)
        dom_logits = self.domain_predictor(h[:, -1, :])    # (B, D)
        dom_probs  = torch.softmax(dom_logits, dim=-1)     # (B, D)

        # Build memory: soft mixture of domain-specific slot selections
        all_slots = []
        for d in range(self.num_domains):
            g = torch.sigmoid(self.gate[d](h)).squeeze(-1)  # (B, L)
            k = min(self.sld, L)
            _, idx = g.topk(k, dim=-1)
            slots_d = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))  # (B, k, H)
            weight_d = dom_probs[:, d].unsqueeze(-1).unsqueeze(-1)      # (B, 1, 1)
            all_slots.append(slots_d * weight_d)
        memory = torch.cat(all_slots, dim=1)   # (B, D*k, H)

        q    = self.q_proj(h[:, -1, :])
        attn = F.softmax(torch.bmm(memory, q.unsqueeze(-1)).squeeze(-1) / H**0.5, dim=-1)
        return self.output((attn.unsqueeze(-1) * memory).sum(1)), dom_logits


def train_model(model, steps=STEPS, with_domain_loss=False):
    opt = Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(steps):
        seq, tgt, dom = make_multi_domain_batch(BATCH)
        if with_domain_loss:
            logits, dom_logits = model(seq)
            loss = F.cross_entropy(logits, tgt) + 0.3 * F.cross_entropy(dom_logits, dom)
        else:
            logits = model(seq)
            loss   = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, n_eval=50, with_domain_logits=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n_eval):
            seq, tgt, _ = make_multi_domain_batch(BATCH)
            if with_domain_logits:
                logits, _ = model(seq)
            else:
                logits = model(seq)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total   += tgt.size(0)
    return correct / total


class Exp251MultiDomainRetrieval(Experiment):
    experiment_id = "exp_25_1"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                      seq_len=SEQ_LEN, num_pairs_per_domain=NUM_PAIRS,
                      num_domains=NUM_DOMAINS, memory_slots=MEMORY_SLOTS,
                      steps=STEPS, batch=BATCH)
        random_baseline = 1.0 / (VOCAB_SIZE // 2)

        print("Training generic slot model (A)...")
        model_A = GenericSlotModel()
        model_A = train_model(model_A, with_domain_loss=False)
        acc_A = evaluate(model_A)
        print(f"  A (generic): acc={acc_A:.3f}")

        print("Training domain-specific slot model (B)...")
        model_B = DomainSlotModel()
        model_B = train_model(model_B, with_domain_loss=True)
        acc_B = evaluate(model_B, with_domain_logits=True)
        print(f"  B (domain-specific): acc={acc_B:.3f}")

        metrics = dict(
            acc_generic=round(acc_A, 4), acc_domain_specific=round(acc_B, 4),
            random_baseline=round(random_baseline, 4),
            generic_below_70pct=(acc_A < 0.70),
        )
        if acc_A < 0.70:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Generic acc={acc_A:.3f}<0.70. Domain-specific: {acc_B:.3f}."
        elif acc_A >= 0.70 and acc_B >= 0.70:
            outcome = OUTCOME_REFUTED
            notes   = f"Generic already achieves {acc_A:.3f}≥0.70. No domain bottleneck."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Generic={acc_A:.3f}, domain-specific={acc_B:.3f}."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp251MultiDomainRetrieval().execute()
