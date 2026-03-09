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

experiment_id = "exp_24_4"
hypothesis = ("Four-hop compositional chains are infeasible at HIDDEN_DIM=64: "
              "accuracy drops >50% vs two-hop even with hop-by-hop training curriculum.")

NUM_ENTITIES   = 32
NUM_ATTRIBUTES = 16
HIDDEN_DIM     = 64
MEMORY_SIZE    = 20
TRAIN_STEPS    = 2000
EVAL_BATCHES   = 200
BATCH_SIZE     = 32
LR             = 3e-4


def make_kb(seed=0):
    torch.manual_seed(seed)
    entity_attr = torch.randint(0, NUM_ATTRIBUTES, (NUM_ENTITIES,))
    colleagues  = torch.arange(NUM_ENTITIES).roll(-1)   # entity i → i+1
    return entity_attr, colleagues


class KB_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.entity_emb = nn.Embedding(NUM_ENTITIES, HIDDEN_DIM)
        self.attr_emb   = nn.Embedding(NUM_ATTRIBUTES, HIDDEN_DIM)
        self.fact_proj  = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)

    def encode_fact(self, ent, attr):
        return self.fact_proj(torch.cat([self.entity_emb(ent), self.attr_emb(attr)], dim=-1))

    def encode_query(self, ent):
        return self.entity_emb(ent)


class MultiHopRetriever(nn.Module):
    def __init__(self, n_hops=2):
        super().__init__()
        self.kb_enc    = KB_Encoder()
        self.hop_nets  = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU())
            for _ in range(n_hops - 1)
        ])
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_ATTRIBUTES)
        self.n_hops     = n_hops

    def retrieve(self, query, memory):
        sims = torch.einsum('bh,bmh->bm', query, memory) / HIDDEN_DIM**0.5
        w    = F.softmax(sims, dim=-1)
        return (w.unsqueeze(-1) * memory).sum(1)

    def forward(self, q_ents, memory):
        q = self.kb_enc.encode_query(q_ents)
        r = self.retrieve(q, memory)
        for hop_net in self.hop_nets:
            q = hop_net(torch.cat([q, r], dim=-1))
            r = self.retrieve(q, memory)
        return self.classifier(r)


def build_memory(batch_size, entity_attr, kb_enc):
    M = min(MEMORY_SIZE, NUM_ENTITIES)
    all_ents  = torch.arange(NUM_ENTITIES)
    all_attrs = entity_attr
    all_facts = kb_enc.encode_fact(all_ents[:M], all_attrs[:M])
    return all_facts.unsqueeze(0).expand(batch_size, -1, -1).detach()


def get_hop_target(q_ents, entity_attr, colleagues, n_hops):
    """Follow the colleague chain n_hops times and return the final entity's attribute."""
    curr = q_ents.clone()
    for _ in range(n_hops):
        curr = colleagues[curr]
    return entity_attr[curr]


def train_model(n_hops, steps=TRAIN_STEPS):
    entity_attr, colleagues = make_kb()
    model = MultiHopRetriever(n_hops=n_hops)
    opt   = Adam(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        B  = BATCH_SIZE
        qe = torch.randint(0, NUM_ENTITIES, (B,))
        tgt = get_hop_target(qe, entity_attr, colleagues, n_hops)
        with torch.no_grad():
            mem = build_memory(B, entity_attr, model.kb_enc)
        logits = model(qe, mem)
        loss   = F.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, entity_attr, colleagues


def evaluate(model, entity_attr, colleagues, n_hops):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            B  = BATCH_SIZE
            qe = torch.randint(0, NUM_ENTITIES, (B,))
            tgt = get_hop_target(qe, entity_attr, colleagues, n_hops)
            mem = build_memory(B, entity_attr, model.kb_enc)
            correct += (model(qe, mem).argmax(-1) == tgt).sum().item()
            total   += B
    return correct / total


class Exp244FourHopInfeasible(Experiment):
    experiment_id = "exp_24_4"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(num_entities=NUM_ENTITIES, num_attributes=NUM_ATTRIBUTES,
                      hidden_dim=HIDDEN_DIM, train_steps=TRAIN_STEPS)
        random_baseline = 1.0 / NUM_ATTRIBUTES

        print("Training 2-hop retriever...")
        model_2hop, ea, col = train_model(n_hops=2)
        acc_2hop = evaluate(model_2hop, ea, col, n_hops=2)
        print(f"  2-hop acc={acc_2hop:.3f}")

        print("Training 4-hop retriever (curriculum: same kb)...")
        model_4hop, ea4, col4 = train_model(n_hops=4)
        acc_4hop = evaluate(model_4hop, ea4, col4, n_hops=4)
        print(f"  4-hop acc={acc_4hop:.3f}")

        drop = (acc_2hop - acc_4hop) / max(acc_2hop, 1e-6)

        metrics = dict(
            acc_2hop=round(acc_2hop, 4), acc_4hop=round(acc_4hop, 4),
            acc_drop_fraction=round(drop, 4), random_baseline=round(random_baseline, 4),
        )
        if drop > 0.50:
            outcome = OUTCOME_SUPPORTED
            notes   = f"4-hop drops {drop*100:.1f}%>50% vs 2-hop ({acc_2hop:.3f} vs {acc_4hop:.3f})."
        elif drop < 0.20:
            outcome = OUTCOME_REFUTED
            notes   = f"4-hop achievable: drop={drop*100:.1f}%<20% vs 2-hop."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Drop={drop*100:.1f}%, between 20% and 50%."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp244FourHopInfeasible().execute()
