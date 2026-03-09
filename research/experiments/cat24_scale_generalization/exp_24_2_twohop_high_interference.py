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

experiment_id = "exp_24_2"
hypothesis = ("Compositional two-hop retrieval sustains >70% accuracy under 60% "
              "entity interference, up from 40% interference tested in exp_13_1.")

NUM_ENTITIES   = 32
NUM_ATTRIBUTES = 16
HIDDEN_DIM     = 64
MEMORY_SIZE    = 20
TRAIN_STEPS    = 2000
EVAL_BATCHES   = 200
BATCH_SIZE     = 32
LR             = 3e-4
INTERFERENCE_FRACS = [0.0, 0.40, 0.60]


def make_kb(seed=0):
    torch.manual_seed(seed)
    entity_attr = torch.randint(0, NUM_ATTRIBUTES, (NUM_ENTITIES,))
    colleagues  = torch.arange(NUM_ENTITIES).roll(-1)
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


class TwoHopRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.kb_enc     = KB_Encoder()
        self.hop2_net   = nn.Sequential(nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU())
        self.classifier = nn.Linear(HIDDEN_DIM, NUM_ATTRIBUTES)

    def retrieve(self, query, memory):
        sims = torch.einsum('bh,bmh->bm', query, memory) / HIDDEN_DIM**0.5
        w    = F.softmax(sims, dim=-1)
        return (w.unsqueeze(-1) * memory).sum(1)

    def forward_single(self, q_ents, memory):
        q = self.kb_enc.encode_query(q_ents)
        return self.classifier(self.retrieve(q, memory))

    def forward_two_hop(self, q_ents, memory):
        q1 = self.kb_enc.encode_query(q_ents)
        r1 = self.retrieve(q1, memory)
        q2 = self.hop2_net(torch.cat([q1, r1], dim=-1))
        r2 = self.retrieve(q2, memory)
        return self.classifier(r2)


def build_memory(batch_size, entity_attr, kb_enc, interference_frac=0.0):
    """Build memory; replace `interference_frac` fraction of attribute labels randomly."""
    B = batch_size
    M = min(MEMORY_SIZE, NUM_ENTITIES)
    all_entities = torch.arange(NUM_ENTITIES)
    all_attrs    = entity_attr.clone()
    if interference_frac > 0.0:
        n_corrupt = int(M * interference_frac)
        corrupt_idx = torch.randperm(M)[:n_corrupt]
        all_attrs[corrupt_idx] = torch.randint(0, NUM_ATTRIBUTES, (n_corrupt,))
    all_facts = kb_enc.encode_fact(all_entities[:M], all_attrs[:M])
    return all_facts.unsqueeze(0).expand(B, -1, -1).detach()


def train_model(steps=TRAIN_STEPS):
    entity_attr, colleagues = make_kb()
    model = TwoHopRetriever()
    opt   = Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        B   = BATCH_SIZE
        qe  = torch.randint(0, NUM_ENTITIES, (B,))
        t1  = entity_attr[qe]
        col = colleagues[qe]
        t2  = entity_attr[col]
        with torch.no_grad():
            mem = build_memory(B, entity_attr, model.kb_enc, interference_frac=0.0)
        l1  = F.cross_entropy(model.forward_single(qe, mem), t1)
        l2  = F.cross_entropy(model.forward_two_hop(qe, mem), t2)
        loss = l1 + l2
        opt.zero_grad(); loss.backward(); opt.step()
    return model, entity_attr, colleagues


def evaluate_at_interference(model, entity_attr, colleagues, interf_frac):
    model.eval()
    correct_2hop = total = 0
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            B   = BATCH_SIZE
            qe  = torch.randint(0, NUM_ENTITIES, (B,))
            col = colleagues[qe]
            t2  = entity_attr[col]
            mem = build_memory(B, entity_attr, model.kb_enc, interf_frac)
            correct_2hop += (model.forward_two_hop(qe, mem).argmax(-1) == t2).sum().item()
            total += B
    return correct_2hop / total


class Exp242TwoHopHighInterference(Experiment):
    experiment_id = "exp_24_2"
    hypothesis    = hypothesis

    def run(self) -> ExperimentResult:
        config = dict(num_entities=NUM_ENTITIES, num_attributes=NUM_ATTRIBUTES,
                      hidden_dim=HIDDEN_DIM, train_steps=TRAIN_STEPS,
                      interference_fracs=INTERFERENCE_FRACS)
        print("Training two-hop retrieval model (clean KB)...")
        model, entity_attr, colleagues = train_model()
        accs = {}
        for frac in INTERFERENCE_FRACS:
            acc = evaluate_at_interference(model, entity_attr, colleagues, frac)
            accs[frac] = round(acc, 4)
            print(f"  interference={frac:.2f}: two_hop_acc={acc:.3f}")

        acc_at_60 = accs[0.60]
        acc_at_0  = accs[0.0]
        retention = acc_at_60 / max(acc_at_0, 1e-6)

        metrics = dict(
            two_hop_acc_by_interference=accs,
            retention_at_60pct_interf=round(retention, 4),
        )
        if acc_at_60 > 0.70:
            outcome = OUTCOME_SUPPORTED
            notes   = f"Two-hop acc at 60% interference={acc_at_60:.3f}>0.70."
        elif acc_at_60 < 0.30:
            outcome = OUTCOME_REFUTED
            notes   = f"Two-hop collapses at 60% interference: acc={acc_at_60:.3f}<0.30."
        else:
            outcome = OUTCOME_INCONCLUSIVE
            notes   = f"Two-hop acc at 60% interference={acc_at_60:.3f}, between 0.30 and 0.70."
        return self.result(outcome, metrics, notes, config)


if __name__ == "__main__":
    Exp242TwoHopHighInterference().execute()
