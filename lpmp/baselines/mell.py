
"""
Minimal-faithful MELL-style baseline in PyTorch.
Core idea (Matsuno & Murata, WWW'18): learn a per-layer vector s_l that modulates
shared node embeddings E. We implement a concise bilinear scoring:
    score_l(u,v) = < E[u] ⊙ s_l, E[v] >
with negative sampling over observed edges across layers.
This aligns with "learning a layer vector that captures layer connectivity" while
keeping training simple and dependency-free.

Reference: Matsuno & Murata (2018). MELL: Effective Embedding Method for Multiplex Networks.
"""
from typing import Dict, List, Tuple
import random
import torch
import torch.nn as nn

class MellScorer(nn.Module):
    def __init__(self, num_nodes: int, layers: List[str], dim: int = 128):
        super().__init__()
        self.E = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.E.weight)
        self.layer_vec = nn.ParameterDict({L: nn.Parameter(torch.ones(dim)) for L in layers})

    def score(self, L: str, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # u,v: (B,) long
        e_u = self.E(u)                      # (B,d)
        e_v = self.E(v)                      # (B,d)
        s = self.layer_vec[L]                # (d,)
        z = (e_u * s) * e_v                  # (B,d)
        return z.sum(-1)                     # (B,)

    def forward(self, L: str, pairs: torch.Tensor) -> torch.Tensor:
        # pairs: (B,2) [u,v]
        return self.score(L, pairs[:,0], pairs[:,1]).unsqueeze(-1)

@torch.no_grad()
def pairs_from_edges(edge_sets: Dict[str, set]) -> List[Tuple[int,int,str]]:
    out = []
    for L, S in edge_sets.items():
        for (u,v) in S:
            out.append((u,v,L))
            out.append((v,u,L))  # treat as directed for sampling richness
    return out

def make_ns_sampler(all_pairs: List[Tuple[int,int,str]], num_nodes: int):
    # Uniform negatives per layer
    by_layer = {}
    for u,v,L in all_pairs:
        by_layer.setdefault(L, set()).add((u,v))
    def sample_neg(L: str, B: int):
        # resample until non-edge
        neg = []
        S = by_layer[L]
        while len(neg) < B:
            u = random.randrange(num_nodes)
            v = random.randrange(num_nodes)
            if u==v: continue
            if (u,v) in S or (v,u) in S: continue
            neg.append((u,v))
        return torch.tensor(neg, dtype=torch.long)
    return sample_neg

def train_mell(edge_sets: Dict[str, set], num_nodes: int, layers: List[str], dim=128, epochs=5, batch_size=1024, lr=1e-3, device="cpu"):
    all_pos = pairs_from_edges(edge_sets)
    model = MellScorer(num_nodes, layers, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    neg_sampler = make_ns_sampler(all_pos, num_nodes)

    # Flatten pos into list per epoch
    import random
    for ep in range(1, epochs+1):
        random.shuffle(all_pos)
        total = 0.0; n=0
        for i in range(0, len(all_pos), batch_size):
            batch = all_pos[i:i+batch_size]
            Ls = [L for (u,v,L) in batch]
            pairs = torch.tensor([[u,v] for (u,v,_) in batch], dtype=torch.long, device=device)
            y_pos = torch.ones(len(batch), 1, device=device)

            # negatives per layer to keep distribution similar
            neg_pairs_list = []
            for L in Ls:
                neg_pairs_list.append(neg_sampler(L, 1)[0].tolist())
            neg_pairs = torch.tensor(neg_pairs_list, dtype=torch.long, device=device)
            y_neg = torch.zeros(len(batch), 1, device=device)

            logits_pos = torch.stack([model.score(L, pairs[j,0], pairs[j,1]) for j,L in enumerate(Ls)], dim=0).unsqueeze(-1)
            logits_neg = torch.stack([model.score(Ls[j], neg_pairs[j,0], neg_pairs[j,1]) for j in range(len(Ls))], dim=0).unsqueeze(-1)
            logits = torch.cat([logits_pos, logits_neg], dim=0)
            y = torch.cat([y_pos, y_neg], dim=0)

            opt.zero_grad(set_to_none=True)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            total += float(loss.item()); n+=1
        print(f"[MELL] epoch={ep} loss={total/max(1,n):.4f}", flush=True)
    return model

@torch.no_grad()
def mell_scores_for_layer(model: MellScorer, target_layer: str, pairs: List[Tuple[int,int]], device="cpu"):
    model.eval()
    P = torch.tensor(pairs, dtype=torch.long, device=device)
    logits = model(target_layer, P).squeeze(-1).cpu()
    return logits


def mell_prepare(edge_sets: Dict[str, set], num_nodes: int, layers: List[str],
                 dim: int = 128, lr: float = 1e-3, device: str = "cpu"):
    """
    Build model + optimizer, precompute positives and a negative sampler.
    Returns: (model, optimizer, neg_sampler, all_pos)
    """
    model = MellScorer(num_nodes, layers, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    all_pos = pairs_from_edges(edge_sets)
    neg_sampler = make_ns_sampler(all_pos, num_nodes)
    return model, opt, neg_sampler, all_pos

def mell_train_one_epoch(model: MellScorer,
                         opt: torch.optim.Optimizer,
                         neg_sampler,
                         all_pos: List[Tuple[int,int,str]],
                         batch_size: int = 1024,
                         device: str = "cpu") -> float:
    """
    One epoch of MeLL training over all_pos with 1 negative per positive (per layer).
    Returns the average BCE loss for this epoch.
    """
    import random
    bce = nn.BCEWithLogitsLoss()
    random.shuffle(all_pos)
    total = 0.0; n = 0
    for i in range(0, len(all_pos), batch_size):
        batch = all_pos[i:i+batch_size]
        Ls = [L for (u,v,L) in batch]
        pairs = torch.tensor([[u,v] for (u,v,_) in batch], dtype=torch.long, device=device)
        y_pos = torch.ones(len(batch), 1, device=device)

        # negatives per layer
        neg_pairs_list = []
        for L in Ls:
            neg_pairs_list.append(neg_sampler(L, 1)[0].tolist())
        neg_pairs = torch.tensor(neg_pairs_list, dtype=torch.long, device=device)
        y_neg = torch.zeros(len(batch), 1, device=device)

        logits_pos = torch.stack([model.score(L, pairs[j,0], pairs[j,1]) for j,L in enumerate(Ls)], dim=0).unsqueeze(-1)
        logits_neg = torch.stack([model.score(Ls[j], neg_pairs[j,0], neg_pairs[j,1]) for j in range(len(Ls))], dim=0).unsqueeze(-1)
        logits = torch.cat([logits_pos, logits_neg], dim=0)
        y = torch.cat([y_pos, y_neg], dim=0)

        opt.zero_grad(set_to_none=True)
        loss = bce(logits, y)
        loss.backward()
        opt.step()

        total += float(loss.item()); n += 1
    return total / max(1, n)