
"""
RMNE baseline (ICML 2022): role-modified random walks + Skip-gram (negative sampling).
We implement:
  - Role discovery via simple WL features (2 iterations) with equality-based role assignment.
  - Role-modified random walk with parameters r (neighbor sampling) and t (role sampling),
    and cross-layer same-individual transitions weighted by log-degree, per paper eqs.
  - Skip-gram with negative sampling to get node embeddings.
"""
from typing import Dict, List, Tuple, Iterable
import math, random
import numpy as np
import torch
import torch.nn as nn

def build_adj_lists(num_nodes: int, edges: List[Tuple[int,int]]):
    adj = [[] for _ in range(num_nodes)]
    deg = [0]*num_nodes
    for u,v in edges:
        if u==v: continue
        adj[u].append(v); adj[v].append(u)
        deg[u]+=1; deg[v]+=1
    return adj, np.array(deg, dtype=np.int64)

def wl_roles_per_layer(num_nodes: int, edge_sets: Dict[str, set], iters: int = 2):
    # Returns dict: (layer -> List[int role_id per node])
    roles = {}
    for L, S in edge_sets.items():
        adj, _ = build_adj_lists(num_nodes, list(S))
        labels = [str(len(adj[i])) for i in range(num_nodes)]  # start with degree
        for _ in range(iters):
            new = []
            for i in range(num_nodes):
                neigh = sorted(labels[j] for j in adj[i])
                new.append(labels[i] + "|" + ",".join(neigh))
            # compress to ids
            uniq = {lab:i for i,lab in enumerate(sorted(set(new)))}
            labels = [f"{uniq[s]}" for s in new]
        # map to integer role ids
        uniq = {lab:i for i,lab in enumerate(sorted(set(labels)))}
        roles[L] = [uniq[s] for s in labels]
    return roles

def role_members(roles: Dict[str, List[int]]):
    # For each layer, map role -> set(nodes)
    per_layer = {}
    for L, arr in roles.items():
        d = {}
        for n, r in enumerate(arr):
            d.setdefault(r, set()).add(n)
        per_layer[L] = d
    return per_layer

def degrees_per_layer(num_nodes: int, edge_sets: Dict[str, set]):
    degs = {}
    for L, S in edge_sets.items():
        _, deg = build_adj_lists(num_nodes, list(S))
        degs[L] = deg
    return degs

class SkipGramNS(nn.Module):
    def __init__(self, num_nodes: int, dim: int = 128):
        super().__init__()
        self.in_emb = nn.Embedding(num_nodes, dim)
        self.out_emb = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.in_emb.weight)
        nn.init.xavier_uniform_(self.out_emb.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor):
        v = self.in_emb(center)     # (B,d)
        u = self.out_emb(context)   # (B,d)
        return (v*u).sum(-1)        # logits

def precompute_transition(edge_sets: Dict[str, set], roles: Dict[str, List[int]], r: float, t: float):
    # Build maps for neighbors, role members per layer
    neighbors = {}
    rolemem = {}
    for L, S in edge_sets.items():
        adj,_ = build_adj_lists(max(max(u,v) for (u,v) in S)+1 if S else 0, list(S))
        neighbors[L] = [set(vs) for vs in adj]
        # role members except self
        R = roles[L]
        rm = role_members({L: R})[L]
        rolemem[L] = {i: (rm[R[i]] - {i}) for i in range(len(R))}
    return neighbors, rolemem

def rmne_walks(num_nodes: int, edge_sets: Dict[str, set], roles: Dict[str, List[int]], K: int = 10, Le: int = 40, r: float = 5.0, t: float = 5.0, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # per-layer neighbor sets & role member sets
    neighbors = {}
    rolemem = {}
    degs = degrees_per_layer(num_nodes, edge_sets)

    for L, S in edge_sets.items():
        adj,_ = build_adj_lists(num_nodes, list(S))
        neighbors[L] = [set(vs) for vs in adj]
        R = roles[L]
        rm = role_members({L: R})[L]
        rolemem[L] = {i: (rm[R[i]] - {i}) for i in range(num_nodes)}

    walks = []
    layers = list(edge_sets.keys())

    def next_step(cur_node: int, cur_layer: str):
        # Candidate set = neighbors U role members at cur_layer U {same node in other layers}
        cand = []
        weights = []
        # neighbors in same layer
        Ns = neighbors[cur_layer][cur_node]
        for v in Ns:
            cand.append((v, cur_layer))
            weights.append(1.0/max(1.0, r))  # delta_r
        # role members (same layer)
        for v in rolemem[cur_layer][cur_node]:
            cand.append((v, cur_layer))
            weights.append(1.0/max(1.0, t))  # beta_t

        # cross-layer same individual mapping v=x
        for L2 in layers:
            if L2 == cur_layer: continue
            # stay at same node index on other layer
            deg = degs[L2][cur_node] if cur_node < len(degs[L2]) else 1
            w = (1.0/max(1.0, t)) * math.log(max(1, int(deg)))
            cand.append((cur_node, L2))
            weights.append(w if w>0 else 1e-6)

        if not cand:
            return cur_node, cur_layer

        s = sum(weights)
        probs = [w/s for w in weights]
        idx = np.random.choice(len(cand), p=probs)
        return cand[idx]

    for start in range(num_nodes):
        for _ in range(K):
            # start layer random
            L0 = random.choice(layers)
            walk = [(start, L0)]
            while len(walk) < Le:
                u, L = walk[-1]
                v, L2 = next_step(u, L)
                walk.append((v, L2))
            # project back to node ids only
            walks.append([u for (u, L) in walk])
    return walks

def build_skipgram_dataset(walks: List[List[int]], window: int = 5):
    pairs = []
    for w in walks:
        for i,u in enumerate(w):
            s = max(0, i-window); e = min(len(w), i+window+1)
            for j in range(s, e):
                if j==i: continue
                pairs.append((u, w[j]))
    return pairs

def train_rmne(num_nodes: int, edge_sets: Dict[str, set], dim=128, K=10, Le=40, window=5, epochs=2, batch_size=2048, lr=1e-3, r=5.0, t=5.0, seed=42, device="cpu"):
    roles = wl_roles_per_layer(num_nodes, edge_sets, iters=2)
    walks = rmne_walks(num_nodes, edge_sets, roles, K=K, Le=Le, r=r, t=t, seed=seed)
    pairs = build_skipgram_dataset(walks, window=window)

    model = SkipGramNS(num_nodes, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)

    def neg_samples(B):
        return torch.from_numpy(rng.integers(0, num_nodes, size=B)).long()

    import math
    for ep in range(1, epochs+1):
        random.shuffle(pairs)
        total=0.0; n=0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            c = torch.tensor([u for (u,v) in batch], dtype=torch.long, device=device)
            p = torch.tensor([v for (u,v) in batch], dtype=torch.long, device=device)
            logits_pos = model(c, p).unsqueeze(-1)
            # one negative per positive
            n_ctx = neg_samples(len(batch)).to(device)
            logits_neg = model(c, n_ctx).unsqueeze(-1)
            logits = torch.cat([logits_pos, logits_neg], dim=0)
            y = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0)

            opt.zero_grad(set_to_none=True)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            total += float(loss.item()); n+=1
        print(f"[RMNE] epoch={ep} loss={total/max(1,n):.4f}", flush=True)

    # Final embedding as input embedding (common choice)
    return model.in_emb.weight.data.detach().cpu()

@torch.no_grad()
def rmne_scores(emb: torch.Tensor, pairs: List[Tuple[int,int]]):
    # Dot-product scorer
    u = torch.tensor([a for a,b in pairs], dtype=torch.long)
    v = torch.tensor([b for a,b in pairs], dtype=torch.long)
    s = (emb[u] * emb[v]).sum(-1)
    return s



def rmne_prepare(num_nodes: int, edge_sets: Dict[str, set], dim=128, K=10, Le=40, window=5,
                 lr=1e-3, r=5.0, t=5.0, seed=42, device="cpu"):
    """
    Build model + optimizer, precompute role-modified walks and skip-gram dataset.
    Returns: (model, optimizer, pairs, rng)
    """
    roles = wl_roles_per_layer(num_nodes, edge_sets, iters=2)
    walks = rmne_walks(num_nodes, edge_sets, roles, K=K, Le=Le, r=r, t=t, seed=seed)
    pairs = build_skipgram_dataset(walks, window=window)
    model = SkipGramNS(num_nodes, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    return model, opt, pairs, rng

def rmne_train_one_epoch(model: SkipGramNS,
                         opt: torch.optim.Optimizer,
                         pairs: List[Tuple[int,int]],
                         num_nodes: int,
                         rng: np.random.Generator,
                         batch_size: int = 2048,
                         device: str = "cpu") -> float:
    """
    One epoch of RMNE Skip-gram NS over precomputed pairs.
    Returns the average BCE loss for this epoch.
    """
    import random
    bce = nn.BCEWithLogitsLoss()
    random.shuffle(pairs)
    total = 0.0; n = 0

    def neg_samples(B):
        return torch.from_numpy(rng.integers(0, num_nodes, size=B)).long().to(device)

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        c = torch.tensor([u for (u,v) in batch], dtype=torch.long, device=device)
        p = torch.tensor([v for (u,v) in batch], dtype=torch.long, device=device)
        logits_pos = model(c, p).unsqueeze(-1)
        n_ctx = neg_samples(len(batch))
        logits_neg = model(c, n_ctx).unsqueeze(-1)

        logits = torch.cat([logits_pos, logits_neg], dim=0)
        y = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0)

        opt.zero_grad(set_to_none=True)
        loss = bce(logits, y)
        loss.backward()
        opt.step()

        total += float(loss.item()); n += 1
    return total / max(1, n)