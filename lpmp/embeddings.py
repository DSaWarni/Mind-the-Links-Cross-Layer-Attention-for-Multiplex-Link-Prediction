
from typing import Dict, List, Tuple
import math, random
import numpy as np
import torch
import torch.nn as nn

# ---- Utilities over edge_sets ----

def build_adj_lists(num_nodes: int, edges: List[Tuple[int,int]]):
    adj = [[] for _ in range(num_nodes)]
    for u,v in edges:
        if u==v: continue
        adj[u].append(v); adj[v].append(u)
    return adj

# K-core numbers (per layer)
def core_numbers(num_nodes: int, edges: List[Tuple[int,int]]):
    adj = [set() for _ in range(num_nodes)]
    for u,v in edges:
        if u==v: continue
        adj[u].add(v); adj[v].add(u)
    deg = [len(adj[i]) for i in range(num_nodes)]
    import heapq
    heap = [(deg[i], i) for i in range(num_nodes)]
    heapq.heapify(heap)
    removed = [False]*num_nodes
    core = [0]*num_nodes
    while heap:
        d, i = heapq.heappop(heap)
        if removed[i]: continue
        core[i] = d
        removed[i] = True
        for j in list(adj[i]):
            adj[j].discard(i)
            if not removed[j]:
                deg[j] -= 1
                heapq.heappush(heap, (deg[j], j))
    return core

# ---- Skip-gram with negative sampling ----
class SkipGramNS(nn.Module):
    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.in_emb = nn.Embedding(num_nodes, dim)
        self.out_emb = nn.Embedding(num_nodes, dim)
        nn.init.xavier_uniform_(self.in_emb.weight)
        nn.init.xavier_uniform_(self.out_emb.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor):
        v = self.in_emb(center)     # (B,d)
        u = self.out_emb(context)   # (B,d)
        return (v*u).sum(-1)        # (B,)

def train_skipgram_from_walks(
    walks: List[List[int]], num_nodes: int, dim: int = 128,
    window: int = 10, epochs: int = 1, batch_size: int = 8192, lr: float = 1e-3, device: str = "cpu",
    neg_k: int = 1
) -> torch.Tensor:
    # Build (center, context) pairs
    pairs = []
    for w in walks:
        for i,u in enumerate(w):
            s = max(0, i-window); e = min(len(w), i+window+1)
            for j in range(s, e):
                if j==i: continue
                pairs.append((u, w[j]))
    if not pairs:
        return torch.zeros((num_nodes, dim), dtype=torch.float32)

    model = SkipGramNS(num_nodes, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(123)

    def neg_samples(B):
        return torch.from_numpy(rng.integers(0, num_nodes, size=B)).long()

    for ep in range(epochs):
        random.shuffle(pairs)
        total=0.0; n=0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            c = torch.tensor([a for a,b in batch], dtype=torch.long, device=device)
            p = torch.tensor([b for a,b in batch], dtype=torch.long, device=device)

            logits_pos = model(c, p).unsqueeze(-1)
            # k negatives per positive
            cn = c.repeat_interleave(neg_k)
            n_ctx = neg_samples(len(batch)*neg_k).to(device)
            logits_neg = model(cn, n_ctx).unsqueeze(-1)

            logits = torch.cat([logits_pos, logits_neg], dim=0)
            y = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0)

            opt.zero_grad(set_to_none=True)
            loss = bce(logits, y)
            loss.backward()
            opt.step()
            total += float(loss.item()); n+=1
        print(f"[SGNS] epoch={ep+1} loss={total/max(1,n):.4f}", flush=True)
    return model.in_emb.weight.data.detach().cpu()

# ---- Node2Vec walks (per layer) ----
def node2vec_walks(adj: List[List[int]], p: float, q: float, walk_length: int, walks_per_node: int, seed: int=42) -> List[List[int]]:
    random.seed(seed)
    walks = []
    N = len(adj)
    for start in range(N):
        if not adj[start]:
            continue
        for _ in range(walks_per_node):
            walk = [start]
            # pick first step uniformly
            nb = adj[start]
            cur = start
            if not nb: continue
            nxt = random.choice(nb)
            walk.append(nxt)
            prev = start; cur = nxt
            while len(walk) < walk_length:
                nbrs = adj[cur]
                if not nbrs: break
                probs = []
                for x in nbrs:
                    if x == prev:        # return
                        probs.append(1.0/p)
                    elif x in adj[prev]: # BFS
                        probs.append(1.0)
                    else:                # DFS
                        probs.append(1.0/q)
                s = sum(probs); probs = [w/s for w in probs]
                idx = np.random.choice(len(nbrs), p=probs)
                nxt = nbrs[idx]
                walk.append(nxt)
                prev, cur = cur, nxt
            walks.append(walk)
    return walks

# ---- Core-biased walks ("core2vec-inspired") ----
def core_biased_walks(adj: List[List[int]], cores: List[int], walk_length: int, walks_per_node: int, temperature: float=1.0, seed: int=42):
    """
    Not a 1:1 reproduction of core2vec, but uses core similarity to bias transitions.
    Transition weight from u->v ∝ exp( -|core[u]-core[v]| / T ).
    """
    random.seed(seed)
    walks = []
    N = len(adj)
    for start in range(N):
        if not adj[start]:
            continue
        for _ in range(walks_per_node):
            walk = [start]
            cur = start
            while len(walk) < walk_length:
                nbrs = adj[cur]
                if not nbrs: break
                weights = []
                for v in nbrs:
                    w = math.exp(-abs(cores[cur]-cores[v]) / max(1e-6, temperature))
                    weights.append(w)
                s = sum(weights); probs = [w/s for w in weights]
                idx = np.random.choice(len(nbrs), p=probs)
                nxt = nbrs[idx]
                walk.append(nxt)
                cur = nxt
            walks.append(walk)
    return walks

# ---- Factory ----
def build_layer_embeddings(
    num_nodes: int,
    edge_sets: Dict[str, set],
    backend: str = "node2vec",
    dim: int = 128,
    # node2vec params
    n2v_p: float = 1.0,
    n2v_q: float = 1.0,
    n2v_walk_len: int = 50,
    n2v_walks_per_node: int = 10,
    sgns_window: int = 10,
    sgns_epochs: int = 1,
    sgns_batch: int = 8192,
    sgns_lr: float = 1e-3,
    sgns_neg_k: int = 1,
    # corewalk params
    core_temp: float = 1.0,
    core_walk_len: int = 40,
    core_walks_per_node: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict layer_name -> (num_nodes, dim) float32 tensor.
    backends: 'node2vec', 'corewalk', 'random'
    """
    out: Dict[str, torch.Tensor] = {}
    for L, S in edge_sets.items():
        edges = list(S)
        adj = build_adj_lists(num_nodes, edges)
        if backend == "random":
            E = torch.randn((num_nodes, dim), dtype=torch.float32)
            #E = torch.randn((num_nodes, dim), dtype=torch.float32, device=device).cpu()
            out[L] = E
            continue

        if backend == "node2vec":
            walks = node2vec_walks(adj, p=n2v_p, q=n2v_q, walk_length=n2v_walk_len, walks_per_node=n2v_walks_per_node, seed=seed)
        elif backend == "corewalk":
            cores = core_numbers(num_nodes, edges)
            walks = core_biased_walks(adj, cores, walk_length=core_walk_len, walks_per_node=core_walks_per_node, temperature=core_temp, seed=seed)
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

        E = train_skipgram_from_walks(
            walks, num_nodes, dim=dim, window=sgns_window, epochs=sgns_epochs, batch_size=sgns_batch,
            lr=sgns_lr, device=device, neg_k=sgns_neg_k
        )
        out[L] = E
    return out
