
"""
HOPLP-MUL-style baseline (Mishra et al., Applied Intelligence 2023).
Core idea: link likelihood via longer paths with damping + layer fusion via layer ranking.
Implementation: dense adjacency powers up to K with geometric decay alpha, per-layer weights w_l (density ranking).
  score(u,v) = sum_l w_l * sum_{k=1..K} alpha^{k-1} [A_l^k]_{uv}
For undirected graphs, A is symmetric (binary). Suitable for small graphs (like CS-Aarhus).
"""
from typing import Dict, List, Tuple
import numpy as np
import torch

def layer_weights_by_density(edge_sets: Dict[str, set], num_nodes: int):
    # density = |E_l| / (N*(N-1)/2)
    N = num_nodes
    denom = max(1, N*(N-1)//2)
    dens = {L: len(S)/denom for L,S in edge_sets.items()}
    # rank-based weights: normalize to sum 1
    vals = np.array([dens[L] for L in edge_sets.keys()], dtype=np.float64)
    if vals.sum() == 0:
        w = {L: 1.0/len(edge_sets) for L in edge_sets.keys()}
    else:
        vals = vals/vals.sum()
        w = {L: float(v) for L,v in zip(edge_sets.keys(), vals)}
    return w

def build_adj(num_nodes: int, edges: List[Tuple[int,int]]):
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for (u,v) in edges:
        if u==v: continue
        A[u,v]=1.0; A[v,u]=1.0
    return A

@torch.no_grad()
def hoplp_scores(edge_sets: Dict[str, set], num_nodes: int, K: int = 3, alpha: float = 0.5, weight_mode: str = "density"):
    # Compute per-layer A^k
    if weight_mode == "density":
        w = layer_weights_by_density(edge_sets, num_nodes)
    else:
        w = {L: 1.0/len(edge_sets) for L in edge_sets.keys()}

    S = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for L, S_l in edge_sets.items():
        A = build_adj(num_nodes, list(S_l))
        Ak = A.clone()
        for k in range(1, K+1):
            coef = (alpha ** (k-1))
            S += w[L] * coef * Ak
            if k < K:
                Ak = Ak @ A
    return S  # dense score matrix

def scores_for_pairs(S: torch.Tensor, pairs: List[Tuple[int,int]]):
    out = []
    for (u,v) in pairs:
        out.append(float(S[u,v]))
    return torch.tensor(out, dtype=torch.float32)
