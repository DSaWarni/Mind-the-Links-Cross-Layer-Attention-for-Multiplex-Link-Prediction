
from typing import Dict, Tuple, List, Set, Iterable, Optional
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass

@dataclass
class MultiplexData:
    layers: List[str]
    num_nodes: int
    edge_index: Dict[str, torch.Tensor]
    edge_sets: Dict[str, Set[Tuple[int,int]]]

def _canon(u:int, v:int)->Tuple[int,int]:
    return (u,v) if u<=v else (v,u)

def load_edges(
    edges_csv_path: str,
    layer_col: Optional[str] = None,
    start_col: Optional[str] = None,
    end_col: Optional[str] = None,
    id_base: int = 1,           # 1 → subtract 1 ; 0 → use as-is
    sep: Optional[str] = None,  # None → auto regex splitter
    has_header: bool = False,   # default tuned for your sample (no header)
    directed: bool = False,     # False → symmetrize
) -> 'MultiplexData':
    """
    Flexible loader for multiplex edge lists.

    Defaults here assume a headerless file with 4 columns:
        Layer  Start_Node  End_Node  Edge_Weight
    where Edge_Weight is ignored.

    You can override column names via layer_col/start_col/end_col if a header exists.
    """
    if sep is None:
        sep = r"\s+|,|;|\t"

    df = pd.read_csv(
        edges_csv_path,
        sep=sep,
        engine="python",
        header=0 if has_header else None
    )
    if not has_header:
        # Assign generic names and map to expected roles
        # Expect at least 3 cols: [layer, start, end, (optional weight)]
        cols = [f"col{i}" for i in range(len(df.columns))]
        df.columns = cols
        # default mapping for 4-col layout
        layer_col = layer_col or cols[0]
        start_col = start_col or cols[1]
        end_col   = end_col   or cols[2]
    else:
        # try resolve case-insensitively if not provided
        cols_lower = {c.lower(): c for c in df.columns}
        def find_one(cands):
            for c in cands:
                if c in cols_lower: return cols_lower[c]
            return None
        layer_col = layer_col or find_one({"layer","layers","layer_id","type","relation","rel","edge_type"})
        start_col = start_col or find_one({"start","source","src","u","from","i","node1"})
        end_col   = end_col   or find_one({"end","target","dst","v","to","j","node2"})
        if any(x is None for x in [layer_col, start_col, end_col]):
            raise ValueError("Could not infer layer/start/end columns from header. Please pass --layer-col/--start-col/--end-col.")

    # Extract arrays and normalize ids
    u_raw = df[start_col].to_numpy()
    v_raw = df[end_col].to_numpy()
    layers = df[layer_col].astype(str).to_numpy()

    # Try integer casting; else factorize
    try:
        u = u_raw.astype(np.int64)
        v = v_raw.astype(np.int64)
        if id_base == 1:
            u -= 1; v -= 1
    except Exception:
        all_nodes = pd.Index(u_raw).append(pd.Index(v_raw)).unique()
        remap = {k:i for i,k in enumerate(all_nodes)}
        u = np.array([remap[x] for x in u_raw], dtype=np.int64)
        v = np.array([remap[x] for x in v_raw], dtype=np.int64)

    num_nodes = int(max(u.max(), v.max())) + 1
    layer_names = list(sorted(pd.unique(layers)))

    edge_index = {}
    edge_sets = {}
    for L in layer_names:
        mask = (layers == L)
        uu = u[mask]; vv = v[mask]
        if directed:
            edges = np.stack([uu, vv], axis=0)
        else:
            edges = np.stack([np.concatenate([uu, vv]), np.concatenate([vv, uu])], axis=0)
        edge_index[L] = torch.from_numpy(edges).long().contiguous()
        if directed:
            es = {(int(a), int(b)) for a,b in zip(uu, vv) if a!=b}
        else:
            es = {_canon(int(a), int(b)) for a,b in zip(uu, vv) if a!=b}
        edge_sets[L] = es

    return MultiplexData(layer_names, num_nodes, edge_index, edge_sets)

def build_union_pairs(edge_sets: Dict[str, Set[Tuple[int,int]]]) -> Set[Tuple[int,int]]:
    union = set()
    for es in edge_sets.values():
        union |= es
    return union

def labels_for_layer(target_layer: str, union_pairs: Iterable[Tuple[int,int]], edge_sets: Dict[str, Set[Tuple[int,int]]]):
    y = []
    for e in union_pairs:
        y.append(1 if e in edge_sets[target_layer] else 0)
    return np.array(y, dtype=np.int64)

def stratified_split(X_idx: np.ndarray, y: np.ndarray, train: float=0.7, val: float=0.15, test: float=0.15, seed: int=0):
    assert abs(train+val+test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx_pos = X_idx[y==1]
    idx_neg = X_idx[y==0]

    def split_indices(idx):
        n = len(idx)
        n_train = int(train*n)
        n_val = int(val*n)
        perm = rng.permutation(idx)
        i_tr = perm[:n_train]
        i_va = perm[n_train:n_train+n_val]
        i_te = perm[n_train+n_val:]
        return i_tr, i_va, i_te

    tr_p, va_p, te_p = split_indices(idx_pos)
    tr_n, va_n, te_n = split_indices(idx_neg)

    tr = np.concatenate([tr_p, tr_n]); rng.shuffle(tr)
    va = np.concatenate([va_p, va_n]); rng.shuffle(va)
    te = np.concatenate([te_p, te_n]); rng.shuffle(te)
    return tr, va, te

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: List[Tuple[int,int]], labels: np.ndarray):
        assert len(pairs) == len(labels)
        self.pairs = pairs
        self.labels = torch.from_numpy(labels.astype(np.float32)).view(-1,1)
        self._pair_idx = torch.as_tensor(pairs, dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {
            "pair_idx": self._pair_idx[idx],
            "y": self.labels[idx],
        }

def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg/ max(1, pos), dtype=torch.float32)


def mask_pairs_from_layer(edge_sets, pairs_to_remove, target_layer):
    """
    Return a shallow-copied edge_sets where target-layer edges in pairs_to_remove are removed.
    pairs_to_remove: iterable of (u,v) with u<=v (union_pairs are canonical).
    """
    new_sets = dict(edge_sets)
    drop = set((min(u, v), max(u, v)) for (u, v) in pairs_to_remove)
    pruned = set(e for e in new_sets[target_layer] if e not in drop)
    new_sets[target_layer] = pruned
    return new_sets

