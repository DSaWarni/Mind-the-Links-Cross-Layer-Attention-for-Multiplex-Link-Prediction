from typing import Dict, List
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv
    _HAVE_PYG = True
except Exception:
    _HAVE_PYG = False


def _get_layer_tensor(d: Dict, L):
    """Robust layer lookup: accept both str/int keys."""
    if L in d:
        return d[L]
    Ls = str(L)
    if Ls in d:
        return d[Ls]
    raise KeyError(f"Missing tensor for layer key={L!r} (also tried {Ls!r}).")


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if not _HAVE_PYG:
            raise ImportError(
                "torch_geometric is required for TransGAT. "
                "Install PyG wheels that match your Torch/CUDA."
            )
        # Out channels per head = out_dim // heads, concat=True → final dim = out_dim
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, in_dim], edge_index: [2, E], both on same device
        h = self.gat(x, edge_index)
        return self.act(h)


class TransGAT(nn.Module):
    """
    Independent per-layer GAT encoders + Transformer over per-layer edge embeddings.
    Token per layer = concat(h[u], h[v])  → d_model must equal 2*d_node.

    By default we EXCLUDE the target layer from the context sequence (include_target_in_context=False).
    """
    def __init__(
        self,
        layer_names: List[str],
        node2vec: Dict[str, torch.Tensor],
        edge_index: Dict[str, torch.Tensor],
        d_node: int,
        d_model: int = 256,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        tfm_layers: int = 2,
        tfm_heads: int = 4,
        tfm_ff: int = 512,
        tfm_dropout: float = 0.1,
        include_target_in_context: bool = False,
    ) -> None:
        super().__init__()
        assert d_model == 2 * d_node, "Expected d_model == 2*d_node (because token = [h[u]||h[v]])."

        self.layer_names = [str(L) for L in layer_names]
        # keep on CPU; move to the right device just-in-time
        self.node2vec: Dict[str, torch.Tensor] = {
            str(L): emb.detach().to(torch.float32).contiguous() for L, emb in node2vec.items()
        }
        self.edge_index = {str(L): ei for L, ei in edge_index.items()}
        self.include_target_in_context = include_target_in_context
        self.d_node = d_node
        self.d_model = d_model

        # Per-layer GAT encoder (dim-preserving)
        self.gats = nn.ModuleDict({
            str(L): GATLayer(d_node, d_node, heads=gat_heads, dropout=gat_dropout)
            for L in self.layer_names
        })

        # Transformer encoder over [CLS, tokens...]
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=tfm_heads,
            dim_feedforward=tfm_ff,
            dropout=tfm_dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=tfm_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def _per_layer_node_emb(self, L: str, device: torch.device) -> torch.Tensor:
        x0 = _get_layer_tensor(self.node2vec, L)
        if x0.device != device:
            x0 = x0.to(device, non_blocking=True)
        ei = _get_layer_tensor(self.edge_index, L)
        if ei.device != device:
            ei = ei.to(device, non_blocking=True)
        h = self.gats[L](x0, ei)  # [N, d_node]
        return h

    @staticmethod
    def _edge_embed(node_emb: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        # pair_idx: [B,2] on same device as node_emb
        uv = node_emb.index_select(0, pair_idx.view(-1))  # [B*2, d_node]
        uv = uv.view(pair_idx.size(0), 2, -1)
        return torch.cat([uv[:, 0, :], uv[:, 1, :]], dim=-1)  # [B, 2*d_node] == d_model

    def forward(self, target_layer: str, pair_idx: torch.Tensor) -> torch.Tensor:
        device = pair_idx.device
        B = pair_idx.size(0)
        toks = [self.cls.expand(B, 1, -1).to(device)]

        tgt = str(target_layer)
        for L in self.layer_names:
            if not self.include_target_in_context and L == tgt:
                continue
            node_emb = self._per_layer_node_emb(L, device)   # [N, d_node]
            e = self._edge_embed(node_emb, pair_idx)         # [B, d_model]
            toks.append(e.unsqueeze(1))

        seq = torch.cat(toks, dim=1)          # [B, 1+num_ctx_layers, d_model]
        h = self.encoder(seq)[:, 0, :]        # [B, d_model]
        return self.head(h)                   # [B,1]
