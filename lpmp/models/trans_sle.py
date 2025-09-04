from typing import Dict, List
import torch
import torch.nn as nn


def _get_layer_tensor(d: Dict, L):
    """Robust layer lookup: accept both str/int keys."""
    if L in d:
        return d[L]
    Ls = str(L)
    if Ls in d:
        return d[Ls]
    raise KeyError(f"Missing tensor for layer key={L!r} (also tried {Ls!r}).")


class SLETransformer(nn.Module):
    """
    Sequence-only: each token is concat([e_L[u], e_L[v]]) for a layer L.
    Token dim = 2*d_node → d_model must equal 2*d_node.

    Defaults aligned with the original intention: include_target_in_context=True.
    """
    def __init__(
        self,
        layer_names: List[str],
        node2vec: Dict[str, torch.Tensor],
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        include_target_in_context: bool = True,
    ) -> None:
        super().__init__()
        self.layer_names = [str(L) for L in layer_names]
        # keep on CPU; move to the right device just-in-time
        self.node2vec: Dict[str, torch.Tensor] = {
            str(L): emb.detach().to(torch.float32).contiguous() for L, emb in node2vec.items()
        }
        self.include_target_in_context = include_target_in_context

        # infer d_node and validate d_model
        any_L = self.layer_names[0]
        d_node = int(_get_layer_tensor(self.node2vec, any_L).shape[1])
        assert d_model == 2 * d_node, "Expected d_model == 2*d_node (token = [e[u]||e[v]])."
        self.d_model = d_model

        # Transformer encoder
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))  # learned CLS
        nn.init.normal_(self.cls, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    @torch.no_grad()
    def _edge_embed_for_layer(self, L: str, pair_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns [B, 2*d_node] concat embeddings for (u,v) on layer L.
        Ensures embeddings are on the same device as pair_idx.
        """
        E = _get_layer_tensor(self.node2vec, L)   # [N, d_node] (CPU)
        if E.device != pair_idx.device:
            E = E.to(pair_idx.device, non_blocking=True)
        uv = E.index_select(0, pair_idx.view(-1))  # [B*2, d_node]
        uv = uv.view(pair_idx.size(0), 2, -1)
        return torch.cat([uv[:, 0, :], uv[:, 1, :]], dim=-1)  # [B, 2*d_node] == d_model

    def _build_sequence(self, target_layer: str, pair_idx: torch.Tensor) -> torch.Tensor:
        B = pair_idx.size(0)
        device = pair_idx.device
        toks = [self.cls.expand(B, 1, -1).to(device)]
        tgt = str(target_layer)
        for L in self.layer_names:
            if not self.include_target_in_context and L == tgt:
                continue
            e = self._edge_embed_for_layer(L, pair_idx)  # [B, d_model]
            toks.append(e.unsqueeze(1))
        return torch.cat(toks, dim=1)  # [B, 1+num_ctx_layers, d_model]

    def forward(self, target_layer: str, pair_idx: torch.Tensor) -> torch.Tensor:
        seq = self._build_sequence(target_layer, pair_idx)
        h = self.encoder(seq)[:, 0, :]
        return self.head(h)  # [B,1]
