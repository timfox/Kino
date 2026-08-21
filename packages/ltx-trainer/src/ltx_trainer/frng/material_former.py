"""MaterialFormer + light-independence loss (paper Sec. 4.2, Eq. 9–12)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ltx_trainer.frng.config import FRNGConfig


class MatDetokenizer(nn.Module):
    """Eq. (11): single-layer MLP to per-Gaussian material features."""

    def __init__(self, token_dim: int, mat_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(token_dim, mat_dim)

    def forward(self, t_mat: Tensor) -> Tensor:
        return self.net(t_mat)


class MaterialFormer(nn.Module):
    """Eq. (9)–(10): Transformer decoder with IDM prior tokens as memory (conditional).

    RelitLRM is **not** executed here: pass frozen ``T_geo``, ``T_app``, ``T_prior`` tensors of shape (B, P, D).
    """

    def __init__(self, cfg: FRNGConfig, *, mat_out_dim: int) -> None:
        super().__init__()
        d = cfg.token_dim
        if d % cfg.matformer_heads != 0:
            raise ValueError("token_dim must be divisible by matformer_heads")
        layer = nn.TransformerDecoderLayer(
            d_model=d,
            nhead=cfg.matformer_heads,
            dim_feedforward=cfg.matformer_dim_feedforward,
            dropout=cfg.matformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.matformer_layers)
        self.mat_head = MatDetokenizer(d, mat_out_dim)

    def forward(self, t_geo: Tensor, t_app: Tensor, t_prior: Tensor) -> tuple[Tensor, Tensor]:
        """Returns material tokens ``t_mat`` (B, P, D) and features ``g_mat`` (B, P, mat_out_dim).

        ``T_all = {T_geo} ⊕ {T_app}`` is sequence concat (paper Eq. 9); decoder output at the first
        ``P`` positions is treated as per-Gaussian material state (appearance tokens attend into geo slots).
        """
        if t_geo.shape[1] != t_app.shape[1]:
            raise ValueError("t_geo and t_app must share patch count P")
        t_all = torch.cat([t_geo, t_app], dim=1)
        h = self.decoder(tgt=t_all, memory=t_prior)
        p = t_geo.shape[1]
        t_mat = h[:, :p, :]
        g_mat = self.mat_head(t_mat)
        return t_mat, g_mat


def light_independence_loss(
    t_mat_a: Tensor,
    t_mat_b: Tensor,
    *,
    lambda_cos: float = 0.2,
    lambda_kld: float = 0.2,
    eps: float = 1e-8,
) -> Tensor:
    """Eq. (12): ``λ_c (1 - cos) + λ_KLD * KL`` with cosine on last dim, discrete KL on softmaxed tokens."""
    # Cosine similarity per position, mean over batch and patches
    cos = F.cosine_similarity(t_mat_a, t_mat_b, dim=-1, eps=eps).mean()
    l_cos = 1.0 - cos
    p = F.softmax(t_mat_a, dim=-1).clamp_min(eps)
    q = F.softmax(t_mat_b, dim=-1).clamp_min(eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1).mean()
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1).mean()
    l_kld = 0.5 * (kl_pq + kl_qp)
    return lambda_cos * l_cos + lambda_kld * l_kld
