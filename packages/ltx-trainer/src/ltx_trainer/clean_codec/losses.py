"""CleanCodec training objectives (§3.2.4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def mel_l2_loss(m: Tensor, m_hat: Tensor) -> Tensor:
    return F.mse_loss(m_hat, m)


def ssl_cosine_loss(s: Tensor, s_hat: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(s.flatten(1), s_hat.flatten(1), dim=-1).mean()


def speaker_cosine_loss(g: Tensor, g_hat: Tensor) -> Tensor:
    return 1.0 - F.cosine_similarity(g, g_hat, dim=-1).mean()


def combined_codec_loss(
    m: Tensor,
    m_hat: Tensor,
    s: Tensor,
    s_hat: Tensor,
    g: Tensor,
    g_hat: Tensor,
    *,
    w_ssl: float = 1.0,
    w_emb: float = 1.0,
) -> dict[str, Tensor]:
    l_mel = mel_l2_loss(m, m_hat)
    l_ssl = ssl_cosine_loss(s, s_hat)
    l_emb = speaker_cosine_loss(g, g_hat)
    total = l_mel + w_ssl * l_ssl + w_emb * l_emb
    return {"total": total, "Lmel": l_mel, "Lssl": l_ssl, "Lemb": l_emb}
