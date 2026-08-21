"""Instruction-following stub: audio tokens + text head (Qwen2.5-Omni style)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.iwslt26_if.config import IWSLT26IFConfig

TASK_TOKENS = (
    "<|asr|>",
    "<|st|>",
    "<|mc|>",
    "<|sqa|>",
    "<|achap|>",
    "<|ssum|>",
    "<|instruct|>",
)


class SpeechIFStub(nn.Module):
    """Minimal dual-modality stub for forward / loss smoke."""

    def __init__(self, cfg: IWSLT26IFConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or IWSLT26IFConfig()
        d = 128
        self.audio_proj = nn.Linear(cfg.demo_audio_len, d)
        self.text_emb = nn.Embedding(256, d)
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=4, batch_first=True, dim_feedforward=d * 2),
            num_layers=2,
        )
        self.out = nn.Linear(d, 256)

    def forward(self, audio_feat: Tensor, text_ids: Tensor) -> Tensor:
        # audio_feat: (B, L), text_ids: (B, T)
        a = self.audio_proj(audio_feat).unsqueeze(1)
        t = self.text_emb(text_ids)
        h = torch.cat([a, t], dim=1)
        h = self.fusion(h)
        return self.out(h[:, -text_ids.shape[1] :, :])
