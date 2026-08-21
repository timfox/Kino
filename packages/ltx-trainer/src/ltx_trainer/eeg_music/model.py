"""Channel-oriented EEG encoder stub."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.eeg_music.augmentation import channel_drop
from ltx_trainer.eeg_music.channel import ChannelWiseTokenizer
from ltx_trainer.eeg_music.config import EegMusicConfig


class ChannelOrientedEEGStub(nn.Module):
    """Tokenizer + transformer + CLS projection."""

    def __init__(self, cfg: EegMusicConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EegMusicConfig()
        d = self.cfg.align_dim
        self.tokenizer = ChannelWiseTokenizer(
            self.cfg.num_channels, patch_size=50, embed_dim=d // 4
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d // 4,
            nhead=4,
            dim_feedforward=d,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj = nn.Linear(d // 4, d)

    def encode(self, x: Tensor, *, drop_channels: bool = True) -> Tensor:
        if drop_channels:
            x = channel_drop(x, self.cfg.channel_dropout)
        tokens = self.tokenizer(x)
        h = self.encoder(tokens)
        cls = h[:, 0, :]
        return self.proj(cls)

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)
