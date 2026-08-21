"""TQCodec model and training pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ltx_trainer.tqcodec.config import TQCodecConfig
from ltx_trainer.tqcodec.losses import codec_generator_loss, log_spectral_distance, snr_db
from ltx_trainer.tqcodec.pqmf import PQMFAnalysis, PQMFSynthesis, SubbandEncoder
from ltx_trainer.tqcodec.quantizer import RSimVQ
from ltx_trainer.tqcodec.seanet import SEANetDecoder, SEANetEncoder


class TQCodec(nn.Module):
    """Encoder → RSimVQ → SEANet decoder with optional PQMF subbands."""

    def __init__(self, cfg: TQCodecConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TQCodecConfig()
        self.analysis = PQMFAnalysis() if self.cfg.use_subband else None
        self.synthesis = PQMFSynthesis() if self.cfg.use_subband else None
        self.subband = SubbandEncoder() if self.cfg.use_subband else None
        self.encoder = SEANetEncoder(self.cfg)
        self.quantizer = RSimVQ(self.cfg.latent_dim, self.cfg.codebook_size, self.cfg.num_codebooks)
        self.decoder = SEANetDecoder(self.cfg)

    def encode(self, x: Tensor) -> Tensor:
        if self.analysis is not None:
            x = self.analysis(x)
        z = self.encoder(x if x.dim() == 3 else x.unsqueeze(1))
        z = z.transpose(1, 2)
        q, _ = self.quantizer(z)
        return q.transpose(1, 2)

    def decode(self, z: Tensor) -> Tensor:
        y = self.decoder(z)
        if self.synthesis is not None and y.shape[1] != 1:
            y = self.synthesis(y)
        return y

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x).transpose(1, 2)
        q, codes = self.quantizer(z)
        y = self.decode(q.transpose(1, 2))
        if y.shape[-1] != x.shape[-1]:
            y = torch.nn.functional.interpolate(y, size=x.shape[-1], mode="linear", align_corners=False)
        return y, {"codes": codes}


def train_step(model: TQCodec, batch: Tensor) -> tuple[Tensor, dict[str, float]]:
    pred, _ = model(batch)
    target = batch if batch.dim() == 3 else batch.unsqueeze(1)
    loss, stats = codec_generator_loss(pred, target)
    stats["loss"] = float(loss.detach())
    stats["lsd"] = log_spectral_distance(pred.detach(), target.detach())
    stats["snr_db"] = snr_db(pred.detach(), target.detach())
    return loss, stats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
