"""FogNet two-stream CLIP model (Sec. 4, Fig. 5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.fognet.attention import CrossStreamAlignment, FogAwareSelection, MutualEnhancement
from ltx_trainer.fognet.classes import NUM_CLASSES
from ltx_trainer.fognet.encoder import ClipTextStub, ClipVisualStub


@dataclass
class FogNetConfig:
    num_classes: int = NUM_CLASSES
    embed_dim: int = 256
    num_frames: int = 8
    use_fas: bool = True
    use_me: bool = True
    use_csa: bool = True


class FogInvariantExtractor(nn.Module):
    def __init__(self, cfg: FogNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        self.fas = FogAwareSelection(d)
        self.me = MutualEnhancement(d)
        self.csa = CrossStreamAlignment()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, vc: Tensor, vf: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return pooled fog/clean embeddings and LTemp."""
        if self.cfg.use_fas:
            ac, af = self.fas(vc, vf)
        else:
            ac, af = vc, vf
        if self.cfg.use_me:
            dc, df = self.me(ac, af)
        else:
            dc, df = ac, af
        if self.cfg.use_csa:
            sc = self.csa.similarity(df, dc)
            temp_loss = self.csa.temporal_loss(sc)
        else:
            temp_loss = torch.zeros((), device=vf.device)
        fog_emb = df.mean(dim=1)
        clean_emb = dc.mean(dim=1)
        return fog_emb, clean_emb, temp_loss


class FogNet(nn.Module):
    """End-to-end fog-invariant action recognition."""

    def __init__(self, cfg: FogNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or FogNetConfig()
        self.visual = ClipVisualStub(self.cfg.embed_dim)
        self.text = ClipTextStub(self.cfg.num_classes, self.cfg.embed_dim)
        self.extractor = FogInvariantExtractor(self.cfg)

    def encode_video(self, video: Tensor) -> Tensor:
        if video.dim() == 4:
            video = video.unsqueeze(0)
        return self.visual(video)

    def forward_train(
        self,
        foggy: Tensor,
        clean: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        vf = self.encode_video(foggy)
        vc = self.encode_video(clean)
        fog_emb, clean_emb, temp_loss = self.extractor(vc, vf)
        text_emb = self.text(labels)
        return fog_emb, clean_emb, text_emb, temp_loss

    @torch.no_grad()
    def forward_infer(self, foggy: Tensor) -> Tensor:
        """Fog-only inference path (Fig. 5): FAS → ME on fog stream."""
        self.eval()
        vf = self.encode_video(foggy)
        ext = self.extractor
        if self.cfg.use_fas:
            _ac, af = ext.fas(vf, vf)
        else:
            af = vf
        if self.cfg.use_me:
            _dc, df = ext.me(af, af)
        else:
            df = af
        fog_emb = F.normalize(df.mean(dim=1), dim=-1)
        text_w = F.normalize(self.text.embed.weight, dim=-1)
        logits = fog_emb @ text_w.T
        return logits

    def predict(self, foggy: Tensor) -> int:
        return int(self.forward_infer(foggy).argmax(dim=-1).item())
