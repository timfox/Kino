"""UniMatch V2 teacher-student segmentation (Fig. 1)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ltx_trainer.weatherproof.augment import strong_augment, weak_augment
from ltx_trainer.weatherproof.classes import NUM_CLASSES
from ltx_trainer.weatherproof.complementary import complementary_dropout
from ltx_trainer.weatherproof.decoder import DPTSegHead
from ltx_trainer.weatherproof.encoder import FrozenDinoStub
from ltx_trainer.weatherproof.losses import SemiSupervisedLossConfig


@dataclass
class UniMatchV2Config:
    num_classes: int = NUM_CLASSES
    feat_channels: int = 256
    confidence_tau: float = 0.95
    crop_size: int = 518


class SegNetwork(nn.Module):
    def __init__(self, cfg: UniMatchV2Config) -> None:
        super().__init__()
        self.encoder = FrozenDinoStub(out_channels=cfg.feat_channels)
        self.decoder = DPTSegHead(cfg.feat_channels, num_classes=cfg.num_classes)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, feat: Tensor, *, out_size: tuple[int, int] | None = None) -> Tensor:
        return self.decoder(feat, out_size=out_size)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        h, w = x.shape[-2:]
        logits = self.decode(self.encode(x), out_size=(h, w))
        return logits.squeeze(0) if squeeze else logits

    def forward_strong_pair(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Two strong views with complementary dropout."""
        h, w = x.shape[-2:]
        s1 = strong_augment(x)
        s2 = strong_augment(x)
        if s1.dim() == 3:
            s1, s2 = s1.unsqueeze(0), s2.unsqueeze(0)
        e1, e2 = self.encode(s1), self.encode(s2)
        e1, e2 = complementary_dropout(e1, e2)
        return self.decode(e1, out_size=(h, w)), self.decode(e2, out_size=(h, w))


class UniMatchV2Seg(nn.Module):
    """Student + EMA teacher for WeatherProof semi-supervised training."""

    def __init__(self, cfg: UniMatchV2Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or UniMatchV2Config()
        self.student = SegNetwork(self.cfg)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.loss_cfg = SemiSupervisedLossConfig(confidence_tau=self.cfg.confidence_tau)

    def pseudo_from_teacher(self, degraded: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Weak aug → teacher logits → pseudo label + confidence mask."""
        weak = weak_augment(degraded)
        if weak.dim() == 3:
            weak = weak.unsqueeze(0)
        with torch.no_grad():
            logits = self.teacher(weak)
            if logits.dim() == 3:
                logits = logits.unsqueeze(0)
            prob = F.softmax(logits, dim=1)
            conf, pseudo = prob.max(dim=1)
            mask = (conf >= self.cfg.confidence_tau).long()
        return pseudo.squeeze(0), mask.squeeze(0), logits

    def trainable_parameters(self):
        return self.student.decoder.parameters()

    def init_teacher_from_student(self) -> None:
        """Sync teacher decoder to student after student weight init."""
        self.teacher.decoder.load_state_dict(self.student.decoder.state_dict())
