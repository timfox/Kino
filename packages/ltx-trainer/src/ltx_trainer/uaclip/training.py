"""Utility-Aware CLIP fine-tuning step from pretrained embeddings (Section 3.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ltx_trainer.uaclip.config import UAClipConfig
from ltx_trainer.uaclip.demand import Platform, visual_utility
from ltx_trainer.uaclip.infonce import bidirectional_infonce, build_score_matrix
from ltx_trainer.uaclip.similarity import l2_normalize


@dataclass
class UAClipTrainReport:
    loss_before: float
    loss_after: float
    clip_loss_before: float
    clip_loss_after: float
    platform: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss_before": round(self.loss_before, 6),
            "loss_after": round(self.loss_after, 6),
            "clip_loss_before": round(self.clip_loss_before, 6),
            "clip_loss_after": round(self.clip_loss_after, 6),
            "platform": self.platform,
            "improved": self.loss_after < self.loss_before,
        }


def _batch_loss(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    utilities: np.ndarray,
    cfg: UAClipConfig,
    *,
    utility_aware: bool,
) -> float:
    scores = build_score_matrix(
        image_embs,
        text_embs,
        utilities,
        alpha_visual=cfg.alpha_visual,
        beta_semantic=cfg.beta_semantic,
        utility_aware=utility_aware,
    )
    return bidirectional_infonce(scores, temperature=cfg.temperature, utility_aware=utility_aware)["loss"]


def train_utility_aware_step(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    attrs_list: list[dict[str, float]],
    platform: Platform,
    cfg: UAClipConfig | None = None,
    *,
    lr: float = 0.05,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, UAClipTrainReport]:
    """
    One FD step on image/text embeddings minimizing Utility-Aware InfoNCE.
    """
    cfg = cfg or UAClipConfig()
    image_embs = l2_normalize(np.asarray(image_embs, dtype=np.float64))
    text_embs = l2_normalize(np.asarray(text_embs, dtype=np.float64))
    utilities = np.array(
        [visual_utility(a, platform, eta=cfg.eta_utility) for a in attrs_list],
        dtype=np.float64,
    )

    clip_before = _batch_loss(image_embs, text_embs, utilities, cfg, utility_aware=False)
    ua_before = _batch_loss(image_embs, text_embs, utilities, cfg, utility_aware=True)

    def ua_loss(im: np.ndarray, tx: np.ndarray) -> float:
        return _batch_loss(l2_normalize(im), l2_normalize(tx), utilities, cfg, utility_aware=True)

    base = ua_loss(image_embs, text_embs)
    grad_i = np.zeros_like(image_embs)
    grad_t = np.zeros_like(text_embs)
    n, d = image_embs.shape
    for i in range(n):
        for j in range(d):
            im2 = image_embs.copy()
            im2[i, j] += eps
            grad_i[i, j] = (ua_loss(im2, text_embs) - base) / eps
            tx2 = text_embs.copy()
            tx2[i, j] += eps
            grad_t[i, j] = (ua_loss(image_embs, tx2) - base) / eps
    image_out = l2_normalize(image_embs - lr * grad_i)
    text_out = l2_normalize(text_embs - lr * grad_t)
    ua_after = ua_loss(image_out, text_out)
    clip_after = _batch_loss(image_out, text_out, utilities, cfg, utility_aware=False)

    report = UAClipTrainReport(
        loss_before=ua_before,
        loss_after=ua_after,
        clip_loss_before=clip_before,
        clip_loss_after=clip_after,
        platform=platform,
    )
    return image_out, text_out, report


def demo_train(
    *,
    seed: int = 0,
    platform: Platform = "amazon",
    n: int = 8,
) -> dict[str, Any]:
    from ltx_trainer.uaclip.visual_attrs import VisualAttributes

    cfg = UAClipConfig(embed_dim=32, temperature=1.0)
    rng = np.random.default_rng(seed)
    d = cfg.embed_dim
    attrs_list = [VisualAttributes.random(rng, platform).to_dict(platform) for _ in range(n)]
    im = l2_normalize(rng.standard_normal((n, d)))
    tx = l2_normalize(rng.standard_normal((n, d)))
    _, _, report = train_utility_aware_step(im, tx, attrs_list, platform, cfg)
    return report.to_dict()
