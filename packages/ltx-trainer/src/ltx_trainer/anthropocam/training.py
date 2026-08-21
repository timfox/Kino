"""Training / inference helpers and parameter sweeps from §5."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.anthropocam.config import (
    CONTENT_LAYER,
    HIGH_RES,
    LATENCY_TARGET_S,
    LOW_RES,
    MOBILE_RESOLUTION,
    OPTIMAL_BATCH_SIZE,
    OPTIMAL_EPOCHS,
    OPTIMAL_STYLE_WEIGHT,
    AnthropoCamConfig,
)
from ltx_trainer.anthropocam.features import ProxyVGGFeatures
from ltx_trainer.anthropocam.gram import gram_matrix
from ltx_trainer.anthropocam.layers import style_layers_for
from ltx_trainer.anthropocam.losses import content_loss, style_loss, total_loss, total_variation_loss
from ltx_trainer.anthropocam.transform_net import AnthropoTransformNet


def build_feature_extractor(cfg: AnthropoCamConfig) -> ProxyVGGFeatures:
    layers = tuple(dict.fromkeys((*cfg.style_layers_for_texture(), cfg.content_layer)))
    return ProxyVGGFeatures(layers, content_layer=cfg.content_layer)


def perceptual_loss_bundle(
    content_image: Tensor,
    style_image: Tensor,
    generated: Tensor,
    cfg: AnthropoCamConfig,
    *,
    extractor: ProxyVGGFeatures | None = None,
) -> dict[str, float]:
    ext = extractor or build_feature_extractor(cfg)
    ext.eval()
    c_feats = ext(content_image)
    s_feats = ext(style_image)
    g_feats = ext(generated)
    style_grams = {k: gram_matrix(v) for k, v in s_feats.items()}
    c_l = content_loss(c_feats[cfg.content_layer], g_feats[cfg.content_layer])
    s_l = style_loss(style_grams, g_feats, cfg.layer_weights)
    tv_l = total_variation_loss(generated)
    tot = total_loss(c_l, s_l, tv_l, content_weight=cfg.content_weight, style_weight=cfg.style_weight, tv_weight=cfg.tv_weight)
    return {
        "content": float(c_l.detach()),
        "style": float(s_l.detach()),
        "tv": float(tv_l.detach()),
        "total": float(tot.detach()),
    }


def style_weight_sensitivity() -> list[dict[str, Any]]:
    """§5.2.1 — w_s sweep with qualitative labels."""
    return [
        {"w_s": 2, "label": "subtle", "note": "high structural stability, weak stylization"},
        {"w_s": 5, "label": "optimal", "note": "balanced texture + semantic legibility"},
        {"w_s": 8, "label": "semantic_erasure", "note": "blocky artifacts, geometry loss"},
    ]


def epoch_convergence_table() -> list[dict[str, Any]]:
    """§5.2.2 — epoch vs quality trade-off."""
    return [
        {"epochs": 1, "quality": "underfit", "note": "fast but weak Anthropocene texture"},
        {"epochs": 10, "quality": "optimal", "note": "best texture depth vs time"},
        {"epochs": 20, "quality": "saturated", "note": "marginal visual gain, overfit risk"},
    ]


def batch_size_table() -> list[dict[str, Any]]:
    """§5.2.3 — batch size stability."""
    return [
        {"batch_size": 4, "stability": "oscillatory", "note": "noisy convergence"},
        {"batch_size": 8, "stability": "optimal", "note": "stable updates, moderate memory"},
        {"batch_size": 16, "stability": "slow", "note": "higher memory and step time"},
    ]


def resolution_latency_table() -> list[dict[str, Any]]:
    """§5.3 — mobile resolution vs texture fidelity."""
    return [
        {
            "width": LOW_RES[0],
            "height": LOW_RES[1],
            "latency_class": "instant",
            "texture": "blocky modular abstraction",
            "mobile_ok": False,
        },
        {
            "width": MOBILE_RESOLUTION[0],
            "height": MOBILE_RESOLUTION[1],
            "latency_class": "target_3_5s",
            "texture": "balanced Anthropocene detail",
            "mobile_ok": True,
        },
        {
            "width": HIGH_RES[0],
            "height": HIGH_RES[1],
            "latency_class": "slow",
            "texture": "fine high-frequency industrial detail",
            "mobile_ok": False,
        },
    ]


def recommended_mobile_config() -> AnthropoCamConfig:
    return AnthropoCamConfig(
        style_weight=OPTIMAL_STYLE_WEIGHT,
        train_epochs=OPTIMAL_EPOCHS,
        train_batch_size=OPTIMAL_BATCH_SIZE,
        mobile_width=MOBILE_RESOLUTION[0],
        mobile_height=MOBILE_RESOLUTION[1],
        latency_target_s=LATENCY_TARGET_S,
    )


def forward_smoke(cfg: AnthropoCamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or recommended_mobile_config()
    net = AnthropoTransformNet(cfg)
    ext = build_feature_extractor(cfg)
    content = torch.rand(1, 3, 64, 64)
    style = torch.rand(1, 3, 64, 64)
    style_id = torch.zeros(1, dtype=torch.long)
    out = net(content, style_id)
    losses = perceptual_loss_bundle(content, style, out, cfg, extractor=ext)
    modular_layers = style_layers_for("modular")
    filament_layers = style_layers_for("filamentous")
    return {
        "output_shape": list(out.shape),
        "losses": losses,
        "content_layer": CONTENT_LAYER,
        "modular_style_layers": list(modular_layers),
        "filamentous_style_layers": list(filament_layers),
        "mobile_resolution": [cfg.mobile_width, cfg.mobile_height],
        "style_weight": cfg.style_weight,
    }
