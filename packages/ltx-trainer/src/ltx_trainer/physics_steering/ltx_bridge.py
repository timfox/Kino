"""LTX validation / delivery QA bridge for physics steering concepts."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.metrics import directional_purity, flip_rate, score_delta
from ltx_trainer.physics_steering.probe import cav_from_weights, fit_logistic_probe
from ltx_trainer.physics_steering.steering import probe_impossible_score, steer_representation


def ltx_integration_notes(cfg: PhysicsSteeringConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PhysicsSteeringConfig()
    return {
        "use_case": "Stress-test LTX validation clips with physics CAV steering on pooled hidden states",
        "hook_point": "Optional VideoMAE / temporal encoder mean-pool before validation_sampler scoring",
        "does_not": "Does not steer LTX DiT latents directly — representation-space audit only",
        "alpha_sweep": f"Try α ∈ {{-{cfg.steering_saturation_alpha}, 0, +{cfg.steering_saturation_alpha}}}",
        "pez_layers": list(cfg.top_pez_layers),
    }


class PhysicsSteeringLTXBridge(nn.Module):
    """Tiny probe + steer head for validation hidden-state QA smoke."""

    def __init__(self, cfg: PhysicsSteeringConfig | None = None, *, dim: int | None = None) -> None:
        super().__init__()
        cfg = cfg or PhysicsSteeringConfig()
        d = dim or cfg.ltx_bridge_dim
        self.dim = d
        self.register_buffer("weight", torch.randn(d))
        self.register_buffer("bias", torch.zeros(()))
        self.register_buffer("cav", torch.randn(d))

    def forward(self, hidden: Tensor, alpha: float = 0.0) -> Tensor:
        """Return steered P(impossible) for (B, D) pooled features."""
        w = self.weight
        b = self.bias
        v = self.cav / (self.cav.norm() + 1e-8)
        f = hidden
        if alpha != 0.0:
            f = steer_representation(hidden, v, alpha)
        return probe_impossible_score(f, w, b)

    def fit_from_labels(self, features: Tensor, labels: Tensor, *, steps: int = 100) -> dict[str, float]:
        w, b, acc = fit_logistic_probe(features, labels, steps=steps)
        self.weight.copy_(w[: self.dim] if w.numel() >= self.dim else w)
        self.bias.copy_(b.reshape(()))
        self.cav.copy_(cav_from_weights(self.weight))
        base = probe_impossible_score(features, self.weight, self.bias)
        steered_f = steer_representation(features, self.cav, alpha=5.0)
        steered = probe_impossible_score(steered_f, self.weight, self.bias)
        delta = steered_f - features
        return {
            "train_accuracy": acc,
            "flip_rate_alpha5": flip_rate((base >= 0.5).long(), (steered >= 0.5).long()),
            "score_delta_alpha5": score_delta(base, steered),
            "directional_purity_alpha5": directional_purity(delta, self.cav),
        }
