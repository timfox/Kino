"""Audit LTX pooled hidden states with physics steering bundles (QA hook)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.physics_steering.bundle import load_steering_bundle
from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.ltx_bridge import PhysicsSteeringLTXBridge
from ltx_trainer.physics_steering.metrics import flip_rate, score_delta
from ltx_trainer.physics_steering.steering import probe_impossible_score, steer_representation


def bridge_from_bundle(bundle_path: str | Path, *, dim: int | None = None) -> PhysicsSteeringLTXBridge:
    """Build :class:`PhysicsSteeringLTXBridge` from ``export_steering_bundle`` JSON."""
    data = load_steering_bundle(bundle_path)
    cfg = PhysicsSteeringConfig()
    d = dim or len(data["probe_weight"])
    bridge = PhysicsSteeringLTXBridge(cfg, dim=d)
    w = data["probe_weight"]
    if w.shape[0] != d:
        w = w[:d]
    bridge.weight.copy_(w)
    bridge.bias.copy_(torch.tensor(data["probe_bias"]))
    bridge.cav.copy_(data["cav_by_layer"].get(data["probe_layer"], data["cav_by_layer"][5]))
    return bridge


def audit_pooled_hidden(
    hidden: Tensor,
    *,
    bundle_path: str | Path | None = None,
    bridge: PhysicsSteeringLTXBridge | None = None,
    alphas: tuple[float, ...] = (-5.0, 0.0, 5.0),
) -> dict[str, Any]:
    """Score ``hidden`` (D,) or (B, D) with baseline and steered P(impossible)."""
    if bridge is None:
        if bundle_path is None:
            raise ValueError("bundle_path or bridge required")
        bridge = bridge_from_bundle(bundle_path, dim=hidden.shape[-1])
    bridge = bridge.to(hidden.device)
    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
    w, b, v = bridge.weight, bridge.bias, bridge.cav / (bridge.cav.norm() + 1e-8)
    base_p = probe_impossible_score(hidden, w, b)
    base_pred = (base_p >= 0.5).long()
    rows: list[dict[str, float]] = []
    for alpha in alphas:
        if alpha == 0.0:
            rows.append({"alpha": alpha, "p_impossible": float(base_p.mean().item())})
            continue
        steered_f = steer_representation(hidden, v, alpha)
        steered_p = probe_impossible_score(steered_f, w, b)
        steered_pred = (steered_p >= 0.5).long()
        rows.append(
            {
                "alpha": float(alpha),
                "p_impossible": float(steered_p.mean().item()),
                "flip_rate": flip_rate(base_pred, steered_pred),
                "score_delta": score_delta(base_p, steered_p),
            }
        )
    return {
        "n": int(hidden.shape[0]),
        "baseline_p_impossible": float(base_p.mean().item()),
        "baseline_pred_impossible_frac": float(base_pred.float().mean().item()),
        "alpha_rows": rows,
    }


def audit_npz_primary_layer(
    npz_path: str | Path,
    bundle_path: str | Path,
    *,
    layer: int | None = None,
) -> dict[str, Any]:
    """Audit all samples at PEZ layer in a collected activations NPZ."""
    from ltx_trainer.physics_steering.activations_io import load_layer_activations

    layer_feats, labels, blocks, meta = load_layer_activations(npz_path)
    data = load_steering_bundle(bundle_path)
    layer = layer if layer is not None else int(data.get("probe_layer", 5))
    hidden = layer_feats[layer]
    audit = audit_pooled_hidden(hidden, bridge=bridge_from_bundle(bundle_path, dim=hidden.shape[-1]))
    audit["labels"] = labels.tolist()
    audit["blocks"] = blocks.tolist() if blocks is not None else None
    audit["metadata"] = meta
    audit["layer"] = layer
    return audit
