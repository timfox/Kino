"""Export / import steering bundles (CAVs + probe weights) for deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.physics_steering.config import PhysicsSteeringConfig


def export_steering_bundle(
    path: str | Path,
    *,
    cav_by_layer: dict[int, Tensor],
    probe_weight: Tensor,
    probe_bias: float | Tensor,
    probe_layer: int,
    metadata: dict[str, Any] | None = None,
    cfg: PhysicsSteeringConfig | None = None,
) -> Path:
    """Write JSON bundle (vectors as lists) for inference hooks."""
    cfg = cfg or PhysicsSteeringConfig()
    payload: dict[str, Any] = {
        "paper": cfg.paper_arxiv,
        "model_id": cfg.model_id,
        "probe_layer": probe_layer,
        "primary_pez_layer": cfg.primary_pez_layer,
        "steering_saturation_alpha": cfg.steering_saturation_alpha,
        "sign_convention": "alpha>0 → impossible; alpha<0 → possible",
        "probe_weight": probe_weight.detach().cpu().tolist(),
        "probe_bias": float(probe_bias) if isinstance(probe_bias, Tensor) else float(probe_bias),
        "cav_by_layer": {str(k): v.detach().cpu().tolist() for k, v in sorted(cav_by_layer.items())},
        "metadata": metadata or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_steering_bundle(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["cav_by_layer"] = {int(k): torch.tensor(v, dtype=torch.float32) for k, v in data["cav_by_layer"].items()}
    data["probe_weight"] = torch.tensor(data["probe_weight"], dtype=torch.float32)
    data["probe_bias"] = float(data["probe_bias"])
    return data
