"""Export / import fitted motion parameters for animated SVG replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ltx_trainer.livesvg.config import LiveSVGConfig


def export_motion_bundle(
    path: str | Path,
    *,
    homographies: Tensor,
    path_deltas: Tensor,
    num_paths: int,
    probe_layer: int = 0,
    metadata: dict[str, Any] | None = None,
    cfg: LiveSVGConfig | None = None,
) -> Path:
    """Write JSON bundle: per-keyframe homographies and path offsets."""
    cfg = cfg or LiveSVGConfig()
    payload: dict[str, Any] = {
        "paper": cfg.paper_arxiv,
        "num_keyframes": int(homographies.shape[0]),
        "num_paths": num_paths,
        "probe_layer": probe_layer,
        "homographies": homographies.detach().cpu().tolist(),
        "path_deltas": path_deltas.detach().cpu().tolist(),
        "metadata": metadata or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_motion_bundle(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["homographies"] = torch.tensor(data["homographies"], dtype=torch.float32)
    data["path_deltas"] = torch.tensor(data["path_deltas"], dtype=torch.float32)
    return data
