"""Save/load mean-pooled layer activations for physics steering experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def save_layer_activations(
    path: str | Path,
    layer_features: dict[int, Tensor],
    labels: Tensor,
    blocks: Tensor | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write ``layer_{l}`` arrays plus ``labels`` (and optional ``blocks``) to NPZ."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "labels": labels.detach().cpu().numpy(),
        "num_layers": len(layer_features),
    }
    if blocks is not None:
        payload["blocks"] = blocks.detach().cpu().numpy()
    if metadata:
        for k, v in metadata.items():
            payload[f"meta_{k}"] = v
    for layer, feats in layer_features.items():
        payload[f"layer_{layer}"] = feats.detach().cpu().numpy()
    import numpy as np

    np.savez_compressed(out, **payload)
    return out


def load_layer_activations(
    path: str | Path,
) -> tuple[dict[int, Tensor], Tensor, Tensor | None, dict[str, Any]]:
    """Load NPZ written by :func:`save_layer_activations`."""
    import numpy as np

    data = np.load(Path(path), allow_pickle=True)
    labels = torch.from_numpy(data["labels"]).long()
    blocks = torch.from_numpy(data["blocks"]).long() if "blocks" in data else None
    layer_features: dict[int, Tensor] = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_", 1)[1])
            layer_features[layer] = torch.from_numpy(data[key]).float()
    meta = {k[5:]: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files if k.startswith("meta_")}
    return layer_features, labels, blocks, meta
