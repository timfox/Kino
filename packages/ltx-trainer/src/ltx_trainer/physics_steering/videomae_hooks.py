"""Optional VideoMAE activation hooks (MCG-NJU/videomae-base).

Requires ``transformers`` and model weights. Collect per-block outputs and mean-pool
(Eq. 1) for ``physics_steering.experiments``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from ltx_trainer.physics_steering.config import PhysicsSteeringConfig
from ltx_trainer.physics_steering.representation import mean_pool_hidden


class VideoMAEActivationCollector:
    """Register forward hooks on transformer blocks; return ``H_l`` per layer."""

    def __init__(self, model: Any, num_layers: int = 12) -> None:
        self.model = model
        self.num_layers = num_layers
        self._cache: dict[int, Tensor] = {}
        self._handles: list[Any] = []

    def _make_hook(self, layer: int) -> Callable[..., None]:
        def hook(_module: Any, _inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            self._cache[layer] = h.detach()

        return hook

    def register(self) -> None:
        blocks = self._transformer_blocks()
        for i, block in enumerate(blocks):
            if i >= self.num_layers:
                break
            self._handles.append(block.register_forward_hook(self._make_hook(i)))

    def _transformer_blocks(self) -> Any:
        """Resolve HF VideoMAE / ViT encoder layer list."""
        candidates = [
            getattr(self.model, "encoder", None),
            getattr(self.model, "videomae", None),
            self.model,
        ]
        for root in candidates:
            if root is None:
                continue
            layer = getattr(root, "layer", None)
            if layer is not None:
                return layer
            enc = getattr(root, "encoder", None)
            if enc is not None:
                layer = getattr(enc, "layer", None)
                if layer is not None:
                    return layer
        raise AttributeError("Could not find transformer blocks on VideoMAE model")

    def clear(self) -> None:
        self._cache.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def pooled_by_layer(self) -> dict[int, Tensor]:
        return {layer: mean_pool_hidden(h) for layer, h in self._cache.items()}


def load_videomae_model(cfg: PhysicsSteeringConfig | None = None) -> tuple[Any, Any]:
    """Load ``cfg.model_id`` via Hugging Face transformers."""
    cfg = cfg or PhysicsSteeringConfig()
    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as e:
        raise ImportError("pip install transformers for VideoMAE activation collection") from e
    try:
        processor = AutoVideoProcessor.from_pretrained(cfg.model_id)
        model = AutoModel.from_pretrained(cfg.model_id)
    except OSError:
        processor = AutoVideoProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = AutoModel.from_pretrained("MCG-NJU/videomae-base")
    return model, processor


def collect_pooled_activations(
    videos: Tensor,
    model: Any,
    *,
    num_layers: int = 12,
) -> dict[int, Tensor]:
    """Forward pass on a video batch; return mean-pooled ``f_l`` per layer (B, D)."""
    from ltx_trainer.physics_steering.representation import batch_mean_pool

    collector = VideoMAEActivationCollector(model, num_layers=num_layers)
    collector.register()
    collector.clear()
    with torch.no_grad():
        model(videos)
    return {layer: batch_mean_pool(collector._cache[layer]) for layer in collector._cache}
