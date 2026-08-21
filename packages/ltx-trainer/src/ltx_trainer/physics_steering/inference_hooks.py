"""Register inference-time CAV hooks on VideoMAE transformer blocks (Eq. 5)."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor

from ltx_trainer.physics_steering.steering import steer_hidden_states
from ltx_trainer.physics_steering.videomae_hooks import VideoMAEActivationCollector


class PhysicsSteeringHookManager:
    """Apply ``H̃ = H + α v`` at selected PEZ layers during forward pass."""

    def __init__(
        self,
        model: Any,
        cav_by_layer: dict[int, Tensor],
        *,
        alpha: float = 5.0,
        num_layers: int = 12,
    ) -> None:
        self.model = model
        self.cav_by_layer = {k: v.detach() for k, v in cav_by_layer.items()}
        self.alpha = alpha
        self._collector = VideoMAEActivationCollector(model, num_layers=num_layers)
        self._steer_handles: list[Any] = []

    def _transformer_blocks(self) -> Any:
        return self._collector._transformer_blocks()

    def _make_steering_hook(self, layer: int, cav: Tensor) -> Callable[..., None]:
        alpha = self.alpha

        def hook(_module: Any, _inp: Any, out: Any) -> Tensor | tuple[Tensor, ...]:
            if isinstance(out, tuple):
                steered = steer_hidden_states(out[0], cav, alpha)
                return (steered, *out[1:])
            return steer_hidden_states(out, cav, alpha)

        return hook

    def register_steering(self) -> None:
        """Add CAV injection hooks (call before forward)."""
        self.remove_steering()
        blocks = self._transformer_blocks()
        for layer, cav in self.cav_by_layer.items():
            if layer >= len(blocks):
                continue
            self._steer_handles.append(blocks[layer].register_forward_hook(self._make_steering_hook(layer, cav)))

    def remove_steering(self) -> None:
        for h in self._steer_handles:
            h.remove()
        self._steer_handles.clear()

    def register_probe_collector(self) -> None:
        """Read-only activation hooks (no steering)."""
        self._collector.register()

    def remove_probe_collector(self) -> None:
        self._collector.remove()

    def pooled_activations(self) -> dict[int, Tensor]:
        return self._collector.pooled_by_layer()

    def clear_cache(self) -> None:
        self._collector.clear()

    def __enter__(self) -> PhysicsSteeringHookManager:
        self.register_steering()
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove_steering()
