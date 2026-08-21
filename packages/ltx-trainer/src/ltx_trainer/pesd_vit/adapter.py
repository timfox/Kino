"""Task-specific bottleneck Adapters for PESD-ViT (Sec. II-B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskAdapter:
    """Bottleneck Adapter: down → LN+GELU → up (zero-init up-projection)."""

    down: np.ndarray  # (d, r)
    up: np.ndarray  # (r, d)
    name: str

    @classmethod
    def init(cls, name: str, d: int, r: int, *, rng: np.random.Generator) -> TaskAdapter:
        down = rng.standard_normal((d, r)) * (1.0 / max(d, 1) ** 0.5)
        up = np.zeros((r, d), dtype=np.float64)
        return cls(down=down, up=up, name=name)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Residual adapter: x + up(gelu(ln(down^T x)))."""
        h = x @ self.down
        h = np.maximum(0.0, h)  # GELU stub (ReLU for numpy smoke)
        return x + h @ self.up


def adapter_param_count(d: int, r: int, num_tasks: int = 3) -> dict[str, float]:
    per_task = d * r + r * d
    return {
        "per_task": float(per_task),
        "total_adapters": float(per_task * num_tasks),
        "full_finetune_swin_t_approx": 28_000_000.0,
        "pct_of_full": float(per_task * num_tasks) / 28_000_000.0 * 100.0,
    }
