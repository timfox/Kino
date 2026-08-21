"""BenchmarkEvaluator stub (Section 3.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ltx_trainer.detectzoo.api import BaseDetector
from ltx_trainer.detectzoo.metrics import metrics_bundle


@dataclass
class DatasetItem:
    input: Any
    label: int  # 0 human, 1 ai


@dataclass
class BenchmarkEvaluator:
    """Runs detectors on a list of DatasetItem and aggregates metrics."""

    dataset_name: str
    items: list[DatasetItem]
    detectors: list[BaseDetector]
    metadata: dict[str, Any] = field(default_factory=dict)

    def run(self) -> dict[str, Any]:
        out: dict[str, Any] = {"dataset": self.dataset_name, "n_samples": len(self.items), "detectors": {}}
        labels = torch.tensor([it.label for it in self.items], dtype=torch.float32)
        for det in self.detectors:
            scores = []
            for it in self.items:
                r = det.predict(it.input)
                scores.append(r.score)
            s = torch.tensor(scores, dtype=torch.float32)
            out["detectors"][det.name] = metrics_bundle(s, labels)
        return out
