"""Toy decoders for LOPEO leakage demonstration."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.lopeo.metrics import decoding_accuracy, pearson_corr


class IdentityMemorizingDecoder:
    """Memorizes attended speaker; inflates accuracy when test speaker was seen in train."""

    def __init__(self, *, leak_boost: float = 0.0) -> None:
        self.leak_boost = leak_boost
        self._seen: set[str] = set()

    def fit(self, trials: list[dict[str, Any]]) -> None:
        self._seen = {str(t.get("attended", "")) for t in trials}

    def evaluate(self, trials: list[dict[str, Any]]) -> float:
        correct = 0
        for t in trials:
            att = str(t.get("attended", ""))
            pred = att if att in self._seen else "unknown"
            if pred == att:
                correct += 1
            elif att in self._seen:
                correct += self.leak_boost
        return correct / max(len(trials), 1)


def envelope_reconstruction_demo(*, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(64)
    pred = target + 0.05 * rng.standard_normal(64)
    rho = pearson_corr(target, pred)
    return {"envelope_rho": round(float(rho), 4), "envelope_mse": round(float(np.mean((target - pred) ** 2)), 4)}


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    demo = envelope_reconstruction_demo(seed=seed)
    trials = [
        {"attended": "spk_a", "unattended": "spk_b"},
        {"attended": "spk_a", "unattended": "spk_b"},
    ]
    dec = IdentityMemorizingDecoder()
    dec.fit(trials)
    acc = dec.evaluate(trials)
    return {**demo, "leakage_accuracy": round(acc, 3)}
