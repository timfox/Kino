"""NN regression stubs for c and κ prediction (Sec. 2.3, 3.3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sle_nn.config import NNArchitecture, SleNNConfig


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def predict_with_noise(
    targets: np.ndarray,
    *,
    test_mse: float,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic predictions achieving a target test MSE."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets, dtype=np.float64)
    noise = rng.normal(scale=np.sqrt(test_mse), size=targets.shape)
    return targets + noise


def deterministic_c_predictions(
    c_true: np.ndarray,
    *,
    cfg: SleNNConfig | None = None,
    seed: int = 0,
) -> np.ndarray:
    cfg = cfg or SleNNConfig()
    return predict_with_noise(c_true, test_mse=cfg.test_mse_deterministic_c, seed=seed)


def sle_kappa_predictions(
    kappa_true: np.ndarray,
    *,
    same_noise: bool,
    cfg: SleNNConfig | None = None,
    seed: int = 0,
) -> np.ndarray:
    cfg = cfg or SleNNConfig()
    target = cfg.test_mse_sle_same_noise if same_noise else cfg.test_mse_sle_different_noise
    return predict_with_noise(kappa_true, test_mse=target, seed=seed)


def sle_trace_kappa_predictions(
    kappa_true: np.ndarray,
    *,
    cfg: SleNNConfig | None = None,
    seed: int = 0,
) -> np.ndarray:
    cfg = cfg or SleNNConfig()
    return predict_with_noise(kappa_true, test_mse=cfg.test_mse_sle_trace_fixed_bm, seed=seed)


def architecture_summary(arch: NNArchitecture | None = None) -> dict[str, Any]:
    arch = arch or NNArchitecture()
    return {
        "deterministic": {
            "layers": list(arch.deterministic_hidden) + [1],
            "activations": ["relu", "relu", "linear"],
            "loss": "mse",
            "optimizer": "adam",
        },
        "sle_kappa": {
            "layers": ["flatten", list(arch.sle_hidden) + [1]],
            "dropout": arch.dropout_rate,
            "activations": ["relu", "relu", "linear"],
            "loss": "mse",
            "optimizer": "adam",
        },
    }
