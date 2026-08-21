"""Paper metrics and reference tables (Sec. 3)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.bilt.config import NUM_SAMPLES, PARAMS_M, SPECTRAL_POINTS


def table1_architecture() -> list[dict[str, Any]]:
  """Layer stack from Table 1."""
  return [
      {"layer": 1, "type": "Conv1D", "params": "128 filters, k=23", "activation": "gelu"},
      {"layer": 2, "type": "LayerNorm"},
      {"layer": 3, "type": "MaxPool1D", "params": "pool=3, stride=1"},
      {"layer": 4, "type": "LayerNorm"},
      {"layer": 5, "type": "ImportanceGating", "params": "3→16→1 MLP", "optional": True},
      {"layer": 6, "type": "PositionalEmbedding", "params": "150×128"},
      {"layer": 7, "type": "CrossAttention", "params": "16 probes, 8 heads"},
      {"layer": 8, "type": "LayerNorm+residual"},
      {"layer": 9, "type": "SelfAttention", "params": "8 heads"},
      {"layer": 10, "type": "LayerNorm+residual"},
      {"layer": 11, "type": "Dense", "params": 256, "activation": "gelu"},
      {"layer": 12, "type": "Dense", "params": 128},
      {"layer": 13, "type": "LayerNorm+residual"},
      {"layer": 14, "type": "Flatten", "params": "16×128=2048"},
      {"layer": 15, "type": "Dense", "params": 64, "activation": "gelu"},
      {"layer": 16, "type": "Latent", "params": "3, L1=1e-4", "activation": "softplus"},
      {"layer": 17, "type": "PhysicsDecoder", "params": "300 linear, no bias"},
      {"layer": 18, "type": "Reshape", "params": "(150, 2)"},
  ]


def table2_training_phases() -> list[dict[str, Any]]:
    return [
        {"phase": 1, "epochs": 1000, "lr": "cos 1e-3→2e-5", "augment": "off"},
        {"phase": 2, "epochs": 4000, "lr": "2e-5 const", "augment": "linear 0→1"},
        {"phase": 3, "epochs": 3000, "lr": "cos 2e-5→2e-6", "augment": "full", "early_stop_patience": 1500},
    ]


def table3_performance() -> list[dict[str, Any]]:
    """Table 3: predictive performance."""
    return [
        {"dataset": "Training (clean)", "r2_mu_a": 0.991, "mape_mu_a": 0.093, "r2_mu_s": 0.991, "mape_mu_s": 0.090},
        {"dataset": "Test (clean)", "r2_mu_a": 0.979, "mape_mu_a": 0.162, "r2_mu_s": 0.975, "mape_mu_s": 0.097},
        {"dataset": "Test (noisy/shifted)", "r2_mu_a": 0.917, "mape_mu_a": 0.253, "r2_mu_s": 0.917, "mape_mu_s": 0.199},
        {"dataset": "Sim spectrometer (no blur)", "r2_mu_a": 0.975, "r2_mu_s": 0.976},
        {"dataset": "Sim spectrometer FWHM≈4.7nm", "r2_mu_a": 0.972, "r2_mu_s": 0.974},
        {"dataset": "Sim spectrometer FWHM≈14.1nm", "r2_mu_a": 0.971, "r2_mu_s": 0.975},
        {"dataset": "Sim spectrometer FWHM≈23.6nm", "r2_mu_a": 0.960, "r2_mu_s": 0.974},
    ]


def table_complexity() -> dict[str, Any]:
    return {
        "params_m": PARAMS_M,
        "spectral_points": SPECTRAL_POINTS,
        "num_samples": NUM_SAMPLES,
        "latent_neurons": 3,
        "constituents": ["intralipid (scatterer)", "red ink", "black ink"],
    }


def r2_score(y_true, y_pred) -> float:
    import torch

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot.item() == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mape(y_true, y_pred, *, threshold_frac: float = 0.01) -> float:
    import torch

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    ymax = y_true.max()
    mask = y_true > threshold_frac * ymax
    if mask.sum() == 0:
        return float("nan")
    rel = ((y_true[mask] - y_pred[mask]).abs() / y_true[mask].abs()).mean()
    return float(rel)
