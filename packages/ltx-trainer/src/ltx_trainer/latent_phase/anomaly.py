"""Latent k-NN anomaly detection (Sec. 5.2, Ascárate et al. 2026 follow-on)."""

from __future__ import annotations

from typing import Any


def unsupervised_ad_results() -> list[dict[str, Any]]:
    return [
        {"dataset": "Mars Rover Mastcam", "model": "VAE+kNN", "auroc": 0.66},
        {"dataset": "Mars Rover Mastcam", "model": "Comp.VAE+kNN", "auroc": 0.76},
        {"dataset": "Galaxy Zoo 64", "model": "VAE+kNN", "auroc": 0.74},
        {"dataset": "Galaxy Zoo 64", "model": "Comp.VAE+kNN", "auroc": 0.79},
    ]


def ood_fpr95_results() -> list[dict[str, Any]]:
    return [
        {"setting": "CIFAR-10 ID / far-OOD avg", "baseline_fpr95": 0.42, "compressed_fpr95": 0.32},
        {"setting": "CIFAR-10 ID / CIFAR-100 near-OOD", "baseline_fpr95": 0.55, "compressed_fpr95": 0.25},
        {"setting": "Imagenette ID / near ImageNet OOD", "baseline_fpr95": 0.38, "compressed_fpr95": 0.28},
    ]


def ad_improvement(dataset: str = "Galaxy Zoo 64") -> dict[str, float]:
    rows = [r for r in unsupervised_ad_results() if r["dataset"] == dataset]
    base = next(r for r in rows if "Comp" not in r["model"])
    comp = next(r for r in rows if "Comp" in r["model"])
    return {
        "delta_auroc": float(comp["auroc"] - base["auroc"]),
        "baseline_auroc": float(base["auroc"]),
        "compressed_auroc": float(comp["auroc"]),
    }
