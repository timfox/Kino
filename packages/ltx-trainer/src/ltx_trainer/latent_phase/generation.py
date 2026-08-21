"""Generation metrics: self-FID and MSE trade-off (Sec. 5.1, Appendix D)."""

from __future__ import annotations

from typing import Any


def generation_results() -> list[dict[str, Any]]:
    """CIFAR-10 and CelebA64 compressed vs standard VAE (Ascárate et al. 2025 follow-on)."""
    return [
        {
            "dataset": "CIFAR-10",
            "model": "VAE",
            "self_fid": 118.0,
            "mse": 24.5,
        },
        {
            "dataset": "CIFAR-10",
            "model": "Comp.VAE",
            "self_fid": 108.0,
            "mse": 23.5,
        },
        {
            "dataset": "CelebA64",
            "model": "VAE",
            "self_fid": 95.0,
            "mse": 18.2,
        },
        {
            "dataset": "CelebA64",
            "model": "Comp.VAE",
            "self_fid": 86.0,
            "mse": 17.4,
        },
    ]


def generation_improvement(dataset: str = "CIFAR-10") -> dict[str, float]:
    rows = [r for r in generation_results() if r["dataset"] == dataset]
    base = next(r for r in rows if r["model"] == "VAE")
    comp = next(r for r in rows if r["model"] == "Comp.VAE")
    return {
        "delta_self_fid": float(base["self_fid"] - comp["self_fid"]),
        "delta_mse": float(base["mse"] - comp["mse"]),
        "self_fid_ratio": float(comp["self_fid"] / base["self_fid"]),
    }
