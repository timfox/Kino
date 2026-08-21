"""Table anchors from Poppi et al. (arXiv:2606.05290)."""

from __future__ import annotations


def table1_flux1_schnell_llama_svd() -> dict[str, float]:
    """Table 1: Flux1-Schnell, Llama3.1-8B, SVD, α=5."""
    return {"asr": 0.038, "clip_sim": 0.308, "fid": 34.4}


def table1_original() -> dict[str, dict[str, float]]:
    """Table 1 original (α=0) rows."""
    return {
        "Flux1-Schnell": {"asr": 0.307, "clip_sim": 0.319, "fid": 29.2},
        "Flux1-Dev": {"asr": 0.286, "clip_sim": 0.309, "fid": 34.5},
        "Qwen-Image": {"asr": 0.384, "clip_sim": 0.332, "fid": 31.5},
        "Z-Image-Turbo": {"asr": 0.304, "clip_sim": 0.319, "fid": 31.7},
    }


def table1_native() -> dict[str, float]:
    """Native target oracle ASR (Flux1-Schnell)."""
    return {"asr": 0.085, "clip_sim": 0.306, "fid": 35.3}


def table1_transferred_svd() -> dict[str, dict[str, float]]:
    """Transferred SVD ASR highlights (Llama source, α=5/3)."""
    return {
        ("Flux1-Schnell", "Llama3.1-8B"): {"asr": 0.038, "clip_sim": 0.308},
        ("Flux1-Dev", "Mistral-7B"): {"asr": 0.037, "clip_sim": 0.278},
        ("Z-Image-Turbo", "Llama3.1-8B"): {"asr": 0.002, "clip_sim": 0.249},
    }


def table2_mma_flux1_dev_svd() -> float:
    """Table 2 MMA-Diffusion ASR for Flux1-Dev + Llama SVD."""
    return 0.020


def alpha_sweep() -> tuple[float, ...]:
    return (-1.0, 0.0, 1.0, 3.0, 5.0, 7.0)
