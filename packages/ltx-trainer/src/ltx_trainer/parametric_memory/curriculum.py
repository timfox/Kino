"""PhoneBook inter-batch curriculum (Appendix D, Tables 5–6)."""

from __future__ import annotations

from typing import Any

# Exposure ratios shared by Qwen and Llama PhoneBook MemFT-SW runs
CURRICULUM_EXPOSURE: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


def phonebook_curriculum_qwen() -> list[dict[str, Any]]:
    """Table 5 — Qwen3-8B-Instruct."""
    return [
        {"length_tokens": 1000, "approx_samples": 100, "lr": 1e-2, "epochs": 300, "boundaries": [20, 40, 60, 80, 300]},
        {"length_tokens": 2000, "approx_samples": 200, "lr": 1e-2, "epochs": 300, "boundaries": [20, 40, 60, 80, 300]},
        {"length_tokens": 4000, "approx_samples": 400, "lr": 1e-2, "epochs": 350, "boundaries": [20, 40, 60, 80, 350]},
        {"length_tokens": 8000, "approx_samples": 800, "lr": 7e-3, "epochs": 350, "boundaries": [30, 60, 90, 120, 350]},
        {"length_tokens": 12000, "approx_samples": 1200, "lr": 5e-3, "epochs": 500, "boundaries": [60, 120, 180, 240, 500]},
        {"length_tokens": 16000, "approx_samples": 1600, "lr": 5e-3, "epochs": 600, "boundaries": [80, 160, 240, 320, 600]},
        {"length_tokens": 24000, "approx_samples": 2400, "lr": 5e-3, "epochs": 600, "boundaries": [80, 160, 240, 320, 600]},
        {"length_tokens": 32000, "approx_samples": 3200, "lr": 5e-3, "epochs": 700, "boundaries": [100, 200, 300, 400, 700]},
    ]


def phonebook_curriculum_llama() -> list[dict[str, Any]]:
    """Table 6 — Llama3.1-8B-Instruct."""
    return [
        {"length_tokens": 1000, "approx_samples": 250, "lr": 1e-2, "epochs": 300, "boundaries": [20, 40, 60, 80, 300]},
        {"length_tokens": 2000, "approx_samples": 500, "lr": 1e-2, "epochs": 400, "boundaries": [40, 80, 120, 160, 400]},
        {"length_tokens": 4000, "approx_samples": 1000, "lr": 7e-3, "epochs": 400, "boundaries": [40, 80, 120, 160, 400]},
        {"length_tokens": 8000, "approx_samples": 2000, "lr": 5e-3, "epochs": 600, "boundaries": [80, 160, 240, 320, 600]},
        {"length_tokens": 12000, "approx_samples": 3000, "lr": 5e-3, "epochs": 700, "boundaries": [100, 200, 300, 400, 700]},
        {"length_tokens": 16000, "approx_samples": 4000, "lr": 5e-3, "epochs": 700, "boundaries": [100, 200, 300, 400, 700]},
        {"length_tokens": 24000, "approx_samples": 6000, "lr": 3e-3, "epochs": 700, "boundaries": [100, 200, 300, 400, 700]},
        {"length_tokens": 32000, "approx_samples": 8000, "lr": 3e-3, "epochs": 800, "boundaries": [120, 240, 360, 480, 800]},
    ]


def exposure_at_epoch(epoch: int, boundaries: list[int]) -> float:
    """Map epoch to curriculum exposure ratio using boundary list."""
    for i, bound in enumerate(boundaries):
        if epoch <= bound:
            return CURRICULUM_EXPOSURE[min(i, len(CURRICULUM_EXPOSURE) - 1)]
    return 1.0
