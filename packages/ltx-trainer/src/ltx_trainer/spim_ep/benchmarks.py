"""Reference results from Vanden Abeele et al. (arXiv:2606.13454)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.spim_ep.config import (
    MNIST_LAYERED_ACC,
    MNIST_TEST_ACC,
    PAPER_ARXIV,
    PAPER_TITLE,
    PAPER_URL,
    WINE_SIM_ACC,
    WINE_TEST_ACC_EXP,
    WINE_TEST_ACC_STD,
)

TABLE_WINE_EXPERIMENT: dict[str, float | str] = {
    "dataset": "UCI Wine",
    "test_accuracy_pct": WINE_TEST_ACC_EXP,
    "test_accuracy_std": WINE_TEST_ACC_STD,
    "simulation_accuracy_pct": WINE_SIM_ACC,
    "single_step_inference_acc_pct": 85.71,
    "n_input": 13,
    "n_hidden": 5,
    "n_output": 3,
    "rank_K": 20,
}

TABLE_MNIST_ALL_TO_ALL: dict[str, float | str] = {
    "Nd": 510,
    "rank_K": 355,
    "test_accuracy_pct": MNIST_TEST_ACC,
    "test_accuracy_std": 0.15,
    "reference_ep_pct": 97.5,
}

TABLE_MNIST_PRECISION: dict[str, dict[str, float]] = {
    "32-bit": {"acc": 97.68, "inference_steps": 6.60},
    "12-bit": {"acc": 97.84, "inference_steps": 6.59},
    "10-bit": {"acc": 97.50, "inference_steps": 5.85},
    "8-bit": {"acc": 97.67, "inference_steps": 5.45},
    "6-bit": {"acc": 96.61, "inference_steps": 5.96},
    "4-bit": {"acc": 90.60, "inference_steps": 4.41},
}

TABLE_APPENDIX_LR: dict[str, float] = {
    "Original (±π/2 and ±π/4)": 97.47,
    "No self-coupling (±π/2, Approx LR)": 48.77,
    "No self-coupling (±π/2, Exact LR)": 96.68,
    "Main (±π/4, Approx LR)": 97.69,
    "Main (±π/4, Exact LR)": 97.78,
}

TABLE_MNIST_LAYERED: dict[str, float] = {
    "test_accuracy_pct": MNIST_LAYERED_ACC,
    "test_accuracy_std": 0.10,
}


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "paper": {"title": PAPER_TITLE, "arxiv": f"arXiv:{PAPER_ARXIV}", "url": PAPER_URL},
        "wine_experiment": TABLE_WINE_EXPERIMENT,
        "mnist_all_to_all": TABLE_MNIST_ALL_TO_ALL,
        "mnist_precision": TABLE_MNIST_PRECISION,
        "appendix_learning_rules": TABLE_APPENDIX_LR,
        "mnist_layered_architecture": TABLE_MNIST_LAYERED,
    }
