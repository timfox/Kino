"""SpecX multimodal spectroscopy benchmark constants (arXiv:2605.18791)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SpecxConfig:
    paper_arxiv: str = "2605.18791"
    paper_title: str = (
        "SpecX: A Large-Scale Benchmark for Multi-Modal Spectroscopy and Cross-Paradigm Evaluation"
    )
    institution: str = "Hunan University"

    total_molecules_filtered: int = 1_701_739
    retention_rate_pct: float = 62.36

    tier_large_molecules: int = 1_000_000
    tier_small_molecules: int = 4_496
    tier_exp_molecules: int = 432

    modalities_all: tuple[str, ...] = (
        "1H-NMR",
        "13C-NMR",
        "HSQC-NMR",
        "IR",
        "MS",
        "UV",
        "Raman",
        "Fluorescence",
    )
    modalities_large_small: tuple[str, ...] = (
        "1H-NMR",
        "13C-NMR",
        "HSQC-NMR",
        "IR",
        "MS",
        "UV",
        "Raman",
    )
    modalities_exp: tuple[str, ...] = ("MS", "UV")

    num_functional_groups: int = 37
    heavy_atom_min: int = 5
    heavy_atom_max: int = 35

    tasks_ml: tuple[str, ...] = (
        "spectra_to_smiles",
        "functional_group_prediction",
        "smiles_to_spectra",
    )
    tasks_qa: tuple[str, ...] = ("qa_smiles_inference", "qa_functional_groups")

    split_strategies: tuple[str, ...] = ("random", "scaffold")

    sources: tuple[str, ...] = (
        "QME14S",
        "ViBench",
        "ChEMBL",
        "Multimodal Spectroscopic Dataset",
        "MassSpecGym",
    )

    mllm_eval_models: tuple[str, ...] = ("DeepSeek-V3", "GPT-4.1-mini", "Qwen2.5-3B")
