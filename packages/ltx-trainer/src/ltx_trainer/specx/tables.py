"""Paper benchmark tables for SpecX (arXiv:2605.18791)."""

from __future__ import annotations

from typing import Any


def table1_benchmark_comparison() -> list[dict[str, Any]]:
    """Table 1 — comparison with existing spectral benchmarks."""
    return [
        {"benchmark": "NovoBench", "scale": "small", "multi_spec": False, "experimental": False, "model_coverage": "MLLM"},
        {"benchmark": "MolPuzzle", "scale": "small", "multi_spec": False, "experimental": False, "model_coverage": "MLLM"},
        {"benchmark": "Multimodal Spec", "scale": "medium", "multi_spec": "partial", "experimental": "partial", "model_coverage": "ML"},
        {"benchmark": "MassSpecGym", "scale": "medium", "multi_spec": False, "experimental": True, "model_coverage": "ML"},
        {"benchmark": "SpecX (Ours)", "scale": "1.7M", "multi_spec": True, "experimental": True, "model_coverage": "MLLM+ML"},
    ]


def table3_elucidation_random_excerpt() -> list[dict[str, Any]]:
    """Table 3 excerpt — Top-1 % (random split, selected modalities)."""
    return [
        {"modality": "1H-NMR", "top1": 51.52, "top5": 65.74, "top10": 68.18},
        {"modality": "13C-NMR", "top1": 47.48, "top5": 65.92, "top10": 69.49},
        {"modality": "13C+1H+MS", "top1": 59.04, "top5": 78.59, "top10": 81.77},
        {"modality": "IR", "top1": 4.04, "top5": 9.48, "top10": 11.31},
        {"modality": "UV-Vis", "top1": 0.69, "top5": 4.85, "top10": 8.78},
        {"modality": "Raman", "top1": 25.59, "top5": 47.17, "top10": 53.23},
    ]


def table3_elucidation_scaffold_excerpt() -> list[dict[str, Any]]:
    """Table 3 excerpt — Top-1 % (scaffold split)."""
    return [
        {"modality": "1H-NMR", "top1": 25.00},
        {"modality": "13C+1H+MS", "top1": 29.66},
        {"modality": "IR", "top1": 0.94},
        {"modality": "UV-Vis", "top1": 0.10},
    ]


def table4_functional_group_random() -> list[dict[str, Any]]:
    """Table 4 excerpt — macro-F1 XGBoost random split."""
    return [
        {"modality": "Raman", "xgboost": 0.958, "cnn1d": 0.878},
        {"modality": "IR", "xgboost": 0.824, "cnn1d": 0.775},
        {"modality": "1H-NMR", "xgboost": 0.716, "cnn1d": 0.670},
        {"modality": "13C-NMR", "xgboost": 0.676, "cnn1d": 0.610},
        {"modality": "MS", "xgboost": 0.627, "cnn1d": 0.608},
        {"modality": "UV-Vis", "xgboost": 0.530, "cnn1d": 0.489},
    ]


def table6_qa_smiles_small_excerpt() -> list[dict[str, Any]]:
    """Table 6 excerpt — MLLM QA-1 on Small subset (random, T1)."""
    return [
        {"setting": "1H-NMR DeepSeek-V3", "t1": 0.006, "t5": 0.011, "t10": 0.011},
        {"setting": "Multi DeepSeek-V3", "t1": 0.015, "t5": 0.035, "t10": 0.042},
        {"setting": "Multi GPT-4.1-mini", "t1": 0.008, "t5": 0.020, "t10": 0.024},
    ]


def table11_subset_modalities() -> list[dict[str, Any]]:
    """Table 11 — subset modality coverage."""
    return [
        {"subset": "Large", "molecules": "~1,000,000", "modalities": 7, "fl_in_benchmark": False},
        {"subset": "Small", "molecules": 4496, "modalities": 7, "fl_in_benchmark": False},
        {"subset": "Exp", "molecules": 432, "modalities": 2, "modalities_list": "MS, UV"},
    ]


def modality_representations() -> list[dict[str, Any]]:
    """Table 2 / 13 summary — data representations per modality."""
    return [
        {"modality": "IR", "representation": "vector 1800 (400–4000 cm⁻¹)"},
        {"modality": "1H-NMR", "representation": "peak list + annotated spectrum"},
        {"modality": "13C-NMR", "representation": "peak list"},
        {"modality": "HSQC-NMR", "representation": "512×512 matrix"},
        {"modality": "MS/MS", "representation": "m/z, intensity, formula"},
        {"modality": "UV", "representation": "~1000 points (200–800 nm)"},
        {"modality": "Raman", "representation": "~1500 points (100–3500 cm⁻¹)"},
        {"modality": "Fluorescence", "representation": "emission ~1000 (300–800 nm)"},
    ]
