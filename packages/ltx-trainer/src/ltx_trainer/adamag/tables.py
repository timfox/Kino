"""Paper tables for AdaMaG (arXiv:2605.20079)."""

from __future__ import annotations

from typing import Any


def table1_optimal_guidance() -> list[dict[str, Any]]:
    """Table 1 — optimal guidance scale (FID×1, IS, SAT)."""
    return [
        {"method": "CFG", "SD3_fid": 32.4, "SD3_is": 33.2, "SD3_sat": 0.53,
         "SD35_fid": 35.8, "Flux_fid": 36.1, "Flux_sat": 0.38},
        {"method": "Rect-CFG++", "SD3_fid": 32.4, "SD3_sat": 0.55, "SD35_fid": 39.6, "Flux_fid": 35.7},
        {"method": "TAG", "SD3_fid": 32.4, "SD35_fid": 35.2, "Flux_fid": 36.1},
        {"method": "APG", "SD3_fid": 39.0, "SD35_fid": 54.9, "Flux_fid": 34.3},
        {"method": "Ours", "SD3_fid": 30.4, "SD3_is": 32.9, "SD3_sat": 0.51,
         "SD35_fid": 32.1, "SD35_sat": 0.48, "Flux_fid": 34.8, "Flux_sat": 0.34},
    ]


def table1_high_guidance() -> list[dict[str, Any]]:
    """Table 1 — high-stress guidance (SD3 ω=15, SD3.5 ω=15, Flux ω=3)."""
    return [
        {"method": "CFG", "SD3_fid": 42.6, "SD3_sat": 0.63, "SD35_fid": 62.7, "Flux_fid": 37.8},
        {"method": "Ours", "SD3_fid": 35.6, "SD3_sat": 0.59, "SD35_fid": 54.6, "Flux_fid": 36.9},
    ]


def table2_compbench_sd3() -> list[dict[str, Any]]:
    """Table 2 — T2I-CompBench (SD3), CFG vs +Ours."""
    return [
        {"method": "CFG", "color": 0.7374, "shape": 0.5789, "texture": 0.7131, "spatial": 0.3211},
        {"method": "+ Ours", "color": 0.8134, "shape": 0.5811, "texture": 0.7790, "spatial": 0.3241},
    ]


def table3_hires_sd3() -> dict[str, Any]:
    """Table 3 — 1024×1024 SD3."""
    return {
        "CFG": {"fid": 23.89, "ir": 0.98, "pickscore": 0.441, "hpsv2": 0.275, "sat": 0.51},
        "Ours": {"fid": 22.34, "ir": 1.043, "pickscore": 0.557, "hpsv2": 0.288, "sat": 0.48},
    }


def table4_beta_ablation() -> list[dict[str, Any]]:
    """Table 4 — score-parallel damping β (SD3-scale study)."""
    return [
        {"beta": "CFG", "fid": 32.64, "is": 33.04, "sat": 0.52, "recall": 0.68},
        {"beta": 0.0, "fid": 31.47, "is": 32.67, "sat": 0.50, "recall": 0.70},
        {"beta": 0.1, "fid": 31.39, "is": 33.17, "sat": 0.50, "recall": 0.70},
        {"beta": 1.0, "fid": 31.55, "is": 33.12, "sat": 0.50, "recall": 0.69},
        {"beta": 5.0, "fid": 32.10, "is": 33.13, "sat": 0.51, "recall": 0.70},
    ]


def table5_gamma_ablation() -> list[dict[str, Any]]:
    """Table 5 — schedule exponent γ at β=0.1."""
    return [
        {"gamma": "β=0.1 only", "fid": 31.39, "is": 33.17, "sat": 0.50, "recall": 0.70},
        {"gamma": 0.1, "fid": 31.15, "is": 32.79, "sat": 0.497, "recall": 0.71},
        {"gamma": 1.0, "fid": 29.97, "is": 33.14, "sat": 0.493, "recall": 0.71},
        {"gamma": 2.0, "fid": 29.55, "is": 33.32, "sat": 0.491, "recall": 0.71},
        {"gamma": 4.0, "fid": 29.29, "is": 33.06, "sat": 0.489, "recall": 0.72},
    ]
