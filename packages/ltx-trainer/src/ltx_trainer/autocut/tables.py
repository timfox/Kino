"""Paper tables (main + ablation + hyperparameters)."""

from __future__ import annotations

from typing import Any


def table1_main_results() -> list[dict[str, Any]]:
    """Table 1 — quantitative results across tasks."""
    return [
        {
            "model": "Qwen3-8B (Caption)",
            "CSA": 0.1374,
            "CRA": 0.01648,
            "VSC": 0.9308,
            "SQ": 79.9947,
            "WCD": 5.2648,
            "MSS": None,
        },
        {
            "model": "Qwen3-8B (Caption + SFT)",
            "CSA": 0.5687,
            "CRA": 0.03022,
            "VSC": 1.1227,
            "SQ": 59.1592,
            "WCD": 6.8230,
            "MSS": None,
        },
        {
            "model": "Qwen2.5-VL-32B Instruct",
            "CSA": 0.6648,
            "CRA": 0.02472,
            "VSC": 0.9980,
            "SQ": 78.3180,
            "WCD": 12.5071,
            "MSS": None,
        },
        {
            "model": "GPT-4o + MGSV",
            "CSA": 0.2689,
            "CRA": 0.07756,
            "VSC": 1.1364,
            "SQ": 83.0290,
            "WCD": 7.7457,
            "MSS": 0.2656,
        },
        {
            "model": "AutoCut (ours)",
            "CSA": 0.6593,
            "CRA": 0.10714,
            "VSC": 1.0360,
            "SQ": 84.6255,
            "WCD": 3.0182,
            "MSS": 0.3475,
        },
    ]


def table2_ablation_training() -> list[dict[str, Any]]:
    return [
        {
            "method": "sft only",
            "CSA": 0.4780,
            "CRA": 0.08242,
            "VSC": 1.0043,
            "SQ": 83.1898,
            "WCD": 4.4346,
        },
        {
            "method": "emb+full+sft",
            "CSA": 0.7170,
            "CRA": 0.05770,
            "VSC": 0.9669,
            "SQ": 78.9644,
            "WCD": 4.4984,
        },
        {
            "method": "emb+sft (ours)",
            "CSA": 0.6593,
            "CRA": 0.10714,
            "VSC": 1.0360,
            "SQ": 84.6255,
            "WCD": 3.0182,
        },
    ]


def table_rqvae_hyperparams() -> list[dict[str, Any]]:
    return [
        {
            "modality": "video",
            "input_feature_size": 128,
            "quantization_heads": 8,
            "codebook_size": 256,
            "encoder_mlp": "[512, 512]",
            "max_epochs": 20,
            "batch_size": 8192,
        },
        {
            "modality": "audio",
            "input_feature_size": 2048,
            "quantization_heads": 8,
            "codebook_size": 256,
            "encoder_mlp": "[1024, 512]",
            "max_epochs": 50,
            "batch_size": 8192,
        },
    ]


def table_alignment_sft_hyperparams() -> list[dict[str, Any]]:
    return [
        {
            "stage": "alignment",
            "base_model": "Qwen3-8B",
            "cutoff": 8096,
            "lr": 5e-5,
            "epochs": 5,
            "packing": True,
            "trainable": "multimodal embeddings only",
        },
        {
            "stage": "sft",
            "base_model": "aligned Qwen3-8B",
            "cutoff": 4000,
            "lr": 1e-5,
            "epochs": 3,
            "packing": False,
            "trainable": "full parameters",
        },
    ]


def user_study_vs_gpt4o() -> dict[str, dict[str, float]]:
    """Fig. 4 win/tie/loss proportions (%)."""
    return {
        "visual_smoothness": {"win": 65, "tie": 18, "loss": 17},
        "logical_consistency": {"win": 68, "tie": 18, "loss": 14},
        "script_visual_alignment": {"win": 70, "tie": 26, "loss": 4},
        "bgm_visual_compatibility": {"win": 52, "tie": 22, "loss": 26},
        "overall_attractiveness": {"win": 65, "tie": 20, "loss": 15},
    }
