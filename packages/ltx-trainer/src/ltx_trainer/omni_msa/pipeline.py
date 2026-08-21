"""Omni-modal MSA framework card, tables, and demos (arXiv:2606.05713)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.omni_msa.config import OmniMsaConfig
from ltx_trainer.omni_msa.layout import LIMITATIONS
from ltx_trainer.omni_msa.mock import compare_readouts_smoke, evaluation_smoke, run_discriminative_smoke


def framework_card(cfg: OmniMsaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or OmniMsaConfig()
    return {
        "name": "Discriminative Hidden-State Readout for Omni-Modal MSA",
        "paper": cfg.paper_arxiv,
        "authors": "Bin Wen, Tien-Ping Tan (Universiti Sains Malaysia)",
        "task": "Continuous multimodal sentiment regression on CMU-MOSI / CMU-MOSEI",
        "idea": (
            "Replace generative numeric-string decoding with a single-forward-pass discriminative "
            "readout: pool last non-padding Thinker hidden state → MLP → sentiment in [−3, +3]. "
            "Backbone: Qwen2.5-Omni-7B Thinker + 4-bit QLoRA (~1.14% trainable)."
        ),
        "readout": {
            "discriminative": "z = h_ℓ, MLP 3584→256→1, L1 on normalized labels",
            "generative_baseline": "Greedy decode numeric string + regex parse (unparsable/OOB)",
        },
        "backbone": cfg.backbone,
        "adaptation": {
            "quant": f"{cfg.quant_bits}-bit {cfg.quant_dtype.upper()}",
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "trainable_pct": cfg.trainable_param_pct,
            "peak_memory_gb": f"{cfg.peak_memory_gb_min}–{cfg.peak_memory_gb_max}",
        },
        "datasets": list(cfg.datasets),
        "metrics": ["MAE", "Corr", "Acc-7", "Acc-2", "F1", "unparsable%", "OOB%"],
        "defaults": cfg.__dict__,
    }


def table_i_mosei_sota() -> list[dict[str, float | str]]:
    """Table 1 — CMU-MOSEI state-of-the-art comparison."""
    rows = [
        ("TFN", 0.593, 0.700, 50.2, 82.5, 82.1),
        ("LMF", 0.623, 0.677, 48.0, 82.0, 82.1),
        ("MFN", 0.568, 0.717, 51.1, 84.0, 83.9),
        ("MulT", 0.580, 0.703, 51.8, 82.5, 82.3),
        ("MISA", 0.555, 0.756, 52.2, 85.5, 85.3),
        ("MAG-BERT", 0.539, 0.753, 52.7, 85.2, 85.1),
        ("Self-MM", 0.530, 0.765, 53.6, 85.2, 85.3),
        ("MMIM", 0.526, 0.772, 54.2, 85.9, 85.3),
        ("ConFEDE", 0.522, 0.780, 54.9, 85.8, 85.8),
        ("DMD", 0.532, 0.766, 54.0, 86.0, 85.9),
        ("ALMT", 0.526, 0.779, 53.7, 86.4, 86.4),
        ("MSAmba", 0.521, 0.781, 54.4, 86.5, 86.4),
        ("MEMMI", 0.526, 0.779, 54.2, 86.0, 86.0),
        ("DecAlign", 0.543, 0.768, 55.0, 86.5, 86.1),
        ("Ours (discriminative)", 0.506, 0.790, 55.0, 87.1, 87.0),
    ]
    return [
        {
            "method": name,
            "mae": mae,
            "corr": corr,
            "acc7": acc7,
            "acc2": acc2,
            "f1": f1,
        }
        for name, mae, corr, acc7, acc2, f1 in rows
    ]


def table_ii_mosi_sota() -> list[dict[str, float | str]]:
    """Table 2 — CMU-MOSI state-of-the-art comparison."""
    rows = [
        ("TFN", 0.947, 0.673, 34.5, 79.1, 79.1),
        ("LMF", 0.950, 0.651, 33.8, 79.2, 79.2),
        ("MulT", 0.880, 0.702, 36.9, 81.0, 81.0),
        ("MISA", 0.777, 0.778, 41.4, 83.5, 83.6),
        ("MAG-BERT", 0.731, 0.789, 43.6, 84.3, 84.3),
        ("Self-MM", 0.713, 0.798, 46.7, 85.0, 84.9),
        ("MMIM", 0.700, 0.800, 46.7, 85.1, 85.0),
        ("ConFEDE", 0.742, 0.784, 42.3, 85.5, 85.5),
        ("DMD", 0.723, 0.794, 45.6, 85.7, 85.6),
        ("ALMT", 0.683, 0.805, 47.9, 85.6, 85.6),
        ("MSAmba", 0.681, 0.806, 47.0, 86.0, 86.0),
        ("Ours (discriminative)", 0.551, 0.888, 52.9, 89.5, 89.5),
    ]
    return [
        {
            "method": name,
            "mae": mae,
            "corr": corr,
            "acc7": acc7,
            "acc2": acc2,
            "f1": f1,
        }
        for name, mae, corr, acc7, acc2, f1 in rows
    ]


def table_iii_seed_stability() -> list[dict[str, float | str]]:
    """Table 3 — four-seed stability on CMU-MOSI."""
    seeds = [
        ("A", 0.551, 0.888, 52.9, 89.5, 89.5),
        ("B", 0.569, 0.865, 52.3, 88.8, 88.7),
        ("C", 0.576, 0.879, 51.0, 88.4, 88.5),
        ("D", 0.584, 0.875, 53.0, 89.8, 89.8),
    ]
    rows = [
        {
            "seed": s,
            "mae": mae,
            "corr": corr,
            "acc7": acc7,
            "acc2": acc2,
            "f1": f1,
        }
        for s, mae, corr, acc7, acc2, f1 in seeds
    ]
    rows.append(
        {
            "seed": "Mean ± std",
            "mae": "0.570 ± 0.014",
            "corr": "0.877 ± 0.009",
            "acc7": "52.3 ± 0.9",
            "acc2": "89.1 ± 0.7",
            "f1": "89.1 ± 0.7",
        }
    )
    return rows


def table_iv_readout_controlled() -> list[dict[str, float | str]]:
    """Table 4 — controlled discriminative vs generative readout."""
    return [
        {
            "dataset": "MOSI",
            "metric": "MAE",
            "discriminative": 0.667,
            "generative_zero_shot": 1.443,
            "generative_trained": 1.521,
            "lower_is_better": True,
        },
        {
            "dataset": "MOSI",
            "metric": "Corr",
            "discriminative": 0.824,
            "generative_zero_shot": 0.491,
            "generative_trained": 0.197,
            "lower_is_better": False,
        },
        {
            "dataset": "MOSI",
            "metric": "Acc-2",
            "discriminative": 85.4,
            "generative_zero_shot": 73.3,
            "generative_trained": 58.2,
            "lower_is_better": False,
        },
        {
            "dataset": "MOSEI",
            "metric": "MAE",
            "discriminative": 0.521,
            "generative_zero_shot": 1.431,
            "generative_trained": None,
            "lower_is_better": True,
        },
        {
            "dataset": "MOSEI",
            "metric": "Corr",
            "discriminative": 0.790,
            "generative_zero_shot": 0.473,
            "generative_trained": None,
            "lower_is_better": False,
        },
        {
            "dataset": "MOSI",
            "metric": "Unparsable %",
            "discriminative": 0.0,
            "generative_zero_shot": 2.8,
            "generative_trained": 0.0,
            "lower_is_better": True,
        },
        {
            "dataset": "MOSI",
            "metric": "OOB %",
            "discriminative": 0.0,
            "generative_zero_shot": 0.05,
            "generative_trained": 0.0,
            "lower_is_better": True,
        },
        {
            "dataset": "MOSI",
            "metric": "Peak mem (GB)",
            "discriminative": 10.78,
            "generative_zero_shot": 10.78,
            "generative_trained": None,
            "lower_is_better": True,
        },
        {
            "dataset": "MOSI",
            "metric": "Inf time (s/sample)",
            "discriminative": 1.14,
            "generative_zero_shot": 1.47,
            "generative_trained": None,
            "lower_is_better": True,
        },
    ]


def table_v_modality_ablation() -> list[dict[str, float | str]]:
    """Table 5 — modality ablation on CMU-MOSI (controlled budget)."""
    return [
        {"configuration": "Text only", "mae": 0.552, "corr": 0.883, "acc2": 87.3, "f1": 87.3, "acc7": None},
        {"configuration": "Text + Video", "mae": 0.631, "corr": 0.858, "acc2": 88.1, "f1": 88.0, "acc7": 46.8},
        {"configuration": "Full (T+V+A)", "mae": 0.667, "corr": 0.824, "acc2": 85.4, "f1": 85.4, "acc7": None},
    ]


def table_vi_audio_denoising() -> list[dict[str, float | str]]:
    """Table 6 — DeepFilterNet denoising ablation on CMU-MOSI."""
    return [
        {"configuration": "Original audio", "mae": 0.598, "corr": 0.878, "acc2": 89.63},
        {"configuration": "Denoised audio", "mae": 0.551, "corr": 0.888, "acc2": 89.52},
    ]


def figure3_readout_summary() -> dict[str, Any]:
    """Fig. 3 — controlled readout comparison highlights on CMU-MOSI."""
    return {
        "mae": {"discriminative": 0.667, "generative_zero_shot": 1.443, "generative_trained": 1.521},
        "corr": {"discriminative": 0.824, "generative_zero_shot": 0.491, "generative_trained": 0.197},
        "acc2": {"discriminative": 85.4, "generative_zero_shot": 73.3, "generative_trained": 58.2},
        "unparsable_pct": {"discriminative": 0.0, "generative_zero_shot": 2.8, "generative_trained": 0.0},
    }


def representation_analysis() -> dict[str, float]:
    """Section 4.7 — kNN label smoothness on CMU-MOSI test hidden states."""
    return {
        "knn_label_diff": 0.950,
        "permuted_baseline": 1.896,
        "reduction_pct": 49.9,
        "k_sensitivity_range_reduction_pct": "0.49–0.51 for k=5,10,20",
    }


def pipeline_demo(cfg: OmniMsaConfig | None = None, *, seed: int = 42) -> dict[str, Any]:
    cfg = cfg or OmniMsaConfig()
    return {
        "discriminative_smoke": run_discriminative_smoke(seed=seed),
        "readout_comparison": compare_readouts_smoke(seed=seed),
        "paper_mosi_mae": 0.551,
        "paper_mosei_mae": 0.506,
        "trainable_param_pct": cfg.trainable_param_pct,
    }


def evaluation_demo(cfg: OmniMsaConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    demo["evaluation_smoke"] = evaluation_smoke(seed=0)
    demo["ours_mosei_mae"] = next(r["mae"] for r in table_i_mosei_sota() if "Ours" in str(r["method"]))
    demo["ours_mosi_corr"] = next(r["corr"] for r in table_ii_mosi_sota() if "Ours" in str(r["method"]))
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_i_mosei_sota": table_i_mosei_sota(),
        "table_ii_mosi_sota": table_ii_mosi_sota(),
        "table_iii_seed_stability": table_iii_seed_stability(),
        "table_iv_readout_controlled": table_iv_readout_controlled(),
        "table_v_modality_ablation": table_v_modality_ablation(),
        "table_vi_audio_denoising": table_vi_audio_denoising(),
        "figure3_readout_summary": figure3_readout_summary(),
        "representation_analysis": representation_analysis(),
    }
