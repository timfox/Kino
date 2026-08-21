"""Framework card, paper tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.ssl_soft_infer.config import SslSoftInferConfig
from ltx_trainer.ssl_soft_infer.phoneme import phoneme_separability_demo
from ltx_trainer.ssl_soft_infer.posterior import posterior_demo, tau_for_dataset


def framework_card(cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    c = cfg or SslSoftInferConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "demo_url": c.demo_url,
        "method": "hard_discrete_train + soft_posterior_infer",
        "ssl_models": {
            "hubert_large": c.hubert_hf,
            "wavlm_large": c.wavlm_hf,
            "layer": c.ssl_layer,
        },
        "cluster_sizes": list(c.cluster_sizes),
        "training": {
            "asr_corpus": c.train_data_asr,
            "synth_corpus": c.train_data_synth,
            "train_assignment": c.asr_train_assign,
            "infer_assignment": c.asr_infer_assign,
        },
        "tau_defaults": {
            "librispeech": c.tau_librispeech,
            "ted2": c.tau_ted2,
            "chime4": c.tau_chime4,
            "erj": c.tau_erj,
        },
        "headline": headline_results(c),
    }


def headline_results(cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    c = cfg or SslSoftInferConfig()
    return {
        "wavlm4096_erj_soft_beats_continuous": c.wavlm4096_erj_soft < c.wavlm_cont_erj,
        "wavlm4096_erj_soft": c.wavlm4096_erj_soft,
        "wavlm4096_vc_spksim_soft": c.wavlm4096_vc_spksim_soft,
        "wavlm4096_asr_ratio_soft": c.wavlm4096_asr_ratio_soft,
        "wavlm4layer_erj_soft_ii": c.wavlm4layer_erj_soft_ii,
    }


def table1_asr_wer() -> list[dict[str, Any]]:
    """Table 1 — LibriSpeech-100h ASR WER/CER [%] (selected rows)."""
    return [
        {
            "ssl": "HuBERT",
            "k": "cont.",
            "train": "-",
            "infer": "-",
            "ls_clean": 3.1,
            "ls_other": 5.7,
            "ted2": 10.5,
            "chime4": 52.7,
            "erj": 50.5,
        },
        {
            "ssl": "HuBERT",
            "k": 4096,
            "train": "hard",
            "infer": "soft",
            "ls_clean": 3.9,
            "ls_other": 6.8,
            "ted2": 11.6,
            "chime4": 54.4,
            "erj": 49.4,
        },
        {
            "ssl": "WavLM",
            "k": "cont.",
            "train": "-",
            "infer": "-",
            "ls_clean": 3.0,
            "ls_other": 5.5,
            "ted2": 7.8,
            "chime4": 16.0,
            "erj": 38.9,
        },
        {
            "ssl": "WavLM",
            "k": 4096,
            "train": "hard",
            "infer": "hard",
            "ls_clean": 3.8,
            "ls_other": 6.6,
            "ted2": 10.1,
            "chime4": 19.3,
            "erj": 41.5,
        },
        {
            "ssl": "WavLM",
            "k": 4096,
            "train": "hard",
            "infer": "soft",
            "ls_clean": 3.7,
            "ls_other": 6.3,
            "ted2": 9.8,
            "chime4": 17.8,
            "erj": 38.8,
        },
        {
            "ssl": "WavLM",
            "k": 1024,
            "train": "soft",
            "infer": "soft",
            "ls_clean": 3.9,
            "ls_other": 6.6,
            "ted2": 10.3,
            "chime4": 19.4,
            "erj": 43.4,
        },
    ]


def table2_synthesis() -> list[dict[str, Any]]:
    """Table 2 — LJSpeech resynthesis + TIMIT any-to-one VC (WavLM)."""
    return [
        {
            "k": "cont.",
            "infer": "-",
            "mcd": 4.17,
            "f0_rmse": 0.188,
            "utmos_resynth": 4.15,
            "wer_resynth": 2.68,
            "ppg_dist": 0.721,
            "f0_corr": 0.501,
            "spksim_vc": 0.727,
            "wer_vc": 3.07,
        },
        {
            "k": 4096,
            "infer": "hard",
            "mcd": 5.61,
            "f0_rmse": 0.293,
            "utmos_resynth": 3.86,
            "wer_resynth": 2.99,
            "ppg_dist": 0.857,
            "f0_corr": 0.371,
            "spksim_vc": 0.806,
            "wer_vc": 6.72,
        },
        {
            "k": 4096,
            "infer": "soft",
            "mcd": 5.46,
            "f0_rmse": 0.287,
            "utmos_resynth": 3.97,
            "wer_resynth": 3.00,
            "ppg_dist": 0.811,
            "f0_corr": 0.397,
            "spksim_vc": 0.820,
            "wer_vc": 5.12,
        },
    ]


def table3_phoneme_separability() -> list[dict[str, Any]]:
    """Table 3 — intra/inter phoneme variance ratio (WavLM, selected)."""
    return [
        {"ssl": "WavLM", "k": 128, "task": "ASR", "ratio_hard": 1.39, "ratio_soft": 1.52},
        {"ssl": "WavLM", "k": 1024, "task": "ASR", "ratio_hard": 1.26, "ratio_soft": 1.33},
        {"ssl": "WavLM", "k": 4096, "task": "ASR", "ratio_hard": 1.16, "ratio_soft": 1.22},
        {"ssl": "WavLM", "k": 4096, "task": "Synth.", "ratio_hard": 1.18, "ratio_soft": 1.24},
    ]


def table4_multilayer_asr() -> list[dict[str, Any]]:
    """Table 4 — four-layer WavLM ASR WER on ERJ."""
    return [
        {"layers": 1, "train": "hard", "infer": "soft", "erj": 38.8},
        {"layers": 4, "train": "hard", "infer": "hard", "erj": 44.1},
        {"layers": 4, "train": "hard", "infer": "soft_i", "erj": 41.5},
        {"layers": 4, "train": "hard", "infer": "soft_ii", "erj": 39.1},
    ]


def experimental_protocol(cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    c = cfg or SslSoftInferConfig()
    return {
        "ssl_layer": c.ssl_layer,
        "kmeans_subset_hours": c.kmeans_hours,
        "cluster_sizes": list(c.cluster_sizes),
        "asr_model": "hybrid CTC/attention (ESPnet)",
        "vocoder": "HiFi-GAN",
        "train_infer_recipe": "hard train, soft infer (proposed)",
        "tau_search": "per-dataset without retraining",
        "ood_eval": list(c.ood_datasets),
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_asr_wer": table1_asr_wer(),
        "table2_synthesis": table2_synthesis(),
        "table3_phoneme_separability": table3_phoneme_separability(),
        "table4_multilayer_asr": table4_multilayer_asr(),
    }


def evaluation_demo(seed: int = 0, cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    c = cfg or SslSoftInferConfig()
    post = posterior_demo(seed=seed, cfg=c)
    phon = phoneme_separability_demo(seed=seed, cfg=c)
    wavlm_soft = next(
        r for r in table1_asr_wer() if r["ssl"] == "WavLM" and r["k"] == 4096 and r["infer"] == "soft"
    )
    return {
        "posterior": post,
        "phoneme": phon,
        "tau_erj": tau_for_dataset("ERJ", c),
        "wavlm4096_erj_soft": wavlm_soft["erj"],
        "beats_continuous_on_erj": wavlm_soft["erj"] < c.wavlm_cont_erj,
    }


def pipeline_demo(seed: int = 42, cfg: SslSoftInferConfig | None = None) -> dict[str, Any]:
    return {
        "framework": framework_card(cfg),
        "evaluation": evaluation_demo(seed=seed, cfg=cfg),
        "benchmarks": benchmarks_bundle(),
        "protocol": experimental_protocol(cfg),
    }
