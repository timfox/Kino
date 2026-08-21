"""Framework card, paper tables, and evaluation demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.iraf.config import IrafConfig
from ltx_trainer.iraf.gate import gate_demo, duplex_loss_stub


def framework_card(cfg: IrafConfig | None = None) -> dict[str, Any]:
    c = cfg or IrafConfig()
    return {
        "paper": c.paper_arxiv,
        "title": c.title,
        "framework": c.framework,
        "task": "noise_robust_full_duplex_spoken_dialogue",
        "method": "frame_level_reliability_gate_before_llm_fusion",
        "backbone": {
            "speech_encoder": c.speech_encoder,
            "llm": c.llm_backbone,
            "codec": c.speech_codec,
            "speaker_encoder": c.speaker_encoder,
            "iraf_layers": c.iraf_transformer_layers,
        },
        "datasets": list(c.datasets),
        "noise_augmentation": c.noise_corpus,
        "loss_weights": {
            "lambda_text": c.lambda_text,
            "lambda_audio": c.lambda_audio,
            "lambda_gate": c.lambda_gate,
        },
        "headline": headline_results(c),
    }


def headline_results(cfg: IrafConfig | None = None) -> dict[str, Any]:
    c = cfg or IrafConfig()
    return {
        "msmarco_iraf_bleu": c.msmarco_iraf_bleu,
        "msmarco_iraf_rsr_pct": c.msmarco_iraf_rsr_pct,
        "instruct_iraf_bleu": c.instruct_iraf_bleu,
        "instruct_iraf_rsr_pct": c.instruct_iraf_rsr_pct,
        "instruct_iraf_ssr_pct": c.instruct_iraf_ssr_pct,
        "msmarco_bleu_rel_improvement_pct": c.msmarco_iraf_bleu_rel_pct,
        "instruct_bleu_rel_improvement_pct": c.instruct_iraf_bleu_rel_pct,
    }


def table1_ms_marco_musan() -> list[dict[str, Any]]:
    """Table 1 — MS MARCO under MUSAN speech interference (ALL column)."""
    return [
        {
            "method": "CleanBase",
            "condition": "interfering_speakers_only",
            "bleu": 0.66,
            "sbert": 0.11,
            "rl_s": 1.46,
            "rsr_pct": 6.2,
        },
        {
            "method": "NoisyAug",
            "condition": "interfering_speakers_only",
            "bleu": 12.74,
            "sbert": 0.506,
            "rl_s": 0.97,
            "rsr_pct": 93.1,
        },
        {
            "method": "IRAF",
            "condition": "interfering_speakers_only",
            "bleu": 14.20,
            "sbert": 0.523,
            "rl_s": 0.96,
            "rsr_pct": 95.7,
        },
        {
            "method": "CleanBase",
            "condition": "interference_and_noise",
            "bleu": 0.00,
            "sbert": 0.03,
            "rl_s": 1.49,
            "rsr_pct": 2.8,
        },
        {
            "method": "NoisyAug",
            "condition": "interference_and_noise",
            "bleu": 11.33,
            "sbert": 0.454,
            "rl_s": 0.98,
            "rsr_pct": 88.2,
        },
        {
            "method": "IRAF",
            "condition": "interference_and_noise",
            "bleu": 12.01,
            "sbert": 0.476,
            "rl_s": 0.94,
            "rsr_pct": 92.5,
        },
    ]


def table2_instructs2s() -> list[dict[str, Any]]:
    """Table 2 — InstructS2S-200K under MUSAN speech interference."""
    return [
        {
            "method": "CleanBase",
            "condition": "interfering_speakers_only",
            "bleu": 1.13,
            "sbert": 0.22,
            "rl_s": 1.39,
            "rsr_pct": 13.9,
            "sl_s": 1.29,
            "ssr_pct": 42.7,
        },
        {
            "method": "NoisyAug",
            "condition": "interfering_speakers_only",
            "bleu": 9.64,
            "sbert": 0.47,
            "rl_s": 0.97,
            "rsr_pct": 69.2,
            "sl_s": 0.74,
            "ssr_pct": 99.0,
        },
        {
            "method": "IRAF",
            "condition": "interfering_speakers_only",
            "bleu": 13.76,
            "sbert": 0.58,
            "rl_s": 0.82,
            "rsr_pct": 91.0,
            "sl_s": 0.73,
            "ssr_pct": 99.8,
        },
        {
            "method": "CleanBase",
            "condition": "interference_and_noise",
            "bleu": 0.91,
            "sbert": 0.21,
            "rl_s": 1.41,
            "rsr_pct": 9.8,
            "sl_s": 1.34,
            "ssr_pct": 40.2,
        },
        {
            "method": "NoisyAug",
            "condition": "interference_and_noise",
            "bleu": 8.32,
            "sbert": 0.44,
            "rl_s": 1.05,
            "rsr_pct": 56.0,
            "sl_s": 0.74,
            "ssr_pct": 99.6,
        },
        {
            "method": "IRAF",
            "condition": "interference_and_noise",
            "bleu": 9.83,
            "sbert": 0.47,
            "rl_s": 0.98,
            "rsr_pct": 69.2,
            "sl_s": 0.73,
            "ssr_pct": 100.0,
        },
    ]


def experimental_protocol(cfg: IrafConfig | None = None) -> dict[str, Any]:
    c = cfg or IrafConfig()
    return {
        "eval_metrics": {
            "response_quality": ["BLEU", "sBERT"],
            "turn_taking": ["response_latency", "response_success_rate"],
            "barge_in": ["stop_latency", "stop_success_rate"],
        },
        "baselines": ["CleanBase", "NoisyAug", "IRAF"],
        "noise": c.noise_corpus,
        "snr_db_ms_marco": list(c.snr_db_ms_marco),
        "snr_db_instructs2s": list(c.snr_db_instructs2s),
        "barge_in_training_prob": c.barge_in_prob,
        "inter_turn_pause_s": c.inter_turn_pause_s,
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_ms_marco": table1_ms_marco_musan(),
        "table2_instructs2s": table2_instructs2s(),
    }


def evaluation_demo(seed: int = 42, cfg: IrafConfig | None = None) -> dict[str, Any]:
    c = cfg or IrafConfig()
    gate = gate_demo(seed=seed, cfg=c)
    iraf_row = next(
        r
        for r in table1_ms_marco_musan()
        if r["method"] == "IRAF" and r["condition"] == "interfering_speakers_only"
    )
    noisy = next(
        r
        for r in table1_ms_marco_musan()
        if r["method"] == "NoisyAug" and r["condition"] == "interfering_speakers_only"
    )
    return {
        "gate": gate,
        "loss_stub": duplex_loss_stub(cfg=c),
        "msmarco_bleu_gain": iraf_row["bleu"] - noisy["bleu"],
        "beats_noisyaug_on_msmarco": iraf_row["bleu"] > noisy["bleu"],
        "beats_noisyaug_on_instruct_rsr": c.instruct_iraf_rsr_pct > c.instruct_noisyaug_rsr_pct,
    }


def pipeline_demo(seed: int = 42, cfg: IrafConfig | None = None) -> dict[str, Any]:
    c = cfg or IrafConfig()
    return {
        "framework": framework_card(c),
        "protocol": experimental_protocol(c),
        "evaluation": evaluation_demo(seed=seed, cfg=c),
        "benchmarks": benchmarks_bundle(),
    }
