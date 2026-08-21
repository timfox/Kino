"""Framework card and benchmark excerpts (arXiv:2605.22083)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.robustspeechflow.config import RobustSpeechFlowConfig
from ltx_trainer.robustspeechflow.layout import LIMITATIONS
from ltx_trainer.robustspeechflow.mock import evaluation_smoke


def framework_card(cfg: RobustSpeechFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RobustSpeechFlowConfig()
    return {
        "name": "RobustSpeechFlow",
        "paper": cfg.paper_arxiv,
        "authors": "Jinhyeok Yang, Hyeongju Kim, Yechan Yu, Joon Byun, Frederik Bous, Juheon Lee",
        "goal": "Improve alignment robustness in flow-matching TTS by penalizing skip/repeat failure modes.",
        "core_idea": {
            "positive": "Flow matching on linear path x_t=(1-t)ε+tx with v=x-ε.",
            "random_negative": "Contrastive FM using in-batch random negative velocities (Eq. 3).",
            "hard_negative": "Same-utterance length-preserving latent corruptions: repeat overwrite (Eq. 4) and skip shift+silence (Eq. 5).",
        },
        "objective": {
            "Lpos": "E ||uθ(x_t,t,c) - (x-ε)||^2",
            "Lrand": "E ||uθ - (x_rand-ε)||^2 (subtracted)",
            "Laug": "E ||uθ - (x_aug-ε)||^2 (subtracted)",
            "Ltotal": "Lpos - λrand·Lrand - λaug·Laug (Eq. 7)",
        },
        "settings": {
            "lambda_rand": cfg.lambda_rand,
            "lambda_aug": cfg.lambda_aug,
            "nfe": list(cfg.nfe_choices),
            "cfg_weight": cfg.cfg_weight,
        },
        "benchmarks": {
            "seed_tts_eval": "public zero-shot benchmark, report WER and SIM (Table 1)",
            "zero500": "internal multilingual benchmark (English+Korean) with diverse speakers/prosody, report CER/WER via Whisper large-v3 (Table 2)",
        },
        "headlines": headline_results(cfg),
        "limitations": LIMITATIONS,
    }


def table1_seed_tts_eval() -> list[dict[str, Any]]:
    """Table 1 excerpt — Seed-TTS-eval (WER↓, SIM↑)."""
    cfg = RobustSpeechFlowConfig()
    return [
        {"model": "Baseline (SupertonicTTS)", "params_b": cfg.params_b, "wer": cfg.seed_tts_eval_baseline_wer, "sim": cfg.seed_tts_eval_sim},
        {"model": "ContrastiveFM", "params_b": cfg.params_b, "wer": cfg.seed_tts_eval_contrastivefm_wer, "sim": cfg.seed_tts_eval_sim},
        {"model": "RobustSpeechFlow", "params_b": cfg.params_b, "wer": cfg.seed_tts_eval_robustspeechflow_wer, "sim": cfg.seed_tts_eval_sim},
    ]


def table2_zero500() -> list[dict[str, Any]]:
    """Table 2 excerpt — ZERO500 at 500k steps (%)."""
    cfg = RobustSpeechFlowConfig()
    rows: list[dict[str, Any]] = []

    def add(model: str, nfe: int, en_cer: float, en_wer: float, ko_cer: float, ko_wer: float) -> None:
        rows.append(
            {
                "model": model,
                "nfe": nfe,
                "en_cer": en_cer,
                "en_wer": en_wer,
                "ko_cer": ko_cer,
                "ko_wer": ko_wer,
            }
        )

    add("Baseline", 12, cfg.zero500_en_baseline_12_cer, cfg.zero500_en_baseline_12_wer, cfg.zero500_ko_baseline_12_cer, cfg.zero500_ko_baseline_12_wer)
    add("Baseline", 24, cfg.zero500_en_baseline_24_cer, cfg.zero500_en_baseline_24_wer, cfg.zero500_ko_baseline_24_cer, cfg.zero500_ko_baseline_24_wer)
    add(
        "ContrastiveFM",
        12,
        cfg.zero500_en_contrastivefm_12_cer,
        cfg.zero500_en_contrastivefm_12_wer,
        cfg.zero500_ko_contrastivefm_12_cer,
        cfg.zero500_ko_contrastivefm_12_wer,
    )
    add(
        "ContrastiveFM",
        24,
        cfg.zero500_en_contrastivefm_24_cer,
        cfg.zero500_en_contrastivefm_24_wer,
        cfg.zero500_ko_contrastivefm_24_cer,
        cfg.zero500_ko_contrastivefm_24_wer,
    )
    add("RobustSpeechFlow", 12, cfg.zero500_en_robust_12_cer, cfg.zero500_en_robust_12_wer, cfg.zero500_ko_robust_12_cer, cfg.zero500_ko_robust_12_wer)
    add("RobustSpeechFlow", 24, cfg.zero500_en_robust_24_cer, cfg.zero500_en_robust_24_wer, cfg.zero500_ko_robust_24_cer, cfg.zero500_ko_robust_24_wer)
    return rows


def headline_results(cfg: RobustSpeechFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RobustSpeechFlowConfig()
    return {
        "seed_tts_eval_wer_baseline_to_robust": (cfg.seed_tts_eval_baseline_wer, cfg.seed_tts_eval_robustspeechflow_wer),
        "seed_tts_eval_wer_delta": round(cfg.seed_tts_eval_robustspeechflow_wer - cfg.seed_tts_eval_baseline_wer, 3),
        "zero500_en_cer_24_baseline_to_robust": (cfg.zero500_en_baseline_24_cer, cfg.zero500_en_robust_24_cer),
        "zero500_ko_cer_24_baseline_to_robust": (cfg.zero500_ko_baseline_24_cer, cfg.zero500_ko_robust_24_cer),
        "lambda_rand": cfg.lambda_rand,
        "lambda_aug": cfg.lambda_aug,
        "params_b": cfg.params_b,
    }


def evaluation_demo(cfg: RobustSpeechFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RobustSpeechFlowConfig()
    return {
        "config": cfg.__dict__,
        "smoke": evaluation_smoke(cfg),
    }


def benchmarks_bundle(cfg: RobustSpeechFlowConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RobustSpeechFlowConfig()
    return {
        "headlines": headline_results(cfg),
        "table1_seed_tts_eval": table1_seed_tts_eval(),
        "table2_zero500": table2_zero500(),
    }

