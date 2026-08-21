"""RobustSpeechFlow stub (arXiv:2605.22083)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobustSpeechFlowConfig:
    paper_arxiv: str = "arXiv:2605.22083"
    title: str = "RobustSpeechFlow: Learning Robust Text-to-Speech Trajectories via Augmentation-based Contrastive Flow Matching"

    # Training objective weights (Sec. 4.3)
    lambda_rand: float = 0.2
    lambda_aug: float = 0.2

    # Inference settings (Sec. 4.3)
    nfe_choices: tuple[int, ...] = (12, 24)
    cfg_weight: float = 3.0

    # Seed-TTS-eval (Table 1) — compact SupertonicTTS family numbers
    params_b: float = 0.06
    seed_tts_eval_baseline_wer: float = 1.44
    seed_tts_eval_contrastivefm_wer: float = 1.41
    seed_tts_eval_robustspeechflow_wer: float = 1.38
    seed_tts_eval_sim: float = 0.60

    # ZERO500 (Table 2) at 500k steps (%)
    # English
    zero500_en_baseline_12_cer: float = 0.55
    zero500_en_baseline_12_wer: float = 1.25
    zero500_en_baseline_24_cer: float = 0.48
    zero500_en_baseline_24_wer: float = 1.18
    zero500_en_contrastivefm_12_cer: float = 0.41
    zero500_en_contrastivefm_12_wer: float = 1.10
    zero500_en_contrastivefm_24_cer: float = 0.39
    zero500_en_contrastivefm_24_wer: float = 1.06
    zero500_en_robust_12_cer: float = 0.43
    zero500_en_robust_12_wer: float = 1.14
    zero500_en_robust_24_cer: float = 0.35
    zero500_en_robust_24_wer: float = 1.03

    # Korean
    zero500_ko_baseline_12_cer: float = 0.93
    zero500_ko_baseline_12_wer: float = 8.46
    zero500_ko_baseline_24_cer: float = 0.81
    zero500_ko_baseline_24_wer: float = 8.40
    zero500_ko_contrastivefm_12_cer: float = 0.77
    zero500_ko_contrastivefm_12_wer: float = 7.92
    zero500_ko_contrastivefm_24_cer: float = 0.65
    zero500_ko_contrastivefm_24_wer: float = 7.72
    zero500_ko_robust_12_cer: float = 0.57
    zero500_ko_robust_12_wer: float = 7.59
    zero500_ko_robust_24_cer: float = 0.57
    zero500_ko_robust_24_wer: float = 7.45

