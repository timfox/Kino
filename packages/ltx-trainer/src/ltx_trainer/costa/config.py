"""CoSTA — cognitive-state-conditioned TTS AD augmentation (Liu et al., arXiv:2606.06170)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostaConfig:
    paper_arxiv: str = "arXiv:2606.06170"
    title: str = (
        "CoSTA: Cognitive-State-Conditioned TTS Data Augmentation Using ASR "
        "Transcripts for Alzheimer's Disease Detection"
    )
    framework: str = "CoSTA"

    dataset: str = "ADReSS"
    train_subjects: int = 108
    test_subjects: int = 48
    tts_train_split: str = "45 AD + 45 HC (5:1 holdout for TTS eval)"
    asr_finetune_corpora: tuple[str, ...] = ("WLS", "Lu", "Kempler")

    # Transcript pool (§2.2)
    asr_model_count: int = 18
    asr_transcripts_per_sample: int = 36
    transcript_pool_size: int = 37  # MT + 36 ASR

    tts_backends: tuple[str, ...] = ("CosyVoice2", "F5-TTS")
    cs_cond_variants: tuple[str, ...] = ("CosyVoice2-AD", "CosyVoice2-HC", "CS-Cond F5-TTS")

    # AD detection backbone (§2.4)
    detector: str = "WavLM + 24-layer Transformer + weighted fusion + attentive pooling"

    # Table 2 baseline + headline (2× aug, audio-only ADReSS test)
    baseline_accuracy_pct: float = 81.67
    best_accuracy_pct: float = 85.83
    gain_over_baseline_pct: float = 4.16  # paper Table 4 (+4.16%); float delta rounds to 4.16

    # Table 2 — CS-Cond CosyVoice2 blue-bold configs (>84%)
    cs_cosy_w2v960_ft_pct: float = 84.17
    cs_cosy_w2v960_large_lv_ft_pct: float = 85.00
    cs_cosy_whisper_large_v3_ft_pct: float = 84.58

    # Table 2 — beat-baseline ratios (2× aug, 37 text sources each)
    cs_cosy_beat_baseline_ratio: str = "28/37"
    pretrained_cosy_beat_baseline_ratio: str = "7/37"
    cs_f5_beat_baseline_ratio: str = "24/37"
    pretrained_f5_beat_baseline_ratio: str = "16/37"

    # ASR-driven > MT-driven (Table 2 footer)
    cs_cosy_asr_beats_mt_ratio: str = "20/36"
    cs_f5_asr_beats_mt_ratio: str = "22/36"

    # Table 3 — TTA @ 2× CS-Cond CosyVoice2
    tta_w2v960_pct: float = 85.42
    tta_w2v960_large_lv_pct: float = 85.83
    tta_whisper_large_v3_pct: float = 85.42
    tta_average_gain_pct: float = 0.98

    # Fig. 3 optimal augmentation factor
    optimal_aug_factor: float = 2.0
    aug_factor_sweep: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

    # Table 1 — CS-Cond CosyVoice2-AD objective metrics
    cs_cosy_ad_mcd: float = 5.436
    cs_cosy_ad_log_f0_rmse: float = 0.305
    cs_cosy_ad_fad: float = 2.192

    # Table 4 comparisons
    whisper_mlp_prior_pct: float = 79.17
    wav2vec2_linear_prior_pct: float = 80.83
    aw_hubert_prior_pct: float = 81.67
    pitch_shift_da_pct: float = 79.17

    # Fig. 2 ASR WER span on ADReSS
    asr_wer_min_pct: float = 26.36
    asr_wer_max_pct: float = 68.55
