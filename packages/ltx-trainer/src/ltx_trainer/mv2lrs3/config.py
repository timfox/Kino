"""MV2LRS3 — matched AVSR generalisability benchmark (arXiv:2606.07259)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Mv2Lrs3Config:
    paper_arxiv: str = "arXiv:2606.07259"
    title: str = "Assessing True Generalisability of Audio-Visual Speech Recognisers"
    benchmark: str = "MultiVSR2LRS3 (MV2LRS3)"
    reference_set: str = "LRS3 Test"
    candidate_pool: str = "MultiVSR English validation"
    release_url: str = "https://github.com/chaufanglin/mv2lrs3"

    lrs3_test_hours: float = 0.9
    lrs3_test_utterances: int = 1321
    mv2lrs3_runs: int = 5
    knn_candidates: int = 5
    feature_dims: int = 8

    # Linear fit (Eq. 1): WER_MV2LRS3 = slope × WER_LRS3 + intercept
    linear_fit_slope: float = 10.4
    linear_fit_intercept: float = 8.1

    matching_factors: tuple[str, ...] = (
        "duration",
        "age",
        "gender",
        "skin_tone",
        "yaw_mean",
        "yaw_std",
        "snr",
        "speech_rate",
    )

    factor_weights: dict[str, float] = field(
        default_factory=lambda: {
            "duration": 100.0,
            "age": 100.0,
            "gender": 50.0,
            "skin_tone": 40.0,
            "yaw_mean": 100.0,
            "yaw_std": 50.0,
            "snr": 70.0,
            "speech_rate": 40.0,
        }
    )

    evaluated_models: tuple[str, ...] = (
        "AV-HuBERT",
        "Auto-AVSR",
        "USR",
        "Whisper-Flamingo",
        "Llama-AVSR",
    )
