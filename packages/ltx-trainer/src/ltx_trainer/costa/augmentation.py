"""CoSTA augmentation + TTA stubs (arXiv:2606.06170)."""

from __future__ import annotations

from typing import Literal

CognitiveState = Literal["AD", "HC"]


def cosyvoice_prompt(cognitive_state: CognitiveState) -> str:
    """Natural-language instruction I(c) for CS-Cond CosyVoice2 (Eq. 1)."""
    if cognitive_state == "AD":
        return (
            "Speak as an older patient with Alzheimer's disease: unnatural pauses, "
            "imprecise articulation, reduced fluency."
        )
    return (
        "Speak as a healthy adult: natural spoken style, clear stable voice, "
        "normal fluent pace."
    )


def build_text_prompt(
    cognitive_state: CognitiveState,
    manual_transcript: str,
    *,
    separator: str = "<|endofprompt|>",
) -> str:
    """Eq. (1): y(c) = [I(c), <|endofprompt|>, yMT]."""
    return f"{cosyvoice_prompt(cognitive_state)}{separator}{manual_transcript}"


def f5_cognition_label(cognitive_state: CognitiveState) -> str:
    return "Alzheimer" if cognitive_state == "AD" else "Health"


def synthesize_stub(
    target_transcript: str,
    cognitive_state: CognitiveState,
    *,
    backend: str = "CosyVoice2",
) -> dict[str, str | float]:
    """Eq. (4) placeholder — no waveform generation."""
    return {
        "backend": backend,
        "cognitive_state": cognitive_state,
        "target_transcript_len": float(len(target_transcript.split())),
        "conditioning": build_text_prompt(cognitive_state, target_transcript)
        if backend == "CosyVoice2"
        else f5_cognition_label(cognitive_state),
    }


def tta_probability_average(
    p_original: tuple[float, float],
    p_synthetic: tuple[float, float],
) -> tuple[float, float]:
    """§2.3.2: P_final = (P_ori + P_syn) / 2 for (P_HC, P_AD)."""
    return (
        (p_original[0] + p_synthetic[0]) / 2.0,
        (p_original[1] + p_synthetic[1]) / 2.0,
    )


def augmentation_factor_curve() -> dict[str, list[float]]:
    """Fig. 3 inverted-U anchor (three ASR sources + average)."""
    factors = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    w2v960 = [81.0, 83.5, 84.17, 84.0, 83.0, 82.5, 82.0]
    w2v960_lv = [82.0, 84.5, 85.00, 84.5, 83.5, 83.0, 82.5]
    whisper_v3 = [81.5, 84.0, 84.58, 84.2, 83.2, 82.8, 82.3]
    avg = [(a + b + c) / 3.0 for a, b, c in zip(w2v960, w2v960_lv, whisper_v3, strict=True)]
    return {
        "factors": factors,
        "fine_tuned_w2v960": w2v960,
        "fine_tuned_w2v960_large_lv": w2v960_lv,
        "fine_tuned_whisper_large_v3": whisper_v3,
        "average": avg,
    }


def optimal_augmentation_factor() -> float:
    curve = augmentation_factor_curve()
    idx = max(range(len(curve["average"])), key=lambda i: curve["average"][i])
    return curve["factors"][idx]
