"""Multi-modal feature extraction stubs (§2.1)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.visa.config import VisaConfig


def librosa_acoustic_descriptors(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16_000,
) -> dict[str, float]:
    """Low-level acoustic descriptors (librosa-compatible stub, numpy-only)."""
    x = np.asarray(waveform, dtype=np.float64).ravel()
    if x.size == 0:
        x = np.zeros(1600)
    rms = float(np.sqrt(np.mean(x**2)))
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-8))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    return {
        "rms_energy": rms,
        "spectral_centroid_hz": centroid,
        "zero_crossing_rate": zcr,
        "duration_s": x.size / sample_rate,
        "sample_rate": sample_rate,
    }


def caption_stub(*, duration_s: float) -> str:
    return (
        f"Audio clip ({duration_s:.1f}s): mixed environmental ambience with "
        "speech-like segments and transient events."
    )


def fuzzy_audioset_match(candidates: list[str]) -> list[dict[str, str]]:
    """Map LLM event candidates to AudioSet labels (stub)."""
    out: list[dict[str, str]] = []
    for i, cand in enumerate(candidates):
        label = cand.strip().lower().replace(" ", "_")
        out.append({"candidate": cand, "audioset_label": label, "score": f"{0.92 - i * 0.05:.2f}"})
    return out


def agentic_sed_pipeline(
    *,
    question: str,
    caption: str,
    choices: list[str],
    duration_s: float,
) -> dict[str, Any]:
    """Agentic SED: LLM candidates → FlexSED → VLM verification (stub)."""
    del question, choices
    candidates = ["footsteps", "door_close"] if "second" in caption.lower() else ["speech", "music"]
    events = fuzzy_audioset_match(candidates)
    timestamps = [(0.5, 2.1), (duration_s * 0.4, duration_s * 0.55)]
    return {
        "event_candidates": events,
        "flexsed_timestamps": timestamps,
        "vlm_verified": True,
        "heatmap_confidence": 0.81,
        "event_count": len(timestamps),
    }


def acoustic_visual_views(waveform: np.ndarray, cfg: VisaConfig | None = None) -> list[dict[str, Any]]:
    """Five acoustic visualizations for VLM spectral reasoning (metadata stub)."""
    cfg = cfg or VisaConfig()
    desc = librosa_acoustic_descriptors(waveform)
    views: list[dict[str, Any]] = []
    for view in cfg.acoustic_views:
        views.append(
            {
                "view": view,
                "shape_hint": "(time, freq)" if view != "rms" else "(time, 1)",
                "peak_energy": desc["rms_energy"],
                "ready_for_vlm": True,
            }
        )
    return views


def multimodal_feature_bundle(
    waveform: np.ndarray,
    *,
    question: str = "",
    choices: list[str] | None = None,
    cfg: VisaConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or VisaConfig()
    choices = choices or ["A", "B", "C", "D"]
    desc = librosa_acoustic_descriptors(waveform)
    cap = caption_stub(duration_s=desc["duration_s"])
    sed = agentic_sed_pipeline(
        question=question,
        caption=cap,
        choices=choices,
        duration_s=desc["duration_s"],
    )
    views = acoustic_visual_views(waveform, cfg)
    return {
        "acoustic_descriptors": desc,
        "semantic_caption": cap,
        "agentic_sed": sed,
        "acoustic_visual_views": views,
        "captioner": cfg.captioner,
        "sed_model": cfg.sed_model,
        "vlm_backbone": cfg.vlm_backbone,
    }
