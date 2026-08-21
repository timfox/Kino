"""Stub metrics for identity, video quality, timbre, lip sync."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricBundle:
    facesim_arc: float
    facesim_cur: float
    fid: float
    fvd: float
    clip_text: float
    speaker_sim: float
    fad: float
    wer_pct: float
    lse_c: float | None = None
    lse_d: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "facesim_arc": round(self.facesim_arc, 4),
            "facesim_cur": round(self.facesim_cur, 4),
            "fid": round(self.fid, 2),
            "fvd": round(self.fvd, 2),
            "clip_text": round(self.clip_text, 2),
            "speaker_sim": round(self.speaker_sim, 4),
            "fad": round(self.fad, 2),
            "wer_pct": round(self.wer_pct, 2),
            "lse_c": None if self.lse_c is None else round(self.lse_c, 2),
            "lse_d": None if self.lse_d is None else round(self.lse_d, 2),
        }


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def stub_identity_metrics(ref_face: np.ndarray, frames: list[np.ndarray]) -> tuple[float, float]:
    """Proxy FaceSim using cosine similarity between ref and frame embeddings."""
    if not frames:
        return 0.0, 0.0
    sims = [cosine_sim(ref_face, f) for f in frames]
    arc = float(np.mean(sims))
    cur = float(np.mean([s * 0.98 + 0.02 for s in sims]))
    return arc, cur


def stub_speaker_sim(ref_audio_emb: np.ndarray, gen_audio_emb: np.ndarray) -> float:
    return cosine_sim(ref_audio_emb, gen_audio_emb)


def stub_clip_text(prompt_emb: np.ndarray, frame_emb: np.ndarray) -> float:
    # CLIP cosine scaled to paper-ish range (~25–28)
    return 25.0 + 3.0 * max(0.0, cosine_sim(prompt_emb, frame_emb))


def stub_fid_fvd(ref_face: np.ndarray, frames: list[np.ndarray]) -> tuple[float, float]:
    if not frames:
        return 999.0, 999.0
    sim = np.mean([cosine_sim(ref_face, f) for f in frames])
    fid = max(50.0, 200.0 * (1.0 - sim))
    fvd = max(100.0, 600.0 * (1.0 - sim))
    return float(fid), float(fvd)
