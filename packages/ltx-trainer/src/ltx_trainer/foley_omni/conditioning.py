"""Structured multimodal conditioning and hybrid injection (Sec. 4.2–4.3)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

WORDS_OPEN = "[WORDS]"
WORDS_CLOSE = "[END_WORDS]"
AUDIO_OPEN = "[AUDIO]"
AUDIO_CLOSE = "[END_AUDIO]"


def format_structured_prompt(
    *,
    words: str = "",
    audio: str = "",
    music: str = "",
) -> str:
    """Build structured caption with explicit field tags (empty fields omitted)."""
    parts: list[str] = []
    if words.strip():
        parts.append(f"{WORDS_OPEN}{words.strip()}{WORDS_CLOSE}")
    if audio.strip():
        parts.append(f"{AUDIO_OPEN}{audio.strip()}{AUDIO_CLOSE}")
    if music.strip():
        parts.append(f"This song features {music.strip()}")
    return " ".join(parts)


def parse_structured_fields(text: str) -> dict[str, str]:
    """Extract [WORDS], [AUDIO], and music caption fragments from structured text."""
    out: dict[str, str] = {"words": "", "audio": "", "music": ""}
    m = re.search(
        rf"{re.escape(WORDS_OPEN)}(.*?){re.escape(WORDS_CLOSE)}",
        text,
        flags=re.DOTALL,
    )
    if m:
        out["words"] = m.group(1).strip()
    m = re.search(
        rf"{re.escape(AUDIO_OPEN)}(.*?){re.escape(AUDIO_CLOSE)}",
        text,
        flags=re.DOTALL,
    )
    if m:
        out["audio"] = m.group(1).strip()
    if "This song features" in text:
        out["music"] = text.split("This song features", 1)[-1].strip()
    return out


def concat_unified_context(
    c_text: np.ndarray,
    c_clip: np.ndarray,
    c_sync: np.ndarray,
) -> np.ndarray:
    """C_uni = [C_text; C_clip; C_sync] along feature axis (Eq. 1)."""
    return np.concatenate([c_text, c_clip, c_sync], axis=-1)


def sync_adapter(c_sync: np.ndarray, target_len: int) -> np.ndarray:
    """Interpolate Synchformer features to audio latent length L (Eq. 2 stub)."""
    if c_sync.ndim == 1:
        c_sync = c_sync.reshape(1, -1)
    src_len = c_sync.shape[0]
    if src_len == target_len:
        return c_sync.astype(np.float64)
    x_src = np.linspace(0.0, 1.0, src_len)
    x_dst = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, c_sync.shape[1]), dtype=np.float64)
    for j in range(c_sync.shape[1]):
        out[:, j] = np.interp(x_dst, x_src, c_sync[:, j])
    return out


def apply_sync_injection(x_t: np.ndarray, z_sync: np.ndarray) -> np.ndarray:
    """x̃_t = x_t + Z_sync (Eq. 3)."""
    if z_sync.shape != x_t.shape:
        if z_sync.ndim == 2 and x_t.ndim == 1:
            z_sync = z_sync.mean(axis=0)
        if z_sync.shape != x_t.shape:
            raise ValueError(f"sync shape {z_sync.shape} != latent {x_t.shape}")
    return x_t + z_sync


def toy_project_conditions(
    *,
    text_dim: int = 8,
    clip_dim: int = 6,
    sync_dim: int = 4,
    latent_len: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministic fake embeddings for smoke tests."""
    rng = np.random.default_rng(seed)
    c_text = rng.standard_normal(text_dim)
    c_clip = rng.standard_normal((latent_len, clip_dim))
    c_sync = rng.standard_normal((max(4, latent_len // 4), sync_dim))
    z_aligned = sync_adapter(c_sync, latent_len)
    z_sync = z_aligned.mean(axis=1) if z_aligned.ndim == 2 else z_aligned
    c_uni = concat_unified_context(c_text, c_clip.mean(axis=0), z_sync)
    x_t = rng.standard_normal(latent_len)
    x_tilde = apply_sync_injection(x_t, z_sync)
    return {
        "c_text": c_text,
        "c_clip": c_clip,
        "c_sync": c_sync,
        "z_sync": z_sync,
        "c_uni": c_uni,
        "x_tilde": x_tilde,
    }
