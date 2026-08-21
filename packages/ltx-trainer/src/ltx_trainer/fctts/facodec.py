"""FACodec-style factorization — §3.1 (toy slots)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FacodecFactors:
    """Disentangled streams used by FC-TTS (cp + zspk only at inference)."""

    z_spk: np.ndarray
    prosody_tokens: np.ndarray
    content_tokens: np.ndarray | None = None
    detail_tokens: np.ndarray | None = None


def split_facodec_streams(
    latent: np.ndarray,
    prosody_dim: int = 8,
    content_dim: int = 16,
    detail_dim: int = 24,
) -> FacodecFactors:
    """Toy split of a flat codec latent into factorized streams."""
    latent = np.asarray(latent, dtype=np.float64).ravel()
    off = 0
    cp = latent[off : off + prosody_dim]
    off += prosody_dim
    cc = latent[off : off + content_dim]
    off += content_dim
    cd = latent[off : off + detail_dim]
    off += detail_dim
    z_spk = latent[off : off + 32] if off + 32 <= latent.size else latent[:32]
    return FacodecFactors(z_spk=z_spk, prosody_tokens=cp, content_tokens=cc, detail_tokens=cd)


def fctts_conditioning(factors: FacodecFactors) -> dict[str, Any]:
    """FC-TTS uses only timbre embedding and prosody tokens — §3.1."""
    return {
        "z_spk": factors.z_spk,
        "cp": factors.prosody_tokens,
        "uses_content": False,
        "uses_detail": False,
        "rationale": "Exclude cc/cd to prevent leakage across timbre/style pathways",
    }
