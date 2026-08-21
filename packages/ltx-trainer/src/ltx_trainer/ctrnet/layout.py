"""CTRnet/PuLSS stub limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — no TF-GridNet training, CHiME-6 audio, or Parakeet fine-tuning in-repo.",
    "FCP and pseudo-label routines are single-frequency toys, not full multi-mic STFT pipelines.",
    "Assumes fixed C=4 speakers per 12 s block; variable speaker count handled only via frame muting concept.",
    "Distributed 24-mic PuLSS uses per-array DNN plus SNR selection — not reimplemented here.",
)
