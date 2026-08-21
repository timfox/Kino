"""Known limitations after local baseline + judger wiring."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Upstream instruction editors (Step-Audio-EditX, Audio-Omni, MMEdit, SmartDJ, Ming-UniAudio) "
    "run via GOPEX_MMAE_*_CMD templates; this repo materializes Identity/Noise locally and scores submissions.",
    "Omni judger attaches PCM-trimmed WAV slices; set GOPEX_MMAE_OMNI_BASE_URL to a Qwen3-Omni-compatible vLLM endpoint.",
    "Full 2k omni eval is resumable via vote cache (score-full --cache-dir); mock mode uses Table 2 anchor rates only.",
    "LTX fold sidecar (mmae + mmae_audio) infers taxonomy from captions when GOPEX_MMAE_ENABLE=1; optional "
    "use_mmae_instruction_weights downweights complex edit proxies during AV-fold training.",
)
