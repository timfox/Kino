"""HoliTok stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No trained HoliTok weights or BigVGAN decoder in GOPEX.",
    "No 48 kHz waveform I/O or LibriSpeech reconstruction eval.",
    "AR+DiT downstream is table anchors + toy flow-matching / CE proxies only.",
    "Speech-only scope; environmental sound and music generalization not evaluated here.",
)
