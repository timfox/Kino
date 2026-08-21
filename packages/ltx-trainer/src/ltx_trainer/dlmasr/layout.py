"""DLM-ASR decoding stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No Whisper-LLaDA / LLaDA-8B weights or LibriSpeech I/O in-tree.",
    "Decoding strategies are numpy confidence proxies, not full block diffusion.",
    "WER/RTF numbers are paper figure anchors (Figs 1–4).",
    "English read speech only; noisy/multilingual generalization not evaluated.",
)
