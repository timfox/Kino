"""ChildVox stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No MLADDC-style download of 17 ChildVox corpora or private NLS/ADOS data.",
    "Encoder/LALM fine-tuning is a LoRA smoke proxy, not full SSAST/Whisper training.",
    "Benchmark numbers are paper table anchors (Tables 3–5, Figure 4).",
    "English-centric; multilingual TinyVox phoneme eval not wired to real IPA targets.",
)
