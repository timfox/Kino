"""MUSTBENCH stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No MTG-Jamendo / Slakh2100 download or MUST checkpoint weights in-tree.",
    "MERT transition encoder and GRPO are numpy proxies, not full Qwen2.5-Omni training.",
    "Benchmark numbers are paper table anchors (Tables 3–4).",
    "English read speech from expert-validated pipeline only; multilingual/noisy music not covered.",
)
