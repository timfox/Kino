"""SVHighlights stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No YouTube download or redistribution of SVHighlights video files.",
    "PSNR alignment runs on synthetic/downsampled frames only in this stub.",
    "TransNet V2, WhisperX, InternVL, and Llama are not invoked — heuristic proxies only.",
    "VTG-tuned baselines (Moment-DETR, TRACE, etc.) are table anchors, not runnable checkpoints.",
    "Evaluation metrics are reference implementations on toy timelines, not full 640-hour eval.",
)
