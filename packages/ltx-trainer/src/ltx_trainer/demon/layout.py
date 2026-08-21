"""DEMON stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No ACE-Step 1.5, StreamDiffusion, or TensorRT engine binaries in-tree.",
    "Ring-buffer tick and SDE blending are algorithmic proxies, not GPU TRT inference.",
    "Throughput and propagation numbers are paper table anchors (Tables 3, 12–15).",
    "No live MIDI/DAW integration; composability workflows are documented only.",
)
