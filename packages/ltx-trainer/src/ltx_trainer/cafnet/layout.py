"""CAFNet stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No MLADDC dataset I/O or trained CAFNet checkpoints in GOPEX.",
    "Feature extraction uses numpy STFT proxies, not librosa MFCC/LFCC/Chroma.",
    "Model forward pass is a shape/loss smoke stub, not full PyTorch CAFNet.",
    "Reported metrics are paper table anchors (Tables 3–11).",
)
