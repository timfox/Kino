"""BTS-CAFE stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No real ICBHI/SPRSound download or multi-client FL training loop.",
    "GIN uses numpy random group-conv proxy, not full spectrogram BTS fine-tuning.",
    "Private hospital client partitions and gradient communication are not simulated.",
    "Benchmark numbers are paper table anchors (Tables 2–4, Fig. 2).",
)
