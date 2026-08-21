"""COMET stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No HTSAT-BERT-ZS / CLAP weights or Clotho/AudioCaps I/O in GOPEX.",
    "PLS-SVD runs on toy random matrices, not full CLAP embedding banks.",
    "PLSHead retrieval and captioning numbers are table anchors only.",
    "Projection decoding memory bank not materialized (linear PD proxy only).",
)
