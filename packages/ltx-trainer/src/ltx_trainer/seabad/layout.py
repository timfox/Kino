"""SEABAD stub limitations."""

from __future__ import annotations

LIMITATIONS: tuple[str, ...] = (
    "Reference stub only — no Zenodo download, Xeno-Canto API, or FAISS index in-repo.",
    "Positive clips from community Xeno-Canto metadata; geographic and temporal bias remain (§5.6).",
    "Negative corpus draws from temperate/global sources; tropical rain, cicada, primate gaps.",
    "Binary presence–absence only — not species classification.",
    "Table 5 CNN baselines use 224×224 mel for validation; edge deployment needs separate quantization (companion SEABADNet).",
)
