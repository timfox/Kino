"""CryAcc stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No Boston Children's vaccination cry corpus or Knowles BU-27135 recordings in-tree.",
    "Feature extraction is numpy/Praat-style proxy math, not Parselmouth on real cry segments.",
    "ICC and bias values are paper Table I–II anchors, not recomputed from local data.",
    "Vocal-tract formant content not captured by chest-surface ACC (paper §IV limitation).",
)
