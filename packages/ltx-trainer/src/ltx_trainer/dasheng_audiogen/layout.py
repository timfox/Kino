"""Dasheng AudioGen stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No DashengTokenizer weights, DiT checkpoint, or 77k-hour ACAVCaps superset in-tree.",
    "Flow-matching generation and T5 conditioning are algorithmic stubs only.",
    "MECAT / AudioCaps metrics are paper table anchors, not live generation runs.",
    "Fixed 10-second clip duration per paper; no variable-length generation.",
)
