"""VOICEGIRAFFE stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "No hour-scale audio corpus or Hub dataset download in-tree.",
    "MC accuracy and table anchors are paper references, not live LALM inference.",
    "E2E / cascade / LRM modes are evaluation-protocol stubs only.",
    "Caption pipeline (Qwen3-Omni, Gemini QA gen) is documented but not executed.",
)
