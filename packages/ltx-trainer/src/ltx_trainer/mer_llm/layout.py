"""GOPEX stub limitations."""

LIMITATIONS: tuple[str, ...] = (
    "Survey taxonomy and table excerpts only — no trained MER-with-LLMs checkpoint.",
    "No live API calls to GPT-4o, Gemini, or proprietary MLLM judges.",
    "Autoregressive stub does not run real multimodal encoders or LLM forward passes.",
    "Dataset counts and metrics are paper anchors; re-verify against arXiv:2605.21239 revisions.",
)
