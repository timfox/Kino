"""LiveK12Bench layout constants."""

from __future__ import annotations

DISCIPLINES: tuple[str, ...] = ("mathematics", "physics", "chemistry", "biology")

MODALITIES: tuple[tuple[str, str], ...] = (
    ("text_only", "TO — parsed textual stem/options"),
    ("text_image", "TI — interleaved text + cropped figures"),
    ("image_only_exam", "IO — full exam page snapshot + question index (end-to-end)"),
)

QUESTION_TYPES: tuple[str, ...] = ("mcq", "fib", "qa")

CHALLENGING_SUBSETS: tuple[tuple[str, str], ...] = (
    ("complex_layout", "Multi-page / detached figures / noisy exam layouts (50/subject)"),
    ("rigorous_process", "Multi-KP items prone to lucky-guess without sound reasoning"),
    ("long_horizon", "High-point problems trapping models in verbose chains"),
)

PROCESS_ERROR_TYPES: tuple[str, ...] = (
    "CIE",  # condition interpretation error
    "LAE",  # logical assumption error
    "DRE",  # deductive reasoning error
)

PIPELINE_STAGES: tuple[str, ...] = (
    "collect_fresh_exam_pdfs",
    "structural_ocr_mineru",
    "llm_variable_template_parse",
    "human_verify_and_dedupe",
    "knowledge_point_tagging",
    "mock_exam_inference",
    "multi_llm_judge_arbitration",
)

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — not running MinerU, vLLM judges, or full 2,114-question eval.",
    "Table metrics are paper-reported constants.",
    "Toy scoring smoke uses synthetic items, not MHS/LiveK12 JSON.",
)
