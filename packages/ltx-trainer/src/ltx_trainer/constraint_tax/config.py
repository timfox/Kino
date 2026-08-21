"""Configuration for Constraint Tax SLM structured-output benchmark (arXiv:2605.26128)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConstraintTaxConfig:
    paper_arxiv: str = "2605.26128"
    paper_title: str = (
        "The Constraint Tax: Measuring Validity-Correctness Tradeoffs "
        "in Structured Outputs for Small Language Models"
    )
    author: str = "Jaideep Ray (jaray@acm.org)"

    main_generations: int = 15_000
    main_instances_per_checkpoint: int = 1_000
    main_task_families: int = 5
    main_modes_direct: tuple[str, ...] = (
        "freeform",
        "freeform_direct",
        "freeform_brief_reasoning",
        "prompt_json",
        "answer_only_schema",
    )

    calendar_examples_per_mode: int = 200
    expanded_models: int = 4
    expanded_examples_per_family: int = 20

    backends: tuple[str, ...] = ("mlx", "hf", "vllm", "sglang")
    boundary_model: str = "Qwen2.5-3B-Instruct"
    main_checkpoints: tuple[str, ...] = (
        "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "SmolLM2-1.7B-Instruct",
    )

    output_modes_full: tuple[str, ...] = field(
        default_factory=lambda: (
            "freeform",
            "freeform_direct",
            "freeform_brief_reasoning",
            "prompt_json",
            "final_only_regex",
            "answer_only_schema",
            "rationale_answer_schema",
            "typed_trace_schema",
            "delayed_constraint",
        )
    )

    task_families: tuple[str, ...] = (
        "arithmetic_two_step",
        "symbolic_string",
        "object_tracking",
        "boolean_logic",
        "tool_call_argument",
    )
