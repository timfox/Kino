"""ID-LoRA: identity-driven in-context LoRA for LTX AV (arXiv:2603.10256)."""

from ltx_trainer.id_lora.benchmarks import BENCHMARK_TABLES, celebvhq_cross_video_row, talkvid_in_domain_row
from ltx_trainer.id_lora.config import IdLoraConfig
from ltx_trainer.id_lora.inference_plan import build_inference_argv, inference_defaults
from ltx_trainer.id_lora.mock import evaluation_smoke
from ltx_trainer.id_lora.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    framework_card,
    knowledge_card,
    training_plan,
)
from ltx_trainer.id_lora.prompts import ParsedIdLoraPrompt, build_structured_prompt, parse_id_lora_prompt

__all__ = [
    "BENCHMARK_TABLES",
    "IdLoraConfig",
    "ParsedIdLoraPrompt",
    "benchmarks_bundle",
    "build_inference_argv",
    "build_structured_prompt",
    "celebvhq_cross_video_row",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "inference_defaults",
    "knowledge_card",
    "parse_id_lora_prompt",
    "talkvid_in_domain_row",
    "training_plan",
]
