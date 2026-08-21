"""PEMark — watermark API responses via key ordering (arXiv:2605.21865)."""

from ltx_trainer.pemark.attacks import deletion_attack, insertion_attack, tamper_attack_values_stub, watermark_similarity
from ltx_trainer.pemark.config import PEMarkConfig
from ltx_trainer.pemark.layout import LIMITATIONS
from ltx_trainer.pemark.pipeline import (
    evaluation_demo,
    fig5_threshold_vs_length_points,
    framework_card,
    pipeline_demo,
    robustness_excerpt,
    table_i_method_comparison,
    table_ii_dataset_configs,
    table_iii_api_latency,
)
from ltx_trainer.pemark.position_encoding import (
    embed_watermark_into_keys,
    extract_watermark_from_keys,
    extract_with_group_voting,
    factorial_decompose,
    factorial_recompose,
    invert_lehmer_coeffs,
    int_to_watermark_bits,
    lehmer_permutation,
    majority_vote_bitstrings,
    min_threshold_T_for_bits,
    watermark_bits_to_int,
)

__all__ = [
    "LIMITATIONS",
    "PEMarkConfig",
    "deletion_attack",
    "embed_watermark_into_keys",
    "evaluation_demo",
    "extract_watermark_from_keys",
    "extract_with_group_voting",
    "factorial_decompose",
    "factorial_recompose",
    "fig5_threshold_vs_length_points",
    "framework_card",
    "insertion_attack",
    "int_to_watermark_bits",
    "invert_lehmer_coeffs",
    "lehmer_permutation",
    "majority_vote_bitstrings",
    "min_threshold_T_for_bits",
    "pipeline_demo",
    "robustness_excerpt",
    "table_i_method_comparison",
    "table_ii_dataset_configs",
    "table_iii_api_latency",
    "tamper_attack_values_stub",
    "watermark_bits_to_int",
    "watermark_similarity",
]

