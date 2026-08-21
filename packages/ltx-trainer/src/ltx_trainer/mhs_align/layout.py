"""MHS alignment pipeline layout."""

from __future__ import annotations

MHS_ATTRIBUTES: tuple[str, ...] = (
    "respect",
    "sentiment",
    "status",
    "hatespeech",
    "genocide",
    "dehumanize",
    "attack_defend",
    "violence",
    "humiliate",
    "insult",
)

BEHAVIORAL_ATTRIBUTES: tuple[str, ...] = (
    "insult",
    "humiliate",
    "violence",
    "attack_defend",
    "dehumanize",
    "genocide",
)

EVALUATIVE_ATTRIBUTES: tuple[str, ...] = (
    "respect",
    "sentiment",
    "status",
    "hatespeech",
)

PIPELINE_STAGES: tuple[str, ...] = (
    "per_attribute_llm_prompt_vanilla_or_persona",
    "extract_ordinal_label_and_token_confidence",
    "spearman_vs_human_per_attribute",
    "confidence_weighted_features_x_i = S_i * C_i",
    "ridge_regress_irt_hate_score",
    "compare_to_direct_prompting_baselines",
)

LIMITATIONS: tuple[str, ...] = (
    "Reference stub — not full MHS re-annotation or vLLM reproduction.",
    "Spearman and R² table values are paper-reported constants, not re-fit on Hub data.",
    "Toy Ridge smoke uses synthetic comments, not the 39k MHS corpus.",
    "Persona conditioning effects on confidence only sketched.",
)
