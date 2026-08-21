"""PDE survey pipeline and limitations."""

from __future__ import annotations

PIPELINE_STAGES: tuple[str, ...] = (
    "formalize_instance_and_dataset_pde",
    "threat_model_query_only_adversary",
    "scenario_taxonomy_table1",
    "attack_methods_mia_and_contamination",
    "defense_mitigation_survey",
    "sota_comparison_table2",
    "open_challenges_future_directions",
)

LIMITATIONS: tuple[str, ...] = (
    "Survey stub only — no LLM pretraining corpus, Min-K%++, or EM-MIA reproduction.",
    "Text-only LLMs; excludes multimodal and low-resource settings per paper Sec. 8.",
    "Toy perplexity/MIA scores are illustrative, not benchmark-faithful attacks.",
    "Table 2 availability flags are paper excerpts, not live repository audits.",
)
