"""AUTOSCIENTISTS configuration (arXiv:2605.28655)."""

from __future__ import annotations

from dataclasses import dataclass, field

AGENT_ROLES = ("analyst", "experiment")
DEFAULT_TEAMS = ("architecture", "schedule", "throughput")
BIOML_DOMAINS = ("biomedical_imaging", "drug_discovery", "protein_engineering", "single_cell_omics")


@dataclass
class AutoScientistsConfig:
    arxiv: str = "2605.28655"
    website: str = "https://autoscientists.openscientist.ai"
    code_repo: str = "https://github.com/mims-harvard/AutoScientists"
    institution: str = "Harvard University"
    llm_backend: str = "claude-sonnet-4.6"
    coding_agent: str = "claude-code"
    default_analysts: int = 3
    default_experiment_agents: int = 6
    noise_band_multiplier: float = 2.0
    stagnation_window: int = 10
    bioml_tasks: int = 24
    bioml_mean_leaderboard_pct: float = 74.40
    autoresearch_mean_leaderboard_pct: float = 66.07
    gpt_baseline_val_bpb: float = 0.998
    gpt_target_val_bpb: float = 0.978
    gpt_experiments_to_target: int = 34
    autoresearch_experiments_to_target: int = 65
    proteingym_kermut_spearman: float = 0.657
    proteingym_autoscientists_spearman: float = 0.700
    ace2_kermut_spearman: float = 0.747
    ace2_autoscientists_spearman: float = 0.840
