"""MMAE benchmark constants (arXiv:2606.07229)."""

from __future__ import annotations

MMAE_PAPER_ARXIV = "arXiv:2606.07229"
MMAE_PAPER_TITLE = (
    "MMAE: A Massive Multitask Audio Editing Benchmark"
)
MMAE_GITHUB = "https://github.com/ddlBoJack/MMAE"
MMAE_HUB_DATASET = "BoJack/MMAE"
MMAE_HUB_ENV = "GOPEX_MMAE_ROOT"
MMAE_JUDGER = "Qwen3-Omni"

# Table 1 key statistics
MMAE_TOTAL_SAMPLES = 2000
MMAE_TOTAL_RUBRICS = 17_741
MMAE_AVG_RUBRICS_PER_SAMPLE = 8.87
MMAE_AVG_IF_RUBRICS = 3.58
MMAE_AVG_CR_RUBRICS = 5.29
MMAE_AVG_DURATION_SEC = 14.46
MMAE_AVG_INSTRUCTION_WORDS = 14.0
MMAE_AVG_OPERATIONS = 1.22
MMAE_AVG_CHOICES = 3.53
MMAE_SHORT_SUBSET_MAX_SEC = 10.0
MMAE_SHORT_SUBSET_COUNT = 801

# Evaluated model names (Table 2)
MMAE_MODELS_FULL = ("Step-Audio-EditX", "Ming-UniAudio")
MMAE_MODELS_SHORT = ("MMEdit", "Audio-Omni", "SmartDJ w/o planner", "SmartDJ w/ planner")
MMAE_BASELINES = ("Identity", "Noise")
