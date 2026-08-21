"""Step 3.7 Flash configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.step37_flash.constants import (
    API_MODEL_ID,
    GITHUB_REPO,
    LICENSE,
    MODEL_PAGE,
)


@dataclass
class Step37FlashConfig:
    name: str = "Step 3.7 Flash"
    title: str = (
        "198B sparse MoE vision-language model for high-frequency agentic production workloads"
    )
    team: str = "StepFun"
    published: str = "2026"
    paper_arxiv: str = ""
    website: str = MODEL_PAGE
    github_repo: str = GITHUB_REPO
    license: str = LICENSE
    api_model_id: str = API_MODEL_ID
    ltx_hook: str = "Optional fast VLM caption backend for dataset_split_and_caption"
    default_reasoning_level: str = "medium"
    citation_note: str = field(
        default="See GitHub README; Apache-2.0 — stepfun-ai/Step-3.7-Flash"
    )
