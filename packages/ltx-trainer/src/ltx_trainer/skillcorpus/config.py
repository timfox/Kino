"""SkillCorpus — curated open SKILL.md ecosystem (arXiv:2607.15557)."""

from __future__ import annotations

from dataclasses import dataclass

PAPER_ARXIV = "2607.15557"
PAPER_TITLE = (
    "SkillCorpus: Consolidating and Evaluating the Open Skill Ecosystem "
    "for Real-World LLM Agents"
)
PAPER_SYSTEM = "SkillCorpus"
PAPER_AUTHORS = (
    "Yanze Wang, Pengfei Yao, Tianyi Sun, Chuanrui Hu, Yan Xiao, "
    "Yunyun Han, Jun Sun, Yafeng Deng (EverMind / Shanda / PKU)"
)
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV}"
PAPER_GITHUB = ""  # release upon acceptance
BENCHMARK = (
    "SkillsBench (87) + GDPVal (220) + QwenClawBench (100); "
    "OpenClaw / Raven harnesses; Qwen3.5-27B / 397B + Opus 4.7 check"
)

COMPONENTS = (
    "multi_source_aggregation",
    "six_stage_curation",
    "three_facet_quality",
    "sixteen_class_taxonomy",
    "retrieve_rerank_select",
)

TAXONOMY_CLASSES = (
    "Dev",
    "Data",
    "Writing",
    "DevOps-Infra",
    "Multimedia",
    "Testing",
    "AI-ML",
    "Workflow",
    "Frontend-UI",
    "Security",
    "Meta",
    "Productivity",
    "Doc-Proc",
    "Comms",
    "Other",
    "Auth",
)

HARD_GATE_FLAGS = (
    "prompt_injection",
    "cmd_injection",
    "unsafe_exec",
    "auth_bypass",
    "csam_risk",
)


@dataclass
class SkillCorpusConfig:
    """Runtime knobs for the CPU stub."""

    n_active_skills: int = 96401
    n_raw_crawl: int = 821000
    embed_dim: int = 1024
    top_k_select: int = 2
    cosine_auto_merge: float = 0.995
    cosine_borderline: float = 0.90
    content_q_weights: tuple[float, float, float] = (0.50, 0.35, 0.15)
