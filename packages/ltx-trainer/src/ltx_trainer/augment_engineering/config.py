"""Augment Engineering configuration (arXiv:2605.26146)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AugmentEngConfig:
    paper_arxiv: str = "2605.26146"
    paper_title: str = (
        "Augment Engineering: A Methodology for Multi-Tool AI Orchestration "
        "Across Professional Domains"
    )
    authors: str = "Elias Calboreanu (Swift North AI Lab, The Swift Group, LLC)"

    orchestration_phases: tuple[str, ...] = (
        "domain_inventory",
        "tool_mapping",
        "skill_transfer_assessment",
        "integration_design",
        "orchestration_execution",
        "portfolio_optimization",
    )

    portability_metrics: tuple[str, ...] = (
        "transfer_velocity",
        "cross_domain_output_quality",
        "orchestration_overhead",
        "coverage_breadth",
    )

    prompt_levels: tuple[str, ...] = ("L1", "L2", "L3", "L4")

    # Paper Table 3: 5 AI tools + 5 infrastructure (portability claim on AI tools)
    ai_tools: tuple[str, ...] = (
        "Claude",
        "Gamma.app",
        "HeyGen",
        "MLX",
        "Tesseract OCR",
    )
    infrastructure: tuple[str, ...] = (
        "Jira",
        "GitHub/Git",
        "Vercel",
        "LaTeX/BibTeX",
        "WeasyPrint",
    )

    case_study_domains: tuple[str, ...] = field(
        default_factory=lambda: (
            "software_development",
            "academic_publication",
            "proposals",
            "curriculum_design",
            "video_production",
            "presentation_design",
            "web_deployment",
        )
    )
