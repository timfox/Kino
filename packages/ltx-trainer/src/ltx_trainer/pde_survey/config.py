"""PDE survey configuration (arXiv:2605.26133)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PdeSurveyConfig:
    paper_arxiv: str = "2605.26133"
    paper_title: str = (
        "Pretraining Data Exposure in Large Language Models: A Survey of "
        "Membership Inference, Data Contamination, and Security Implications"
    )
    authors: str = "Ziyi Tong, Feifei Sun, Le Minh Nguyen (JAIST)"

    exposure_levels: tuple[str, ...] = ("instance", "dataset")
    domains: tuple[str, ...] = ("data_contamination", "membership_inference")

    # Toy MIA / perplexity thresholds for smoke
    member_ppl_ceiling: float = 12.0
    nonmember_ppl_floor: float = 28.0

    scenarios: tuple[str, ...] = field(
        default_factory=lambda: (
            "nlp_benchmark_contamination",
            "personal_data_exposure",
            "copyright_ip_risks",
            "code_software_security",
        )
    )
