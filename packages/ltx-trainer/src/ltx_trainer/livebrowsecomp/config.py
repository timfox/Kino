"""LiveBrowseComp configuration (arXiv:2605.28721)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiveBrowseCompConfig:
    """Reference constants from paper."""

    arxiv: str = "2605.28721"
    hub_dataset: str = "Forival/LiveBrowseComp"
    num_questions: int = 335
    recency_days: int = 90
    human_solve_rate_browsecomp: float = 0.30
    human_solve_rate_live: float = 0.31
    # Pilot closed-book pass@4 averages (Fig 2 narrative)
    closed_book_avg_pass4: float = 38.9
    # Trajectory (§2.3)
    model_originated_query_rate: float = 0.55  # >50%, increases to >60%
    evidence_use_rate_after_retrieval: float = 0.30  # below 1/3
    seed_sources: tuple[str, ...] = (
        "GDELT",
        "TMDB",
        "RAWG",
        "CVE/NVD",
        "SportsDB",
        "USGS",
    )
