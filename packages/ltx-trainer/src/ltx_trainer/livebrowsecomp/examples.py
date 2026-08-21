"""Bundled demo trajectories and filter examples (offline)."""

from __future__ import annotations

from ltx_trainer.livebrowsecomp.filters import SeedEvent, filter_seed_pipeline
from ltx_trainer.livebrowsecomp.trajectory import RetrievalHit, SearchRound, SearchTrajectory


def demo_ikd_trajectory() -> SearchTrajectory:
    """Model-led search: query terms appear in reasoning before retrieval."""
    return SearchTrajectory(
        question_id="demo-ikd",
        rounds=[
            SearchRound(
                round_idx=0,
                query="WorkTime vulnerability Canada bird vendor WHO report",
                reasoning_before="The product may be WorkTime based on labor-themed naming.",
                hits=[RetrievalHit(snippet="Unrelated sports result.", is_gold=False)],
                reasoning_after="Need CVE cross-reference.",
            ),
            SearchRound(
                round_idx=1,
                query="email marketing PHP vulnerability 2020",
                reasoning_before="Email marketing tool vulnerability year is likely 2020.",
                hits=[
                    RetrievalHit(
                        snippet="CVE-2020-XXXX email marketing PHP file upload.",
                        is_gold=True,
                        is_evidence=True,
                    )
                ],
                reasoning_after="Confirmed 2020.",
            ),
        ],
        final_answer="<answer>2020</answer>",
    )


def demo_filter_examples() -> list[dict]:
    events = [
        SeedEvent("tmdb", "m1", 45, {"popularity": 0.5, "vote_count": 5, "revenue": 0}),
        SeedEvent("cve", "c1", 20, {"cvss": 9.2, "single_product": True, "published_days_ago": 15}),
        SeedEvent("sports", "s1", 30, {"sport": "cricket", "attendance": 12000}),
    ]
    return [filter_seed_pipeline(e) for e in events]
