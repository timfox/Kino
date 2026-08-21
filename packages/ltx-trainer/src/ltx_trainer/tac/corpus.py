"""TaC example corpus for batch evaluation."""

from __future__ import annotations

from ltx_trainer.tac.pipeline import TaCExample, demo_film_example


def builtin_corpus() -> list[TaCExample]:
    """Small multi-example set for regression (no HotpotQA download)."""
    film = demo_film_example()
    return [
        film,
        TaCExample(
            example_id="capital",
            question="What is the capital of France?",
            context="Document: France is a country in Western Europe. Document: Paris is its capital and largest city.",
            gold="Paris",
            dataset="hotpotqa",
        ),
        TaCExample(
            example_id="year",
            question="In what year did the Apollo 11 mission land on the Moon?",
            context="Document: Apollo 11 was the spaceflight that landed the first humans on the Moon in 1969.",
            gold="1969",
            dataset="hotpotqa",
        ),
    ]
