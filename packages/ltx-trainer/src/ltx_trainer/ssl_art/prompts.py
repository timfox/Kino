"""Text prompts for zero-shot style/genre classification (Sec. 4.2)."""

from __future__ import annotations


def style_prompt(class_name: str) -> str:
    return f"A painting in the {class_name} style."


def genre_prompt(genre_name: str) -> str:
    return f"A {genre_name} painting."


def build_label_prompts(labels: list[str], *, task: str = "style") -> list[str]:
    if task == "genre":
        return [genre_prompt(g) for g in labels]
    return [style_prompt(s) for s in labels]
