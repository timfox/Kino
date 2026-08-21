"""Prompt templates — § D.2.2 Figs. 13–18."""

from __future__ import annotations

from typing import Literal


def vision_count_prompt(*, board_label: str = "chessboard", synthetic: bool = True) -> str:
    if synthetic:
        return f"Question: How many black stones are there on the {board_label} ?"
    return (
        "You are a precise counting engine. Analyze the Go board. "
        "Format: Verification: <Dense Data> → FINAL ANSWER: <Result>.\n"
        f"Question: Analyze the {board_label}. Count the black stones."
    )


def text_count_prompt(*, target_char: str = "c") -> str:
    return f"Question: How many {target_char} letters are there in the given string ?"


def comparative_vision_text_prompt() -> str:
    return (
        "Question: Is the number of c letters in the given string the same as "
        "the number of black stones on the chessboard ?"
    )


def comparative_text_text_prompt() -> str:
    return "Question: Are the number of c letters in both input strings the same ?"


def prompt_bundle(kind: Literal["vision", "text", "v2t_compare", "t2t_compare"] = "vision") -> dict[str, str]:
    mapping = {
        "vision": vision_count_prompt(),
        "text": text_count_prompt(),
        "v2t_compare": comparative_vision_text_prompt(),
        "t2t_compare": comparative_text_text_prompt(),
    }
    return {"task": kind, "template": mapping[kind]}
