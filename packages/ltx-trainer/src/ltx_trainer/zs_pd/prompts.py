"""Prompt templates — Table 2."""

from __future__ import annotations

LLM_PROMPT_TEMPLATE = (
    "Task: You are a clinical classification model. Based on the audio features "
    "extracted from a person's speech, classify whether a person has Parkinson's "
    "disease or not. Output 1 if the person has Parkinson's disease, or 0 if the "
    "person is healthy. Instruction: Respond with exactly one token: 0 or 1. "
    "Now solve the following. Input: {list} Output:"
)

LALM_PROMPT_TEMPLATE = (
    "You are an audio analysis model. Your task is to decide whether the speech "
    "characteristics are more consistent with healthy control speech or Parkinson's. "
    "Consider acoustic cues, including pitch variability, loudness variability over "
    "time, articulation precision of consonants, voice quality (breathy, hoarse, "
    "strained), speech rate, and rhythm. Make a balanced decision based only on the "
    "provided audio. Output only a single digit: 0 = Healthy or 1 = Parkinson's disease."
)


def build_llm_prompt(feature_list: str) -> str:
    return LLM_PROMPT_TEMPLATE.format(list=feature_list)
