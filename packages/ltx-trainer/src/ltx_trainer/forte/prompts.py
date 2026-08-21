"""LLM elaboration and CapBench evaluation prompts (arXiv:2606.05812)."""

from __future__ import annotations

POSITIVE_ELABORATION_TEMPLATE = """You are an expert in audio scene description and sound event recognition. Given a short text query that describes an audio event or scene, generate {n_pos} enriched elaborations. Each elaboration should preserve the core sound event while adding precise acoustic and contextual detail that would help distinguish the target audio from similar but incorrect matches.

Rules:
- Do NOT change or substitute the core sound event.
- Add at most 3 new descriptors per elaboration (acoustic quality, temporal pattern, spatial environment, intensity).
- Do NOT introduce new sound sources not implied by the original query.
- Each elaboration must be a single fluent grammatical sentence of at most 20 words.
- Output as a numbered list, one elaboration per line.

Query: {query}
Elaborations:"""

CONTRASTIVE_NEGATIVE_TEMPLATE = """You are an expert in audio scene description and sound event recognition. Given a short text query, generate {n_neg} contrastive negatives. A contrastive negative describes an audio event that shares surface similarity with the query but differs critically in acoustic character.

Rules:
- Keep the same primary sound source.
- Change exactly one acoustic dimension to something clearly different.
- Each negative must be a single fluent grammatical sentence of at most 20 words.
- Output as a numbered list, one negative per line.

Query: {query}
Contrastive negatives:"""

CAPBENCH_CAPTION_PROMPT = """Given image frames uniformly sampled from a video clip, describe the video in detail, focusing on main subjects, actions, and background. Pay attention to details that may violate trust-and-safety policy. Maximum length: 150 words."""


def format_positive_prompt(query: str, *, n_pos: int = 2) -> str:
    return POSITIVE_ELABORATION_TEMPLATE.format(n_pos=n_pos, query=query.strip())


def format_negative_prompt(query: str, *, n_neg: int = 3) -> str:
    return CONTRASTIVE_NEGATIVE_TEMPLATE.format(n_neg=n_neg, query=query.strip())


def prompt_bundle() -> dict[str, object]:
    return {
        "elaboration_llm": "Mistral-7B-Instruct-v0.3",
        "n_pos_default": 2,
        "n_neg_default": 3,
        "verbaliser_templates": 9,
    }
