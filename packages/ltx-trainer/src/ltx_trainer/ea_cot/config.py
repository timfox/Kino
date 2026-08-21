"""Entity-Aware CoT for SLLM reasoning (arXiv:2606.04474)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EACoTConfig:
    paper_arxiv: str = "arXiv:2606.04474"
    title: str = (
        "Entity Binding Failures in Speech LLM Reasoning: "
        "Diagnosis and Chain-of-Thought Intervention"
    )
    benchmark: str = "VoiceBench BBH (4 categories, 1,000 items)"
    categories: tuple[str, ...] = ("hyperbaton", "navigate", "sports understanding", "web of lies")
    category_abbrev: tuple[str, ...] = ("HYP", "NAV", "SPO", "WOL")

    models: tuple[str, ...] = ("Qwen/Qwen2.5-Omni-7B", "microsoft/Phi-4-multimodal-instruct")
    baseline_tokens: int = 256
    ea_cot_tokens: int = 1024

    # Table 1 headline: Phi-4 WOL S2T +24.4 pp
    max_wol_s2t_gain_pp: float = 24.4
    generic_cot_wol_gain_pp: float = 2.4

    demo_entity_dim: int = 32
    demo_claim_len: int = 16
