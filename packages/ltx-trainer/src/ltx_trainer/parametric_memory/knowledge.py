"""Paper facts card for agents (arXiv:2605.30260)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.parametric_memory.config import ParametricMemoryConfig
from ltx_trainer.parametric_memory.phase_transition import L_CRIT


def knowledge_card(cfg: ParametricMemoryConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ParametricMemoryConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": "How LoRA Remembers? A Parametric Memory Law for LLM Finetuning",
        "code": cfg.code_url,
        "law": "ΔL(r, ℓ) = C · r^α · ℓ^(-β) + b",
        "l_crit": cfg.l_crit,
        "l_crit_note": "L_crit = ln(2): greedy verbatim recall when P_target > 0.5",
        "memft": list(cfg.memft_variants),
        "benchmarks": [
            "Long-Context Memorization Stress Test (random-token LongBench mixtures)",
            "PhoneBook (deduplicated name→number, answer-only length buckets)",
            "Linear rule f(x,y)=3x+5y+7 (memory vs generalization)",
        ],
        "models": list(cfg.models),
        "lora_probe": "Frozen base + LoRA on selected MLP layers; rank r is capacity knob",
        "metrics": ["L (answer-only CE)", "Acctok", "AccEM under greedy decode"],
        "saturation_filter": f"Exclude samples with L_final ≤ {cfg.saturation_loss} when fitting law",
    }


def references_bibtex() -> str:
    return """@article{xu2026parametricmemory,
  title={How LoRA Remembers? A Parametric Memory Law for LLM Finetuning},
  author={Xu, Ziwen and Hong, Haiwen and Yu, Linsong and others},
  journal={arXiv preprint arXiv:2605.30260},
  year={2026}
}"""
