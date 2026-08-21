"""TaC / TaC-C pipeline and evaluation demo (arXiv:2605.28713)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ltx_trainer.tac.baselines import (
    DATASET_STATS,
    LOCOMO_4X,
    PILOT_STUDY,
    TABLE1_LLAMA_4X,
    TABLE1_LLAMA_8X,
    TABLE1_QWEN3_4X,
    TABLE1_QWEN3_8X,
    TABLE3_TRANSFER_8X,
    TABLE5_ABLATION,
    THINKER_SCALE_EM,
    TRACE_EXAMPLES,
)
from ltx_trainer.tac.config import TaCConfig
from ltx_trainer.tac.metrics import exact_match, f1_score, utility_reward
from ltx_trainer.tac.prompts import answerer_prompt, thinker_prompt, wrap_thinking
from ltx_trainer.tac.rewards import (
    extract_thinking,
    grpo_group_advantages,
    total_reward,
)


@dataclass
class TaCExample:
    """Single long-context QA instance for stub evaluation."""

    example_id: str
    question: str
    context: str
    gold: str
    dataset: str = "hotpotqa"


def demo_film_example() -> TaCExample:
    """Case study from paper Appendix E (13 At A Table vs São Paulo)."""
    docs = [
        'Document: 13 at a Table ("Tredici a tavola") is a 2004 Italian comedy film by Enrico Oldoini.',
        "Document: São Paulo, Sociedade Anônima is a 1965 Brazilian drama film by Luis Sérgio Person.",
        "Document: Trolleybuses in São Paulo: SPTrans (1949) and EMTU (1988) systems.",
        "Document: São Luís may refer to several places in the Portuguese-speaking world.",
        "Document: Copa São Paulo de Futebol Júnior is an under-20 football competition.",
    ]
    context = "\n".join(docs)
    return TaCExample(
        example_id="film-year",
        question="Which film came out earlier, 13 At A Table or São Paulo, Sociedade Anônima?",
        context=context,
        gold="São Paulo, Sociedade Anônima",
        dataset="hotpotqa",
    )


def _stub_thinker_trace(example: TaCExample, budget: int) -> str:
    """Deterministic compressed trace (no LLM call)."""
    if "earlier" in example.question.lower() and "film" in example.question.lower():
        body = (
            "13 at a Table is a 2004 Italian comedy (Enrico Oldoini). "
            "São Paulo, Sociedade Anônima is a 1965 Brazilian drama (Luis Sérgio Person). "
            "Compare release years 2004 vs 1965; the 1965 film is earlier. "
            "Do not answer yet — evidence only for downstream use."
        )
    else:
        body = f"Query-relevant summary for: {example.question[:80]}. Key facts from context preserved."
    return wrap_thinking(body)


def _stub_answerer(question: str, thinking: str, gold: str) -> str:
    """Heuristic answerer for smoke tests."""
    t = thinking.lower()
    if "1965" in t and "2004" in t and "earlier" in question.lower():
        return "São Paulo, Sociedade Anônima"
    if gold.lower() in t:
        return gold
    return gold.split()[0] if gold else "unknown"


def tac_vanilla(
    example: TaCExample,
    *,
    compression_ratio: int = 4,
    cfg: TaCConfig | None = None,
) -> dict[str, Any]:
    """Prompt-only TaC (TaC-Vanilla): thinking trace as compressed context."""
    cfg = cfg or TaCConfig()
    ctx_len = max(1, len(example.context) // 4)
    budget = max(32, ctx_len // compression_ratio)
    raw = _stub_thinker_trace(example, budget)
    thinking = extract_thinking(raw, cfg)
    pred = _stub_answerer(example.question, thinking, example.gold)
    return {
        "thinking": thinking,
        "raw_output": raw,
        "prediction": pred,
        "budget": budget,
        "context_tokens_est": ctx_len,
        "thinking_tokens_est": len(thinking) // 4,
        "em": exact_match(pred, example.gold),
        "f1": f1_score(pred, example.gold),
        "utility": utility_reward(pred, example.gold),
    }


def tac_c_select_trace(
    example: TaCExample,
    *,
    compression_ratio: int = 4,
    cfg: TaCConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """TaC-C stub: sample group of traces, score with TaC-C rewards, pick best."""
    cfg = cfg or TaCConfig()
    ctx_len = max(1, len(example.context) // 4)
    budget = max(32, ctx_len // compression_ratio)
    rng = random.Random(seed)
    candidates: list[str] = []
    # Good trace
    candidates.append(_stub_thinker_trace(example, budget))
    # Hack trace (direct answer)
    candidates.append(
        wrap_thinking(f"The answer is {example.gold}.", cfg)
    )
    # Overlong trace
    candidates.append(wrap_thinking(" ".join(["padding context"] * budget * 3), cfg))
    # Fill group
    while len(candidates) < cfg.grpo_group_size:
        candidates.append(
            wrap_thinking(
                f"Partial evidence for: {example.question}. Fact {rng.randint(0, 99)}.",
                cfg,
            )
        )

    scored: list[dict[str, Any]] = []
    for raw in candidates[: cfg.grpo_group_size]:
        thinking = extract_thinking(raw, cfg)
        pred = _stub_answerer(example.question, thinking, example.gold)
        rb = total_reward(raw, budget=budget, prediction=pred, gold=example.gold, cfg=cfg)
        scored.append({"raw": raw, "prediction": pred, "reward": rb})

    rewards = [s["reward"].total for s in scored]
    advantages = grpo_group_advantages(rewards)
    best_idx = max(range(len(scored)), key=lambda i: scored[i]["reward"].total)
    best = scored[best_idx]
    return {
        "budget": budget,
        "compression_ratio": compression_ratio,
        "candidates": len(scored),
        "advantages": advantages,
        "best_index": best_idx,
        "best_reward": {
            "total": best["reward"].total,
            "utility": best["reward"].utility,
            "budget": best["reward"].budget,
            "hack_gate": best["reward"].hack_gate,
            "is_hack": best["reward"].is_hack,
        },
        "prediction": best["prediction"],
        "thinking": extract_thinking(best["raw"], cfg),
        "em": exact_match(best["prediction"], example.gold),
        "f1": f1_score(best["prediction"], example.gold),
    }


def framework_card(cfg: TaCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TaCConfig()
    return {
        "name": "Thinking as Compression (TaC)",
        "paper": "arXiv:2605.28713",
        "authors": "Ma et al. (Baidu, Xi'an Jiaotong University, …)",
        "paradigm": "Thinking trace o ~ p(o|q,C,B); answer ŷ ~ p(ŷ|q,o); |o| ≤ B",
        "variants": {
            "TaC-Vanilla": "Prompt-only thinker → trace as compressed context",
            "TaC-C": "GRPO + utility + budget + anti-hack rewards on thinker",
        },
        "thinker_answerer_decoupled": True,
        "compression_ratios": list(cfg.compression_ratios),
        "datasets": list(cfg.datasets),
        "paper_metrics_qwen3_8b": {
            "4x": {"f1": cfg.paper_f1_4x, "em": cfg.paper_em_4x},
            "8x": {"f1": cfg.paper_f1_8x, "em": cfg.paper_em_8x},
        },
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28713",
        "positioning": [
            "vs task-agnostic compression (LLMLingua, ICAE, PCC): query-conditioned organization",
            "vs task-aware filtering (EXIT, Provence, LongLLMLingua): dynamic read-organize-integrate",
            "vs CoT: trace is reusable compressed context, not just rationale for same model",
        ],
        "limitations": [
            "Extremely long contexts remain challenging",
            "Code / agent trajectories under-explored",
            "Full VeRL training not bundled in stub",
        ],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table1_qwen3_4x": TABLE1_QWEN3_4X,
        "table1_qwen3_8x": TABLE1_QWEN3_8X,
        "table1_llama_4x": TABLE1_LLAMA_4X,
        "table1_llama_8x": TABLE1_LLAMA_8X,
        "table3_transfer_8x": TABLE3_TRANSFER_8X,
        "table5_ablation": TABLE5_ABLATION,
        "table6_dataset_stats": DATASET_STATS,
        "pilot_study_fig2": PILOT_STUDY,
        "thinker_scale_fig4": THINKER_SCALE_EM,
        "trace_examples_table2": TRACE_EXAMPLES,
        "locomo_4x": LOCOMO_4X,
    }


def evaluation_demo(seed: int = 42, *, compression_ratio: int = 4) -> dict[str, Any]:
    ex = demo_film_example()
    vanilla = tac_vanilla(ex, compression_ratio=compression_ratio)
    constrained = tac_c_select_trace(ex, compression_ratio=compression_ratio, seed=seed)
    return {
        "seed": seed,
        "compression_ratio": compression_ratio,
        "example_id": ex.example_id,
        "question": ex.question,
        "gold": ex.gold,
        "tac_vanilla": vanilla,
        "tac_c": constrained,
        "paper_avg_qwen3_tac_c": {
            "4x_f1": 65.12,
            "4x_em": 51.17,
            "8x_f1": 65.42,
            "8x_em": 51.79,
        },
        "thinker_prompt_preview": thinker_prompt(
            ex.context[:200] + "...",
            ex.question,
            target_think_len=vanilla["budget"],
            context_token_len=vanilla["context_tokens_est"],
            comp_ratio=compression_ratio,
        )[:400],
    }
