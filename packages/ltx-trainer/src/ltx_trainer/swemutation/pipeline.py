"""SWE-Mutation benchmarks, tables, and demo evaluation (Sec. 4)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.swemutation.config import MULTILINGUAL_LANGS, SWEMutationConfig, TaskName
from ltx_trainer.swemutation.metrics import (
    absolute_mutation_score,
    pass_at_1,
    relative_detection_rate,
    verified_reproduction_rate,
)
from ltx_trainer.swemutation.mutation import compare_mutation_methods, mutation_strategy_catalog


def benchmark_table_test_repair() -> dict[str, dict[str, float]]:
    """Table 2 — test repair (selected models, both frameworks)."""
    return {
        "claude_sonnet_4_5": {
            "mini_pass": 97.20,
            "mini_vrr": 42.60,
            "mini_rdr": 79.30,
            "cc_pass": 99.80,
            "cc_vrr": 59.20,
            "cc_rdr": 81.15,
        },
        "deepseek_v3_1": {
            "mini_pass": 96.60,
            "mini_vrr": 33.00,
            "mini_rdr": 66.41,
            "cc_pass": 96.80,
            "cc_vrr": 58.20,
            "cc_rdr": 68.36,
        },
        "gpt_oss_120b": {
            "mini_pass": 74.80,
            "mini_vrr": 24.80,
            "mini_rdr": 36.31,
            "cc_pass": 86.80,
            "cc_vrr": 36.40,
            "cc_rdr": 39.28,
        },
    }


def benchmark_table_test_generation() -> dict[str, dict[str, float]]:
    """Table 3 — test generation."""
    return {
        "claude_sonnet_4_5": {
            "mini_pass": 96.20,
            "mini_vrr": 29.80,
            "mini_rdr": 63.70,
            "cc_pass": 98.00,
            "cc_vrr": 40.40,
            "cc_rdr": 71.71,
        },
        "deepseek_v3_1": {
            "mini_pass": 88.20,
            "mini_vrr": 10.20,
            "mini_rdr": 36.15,
            "cc_pass": 94.00,
            "cc_vrr": 20.40,
            "cc_rdr": 39.09,
        },
        "qwen3_coder": {
            "mini_pass": 86.20,
            "mini_vrr": 12.40,
            "mini_rdr": 33.33,
            "cc_pass": 95.20,
            "cc_vrr": 26.80,
            "cc_rdr": 33.21,
        },
    }


def benchmark_table_multilingual() -> dict[str, dict[str, float]]:
    """Table 4 — test repair, 9 languages averaged (Mini-Swe-Agent)."""
    return {
        "claude_sonnet_4_5": {"pass": 91.33, "vrr": 33.33, "rdr": 58.33},
        "deepseek_v3_1": {"pass": 86.00, "vrr": 20.33, "rdr": 36.67},
        "qwen3_coder": {"pass": 83.67, "vrr": 16.33, "rdr": 38.13},
        "kimi_k2": {"pass": 80.67, "vrr": 17.00, "rdr": 41.00},
        "glm_4_6": {"pass": 81.67, "vrr": 15.33, "rdr": 42.05},
    }


def mutation_strategy_rdr_table() -> dict[str, dict[str, float]]:
    """Table 5 — RDR by mutation strategy (test generation, agentic column)."""
    models = [
        "claude_sonnet_4_5",
        "claude_sonnet_3_7",
        "deepseek_v3_1",
        "qwen3_coder",
        "kimi_k2",
        "glm_4_6",
        "gpt_oss_120b",
    ]
    rule_based = [75.43, 73.25, 72.92, 72.16, 74.12, 73.88, 55.55]
    few_shot = [69.52, 55.25, 52.86, 50.18, 62.43, 59.55, 35.27]
    agentic = [63.70, 37.47, 36.15, 33.33, 42.59, 39.79, 25.61]
    return {
        m: {"rule_based": rb, "few_shot": fs, "agentic": ag}
        for m, rb, fs, ag in zip(models, rule_based, few_shot, agentic, strict=True)
    }


def mutant_quality_table() -> dict[str, dict[str, float]]:
    """Appendix Table 7 — compilability, realistic, coupling rates."""
    return {
        "rule_based": {"compilability": 100.0, "realistic": 38.0, "coupling": 39.0},
        "few_shot": {"compilability": 84.0, "realistic": 72.0, "coupling": 59.0},
        "agentic": {"compilability": 93.0, "realistic": 100.0, "coupling": 70.0},
    }


def locate_ablation_table() -> dict[str, dict[str, float]]:
    """Appendix Table 8 — Locate module ablation."""
    return {
        "wo_locate": {"validity_rate": 84.0, "f2p_trigger_rate": 15.0},
        "w_locate": {"validity_rate": 92.0, "f2p_trigger_rate": 69.0},
    }


def demo_instance_evaluation(
    *,
    pass_ok: bool = True,
    reproduces_bug: bool = True,
    passes_golden: bool = True,
    mutants_total: int = 5,
    mutants_killed_base: tuple[int, ...] = (0, 1),
    mutants_killed_gen: tuple[int, ...] = (0, 1, 2, 3),
    m_base_nonempty: bool = True,
) -> dict[str, float]:
    """Single-instance metric demo."""
    vrr = 100.0 if (pass_ok and reproduces_bug and passes_golden) else 0.0
    if m_base_nonempty:
        base_set = set(mutants_killed_base)
        all_idx = list(range(mutants_total))
        gen_set = set(mutants_killed_gen)
        survivors = [i for i in all_idx if i not in base_set]
        newly = len([i for i in gen_set if i in survivors])
        rdr = newly / len(survivors) * 100.0 if survivors else 0.0
    else:
        rdr = absolute_mutation_score(len(mutants_killed_gen), mutants_total)
    return {
        "pass_at_1": 100.0 if pass_ok else 0.0,
        "vrr": vrr,
        "rdr": rdr,
    }


def demo_micro_rdr() -> float:
    """Multi-instance RDR demo matching Eq. 1."""
    # instance 1: 4 mutants, base killed {0}, gen kills {2,3} of survivors {1,2,3}
    killed_gen = [{0, 2, 3}, {1, 2}]
    killed_base = [{0}, {1}]
    totals = [4, 3]
    return relative_detection_rate(killed_gen, killed_base, totals)


def dataset_card(cfg: SWEMutationConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SWEMutationConfig()
    return {
        "name": "SWE-Mutation",
        "paper": "arXiv:2605.22175",
        "title": "Can LLMs Generate Reliable Test Suites in Software Engineering?",
        "source": "SWE-bench Verified + SWE-bench-Multilingual",
        "python_instances": cfg.python_instances,
        "multilingual_instances": cfg.multilingual_instances,
        "total_mutants": cfg.total_mutants,
        "languages": list(MULTILINGUAL_LANGS),
        "tasks": ["test_generation", "test_repair"],
        "frameworks": ["mini_swe_agent", "claude_code"],
        "mutation_modules": ["locate", "mutation", "judge", "self_play"],
        "metrics": ["pass_at_1", "vrr", "rdr"],
        "code": "https://github.com/Sunny4Coding/SWE-Mutation",
    }


def evaluate_task_metrics(
    results: list[dict[str, bool]],
    *,
    task: TaskName = "test_repair",
) -> dict[str, float]:
    """Aggregate Pass@1 and VRR from per-instance boolean flags."""
    n = len(results)
    if n == 0:
        return {"pass_at_1": 0.0, "vrr": 0.0}
    pass_ok = sum(1 for r in results if r.get("pass_at_1", False))
    vrr_ok = sum(
        1
        for r in results
        if r.get("pass_at_1", False) and r.get("reproduces_bug", False) and r.get("passes_golden", False)
    )
    return {
        "pass_at_1": pass_at_1(pass_ok, n),
        "vrr": verified_reproduction_rate(vrr_ok, n),
        "task": task,
    }
