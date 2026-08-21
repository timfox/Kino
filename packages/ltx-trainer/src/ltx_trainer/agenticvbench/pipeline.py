"""AgenticVBench framework API (arXiv:2605.27705)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.agenticvbench.benchmarks import benchmarks_bundle
from ltx_trainer.agenticvbench.config import TASK_FAMILIES, AgenticVBenchConfig
from ltx_trainer.agenticvbench.examples import (
    demo_assembly_perfect,
    demo_repair_timeline,
    demo_repair_window,
    demo_repurpose_rubric,
    demo_sequencing_perfect,
)


def framework_card(cfg: AgenticVBenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AgenticVBenchConfig()
    return {
        "name": "AgenticVBench",
        "paper": f"arXiv:{cfg.arxiv}",
        "website": cfg.website,
        "tagline": "Can AI agents complete real-world post-production tasks?",
        "task_families": list(TASK_FAMILIES),
        "tasks_per_family": dict(cfg.tasks_per_family),
        "total_tasks": cfg.num_tasks,
        "evaluation": {
            "assembly": "chance-corrected slot accuracy (programmatic)",
            "repair": "linear reward vs broken/golden references (programmatic)",
            "sequencing": "(1-ND)*LIS*ADJ on clip permutations (programmatic)",
            "repurpose": "binary expert rubrics + format verifier",
        },
        "rollouts": {
            "reps": cfg.rollout_reps,
            "max_iterations": cfg.max_agent_iterations,
            "timeout_s": cfg.per_task_timeout_s,
        },
        "human_reference": dict(cfg.human_reference),
        "best_stack_overall": cfg.best_stack_overall,
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.27705",
        "authors": "Cao, Zheng, Song, Hu (Philo Labs Research)",
        "problem": "Multimodal agents on long-horizon video post-production, not single-shot VQA",
        "task_families": {
            "assembly": "storyboard-aligned clip selection from golden + distractors",
            "repair": "localize and fix injected A/V/timeline defects",
            "sequencing": "recover narrative order from shuffled shots",
            "repurpose": "brief-driven long-to-short deliverable (recap/trailer/social)",
        },
        "headline_result": "Best stack ~31% vs human expert ~81–95% per family",
        "harness_finding": "Harness choice shifts scores up to 20pp; typed multimodal tools matter",
        "failure_modes": {
            "repurpose": "long-context information loss (83%)",
            "repair": "temporal reasoning / wrong cut window (65%)",
        },
        "website": "https://agenticvbench.com",
    }


def evaluation_demo() -> dict[str, Any]:
    asm = demo_assembly_perfect()
    seq = demo_sequencing_perfect()
    rep_w = demo_repair_window()
    rep_t = demo_repair_timeline()
    rpu = demo_repurpose_rubric()
    bench = benchmarks_bundle()
    return {
        "assembly": asm,
        "sequencing": seq,
        "repair_window": rep_w,
        "repair_timeline": rep_t,
        "repurpose": rpu,
        "benchmarks": {
            "best_stack": bench["best_stack_by_family"],
            "human_reference": bench["human_reference"],
        },
        "all_verifiers_ok": (
            asm["score"] >= 0.99
            and seq["score"] >= 0.99
            and seq["strict_match"]
            and rep_w["reward"] > 0.5
            and rep_t["range_match"] >= 0.99
        ),
    }


def evaluation_smoke(cfg: AgenticVBenchConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo()
    bench = benchmarks_bundle()
    return {
        "paper": "arXiv:2605.27705",
        "all_verifiers_ok": demo["all_verifiers_ok"],
        "sequencing_score": demo["sequencing"]["score"],
        "assembly_score": demo["assembly"]["score"],
        "best_overall": bench["best_overall"],
        "ok": demo["all_verifiers_ok"] and bench["best_overall"] <= 0.35,
    }
