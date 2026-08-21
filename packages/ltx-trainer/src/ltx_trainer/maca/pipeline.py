"""MACA framework card, paper tables, and smoke demos (arXiv:2605.25746)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.maca.config import MACAConfig
from ltx_trainer.maca.embeddings import AgentProfile
from ltx_trainer.maca.layout import architecture_layout, paper_limitations
from ltx_trainer.maca.orchestration import compare_with_without_graphspec, grpo_update_demo
from ltx_trainer.maca.prior import build_graphspec


def _default_agents() -> list[AgentProfile]:
    # Token costs are just reference values for simulated efficiency.
    return [
        AgentProfile("TaskPlanner", "route task, propose minimal agent set", 180),
        AgentProfile("AlgorithmDesigner", "design strategy, data structures", 260),
        AgentProfile("CodeWriting", "write executable solution", 520),
        AgentProfile("CodeReviewer", "check correctness and style", 240),
        AgentProfile("UnitTestWriter", "write tests and edge cases", 240),
        AgentProfile("EdgeCaseHunter", "find adversarial cases", 200),
        AgentProfile("BugFixer", "patch failing implementation", 380),
        AgentProfile("Summarizer", "compress state", 160),
        AgentProfile("BudgetController", "reduce token waste", 120),
        AgentProfile("RedTeamCritic", "stress-test assumptions", 220),
        # A couple of non-code agents used in the paper's pools:
        AgentProfile("MathSolver", "solve derived equations reliably", 420),
        AgentProfile("ArithmeticChecker", "recompute arithmetic quickly", 160),
    ]


def framework_card(cfg: MACAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MACAConfig()
    return {
        "name": "MACA (Multi-Agent Coordination Adaptation)",
        "paper": cfg.paper_arxiv,
        "github": cfg.github,
        "idea": "Posterior inference over structure G and orchestration τ; GraphSpec prior guides token-aware orchestration.",
        "components": ["GraphSpec structural prior (Z_prior, P_prior)", "Hard mask + soft KL anchoring", "GRPO-style policy optimization"],
        "default_gamma": cfg.relevance_threshold_gamma,
        "default_lambda": cfg.kl_alpha,
        "default_budget_controller": True,
    }


def table_main_results_llama31_8b() -> dict[str, dict[str, dict[str, float]]]:
    """Table 1 excerpt (Llama-3.1-8B) — accuracy and avg cost on 6 benchmarks."""
    return {
        "HumanEval": {
            "Puppeteer": {"acc": 71.49, "avg_cost": 2798.3},
            "MACA": {"acc": 75.76, "avg_cost": 2100.1},
        },
        "MBPP": {
            "Puppeteer": {"acc": 47.17, "avg_cost": 3696.5},
            "MACA": {"acc": 49.23, "avg_cost": 2412.3},
        },
        "MMLU-Pro": {
            "Puppeteer": {"acc": 51.25, "avg_cost": 4097.8},
            "MACA": {"acc": 52.67, "avg_cost": 2117.8},
        },
        "ARC-C": {
            "Puppeteer": {"acc": 85.57, "avg_cost": 3753.3},
            "MACA": {"acc": 87.75, "avg_cost": 1656.3},
        },
        "SVAMP": {
            "Puppeteer": {"acc": 94.18, "avg_cost": 4204.7},
            "MACA": {"acc": 96.00, "avg_cost": 2057.2},
        },
        "GSM-Hard": {
            "Puppeteer": {"acc": 49.25, "avg_cost": 4196.6},
            "MACA": {"acc": 50.30, "avg_cost": 1602.5},
        },
    }


def table_ablation_arc_gsm() -> dict[str, dict[str, float]]:
    """Table 3 excerpt — ARC-C and GSM-Hard ablations."""
    return {
        "MACA": {"arc_acc": 87.75, "arc_cost": 1656.3, "gsm_acc": 50.30, "gsm_cost": 1602.5},
        "w/o Z_prior": {"arc_acc": 82.64, "arc_cost": 1920.4, "gsm_acc": 44.37, "gsm_cost": 1885.2},
        "w/o P_prior": {"arc_acc": 81.31, "arc_cost": 1896.7, "gsm_acc": 43.92, "gsm_cost": 1830.8},
        "w/o GraphSpec": {"arc_acc": 72.35, "arc_cost": 2285.9, "gsm_acc": 36.80, "gsm_cost": 2050.1},
        "w/o policy": {"arc_acc": 85.53, "arc_cost": 2109.6, "gsm_acc": 47.96, "gsm_cost": 1785.5},
    }


def table_sensitivity() -> dict[str, dict[str, float]]:
    """Figure 7 qualitative defaults."""
    return {
        "gamma": {"recommended": 0.4, "interpretation": 0.0},
        "lambda": {"recommended": 0.7, "interpretation": 0.0},
    }


def graphspec_demo(task_text: str = "Write Python code: is_palindrome(s) with alnum filtering") -> dict[str, Any]:
    cfg = MACAConfig()
    agents = _default_agents()
    gs = build_graphspec(task_text, agents, budget_tokens=1800, cfg=cfg)
    mask = gs.mask(z_threshold=0.0, p_threshold=0.15)
    return {
        "task": task_text,
        "agents": list(gs.agent_names),
        "z_prior": {n: float(v) for n, v in zip(gs.agent_names, gs.z_prior.tolist(), strict=True)},
        "edge_mass_mean": float(gs.p_prior.mean().item()),
        "mask_density": float(mask.to(torch.float32).mean().item()),
    }


def evaluation_demo(cfg: MACAConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MACAConfig()
    agents = _default_agents()
    task = "Write Python code: implement is_palindrome(s) ignoring case and non-alphanumeric"
    compare = compare_with_without_graphspec(task, agents, budget_tokens=1800, cfg=cfg)
    update = grpo_update_demo(task, agents, budget_tokens=1800, cfg=cfg, with_graphspec=True)
    return {
        "graphspec_demo": graphspec_demo(task),
        "compare": compare,
        "grpo_update": update,
        "architecture_layout": architecture_layout(),
        "limitations": paper_limitations(),
        "paper_tables": {
            "main_results_llama31_8b": table_main_results_llama31_8b(),
            "ablation_arc_gsm": table_ablation_arc_gsm(),
        },
    }

