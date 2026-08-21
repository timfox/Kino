"""AUTOSCIENTISTS API (arXiv:2605.28666)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.autoscientists.benchmarks import benchmarks_bundle
from ltx_trainer.autoscientists.config import AGENT_ROLES, AutoScientistsConfig
from ltx_trainer.autoscientists.examples import demo_gpt_speedup, demo_noise_gate_confirm, demo_parallel_teams


def framework_card(cfg: AutoScientistsConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AutoScientistsConfig()
    return {
        "name": "AUTOSCIENTISTS",
        "paper": f"arXiv:{cfg.arxiv}",
        "website": cfg.website,
        "code_repo": cfg.code_repo,
        "institution": cfg.institution,
        "design": "Decentralized self-organizing agent teams; no central planner",
        "shared_state": ["champion p*", "experiment log L", "forum F", "team queues Qk", "dead-ends Dk"],
        "roles": list(AGENT_ROLES),
        "phases": ["discussion", "parallel_execution", "re_discussion_on_stagnation"],
        "default_team": {
            "analysts": cfg.default_analysts,
            "experiment_agents": cfg.default_experiment_agents,
        },
        "benchmarks": {
            "bioml_tasks": cfg.bioml_tasks,
            "bioml_mean_lb_pct": cfg.bioml_mean_leaderboard_pct,
        },
        "llm": {"coding_agent": cfg.coding_agent, "model": cfg.llm_backend},
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28655",
        "authors": "Gao, Fang, Zitnik (Harvard)",
        "problem": "Long-running ML experimentation needs parallel hypotheses, failure memory, team re-org",
        "mechanisms": [
            "Forum critique before experiments run",
            "Analyst queues + experiment agents in parallel teams",
            "Noise-aware champion promotion (anti pollution)",
            "Discussion-triggered team reformation on stagnation",
        ],
        "results": {
            "bioml_mean_leaderboard_pct": "74.4% vs 66.1% Autoresearch",
            "gpt_speedup": "1.9× fewer experiments to matched val_bpb",
            "proteingym": "0.700 vs 0.657 Kermut avg Spearman (217 assays)",
            "ace2_dev": "0.840 vs 0.747 Spearman",
        },
        "website": "https://autoscientists.openscientist.ai",
        "repo": "https://github.com/mims-harvard/AutoScientists",
    }


def evaluation_demo() -> dict[str, Any]:
    gpt = demo_gpt_speedup()
    teams = demo_parallel_teams()
    gate = demo_noise_gate_confirm()
    return {
        "gpt_speedup": gpt,
        "parallel_teams": teams,
        "noise_gate": gate,
        "all_ok": (
            gpt["speedup_factor"] >= 1.5
            and teams["ok"]
            and gate["clear_win"]
            and gate["borderline_confirmed"]
            and not gate["borderline_rejected"]
        ),
    }


def evaluation_smoke(cfg: AutoScientistsConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo()
    bench = benchmarks_bundle()
    return {
        "paper": "arXiv:2605.28655",
        "all_ok": demo["all_ok"],
        "bioml_delta_pp": bench["bioml_mean_leaderboard_pct"]["delta_pp"],
        "gpt_speedup": demo["gpt_speedup"]["speedup_factor"],
        "ok": demo["all_ok"] and bench["bioml_mean_leaderboard_pct"]["delta_pp"] > 7.0,
    }
