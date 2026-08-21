"""Role-Agent public API and smoke entrypoints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.role_agent.benchmarks import benchmarks_bundle
from ltx_trainer.role_agent.config import RoleAgentConfig
from ltx_trainer.role_agent.examples import demo_aiw_failure_curriculum, demo_dual_role_loop, demo_wia_toy_rollout
from ltx_trainer.role_agent.failure_modes import FAILURE_MODE_COUNTS
from ltx_trainer.role_agent.training import run_toy_training


def framework_card(cfg: RoleAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RoleAgentConfig()
    return {
        "name": "Role-Agent",
        "paper": f"arXiv:{cfg.arxiv}",
        "github": cfg.github,
        "institution": cfg.institution,
        "design": "Single LLM dual-role bootstrapped co-evolution",
        "modules": {
            "WIA": "World-In-Agent — predictive state rewards + GiGPO-style advantages",
            "AIW": "Agent-In-World — failure mode memory + curriculum resampling",
        },
        "benchmarks": ["ALFWorld", "WebShop", "search-augmented QA (NQ, HotpotQA, …)"],
        "backbone_default": cfg.backbone_default,
        "hyperparameters": {
            "group_size": cfg.group_size,
            "state_similarity_threshold": cfg.state_similarity_threshold,
            "advantage_alpha": cfg.advantage_alpha,
            "prediction_horizon_frac": cfg.prediction_horizon_frac,
            "t_max": cfg.t_max,
        },
        "failure_mode_counts": FAILURE_MODE_COUNTS,
    }


def knowledge_card() -> dict[str, Any]:
    bench = benchmarks_bundle()
    return {
        "arxiv": "2606.10917",
        "authors": "Wang, Ma, Yang, Huang, Wang, Wang, Chu (USTC + AMAP)",
        "problem": "Sparse static env feedback limits agentic RL generalization",
        "mechanisms": [
            "WIA: LMS predictive reward modulates task return multiplicatively",
            "GiGPO-style state grouping + mixed A^S + α·A^E advantages",
            "AIW: LLM failure reflection → retrieve similar tasks → reshape p_D",
        ],
        "results": {
            "alfworld_qwen15b": bench["table1_qwen15b"]["Role-Agent"]["alfworld_all"],
            "webshop_qwen15b": bench["table1_qwen15b"]["Role-Agent"]["webshop_succ"],
            "search_qa_avg_qwen3b": bench["table2_search_qa_qwen3b"]["Role-Agent"]["avg"],
            "vs_gigpo_alfworld_pp": bench["role_agent_vs_gigpo_alfworld_pp"],
            "vs_gigpo_webshop_pp": bench["role_agent_vs_gigpo_webshop_pp"],
        },
        "github": "https://github.com/AMAP-ML/roleagent",
    }


def evaluation_demo(cfg: RoleAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RoleAgentConfig()
    dual = demo_dual_role_loop()
    toy_train = run_toy_training(iterations=12, seed=7, group_size=2)
    return {
        "wia_toy": demo_wia_toy_rollout(),
        "aiw_toy": demo_aiw_failure_curriculum(),
        "dual_role": dual,
        "toy_training": toy_train,
        "prediction_horizon_alfworld": cfg.prediction_horizon("alfworld"),
        "paper_tables": benchmarks_bundle(),
        "all_ok": dual["ok"] and toy_train["ok"],
    }


def evaluation_smoke(cfg: RoleAgentConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RoleAgentConfig()
    demo = evaluation_demo(cfg)
    bench = benchmarks_bundle()
    t1 = bench["table1_qwen15b"]
    t3 = bench["table3_ablation"]
    ok = (
        demo["all_ok"]
        and t1["Role-Agent"]["alfworld_all"] == 90.9
        and t1["Role-Agent"]["webshop_succ"] == 71.9
        and t1["Role-Agent"]["alfworld_all"] > t1["GiGPO"]["alfworld_all"]
        and t3["w/o AIW"]["webshop"] < t3["Role-Agent"]["webshop"]
        and bench["ablation_both_beat_gigpo"]
        and cfg.prediction_horizon("alfworld") == 3  # 5% of 50 rounded
    )
    return {
        "package": "ltx_trainer.role_agent",
        "status": "smoke_ok" if ok else "smoke_fail",
        "paper": cfg.title,
        "arxiv": cfg.arxiv,
        "ok": ok,
        "alfworld_all": t1["Role-Agent"]["alfworld_all"],
        "webshop_succ": t1["Role-Agent"]["webshop_succ"],
        "gigpo_delta_alfworld_pp": bench["role_agent_vs_gigpo_alfworld_pp"],
        "aiw_ablation_drop_webshop_pp": bench["ablation_aiw_drop_webshop_pp"],
    }
