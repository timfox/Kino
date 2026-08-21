"""LiveBrowseComp framework and evaluation (arXiv:2605.28721)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.livebrowsecomp.agent import scaffold_spec
from ltx_trainer.livebrowsecomp.benchmarks import (
    CATEGORY_DISTRIBUTION,
    CORRELATION,
    EXAMPLE_QUESTIONS,
    TABLE3_MODEL_SCORES,
    TABLE5_PER_DOMAIN,
)
from ltx_trainer.livebrowsecomp.config import LiveBrowseCompConfig
from ltx_trainer.livebrowsecomp.eval import (
    evaluate_predictions,
    iter_builtin_smoke_predictions,
)
from ltx_trainer.livebrowsecomp.examples import demo_filter_examples, demo_ikd_trajectory
from ltx_trainer.livebrowsecomp.ikd_diagnostics import (
    compute_ikd_from_trajectories,
    ikd_summary,
)
from ltx_trainer.livebrowsecomp.io import checkout_status, dataset_stats, load_items
from ltx_trainer.livebrowsecomp.trajectory import analyze_trajectory


def framework_card(cfg: LiveBrowseCompConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LiveBrowseCompConfig()
    ikd = ikd_summary()
    return {
        "name": "LiveBrowseComp",
        "paper": f"arXiv:{cfg.arxiv}",
        "hub_dataset": cfg.hub_dataset,
        "num_questions": cfg.num_questions,
        "recency_window_days": cfg.recency_days,
        "ikd": {
            "definition": "Intrinsic Knowledge Dependence — agents guess from parametric memory and use search to verify",
            "closed_book_avg_pass4_static": ikd.closed_book_avg,
            "evidence_blocked_collapse": ikd.blocked_worse_than_closed,
            "model_originated_queries": ikd.model_originated_query_fraction,
        },
        "construction": {
            "sources": list(cfg.seed_sources),
            "filters": ["90-day recency", "long-tail obscurity", "answer stability"],
            "verification": ["correctness+uniqueness", "30min difficulty", "temporality"],
        },
        "scaffold": scaffold_spec(),
        "human_calibration": {
            "browsecomp_solve_rate": cfg.human_solve_rate_browsecomp,
            "livebrowsecomp_solve_rate": cfg.human_solve_rate_live,
        },
        "checkout": checkout_status(),
    }


def knowledge_card() -> dict[str, Any]:
    return {
        "arxiv": "2605.28721",
        "authors": "Fan, Wang, Chu et al. (HIT, Xiaohongshu)",
        "problem": "Static search benchmarks reward memory-backed verification, not discovery",
        "diagnostics": [
            "Q1 closed-book — how much is answerable without tools",
            "Q2 evidence-blocked — search without gold/evidence docs",
            "Q3 trajectory — model- vs retrieval-originated queries",
        ],
        "livebrowsecomp_design": [
            "335 human-authored questions (encrypted on Hub)",
            "Facts from last 90 days, long-tail, stable answers",
            "Closed-book ≤2% on LiveBrowseComp vs up to 44.5% on BrowseComp",
        ],
        "hub": "https://huggingface.co/datasets/Forival/LiveBrowseComp",
        "env": {"GOPEX_LIVEBROWSECOMP_CACHE": "local JSONL cache directory"},
        "related": ["BrowseComp", "BrowseComp-Plus", "LiveBench", "FreshQA"],
    }


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table3_scores": TABLE3_MODEL_SCORES,
        "category_distribution": CATEGORY_DISTRIBUTION,
        "table5_per_domain": TABLE5_PER_DOMAIN,
        "correlation": CORRELATION,
        "example_questions": EXAMPLE_QUESTIONS,
        "paper_trajectory_stats": __import__(
            "ltx_trainer.livebrowsecomp.ikd_diagnostics", fromlist=["TRAJECTORY_STATS"]
        ).TRAJECTORY_STATS,
    }


def score_drop_analysis() -> list[dict[str, Any]]:
    rows = []
    for r in TABLE3_MODEL_SCORES:
        rows.append(
            {
                "model": r["model"],
                "browsecomp": r["BrowseComp"],
                "live": r["LiveBrowseComp"],
                "drop_points": round(r["BrowseComp"] - r["LiveBrowseComp"], 1),
            }
        )
    return sorted(rows, key=lambda x: -x["drop_points"])


def evaluation_demo(seed: int = 42, *, load_limit: int = 5, download: bool = True) -> dict[str, Any]:
    _ = seed
    traj = demo_ikd_trajectory()
    traj_report = analyze_trajectory(traj)
    ikd_live = compute_ikd_from_trajectories([traj])

    eval_block: dict[str, Any] = {"dataset_loaded": False}
    try:
        items = load_items(download=download, limit=load_limit)
        preds = iter_builtin_smoke_predictions(items)
        report = evaluate_predictions(items, preds)
        eval_block = {
            "dataset_loaded": True,
            "dataset_stats": dataset_stats(items),
            "smoke_eval": report.to_dict(),
            "sample_idx": items[0].idx if items else None,
            "sample_answer_len": len(items[0].answer) if items else 0,
        }
    except (FileNotFoundError, OSError, ImportError) as e:
        eval_block["load_error"] = str(e)

    return {
        "seed": seed,
        "ikd_trajectory_demo": traj_report.to_dict(),
        "ikd_computed": ikd_live,
        "seed_filters_demo": demo_filter_examples(),
        "live_eval": eval_block,
        "ranking_shift_example": {
            "glm51_browsecomp": 68.0,
            "glm51_live": 33.9,
            "dsv32_browsecomp": 51.4,
            "dsv32_live": 37.6,
        },
        "score_drops_top3": score_drop_analysis()[:3],
        "correlation": CORRELATION,
    }


def evaluation_smoke(cfg: LiveBrowseCompConfig | None = None) -> dict[str, Any]:
    _ = cfg
    demo = evaluation_demo(seed=0, load_limit=3, download=True)
    traj = demo["ikd_trajectory_demo"]
    live = demo.get("live_eval", {})
    pass_ok = live.get("smoke_eval", {}).get("pass_at_4", 0) >= 20.0 if live.get("dataset_loaded") else True
    return {
        "paper": "arXiv:2605.28721",
        "num_questions": 335,
        "ikd_model_originated_rate": traj.get("model_originated_rate", 0) > 0.5,
        "dataset_loaded": live.get("dataset_loaded", False),
        "smoke_pass_at_4": live.get("smoke_eval", {}).get("pass_at_4"),
        "smoke_eval_ok": pass_ok,
    }
