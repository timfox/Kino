"""Framework card and benchmarks for ITS-AVGen."""

from __future__ import annotations

from typing import Any

from ltx_trainer.its_avgen.config import ItsAvgenConfig
from ltx_trainer.its_avgen.search import best_of_n, evo_search_smoke


def framework_card(cfg: ItsAvgenConfig | None = None) -> dict[str, Any]:
    cfg = cfg or ItsAvgenConfig()
    return {
        "name": "ITS-AVGen",
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "project": cfg.project_url,
        "modes": ["Best-of-N", "EvoSearch"],
        "verifiers": ["VideoReward-TA (text alignment)", "JavisScore (AV sync)"],
        "mitigation": "ARW adaptive reweighting vs verifier hacking",
        "training_free": True,
    }


def table_main_results() -> list[dict[str, Any]]:
    return [
        {"method": "Single sample", "vbench_t2v": "baseline", "av_align": "baseline"},
        {"method": "Best-of-N (VR only)", "vbench_t2v": "+", "av_align": "− (hacking)"},
        {"method": "Best-of-N (VR+JS)", "vbench_t2v": "+", "av_align": "+"},
        {"method": "EvoSearch (VR+JS+ARW)", "vbench_t2v": "++", "av_align": "++"},
    ]


def benchmarks_bundle() -> dict[str, Any]:
    return {"main_results": table_main_results(), "recommended_n": 4}


def evaluation_demo(*, seed: int = 0, n: int = 4) -> dict[str, Any]:
    rng = __import__("numpy").random.default_rng(seed)
    candidates = [
        {
            "text_overlap": float(rng.uniform(0.3, 0.9)),
            "motion_stability": float(rng.uniform(0.3, 0.9)),
            "av_sync": float(rng.uniform(0.3, 0.9)),
            "fine_grained_match": float(rng.uniform(0.3, 0.9)),
            "vr_hack_risk": float(rng.uniform(0.0, 0.5)),
        }
        for _ in range(n)
    ]
    bon = best_of_n(candidates, seed=seed)
    evo = evo_search_smoke(seed=seed + 100)
    return {"paper": ItsAvgenConfig().paper_arxiv, "best_of_n": bon, "evo_search": evo}


def delivery_plan(*, n_candidates: int = 4) -> dict[str, Any]:
    """Env knobs for HQ delivery without retraining."""
    return {
        "env": {
            "GOPEX_ITS_ENABLED": "1",
            "GOPEX_ITS_N_CANDIDATES": str(n_candidates),
            "GOPEX_ITS_ARW": "1",
            "GOPEX_INFER_ADAMAG": "1",
        },
        "note": "Generate N clips per prompt; rank with VR+JS proxies; keep best.",
    }
