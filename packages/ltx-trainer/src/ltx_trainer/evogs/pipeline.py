"""Framework card, evaluation demo, smoke (arXiv:2606.07179)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.evogs.benchmarks import benchmarks_bundle, ours_beats_baselines
from ltx_trainer.evogs.config import EvoGSConfig
from ltx_trainer.evogs.layout import LIMITATIONS
from ltx_trainer.evogs.simulation import full_pipeline_demo, train_step_torch


def framework_card(cfg: EvoGSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EvoGSConfig()
    p = cfg.params
    return {
        "name": "EvoGS",
        "paper": cfg.paper_arxiv,
        "title": cfg.paper_title,
        "task": "Continuous-layered progressive 3DGS for scalable streaming",
        "paradigm": "Evolution Tree with wavelet-inspired parent-child refinement (Option D)",
        "refinement": "C1 = P + ψ, C2 = P − α ⊙ ψ",
        "quality_levels": p.num_levels,
        "image_pyramid": list(p.downsample_factors),
        "training": {
            "L0_iters": p.base_iters,
            "Li_iters": p.refine_iters,
            "densify_every": p.densify_interval,
            "lambda_dssim": p.lambda_dssim,
        },
        "advantages": [
            "Ghost splats < 25% vs > 65% discrete layering",
            "Up to 2.4× lower transmission payload",
            "Up to 5.5× lower GPU leaf footprint",
            "Smooth progressive quality (Fig. 10–11)",
            "Compressible ψ residuals (Table 5)",
        ],
        "datasets": list(cfg.datasets),
        "baselines": list(cfg.baselines),
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return list(LIMITATIONS)


def evaluation_demo(cfg: EvoGSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EvoGSConfig()
    demo = full_pipeline_demo(cfg, seed=0)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "benchmarks": benchmarks_bundle(),
        "beats": ours_beats_baselines(),
        "pipeline_demo": demo,
        "train_step": train_step_torch(0),
    }


def evaluation_smoke(cfg: EvoGSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EvoGSConfig()
    demo = evaluation_demo(cfg)
    beats = demo["beats"]
    pipe = demo["pipeline_demo"]
    stream = pipe["streaming"]["quality_metrics"]

    assert beats["ghost_under_25pct"]
    assert beats["ghost_below_lapis"]
    assert beats["asym_beats_sym_L3"]
    assert beats["compressed_under_100mb"]
    assert beats["storage_beats_lapis"]
    assert pipe["compression"]["compression_ratio"] > 1.0
    assert stream["continuous_monotone"]
    assert stream["smoothness_gain"] > 1.0
    assert pipe["tree"]["ghost_ratio"] < 0.35

    ts = demo["train_step"]
    if ts.get("torch"):
        assert ts["loss"] >= 0.0

    return {
        "status": "ok",
        "paper": cfg.paper_arxiv,
        "ghost_ratio": pipe["tree"]["ghost_ratio"],
        "compression_ratio": pipe["compression"]["compression_ratio"],
        "smoothness_gain": stream["smoothness_gain"],
        "demo_keys": list(demo.keys()),
    }
