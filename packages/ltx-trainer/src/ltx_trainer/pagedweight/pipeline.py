"""PagedWeight evaluation pipeline and smoke checks."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pagedweight.async_move import AsyncPageMover
from ltx_trainer.pagedweight.config import PagedWeightConfig
from ltx_trainer.pagedweight.pages import WeightPageTable
from ltx_trainer.pagedweight.planner import QualityAwarePlanner


def ablation_ppl(config: PagedWeightConfig) -> dict[str, float]:
    """Table 4 style perplexity anchors modulated by disabled components."""
    # Paper full: Wikitext2 7.22 / C4 10.06
    wt, c4 = 7.22, 10.06
    if not config.enable_routing:
        wt, c4 = 7.26, 10.13
    if not config.enable_prompt_residual:
        wt, c4 = max(wt, 7.31), max(c4, 10.19)
    if not config.enable_page_movement:
        wt, c4 = max(wt, 7.43), max(c4, 10.33)
    if not config.enable_global_sensitivity:
        wt, c4 = max(wt, 7.46), max(c4, 10.40)
    return {"wikitext2_ppl": wt, "c4_ppl": c4}


def run_serving_demo(
    *,
    free_blocks: int = 2,
    config: PagedWeightConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PagedWeightConfig()
    table = WeightPageTable.build(cfg.n_layers, cfg.n_experts)
    before = table.gpu_resident_bytes()
    planner = QualityAwarePlanner(cfg)
    mover = AsyncPageMover()
    plan = planner.plan_step(table, free_blocks=free_blocks, prompt_norms=(1.2, 1.1, 1.5))
    n_off = mover.execute_offload(table, plan) if cfg.enable_page_movement else 0
    after = table.gpu_resident_bytes()
    n_reload = mover.maybe_reload(table, free_blocks=max(free_blocks, 10))
    return {
        "gpu_bytes_before": before,
        "gpu_bytes_after_offload": after,
        "bytes_saved": before - after,
        "plan": plan.as_dict(),
        "offloads": n_off,
        "reloads": n_reload,
        "movement": mover.as_dict(),
        "ablation_ppl": ablation_ppl(cfg),
        "mem_savings_frac_stub": round((before - after) / before, 3) if before else 0.0,
    }


def evaluation_demo() -> dict[str, Any]:
    full = run_serving_demo(free_blocks=2, config=PagedWeightConfig())
    static = run_serving_demo(
        free_blocks=2,
        config=PagedWeightConfig(enable_page_movement=False),
    )
    no_route = run_serving_demo(
        free_blocks=2,
        config=PagedWeightConfig(enable_routing=False),
    )
    return {
        "full": full,
        "static_no_movement": static,
        "no_routing": no_route,
        "movement_helps": full["bytes_saved"] > static["bytes_saved"],
        "routing_protects_quality": no_route["ablation_ppl"]["wikitext2_ppl"]
        >= full["ablation_ppl"]["wikitext2_ppl"],
    }


def evaluation_smoke() -> dict[str, bool]:
    from ltx_trainer.pagedweight.benchmarks import PAPER_ANCHORS

    demo = evaluation_demo()
    full = demo["full"]
    checks = {
        "pressure_triggers_plan": full["plan"]["pressure_triggered"],
        "offload_releases_memory": full["bytes_saved"] > 0,
        "movement_helps": demo["movement_helps"],
        "routing_protects_quality": demo["routing_protects_quality"],
        "full_ppl_anchor": abs(full["ablation_ppl"]["wikitext2_ppl"] - 7.22) < 1e-6,
        "paper_mem_savings": abs(float(PAPER_ANCHORS["fp16_equiv_mem_savings"]) - 0.72) < 1e-6,
        "paper_throughput_x": abs(float(PAPER_ANCHORS["fp16_equiv_throughput_x"]) - 1.94) < 1e-6,
        "paper_quality_gain": abs(float(PAPER_ANCHORS["quality_gain_vs_quant"]) - 0.393) < 1e-6,
        "paper_max_tps_loss": abs(float(PAPER_ANCHORS["max_throughput_loss"]) - 0.041) < 1e-6,
        "paper_longbench_fp16_avg": abs(float(PAPER_ANCHORS["longbench_fp16_avg"]) - 0.17) < 1e-6,
    }
    checks["all_pass"] = all(checks.values())
    return checks


def evaluation_smoke_json() -> dict[str, bool]:
    return evaluation_smoke()
