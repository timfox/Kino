"""End-to-end FSC-Net pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.fsc_net.config import FscNetConfig
from ltx_trainer.fsc_net.pipeline import evaluation_demo, table1_vctk, table3_ablation


def pipeline_demo(*, seed: int = 0, cfg: FscNetConfig | None = None) -> dict[str, Any]:
    cfg = cfg or FscNetConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    fsc_4k = next(r for r in table1_vctk() if r["model"] == "FSC-Net" and r["scenario"] == "4_khz_to_48_khz")
    ablation_c = next(r for r in table3_ablation() if "Progressive" in r["model"])
    baseline = next(r for r in table3_ablation() if "Baseline" in r["model"])
    return {
        "demo": demo,
        "pesq_gain_vs_baseline_ablation": round(ablation_c["pesq"] - baseline["pesq"], 4),
        "lsd_4k": fsc_4k["lsd"],
        "params_m": cfg.params_m,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["lsd_4k"] == 0.8771
    assert out["params_m"] == 1.54
    assert out["pesq_gain_vs_baseline_ablation"] == 0.2873
    return {"status": "ok", "lsd_4k": out["lsd_4k"], "params_m": out["params_m"]}
