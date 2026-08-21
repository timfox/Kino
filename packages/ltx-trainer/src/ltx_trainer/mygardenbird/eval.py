"""End-to-end MyGardenBird pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mygardenbird.config import MygardenbirdConfig
from ltx_trainer.mygardenbird.pipeline import evaluation_demo, headline_results, table7_cnn_accuracy


def pipeline_demo(*, seed: int = 0, cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    demo = evaluation_demo(seed=seed, cfg=c)
    eff = next(r for r in table7_cnn_accuracy() if r["model"] == "EfficientNet-B0")
    return {
        "demo": demo,
        "best_cnn_acc_16k": eff["acc_16k"],
        "birdnet_beats_cnn": c.birdnet_acc_16k > eff["acc_16k"],
        "headline": headline_results(c),
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["birdnet_beats_cnn"]
    assert out["best_cnn_acc_16k"] == 96.39
    return {"status": "ok", "efficientnet_acc_16k": out["best_cnn_acc_16k"]}
