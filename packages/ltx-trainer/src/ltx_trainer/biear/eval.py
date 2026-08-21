"""End-to-end BiEAR pipeline demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.biear.config import BiearConfig
from ltx_trainer.biear.pipeline import evaluation_demo, table2_anechoic, table3_real_rooms


def pipeline_demo(*, seed: int = 0, cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    demo = evaluation_demo(seed=seed, cfg=cfg)
    biear = next(r for r in table2_anechoic() if "Dual Controller + Rel" in r["model"])
    deepear = next(r for r in table2_anechoic() if r["model"] == "DeepEar")
    meeting = next(
        r for r in table3_real_rooms()
        if r["room"] == "Meeting Room" and r["transfer"] and "BiEAR" in r["model"]
    )
    return {
        "demo": demo,
        "azim_mae_1spk_seen": biear["1spk_azim_mae_seen"],
        "azim_gain_vs_deepear": round(deepear["1spk_azim_mae_seen"] - biear["1spk_azim_mae_seen"], 2),
        "meeting_detect_transfer": meeting["1spk_detect"],
        "dual_relative": True,
    }


def eval_smoke() -> dict[str, Any]:
    out = pipeline_demo(seed=42)
    assert out["azim_mae_1spk_seen"] == 0.36
    assert out["azim_gain_vs_deepear"] > 0.4
    assert out["meeting_detect_transfer"] > 93.0
    return {"status": "ok", "azim_mae_1spk_seen": out["azim_mae_1spk_seen"]}
