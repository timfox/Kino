"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.biear.config import BiearConfig
from ltx_trainer.biear.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table2_anechoic,
    table3_real_rooms,
)


def evaluation_smoke(cfg: BiearConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BiearConfig()
    demo = evaluation_demo(seed=0, cfg=cfg)
    fw = demo["framework"]
    assert fw["paper"] == "arXiv:2606.06795"
    assert fw["n_sectors"] == 8

    biear = next(r for r in table2_anechoic() if "Dual Controller + Rel" in r["model"])
    passive = next(r for r in table2_anechoic() if "w/o Controller" in r["model"])
    assert biear["1spk_azim_mae_seen"] < passive["1spk_azim_mae_seen"]
    assert biear["1spk_azim_mae_seen"] == 0.36

    meeting = next(
        r for r in table3_real_rooms()
        if r["room"] == "Meeting Room" and r["transfer"] and "BiEAR" in r["model"]
    )
    assert meeting["1spk_detect"] == 93.74

    b = benchmarks_bundle()
    assert len(b["table1_brir_datasets"]) == 3

    return {
        "status": "ok",
        "paper": fw["paper"],
        "azim_mae_1spk_seen": cfg.azim_mae_1spk_seen,
        "meeting_detect_transfer": cfg.meeting_detect_transfer,
    }
