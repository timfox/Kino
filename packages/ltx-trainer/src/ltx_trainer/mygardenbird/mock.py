"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.mygardenbird.config import MygardenbirdConfig
from ltx_trainer.mygardenbird.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table7_cnn_accuracy,
)
from ltx_trainer.mygardenbird.species import verify_balance


def evaluation_smoke(cfg: MygardenbirdConfig | None = None) -> dict[str, Any]:
    c = cfg or MygardenbirdConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    bal = verify_balance(c)
    assert bal["balanced_600_per_class"]
    assert bal["clips_16k_total"] == c.clips_16k
    assert bal["xc_files_total"] == c.source_recordings
    assert demo["split"]["no_source_leakage"]

    eff = next(r for r in table7_cnn_accuracy() if r["model"] == "EfficientNet-B0")
    bird = next(r for r in table7_cnn_accuracy() if "BirdNET" in r["model"])
    assert eff["acc_16k"] == c.efficientnet_acc_16k
    assert bird["acc_16k"] == c.birdnet_acc_16k
    assert eff["acc_16k"] > 92.0

    b = benchmarks_bundle()
    assert len(b["table2_species_composition"]) == 12
    assert b["table3_mip_splits"][-1]["total"] == 7200

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "clips_16k": c.clips_16k,
        "birdnet_acc_16k": c.birdnet_acc_16k,
        "efficientnet_acc_16k": c.efficientnet_acc_16k,
    }
