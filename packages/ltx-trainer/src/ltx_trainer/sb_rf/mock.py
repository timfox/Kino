"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.sb_rf.config import SbRfConfig
from ltx_trainer.sb_rf.flow import velocity_matching_loss, velocity_target
from ltx_trainer.sb_rf.pipeline import benchmarks_bundle, evaluation_demo, table1_vbdmd, table2_low_snr


def evaluation_smoke(cfg: SbRfConfig | None = None) -> dict[str, Any]:
    c = cfg or SbRfConfig()
    demo = evaluation_demo(seed=0, cfg=c)
    sb_rf_a = next(r for r in table1_vbdmd(c) if r["method"] == "SB-RF")
    cose = next(r for r in table1_vbdmd(c) if r["method"] == "COSE")
    sb_rf_b = next(r for r in table2_low_snr(c) if r["method"] == "SB-RF" and r["nfe"] == 1)
    bb_rf_b = next(r for r in table2_low_snr(c) if r["method"] == "BB-RF")

    assert c.sb_rf_track_a_pesq == 3.39
    assert c.sb_rf_track_a_si_sdr == 19.5
    assert c.sb_rf_track_b_pesq == 2.56
    assert c.nfe_default == 1
    assert sb_rf_a["pesq"] > cose["pesq"]
    assert abs(sb_rf_a["pesq"] - cose["pesq"] - c.track_a_pesq_gain_vs_cose) < 1e-6
    assert abs(sb_rf_b["pesq"] - bb_rf_b["pesq"] - c.track_b_pesq_gain_vs_bb_rf) < 1e-6
    assert demo["sb_rf_beats_bb_rf_track_a"] is True
    assert demo["velocity_loss"] < 0.01
    assert len(benchmarks_bundle(c)["track_a_vbdmd"]) == 9

    x = np.ones((4, 4), dtype=np.complex128)
    y = np.ones((4, 4), dtype=np.complex128) * 2
    assert float(velocity_matching_loss(velocity_target(x, y), x, y)) == 0.0

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "track_a_pesq": c.sb_rf_track_a_pesq,
        "track_a_si_sdr": c.sb_rf_track_a_si_sdr,
        "track_b_pesq": c.sb_rf_track_b_pesq,
        "track_b_estoi": c.sb_rf_track_b_estoi,
        "nfe": c.nfe_default,
    }
