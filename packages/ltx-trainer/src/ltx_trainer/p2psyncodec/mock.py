"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.p2psyncodec.config import P2PSynCodecConfig
from ltx_trainer.p2psyncodec.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_objective,
    table2_pseudo_vq_ablation,
)
from ltx_trainer.p2psyncodec.quantizer import bitrate_kbps


def evaluation_smoke(cfg: P2PSynCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or P2PSynCodecConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    p2p = next(r for r in table1_objective() if r["codec"] == "P2PSynCodec")
    assert p2p["libritts_utmos"] == c.t1_utmos

    kbps = bitrate_kbps(fs=c.libritts_sample_rate_hz, cfg=c)
    assert abs(kbps - c.libritts_target_kbps) < 0.01

    t2_n3 = next(r for r in table2_pseudo_vq_ablation() if r["n_pseudo"] == 3)
    assert t2_n3["all_utmos"] == c.t2_n3_all_utmos
    assert demo["optimal_n_pseudo"] == c.pseudo_vq_count
    assert demo["beats_mdctcodec_utmos"]
    assert len(benchmarks_bundle()["fig3_abx_high_bitrate"]) == 5

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "libritts_kbps": c.libritts_target_kbps,
        "libritts_utmos": c.t1_utmos,
        "bitrate_saving_pct": c.bitrate_saving_pct,
        "pseudo_vq_count": c.pseudo_vq_count,
    }
