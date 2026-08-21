"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.vocodec.config import VoCodecConfig
from ltx_trainer.vocodec.pipeline import (
    benchmarks_bundle,
    evaluation_demo,
    table1_libritts,
    table3_voicing_analysis,
)
from ltx_trainer.vocodec.quantizer import bitrate_kbps_simplified


def evaluation_smoke(cfg: VoCodecConfig | None = None) -> dict[str, Any]:
    c = cfg or VoCodecConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    vocodec = next(r for r in table1_libritts() if r["codec"] == "VoCodec")
    assert vocodec["lsd"] == c.t1_lsd
    assert vocodec["mushra"] == c.t1_mushra

    kbps = bitrate_kbps_simplified(
        fs=c.libritts_sample_rate_hz,
        voiced_ratio=c.libritts_voiced_ratio,
        cfg=c,
    )
    assert abs(kbps - c.libritts_target_kbps) < 0.05

    t3 = next(r for r in table3_voicing_analysis() if r["variant"] == "VoCodec")
    reversed_row = next(r for r in table3_voicing_analysis() if r["variant"] == "VoCodec-r")
    assert t3["lsd_v"] < reversed_row["lsd_v"]
    assert t3["stoi"] > reversed_row["stoi"]

    assert demo["beats_streamcodec_stoi"]
    assert demo["libritts_bitrate_kbps"] == c.libritts_target_kbps
    assert len(benchmarks_bundle()["table2_vctk"]) == 6

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "libritts_kbps": c.libritts_target_kbps,
        "libritts_mushra": c.t1_mushra,
        "bitrate_saving_pct": c.bitrate_saving_pct,
    }
