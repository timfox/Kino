"""CPU evaluation smoke."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dbhn_net.config import DbhnNetConfig
from ltx_trainer.dbhn_net.pipeline import benchmarks_bundle, evaluation_demo, table3_mamba_ablation


def evaluation_smoke(cfg: DbhnNetConfig | None = None) -> dict[str, Any]:
    c = cfg or DbhnNetConfig()
    demo = evaluation_demo(seed=0, cfg=c)

    mamba_row = next(r for r in table3_mamba_ablation(c) if r["model"] == "DBHN-Net (OURS)")
    assert mamba_row["macs"] == c.macs_dbhn
    assert mamba_row["pesq"] == c.full_pesq

    assert c.full_pesq > c.wo_ann_pesq
    assert c.full_pesq > c.wo_snn_pesq
    assert c.macs_dbhn < c.macs_lstm
    assert c.macs_dbhn < c.macs_transformer
    assert c.macs_dbhn < c.macs_bsdb
    assert demo["loss"]["total"] >= 0.0
    assert len(benchmarks_bundle(c)["table2_dual_branch"]) == 3

    return {
        "status": "ok",
        "paper": c.paper_arxiv,
        "wsj0_pesq": c.wsj0_pesq_avg,
        "dns_wb_pesq": c.dns_wb_pesq,
        "dns_sisdr": c.dns_sisdr,
        "macs_gps": c.macs_dbhn,
        "complexity_reduction_x": c.complexity_reduction_x,
        "full_pesq": c.full_pesq,
    }
