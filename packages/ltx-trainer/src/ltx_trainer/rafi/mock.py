"""RaFI evaluation smoke for paper-stub validation."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke(cfg=None) -> dict[str, Any]:
    config_mod = load_sibling(__file__, "config")
    pipeline_mod = load_sibling(__file__, "pipeline")
    c = cfg or config_mod.RafiConfig()
    demo = pipeline_mod.evaluation_demo(c)
    return {
        "paper": c.paper_arxiv,
        "forward_total_rays": int(demo["forward_total_rays"]),
        "received_rank0": int(demo["received_rank0"]),
        "paper_intranode_gbps": float(demo["paper_intranode_gbps"]),
        "intranode_rays_per_sec": float(demo["throughput"]["intranode_rays_per_sec"]),
    }
