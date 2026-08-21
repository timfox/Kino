"""End-to-end PCCL demo."""

from __future__ import annotations

from typing import Any

from ltx_trainer.pccl.benchmarks import (
    process_group_speedup_anchors,
    scalability_anchors,
    table_i_synthesizer_comparison,
)
from ltx_trainer.pccl.config import PcclConfig
from ltx_trainer.pccl.process_group import overlay_process_groups, speedup_vs_direct
from ltx_trainer.pccl.synthesis import synthesize


def run_demo(cfg: PcclConfig | None = None) -> dict[str, Any]:
    cfg = cfg or PcclConfig()
    pg_demo = PcclConfig(
        n_npus=9,
        mesh_width=3,
        topology="2d_mesh",
        collective="all_gather",
        process_group=[1, 2, 3],
    )
    a2a_demo = PcclConfig(
        n_npus=16,
        mesh_width=4,
        topology="2d_mesh",
        collective="all_to_all",
        process_group=[1, 2, 3, 4],
    )
    return {
        "all_gather_pg": synthesize(pg_demo),
        "all_to_all_pg": synthesize(a2a_demo),
        "multi_group": overlay_process_groups(
            9,
            [
                {"npus": [1, 2, 3], "collective": "all_to_allv"},
                {"npus": [7, 8, 9], "collective": "all_gather"},
            ],
        ),
        "speedup": speedup_vs_direct(n_process_groups=2),
        "scalability": scalability_anchors(),
        "comparison": table_i_synthesizer_comparison(),
        "process_group_anchors": process_group_speedup_anchors(),
    }
