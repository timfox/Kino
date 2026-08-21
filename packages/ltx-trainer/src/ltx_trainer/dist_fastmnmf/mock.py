"""Distributed FastMNMF smoke evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.dist_fastmnmf.block_scm import block_diag_scm, per_subarray_joint_diagonalizable
from ltx_trainer.dist_fastmnmf.complexity import asymptotic_speedup_equal_subarrays
from ltx_trainer.dist_fastmnmf.config import DistFastmnmfConfig, TimingResults


def evaluation_smoke(cfg: DistFastmnmfConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DistFastmnmfConfig()
    rng = np.random.default_rng(0)
    blocks = [rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)) for _ in range(3)]
    scm = block_diag_scm(blocks)
    blocks_per_source = [[b for b in blocks] for _ in range(2)]
    jd = per_subarray_joint_diagonalizable(blocks_per_source)
    speed = asymptotic_speedup_equal_subarrays(l_subarrays=3)
    timing = TimingResults()
    speedup_x = round(timing.fastmnmf_all_subarrays_s / timing.distributed_fastmnmf_s, 2)
    return {
        "block_scm_shape": list(scm.shape),
        "joint_diagonalizable_toy": jd,
        "inversion_speedup_factor": speed["inversion_factor"],
        "paper_mean_sdr_3src_distributed_db": 13.4,
        "speedup_vs_all_subarrays_x": speedup_x,
    }
