"""Runtime configuration for TME / Ozaki II projections."""

from __future__ import annotations

from dataclasses import dataclass, field

from ltx_trainer.fp8_ozaki.constants import DEFAULT_BANDWIDTH_MULTIPLIER, DEFAULT_MODULI_COUNT


@dataclass
class Fp8OzakiConfig:
    """TME model knobs for a single projection run."""

    gpu: str = "B300"
    moduli_count: int = DEFAULT_MODULI_COUNT
    bandwidth_multiplier: float = DEFAULT_BANDWIDTH_MULTIPLIER
    substrate: str = "fp8"  # fp8 | int8
    workloads: tuple[str, ...] = (
        "dense_gemm",
        "batched_gemv_b8",
        "batched_gemv_b2",
        "stencil_7pt",
        "spmv",
    )
    roofline_oi_samples: tuple[float, ...] = field(
        default_factory=lambda: (0.1, 0.2, 0.5, 1.0, 1.5, 4.0, 10.0, 50.0, 100.0)
    )
