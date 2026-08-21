"""GPU architectural baselines (Table 2, arXiv:2606.06510)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GpuId = Literal["H100", "B200", "B300", "R200"]


@dataclass(frozen=True)
class GpuArch:
    """Per-GPU parameters for the TME model (dense rates unless noted)."""

    name: GpuId
    fp64_vector_tflops: float
    fp64_tensor_tflops: float | None
    fp8_tensor_tflops: float
    int8_tensor_tops: float
    hbm_tbs: float
    hbm_gib: tuple[int, int]
    emulated_dgemm_tflops: float | None = None
    fp4_tflops: float | None = None

    @property
    def fp64_native_tflops(self) -> float:
        """Peak fp64 for matrix kernels: prefer tensor path when listed."""
        if self.fp64_tensor_tflops is not None:
            return self.fp64_tensor_tflops
        return self.fp64_vector_tflops

    @property
    def memory_ridge_oi(self) -> float:
        """Native fp64 ridge point P_fp64 / B_mem (FLOPs/Byte)."""
        return self.fp64_native_tflops / self.hbm_tbs

    @property
    def fp8_fp64_ratio(self) -> float:
        return self.fp8_tensor_tflops / self.fp64_native_tflops


ARCHITECTURES: dict[GpuId, GpuArch] = {
    "H100": GpuArch(
        name="H100",
        fp64_vector_tflops=34.0,
        fp64_tensor_tflops=67.0,
        fp8_tensor_tflops=1979.0,
        int8_tensor_tops=1979.0,
        hbm_tbs=3.35,
        hbm_gib=(80, 80),
    ),
    "B200": GpuArch(
        name="B200",
        fp64_vector_tflops=40.0,
        fp64_tensor_tflops=40.0,
        fp8_tensor_tflops=4500.0,
        int8_tensor_tops=155.0,
        hbm_tbs=8.0,
        hbm_gib=(180, 192),
        fp4_tflops=7000.0,
    ),
    "B300": GpuArch(
        name="B300",
        fp64_vector_tflops=1.3,
        fp64_tensor_tflops=1.2,
        fp8_tensor_tflops=5000.0,
        int8_tensor_tops=165.0,
        hbm_tbs=8.0,
        hbm_gib=(279, 288),
        fp4_tflops=10000.0,
    ),
    "R200": GpuArch(
        name="R200",
        fp64_vector_tflops=33.0,
        fp64_tensor_tflops=None,
        fp8_tensor_tflops=4000.0,
        int8_tensor_tops=250.0,
        hbm_tbs=22.0,
        hbm_gib=(288, 288),
        emulated_dgemm_tflops=200.0,
        fp4_tflops=35000.0,
    ),
}


def substrate_throughput(arch: GpuArch, substrate: str) -> float:
    if substrate == "int8":
        return arch.int8_tensor_tops
    return arch.fp8_tensor_tflops


def table_2_architectures() -> list[dict[str, float | str | None]]:
    """Paper Table 2 export."""
    rows: list[dict[str, float | str | None]] = []
    for gpu in ("H100", "B200", "B300", "R200"):
        a = ARCHITECTURES[gpu]  # type: ignore[index]
        rows.append(
            {
                "gpu": gpu,
                "fp64_vector_tflops": a.fp64_vector_tflops,
                "fp64_tensor_tflops": a.fp64_tensor_tflops,
                "fp8_tensor_tflops": a.fp8_tensor_tflops,
                "int8_tensor_tops": a.int8_tensor_tops,
                "hbm_tbs": a.hbm_tbs,
                "hbm_gib_min": a.hbm_gib[0],
                "hbm_gib_max": a.hbm_gib[1],
                "emulated_dgemm_tflops": a.emulated_dgemm_tflops,
                "fp8_fp64_ratio": round(a.fp8_fp64_ratio, 1),
                "memory_ridge_oi": round(a.memory_ridge_oi, 2),
            }
        )
    return rows
