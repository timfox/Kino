"""Tensor–Memory Equilibrium (TME) performance model (§4)."""

from __future__ import annotations

from dataclasses import dataclass

from ltx_trainer.fp8_ozaki.arch import ARCHITECTURES, GpuArch, GpuId, substrate_throughput


@dataclass(frozen=True)
class EmulationParams:
    """Ozaki II emulation parameters α, β, γ."""

    alpha: int  # low-precision MMAs per fp64 op ≈ moduli count r
    beta: float  # bandwidth multiplier (1 = register-fused)
    gamma_per_output: float = 0.0  # Garner reconstruction; amortised to 0 for k ≫ r²


def native_throughput_tflops(oi: float, arch: GpuArch) -> float:
    """Achieved native fp64 throughput at operational intensity OI (FLOPs/Byte)."""
    mem_roof = oi * arch.hbm_tbs
    return min(arch.fp64_native_tflops, mem_roof)


def emulated_throughput_tflops(
    oi: float,
    arch: GpuArch,
    params: EmulationParams,
    substrate: str = "fp8",
) -> float:
    """Achieved Ozaki II fp64-equivalent throughput."""
    p_low = substrate_throughput(arch, substrate)
    compute_ceiling = p_low / params.alpha
    mem_roof = (oi * arch.hbm_tbs) / params.beta
    return min(compute_ceiling, mem_roof)


def ozaki_speedup(
    oi: float,
    gpu: GpuId,
    *,
    moduli_count: int = 10,
    beta: float = 1.0,
    substrate: str = "fp8",
) -> float:
    arch = ARCHITECTURES[gpu]
    params = EmulationParams(alpha=moduli_count, beta=beta)
    nat = native_throughput_tflops(oi, arch)
    emu = emulated_throughput_tflops(oi, arch, params, substrate)
    if nat <= 0:
        return float("inf")
    return emu / nat


def crossover_intensity(
    arch: GpuArch,
    params: EmulationParams,
    substrate: str = "fp8",
) -> float:
    """OI where emulated compute ceiling meets native fp64 compute roof."""
    p_low = substrate_throughput(arch, substrate)
    emu_compute_oi = (p_low / params.alpha) / arch.hbm_tbs * params.beta
    return emu_compute_oi


def roofline_curve(
    gpu: GpuId,
    oi_samples: tuple[float, ...],
    *,
    moduli_count: int = 10,
    beta: float = 1.0,
    substrate: str = "fp8",
) -> dict[str, list[float]]:
    """Native vs Ozaki II throughput samples for roofline plotting."""
    arch = ARCHITECTURES[gpu]
    params = EmulationParams(alpha=moduli_count, beta=beta)
    nat: list[float] = []
    emu: list[float] = []
    mem: list[float] = []
    for oi in oi_samples:
        nat.append(round(native_throughput_tflops(oi, arch), 3))
        emu.append(round(emulated_throughput_tflops(oi, arch, params, substrate), 3))
        mem.append(round(oi * arch.hbm_tbs, 3))
    return {
        "oi": list(oi_samples),
        "native_fp64_tflops": nat,
        "ozaki_fp8_tflops": emu,
        "memory_roof_tflops": mem,
        "native_compute_ceiling": arch.fp64_native_tflops,
        "ozaki_compute_ceiling": round(substrate_throughput(arch, substrate) / moduli_count, 1),
        "ridge_oi_native": round(arch.memory_ridge_oi, 3),
    }


def ozaki_i_slice_count(
    substrate: str,
    k: int,
    *,
    wacc: int | None = None,
    input_mantissa_bits: int | None = None,
) -> int:
    """Accumulator-bound slice count from Eq. (3), Table 1."""
    defaults = {
        "fp16": (24, 11),
        "int8": (31, 7),
        "fp8": (24, 4),
    }
    if substrate not in defaults:
        raise ValueError(f"unknown substrate: {substrate}")
    w, b_in = defaults[substrate] if wacc is None else (wacc, input_mantissa_bits or defaults[substrate][1])
    logk = (k - 1).bit_length() if k > 1 else 0
    b_star = (w - logk) / 2.0
    if b_star <= 0:
        return 53
    if b_star >= b_in:
        return 1
    import math

    return max(1, math.ceil(53 / b_star))
