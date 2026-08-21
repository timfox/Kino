"""Ozaki Scheme II: scaling, modular residues, and cost model."""

from __future__ import annotations

import math
from typing import Any

from ltx_trainer.fp8_ozaki.garner import garner_reconstruct


# Pairwise-coprime moduli suitable for int9/fp8 residue planes (illustrative set).
DEFAULT_MODULI: tuple[int, ...] = (
    251,
    241,
    239,
    233,
    229,
    227,
    223,
    211,
    199,
    197,
)


def integer_scale_matrix(values: list[float], scale: float = 1.0) -> list[int]:
    """Phase 1: ⌊scale · x⌉ to nearest integer (fp64 operands)."""
    return [int(round(v * scale)) for v in values]


def residue_plane(value: int, modulus: int) -> int:
    """Residue for modular GEMM plane."""
    return value % modulus


def product_modulus_bound(moduli: list[int]) -> int:
    """M = ∏ m_i from Eq. (5)."""
    m = 1
    for mi in moduli:
        m *= mi
    return m


def required_moduli_count(max_product: int, moduli: tuple[int, ...] = DEFAULT_MODULI) -> int:
    """Minimum r such that ∏ m_i > 2 · max_product."""
    prod = 1
    for i, mi in enumerate(moduli, start=1):
        prod *= mi
        if prod > 2 * max_product:
            return i
    return len(moduli)


def fp8_mma_count_per_residue(moduli_count: int, *, karatsuba_factor: int = 3) -> int:
    """(3r + 1) FP8 MMAs per fp64-equivalent product (§2.4 Imamura refinement)."""
    return karatsuba_factor * moduli_count + 1


def ozaki_ii_cost_card(
    *,
    moduli_count: int = 10,
    substrate: str = "fp8",
    inner_dim: int = 4096,
) -> dict[str, Any]:
    """Algorithmic cost summary for Ozaki II."""
    moduli = list(DEFAULT_MODULI[:moduli_count])
    m_bound = product_modulus_bound(moduli)
    if substrate == "fp8":
        mmas_per_fp64 = fp8_mma_count_per_residue(moduli_count)
        scaling = "linear in r; fp8 quantisation trick [37]"
    else:
        mmas_per_fp64 = moduli_count + 1
        scaling = "linear in r; int8 modular GEMM [31]"
    return {
        "scheme": "ozaki_ii",
        "substrate": substrate,
        "moduli_count": moduli_count,
        "moduli": moduli,
        "product_bound": m_bound,
        "low_precision_gemms_per_fp64_op": moduli_count,
        "fp8_mmas_per_fp64_op": mmas_per_fp64 if substrate == "fp8" else None,
        "reconstruction": "garner_crt",
        "inner_dim": inner_dim,
        "error_bound": "componentwise ≤ u_fp64 + integer rounding [32]",
        "phases": ["integer_scaling", "modular_gemms", "crt_reconstruction"],
    }


def demo_dot_product_fp64(
    a: list[float],
    b: list[float],
    *,
    moduli_count: int = 4,
    scale: float = 1e6,
) -> dict[str, Any]:
    """Tiny CPU reference: Ozaki-II style modular dot product."""
    moduli = list(DEFAULT_MODULI[:moduli_count])
    ai = integer_scale_matrix(a, scale)
    bi = integer_scale_matrix(b, scale)
    residues: list[int] = []
    for mi in moduli:
        acc = 0
        for av, bv in zip(ai, bi, strict=True):
            acc = (acc + (av % mi) * (bv % mi)) % mi
        residues.append(acc)
    exact_int = sum(av * bv for av, bv in zip(ai, bi, strict=True))
    recon = garner_reconstruct(residues, moduli)
    fp64_native = sum(x * y for x, y in zip(a, b, strict=True))
    fp64_emu = recon / (scale * scale)
    rel_err = abs(fp64_emu - fp64_native) / max(abs(fp64_native), 1e-30)
    return {
        "fp64_native": fp64_native,
        "fp64_emulated": fp64_emu,
        "exact_integer": exact_int,
        "crt_reconstructed": recon,
        "relative_error": rel_err,
        "moduli_count": moduli_count,
        "within_fp64_ulp": rel_err < 2**-52 * 10,
    }
