"""Paper table anchors — Tables 1–8, COMBVD STRESS."""

from __future__ import annotations

from typing import Any


def operating_guidelines() -> list[str]:
    return [
        "Oklch+: L′ = L^α, C′ = C^n/(C^n + σ^n), Euclidean ΔE in transformed Oklab.",
        "Optimized on COMBVD (3,813 suprathreshold pairs): α=0.73, n=0.87, σ=0.34.",
        "NR chroma preserves achromatic point C′=0; sigmoid fails f(0)=0 at optimum.",
        "STRESS uses García et al. Eqs. (1–2); lower is better.",
        "Euclidean distance in Oklch+ supports perceptually uniform interpolation.",
        "COMBVD is sRGB-centered (99.2% pairs C < 0.20); high-chroma validation is future work.",
    ]


def table1_chroma_functions() -> dict[str, Any]:
    """Table 1: chroma transform comparison on full COMBVD."""
    return {
        "power": {"params": 2, "stress": 29.99, "notes": "unbounded"},
        "log": {"params": 2, "stress": 34.36, "beta": 4.92},
        "sigmoid": {"params": 3, "stress": 44.45, "f0_zero": False},
        "naka_rushton": {"params": 3, "stress": 29.09, "adopted": True, "f0_zero": True},
    }


def table2_overall_stress() -> dict[str, Any]:
    """Table 2: overall COMBVD STRESS."""
    return {
        "oklab": {"stress": 47.35, "params": 0},
        "power_lc": {"stress": 29.99, "params": 2, "alpha": 0.52, "gamma": 0.72},
        "oklch_plus": {"stress": 29.09, "params": 3},
        "ciede2000": {"stress": 29.13, "params": 17},
        "helmlab_ref": {"stress": 22.48, "params": 72, "comparable": False},
    }


def table3_subdataset_stress() -> dict[str, dict[str, float]]:
    """Table 3: STRESS by COMBVD sub-dataset."""
    return {
        "D65": {"oklab": 51.45, "oklch_plus": 23.96, "ciede2000": 24.12},
        "C": {"oklab": 41.69, "oklch_plus": 28.11, "ciede2000": 28.87},
        "M": {"oklab": 42.22, "oklch_plus": 34.29, "ciede2000": 35.05},
        "LEEDS": {"oklab": 45.01, "oklch_plus": 24.27, "ciede2000": 19.28},
        "RIT_DuPont": {"oklab": 31.76, "oklch_plus": 25.16, "ciede2000": 19.49},
        "WITT": {"oklab": 45.15, "oklch_plus": 34.05, "ciede2000": 30.27},
        "ALL": {"oklab": 47.35, "oklch_plus": 29.09, "ciede2000": 29.13},
    }


def table4_attribute_stress() -> dict[str, Any]:
    """Table 4: attribute-stratified STRESS."""
    return {
        "all": {"n": 3813, "ciede2000": 29.13, "oklab": 47.35, "oklch_plus": 29.09},
        "lightness": {"n": 142, "ciede2000": 31.29, "oklab": 28.33, "oklch_plus": 26.97},
        "chroma": {"n": 122, "ciede2000": 23.70, "oklab": 29.61, "oklch_plus": 27.13},
        "hue": {"n": 122, "ciede2000": 17.66, "oklab": 28.68, "oklch_plus": 22.68},
    }


def table5_cross_validation() -> dict[str, Any]:
    """Table 5: held-out BFD-P D65 cross-validation."""
    return {
        "train_pairs": 1785,
        "test_pairs": 2028,
        "test_bfd_p_d65": {
            "oklab": 51.45,
            "oklch_plus_cv": 26.14,
            "ciede2000_ref": 24.12,
        },
        "train_oklch_plus_cv": 33.02,
        "train_oklab": 42.62,
    }


def table6_cam16_appearance() -> dict[str, float]:
    """Table 6: STRESS with CAM16-UCS as ΔV (sRGB)."""
    return {"oklab": 33.42, "oklch_plus": 16.61}


def table7_gamut_cam16() -> dict[str, dict[str, float]]:
    """Table 7: CAM16 ratio uniformity by gamut."""
    return {
        "pointers": {"outside_srgb_pct": 0, "oklab": 38.95, "oklch_plus": 18.76},
        "srgb": {"outside_srgb_pct": 0, "oklab": 33.42, "oklch_plus": 16.61},
        "display_p3": {"outside_srgb_pct": 37, "oklab": 29.64, "oklch_plus": 16.85},
        "rec2020": {"outside_srgb_pct": 58, "oklab": 25.90, "oklch_plus": 18.38},
    }


def table8_appearance_vs_discrimination() -> dict[str, dict[str, Any]]:
    """Table 8: Oklch+ advantage (ΔSTRESS Oklab − Oklch+) by attribute."""
    return {
        "lightness": {"combvd": 1.36, "cam16": 10.72, "consistent": True},
        "chroma": {"combvd": 2.48, "cam16": -0.37, "consistent": False},
        "hue": {"combvd": 6.00, "cam16": -6.89, "consistent": False},
    }
