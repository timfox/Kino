"""Sparse 4D BCV framework layout and methodology notes."""

from __future__ import annotations

EVALUATION_METRICS: tuple[dict[str, str], ...] = (
    {
        "name": "MSE",
        "category": "global_error",
        "measures": "Mean-squared voxel-wise intensity deviation",
    },
    {
        "name": "PSNR",
        "category": "global_error",
        "measures": "Log-scaled peak signal relative to MSE",
    },
    {
        "name": "DSSIM",
        "category": "structural",
        "measures": "Dissimilarity SSIM (luminance, contrast, structure)",
    },
    {
        "name": "NMI",
        "category": "structural",
        "measures": "Normalized mutual information",
    },
    {
        "name": "NCC",
        "category": "structural",
        "measures": "Normalized cross-correlation (linear agreement)",
    },
    {
        "name": "FHC",
        "category": "structural",
        "measures": "Fourier hypershell correlation; half-bit resolution",
    },
)

COMPARISON_MODES: tuple[str, ...] = (
    "subset_vs_ground_truth",
    "subset_vs_pseudo_reference",
    "cross_validation_interlaced",
)

LIMITATIONS: tuple[str, ...] = (
    "Demonstrated on simulated water-droplet collisions only (not real experimental noise).",
    "Assumes temporal sampling at or above Nyquist; spatial sparsity is the focus.",
    "Physics-based validation (NS–CH fields) not incorporated.",
    "Performance estimates are dataset-dependent; larger independent data improves reliability.",
)

RECOMMENDED_PRACTICE: tuple[str, ...] = (
    "Prefer NCC (real space) + FHC (frequency space) over MSE/PSNR in ultra-sparse regimes.",
    "Use interlaced-time cross-validation for true 4D independence (not 3D-only).",
    "Monitor variance across bootstrap subsets; <1% std indicates stable regime.",
    "Subset-to-full plateau suggests pseudo-reference ỹ is near-optimal.",
)
