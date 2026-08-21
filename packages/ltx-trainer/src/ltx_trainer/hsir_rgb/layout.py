"""Fig. 2 tensor layout + paper scope notes (arXiv:2605.24769)."""

from __future__ import annotations

from typing import Any


def architecture_layout(
    *,
    height: int,
    width: int,
    num_bands: int,
    num_groups: int,
    inner_channels: int,
) -> dict[str, Any]:
    """Outer denoiser D^h_σ wiring (Fig. 2) — shapes only."""
    kc = num_groups * inner_channels
    return {
        "noisy_input_y": {"shape": ["B", num_bands, height, width], "role": "Observed HS cube"},
        "encoder_E": {
            "shape": [kc, num_bands],
            "role": "Linear spectral projection Ψ^e_k; QR orthonormal columns when Kc ≥ C",
        },
        "latent_groups_z": {
            "shape": ["B", num_groups, inner_channels, height, width],
            "role": "Stacked low-dim groups fed to frozen D^l_σ",
        },
        "inner_denoiser_Dl": {
            "shape": ["B", inner_channels, height, width],
            "role": "Frozen RGB/mono DRUNet (trainable: none)",
        },
        "decoder_F": {"shape": [num_bands, kc], "role": "Linear aggregation F = E^T (Moore–Penrose when Kc < C)"},
        "output_x_hat": {"shape": ["B", num_bands, height, width], "role": "D^h_σ(y)"},
    }


def paper_limitations() -> list[dict[str, str]]:
    """Conclusion / future work (Sec. 4)."""
    return [
        {
            "id": "frozen_rgb_black_box",
            "summary": "Inner DRUNet stays frozen; full RGB prior may be underused.",
            "symptom": "Future work: lightweight stats/noise matching while keeping generality.",
        },
        {
            "id": "pca_projection_mismatch",
            "summary": "PCA subspaces need not align with RGB denoiser latents.",
            "symptom": "Table 3: PCA proj. SAM ≫ proposed learned projection.",
        },
        {
            "id": "kc_lt_c_reconstruction",
            "summary": "When Kc < C, FE ≠ I and only minimum-norm least-squares recovery holds.",
            "symptom": "Grp. K = 1 underperforms full K = 11 stack (Table 3).",
        },
    ]
