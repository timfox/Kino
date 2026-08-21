"""Configuration for HistoBIT3D virtual staining (Song et al., arXiv:2605.22000)."""

from __future__ import annotations

from dataclasses import dataclass

BASELINES: tuple[str, ...] = (
    "cyclegan",
    "stable",
    "cyclediffusion",
    "uvcganv2",
    "base_style",
)

DATASET_SUBSETS: tuple[str, ...] = (
    "duodenum_paired_bit_fluoro",
    "duodenum_bit_only_crypts",
    "duodenum_bit_only_muscularis",
    "kidney_2d_normal_cancer",
)

MSC_SCALES: tuple[int, ...] = (1, 2, 4, 16)


@dataclass
class HistoBIT3DConfig:
    """Defaults from paper Sec. 2–3."""

    image_size: int = 512
    bit_channels: int = 3  # original, inverted, original
    background_sigma: float = 30.0
    # Loss weights (Sec. 3.1)
    lambda_cycle: float = 10.0
    lambda_idt: float = 0.5
    lambda_msc: float = 1.0
    lambda_style: float = 1.0
    # Style prototype EMA
    style_ema_alpha: float = 0.99
    # Table 1 — Our Model
    reference_fid: float = 60.69
    reference_kid: float = 0.0417
    reference_dice_3d: float = 0.594
    reference_nuclei_volume_um3: float = 408.3
    reference_hd95_um: float = 4.04
    ground_truth_nuclei_volume_um3: float = 405.6
    # Dataset scale (Sec. 2.1)
    patches_per_subset: int = 5000
    z_stack_um_min: float = 30.0
    z_stack_um_max: float = 50.0
    led_wavelength_nm: float = 660.0
