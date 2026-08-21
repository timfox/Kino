"""HistoBIT3D pipeline glue and benchmark tables (Sec. 3)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ltx_trainer.histobit3d.config import BASELINES, DATASET_SUBSETS, HistoBIT3DConfig
from ltx_trainer.histobit3d.losses import (
    adain_token,
    bidirectional_msc_loss,
    demo_feature_maps,
    style_statistics_loss,
    total_histobit_loss,
    update_style_prototype,
)
from ltx_trainer.histobit3d.metrics import evaluate_3d_segmentation
from ltx_trainer.histobit3d.preprocessing import bit_three_channel_stack, scale_to_uint8


def benchmark_table_sota(cfg: HistoBIT3DConfig | None = None) -> dict[str, dict[str, float]]:
    """Table 1 — realism and 3D fidelity."""
    cfg = cfg or HistoBIT3DConfig()
    return {
        "cyclegan": {"fid": 136.01, "kid": 0.1556, "dice_3d": 0.360, "nuclei_volume_um3": 377.2, "hd95_um": 5.59},
        "stable": {"fid": 86.54, "kid": 0.0684, "dice_3d": 0.418, "nuclei_volume_um3": 331.9, "hd95_um": 5.38},
        "cyclediffusion": {"fid": 106.83, "kid": 0.1138, "dice_3d": 0.359, "nuclei_volume_um3": 233.2, "hd95_um": 4.77},
        "uvcganv2": {"fid": 95.80, "kid": 0.0821, "dice_3d": 0.515, "nuclei_volume_um3": 386.7, "hd95_um": 4.67},
        "base_style": {"fid": 62.45, "kid": 0.0434, "dice_3d": 0.583, "nuclei_volume_um3": 416.6, "hd95_um": 4.22},
        "our_model": {
            "fid": cfg.reference_fid,
            "kid": cfg.reference_kid,
            "dice_3d": cfg.reference_dice_3d,
            "nuclei_volume_um3": cfg.reference_nuclei_volume_um3,
            "hd95_um": cfg.reference_hd95_um,
        },
    }


def ablation_table() -> list[dict[str, Any]]:
    """Incremental ablation: Base → Base+Style → Our Model."""
    tab = benchmark_table_sota()
    return [
        {"variant": "uvcganv2_base", **{k: tab["uvcganv2"][k] for k in ("fid", "kid", "dice_3d", "hd95_um")}},
        {"variant": "base_style", **{k: tab["base_style"][k] for k in ("fid", "kid", "dice_3d", "hd95_um")}},
        {"variant": "our_model", **{k: tab["our_model"][k] for k in ("fid", "kid", "dice_3d", "hd95_um")}},
    ]


def training_step_demo(cfg: HistoBIT3DConfig | None = None) -> dict[str, float]:
    """Smoke training step with synthetic feature maps."""
    cfg = cfg or HistoBIT3DConfig()
    bf = demo_feature_maps()
    bb = demo_feature_maps()
    hf = demo_feature_maps()
    hb = demo_feature_maps()
    msc = bidirectional_msc_loss(bf, bb, hf, hb)
    s_fake = torch.randn(2, 256)
    s_real = torch.randn(2, 256)
    style = style_statistics_loss(s_fake, s_real)
    proto = update_style_prototype(None, s_real[0], alpha=cfg.style_ema_alpha)
    fused = adain_token(s_fake[0], proto)
    losses = total_histobit_loss(
        cycle_loss=torch.tensor(0.5),
        identity_loss=torch.tensor(0.1),
        msc_loss=msc,
        style_loss=style,
        cfg=cfg,
    )
    return {k: float(v.item()) for k, v in losses.items()}


def preprocess_bit_slice(bit_2d: Tensor) -> Tensor:
    """Full BIT preprocessing pipeline for one axial slice."""
    stack = bit_three_channel_stack(bit_2d)
    return scale_to_uint8(stack)


def dataset_card(cfg: HistoBIT3DConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HistoBIT3DConfig()
    return {
        "name": "HistoBIT3D",
        "paper": "arXiv:2605.22000",
        "title": "Virtual 3D H&E Staining from Phase-contrast BIT",
        "modalities": ["BIT", "fluorescence_nuclei", "FFPE_H&E_unpaired"],
        "subsets": list(DATASET_SUBSETS),
        "patches_per_subset": cfg.patches_per_subset,
        "image_size": cfg.image_size,
        "bit_input_channels": cfg.bit_channels,
        "loss_weights": {
            "lambda_cycle": cfg.lambda_cycle,
            "lambda_idt": cfg.lambda_idt,
            "lambda_msc": cfg.lambda_msc,
            "lambda_style": cfg.lambda_style,
        },
        "baselines": list(BASELINES),
        "code": "https://github.com/aasong113/HistoBIT3D_VirtualStaining",
    }


def demo_segmentation_metrics() -> dict[str, float]:
    """Synthetic 3D mask evaluation."""
    gt = torch.zeros(32, 64, 64)
    gt[:, 20:28, 20:28] = 1.0
    pred = gt.clone()
    pred[:, 21:29, 21:29] = 1.0  # slight shift
    return evaluate_3d_segmentation(pred, gt, voxel_um3=0.5)
