"""MSFET-E2V: event-to-video via spatio-temporal + wavelet transformer (arXiv:2605.25804).

Reference: Haar DWT/CDAM/WSB/RGD modules, paper Tables I–VII, smoke training demo.
Full ESIM training and EVREAL evaluation are external.
"""

from ltx_trainer.msfet_e2v.config import MSFETE2VConfig
from ltx_trainer.msfet_e2v.losses import reconstruction_loss, temporal_consistency_loss, total_loss
from ltx_trainer.msfet_e2v.metrics import psnr, ssim_proxy
from ltx_trainer.msfet_e2v.model import MSFETE2V
from ltx_trainer.msfet_e2v.pipeline import (
    evaluation_demo,
    framework_card,
    table_ablation_cdam_subbands,
    table_ablation_depth,
    table_ablation_modules,
    table_ablation_voxel_bins,
    table_brisque,
    table_inference_efficiency,
    table_quantitative_e2v,
    training_step_demo,
)
from ltx_trainer.msfet_e2v.voxel import events_to_voxel
from ltx_trainer.msfet_e2v.wavelet import haar_dwt2d, haar_idwt2d

__all__ = [
    "MSFETE2V",
    "MSFETE2VConfig",
    "evaluation_demo",
    "events_to_voxel",
    "framework_card",
    "haar_dwt2d",
    "haar_idwt2d",
    "psnr",
    "reconstruction_loss",
    "ssim_proxy",
    "table_ablation_cdam_subbands",
    "table_ablation_depth",
    "table_ablation_modules",
    "table_ablation_voxel_bins",
    "table_brisque",
    "table_inference_efficiency",
    "table_quantitative_e2v",
    "temporal_consistency_loss",
    "total_loss",
    "training_step_demo",
]
