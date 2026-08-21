"""Pretrained RGB / mono denoisers for hyperspectral restoration (Picone, Jouni, Dalla Mura, arXiv:2605.24769)."""

from ltx_trainer.hsir_rgb.config import HSIRRgbConfig
from ltx_trainer.hsir_rgb.encoders import EncoderMode, denoise_by_mode, denoise_sequential
from ltx_trainer.hsir_rgb.inner import awgn_inner_denoiser, rgb_band_teaser_denoise
from ltx_trainer.hsir_rgb.layout import architecture_layout, paper_limitations
from ltx_trainer.hsir_rgb.metrics import hs_psnr, hs_ssim_band_mean, spectral_angle_mapper
from ltx_trainer.hsir_rgb.operators import add_awgn, degrade
from ltx_trainer.hsir_rgb.pipeline import (
    ablation_smoke_demo,
    adapter_training_loss,
    dataset_manifest,
    evaluation_demo,
    fig1_harvard_teaser,
    framework_card,
    harvard_untrained_identity_demo,
    pnp_restore_demo,
    qr_identity_reconstruction_error,
    references_bibtex,
    table_ablation_cave_sigma010,
    table_denoising,
    table_related_work,
    table_tasks_cave,
    task_protocol,
    training_step_demo,
)
from ltx_trainer.hsir_rgb.pnp import pnp_hqs_restore
from ltx_trainer.hsir_rgb.projection import (
    SpectralPnPAdapter,
    decode_stacked,
    encode_groups,
    encoder_decoder_from_unconstrained,
    hyperspectral_denoise,
)
from ltx_trainer.hsir_rgb.stability import lipschitz_surrogate, verify_qr_nonexpansive_identity_inner
from ltx_trainer.hsir_rgb.viz import extract_rgb_viz_cube, psnr_on_viz_bands

__all__ = [
    "HSIRRgbConfig",
    "EncoderMode",
    "SpectralPnPAdapter",
    "ablation_smoke_demo",
    "adapter_training_loss",
    "add_awgn",
    "architecture_layout",
    "awgn_inner_denoiser",
    "dataset_manifest",
    "decode_stacked",
    "degrade",
    "denoise_by_mode",
    "denoise_sequential",
    "encode_groups",
    "encoder_decoder_from_unconstrained",
    "evaluation_demo",
    "extract_rgb_viz_cube",
    "fig1_harvard_teaser",
    "framework_card",
    "harvard_untrained_identity_demo",
    "hs_psnr",
    "hs_ssim_band_mean",
    "hyperspectral_denoise",
    "lipschitz_surrogate",
    "paper_limitations",
    "pnp_hqs_restore",
    "pnp_restore_demo",
    "psnr_on_viz_bands",
    "qr_identity_reconstruction_error",
    "references_bibtex",
    "rgb_band_teaser_denoise",
    "spectral_angle_mapper",
    "table_ablation_cave_sigma010",
    "table_denoising",
    "table_related_work",
    "table_tasks_cave",
    "task_protocol",
    "training_step_demo",
    "verify_qr_nonexpansive_identity_inner",
]
