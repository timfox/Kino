"""Train-time use of GOPEX AV fold sidecars (QoMEX VSR, AdaMaG, FMelCodec, etc.).

Sidecars are written at preprocess when ``GOPEX_AV_FOLD_HOOKS`` is set (see ``fold_registry``).
This module applies **per-sample loss weights** and light regularizers without changing latent shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ltx_trainer.config import AvFoldTrainingConfig


def _latent_shard_batch(batch: dict[str, Any]) -> dict[str, Any] | None:
    """Precomputed shard dict (contains ``latents`` tensor + fold sidecars)."""
    for key in ("latents", "latent_conditions", "video_latents"):
        src = batch.get(key)
        if isinstance(src, dict) and "latents" in src:
            return src
    return None


def _audio_shard_batch(batch: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("audio_latents", "audio_conditions"):
        src = batch.get(key)
        if isinstance(src, dict) and "latents" in src:
            return src
    return None


def _fold_nested(shard: dict[str, Any] | None, fold_key: str) -> dict[str, Any] | None:
    if not isinstance(shard, dict):
        return None
    fold = shard.get(fold_key)
    return fold if isinstance(fold, dict) else None


def _collated_scalar_field(batch: dict[str, Any], *, fold_key: str, field: str, audio: bool = False) -> Tensor | None:
    """Read a float sidecar field after default DataLoader collate."""
    src = _audio_shard_batch(batch) if audio else _latent_shard_batch(batch)
    if not isinstance(src, dict):
        return None
    fold = _fold_nested(src, fold_key)
    if fold is None:
        return None
    val = fold.get(field)
    if isinstance(val, Tensor):
        return val.detach().float().reshape(-1)
    if isinstance(val, (list, tuple)):
        return torch.tensor([float(x) for x in val], dtype=torch.float32)
    if isinstance(val, (int, float)):
        b = 1
        lat = src.get("latents")
        if isinstance(lat, Tensor) and lat.ndim >= 1:
            b = int(lat.shape[0])
        return torch.full((b,), float(val), dtype=torch.float32)
    return None


def fuse_flow_fusion_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight metric fusion-ready clips (FUSE MCM/CVGC proxy on latents)."""
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_fuse_flow_weights,
        fold_key="fuse_flow",
        field="fusion_readiness_proxy",
        min_weight=cfg.min_fuse_flow_weight,
    )


def fuse_flow_geometry_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Downweight unstable geometry (complements phyworld temporal physics)."""
    if not cfg.downweight_low_geometry_stability:
        return None
    stab = _collated_scalar_field(batch, fold_key="fuse_flow", field="geometry_stability_proxy")
    if stab is None:
        return None
    floor = 1.0 - cfg.low_geometry_penalty
    return floor + (1.0 - floor) * stab.clamp(0.0, 1.0)


def physics_faithfulness_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map PhyWorld ``physics_proxy`` (1–5) to per-sample weights in ``[min_quality_weight, 1]``."""
    if not cfg.downweight_low_physics:
        return None
    proxy = _collated_scalar_field(batch, fold_key="phyworld", field="physics_proxy")
    if proxy is None:
        return None
    norm = ((proxy - 1.0) / 4.0).clamp(0.0, 1.0)
    floor = 1.0 - cfg.low_physics_penalty
    return floor + (1.0 - floor) * norm


def avbench_alignment_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map AVBench ``av_align_proxy`` (0–1) to per-sample weights."""
    if not cfg.use_avbench_alignment_weights:
        return None
    align = _collated_scalar_field(batch, fold_key="avbench", field="av_align_proxy")
    if align is None:
        return None
    norm = align.clamp(0.0, 1.0)
    return cfg.min_avbench_weight + (1.0 - cfg.min_avbench_weight) * norm


def mmae_instruction_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map MMAE ``edit_complexity_proxy`` (0–1) to per-sample weights on audio/video sidecars."""
    if not cfg.use_mmae_instruction_weights:
        return None
    proxy = _collated_scalar_field(batch, fold_key="mmae", field="edit_complexity_proxy")
    if proxy is None:
        proxy = _collated_scalar_field(batch, fold_key="mmae", field="edit_complexity_proxy", audio=True)
    if proxy is None:
        return None
    norm = proxy.clamp(0.0, 1.0)
    return cfg.min_mmae_weight + (1.0 - cfg.min_mmae_weight) * norm


def mtavg2_expressiveness_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Downweight clips with high MTAVG2 cinematic expressiveness failure risk."""
    if not cfg.use_mtavg2_expressiveness_weights:
        return None
    risk = _collated_scalar_field(batch, fold_key="mtavg2", field="expressiveness_risk_proxy")
    if risk is None:
        return None
    norm = risk.clamp(0.0, 1.0)
    floor = max(cfg.min_mtavg2_weight, 1.0 - cfg.mtavg2_risk_penalty)
    return floor + (1.0 - floor) * (1.0 - norm)


def growloop_humanlikeness_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map GrowLoop final_score_bucket (0–4) to audio-sidecar weights."""
    if not cfg.use_growloop_weights:
        return None
    bucket = _collated_scalar_field(batch, fold_key="growloop", field="final_score_bucket", audio=True)
    if bucket is None:
        return None
    norm = (bucket / 4.0).clamp(0.0, 1.0)
    return cfg.min_growloop_weight + (1.0 - cfg.min_growloop_weight) * norm


def video_quality_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map VSR-VQA ``mos_proxy`` (1–5) to per-sample weights in ``[min_quality_weight, 1]``."""
    if not cfg.use_vsr_quality_weights:
        return None
    mos = _collated_scalar_field(batch, fold_key="vsr_vqa", field="mos_proxy")
    if mos is None:
        return None
    norm = ((mos - 1.0) / 4.0).clamp(0.0, 1.0)
    return cfg.min_quality_weight + (1.0 - cfg.min_quality_weight) * norm


def erp_boiqa_quality_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map VUGA / VU-BOIQA ``mos_proxy`` (1–5) to per-sample weights for ERP clips."""
    if not cfg.use_erp_boiqa_weights:
        return None
    vuga = _collated_scalar_field(batch, fold_key="vuga", field="mos_proxy")
    vub = _collated_scalar_field(batch, fold_key="vuboiqa", field="mos_proxy")
    if vuga is None and vub is None:
        return None
    if vuga is not None and vub is not None:
        mos = (vuga + vub) * 0.5
    else:
        mos = vuga if vuga is not None else vub
    assert mos is not None
    norm = ((mos - 1.0) / 4.0).clamp(0.0, 1.0)
    return cfg.min_erp_boiqa_weight + (1.0 - cfg.min_erp_boiqa_weight) * norm


def s3po_wss_quality_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map S3PO ``wss_quality_proxy`` / ``mos_proxy`` (1–5) to per-sample weights."""
    if not cfg.use_s3po_wss_weights:
        return None
    mos = _collated_scalar_field(batch, fold_key="s3po", field="wss_quality_proxy")
    if mos is None:
        mos = _collated_scalar_field(batch, fold_key="s3po", field="mos_proxy")
    if mos is None:
        return None
    norm = ((mos - 1.0) / 4.0).clamp(0.0, 1.0)
    return cfg.min_s3po_wss_weight + (1.0 - cfg.min_s3po_wss_weight) * norm


def lucky_hdr_merge_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map LuckyHDR ``merge_readiness_proxy`` (0–1) to per-sample weights."""
    if not cfg.use_lucky_hdr_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="lucky_hdr", field="merge_readiness_proxy")
    if ready is None:
        mos = _collated_scalar_field(batch, fold_key="lucky_hdr", field="mos_proxy")
        if mos is None:
            return None
        norm = ((mos - 1.0) / 4.0).clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_lucky_hdr_weight + (1.0 - cfg.min_lucky_hdr_weight) * norm


def era_defocus_sharpness_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map ErA ``deblur_readiness_proxy`` / ``sharpness_proxy`` (0–1) to per-sample weights."""
    if not cfg.use_era_defocus_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="era_defocus", field="deblur_readiness_proxy")
    if ready is None:
        sharp = _collated_scalar_field(batch, fold_key="era_defocus", field="sharpness_proxy")
        if sharp is None:
            mos = _collated_scalar_field(batch, fold_key="era_defocus", field="mos_proxy")
            if mos is None:
                return None
            norm = ((mos - 1.0) / 4.0).clamp(0.0, 1.0)
        else:
            norm = sharp.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_era_defocus_weight + (1.0 - cfg.min_era_defocus_weight) * norm


def flood_physics_hydrology_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map ``inundation_readiness_proxy`` / ``physics_consistency_proxy`` (0–1) to per-sample weights."""
    if not cfg.use_flood_physics_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="flood_physics", field="inundation_readiness_proxy")
    if ready is None:
        phys = _collated_scalar_field(batch, fold_key="flood_physics", field="physics_consistency_proxy")
        if phys is None:
            extent = _collated_scalar_field(batch, fold_key="flood_physics", field="flood_extent_proxy")
            if extent is None:
                return None
            norm = extent.clamp(0.0, 1.0)
        else:
            norm = phys.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_flood_physics_weight + (1.0 - cfg.min_flood_physics_weight) * norm


def dilated_sym_diff_change_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map ``change_detection_readiness_proxy`` / ``boundary_contrast_proxy`` to per-sample weights."""
    if not cfg.use_dilated_sym_diff_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="dilated_sym_diff", field="change_detection_readiness_proxy")
    if ready is None:
        boundary = _collated_scalar_field(batch, fold_key="dilated_sym_diff", field="boundary_contrast_proxy")
        if boundary is None:
            return None
        norm = boundary.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_dilated_sym_diff_weight + (1.0 - cfg.min_dilated_sym_diff_weight) * norm


def dynamic_gp_spatiotemporal_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map ``filter_readiness_proxy`` / ``basis_fit_proxy`` (0–1) to per-sample weights."""
    if not cfg.use_dynamic_gp_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="dynamic_gp", field="filter_readiness_proxy")
    if ready is None:
        basis = _collated_scalar_field(batch, fold_key="dynamic_gp", field="basis_fit_proxy")
        if basis is None:
            temporal = _collated_scalar_field(batch, fold_key="dynamic_gp", field="temporal_smoothness_proxy")
            if temporal is None:
                return None
            norm = temporal.clamp(0.0, 1.0)
        else:
            norm = basis.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_dynamic_gp_weight + (1.0 - cfg.min_dynamic_gp_weight) * norm


def branch_energy_localization_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map branch_energy localization / Tellegen readiness proxies to per-sample weights."""
    if not cfg.use_branch_energy_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="branch_energy", field="localization_readiness_proxy")
    if ready is None:
        balance = _collated_scalar_field(batch, fold_key="branch_energy", field="tellegen_balance_proxy")
        if balance is None:
            harmonic = _collated_scalar_field(batch, fold_key="branch_energy", field="harmonic_spread_proxy")
            if harmonic is None:
                return None
            norm = harmonic.clamp(0.0, 1.0)
        else:
            norm = balance.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_branch_energy_weight + (1.0 - cfg.min_branch_energy_weight) * norm


def lora_hd_attn_rank_one_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Map lora_hd_attn rank-one / effective-noise proxies to per-sample weights."""
    if not cfg.use_lora_hd_attn_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="lora_hd_attn", field="lora_readiness_proxy")
    if ready is None:
        align = _collated_scalar_field(batch, fold_key="lora_hd_attn", field="rank_one_alignment_proxy")
        if align is None:
            noise = _collated_scalar_field(batch, fold_key="lora_hd_attn", field="effective_noise_proxy")
            if noise is None:
                return None
            norm = (1.0 - noise).clamp(0.0, 1.0)
        else:
            norm = align.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_lora_hd_attn_weight + (1.0 - cfg.min_lora_hd_attn_weight) * norm


def latenthdr_exposure_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight clips with high latenthdr exposure_readiness_proxy (HDR EV stacks)."""
    if not cfg.use_latenthdr_weights:
        return None
    ready = _collated_scalar_field(batch, fold_key="latenthdr", field="exposure_readiness_proxy")
    if ready is None:
        cov = _collated_scalar_field(batch, fold_key="latenthdr", field="bracket_coverage_proxy")
        if cov is None:
            return None
        norm = cov.clamp(0.0, 1.0)
    else:
        norm = ready.clamp(0.0, 1.0)
    return cfg.min_latenthdr_weight + (1.0 - cfg.min_latenthdr_weight) * norm


def _readiness_weights(
    batch: dict[str, Any],
    cfg: "AvFoldTrainingConfig",
    *,
    enabled: bool,
    fold_key: str,
    field: str,
    min_weight: float,
    audio: bool = False,
) -> Tensor | None:
    if not enabled:
        return None
    ready = _collated_scalar_field(batch, fold_key=fold_key, field=field, audio=audio)
    if ready is None:
        return None
    norm = ready.clamp(0.0, 1.0)
    return min_weight + (1.0 - min_weight) * norm


def _cfg_readiness_weights(
    batch: dict[str, Any],
    cfg: "AvFoldTrainingConfig",
    *,
    use_flag: str,
    fold_key: str,
    field: str,
    min_weight_attr: str,
    audio: bool = False,
) -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=getattr(cfg, use_flag),
        fold_key=fold_key,
        field=field,
        min_weight=getattr(cfg, min_weight_attr),
        audio=audio,
    )


# Remaining ERP / pano fold hooks (Round 19): config flag, fold key, field, min-weight attr, log metric
_PANO360_READINESS_WEIGHTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("use_cubediff_weights", "cubediff", "panorama_readiness", "min_cubediff_weight", "av_fold/cubediff_weight_mean"),
    ("use_spherediff_weights", "spherediff", "spherical_readiness", "min_spherediff_weight", "av_fold/spherediff_weight_mean"),
    ("use_erpgs_weights", "erpgs", "splat_readiness", "min_erpgs_weight", "av_fold/erpgs_weight_mean"),
    ("use_sphereuformer_weights", "sphereuformer", "perception_readiness", "min_sphereuformer_weight", "av_fold/sphereuformer_weight_mean"),
    ("use_sphere_depth_weights", "sphere_depth", "calibration_readiness", "min_sphere_depth_weight", "av_fold/sphere_depth_weight_mean"),
    ("use_nvc_erp_qpa_weights", "nvc_erp_qpa", "qpa_readiness", "min_nvc_erp_qpa_weight", "av_fold/nvc_erp_qpa_weight_mean"),
    ("use_mtpano_weights", "mtpano", "multitask_dense_readiness", "min_mtpano_weight", "av_fold/mtpano_weight_mean"),
    ("use_panoworld_x_weights", "panoworld_x", "explorable_world_readiness", "min_panoworld_x_weight", "av_fold/panoworld_x_weight_mean"),
    ("use_pano_affordance_weights", "pano_affordance", "affordance_readiness", "min_pano_affordance_weight", "av_fold/pano_affordance_weight_mean"),
    ("use_panogsdet_weights", "panogsdet", "gs_detection_readiness", "min_panogsdet_weight", "av_fold/panogsdet_weight_mean"),
    ("use_panolm_weights", "panolm", "panovqa_readiness", "min_panolm_weight", "av_fold/panolm_weight_mean"),
    ("use_pano_flight_weights", "pano_flight", "survey_task_readiness", "min_pano_flight_weight", "av_fold/pano_flight_weight_mean"),
    ("use_pano360_weights", "pano360", "photogrammetry_readiness", "min_pano360_weight", "av_fold/pano360_weight_mean"),
)

# Audio / V2A readiness hooks: config flag, fold key, field, min-weight attr, log metric, audio shard
_AUDIO_READINESS_WEIGHTS: tuple[tuple[str, str, str, str, str, bool], ...] = (
    ("use_holitok_weights", "holitok", "holistic_readiness_proxy", "min_holitok_weight", "av_fold/holitok_weight_mean", True),
    ("use_flatsounds_weights", "flatsounds", "onset_strength_proxy", "min_flatsounds_weight", "av_fold/flatsounds_weight_mean", False),
)


def sdr2hdr_merge_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_sdr2hdr_weights,
        fold_key="sdr2hdr",
        field="merge_readiness_proxy",
        min_weight=cfg.min_sdr2hdr_weight,
    )


def x2hdr_alignment_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_x2hdr_weights,
        fold_key="x2hdr",
        field="pu21_alignment_readiness",
        min_weight=cfg.min_x2hdr_weight,
    )


def lf_diff_bracket_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_lf_diff_weights,
        fold_key="lf_diff",
        field="bracket_readiness",
        min_weight=cfg.min_lf_diff_weight,
    )


def diffhdr_log_gamma_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_diffhdr_weights,
        fold_key="diffhdr",
        field="log_gamma_readiness",
        min_weight=cfg.min_diffhdr_weight,
    )


def vdp_hdr_fusion_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_vdp_hdr_weights,
        fold_key="vdp_hdr",
        field="fusion_readiness_proxy",
        min_weight=cfg.min_vdp_hdr_weight,
    )


def pantheon360_cache_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_pantheon360_weights,
        fold_key="pantheon360",
        field="cache_fusion_readiness",
        min_weight=cfg.min_pantheon360_weight,
    )


def mirage_readout_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_mirage_weights,
        fold_key="mirage",
        field="readout_coverage",
        min_weight=cfg.min_mirage_readout_weight,
    )


def semantic_stitch_seam_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_semantic_stitch_weights,
        fold_key="semantic_stitch",
        field="seam_coherence",
        min_weight=cfg.min_semantic_stitch_weight,
    )


def sphere360_timelapse_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_sphere360_weights,
        fold_key="sphere360",
        field="timelapse_readiness",
        min_weight=cfg.min_sphere360_weight,
    )


def panoworld_panospace_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_panoworld_weights,
        fold_key="panoworld",
        field="panospace_readiness",
        min_weight=cfg.min_panoworld_weight,
    )


def gimbal360_canonical_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_gimbal360_weights,
        fold_key="gimbal360",
        field="canonical_erp_readiness",
        min_weight=cfg.min_gimbal360_weight,
    )


def panoenv_spatial_vqa_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_panoenv_weights,
        fold_key="panoenv",
        field="spatial_vqa_readiness",
        min_weight=cfg.min_panoenv_weight,
    )


def weatherproof_robustness_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_weatherproof_weights,
        fold_key="weatherproof",
        field="robustness_readiness",
        min_weight=cfg.min_weatherproof_weight,
    )


def dense360_omni_vlm_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_dense360_weights,
        fold_key="dense360",
        field="omni_vlm_readiness",
        min_weight=cfg.min_dense360_weight,
    )


def cross360_depth_fusion_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_cross360_weights,
        fold_key="cross360",
        field="depth_fusion_readiness",
        min_weight=cfg.min_cross360_weight,
    )


def anything360_conditioning_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_anything360_weights,
        fold_key="anything360",
        field="conditioning_readiness",
        min_weight=cfg.min_anything360_weight,
    )


def spherefusion_depth_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_spherefusion_weights,
        fold_key="spherefusion",
        field="depth_readiness",
        min_weight=cfg.min_spherefusion_weight,
    )


def raim_mef_fusion_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_raim_mef_weights,
        fold_key="raim_mef",
        field="mef_fusion_readiness",
        min_weight=cfg.min_raim_mef_weight,
    )


def lumivid_logc3_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_lumivid_weights,
        fold_key="lumivid",
        field="logc3_alignment_readiness",
        min_weight=cfg.min_lumivid_weight,
    )


def stem2_isotonic_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_stem2_sdr_hdr_weights,
        fold_key="stem2_sdr_hdr",
        field="isotonic_readiness_proxy",
        min_weight=cfg.min_stem2_sdr_hdr_weight,
    )


def physthdr_gs_training_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_physthdr_gs_weights,
        fold_key="physthdr_gs",
        field="gs_training_readiness",
        min_weight=cfg.min_physthdr_gs_weight,
    )


def modulo_spike_unwrap_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_modulo_spike_hdr_weights,
        fold_key="modulo_spike_hdr",
        field="unwrap_readiness_proxy",
        min_weight=cfg.min_modulo_spike_hdr_weight,
    )


def core_kd_drift_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Downweight high CoRe-KD state drift (missing-modality / PoE mismatch proxy)."""
    if not cfg.downweight_high_core_kd_drift:
        return None
    drift = _collated_scalar_field(batch, fold_key="core_kd", field="state_drift_proxy")
    if drift is None:
        return None
    norm = (drift / 0.5).clamp(0.0, 1.0)
    return (1.0 - cfg.core_kd_drift_penalty * norm).clamp(0.1, 1.0)


def entroad_anomaly_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Downweight localized entropy spikes (EntroAD ZSAD proxy)."""
    if not cfg.downweight_entroad_anomaly:
        return None
    risk = _collated_scalar_field(batch, fold_key="entroad", field="localized_entropy_spike_proxy")
    if risk is None:
        return None
    norm = (risk / 1.0).clamp(0.0, 1.0)
    return (1.0 - cfg.entroad_anomaly_penalty * norm).clamp(0.1, 1.0)


def eigenet_reverb_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio when eigenet decay_ratio_proxy suggests reverberant tail energy."""
    if not cfg.use_eigenet_reverb_weights:
        return None
    decay = _collated_scalar_field(batch, fold_key="eigenet", field="decay_ratio_proxy", audio=True)
    if decay is None:
        return None
    norm = (decay / 2.0).clamp(0.0, 1.0)
    return cfg.min_eigenet_weight + (1.0 - cfg.min_eigenet_weight) * norm


def planaudio_composition_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio when planaudio semantic coverage is high."""
    if not cfg.use_planaudio_composition_weights:
        return None
    cov = _collated_scalar_field(
        batch, fold_key="planaudio", field="semantic_coverage_factor_proxy", audio=True
    )
    if cov is None:
        return None
    norm = cov.clamp(0.0, 1.0)
    return cfg.min_planaudio_weight + (1.0 - cfg.min_planaudio_weight) * norm


def ag_repa_causal_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio when FoG-A proxy suggests early-layer causal leverage (AG-REPA)."""
    if not cfg.use_ag_repa_causal_weights:
        return None
    fog = _collated_scalar_field(batch, fold_key="ag_repa", field="fog_a_proxy", audio=True)
    if fog is None:
        return None
    norm = (fog / cfg.ag_repa_fog_scale).clamp(0.0, 1.0)
    return cfg.min_ag_repa_weight + (1.0 - cfg.min_ag_repa_weight) * norm


def forte_t2a_alignment_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight video clips with high FORTE t2a_align_proxy."""
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_forte_t2a_weights,
        fold_key="forte",
        field="t2a_align_proxy",
        min_weight=cfg.min_forte_t2a_weight,
    )


def forte_audio_alignment_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio shards with high forte_audio t2a_align_proxy."""
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_forte_audio_weights,
        fold_key="forte_audio",
        field="t2a_align_proxy",
        min_weight=cfg.min_forte_audio_weight,
        audio=True,
    )


def omnicustom_video_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight video when sync AV customization sidecar identity_proxy is high."""
    if not cfg.use_omnicustom_video_weights:
        return None
    ident = _collated_scalar_field(batch, fold_key="omnicustom", field="identity_proxy")
    if ident is None:
        return None
    norm = ident.clamp(0.0, 1.0)
    return cfg.min_omnicustom_identity_weight + (1.0 - cfg.min_omnicustom_identity_weight) * norm


def omnicustom_timbre_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio when omnicustom timbre_slot sidecar is strong."""
    if not cfg.use_omnicustom_timbre_weights:
        return None
    timbre = _collated_scalar_field(batch, fold_key="omnicustom", field="timbre_proxy", audio=True)
    if timbre is None:
        return None
    norm = timbre.clamp(0.0, 1.0)
    return cfg.min_omnicustom_timbre_weight + (1.0 - cfg.min_omnicustom_timbre_weight) * norm


def id_lora_video_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight video when ID-LoRA structured prompt identity_proxy is high."""
    if not cfg.use_id_lora_video_weights:
        return None
    ident = _collated_scalar_field(batch, fold_key="id_lora", field="identity_proxy")
    if ident is None:
        return None
    norm = ident.clamp(0.0, 1.0)
    return cfg.min_id_lora_identity_weight + (1.0 - cfg.min_id_lora_identity_weight) * norm


def id_lora_audio_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio when ID-LoRA speaker_identity_proxy is high."""
    if not cfg.use_id_lora_audio_weights:
        return None
    speaker = _collated_scalar_field(batch, fold_key="id_lora", field="speaker_identity_proxy", audio=True)
    if speaker is None:
        return None
    norm = speaker.clamp(0.0, 1.0)
    return cfg.min_id_lora_speaker_weight + (1.0 - cfg.min_id_lora_speaker_weight) * norm


def autocut_video_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight video when ad-editing narrative_proxy is high (AutoCut)."""
    if not cfg.use_autocut_video_weights:
        return None
    narr = _collated_scalar_field(batch, fold_key="autocut", field="narrative_proxy")
    if narr is None:
        narr = _collated_scalar_field(batch, fold_key="autocut", field="edit_ready_proxy")
    if narr is None:
        return None
    norm = narr.clamp(0.0, 1.0)
    return cfg.min_autocut_narrative_weight + (1.0 - cfg.min_autocut_narrative_weight) * norm


def autocut_bgm_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight audio when autocut bgm_proxy suggests script–BGM alignment readiness."""
    if not cfg.use_autocut_bgm_weights:
        return None
    bgm = _collated_scalar_field(batch, fold_key="autocut", field="bgm_proxy", audio=True)
    if bgm is None:
        return None
    norm = bgm.clamp(0.0, 1.0)
    return cfg.min_autocut_bgm_weight + (1.0 - cfg.min_autocut_bgm_weight) * norm


def svhighlights_saliency_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Upweight video when svhighlights saliency_norm is high (TF-SELECTOR sidecar)."""
    return _readiness_weights(
        batch,
        cfg,
        enabled=cfg.use_svhighlights_saliency_weights,
        fold_key="svhighlights",
        field="saliency_norm",
        min_weight=cfg.min_svhighlights_saliency_weight,
    )


def _audio_stability_weights(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Downweight clips with high degraded-frame ratio or repeat-augment instability."""
    if not cfg.downweight_degraded_audio and not cfg.downweight_unstable_audio and not cfg.use_speech_mos_weights:
        return None
    w: Tensor | None = None
    if cfg.use_speech_mos_weights:
        pseudo = _collated_scalar_field(
            batch, fold_key="speech_quality_emb", field="pseudo_mos_mean", audio=True
        )
        if pseudo is not None:
            # Proxy MOS is stored near 0..1 by fold.py; tolerate 1..5 if a future evaluator writes that scale.
            norm = torch.where(pseudo > 1.5, (pseudo - 1.0) / 4.0, pseudo).clamp(0.0, 1.0)
            factor = cfg.min_audio_quality_weight + (1.0 - cfg.min_audio_quality_weight) * norm
            w = factor if w is None else w * factor
    if cfg.downweight_degraded_audio:
        deg = _collated_scalar_field(
            batch, fold_key="speech_quality_emb", field="degraded_frame_ratio", audio=True
        )
        if deg is not None:
            factor = (1.0 - cfg.degraded_audio_penalty * deg.clamp(0.0, 1.0)).clamp(0.1, 1.0)
            w = factor if w is None else w * factor
    if cfg.downweight_unstable_audio:
        delta = _collated_scalar_field(
            batch, fold_key="robustspeechflow", field="repeat_delta_mean", audio=True
        )
        if delta is not None:
            # Typical deltas are small; scale so 0.15 → ~0.35 penalty
            instability = (delta / 0.15).clamp(0.0, 1.0)
            factor = (1.0 - cfg.unstable_audio_penalty * instability).clamp(0.1, 1.0)
            w = factor if w is None else w * factor
    return w


def fmelcodec_coding_regularizer(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Optional scalar: mean proxy coding loss (encourages stable mel-code statistics)."""
    if cfg.lambda_fmelcodec_coding <= 0:
        return None
    coding = _collated_scalar_field(batch, fold_key="fmelcodec", field="coding_loss_proxy", audio=True)
    if coding is None:
        return None
    # Sidecar proxies are unnormalized (often 1e3+); clamp like wavenext2 so aux loss cannot dominate FM.
    return coding.clamp(min=0.0, max=10.0).mean()


def wavenext2_spectral_regularizer(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """Optional scalar: spectral envelope probe from WaveNeXt 2 sidecars."""
    if cfg.lambda_wavenext2_spectral <= 0:
        return None
    band = _collated_scalar_field(batch, fold_key="wavenext2", field="stft_band_energy", audio=True)
    if band is None:
        return None
    # Large latent spectral energy is a proxy for hiss/instability; keep the term bounded.
    return band.clamp(min=0.0, max=10.0).mean()


def _video_latents_tensor(batch: dict[str, Any]) -> Tensor | None:
    shard = _latent_shard_batch(batch)
    if shard is None:
        return None
    lat = shard.get("latents")
    if isinstance(lat, Tensor):
        return lat
    return None


def group_action_temporal_loss(batch: dict[str, Any], cfg: "AvFoldTrainingConfig") -> Tensor | None:
    """First vs last latent frame identity (group-action WM regularizer)."""
    if cfg.lambda_group_action_temporal <= 0:
        return None
    z = _video_latents_tensor(batch)
    if z is None or z.ndim < 4:
        return None
    # [B, C, F, H, W] or [B, C, T, H, W]
    if z.shape[2] < 2:
        return None
    from ltx_trainer.group_action.losses import identity_loss

    z0 = z[:, :, 0]
    z1 = z[:, :, -1]
    return cfg.lambda_group_action_temporal * identity_loss(z1, z0)


def apply_av_fold_training(
    per_sample_loss: Tensor,
    batch: dict[str, Any],
    *,
    cfg: "AvFoldTrainingConfig",
    with_audio: bool,
) -> tuple[Tensor, dict[str, float]]:
    """Apply quality weights and optional regularizers to ``[B,]`` diffusion loss."""
    metrics: dict[str, float] = {}
    loss = per_sample_loss

    v_w = video_quality_weights(batch, cfg)
    if v_w is not None:
        loss = loss * v_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/vsr_weight_mean"] = float(v_w.mean().item())

    erp_w = erp_boiqa_quality_weights(batch, cfg)
    if erp_w is not None:
        loss = loss * erp_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/erp_boiqa_weight_mean"] = float(erp_w.mean().item())

    s3_w = s3po_wss_quality_weights(batch, cfg)
    if s3_w is not None:
        loss = loss * s3_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/s3po_wss_weight_mean"] = float(s3_w.mean().item())

    lh_w = lucky_hdr_merge_weights(batch, cfg)
    if lh_w is not None:
        loss = loss * lh_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/lucky_hdr_weight_mean"] = float(lh_w.mean().item())

    ed_w = era_defocus_sharpness_weights(batch, cfg)
    if ed_w is not None:
        loss = loss * ed_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/era_defocus_weight_mean"] = float(ed_w.mean().item())

    fp_w = flood_physics_hydrology_weights(batch, cfg)
    if fp_w is not None:
        loss = loss * fp_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/flood_physics_weight_mean"] = float(fp_w.mean().item())

    dsd_w = dilated_sym_diff_change_weights(batch, cfg)
    if dsd_w is not None:
        loss = loss * dsd_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/dilated_sym_diff_weight_mean"] = float(dsd_w.mean().item())

    dgp_w = dynamic_gp_spatiotemporal_weights(batch, cfg)
    if dgp_w is not None:
        loss = loss * dgp_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/dynamic_gp_weight_mean"] = float(dgp_w.mean().item())

    be_w = branch_energy_localization_weights(batch, cfg)
    if be_w is not None:
        loss = loss * be_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/branch_energy_weight_mean"] = float(be_w.mean().item())

    lha_w = lora_hd_attn_rank_one_weights(batch, cfg)
    if lha_w is not None:
        loss = loss * lha_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/lora_hd_attn_weight_mean"] = float(lha_w.mean().item())

    ld_w = latenthdr_exposure_weights(batch, cfg)
    if ld_w is not None:
        loss = loss * ld_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/latenthdr_weight_mean"] = float(ld_w.mean().item())

    ff_fusion = fuse_flow_fusion_weights(batch, cfg)
    if ff_fusion is not None:
        loss = loss * ff_fusion.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/fuse_flow_fusion_weight_mean"] = float(ff_fusion.mean().item())

    ff_geom = fuse_flow_geometry_weights(batch, cfg)
    if ff_geom is not None:
        loss = loss * ff_geom.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/fuse_flow_geometry_weight_mean"] = float(ff_geom.mean().item())

    ft_w = forte_t2a_alignment_weights(batch, cfg)
    if ft_w is not None:
        loss = loss * ft_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/forte_t2a_weight_mean"] = float(ft_w.mean().item())

    for weight_fn, metric_key in (
        (sdr2hdr_merge_weights, "av_fold/sdr2hdr_weight_mean"),
        (x2hdr_alignment_weights, "av_fold/x2hdr_weight_mean"),
        (lf_diff_bracket_weights, "av_fold/lf_diff_weight_mean"),
        (diffhdr_log_gamma_weights, "av_fold/diffhdr_weight_mean"),
        (vdp_hdr_fusion_weights, "av_fold/vdp_hdr_weight_mean"),
        (semantic_stitch_seam_weights, "av_fold/semantic_stitch_weight_mean"),
        (pantheon360_cache_weights, "av_fold/pantheon360_weight_mean"),
        (mirage_readout_weights, "av_fold/mirage_weight_mean"),
        (sphere360_timelapse_weights, "av_fold/sphere360_weight_mean"),
        (panoworld_panospace_weights, "av_fold/panoworld_weight_mean"),
        (gimbal360_canonical_weights, "av_fold/gimbal360_weight_mean"),
        (panoenv_spatial_vqa_weights, "av_fold/panoenv_weight_mean"),
        (weatherproof_robustness_weights, "av_fold/weatherproof_weight_mean"),
        (dense360_omni_vlm_weights, "av_fold/dense360_weight_mean"),
        (cross360_depth_fusion_weights, "av_fold/cross360_weight_mean"),
        (anything360_conditioning_weights, "av_fold/anything360_weight_mean"),
        (spherefusion_depth_weights, "av_fold/spherefusion_weight_mean"),
        (raim_mef_fusion_weights, "av_fold/raim_mef_weight_mean"),
        (lumivid_logc3_weights, "av_fold/lumivid_weight_mean"),
        (stem2_isotonic_weights, "av_fold/stem2_sdr_hdr_weight_mean"),
        (physthdr_gs_training_weights, "av_fold/physthdr_gs_weight_mean"),
        (modulo_spike_unwrap_weights, "av_fold/modulo_spike_hdr_weight_mean"),
    ):
        w = weight_fn(batch, cfg)
        if w is not None:
            loss = loss * w.to(loss.device)
            if cfg.log_metrics:
                metrics[metric_key] = float(w.mean().item())

    for use_flag, fold_key, field, min_attr, metric_key in _PANO360_READINESS_WEIGHTS:
        w = _cfg_readiness_weights(
            batch,
            cfg,
            use_flag=use_flag,
            fold_key=fold_key,
            field=field,
            min_weight_attr=min_attr,
        )
        if w is not None:
            loss = loss * w.to(loss.device)
            if cfg.log_metrics:
                metrics[metric_key] = float(w.mean().item())

    for use_flag, fold_key, field, min_attr, metric_key, is_audio in _AUDIO_READINESS_WEIGHTS:
        w = _cfg_readiness_weights(
            batch,
            cfg,
            use_flag=use_flag,
            fold_key=fold_key,
            field=field,
            min_weight_attr=min_attr,
            audio=is_audio,
        )
        if w is not None:
            loss = loss * w.to(loss.device)
            if cfg.log_metrics:
                metrics[metric_key] = float(w.mean().item())

    p_w = physics_faithfulness_weights(batch, cfg)
    if p_w is not None:
        loss = loss * p_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/physics_weight_mean"] = float(p_w.mean().item())

    av_w = avbench_alignment_weights(batch, cfg)
    if av_w is not None:
        loss = loss * av_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/avbench_weight_mean"] = float(av_w.mean().item())

    mmae_w = mmae_instruction_weights(batch, cfg)
    if mmae_w is not None:
        loss = loss * mmae_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/mmae_weight_mean"] = float(mmae_w.mean().item())

    mt_w = mtavg2_expressiveness_weights(batch, cfg)
    if mt_w is not None:
        loss = loss * mt_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/mtavg2_weight_mean"] = float(mt_w.mean().item())

    if with_audio:
        gl_w = growloop_humanlikeness_weights(batch, cfg)
        if gl_w is not None:
            loss = loss * gl_w.to(loss.device)
            if cfg.log_metrics:
                metrics["av_fold/growloop_weight_mean"] = float(gl_w.mean().item())

    oc_v = omnicustom_video_weights(batch, cfg)
    if oc_v is not None:
        loss = loss * oc_v.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/omnicustom_video_weight_mean"] = float(oc_v.mean().item())

    id_v = id_lora_video_weights(batch, cfg)
    if id_v is not None:
        loss = loss * id_v.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/id_lora_video_weight_mean"] = float(id_v.mean().item())

    ac_v = autocut_video_weights(batch, cfg)
    if ac_v is not None:
        loss = loss * ac_v.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/autocut_video_weight_mean"] = float(ac_v.mean().item())

    sh_w = svhighlights_saliency_weights(batch, cfg)
    if sh_w is not None:
        loss = loss * sh_w.to(loss.device)
        if cfg.log_metrics:
            metrics["av_fold/svhighlights_weight_mean"] = float(sh_w.mean().item())

    for weight_fn, metric_key in (
        (core_kd_drift_weights, "av_fold/core_kd_weight_mean"),
        (entroad_anomaly_weights, "av_fold/entroad_weight_mean"),
    ):
        w = weight_fn(batch, cfg)
        if w is not None:
            loss = loss * w.to(loss.device)
            if cfg.log_metrics:
                metrics[metric_key] = float(w.mean().item())

    # Product of multiplicative fold weights (before aux regularizers).
    if cfg.log_metrics or cfg.normalize_weight_product:
        with torch.no_grad():
            eff = (loss.detach() / per_sample_loss.detach().clamp_min(1e-8)).float()
            prod_mean = float(eff.mean().item())
        if cfg.log_metrics:
            metrics["av_fold/weight_product_mean"] = prod_mean
        if (
            cfg.normalize_weight_product
            and prod_mean > 0.0
            and prod_mean < float(cfg.min_weight_product)
        ):
            scale = float(cfg.min_weight_product) / prod_mean
            loss = loss * scale
            if cfg.log_metrics:
                metrics["av_fold/weight_product_scale"] = scale
                metrics["av_fold/weight_product_mean"] = float(cfg.min_weight_product)

    proxy = _collated_scalar_field(batch, fold_key="phyworld", field="physics_proxy")
    if proxy is not None and cfg.lambda_phyworld_dpo > 0:
        from ltx_trainer.phyworld.training import physics_ranking_preference_loss

        dpo = physics_ranking_preference_loss(
            loss,
            proxy.to(loss.device),
            beta=cfg.phyworld_dpo_beta,
        )
        if dpo is not None:
            loss = loss + cfg.lambda_phyworld_dpo * dpo
            if cfg.log_metrics:
                metrics["av_fold/phyworld_dpo"] = float(dpo.item())

    if proxy is not None and cfg.lambda_phyworld_proxy_target > 0:
        from ltx_trainer.phyworld.training import physics_proxy_regularizer

        reg = physics_proxy_regularizer(proxy.to(loss.device))
        loss = loss + cfg.lambda_phyworld_proxy_target * reg
        if cfg.log_metrics:
            metrics["av_fold/phyworld_proxy_reg"] = float(reg.item())

    if cfg.lambda_fuse_flow_geometry > 0:
        from ltx_trainer.fuse_flow.training import fuse_flow_geometry_regularizer

        ff_reg = fuse_flow_geometry_regularizer(batch, cfg)
        if ff_reg is not None:
            loss = loss + ff_reg.to(loss.device)
            if cfg.log_metrics:
                metrics["av_fold/fuse_flow_geometry_reg"] = float(ff_reg.item())

    drift = _collated_scalar_field(batch, fold_key="lamo", field="motion_drift_norm")
    if drift is not None:
        if cfg.log_lamo_motion_drift and cfg.log_metrics:
            metrics["av_fold/lamo_motion_drift_mean"] = float(drift.mean().item())
        if cfg.lambda_lamo_drift > 0:
            lamo_term = cfg.lambda_lamo_drift * drift.to(loss.device).mean()
            loss = loss + lamo_term
            if cfg.log_metrics:
                metrics["av_fold/lamo_drift_loss"] = float(lamo_term.item())

    if cfg.log_bernini_segments and cfg.log_metrics:
        segs = _collated_scalar_field(batch, fold_key="bernini", field="num_segments")
        if segs is not None:
            metrics["av_fold/bernini_num_segments_mean"] = float(segs.mean().item())

    if cfg.log_adamag_guidance_norm and cfg.log_metrics:
        gn = _collated_scalar_field(batch, fold_key="adamag", field="adamag_norm")
        if gn is not None:
            metrics["av_fold/adamag_norm_mean"] = float(gn.mean().item())

    if cfg.log_research_fold_metrics and cfg.log_metrics:
        for fold_key, field, metric in (
            ("core_kd", "state_drift_proxy", "av_fold/core_kd_drift_mean"),
            ("vlm_count", "latent_density_proxy", "av_fold/vlm_count_density_mean"),
            ("entroad", "localized_entropy_spike_proxy", "av_fold/entroad_spike_mean"),
            ("flatsounds", "onset_strength_proxy", "av_fold/flatsounds_onset_mean"),
            ("mtavg2", "expressiveness_risk_proxy", "av_fold/mtavg2_risk_mean"),
            ("fuse_flow", "fusion_readiness_proxy", "av_fold/fuse_flow_fusion_ready_mean"),
            ("fuse_flow", "geometry_stability_proxy", "av_fold/fuse_flow_geometry_mean"),
            ("omnicustom", "identity_proxy", "av_fold/omnicustom_identity_mean"),
            ("id_lora", "identity_proxy", "av_fold/id_lora_identity_mean"),
            ("autocut", "narrative_proxy", "av_fold/autocut_narrative_mean"),
            ("vuga", "mos_proxy", "av_fold/vuga_mos_mean"),
            ("vuboiqa", "quality_proxy", "av_fold/vuboiqa_quality_mean"),
            ("s3po", "wss_quality_proxy", "av_fold/s3po_wss_mean"),
            ("lucky_hdr", "merge_readiness_proxy", "av_fold/lucky_hdr_ready_mean"),
            ("era_defocus", "deblur_readiness_proxy", "av_fold/era_defocus_ready_mean"),
            ("era_defocus", "sharpness_proxy", "av_fold/era_defocus_sharpness_mean"),
            ("flood_physics", "inundation_readiness_proxy", "av_fold/flood_physics_ready_mean"),
            ("flood_physics", "physics_consistency_proxy", "av_fold/flood_physics_consistency_mean"),
            ("dilated_sym_diff", "change_detection_readiness_proxy", "av_fold/dilated_sym_diff_ready_mean"),
            ("dilated_sym_diff", "boundary_contrast_proxy", "av_fold/dilated_sym_diff_boundary_mean"),
            ("dynamic_gp", "filter_readiness_proxy", "av_fold/dynamic_gp_ready_mean"),
            ("dynamic_gp", "basis_fit_proxy", "av_fold/dynamic_gp_basis_mean"),
            ("dynamic_gp", "temporal_smoothness_proxy", "av_fold/dynamic_gp_temporal_mean"),
            ("branch_energy", "localization_readiness_proxy", "av_fold/branch_energy_ready_mean"),
            ("branch_energy", "tellegen_balance_proxy", "av_fold/branch_energy_balance_mean"),
            ("branch_energy", "harmonic_spread_proxy", "av_fold/branch_energy_harmonic_mean"),
            ("lora_hd_attn", "lora_readiness_proxy", "av_fold/lora_hd_attn_ready_mean"),
            ("lora_hd_attn", "rank_one_alignment_proxy", "av_fold/lora_hd_attn_align_mean"),
            ("lora_hd_attn", "effective_noise_proxy", "av_fold/lora_hd_attn_noise_mean"),
            ("latenthdr", "exposure_readiness_proxy", "av_fold/latenthdr_ready_mean"),
            ("sdr2hdr", "merge_readiness_proxy", "av_fold/sdr2hdr_ready_mean"),
            ("x2hdr", "pu21_alignment_readiness", "av_fold/x2hdr_ready_mean"),
            ("lf_diff", "bracket_readiness", "av_fold/lf_diff_ready_mean"),
            ("diffhdr", "log_gamma_readiness", "av_fold/diffhdr_ready_mean"),
            ("vdp_hdr", "fusion_readiness_proxy", "av_fold/vdp_hdr_ready_mean"),
            ("pantheon360", "cache_fusion_readiness", "av_fold/pantheon360_ready_mean"),
            ("mirage", "readout_coverage", "av_fold/mirage_ready_mean"),
            ("spherefusion", "depth_readiness", "av_fold/spherefusion_ready_mean"),
            ("dense360", "omni_vlm_readiness", "av_fold/dense360_ready_mean"),
            ("cross360", "depth_fusion_readiness", "av_fold/cross360_ready_mean"),
            ("anything360", "conditioning_readiness", "av_fold/anything360_ready_mean"),
            ("raim_mef", "mef_fusion_readiness", "av_fold/raim_mef_ready_mean"),
            ("lumivid", "logc3_alignment_readiness", "av_fold/lumivid_ready_mean"),
            ("stem2_sdr_hdr", "isotonic_readiness_proxy", "av_fold/stem2_ready_mean"),
            ("physthdr_gs", "gs_training_readiness", "av_fold/physthdr_gs_ready_mean"),
            ("modulo_spike_hdr", "unwrap_readiness_proxy", "av_fold/modulo_spike_ready_mean"),
            ("cubediff", "panorama_readiness", "av_fold/cubediff_ready_mean"),
            ("spherediff", "spherical_readiness", "av_fold/spherediff_ready_mean"),
            ("erpgs", "splat_readiness", "av_fold/erpgs_ready_mean"),
            ("sphereuformer", "perception_readiness", "av_fold/sphereuformer_ready_mean"),
            ("sphere_depth", "calibration_readiness", "av_fold/sphere_depth_ready_mean"),
            ("nvc_erp_qpa", "qpa_readiness", "av_fold/nvc_erp_qpa_ready_mean"),
            ("mtpano", "multitask_dense_readiness", "av_fold/mtpano_ready_mean"),
            ("panoworld_x", "explorable_world_readiness", "av_fold/panoworld_x_ready_mean"),
            ("pano_affordance", "affordance_readiness", "av_fold/pano_affordance_ready_mean"),
            ("panogsdet", "gs_detection_readiness", "av_fold/panogsdet_ready_mean"),
            ("panolm", "panovqa_readiness", "av_fold/panolm_ready_mean"),
            ("pano_flight", "survey_task_readiness", "av_fold/pano_flight_ready_mean"),
            ("pano360", "photogrammetry_readiness", "av_fold/pano360_ready_mean"),
            ("holitok", "holistic_readiness_proxy", "av_fold/holitok_ready_mean"),
            ("sphere360", "timelapse_readiness", "av_fold/sphere360_ready_mean"),
            ("panoworld", "panospace_readiness", "av_fold/panoworld_ready_mean"),
            ("gimbal360", "canonical_erp_readiness", "av_fold/gimbal360_ready_mean"),
            ("panoenv", "spatial_vqa_readiness", "av_fold/panoenv_ready_mean"),
            ("weatherproof", "robustness_readiness", "av_fold/weatherproof_ready_mean"),
        ):
            val = _collated_scalar_field(batch, fold_key=fold_key, field=field)
            if val is not None:
                metrics[metric] = float(val.mean().item())

    ga = group_action_temporal_loss(batch, cfg)
    if ga is not None:
        loss = loss + ga
        if cfg.log_metrics:
            metrics["av_fold/group_action_temporal"] = float(ga.item())

    if with_audio:
        a_w = _audio_stability_weights(batch, cfg)
        if a_w is not None:
            loss = loss * a_w.to(loss.device)
            if cfg.log_metrics:
                metrics["av_fold/audio_weight_mean"] = float(a_w.mean().item())

        for weight_fn, metric_key in (
            (eigenet_reverb_weights, "av_fold/eigenet_weight_mean"),
            (planaudio_composition_weights, "av_fold/planaudio_weight_mean"),
            (ag_repa_causal_weights, "av_fold/ag_repa_weight_mean"),
            (forte_audio_alignment_weights, "av_fold/forte_audio_weight_mean"),
            (omnicustom_timbre_weights, "av_fold/omnicustom_timbre_weight_mean"),
            (id_lora_audio_weights, "av_fold/id_lora_audio_weight_mean"),
            (autocut_bgm_weights, "av_fold/autocut_bgm_weight_mean"),
        ):
            w = weight_fn(batch, cfg)
            if w is not None:
                loss = loss * w.to(loss.device)
                if cfg.log_metrics:
                    metrics[metric_key] = float(w.mean().item())

        if cfg.log_research_fold_metrics and cfg.log_metrics:
            decay = _collated_scalar_field(batch, fold_key="eigenet", field="decay_ratio_proxy", audio=True)
            if decay is not None:
                metrics["av_fold/eigenet_decay_mean"] = float(decay.mean().item())
            comp = _collated_scalar_field(
                batch, fold_key="planaudio", field="composite_complexity_proxy", audio=True
            )
            if comp is not None:
                metrics["av_fold/planaudio_complexity_mean"] = float(comp.mean().item())
            fog = _collated_scalar_field(batch, fold_key="ag_repa", field="fog_a_proxy", audio=True)
            if fog is not None:
                metrics["av_fold/ag_repa_fog_mean"] = float(fog.mean().item())
            forte_a = _collated_scalar_field(batch, fold_key="forte_audio", field="t2a_align_proxy", audio=True)
            if forte_a is not None:
                metrics["av_fold/forte_audio_align_mean"] = float(forte_a.mean().item())
            forte_v = _collated_scalar_field(batch, fold_key="forte", field="t2a_align_proxy")
            if forte_v is not None:
                metrics["av_fold/forte_t2a_align_mean"] = float(forte_v.mean().item())
            timbre = _collated_scalar_field(
                batch, fold_key="omnicustom", field="timbre_proxy", audio=True
            )
            if timbre is not None:
                metrics["av_fold/omnicustom_timbre_mean"] = float(timbre.mean().item())
            speaker = _collated_scalar_field(
                batch, fold_key="id_lora", field="speaker_identity_proxy", audio=True
            )
            if speaker is not None:
                metrics["av_fold/id_lora_speaker_mean"] = float(speaker.mean().item())
            bgm = _collated_scalar_field(batch, fold_key="autocut", field="bgm_proxy", audio=True)
            if bgm is not None:
                metrics["av_fold/autocut_bgm_mean"] = float(bgm.mean().item())

        coding = fmelcodec_coding_regularizer(batch, cfg)
        if coding is not None:
            loss = loss + cfg.lambda_fmelcodec_coding * coding.to(loss.device)
            if cfg.log_metrics:
                metrics["av_fold/fmelcodec_coding"] = float(coding.item())

        spectral = wavenext2_spectral_regularizer(batch, cfg)
        if spectral is not None:
            loss = loss + cfg.lambda_wavenext2_spectral * spectral.to(loss.device)
            if cfg.log_metrics:
                metrics["av_fold/wavenext2_spectral"] = float(spectral.item())

    return loss, metrics


def sidecar_coverage_summary(batch: dict[str, Any]) -> dict[str, bool]:
    """Which fold keys are present in this collated batch (for logging)."""
    from ltx_trainer.fold_registry import expected_audio_fold_keys, expected_video_fold_keys

    out: dict[str, bool] = {}
    lat = _latent_shard_batch(batch)
    if lat is not None:
        for key in expected_video_fold_keys():
            out[key] = key in lat
    aud = _audio_shard_batch(batch)
    if aud is not None:
        for key in expected_audio_fold_keys():
            out[key] = key in aud
    return out
