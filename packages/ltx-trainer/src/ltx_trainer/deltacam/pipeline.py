"""DeltaCam framework card, paper tables, and smoke demos (arXiv:2605.25266)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.deltacam.ccm import CameraConditioningModule
from ltx_trainer.deltacam.config import DeltaCamConfig
from ltx_trainer.deltacam.delta_params import dual_effect_ramp, single_effect_ramp, trajectory_to_delta
from ltx_trainer.deltacam.effect_attention import EffectAttentionBlock, concat_delta_plucker_cond
from ltx_trainer.deltacam.exif import ExifMetadataTokenizer, normalize_metadata_vector
from ltx_trainer.deltacam.integration import encode_style_trajectory
from ltx_trainer.deltacam.layout import conditioning_layout, paper_limitations
from ltx_trainer.deltacam.adaptors import StyleToDeltaAdaptor
from ltx_trainer.deltacam.metrics import table_wclip_sensitivity, wclip
from ltx_trainer.deltacam.plucker import plucker_map_pinhole, zeros_plucker_map
from ltx_trainer.deltacam.proxies import SceneProxyMaps, assert_proxy_shapes, mask_proxy_streams
from ltx_trainer.deltacam.temporal import TemporalStyleEncoder
from ltx_trainer.deltacam.style import (
    DisentangledStyleEmbedder,
    StyleTrajectoryHead,
    style_extraction_loss,
)


def framework_card(cfg: DeltaCamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeltaCamConfig()
    return {
        "name": "DeltaCam",
        "paper": cfg.paper_arxiv,
        "backbone": cfg.backbone,
        "task": "Video-to-video with Δ-parameterized intrinsic camera control",
        "control_modes": ["sliders", "style_embedding", "exif_metadata"],
        "components": [
            "Camera Conditioning Module (CCM)",
            "Δ-parameterized FiLM (optical / sensory / ISP)",
            "Scene proxy maps G (depth / flow / perspective)",
            "Plücker ray extrinsics P",
            "Disentangled style extractor + style→Δ adaptor",
            "TemporalStyleEncoder (shared over frames)",
            "EXIF metadata tokenizer (stage 3)",
            "EffectAttentionBlock (stage-1 latent cond stub; full DiT blocks external)",
        ],
        "training_stages": [
            "Stage 1: CCM + effect attention (synthetic)",
            "Stage 2: style transfer adaptor",
            "Stage 3: real EXIF matching (<0.1% params)",
        ],
        "datasets": ["RealCamVid", "MultiCam", "RealBokeh", "NeuralCam"],
    }


def dataset_manifest() -> list[dict[str, Any]]:
    """Paper datasets with stage hints (paths are placeholders for a real dataloader)."""
    root = "${GOPEX_DELTACAM_DATA}"
    return [
        {
            "name": "RealCamVid",
            "stages": [1],
            "role": "synthetic intrinsic trajectories + proxy supervision",
            "path_placeholder": f"{root}/realcamvid",
        },
        {
            "name": "MultiCam",
            "stages": [1],
            "role": "multi-camera extrinsics + intrinsics",
            "path_placeholder": f"{root}/multicam",
        },
        {
            "name": "RealBokeh",
            "stages": [2],
            "role": "style / bokeh transfer pairs",
            "path_placeholder": f"{root}/realbokeh",
        },
        {
            "name": "NeuralCam",
            "stages": [2, 3],
            "role": "EXIF-aligned real video",
            "path_placeholder": f"{root}/neuralcam",
        },
    ]


def camera_intrinsic_vocabulary(cfg: DeltaCamConfig | None = None) -> list[str]:
    """Ordered intrinsic keys for prompts and UI (matches ``cfg.ranges``)."""
    cfg = cfg or DeltaCamConfig()
    return list(cfg.ranges.keys())


def intrinsic_groups(cfg: DeltaCamConfig | None = None) -> dict[str, list[str]]:
    """FiLM groups from ``DeltaCamConfig`` (Fig. 5) — only keys present in ``ranges``."""
    cfg = cfg or DeltaCamConfig()
    keys = set(cfg.ranges)

    def _filt(names: tuple[str, ...]) -> list[str]:
        return [k for k in names if k in keys]

    return {
        "optical": _filt(cfg.optical_params),
        "sensory": _filt(cfg.sensory_params),
        "isp": _filt(cfg.isp_params),
    }


def intrinsic_param_group(param: str, cfg: DeltaCamConfig | None = None) -> str:
    """Return ``"optical"``, ``"sensory"``, or ``"isp"`` for a range key."""
    cfg = cfg or DeltaCamConfig()
    if param not in cfg.ranges:
        raise KeyError(f"Unknown intrinsic {param!r}")
    if param in cfg.optical_params:
        return "optical"
    if param in cfg.sensory_params:
        return "sensory"
    if param in cfg.isp_params:
        return "isp"
    raise KeyError(f"{param!r} not assigned to optical/sensory/isp tuples")


def style_extraction_loss_weights(cfg: DeltaCamConfig | None = None) -> dict[str, float]:
    """Eq. (4) style-path λ weights (trajectory, content, style, MI)."""
    cfg = cfg or DeltaCamConfig()
    return {
        "lambda_tau": cfg.lambda_tau,
        "lambda_content": cfg.lambda_content,
        "lambda_style": cfg.lambda_style,
        "lambda_mi": cfg.lambda_mi,
    }


def curriculum_stages(cfg: DeltaCamConfig | None = None) -> list[dict[str, Any]]:
    """Three-stage schedule from ``DeltaCamConfig`` (Sec. 3.5 — iteration counts only)."""
    cfg = cfg or DeltaCamConfig()
    return [
        {
            "stage": 1,
            "focus": "CCM + effect attention (synthetic proxies)",
            "iterations": cfg.stage1_iterations,
        },
        {
            "stage": 2,
            "focus": "style→Δ adaptor + disentangled style extraction",
            "iterations": cfg.stage2_iterations,
        },
        {
            "stage": 3,
            "focus": "EXIF tokenizer alignment (paper: <0.1% trainable params)",
            "iterations": cfg.stage3_iterations,
            "trainable_param_fraction_cap": cfg.stage3_trainable_param_fraction,
        },
    ]


def table_related_work() -> dict[str, dict[str, str]]:
    """Table 1 — comparison vs prior work (✓ / partial / ✗ as ``full`` / ``partial`` / ``no``)."""
    return {
        "GenPhoto": {
            "video_to_video": "no",
            "extrinsic_control": "no",
            "intrinsic_control": "full",
            "precise_control": "partial",
            "smooth_control": "partial",
            "style_mixing": "partial",
            "style_transfer": "no",
            "camera_matching": "no",
        },
        "AC3D": {
            "video_to_video": "no",
            "extrinsic_control": "full",
            "intrinsic_control": "no",
            "precise_control": "no",
            "smooth_control": "partial",
            "style_mixing": "no",
            "style_transfer": "no",
            "camera_matching": "no",
        },
        "Akira": {
            "video_to_video": "no",
            "extrinsic_control": "full",
            "intrinsic_control": "partial",
            "precise_control": "partial",
            "smooth_control": "full",
            "style_mixing": "no",
            "style_transfer": "no",
            "camera_matching": "no",
        },
        "VACE": {
            "video_to_video": "partial",
            "extrinsic_control": "no",
            "intrinsic_control": "no",
            "precise_control": "partial",
            "smooth_control": "partial",
            "style_mixing": "no",
            "style_transfer": "partial",
            "camera_matching": "no",
        },
        "ReCamMaster": {
            "video_to_video": "full",
            "extrinsic_control": "full",
            "intrinsic_control": "no",
            "precise_control": "partial",
            "smooth_control": "full",
            "style_mixing": "no",
            "style_transfer": "no",
            "camera_matching": "no",
        },
        "Ours": {
            "video_to_video": "full",
            "extrinsic_control": "full",
            "intrinsic_control": "full",
            "precise_control": "full",
            "smooth_control": "full",
            "style_mixing": "full",
            "style_transfer": "full",
            "camera_matching": "full",
        },
    }


def table_single_effect_control() -> dict[str, dict[str, dict[str, float]]]:
    """Table 2 — single-effect control + VBench temporal / smooth (paper columns)."""
    return {
        "Bokeh": {
            "VACE": {"psnr": 19.86, "ssim": 0.74, "lpips": 0.36, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.89},
            "CogVideoX": {"psnr": 21.27, "ssim": 0.68, "lpips": 0.32, "vbench_temp": 0.96, "vbench_smooth": 0.99, "wclip5": 0.92},
            "Ours": {"psnr": 25.21, "ssim": 0.79, "lpips": 0.29, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.94},
        },
        "Motion Blur": {
            "VACE": {"psnr": 20.49, "ssim": 0.73, "lpips": 0.41, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.89},
            "CogVideoX": {"psnr": 21.70, "ssim": 0.72, "lpips": 0.33, "vbench_temp": 0.96, "vbench_smooth": 0.99, "wclip5": 0.92},
            "Ours": {"psnr": 20.82, "ssim": 0.74, "lpips": 0.31, "vbench_temp": 0.97, "vbench_smooth": 0.99, "wclip5": 0.94},
        },
        "Lens Dist.": {
            "VACE": {"psnr": 11.55, "ssim": 0.37, "lpips": 0.69, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.83},
            "CogVideoX": {"psnr": 12.27, "ssim": 0.40, "lpips": 0.62, "vbench_temp": 0.96, "vbench_smooth": 0.98, "wclip5": 0.88},
            "Ours": {"psnr": 13.92, "ssim": 0.44, "lpips": 0.49, "vbench_temp": 0.98, "vbench_smooth": 0.98, "wclip5": 0.91},
        },
        "Exp. Time": {
            "VACE": {"psnr": 15.35, "ssim": 0.75, "lpips": 0.30, "vbench_temp": 0.98, "vbench_smooth": 0.98, "wclip5": 0.90},
            "CogVideoX": {"psnr": 16.14, "ssim": 0.65, "lpips": 0.29, "vbench_temp": 0.96, "vbench_smooth": 0.99, "wclip5": 0.94},
            "Ours": {"psnr": 21.08, "ssim": 0.74, "lpips": 0.22, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.96},
        },
        "Color Temp.": {
            "VACE": {"psnr": 12.42, "ssim": 0.55, "lpips": 0.50, "vbench_temp": 0.98, "vbench_smooth": 0.99, "wclip5": 0.84},
            "CogVideoX": {"psnr": 19.30, "ssim": 0.71, "lpips": 0.26, "vbench_temp": 0.97, "vbench_smooth": 0.99, "wclip5": 0.95},
            "Ours": {"psnr": 23.51, "ssim": 0.90, "lpips": 0.11, "vbench_temp": 0.98, "vbench_smooth": 0.98, "wclip5": 0.98},
        },
        "Focal Len.": {
            "VACE": {"psnr": 11.14, "ssim": 0.40, "lpips": 0.67, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.84},
            "CogVideoX": {"psnr": 11.61, "ssim": 0.43, "lpips": 0.62, "vbench_temp": 0.96, "vbench_smooth": 0.99, "wclip5": 0.87},
            "Ours": {"psnr": 16.51, "ssim": 0.55, "lpips": 0.33, "vbench_temp": 0.96, "vbench_smooth": 0.98, "wclip5": 0.96},
        },
        "Average": {
            "VACE": {"psnr": 15.13, "ssim": 0.59, "lpips": 0.49, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.87},
            "CogVideoX": {"psnr": 17.05, "ssim": 0.60, "lpips": 0.41, "vbench_temp": 0.96, "vbench_smooth": 0.99, "wclip5": 0.91},
            "Ours": {"psnr": 20.18, "ssim": 0.69, "lpips": 0.29, "vbench_temp": 0.97, "vbench_smooth": 0.98, "wclip5": 0.95},
        },
    }


def table_style_extraction() -> dict[str, dict[str, float]]:
    """Table 3 — per-effect trajectory NCC, style InfoNCE, and Valid Frac. (paper)."""
    return {
        "Focal length": {
            "B1_ncc": 0.32,
            "B2_ncc": 0.75,
            "ours_ncc": 0.84,
            "style_infonce": 0.38,
            "valid_frac": 0.90,
        },
        "Lens distortion": {
            "B1_ncc": 0.22,
            "B2_ncc": 0.25,
            "ours_ncc": 0.45,
            "style_infonce": 0.47,
            "valid_frac": 1.00,
        },
        "Bokeh": {
            "B1_ncc": 0.27,
            "B2_ncc": 0.35,
            "ours_ncc": 0.82,
            "style_infonce": 2.84,
            "valid_frac": 0.76,
        },
        "Exposure": {
            "B1_ncc": 0.77,
            "B2_ncc": 0.80,
            "ours_ncc": 0.89,
            "style_infonce": 0.03,
            "valid_frac": 0.80,
        },
        "Motion blur": {
            "B1_ncc": 0.09,
            "B2_ncc": 0.25,
            "ours_ncc": 0.61,
            "style_infonce": 2.93,
            "valid_frac": 0.93,
        },
        "Color temp.": {
            "B1_ncc": 0.78,
            "B2_ncc": 0.83,
            "ours_ncc": 0.90,
            "style_infonce": 0.03,
            "valid_frac": 0.79,
        },
        "Bokeh focus dist.": {
            "B1_ncc": 0.13,
            "B2_ncc": 0.20,
            "ours_ncc": 0.43,
            "style_infonce": 0.82,
            # Paper Table 3 omits Valid Frac. for this row; placeholder for tooling parity.
            "valid_frac": 0.79,
        },
    }


def table_ablation() -> dict[str, dict[str, float]]:
    """Table 4 — component and parameterization ablations."""
    return {
        "Full model (all streams)": {"psnr": 22.65, "ssim": 0.875, "lpips": 0.204},
        "w/o Depth stream": {"psnr": 22.50, "ssim": 0.875, "lpips": 0.201},
        "w/o Optical flow": {"psnr": 22.68, "ssim": 0.876, "lpips": 0.201},
        "w/o Perspective field": {"psnr": 22.63, "ssim": 0.875, "lpips": 0.202},
        "w/o Source RGB": {"psnr": 22.63, "ssim": 0.873, "lpips": 0.208},
        "Absolute (no Δ)": {"psnr": 11.88, "ssim": 0.360, "lpips": 0.738},
        "Delta (normalized)": {"psnr": 23.29, "ssim": 0.962, "lpips": 0.103},
        "Multi: Bokeh + Exposure": {"psnr": 21.57, "ssim": 0.854, "lpips": 0.274},
    }


def training_step_demo(cfg: DeltaCamConfig | None = None) -> dict[str, float]:
    """Smoke: Δ trajectory → CCM FiLM → style loss → EXIF → wCLIP → geometry stubs."""
    cfg = cfg or DeltaCamConfig()
    torch.manual_seed(25266)
    t_len = 16
    c, h, w = 32, 8, 8
    num_params = len(cfg.ranges)

    _ = single_effect_ramp("aperture_f", start=8.0, end=1.4, num_frames=t_len, cfg=cfg)
    delta = trajectory_to_delta(
        torch.stack(
            [
                torch.tensor([cfg.ranges[k].m_max - i * 0.5 for i, k in enumerate(cfg.ranges)]).float()
                for _ in range(t_len)
            ]
        ),
        cfg,
    )
    h_feat = torch.randn(c, h, w)
    ccm = CameraConditioningModule(c, cfg)
    modulated = ccm(h_feat, delta[0])

    embed = DisentangledStyleEmbedder(in_dim=64, style_dim=cfg.style_dim, content_dim=cfg.style_dim)
    head = StyleTrajectoryHead(cfg.style_dim, num_params)
    x = torch.randn(4, 64)
    z_c, z_s = embed(x)
    tau_pred = head(z_s)
    tau_gt = torch.randn_like(tau_pred)
    losses = style_extraction_loss(
        tau_pred,
        tau_gt,
        z_c,
        z_c.roll(1, 0),
        z_s,
        z_s.roll(1, 0),
        lambda_tau=cfg.lambda_tau,
        lambda_c=cfg.lambda_content,
        lambda_s=cfg.lambda_style,
        lambda_mi=cfg.lambda_mi,
    )

    meta = normalize_metadata_vector([50.0, 2.0, 0.1, 400.0, 0.0, 5500.0, 1.0 / 60.0], cfg)
    exif_tok = ExifMetadataTokenizer(num_meta=num_params, embed_dim=64)
    xi = exif_tok(meta)

    meta_traj = torch.stack([meta + 0.01 * torch.randn_like(meta) for _ in range(t_len)])
    xi_seq = exif_tok(meta_traj)
    assert xi_seq.shape == (t_len, 64)

    temporal = TemporalStyleEncoder(d_model=64, nhead=4, num_layers=1, dim_feedforward=256)
    tfeats = torch.randn(2, t_len, 64)
    emb_st = DisentangledStyleEmbedder(in_dim=64, style_dim=32, content_dim=32)
    head_st = StyleTrajectoryHead(32, num_params)
    tau_seq, _, _ = encode_style_trajectory(
        tfeats, temporal=temporal, embedder=emb_st, head=head_st
    )
    assert tau_seq.shape == (2, t_len, num_params)

    dual_d = dual_effect_ramp(
        "aperture_f",
        "exposure_ev",
        start_a=8.0,
        end_a=2.0,
        start_b=-1.0,
        end_b=1.0,
        num_frames=t_len,
        cfg=cfg,
    )

    adaptor = StyleToDeltaAdaptor(cfg.style_dim, num_params, exif_dim=64)
    z_style_1 = torch.randn(cfg.style_dim)
    delta_hat = adaptor(z_style_1, xi.squeeze(0))
    assert delta_hat.shape == (1, num_params)

    k = torch.tensor([[500.0, 0.0, 256.0], [0.0, 500.0, 256.0], [0.0, 0.0, 1.0]])
    c2w = torch.eye(4)
    plk = plucker_map_pinhole(k, c2w, h, w)
    plk_zero = zeros_plucker_map(h, w)
    cond_eff = concat_delta_plucker_cond(delta[0], plk)
    z_lat = torch.randn(1, c, h, w)
    effect_attn = EffectAttentionBlock(c, cond_dim=cond_eff.shape[-1])
    z_eff = effect_attn(z_lat, cond_eff)
    depth = torch.randn(t_len, h, w)
    flow = torch.randn(t_len, 2, h, w)
    maps = SceneProxyMaps(depth=depth, optical_flow=flow, source_rgb=torch.randn(t_len, 3, h, w))
    assert_proxy_shapes(maps, t=t_len, h=h, w=w)
    maps_masked = mask_proxy_streams(maps, {"depth"})

    feats = F.normalize(torch.randn(t_len, 32), dim=-1)
    w5 = wclip(feats, feats, window=cfg.wclip_window)

    return {
        "modulated_mean": float(modulated.mean().detach()),
        "style_loss": float(losses["total"].detach()),
        "exif_embed_norm": float(xi.norm().detach()),
        "wclip5_self": w5,
        "delta_aperture_end": float(delta[-1, list(cfg.ranges.keys()).index("aperture_f")]),
        "plucker_map_mean": float(plk.mean().detach()),
        "plucker_zero_sum": float(plk_zero.sum().detach()),
        "adaptor_delta_norm": float(delta_hat.norm().detach()),
        "proxy_depth_none": maps_masked.depth is None,
        "temporal_out_std": float(tau_seq.std().detach()),
        "style_trajectory_norm": float(tau_seq.norm().detach()),
        "dual_effect_delta_norm": float(dual_d.norm().detach()),
        "effect_attn_residual_l1": float((z_eff - z_lat).abs().mean().detach()),
    }


def evaluation_demo(cfg: DeltaCamConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DeltaCamConfig()
    step = training_step_demo(cfg)
    tab2 = table_single_effect_control()
    tab3 = table_style_extraction()
    tab4 = table_ablation()
    tw = table_wclip_sensitivity()
    rw = table_related_work()

    ours_avg = tab2["Average"]["Ours"]
    vace_avg = tab2["Average"]["VACE"]
    cog_avg = tab2["Average"]["CogVideoX"]
    stages = curriculum_stages(cfg)
    curriculum_total = sum(int(s["iterations"]) for s in stages)
    igroups = intrinsic_groups(cfg)

    return {
        **step,
        "ours_beats_vace_psnr": ours_avg["psnr"] > vace_avg["psnr"],
        "ours_beats_cog_psnr": ours_avg["psnr"] > cog_avg["psnr"],
        "delta_beats_absolute_psnr": tab4["Delta (normalized)"]["psnr"] > tab4["Absolute (no Δ)"]["psnr"],
        "color_temp_ours_ncc": tab3["Color temp."]["ours_ncc"],
        "bokeh_ours_ncc": tab3["Bokeh"]["ours_ncc"],
        "wclip5_ours_avg": ours_avg["wclip5"],
        "wclip5_table_ours": tw["Ours"]["wclip5"],
        "bokeh_focus_ncc": tab3["Bokeh focus dist."]["ours_ncc"],
        "table1_ours_unique_camera_matching": rw["Ours"]["camera_matching"] == "full"
        and rw["VACE"]["camera_matching"] == "no",
        "ours_vbench_temp_avg": ours_avg["vbench_temp"],
        "dataset_manifest_len": len(dataset_manifest()),
        "focal_valid_frac": tab3["Focal length"]["valid_frac"],
        "curriculum_total_iterations": curriculum_total,
        "curriculum_stage_count": len(stages),
        "style_loss_weights": style_extraction_loss_weights(cfg),
        "intrinsic_groups": igroups,
        "aperture_f_group": intrinsic_param_group("aperture_f", cfg),
        "limitations_count": len(paper_limitations()),
        "conditioning_keys": list(
            conditioning_layout(
                height=cfg.resolution[0], width=cfg.resolution[1], num_intrinsics=len(cfg.ranges)
            ).keys()
        ),
    }
