"""Paper tables, framework card, and adapter smoke (arXiv:2605.24769)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.hsir_rgb.config import HSIRRgbConfig
from ltx_trainer.hsir_rgb.encoders import EncoderMode, denoise_by_mode, denoise_sequential
from ltx_trainer.hsir_rgb.inner import awgn_inner_denoiser
from ltx_trainer.hsir_rgb.layout import architecture_layout, paper_limitations
from ltx_trainer.hsir_rgb.metrics import hs_psnr, hs_ssim_band_mean, spectral_angle_mapper
from ltx_trainer.hsir_rgb.operators import degrade
from ltx_trainer.hsir_rgb.pnp import pnp_hqs_restore
from ltx_trainer.hsir_rgb.projection import SpectralPnPAdapter, encoder_decoder_from_unconstrained
from ltx_trainer.hsir_rgb.stability import verify_qr_nonexpansive_identity_inner
from ltx_trainer.hsir_rgb.viz import psnr_on_viz_bands


def references_bibtex() -> str:
    """Primary citation for HSIR-RGB (stub BibTeX for docs / tooling)."""
    return (
        "@misc{picone2026hsirrgb,\n"
        "  title={Leveraging Pretrained RGB Denoisers for Hyperspectral Image Restoration},\n"
        "  author={Picone, Daniele and Jouni, Mohamad and Dalla Mura, Mauro},\n"
        "  year={2026},\n"
        "  eprint={2605.24769},\n"
        "  archivePrefix={arXiv},\n"
        "  primaryClass={eess.IV},\n"
        "}\n"
    )


def framework_card(cfg: HSIRRgbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HSIRRgbConfig()
    return {
        "name": "Pretrained RGB denoiser → hyperspectral restoration",
        "paper": cfg.paper_arxiv,
        "datasets": list(cfg.datasets),
        "bands": cfg.num_spectral_bands,
        "inner_denoiser": cfg.inner_denoiser_name,
        "projection": (
            f"K={cfg.num_projection_groups} groups × c={cfg.inner_channels}; "
            f"E matrix shape {cfg.latent_spectral_dim}×{cfg.num_spectral_bands} (QR orthonormal columns)"
        ),
        "training": "Only linear spectral adapter (QR E); inner denoiser frozen",
        "pnp": cfg.pnp_framework,
        "metrics": ["PSNR", "SAM (degrees)", "SSIM", "time/patch"],
        "ablation_modes": [m.value for m in EncoderMode],
    }


def table_related_work() -> dict[str, dict[str, str]]:
    """Qualitative comparison vs HS-specific and RGB-assisted baselines (Sec. 1)."""
    return {
        "DPHSIR": {
            "frozen_rgb_denoiser": "no",
            "hyperspectral_denoiser": "yes",
            "pnp_compatible": "yes",
            "projection_adapter": "no",
        },
        "HIR-Diff": {
            "frozen_rgb_denoiser": "no",
            "hyperspectral_denoiser": "diffusion",
            "pnp_compatible": "partial",
            "projection_adapter": "no",
        },
        "RGB-assisted diffusion": {
            "frozen_rgb_denoiser": "no",
            "hyperspectral_denoiser": "joint",
            "pnp_compatible": "no",
            "projection_adapter": "no",
        },
        "Deep HS Prior": {
            "frozen_rgb_denoiser": "no",
            "hyperspectral_denoiser": "dip",
            "pnp_compatible": "partial",
            "projection_adapter": "no",
        },
        "Ours": {
            "frozen_rgb_denoiser": "yes",
            "hyperspectral_denoiser": "adapter_lift",
            "pnp_compatible": "yes",
            "projection_adapter": "yes",
        },
    }


def task_protocol(cfg: HSIRRgbConfig | None = None) -> dict[str, dict[str, float | int | str]]:
    """Sec. 3.1 degradation settings for denoise / deblur / SISR."""
    cfg = cfg or HSIRRgbConfig()
    return {
        "denoise_sigma_010": {"operator": "identity", "noise_sigma": 0.10},
        "denoise_sigma_020": {"operator": "identity", "noise_sigma": 0.20},
        "deblur": {"operator": "gaussian_blur", "noise_sigma": 0.05, "kernel_size": 5},
        "superres_x4": {
            "operator": "downsample_x4",
            "noise_sigma": 0.005,
            "scale": 4,
        },
        "training_awgn_sigmas": list(cfg.train_noise_sigmas),
        "patch_size": cfg.patch_size,
    }


def harvard_untrained_identity_demo(cfg: HSIRRgbConfig | None = None) -> dict[str, float]:
    """Harvard-style untrained identity encoder (sequential RGB, no learned E)."""
    cfg = cfg or HSIRRgbConfig()
    torch.manual_seed(24769)
    x = torch.rand(1, cfg.num_spectral_bands, 16, 16)
    y, _ = degrade(x, "denoise", noise_sigma=0.10)
    inner = awgn_inner_denoiser(0.10)
    x_hat = denoise_sequential(y, inner, inner_c=cfg.inner_channels)
    return {
        "mse": float(torch.nn.functional.mse_loss(x_hat, x).item()),
        "psnr_full": hs_psnr(x_hat, x),
        "psnr_viz_bands": psnr_on_viz_bands(x_hat, x, cfg=cfg),
    }


def dataset_manifest(cfg: HSIRRgbConfig | None = None) -> list[dict[str, Any]]:
    """CAVE / Harvard protocol (Sec. 3.1)."""
    cfg = cfg or HSIRRgbConfig()
    root = "${GOPEX_HSIR_RGB_DATA}"
    tr, va, te = cfg.train_val_test_split
    return [
        {
            "name": "CAVE",
            "bands": cfg.num_spectral_bands,
            "patch_size": cfg.patch_size,
            "split": {"train": tr, "val": va, "test": te},
            "path_placeholder": f"{root}/cave",
        },
        {
            "name": "Harvard",
            "bands": cfg.num_spectral_bands,
            "patch_size": cfg.patch_size,
            "split": {"train": tr, "val": va, "test": te},
            "path_placeholder": f"{root}/harvard",
            "note": "Paper also reports untrained identity encoder on Harvard",
        },
    ]


def fig1_harvard_teaser() -> dict[str, Any]:
    """Fig. 1 — RGB vs HS denoiser PSNR on bands {24, 14, 4}, σ = 0.10 (paper caption)."""
    return {
        "dataset": "Harvard",
        "sigma": 0.10,
        "viz_bands_1based": [24, 14, 4],
        "reference_psnr_db": 20.00,
        "rgb_drunet_psnr_db": 30.42,
        "hs_grunet_psnr_db": 30.35,
        "inner_denoiser": "DRUNet [8] vs GRUNet [9]",
    }


def adapter_training_loss(
    x_hat: Tensor,
    x: Tensor,
) -> Tensor:
    """Eq. (6): ``‖F D(Ey) − x‖²`` (MSE surrogate)."""
    return F.mse_loss(x_hat, x)


def table_denoising() -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Table 1 — denoising (mean values; ± std omitted for tooling)."""
    return {
        "sigma_0.10": {
            "CAVE": {
                "SST": {"psnr": 36.35, "sam": 11.59, "ssim": 0.9607, "time_s": 0.7712},
                "SERT": {"psnr": 38.04, "sam": 8.73, "ssim": 0.9685, "time_s": 0.0655},
                "MAC-Net": {"psnr": 33.20, "sam": 26.37, "ssim": 0.8251, "time_s": 2.936},
                "T3SC": {"psnr": 34.16, "sam": 14.93, "ssim": 0.8860, "time_s": 0.1850},
                "GRUNet": {"psnr": 39.47, "sam": 9.28, "ssim": 0.9636, "time_s": 0.1942},
                "Deep HS Prior": {"psnr": 29.05, "sam": 13.28, "ssim": 0.8576, "time_s": 37.80},
                "Proposed": {"psnr": 44.61, "sam": 4.89, "ssim": 0.9805, "time_s": 0.2817},
            },
            "Harvard": {
                "SST": {"psnr": 28.63, "sam": 10.90, "ssim": 0.7527, "time_s": 0.7712},
                "SERT": {"psnr": 28.74, "sam": 10.38, "ssim": 0.7520, "time_s": 0.0655},
                "MAC-Net": {"psnr": 27.93, "sam": 13.78, "ssim": 0.7251, "time_s": 2.936},
                "T3SC": {"psnr": 28.07, "sam": 11.82, "ssim": 0.7279, "time_s": 0.1850},
                "GRUNet": {"psnr": 29.39, "sam": 10.09, "ssim": 0.7627, "time_s": 0.1942},
                "Deep HS Prior": {"psnr": 24.46, "sam": 12.67, "ssim": 0.5554, "time_s": 37.80},
                "Proposed": {"psnr": 30.51, "sam": 8.52, "ssim": 0.8219, "time_s": 0.2817},
            },
        },
        "sigma_0.20": {
            "CAVE": {
                "SST": {"psnr": 34.82, "sam": 14.53, "ssim": 0.9438, "time_s": 0.7637},
                "SERT": {"psnr": 36.39, "sam": 11.49, "ssim": 0.9530, "time_s": 0.0671},
                "MAC-Net": {"psnr": 27.77, "sam": 44.39, "ssim": 0.5911, "time_s": 3.708},
                "T3SC": {"psnr": 33.66, "sam": 15.71, "ssim": 0.8714, "time_s": 0.1826},
                "GRUNet": {"psnr": 37.17, "sam": 13.45, "ssim": 0.9395, "time_s": 0.1971},
                "Deep HS Prior": {"psnr": 28.48, "sam": 16.16, "ssim": 0.8302, "time_s": 37.79},
                "Proposed": {"psnr": 41.91, "sam": 6.32, "ssim": 0.9694, "time_s": 0.2846},
            },
            "Harvard": {
                "SST": {"psnr": 27.70, "sam": 11.44, "ssim": 0.7298, "time_s": 0.7637},
                "SERT": {"psnr": 27.83, "sam": 10.98, "ssim": 0.7315, "time_s": 0.0671},
                "MAC-Net": {"psnr": 25.11, "sam": 18.75, "ssim": 0.6333, "time_s": 3.708},
                "T3SC": {"psnr": 27.49, "sam": 12.10, "ssim": 0.7077, "time_s": 0.1826},
                "GRUNet": {"psnr": 28.27, "sam": 10.78, "ssim": 0.7317, "time_s": 0.1971},
                "Deep HS Prior": {"psnr": 25.14, "sam": 11.07, "ssim": 0.6069, "time_s": 37.79},
                "Proposed": {"psnr": 28.45, "sam": 9.99, "ssim": 0.7456, "time_s": 0.2846},
            },
        },
    }


def table_tasks_cave() -> dict[str, dict[str, dict[str, float]]]:
    """Table 2 — deblurring and ×4 SR on CAVE (means only)."""
    return {
        "deblur": {
            "ADMM TV": {"psnr": 27.25, "sam": 39.45, "ssim": 0.3680, "time_s": 2.297},
            "DPHSIR": {"psnr": 38.40, "sam": 8.18, "ssim": 0.9518, "time_s": 16.52},
            "Proposed": {"psnr": 39.48, "sam": 4.44, "ssim": 0.9704, "time_s": 6.991},
        },
        "superres_x4": {
            "ADMM TV": {"psnr": 25.42, "sam": 41.27, "ssim": 0.2820, "time_s": 5.743},
            "DPHSIR": {"psnr": 31.58, "sam": 9.04, "ssim": 0.9210, "time_s": 16.83},
            "Deep HS Prior": {"psnr": 36.29, "sam": 5.90, "ssim": 0.9495, "time_s": 676.4},
            "HSISR": {"psnr": 30.58, "sam": 10.15, "ssim": 0.8990, "time_s": 0.025},
            "Proposed": {"psnr": 32.02, "sam": 4.97, "ssim": 0.9542, "time_s": 21.34},
        },
    }


def table_ablation_cave_sigma010() -> dict[str, dict[str, float]]:
    """Table 3 — encoder design on CAVE, σ = 0.10 (means only)."""
    return {
        "Seq. Mono": {"psnr": 41.43, "sam": 6.01, "ssim": 0.9682, "time_s": 0.7625},
        "Seq. RGB": {"psnr": 43.33, "sam": 5.22, "ssim": 0.9762, "time_s": 0.2804},
        "Random": {"psnr": 42.59, "sam": 5.57, "ssim": 0.9744, "time_s": 0.2839},
        "PCA proj.": {"psnr": 33.86, "sam": 25.01, "ssim": 0.6839, "time_s": 1.717},
        "Grp. K = 1": {"psnr": 31.75, "sam": 22.85, "ssim": 0.9206, "time_s": 0.0335},
        "Proposed": {"psnr": 44.61, "sam": 4.89, "ssim": 0.9806, "time_s": 0.2817},
    }


def ablation_smoke_demo(cfg: HSIRRgbConfig | None = None) -> dict[str, float]:
    """Compare Table 3 encoder routes on a tiny cube with a smoothing inner denoiser."""
    cfg = cfg or HSIRRgbConfig()
    torch.manual_seed(24769)
    x = torch.rand(1, cfg.num_spectral_bands, 12, 12)
    y, _ = degrade(x, "denoise", noise_sigma=0.1)
    inner = awgn_inner_denoiser(0.1)
    e_tilde = torch.randn(cfg.latent_spectral_dim, cfg.num_spectral_bands) * 0.05
    out: dict[str, float] = {}
    for mode in (
        EncoderMode.SEQUENTIAL_MONO,
        EncoderMode.SEQUENTIAL_RGB,
        EncoderMode.RANDOM,
        EncoderMode.PCA,
        EncoderMode.GROUP_K1,
        EncoderMode.PROPOSED_QR,
    ):
        x_hat = denoise_by_mode(
            y,
            mode,
            inner,
            inner_c=cfg.inner_channels,
            num_groups=cfg.num_projection_groups,
            e_tilde=e_tilde,
            pca_reference=x,
        )
        out[f"mse_{mode.value}"] = float(F.mse_loss(x_hat, x).item())
    return out


def pnp_restore_demo(cfg: HSIRRgbConfig | None = None) -> dict[str, float]:
    """HQS + proposed QR adapter on denoise / deblur / SR stubs."""
    cfg = cfg or HSIRRgbConfig()
    torch.manual_seed(24769)
    x = torch.rand(1, cfg.num_spectral_bands, 24, 24)
    inner = awgn_inner_denoiser(0.05)
    adapter = SpectralPnPAdapter(cfg.num_spectral_bands, cfg.num_projection_groups, cfg.inner_channels)
    denoiser = lambda t: adapter(t, inner=inner)

    results: dict[str, float] = {}
    for task, ns in (("denoise", 0.10), ("deblur", 0.05), ("superres_x4", 0.005)):
        y, _ = degrade(x, task, noise_sigma=ns, scale=4)
        x_hat = pnp_hqs_restore(
            y if task != "superres_x4" else y,
            denoiser,
            task=task,
            num_iters=cfg.hqs_iterations,
            mu=cfg.hqs_mu,
            scale=4,
        )
        if task == "superres_x4":
            x_ref = torch.nn.functional.interpolate(
                x, size=x_hat.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            x_ref = x
        results[f"psnr_{task}"] = hs_psnr(x_hat, x_ref)
    return results


def training_step_demo(cfg: HSIRRgbConfig | None = None) -> dict[str, float]:
    """Smoke: QR adapter + identity inner denoiser + Eq. (6) MSE backward on ``\\tilde{E}``."""
    cfg = cfg or HSIRRgbConfig()
    torch.manual_seed(24769)
    b, C, h, w = 2, cfg.num_spectral_bands, 16, 16
    adapter = SpectralPnPAdapter(C, cfg.num_projection_groups, cfg.inner_channels)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    x = torch.rand(b, C, h, w)
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    inner = lambda z: z
    for _ in range(3):
        opt.zero_grad()
        x_hat = adapter(y, inner=inner)
        loss = adapter_training_loss(x_hat, x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        e, f_mat = adapter.encode_decode_matrices()
        fe = f_mat @ e
        ortho_err = float(torch.linalg.matrix_norm(fe - torch.eye(C, device=fe.device)).item())
        x_hat = adapter(y, inner=inner)
        ps = hs_psnr(x_hat, x)
        sam = spectral_angle_mapper(x_hat, x)
        ssim = hs_ssim_band_mean(x_hat, x)
    return {
        "mse_after_steps": float(F.mse_loss(x_hat, x).item()),
        "orthonormal_fe_error": ortho_err,
        "psnr": ps,
        "sam_deg": sam,
        "ssim_mean_bands": ssim,
    }


def evaluation_demo(cfg: HSIRRgbConfig | None = None) -> dict[str, Any]:
    cfg = cfg or HSIRRgbConfig()
    step = training_step_demo(cfg)
    abl = ablation_smoke_demo(cfg)
    pnp = pnp_restore_demo(cfg)
    t1 = table_denoising()
    t2 = table_tasks_cave()
    t3 = table_ablation_cave_sigma010()
    fig1 = fig1_harvard_teaser()
    rw = table_related_work()
    stab = verify_qr_nonexpansive_identity_inner(
        torch.randn(cfg.latent_spectral_dim, cfg.num_spectral_bands) * 0.05,
        torch.rand(1, cfg.num_spectral_bands, 8, 8),
        inner_c=cfg.inner_channels,
    )
    harv = harvard_untrained_identity_demo(cfg)
    return {
        **step,
        **{f"abl_{k}": v for k, v in abl.items()},
        **{f"pnp_{k}": v for k, v in pnp.items()},
        **{f"stab_{k}": v for k, v in stab.items() if k != "within_tol"},
        "stab_within_tol": stab["within_tol"],
        "harvard_untrained_mse": harv["mse"],
        "harvard_untrained_viz_psnr": harv["psnr_viz_bands"],
        "proposed_beats_grunet_cave_psnr_010": t1["sigma_0.10"]["CAVE"]["Proposed"]["psnr"]
        > t1["sigma_0.10"]["CAVE"]["GRUNet"]["psnr"],
        "proposed_best_sam_ablation": min(t3[m]["sam"] for m in t3) == t3["Proposed"]["sam"],
        "deblur_proposed_beats_dp_hsir": t2["deblur"]["Proposed"]["psnr"] > t2["deblur"]["DPHSIR"]["psnr"],
        "sr_proposed_sam_below_dp": t2["superres_x4"]["Proposed"]["sam"] < t2["superres_x4"]["DPHSIR"]["sam"],
        "latent_dim": cfg.latent_spectral_dim,
        "dataset_manifest_len": len(dataset_manifest(cfg)),
        "limitations_count": len(paper_limitations()),
        "fig1_rgb_beats_hs": fig1["rgb_drunet_psnr_db"] > fig1["hs_grunet_psnr_db"],
        "table_rw_ours_frozen_rgb": rw["Ours"]["frozen_rgb_denoiser"] == "yes",
        "seq_rgb_mse_le_seq_mono": abl["mse_seq_rgb"] <= abl["mse_seq_mono"],
        "architecture_keys": list(
            architecture_layout(
                height=cfg.patch_size,
                width=cfg.patch_size,
                num_bands=cfg.num_spectral_bands,
                num_groups=cfg.num_projection_groups,
                inner_channels=cfg.inner_channels,
            ).keys()
        ),
    }


def qr_identity_reconstruction_error(e_tilde: Tensor) -> float:
    """``‖FE − I‖`` for QR-derived ``E, F`` (should be ~0 when ``Kc ≥ C``)."""
    e, f = encoder_decoder_from_unconstrained(e_tilde)
    c = e.shape[1]
    fe = f @ e
    return float(torch.linalg.matrix_norm(fe - torch.eye(c, device=fe.device, dtype=fe.dtype)).item())
