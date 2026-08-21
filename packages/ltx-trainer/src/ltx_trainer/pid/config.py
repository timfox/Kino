"""PiD — Pixel diffusion Decoder (Lu et al., arXiv:2605.23902)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PiDConfig:
    """Defaults from Sec. 3–4 and Appendix (PixelDiT 1.3B + adapter)."""

    name: str = "PiD"
    paper_arxiv: str = "arXiv:2605.23902"
    website: str = "https://research.nvidia.com/labs/sil/projects/pid/"
    title: str = "Fast and High-Resolution Latent Decoding with Pixel Diffusion"

    # Backbone (Sec. 4.2)
    base_model: str = "PixelDiT-1.3B"
    patch_size: int = 16
    hidden_dim: int = 1536
    text_encoder: str = "Gemma-2-2B-it"
    text_dim: int = 2304

    # Latent conditioning (Eq. 3–6)
    sigma_max: float = 0.8
    gate_alpha: float = 5.0
    gate_bias: float = 2.0
    adapter_channels: int = 512
    inject_every_n_blocks: int = 2

    # Upscale / decode
    default_scale: int = 4
    scales_by_latent: dict[str, int] = field(
        default_factory=lambda: {
            "vae_flux1": 4,
            "vae_sd3": 4,
            "vae_flux2": 4,
            "vae_zimage": 4,
            "dinov2": 4,
            "siglip": 8,
        }
    )

    # DMD2 student (Sec. 3.4, 4.2)
    student_steps: int = 4
    dmd2_sigmas: tuple[float, ...] = (0.999, 0.866, 0.634, 0.342)

    # LDM early exit presets (Sec. 3.4, Fig. 8)
    early_exit_presets: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "flux1_dev": (24, 28),
            "sd3_medium": (24, 28),
            "flux2_dev": (45, 50),
            "z_image": (45, 50),
            "dinov2_ditdh": (50, 50),
            "siglip_scale_rae": (50, 50),
        }
    )

    # Training (Sec. 4.2)
    prior_lr: float = 2e-5
    adapter_lr: float = 5e-5
    distill_lr: float = 1e-5
    caption_dropout: float = 0.1
    latent_dropout: float = 0.1

    # Reported latency anchors (Table 3, ms compile GB200 @ 2048)
    latency_gb200_2048_compile_ms: float = 211.2
    memory_gb_2048: float = 13.0

    ltx_hook: str = (
        "Replace VAE decode + SR cascade for 4×/8× LTX delivery; "
        "decode partially denoised latents for LDM early exit"
    )
