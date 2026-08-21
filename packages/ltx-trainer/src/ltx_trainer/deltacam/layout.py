"""Static conditioning layout + paper limitation notes (Sec. 4.5, Fig. 13)."""

from __future__ import annotations

from typing import Any


def conditioning_layout(
    *,
    height: int,
    width: int,
    num_intrinsics: int,
    plucker_channels: int = 6,
) -> dict[str, Any]:
    """Tensor roles for CCM + extrinsic branch (Fig. 3–5) — shapes only, no VDM internals."""
    return {
        "plucker_ray_map": {
            "shape": [height, width, plucker_channels],
            "role": "Extrinsic pose P relative to anchor; zeros for same-view intrinsic edit",
        },
        "scene_proxy_depth": {"shape": ["T", height, width], "role": "Monocular depth proxy"},
        "scene_proxy_flow": {"shape": ["T", 2, height, width], "role": "Optical flow proxy"},
        "scene_proxy_rgb": {"shape": ["T", 3, height, width], "role": "Source RGB backbone signal"},
        "delta_intrinsics": {"shape": ["T", num_intrinsics], "role": "Δθ_t normalized per Eq. (2) groups"},
        "ccm_spatial_latent": {
            "shape": ["C", height, width],
            "role": "VAE-encoded proxy features before/after FiLM",
        },
    }


def paper_limitations() -> list[dict[str, str]]:
    """Sec. 4.5 — when deterministic proxies fail, conditioning can leak artifacts."""
    return [
        {
            "id": "proxy_depth_flow",
            "summary": "Heavy motion blur, severe occlusion, or complex transparency break depth/flow.",
            "symptom": "Bokeh / geometric effects may leak or distort (Fig. 13).",
        },
        {
            "id": "spatial_style_leakage",
            "summary": "Defocus and motion blur are depth/velocity coupled.",
            "symptom": "Style InfoNCE stays high for bokeh vs photometric effects (Table 3).",
        },
        {
            "id": "backbone_scale",
            "summary": "Paper uses Wan-2.1 1.3B at 480×832.",
            "symptom": "Longer clips and larger backbones are left to future work.",
        },
    ]
