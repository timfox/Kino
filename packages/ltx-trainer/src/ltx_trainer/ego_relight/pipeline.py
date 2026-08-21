"""EgoRelight end-to-end stubs and framework card."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.ego_relight.benchmarks import benchmarks_bundle
from ltx_trainer.ego_relight.config import EgoRelightConfig
from ltx_trainer.ego_relight.hdr import finlayson_ldr_to_hdr, optimize_hdr_color_correction
from ltx_trainer.ego_relight.lighting import diffuse_maps, sample_specular_rays
from ltx_trainer.ego_relight.perception import encode_depth_guidance, unproject_depth


def framework_card(cfg: EgoRelightConfig | None = None) -> dict[str, Any]:
    cfg = cfg or EgoRelightConfig()
    return {
        "name": cfg.paper_title,
        "arxiv": f"arXiv:{cfg.paper_arxiv}",
        "authors": cfg.authors,
        "problem": (
            "MR telepresence needs egocentric full-body capture, relightable appearance, "
            "and HDR environment estimation from a single HMD."
        ),
        "components": {
            "egocentric_perception": "Stereo down-facing pose + depth -> depth-conditioned AnimationNet",
            "relightable_avatar": "GeoLifting + DiffuseNet + SpecularNet (diffuse/specular separation)",
            "hdr_capture": "360° LDR scan + Finlayson color correction via avatar inverse rendering",
        },
        "reference_metrics": {
            "psnr_subject1": cfg.ref_psnr,
            "ssim_subject1": cfg.ref_ssim,
            "runtime_ms": cfg.runtime_ms,
        },
        "benchmarks": benchmarks_bundle(),
    }


def run_perception_smoke(cfg: EgoRelightConfig | None = None) -> dict[str, Any]:
    """Synthetic depth + mesh vertex correspondence (Sec. 6)."""
    cfg = cfg or EgoRelightConfig()
    vertex = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    vn = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    points = np.array([[0.01, 0.0, 1.02], [2.0, 0.0, 1.0]], dtype=np.float32)
    pnormals = np.tile(vn, (2, 1))
    xi = encode_depth_guidance(
        vertex,
        vn,
        points,
        pnormals,
        epsilon_d=cfg.epsilon_depth_m,
        epsilon_n=cfg.epsilon_normal,
    )
    depth = np.ones((8, 8), dtype=np.float32)
    K = np.array([[100.0, 0, 4], [0, 100.0, 4], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    pts = unproject_depth(depth, K, T)
    return {
        "xi_norm": float(np.linalg.norm(xi)),
        "n_unprojected": int(pts.shape[0]),
        "xi_valid": bool(np.linalg.norm(xi) > 0),
    }


def relight_frame(
    normal: np.ndarray,
    view_dir: np.ndarray,
    albedo: float,
    *,
    cfg: EgoRelightConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """CPU relighting stub using diffuse maps + top-k specular rays."""
    cfg = cfg or EgoRelightConfig()
    rng = np.random.default_rng(seed)
    L = min(cfg.num_lights, 64)
    light_dirs = normalize_rows(rng.standard_normal((L, 3)))
    intensities = np.abs(rng.standard_normal((L, 3))).astype(np.float32)
    vis = (rng.random(L) > 0.3).astype(np.float32)
    n = normal.astype(np.float64)
    n = n / max(np.linalg.norm(n), 1e-8)
    normals = np.tile(n, (1, 1))
    rho, chi = diffuse_maps(normals, light_dirs, intensities, vis[None, :])
    rays = sample_specular_rays(
        n,
        view_dir / max(np.linalg.norm(view_dir), 1e-8),
        light_dirs,
        intensities,
        vis,
        num_rays=min(8, cfg.num_specular_rays),
        alpha=cfg.blinn_phong_alpha,
    )
    cdiff = float(albedo * chi[0] / max(rho[0], 1e-3))
    cspec = float(0.15 * rays[:, -1].mean())
    return {
        "rho": float(rho[0]),
        "chi": float(chi[0]),
        "cdiff": cdiff,
        "cspec": cspec,
        "composite": float(np.clip(cdiff + cspec, 0, 1)),
        "n_specular_rays": int(rays.shape[0]),
    }


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-8)).astype(np.float32)


def run_appearance_torch_smoke() -> dict[str, Any]:
    """Optional torch forward through GeoLifting / Diffuse / Specular stubs."""
    try:
        import torch
        from ltx_trainer.ego_relight.appearance import DiffuseNetStub, GeoLiftingStub, SpecularNetStub, composite_shading
    except ImportError:
        return {"torch": False}
    b, h, w = 1, 16, 16
    stack = torch.randn(b, 9, h, w)
    geo = GeoLiftingStub()
    albedo, nmap = geo(stack)
    rho = torch.rand(b, 1, h, w)
    chi = torch.rand(b, 1, h, w)
    diff = DiffuseNetStub()
    cdiff = diff(albedo, rho, nmap, chi)
    rays = torch.rand(b, 4, h, w, 5)
    spec = SpecularNetStub()
    cspec = spec(albedo, rho, nmap, rays)
    out = composite_shading(cdiff[0, 0].detach().numpy(), cspec[0, 0].detach().numpy())
    return {"torch": True, "shading_mean": float(out.mean())}


def evaluation_demo(cfg: EgoRelightConfig | None = None) -> dict[str, Any]:
    from ltx_trainer.ego_relight.mock import evaluation_smoke

    return evaluation_smoke(cfg)
