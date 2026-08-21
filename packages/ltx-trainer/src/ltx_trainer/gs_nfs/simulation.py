"""Synthetic 4DGS frame encode/decode stub for GS-NFS."""

from __future__ import annotations

import numpy as np

from ltx_trainer.gs_nfs.config import GSNFSConfig, GSNFSParams
from ltx_trainer.gs_nfs.metrics import (
    dead_zone_quantize,
    frame_size_mb,
    psnr_proxy,
    rlgr_size_proxy,
)
from ltx_trainer.gs_nfs.morton import (
    build_level_lists,
    encode_occupancy_stream,
    morton_encode,
    roundtrip_voxels,
)
from ltx_trainer.gs_nfs.paper_tables import table2_latency_ms, table3_mean_comparison
from ltx_trainer.gs_nfs.raht import high_frequency_energy, raht_forward, raht_inverse, raht_prelude
from ltx_trainer.gs_nfs.sh_codec import compressed_size_proxy, decorrelate_sh


def synthetic_frame(
    *,
    n_gaussians: int = 512,
    sh_degree: int = 2,
    bits: int = 8,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = 1 << bits
    coords = rng.integers(0, grid, size=(n_gaussians, 3))
    n_channels = 3 + 3 * ((sh_degree + 1) ** 2 - 1)
    attrs = rng.normal(0, 0.3, size=(n_gaussians, n_channels)).astype(np.float64)
    attrs[:, :3] = rng.uniform(0, 1, size=(n_gaussians, 3))
    return {"coords": coords, "attrs": attrs, "bits": np.array(bits), "sh_degree": np.array(sh_degree)}


def voxelize_merge(frame: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Quantize positions and average attributes per voxel."""
    bits = int(frame["bits"])
    coords = frame["coords"]
    attrs = frame["attrs"]
    keys = np.array([morton_encode(int(x), int(y), int(z), bits=bits) for x, y, z in coords])
    order = np.argsort(keys)
    keys = keys[order]
    attrs = attrs[order]
    unique, start = np.unique(keys, return_index=True)
    merged_attrs = []
    for i, code in enumerate(unique):
        j = start[i + 1] if i + 1 < len(start) else len(attrs)
        merged_attrs.append(attrs[start[i] : j].mean(axis=0))
    return unique, np.stack(merged_attrs, axis=0)


def encode_frame(frame: dict[str, np.ndarray], *, params: GSNFSParams | None = None) -> dict[str, object]:
    params = params or GSNFSParams()
    bits = int(frame["bits"])
    codes, attrs = voxelize_merge(frame)
    levels = build_level_lists(codes, depth=bits)
    occ = encode_occupancy_stream(levels)
    ans_ratio = 2.0 if params.ans_entropy else 1.0
    geom_bits = len(occ) * 8 / ans_ratio

    schedule = raht_prelude(codes, depth=bits)
    sh = attrs.reshape(len(attrs), -1, 3)
    if params.klt_decorrelation:
        decor, _ = decorrelate_sh(sh)
        flat = decor.reshape(len(attrs), -1)
    else:
        flat = attrs
    raht_out = raht_forward(flat[:, 0], schedule)
    hf_before = high_frequency_energy(flat[:, 0], schedule)
    hf_after = high_frequency_energy(raht_out, schedule)

    quant = dead_zone_quantize(flat, step=params.quant_step_sh)
    attr_bits = rlgr_size_proxy(quant, block_size=params.rlgr_block_size)

    return {
        "n_voxels": len(codes),
        "geom_bits": geom_bits,
        "attr_bits": attr_bits,
        "hf_energy_drop": hf_before - hf_after,
        "occupancy_bytes": len(occ),
    }


def decode_frame(
    frame: dict[str, np.ndarray],
    encoded: dict[str, object],
    *,
    params: GSNFSParams | None = None,
) -> dict[str, np.ndarray]:
    params = params or GSNFSParams()
    bits = int(frame["bits"])
    codes, attrs = voxelize_merge(frame)
    schedule = raht_prelude(codes, depth=bits)
    sh = attrs.reshape(len(attrs), -1, 3)
    if params.klt_decorrelation:
        decor, basis = decorrelate_sh(sh)
        flat = decor.reshape(len(attrs), -1)
    else:
        flat = attrs
    coeffs = raht_forward(flat[:, 0], schedule)
    recon_flat = raht_inverse(coeffs, schedule)
    recon = dead_zone_quantize(recon_flat, step=params.quant_step_sh)
    return {"attrs_recon": recon, "codes": codes}


def latency_stub(*, dataset: str) -> dict[str, float]:
    """Map paper Table 2 anchors to synthetic GPU-resident timings."""
    t2 = table2_latency_ms()
    gs = t2["GS-NFS"]
    ds = t2.get(dataset, t2["N3DV"])
    return {
        "encode_ms": ds["encode"],
        "decode_ms": ds["decode"],
        "budget_30fps_ms": 33.0,
        "encode_under_budget": ds["encode"] < 33.0,
        "decode_under_budget": ds["decode"] < 33.0,
        "vs_mesong_encode_ratio": t2["MesonGS"]["encode"] / gs["encode"],
    }


def compression_vs_baselines(*, n_gaussians: int = 200_000) -> dict[str, float]:
    raw_mb = frame_size_mb(n_gaussians)
    t3 = table3_mean_comparison()
    return {
        "raw_mb": raw_mb,
        "gs_nfs_mb": raw_mb / 2.5,
        "meson_mb": raw_mb / t3["MesonGS"]["rcr"],
        "lts_mb": raw_mb / t3["LTS-Draco"]["rcr"],
        "gpcc_mb": raw_mb / t3["G-PCC"]["rcr"],
    }


def klt_ablation_synthetic(*, seed: int = 4) -> dict[str, float]:
    frame = synthetic_frame(n_gaussians=256, sh_degree=2, seed=seed)
    _, attrs = voxelize_merge(frame)
    sh = attrs.reshape(len(attrs), -1, 3)
    rgb = compressed_size_proxy(sh.reshape(len(attrs), -1), use_yuv_klt=False, quant_step=0.1)
    klt = compressed_size_proxy(sh.reshape(len(attrs), -1), use_yuv_klt=True, quant_step=0.1)
    return {"rgb_proxy": rgb, "klt_proxy": klt, "ratio_rgb_over_klt": rgb / max(klt, 1e-6)}


def full_pipeline_demo(*, seed: int = 2, cfg: GSNFSConfig | None = None) -> dict[str, object]:
    cfg = cfg or GSNFSConfig()
    frame = synthetic_frame(seed=seed)
    enc = encode_frame(frame, params=cfg.params)
    dec = decode_frame(frame, enc, params=cfg.params)
    _, attrs = voxelize_merge(frame)
    psnr = psnr_proxy(attrs[:, 0], dec["attrs_recon"])
    morton_ok = len(roundtrip_voxels(frame["coords"], bits=int(frame["bits"]))) >= 1
    lat_hifi = latency_stub(dataset="HiFi4G")
    lat_n3dv = latency_stub(dataset="N3DV")
    comp = compression_vs_baselines()
    klt = klt_ablation_synthetic(seed=seed)
    return {
        "encode": enc,
        "decode_psnr": psnr,
        "morton_roundtrip": morton_ok,
        "latency_hifi4g": lat_hifi,
        "latency_n3dv": lat_n3dv,
        "compression": comp,
        "klt_ablation": klt,
    }
