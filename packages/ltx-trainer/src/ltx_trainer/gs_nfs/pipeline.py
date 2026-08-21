"""Framework card, demos, smoke for GS-NFS (arXiv:2606.05650)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gs_nfs.config import GSNFSConfig
from ltx_trainer.gs_nfs.paper_tables import (
    dataset_catalog,
    jetson_decode_ms,
    table2_latency_ms,
    table3_mean_comparison,
    table_a8_klt_sizes_mb,
)
from ltx_trainer.gs_nfs.simulation import full_pipeline_demo, klt_ablation_synthetic, latency_stub


def framework_card(cfg: GSNFSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GSNFSConfig()
    p = cfg.params
    return {
        "name": "GS-NFS",
        "paper": cfg.paper_arxiv,
        "title": "Bandwidth-adaptive Streaming of Dynamic Gaussian Splats and Point Clouds",
        "task": "Post-training GPU 4DGS / point-cloud compression at 30 fps",
        "pipeline": [
            "Voxelize + merge Gaussians (Morton sort)",
            "GPU octree occupancy + ANS entropy (positions)",
            "RAHT prelude + parallel attribute transform",
            "Dead-zone quantize + block RLGR entropy",
            "SH RGB→YUV + per-channel KLT decorrelation",
        ],
        "components_gpu": [
            "octree_encode",
            "octree_decode",
            "raht_prelude",
            "raht_transform",
            "rlgr_blocks",
            "klt_sh",
        ],
        "datasets": list(cfg.datasets),
        "baselines": list(cfg.baselines),
        "bytes_per_gaussian": cfg.bytes_per_gaussian,
        "params": {
            "octree_depth": p.octree_depth,
            "rlgr_block_size": p.rlgr_block_size,
            "klt_decorrelation": p.klt_decorrelation,
        },
        "packages": list(cfg.packages),
    }


def paper_limitations() -> list[str]:
    return [
        "Numpy stub — no CUDA kernels, nvCOMP ANS, or custom RLGR GPU blocks.",
        "Latency numbers mirror Table 2 anchors; synthetic pipeline validates ordering only.",
        "No trained 3DGS frames from HiFi4G/N3DV; Morton/RAHT roundtrip on synthetic voxels.",
        "V3-2D inter-frame GOP compression not modeled; GS-NFS is intra-frame G-PCC style.",
        "Mobile Jetson decode uses paper Table 4 anchors, not on-device timing.",
    ]


def evaluation_demo(cfg: GSNFSConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GSNFSConfig()
    demo = full_pipeline_demo(seed=7, cfg=cfg)
    return {
        "framework": framework_card(cfg),
        "limitations": paper_limitations(),
        "dataset": dataset_catalog(),
        "tables": {
            "table2_latency_ms": table2_latency_ms(),
            "table3_comparison": table3_mean_comparison(),
            "table_a8_klt_mb": table_a8_klt_sizes_mb(),
            "jetson_decode_ms": jetson_decode_ms(),
        },
        "pipeline_demo": demo,
        "latency_n3dv": latency_stub(dataset="N3DV"),
    }


def evaluation_smoke(cfg: GSNFSConfig | None = None) -> dict[str, Any]:
    demo = evaluation_demo(cfg)
    t2 = demo["tables"]["table2_latency_ms"]
    t3 = demo["tables"]["table3_comparison"]
    a8 = demo["tables"]["table_a8_klt_mb"]["flame_salmon"]
    lat = demo["latency_n3dv"]
    pipe = demo["pipeline_demo"]

    assert abs(t2["GS-NFS"]["encode"] - 23.0) < 0.5
    assert abs(t2["HiFi4G"]["decode"] - 14.0) < 0.5
    assert lat["encode_under_budget"] and lat["decode_under_budget"]
    assert lat["vs_mesong_encode_ratio"] > 100.0
    assert t3["MesonGS"]["rcr"] > 1.5
    assert t3["LTS-Draco"]["rcr"] > 2.5
    assert a8["rgb"] / a8["klt"] > 1.35
    assert pipe["morton_roundtrip"]
    assert pipe["encode"]["hf_energy_drop"] >= 0.0
    assert pipe["klt_ablation"]["ratio_rgb_over_klt"] > 1.05

    return {
        "status": "ok",
        "paper": (cfg or GSNFSConfig()).paper_arxiv,
        "n3dv_encode_ms": lat["encode_ms"],
        "mesong_encode_speedup": lat["vs_mesong_encode_ratio"],
        "klt_rgb_over_klt": a8["rgb"] / a8["klt"],
        "demo_keys": list(demo.keys()),
    }
