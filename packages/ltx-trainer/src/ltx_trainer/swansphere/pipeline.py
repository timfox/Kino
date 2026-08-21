"""SwanSphere framework card, benchmarks, demos (arXiv:2605.30940)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.swansphere.config import SwanSphereConfig
from ltx_trainer.swansphere.layout import LIMITATIONS
from ltx_trainer.swansphere.odpo import odpo_reward, rank_candidates
from ltx_trainer.swansphere.streaming import latency_breakdown, stream_patch_latents
from ltx_trainer.swansphere.svac import svac_batch_loss


def framework_card(cfg: SwanSphereConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwanSphereConfig()
    return {
        "name": "SwanSphere",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_url,
        "task": "Streaming panoramic video / text → First-Order Ambisonics (FOA)",
        "idea": (
            "Causal Spatial LM plans patch-level semantics; LocDiT flow-matches continuous "
            "FOA latents per patch for low first-chunk latency. SVAC aligns VideoMAE-V2 with "
            "AudioMAE; multi-objective ODPO refines spatial/semantic/fidelity preferences."
        ),
        "architecture": {
            "spatial_lm": "causal inter-patch planning",
            "locdit": f"intra-patch flow matching ({cfg.locdit_steps} steps)",
            "vae": f"4-ch FOA continuous latent d={cfg.latent_dim} @ {cfg.latent_fps} FPS",
            "streaming": (
                f"patch={cfg.patch_frames} frames, stride={cfg.patch_stride}, "
                f"context={cfg.causal_patches} patches"
            ),
        },
        "data": {
            "video_foa_pairs": cfg.dataset_pairs,
            "hours": cfg.dataset_hours,
            "spatial_captions": cfg.caption_samples,
        },
        "latency": latency_breakdown(cfg),
        "defaults": cfg.__dict__,
    }


def table_video_to_foa() -> list[dict[str, float | str]]:
    """Table 1 — video-to-spatial audio on hybrid test set."""
    return [
        {
            "method": "Ground Truth",
            "params": "-",
            "inf_time": "-",
            "fd": None,
            "kl": None,
            "delta_theta": None,
            "delta_phi": None,
            "delta_angular": None,
            "mos_sq": 4.60,
            "mos_af": 4.58,
        },
        {
            "method": "MMAudio+AS",
            "params": "1.03B",
            "inf_time": "2.76s",
            "fd": 261.65,
            "kl": 2.43,
            "delta_theta": None,
            "delta_phi": None,
            "delta_angular": None,
            "mos_sq": 3.91,
            "mos_af": 3.60,
        },
        {
            "method": "Diff-Foley+AS",
            "params": "0.94B",
            "inf_time": "2.03s",
            "fd": 304.03,
            "kl": 3.12,
            "delta_theta": None,
            "delta_phi": None,
            "delta_angular": None,
            "mos_sq": 3.68,
            "mos_af": 3.26,
        },
        {
            "method": "ViSAGe",
            "params": "0.36B",
            "inf_time": "20.19s",
            "fd": 232.17,
            "kl": 2.67,
            "delta_theta": 1.57,
            "delta_phi": 0.63,
            "delta_angular": 1.59,
            "mos_sq": 3.82,
            "mos_af": 3.78,
        },
        {
            "method": "OmniAudio",
            "params": "1.22B",
            "inf_time": "0.85s",
            "fd": 157.67,
            "kl": 1.93,
            "delta_theta": 1.25,
            "delta_phi": 0.47,
            "delta_angular": 1.27,
            "mos_sq": 4.12,
            "mos_af": 4.27,
        },
        {
            "method": "SwanSphere",
            "params": "1.09B",
            "inf_time": "0.21s/9.13s",
            "fd": 120.28,
            "kl": 1.36,
            "delta_theta": 1.14,
            "delta_phi": 0.40,
            "delta_angular": 1.03,
            "mos_sq": 4.32,
            "mos_af": 4.44,
        },
    ]


def table_text_to_foa() -> list[dict[str, float | str]]:
    """Table 2 — text-to-spatial audio."""
    return [
        {"method": "Ground Truth", "fd": None, "kl": None, "mos_sq": 4.65, "mos_af": 4.76},
        {"method": "MMAudio+AS", "fd": 313.26, "kl": 2.77, "mos_sq": 3.75, "mos_af": 3.44},
        {"method": "AudioLDM-2+AS", "fd": 294.17, "kl": 2.45, "mos_sq": 3.86, "mos_af": 3.53},
        {"method": "Tango2+AS", "fd": 235.71, "kl": 2.42, "mos_sq": 3.95, "mos_af": 3.27},
        {"method": "OmniAudio(text)", "fd": 174.13, "kl": 1.83, "mos_sq": 4.11, "mos_af": 4.16},
        {"method": "SwanSphere", "fd": 142.80, "kl": 1.43, "mos_sq": 4.31, "mos_af": 4.43},
    ]


def table_svac_ablation() -> list[dict[str, float | str]]:
    """Table 3 — SVAC ablation."""
    return [
        {"model": "Ours", "fd": 120.28, "kl": 1.36, "delta_theta": 1.14, "delta_phi": 0.40, "delta_angular": 1.03},
        {"model": "sem-only", "fd": 127.12, "kl": 1.41, "delta_theta": 1.26, "delta_phi": 0.49, "delta_angular": 1.12},
        {"model": "CLIP", "fd": 140.28, "kl": 1.44, "delta_theta": 1.31, "delta_phi": 0.55, "delta_angular": 1.34},
    ]


def table_model_ablation() -> list[dict[str, float | str]]:
    """Table 4 — capacity, ODPO, DiT paradigm."""
    return [
        {
            "model": "SwanSphere-L",
            "params": "1.09B",
            "inf_time": "0.21s/9.13s",
            "fd": 120.28,
            "kl": 1.36,
            "delta_theta": 1.14,
            "delta_phi": 0.40,
            "delta_angular": 1.03,
        },
        {
            "model": "SwanSphere-M",
            "params": "0.62B",
            "inf_time": "0.17s/7.60s",
            "fd": 132.52,
            "kl": 1.43,
            "delta_theta": 1.28,
            "delta_phi": 0.46,
            "delta_angular": 1.16,
        },
        {
            "model": "SwanSphere-S",
            "params": "0.43B",
            "inf_time": "0.13s/6.10s",
            "fd": 139.81,
            "kl": 1.58,
            "delta_theta": 1.45,
            "delta_phi": 0.53,
            "delta_angular": 1.33,
        },
        {
            "model": "SwanSphere-L w/o ODPO",
            "params": "1.09B",
            "inf_time": "0.21s/9.13s",
            "fd": 133.91,
            "kl": 1.44,
            "delta_theta": 1.21,
            "delta_phi": 0.43,
            "delta_angular": 1.22,
        },
        {
            "model": "DiT",
            "params": "1.11B",
            "inf_time": "6.47s",
            "fd": 123.08,
            "kl": 1.36,
            "delta_theta": 0.91,
            "delta_phi": 0.46,
            "delta_angular": 1.14,
        },
    ]


def table_wcs() -> list[dict[str, float | str]]:
    """Table 5 — independent SELD wCS evaluator."""
    return [
        {"model": "MMAudio+AS", "wcs": 0.32},
        {"model": "Diff-Foley+AS", "wcs": 0.27},
        {"model": "ViSAGe", "wcs": 0.35},
        {"model": "OmniAudio", "wcs": 0.41},
        {"model": "Ours w/o ODPO", "wcs": 0.52},
        {"model": "Ours", "wcs": 0.63},
    ]


def pipeline_demo(cfg: SwanSphereConfig | None = None) -> dict[str, Any]:
    cfg = cfg or SwanSphereConfig()
    frames = cfg.patch_frames * 3
    stream = stream_patch_latents(frames, cfg=cfg, seed=11)
    feats_v = [np.random.default_rng(i).standard_normal(16) for i in range(4)]
    feats_a = [np.random.default_rng(i + 10).standard_normal(16) for i in range(4)]
    svac_loss = svac_batch_loss(feats_v, feats_a, full_physics=True)
    svac_sem = svac_batch_loss(feats_v, feats_a, full_physics=False)
    metrics = [(1.2, 0.7, 0.5), (0.9, 0.8, 0.3), (1.5, 0.5, 0.8)]
    win, lose = rank_candidates(metrics)
    return {
        **stream,
        "svac_loss_full": svac_loss,
        "svac_loss_semantic_only": svac_sem,
        "odpo_winner": win,
        "odpo_loser": lose,
        "odpo_top_reward": odpo_reward(*metrics[win]),
    }


def evaluation_demo(cfg: SwanSphereConfig | None = None) -> dict[str, Any]:
    demo = pipeline_demo(cfg)
    demo["framework"] = framework_card(cfg)
    demo["limitations"] = LIMITATIONS
    rows = {r["method"]: r for r in table_video_to_foa() if r["method"] == "SwanSphere"}
    demo["swansphere_fd"] = rows["SwanSphere"]["fd"]
    demo["swansphere_angular"] = rows["SwanSphere"]["delta_angular"]
    return demo


def benchmarks_bundle() -> dict[str, Any]:
    return {
        "table_video_to_foa": table_video_to_foa(),
        "table_text_to_foa": table_text_to_foa(),
        "table_svac_ablation": table_svac_ablation(),
        "table_model_ablation": table_model_ablation(),
        "table_wcs": table_wcs(),
    }


def knowledge_summary() -> dict[str, Any]:
    return {
        "arxiv": "2605.30940",
        "venue": "ICML 2026 (PMLR 306)",
        "modalities": ["panoramic_video", "spatial_caption", "text"],
        "output": "FOA (W,X,Y,Z)",
        "datasets": ["SwanSphere 165k pairs", "Sphere360", "YT-Ambigen", "V2ST captions ~3.1k"],
        "baselines": ["MMAudio+AS", "OmniAudio", "ViSAGe", "Diff-Foley+AS"],
        "metrics": ["FD", "KL", "DoA angular error", "MOS-SQ", "MOS-AF", "wCS"],
    }
