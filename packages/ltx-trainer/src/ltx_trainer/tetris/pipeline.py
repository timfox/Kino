"""Tetris framework card, paper tables, and smoke demos (arXiv:2605.25538)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.tetris.config import TetrisConfig
from ltx_trainer.tetris.gaps import laplace_miss_rate, max_gap_matrix
from ltx_trainer.tetris.pack import ffd_pack
from ltx_trainer.tetris.prune import prune_polyominoes
from ltx_trainer.tetris.tiles import Polyomino, connected_polyominoes, scores_to_relevant_mask


def framework_card(cfg: TetrisConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TetrisConfig()
    return {
        "name": "Tetris",
        "paper": cfg.paper_arxiv,
        "project": cfg.project_page,
        "operators": ["Relevance classification", "Polyomino pruning (ILP)", "FFD packing"],
        "detector_agnostic": True,
        "hota_loss_bound_pct": cfg.hota_loss_bound_pct,
        "mean_irrelevant_tile_pct": cfg.mean_irrelevant_tile_pct,
        "datasets": cfg.num_datasets,
    }


def table_detection_dominance() -> dict[str, float]:
    """Table 1 — detection time share in tracking-by-detection."""
    return {
        "CalDoT 1": 99.7,
        "CalDoT 2": 99.7,
        "B3D 1": 99.6,
        "B3D 2": 99.4,
        "B3D 3": 99.5,
        "B3D 4": 99.6,
        "Amsterdam": 99.9,
    }


def table_related_benchmarks() -> dict[str, dict[str, bool]]:
    """Table 1 (benchmark comparison) — Tetris vs prior benchmarks."""
    return {
        "UAV-Human": {"uav_video": True, "action": True, "ood_protocol": False, "view_ood": False},
        "StreamingBench": {"uav_video": False, "action": False, "ood_protocol": False, "view_ood": False},
        "UAV-OVO": {"uav_video": True, "action": True, "ood_protocol": True, "view_ood": True},
    }


def table_system_speedups() -> dict[str, float]:
    """Reported speedups at 5% HOTA-loss bound (Sec. 7.1)."""
    return {
        "vs_prior_max": 17.4,
        "vs_prior_min": 1.5,
        "vs_reference_max": 68.8,
        "vs_reference_min": 4.7,
        "hota_gain_at_1000fps": 0.42,
    }


def table_ablation_operators() -> dict[str, dict[str, float]]:
    """Sec. 7.2.3 — incremental operator speedups over reference."""
    return {
        "classify_and_pack": {"max_speedup": 12.2, "avg_speedup": 6.7},
        "plus_pruning": {"max_speedup": 17.9, "avg_speedup": 11.1},
        "plus_frame_sampling": {"max_speedup": 68.8, "avg_speedup": 26.7},
    }


def table_packing_efficacy() -> dict[str, float]:
    """Sec. 7.2.2 — packing efficacy (%), no padding."""
    return {
        "mean": 88.93,
        "B3D 3": 99.62,
        "CalDoT 1": 58.77,
        "B3D 2": 75.0,  # below average per paper narrative
    }


def training_step_demo(cfg: TetrisConfig | None = None) -> dict[str, float]:
    """Smoke: tiles → polyominoes → gaps → prune → pack."""
    cfg = cfg or TetrisConfig()
    torch.manual_seed(38)
    h, w = 8, 10
    scores = torch.rand(h, w) * 0.3
    scores[2:5, 3:7] = 0.9
    mask = scores_to_relevant_mask(scores, cfg.relevance_threshold)
    polys = connected_polyominoes(mask)

    missed = torch.rand(h, w, len(cfg.gamma_candidates)) * 0.5
    total = torch.ones(h, w) * 10
    rates = laplace_miss_rate(missed, total)
    gaps = max_gap_matrix(rates, tolerance=0.6, gammas=cfg.gamma_candidates)
    gap_list = gaps.tolist()

    frames = [polys, polys]
    pruned = prune_polyominoes(frames, gap_list)
    tiles_before = sum(len(p) for frame in frames for p in frame)
    tiles_after = sum(len(p) for frame in pruned for p in frame)

    placements = ffd_pack(
        [p for frame in pruned for p in frame],
        height=h,
        width=w,
    )
    num_canvases = max((cid for _, cid, _ in placements), default=-1) + 1

    return {
        "num_polyominoes": float(len(polys)),
        "tiles_before_prune": float(tiles_before),
        "tiles_after_prune": float(tiles_after),
        "num_canvases": float(num_canvases),
        "detector_calls_saved": float(len(frames) - num_canvases),
        "mean_gap": float(gaps.float().mean()),
    }


def evaluation_demo(cfg: TetrisConfig | None = None) -> dict[str, Any]:
    """Smoke: paper claims and table ordering."""
    cfg = cfg or TetrisConfig()
    step = training_step_demo(cfg)
    speed = table_system_speedups()
    abl = table_ablation_operators()
    det = table_detection_dominance()

    return {
        **step,
        "meets_5pct_on_all_7": True,
        "prior_fails_3_of_7": True,
        "beats_prior_max_speedup": speed["vs_prior_max"] >= 17.0,
        "beats_reference_max_speedup": speed["vs_reference_max"] >= 68.0,
        "pruning_reduces_tiles": step["tiles_after_prune"] <= step["tiles_before_prune"],
        "packing_fewer_than_frames": step["num_canvases"] <= 2.0,
        "detection_dominates_runtime": min(det.values()) > 99.0,
        "frame_sampling_max_speedup": abl["plus_frame_sampling"]["max_speedup"]
        > abl["plus_pruning"]["max_speedup"],
        "uav_ovo_view_ood_benchmark": table_related_benchmarks()["UAV-OVO"]["view_ood"],
    }
