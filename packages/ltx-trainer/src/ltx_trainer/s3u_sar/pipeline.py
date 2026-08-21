"""Toy S³U-SAR loss demo on synthetic keypoints."""

from __future__ import annotations

from typing import Any

from ltx_trainer.s3u_sar.benchmarks import summary_anchors
from ltx_trainer.s3u_sar.config import S3USarConfig
from ltx_trainer.s3u_sar.constants import VISIBILITY_DEGRADED, VISIBILITY_SALIENT
from ltx_trainer.s3u_sar.losses import (
    confidence_gated_total,
    entropy_loss,
    heatmap_mse_loss,
    hetero_loss,
    spatial_softmax,
    topo_loss,
)
from ltx_trainer.s3u_sar.structure import DEFAULT_TOPOLOGY_EDGES, valid_topology_edges


def _synthetic_aircraft_keypoints() -> tuple[list[tuple[float, float]], list[int]]:
    """Stub 10-point aircraft layout in normalized coords."""
    xy = [
        (0.85, 0.50),  # nose tip
        (0.70, 0.50),  # nose root
        (0.45, 0.15),  # wing tip L
        (0.45, 0.85),  # wing tip R
        (0.55, 0.35),  # wing root L
        (0.55, 0.65),  # wing root R
        (0.10, 0.50),  # tail tip
        (0.25, 0.50),  # tail root
        (0.50, 0.30),  # engine L
        (0.50, 0.70),  # engine R
    ]
    vis = [VISIBILITY_SALIENT] * 8 + [VISIBILITY_DEGRADED, VISIBILITY_DEGRADED]
    return xy, vis


def run_demo(cfg: S3USarConfig | None = None) -> dict[str, Any]:
    cfg = cfg or S3USarConfig()
    gt_xy, vis = _synthetic_aircraft_keypoints()
    # Perturbed prediction
    pred_xy = [(x + 0.02, y - 0.01) for x, y in gt_xy]
    lambdas = [cfg.lambda_high if v == VISIBILITY_SALIENT else cfg.lambda_low for v in vis]
    edges = valid_topology_edges(vis, DEFAULT_TOPOLOGY_EDGES)

    flat_gt = [x for p in gt_xy for x in p]
    flat_pred = [x + 0.01 for x in flat_gt]
    lmse = heatmap_mse_loss([flat_pred], [flat_gt])
    lhet = hetero_loss(pred_xy, gt_xy, lambdas)
    ltop = topo_loss(pred_xy, gt_xy, edges, gamma=cfg.gamma_topo)

    probs = spatial_softmax([0.1, 0.05, 0.6, 0.15, 0.1], cfg.temperature)
    lent = entropy_loss(probs)
    confidences = [max(probs)] * len(gt_xy)

    total = confidence_gated_total(
        lmse,
        [lhet / len(gt_xy)] * len(gt_xy),
        [lent] * len(gt_xy),
        [(confidences[i], confidences[j], ltop / max(len(edges), 1)) for i, j in edges],
        confidences=confidences,
        alpha=cfg.alpha,
        beta=cfg.beta,
        mu=cfg.mu,
    )

    summary = summary_anchors()
    return {
        "config": {"temperature": cfg.temperature, "alpha": cfg.alpha, "mu": cfg.mu},
        "num_keypoints": len(gt_xy),
        "valid_edges": len(edges),
        "losses": {
            "Lmse": round(lmse, 6),
            "Lhetero": round(lhet, 6),
            "Ltopo": round(ltop, 6),
            "Lentropy": round(lent, 6),
            "Ltotal": round(total, 6),
        },
        "beats_hrnet_w32": summary["AP_improvement_vs_hrnet_w32"] >= 4.0,
        "summary": summary,
    }
