"""Synthetic die comparison demo — Figs. 1 & 3."""

from __future__ import annotations

from typing import Any

from ltx_trainer.dilated_sym_diff.benchmarks import summary_anchors
from ltx_trainer.dilated_sym_diff.config import DilatedSymDiffConfig
from ltx_trainer.dilated_sym_diff.metrics import choose_radius, iou, iou_vs_radius
from ltx_trainer.dilated_sym_diff.morphology import dilated_symmetric_difference, symmetric_difference
from ltx_trainer.dilated_sym_diff.synthetic import fig3_die_pair


def run_demo(cfg: DilatedSymDiffConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DilatedSymDiffConfig()
    a, b, ref, delta = fig3_die_pair()
    radii = list(range(0, 31, 1))
    curve = iou_vs_radius(a, b, ref, radii)
    r0_iou = next(score for r, score in curve if r == 0)
    r8_iou = next(score for r, score in curve if r == cfg.default_radius)
    chosen = choose_radius(curve, delta_align=delta, target_iou=0.95)
    sym = symmetric_difference(a, b)
    diff_r8 = dilated_symmetric_difference(a, b, cfg.default_radius)
    summary = summary_anchors()
    return {
        "config": {"default_radius": cfg.default_radius, "delta_align": round(delta, 2)},
        "symmetric_diff_pixels": int(sym.sum()),
        "dilated_diff_r8_pixels": int(diff_r8.sum()),
        "IoU_r0": round(r0_iou, 3),
        "IoU_r8": round(r8_iou, 3),
        "chosen_r": chosen,
        "curve_sample": [{"r": r, "IoU": round(s, 3)} for r, s in curve if r in (0, 7, 8, 30)],
        "paper_IoU_at_r8": summary["IoU_at_r8"],
        "summary": summary,
    }
