"""CADENet detection smoke (DAWN-style metrics)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.smoke_util import load_sibling


def evaluation_smoke() -> dict[str, Any]:
    schema = load_sibling(__file__, "schema")
    metrics = load_sibling(__file__, "metrics")
    pred = [schema.Detection(0.1, 0.1, 0.5, 0.5, 0.9, 0)]
    gt = [schema.Detection(0.12, 0.12, 0.48, 0.48, 1.0, 0)]
    f1 = metrics.detection_f1(pred, gt)
    prec, rec, _ = metrics.precision_recall_f1(pred, gt)
    return {
        "paper": "arXiv:2605.19695",
        "detection_f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }
